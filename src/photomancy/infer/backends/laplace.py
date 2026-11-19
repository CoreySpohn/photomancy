"""Laplace backend: MAP + Hessian -> GaussianPosterior.

The MAP is found with ``optax.lbfgs`` (a quasi-Newton line search that
converges to high precision). The covariance is the inverse of the
negative-log-density Hessian at the MAP, computed by Hessian-vector products
(forward-over-reverse AD) and regularized by flooring its eigenvalues at
``min_eigenvalue`` -- so a direction the data barely constrains gets a bounded
variance instead of inverting to an enormous one. The single-mode fit lives in
``laplace_fit`` so the multi-start mixture backend can ``vmap`` it.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from photomancy.infer.backends._flat import (
    flat_density,
    mixture_importance,
    start_points,
)
from photomancy.infer.backends.base import AbstractBackend, BackendDraws, LegacyBackend
from photomancy.infer.legacy_posterior import GaussianPosterior, MixturePosterior


def _map_optimize(neg_logdensity, init, n_steps):
    """Minimize ``neg_logdensity`` from ``init`` with ``optax.lbfgs``."""
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(neg_logdensity)

    def step(carry, _):
        z, state = carry
        value, grad = value_and_grad(z, state=state)
        updates, state = opt.update(
            grad, state, z, value=value, grad=grad, value_fn=neg_logdensity
        )
        z = optax.apply_updates(z, updates)
        return (z, state), None

    (z_map, _), _ = jax.lax.scan(step, (init, opt.init(init)), None, length=n_steps)
    return z_map


def laplace_covariance(neg_logdensity, z_map, min_eigenvalue):
    """Eigenvalue-clamped inverse Hessian of ``neg_logdensity`` at ``z_map``."""
    grad_fn = jax.grad(neg_logdensity)

    def hvp(v):
        return jax.jvp(grad_fn, (z_map,), (v,))[1]

    d = z_map.shape[0]
    hess = jax.vmap(hvp)(jnp.eye(d, dtype=z_map.dtype))
    hess = 0.5 * (hess + hess.T)
    eigvals, eigvecs = jnp.linalg.eigh(hess)
    eigvals = jnp.maximum(eigvals, min_eigenvalue)
    precision = (eigvecs * eigvals) @ eigvecs.T
    return jnp.linalg.inv(precision)


@eqx.filter_jit
def laplace_fit(logdensity, init, n_steps, min_eigenvalue):
    """Single-mode Laplace fit: MAP, covariance, and Laplace log-evidence.

    ``filter_jit``-compiled: when ``logdensity`` is a ``SceneLogDensity`` Module its
    array leaves (e.g. a forward's PSF datacube) are traced inputs, not baked-in
    constants. ``n_steps`` / ``min_eigenvalue`` are Python scalars and stay static.

    Args:
        logdensity: ``z -> scalar`` log-density over the flat parameter position.
        init: Initial flat position. Shape ``(d,)``.
        n_steps: Number of L-BFGS iterations.
        min_eigenvalue: Floor on the precision (negative-Hessian) eigenvalues.

    Returns:
        A tuple ``(z_map, cov, log_evidence)``: the MAP point ``(d,)``, the
        regularized covariance ``(d, d)``, and the scalar Laplace ``log Z``.
    """

    def neg(z):
        return -logdensity(z)

    z_map = _map_optimize(neg, init, n_steps)
    cov = laplace_covariance(neg, z_map, min_eigenvalue)
    d = z_map.shape[0]
    _, logdet_cov = jnp.linalg.slogdet(cov)
    log_z = logdensity(z_map) + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * logdet_cov
    return z_map, cov, log_z


class LaplaceBackend(LegacyBackend):
    """MAP + Laplace approximation: a Gaussian posterior around the mode.

    Optimizes the logdensity to its MAP with L-BFGS, takes the eigenvalue-clamped
    inverse Hessian there as the covariance, and returns a ``GaussianPosterior``
    carrying the Laplace log-evidence. Exact for Gaussian targets.

    Args:
        n_steps: Number of L-BFGS iterations for the MAP search.
        min_eigenvalue: Floor on the precision eigenvalues. The default is a tiny
            positive-definiteness safeguard; pass a larger value (the orbit
            unconstrained space uses ``1.0``) to bound the variance of directions
            the data barely constrains.
    """

    n_steps: int = 100
    min_eigenvalue: float = 1e-10

    def run(self, logdensity, init, key=None):
        """Fit the Laplace approximation and return a GaussianPosterior."""
        z_map, cov, log_z = laplace_fit(
            logdensity, init, self.n_steps, self.min_eigenvalue
        )
        return GaussianPosterior(mean=z_map, cov=cov, evidence=log_z)


class LaplaceMixtureBackend(LegacyBackend):
    """Multi-start MAP + Laplace: an evidence-weighted mixture of Gaussians.

    Runs the single-mode Laplace fit from each of ``K`` starts (``init`` has shape
    ``(K, d)``, one row per start), then assembles a ``MixturePosterior`` whose
    modes are weighted by their per-mode Laplace log-evidence. This is the EIG /
    value-of-information substrate: it carries the per-mode Gaussians and
    approximates the total ``log Z`` by ``logsumexp`` over the modes. Pass the
    starts from a blind initializer (the orbix Thiele-Innes grid maxima) to cover
    the period aliases.

    Args:
        n_steps: L-BFGS iterations per start.
        min_eigenvalue: Floor on each mode's precision eigenvalues.
    """

    n_steps: int = 100
    min_eigenvalue: float = 1e-10

    def run(self, logdensity, init, key=None):
        """Fit one Laplace mode per start in ``init`` (shape ``(K, d)``)."""

        def fit_one(z0):
            return laplace_fit(logdensity, z0, self.n_steps, self.min_eigenvalue)

        means, covs, log_evidences = jax.vmap(fit_one)(init)
        return MixturePosterior(means=means, covs=covs, log_evidences=log_evidences)


class LaplaceInfo(eqx.Module):
    """The Gaussian mixture proposal of a :class:`Laplace` or :class:`Pathfinder` run.

    Coordinates are the unconstrained flat vector of ``fit.to_real`` (sorted entry
    keys, each leaf row-major).

    Attributes:
        means: Component means ``(K, d)``.
        covs: Component covariances ``(K, d, d)``.
        mode_log_evidence: Per-component log evidence (Laplace) or ELBO (Pathfinder)
            that sets the proposal weights ``(K,)``.
        component: Proposal component each draw came from ``(N,)``.
        proposal_log_prob: Log density of the proposal at each draw ``(N,)``.
    """

    means: jax.Array
    covs: jax.Array
    mode_log_evidence: jax.Array
    component: jax.Array
    proposal_log_prob: jax.Array


class Laplace(AbstractBackend):
    """MAP plus Hessian proposals and an importance-weighted posterior.

    From each of ``n_modes`` starts the density is maximized with L-BFGS and the
    eigenvalue-clamped inverse Hessian is the covariance, in unconstrained
    coordinates. The proposal is the mixture of those Gaussians weighted by their
    Laplace evidences; ``n_draws`` draws are weighted by ``density / proposal``. The
    evidence is the importance mean, so it is exact in expectation whatever the
    Gaussian's quality, and the weights carry the non-Gaussian correction.
    ``log_target`` is the fit's density; the proposal is only in ``info``.

    ``run(fit, Laplace(init=posterior), key)`` is the route to an evidence for a NUTS
    result. Angle coordinates of an orbit block are plain reals whose density repeats
    every ``2 pi``; the estimate is the integral over the one copy the proposal covers,
    so a posterior wider than a cell is not supported.

    Args:
        n_draws: Importance draws.
        n_modes: Number of starts and proposal components.
        n_steps: L-BFGS iterations per start.
        min_eigenvalue: Floor on the precision eigenvalues.
        init: ``None`` for prior draws as starts, one constrained draw, or a
            ``Posterior`` whose weights pick the starts.
    """

    n_draws: int = eqx.field(static=True, default=2000)
    n_modes: int = eqx.field(static=True, default=1)
    n_steps: int = eqx.field(static=True, default=100)
    min_eigenvalue: float = 1e-10
    init: object = None

    def _run(self, fit, key):
        """Fit the modes, then draw and weight from their mixture."""
        k_start, k_draw = jax.random.split(key)
        density = flat_density(fit)
        starts = start_points(fit, self.init, k_start, self.n_modes)
        means, covs, log_z = jax.vmap(
            lambda z0: laplace_fit(density, z0, self.n_steps, self.min_eigenvalue)
        )(starts)
        x, log_w, evidence, se, comp, proposal = mixture_importance(
            fit, density, means, covs, log_z, self.n_draws, k_draw
        )
        return BackendDraws(
            samples=x,
            log_weights=log_w,
            log_evidence=evidence,
            log_evidence_se=se,
            info=LaplaceInfo(means, covs, log_z, comp, proposal),
            kind="monte_carlo",
        )
