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

from photomancy.infer.backends._flat import flat_density, start_points
from photomancy.infer.backends.base import AbstractBackend, LegacyBackend
from photomancy.infer.backends.importance import ImportanceInfo, modes_importance
from photomancy.infer.legacy_posterior import GaussianPosterior, MixturePosterior
from photomancy.infer.posterior import Posterior


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


def _hessian(neg_logdensity, z_map):
    """Symmetrized Hessian of ``neg_logdensity`` at ``z_map`` (forward over reverse)."""
    grad_fn = jax.grad(neg_logdensity)

    def hvp(v):
        return jax.jvp(grad_fn, (z_map,), (v,))[1]

    d = z_map.shape[0]
    hess = jax.vmap(hvp)(jnp.eye(d, dtype=z_map.dtype))
    return 0.5 * (hess + hess.T)


def laplace_covariance(neg_logdensity, z_map, min_eigenvalue):
    """Eigenvalue-clamped inverse Hessian of ``neg_logdensity`` at ``z_map``."""
    eigvals, eigvecs = jnp.linalg.eigh(_hessian(neg_logdensity, z_map))
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


@eqx.filter_jit
def _laplace_mode(logdensity, init, n_steps, min_eigenvalue):
    """One mode with the diagnostics that decide whether it is a usable proposal.

    Returns ``(z_map, cov, log_z, min_raw_eigenvalue, grad_norm, newton_decrement)``:
    the covariance floors the Hessian eigenvalues at ``min_eigenvalue``; the raw
    eigenvalue and the gradient are those of the unregularized negative log density at
    ``z_map``, and the Newton decrement ``0.5 g' cov g`` is the number of nats the point
    still lies below the Gaussian's peak.
    """

    def neg(z):
        return -logdensity(z)

    z_map = _map_optimize(neg, init, n_steps)
    grad = jax.grad(neg)(z_map)
    eigvals, eigvecs = jnp.linalg.eigh(_hessian(neg, z_map))
    floored = jnp.maximum(eigvals, min_eigenvalue)
    cov = (eigvecs / floored) @ eigvecs.T
    d = z_map.shape[0]
    log_z = (
        logdensity(z_map)
        + 0.5 * d * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.sum(jnp.log(floored))
    )
    return (
        z_map,
        cov,
        log_z,
        jnp.min(eigvals),
        jnp.linalg.norm(grad),
        0.5 * grad @ cov @ grad,
    )


class LaplaceInfo(eqx.Module):
    """The modes of a :class:`Laplace` run and which of them the proposal kept.

    Coordinates are the unconstrained flat vector of ``fit.to_real`` (sorted entry
    keys, each leaf row-major).

    Attributes:
        means: Mode locations ``(K, d)``.
        covs: Floored inverse Hessians ``(K, d, d)``.
        mode_log_evidence: Laplace log evidence of each mode, which sets its weight in
            the proposal ``(K,)``.
        kept: Whether each mode entered the proposal ``(K,)``.
        n_modes_kept: Number of modes kept; zero means the slice is NaN.
        min_raw_eigenvalue: Smallest eigenvalue of each mode's unregularized Hessian of
            the negative log density ``(K,)``; a large negative value is an indefinite
            point, not a mode.
        grad_norm: Norm of the gradient of the negative log density at each mode
            ``(K,)``.
        newton_decrement: ``0.5 g' cov g`` at each mode, in nats ``(K,)``.
        importance: The :class:`~photomancy.infer.backends.importance.ImportanceInfo`.
    """

    means: jax.Array
    covs: jax.Array
    mode_log_evidence: jax.Array
    kept: jax.Array
    n_modes_kept: jax.Array
    min_raw_eigenvalue: jax.Array
    grad_norm: jax.Array
    newton_decrement: jax.Array
    importance: ImportanceInfo


class Laplace(AbstractBackend):
    """A Gaussian at each mode, wrapped on periodic coordinates, importance weighted.

    1. From each of ``n_modes`` starts the density is maximized with L-BFGS. The
       proposal is one Gaussian at the mode with the inverse Hessian as covariance, in
       the unconstrained coordinates, wrapped on every coordinate the fit declares
       periodic (several modes make a mixture weighted by the Laplace evidences).
    2. The draws are weighted ``density / proposal`` with the wrapped proposal density.
       The weights correct the Gaussian to the posterior; the evidence is the importance
       mean and is consistent whenever the weights have a finite mean.
    3. The evidence and its standard error are NaN when the Pareto k-hat of the
       weights is above ``khat_threshold(n_draws)`` (``info.importance.khat``), as for
       :class:`~photomancy.infer.backends.importance.Importance`: a Gaussian that
       misses the tail of the posterior gives weights of infinite variance and an
       optimistic error. The draws and weights are still returned and are not gated;
       the operations of a ``Posterior`` gate them.
    4. Expected to be poor: a posterior that is flat or nearly flat in an angle (a
       Gaussian cannot match a flat density on a circle), a posterior with more than one
       mode, and a mode near a zero-density region of a block.
    5. In those cases the cloud is a seed for
       ``AdaptiveImportance(MixtureProposal.from_cloud(key, z, log_weights,
       n_centers=...), n_draws)``, with ``z`` the flat vectors of the cloud.
    6. Started from the prior, the modes found can miss the posterior: on a concentrated
       orbit fit two of four keys kept only modes whose cloud had a Kish size near 1
       and a k-hat above 3 (log evidence about 1200 nats low), and the other two a
       k-hat of 0.7 to 0.9 with an evidence about three stated errors low. All four
       return a NaN evidence; seed ``AdaptiveImportance`` from the cloud (item 5), or
       start from a draw that is already near the posterior.
    7. Widening the covariance is not a remedy: it makes the weights worse.

    A mode is dropped from the proposal when it is not a mode: when the unregularized
    Hessian has an eigenvalue below ``-min_eigenvalue`` (an indefinite point, as where
    an optimizer stalls against a wall) or the Newton decrement exceeds
    ``max_decrement`` nats (the gradient has not converged). A direction with
    curvature between ``-min_eigenvalue`` and ``min_eigenvalue`` is flat, not wrong,
    and gets the variance ``1 / min_eigenvalue``. With no mode left the slice is NaN
    (draws, weights, evidence and standard error) and ``info.n_modes_kept`` is zero;
    ``info.min_raw_eigenvalue`` and ``info.grad_norm`` say why.

    ``run(fit, Laplace(init=posterior), key)`` is the route to an evidence for a NUTS
    result.

    Args:
        n_draws: Importance draws.
        n_modes: Number of starts and proposal components.
        n_steps: L-BFGS iterations per start.
        min_eigenvalue: Floor on the precision eigenvalues of the proposal covariance
            and the indefiniteness threshold, in units of inverse squared unconstrained
            coordinate (the standard deviation of a flat direction is at most
            ``1 / sqrt(min_eigenvalue)``). It trades a narrow proposal, whose weights
            have a heavy right tail that a few thousand draws under-sample (a bias of
            about 0.02 nat on a prior-only orbit fit at 0.1), against a wide one,
            whose weights vary more (variance). The default 0.03 is unbiased on that
            fit.
        max_decrement: Largest Newton decrement, in nats, of a mode that is kept.
        init: ``None`` for prior draws as starts, one constrained draw, or a
            ``Posterior`` whose weights pick the starts.
    """

    n_draws: int = eqx.field(static=True, default=2000)
    n_modes: int = eqx.field(static=True, default=1)
    n_steps: int = eqx.field(static=True, default=100)
    min_eigenvalue: float = 0.03
    max_decrement: float = 1.0
    init: Posterior | dict | None = None

    @eqx.filter_jit
    def _run(self, fit, key, *, batch_size=None):
        """Find the modes, then draw and weight from the mixture of the kept ones."""
        k_start, k_draw = jax.random.split(key)
        density = flat_density(fit)
        starts = start_points(fit, self.init, k_start, self.n_modes)
        means, covs, log_z, min_eig, grad_norm, decrement = jax.vmap(
            lambda z0: _laplace_mode(density, z0, self.n_steps, self.min_eigenvalue)
        )(starts)
        kept = (
            jnp.all(jnp.isfinite(means), axis=-1)
            & jnp.isfinite(log_z)
            & (min_eig >= -self.min_eigenvalue)
            & (decrement <= self.max_decrement)
        )
        draws = modes_importance(
            fit, k_draw, means, covs, log_z, kept, self.n_draws, batch_size
        )
        info = LaplaceInfo(
            means,
            covs,
            log_z,
            kept,
            jnp.sum(kept),
            min_eig,
            grad_norm,
            decrement,
            draws.info,
        )
        return eqx.tree_at(lambda d: d.info, draws, info)
