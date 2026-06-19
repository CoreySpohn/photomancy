"""Laplace backend: MAP + Hessian -> GaussianPosterior.

The MAP is found with ``optax.lbfgs`` (the Kidger first-order stack; a quasi-Newton
line search that converges to high precision). The covariance is the inverse of
the negative-log-density Hessian at the MAP, computed by Hessian-vector products
(forward-over-reverse AD) and regularized by flooring its eigenvalues at
``min_eigenvalue`` -- so a direction the data barely constrains gets a bounded
variance instead of inverting to an enormous one. The single-mode fit lives in
``laplace_fit`` so the multi-start mixture backend can ``vmap`` it.
"""

import jax
import jax.numpy as jnp
import optax

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import GaussianPosterior, MixturePosterior


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


def _laplace_covariance(neg_logdensity, z_map, min_eigenvalue):
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


def laplace_fit(logdensity, init, n_steps, min_eigenvalue):
    """Single-mode Laplace fit: MAP, covariance, and Laplace log-evidence.

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
    cov = _laplace_covariance(neg, z_map, min_eigenvalue)
    d = z_map.shape[0]
    _, logdet_cov = jnp.linalg.slogdet(cov)
    log_z = logdensity(z_map) + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * logdet_cov
    return z_map, cov, log_z


class LaplaceBackend(AbstractBackend):
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


class LaplaceMixtureBackend(AbstractBackend):
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
