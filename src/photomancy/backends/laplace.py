"""Laplace backend: MAP + Hessian -> GaussianPosterior."""

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import GaussianPosterior


class LaplaceBackend(AbstractBackend):
    """MAP + Laplace approximation: a Gaussian posterior around the mode.

    Optimizes the logdensity to its MAP, takes the Hessian there for the
    covariance, and returns a ``GaussianPosterior`` carrying the Laplace
    log-evidence. Exact for Gaussian targets.
    """

    def run(self, logdensity, init, key=None):
        """Fit the Laplace approximation and return a GaussianPosterior."""

        def neg(z):
            return -logdensity(z)

        z_map = minimize(neg, init, method="BFGS").x
        hess = jax.hessian(neg)(z_map)
        cov = jnp.linalg.inv(hess)
        d = z_map.shape[0]
        _, logdet_cov = jnp.linalg.slogdet(cov)
        log_z = logdensity(z_map) + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * logdet_cov
        return GaussianPosterior(mean=z_map, cov=cov, evidence=log_z)
