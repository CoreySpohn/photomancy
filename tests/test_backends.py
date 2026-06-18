"""Tests for inference backends."""

import jax.numpy as jnp

from photomancy.backends import LaplaceBackend
from photomancy.posterior import GaussianPosterior


def test_laplace_backend_recovers_gaussian_map_and_covariance():
    """LaplaceBackend recovers a Gaussian target's MAP and covariance."""
    true_mean = jnp.array([1.0, -2.0])
    true_cov = jnp.array([[0.25, 0.0], [0.0, 1.0]])
    prec = jnp.linalg.inv(true_cov)

    def logdensity(z):
        d = z - true_mean
        return -0.5 * d @ prec @ d

    post = LaplaceBackend().run(logdensity, init=jnp.zeros(2))

    assert isinstance(post, GaussianPosterior)
    assert jnp.allclose(post.mean, true_mean, atol=1e-3)
    assert jnp.allclose(post.cov, true_cov, atol=1e-3)
