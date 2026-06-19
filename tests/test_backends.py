"""Tests for inference backends."""

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from photomancy.backends import LaplaceBackend, LaplaceMixtureBackend
from photomancy.posterior import GaussianPosterior, MixturePosterior


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


def test_laplace_backend_clamps_covariance_in_flat_directions():
    """The min_eigenvalue floor caps the covariance where the target is flat.

    A direction with near-zero curvature would invert to an enormous variance;
    the eigenvalue floor on the precision bounds it at ``1 / min_eigenvalue``,
    while leaving the well-constrained direction essentially unchanged.
    """
    prec = jnp.array([[4.0, 0.0], [0.0, 1e-6]])

    def logdensity(z):
        return -0.5 * z @ prec @ z

    min_eig = 1.0
    post = LaplaceBackend(min_eigenvalue=min_eig).run(logdensity, init=jnp.zeros(2))

    assert post.cov[1, 1] <= 1.0 / min_eig + 1e-5
    assert jnp.allclose(post.cov[0, 0], 0.25, atol=1e-3)


def test_laplace_mixture_recovers_both_modes_and_weights_by_evidence():
    """Multi-start Laplace finds each mode and weights them by local evidence."""
    centers = jnp.array([-4.0, 4.0])
    log_w = jnp.log(jnp.array([0.9, 0.1]))

    def logdensity(z):
        # A two-component, well-separated Gaussian-mixture density over z in R^1.
        return logsumexp(-0.5 * (z[0] - centers) ** 2 + log_w)

    inits = jnp.array([[-4.0], [4.0]])  # one start near each mode
    post = LaplaceMixtureBackend(min_eigenvalue=1e-6).run(logdensity, inits)

    assert isinstance(post, MixturePosterior)
    assert jnp.allclose(post.means[:, 0], centers, atol=1e-2)

    weights = jnp.exp(post.log_weights)
    assert weights[0] > weights[1]
    assert jnp.allclose(weights, jnp.array([0.9, 0.1]), atol=0.05)
