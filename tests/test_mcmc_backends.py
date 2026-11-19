"""Tests for the BlackJAX MCMC backends (NUTS, MCLMC)."""

import jax
import jax.numpy as jnp
import pytest

from photomancy.infer.backends import MCLMCBackend, NUTSBackend
from photomancy.infer.legacy_posterior import SamplePosterior

# Full sampler runs belong in the nightly tier.
pytestmark = pytest.mark.slow


def test_nuts_backend_samples_concentrate_at_truth():
    """NUTSBackend returns a SamplePosterior whose draws sit at the mode."""
    true_mean = jnp.array([1.0, -2.0])
    prec = jnp.linalg.inv(jnp.array([[0.25, 0.0], [0.0, 1.0]]))

    def logdensity(z):
        d = z - true_mean
        return -0.5 * d @ prec @ d

    post = NUTSBackend(n_warmup=300, n_samples=2000).run(
        logdensity, init=jnp.zeros(2), key=jax.random.key(0)
    )

    assert isinstance(post, SamplePosterior)
    assert post.samples.shape == (2000, 2)
    assert jnp.allclose(jnp.mean(post.samples, axis=0), true_mean, atol=0.1)
    # plain MCMC carries no evidence estimate
    assert jnp.isnan(post.evidence)

    # the unified resample interface still works (equal weights -> uniform draw)
    draws = post.sample(jax.random.key(1), 500)
    assert draws.shape == (500, 2)


def test_mclmc_backend_samples_concentrate_at_truth():
    """MCLMCBackend returns a SamplePosterior whose draws sit at the mode."""
    true_mean = jnp.array([1.0, -2.0])
    prec = jnp.linalg.inv(jnp.array([[0.25, 0.0], [0.0, 1.0]]))

    def logdensity(z):
        d = z - true_mean
        return -0.5 * d @ prec @ d

    post = MCLMCBackend(n_tune=2000, n_samples=4000).run(
        logdensity, init=jnp.zeros(2), key=jax.random.key(0)
    )

    assert isinstance(post, SamplePosterior)
    assert post.samples.shape == (4000, 2)
    assert jnp.allclose(jnp.mean(post.samples, axis=0), true_mean, atol=0.15)
    # gradient-based MCMC carries no evidence estimate
    assert jnp.isnan(post.evidence)

    # the unified resample interface still works (equal weights -> uniform draw)
    draws = post.sample(jax.random.key(1), 500)
    assert draws.shape == (500, 2)
