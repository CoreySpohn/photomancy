"""Tests for general posterior utilities (manifold projection, capped sampling)."""

import jax
import jax.numpy as jnp

from photomancy.posterior import MixturePosterior
from photomancy.posterior_utils import project_samples, sample_capped


def test_sample_capped_limits_variance():
    """sample_capped caps a mode's covariance eigenvalues at max_variance."""
    post = MixturePosterior(
        means=jnp.zeros((1, 2)),
        covs=(100.0 * jnp.eye(2))[None],  # huge raw spread
        log_evidences=jnp.zeros(1),
    )
    z = sample_capped(post, jax.random.key(0), 4000, max_variance=1.0)
    assert z.shape == (4000, 2)
    # empirical variance is bounded near the cap, far below the raw 100
    assert jnp.all(jnp.var(z, axis=0) < 2.0)


def test_project_samples_pulls_to_mode():
    """project_samples re-optimizes samples toward a logdensity's mode."""
    mu = jnp.array([2.0, -1.0])

    def logdensity(z):
        return -0.5 * jnp.sum((z - mu) ** 2)

    samples = mu + 3.0 * jax.random.normal(jax.random.key(1), (16, 2))
    projected = project_samples(samples, logdensity, n_steps=300, lr=0.1)

    assert projected.shape == (16, 2)
    # every projected sample is closer to the mode than it started
    d0 = jnp.linalg.norm(samples - mu, axis=1)
    d1 = jnp.linalg.norm(projected - mu, axis=1)
    assert jnp.all(d1 < d0)
    assert jnp.allclose(projected, mu, atol=0.3)
