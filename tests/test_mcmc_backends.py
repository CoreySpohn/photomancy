"""Tests for the BlackJAX MCMC backends (NUTS, adaptive-tempered SMC)."""

import jax
import jax.numpy as jnp

from photomancy.backends import MCLMCBackend, NUTSBackend, SMCBackend
from photomancy.posterior import SamplePosterior


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


def test_smc_backend_recovers_posterior_and_evidence():
    """Adaptive-tempered SMC recovers the conjugate posterior mean and log Z.

    Prior ``N(0, tau^2)`` times likelihood ``N(y; z, s^2)`` has a closed-form
    posterior and evidence ``Z = N(y; 0, tau^2 + s^2)``, so SMC's tempering log Z
    has a known target.
    """
    tau, s, y = 2.0, 1.0, 1.0

    def logprior(z):
        return -0.5 * jnp.sum(z**2) / tau**2 - 0.5 * jnp.log(2 * jnp.pi * tau**2)

    def loglikelihood(z):
        return -0.5 * jnp.sum((y - z) ** 2) / s**2 - 0.5 * jnp.log(2 * jnp.pi * s**2)

    analytic_log_z = -0.5 * jnp.log(2 * jnp.pi * (tau**2 + s**2)) - 0.5 * y**2 / (
        tau**2 + s**2
    )
    post_mean = (y / s**2) / (1 / tau**2 + 1 / s**2)

    k_init, k_run = jax.random.split(jax.random.key(0))
    init_particles = tau * jax.random.normal(k_init, (3000, 1))
    post = SMCBackend().run(logprior, loglikelihood, init_particles, k_run)

    assert isinstance(post, SamplePosterior)
    weights = jnp.exp(post.log_weights)
    wmean = jnp.average(post.samples[:, 0], weights=weights)
    assert jnp.allclose(wmean, post_mean, atol=0.1)
    assert jnp.allclose(post.evidence, analytic_log_z, atol=0.2)
