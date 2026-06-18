"""Tests for the unified Posterior types."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from photomancy.posterior import GaussianPosterior, MixturePosterior


def test_gaussian_posterior_samples_log_probs_and_exposes_evidence():
    """A GaussianPosterior samples, peaks at its mean, and exposes evidence."""
    mean = jnp.array([1.0, -2.0])
    cov = jnp.array([[0.25, 0.0], [0.0, 1.0]])
    post = GaussianPosterior(mean=mean, cov=cov, evidence=jnp.asarray(-3.0))

    # log_prob peaks at the mean
    assert post.log_prob(mean) > post.log_prob(mean + jnp.array([2.0, 2.0]))

    # sample shape + recovers the mean in expectation
    samples = post.sample(jax.random.key(0), 5000)
    assert samples.shape == (5000, 2)
    assert jnp.allclose(samples.mean(axis=0), mean, atol=0.1)

    # evidence is exposed
    assert jnp.allclose(post.evidence, -3.0)


def test_mixture_posterior_weights_by_evidence_and_dominant_mode():
    """A MixturePosterior weights modes by evidence and marginalizes log Z."""
    means = jnp.array([[0.0], [10.0]])
    covs = jnp.array([[[1.0]], [[1.0]]])
    log_evidences = jnp.array([0.0, -5.0])  # mode 0 dominates

    post = MixturePosterior(means=means, covs=covs, log_evidences=log_evidences)

    # total evidence marginalizes over modes
    assert jnp.allclose(post.evidence, logsumexp(log_evidences))
    # the dominant mode controls the density
    assert post.log_prob(jnp.array([0.0])) > post.log_prob(jnp.array([10.0]))
    # samples concentrate on the dominant mode
    samples = post.sample(jax.random.key(0), 4000)
    assert samples.shape == (4000, 1)
    assert jnp.mean(jnp.abs(samples[:, 0]) < 3.0) > 0.9
