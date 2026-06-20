"""Tests for the unified Posterior types."""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import logsumexp

from photomancy.posterior import (
    AbstractPosterior,
    GaussianPosterior,
    MixturePosterior,
    SamplePosterior,
    cluster_to_mixture,
)


def test_posterior_subtypes_share_abstract_base_with_scalar_evidence():
    """Gaussian and Mixture are AbstractPosterior with a scalar ``.evidence``.

    The base reconciles ``evidence`` as a field (Gaussian) vs a property
    (Mixture) through ``AbstractVar``, so consumers type against one interface.
    """
    gaussian = GaussianPosterior(
        mean=jnp.zeros(2), cov=jnp.eye(2), evidence=jnp.asarray(-3.0)
    )
    mixture = MixturePosterior(
        means=jnp.zeros((2, 2)),
        covs=jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        log_evidences=jnp.asarray([-1.0, -2.0]),
    )

    assert isinstance(gaussian, AbstractPosterior)
    assert isinstance(mixture, AbstractPosterior)
    assert jnp.ndim(gaussian.evidence) == 0
    assert jnp.ndim(mixture.evidence) == 0


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


def test_sample_posterior_resamples_by_weight_and_reports_evidence():
    """SamplePosterior resamples stored particles by weight and exposes log Z."""
    samples = jnp.array([[0.0, 0.0], [5.0, 5.0]])
    log_weights = jnp.log(jnp.array([0.99, 0.01]))
    post = SamplePosterior(
        samples=samples, log_weights=log_weights, evidence=jnp.asarray(-2.0)
    )

    assert isinstance(post, AbstractPosterior)

    draws = post.sample(jax.random.key(0), 2000)
    assert draws.shape == (2000, 2)
    near_first = jnp.mean(jnp.all(jnp.abs(draws - samples[0]) < 1e-6, axis=1))
    assert near_first > 0.9

    assert jnp.allclose(post.evidence, -2.0)


def test_sample_posterior_log_prob_unsupported():
    """A sample-based posterior carries no closed-form density."""
    post = SamplePosterior(
        samples=jnp.zeros((3, 2)),
        log_weights=jnp.zeros(3),
        evidence=jnp.asarray(jnp.nan),
    )
    with pytest.raises(NotImplementedError):
        post.log_prob(jnp.zeros(2))


def test_sample_posterior_param_names_and_sample_dict():
    """param_names labels the columns; sample_dict returns named draws."""
    samples = jnp.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    post = SamplePosterior(
        samples=samples,
        log_weights=jnp.zeros(3),
        evidence=jnp.asarray(jnp.nan),
        param_names=("a", "b"),
    )
    drawn = post.sample(jax.random.key(0), 5)
    assert drawn.shape == (5, 2)
    d = post.sample_dict(jax.random.key(0), 5)
    assert set(d) == {"a", "b"}
    assert d["a"].shape == (5,)


def test_cluster_to_mixture_recovers_two_clusters():
    """Two separated clusters become a 2-mode mixture with correct means/weights."""
    k0, k1 = jax.random.split(jax.random.key(1))
    a = jax.random.normal(k0, (200, 2)) * 0.2
    b = jax.random.normal(k1, (200, 2)) * 0.2 + jnp.array([5.0, 5.0])
    samples = jnp.concatenate([a, b], axis=0)
    post = SamplePosterior(
        samples=samples,
        log_weights=jnp.zeros(400),
        evidence=jnp.asarray(jnp.nan),
        param_names=("x", "y"),
    )
    mix = cluster_to_mixture(post, 2, key=jax.random.key(2))
    assert mix.means.shape == (2, 2)
    assert mix.covs.shape == (2, 2, 2)
    centers = jnp.sort(mix.means[:, 0])
    assert jnp.allclose(centers, jnp.array([0.0, 5.0]), atol=0.5)
    assert jnp.allclose(jnp.exp(mix.log_weights), jnp.array([0.5, 0.5]), atol=0.1)


def test_gaussian_posterior_to_prior_is_matching_jointprior():
    """GaussianPosterior.to_prior() is a JointPrior with the same MVN density."""
    from photomancy.priors import JointPrior

    mean = jnp.array([1.0, -2.0])
    cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    post = GaussianPosterior(mean=mean, cov=cov, evidence=jnp.asarray(-3.0))

    prior = post.to_prior()
    assert isinstance(prior, JointPrior)
    z = jnp.array([0.5, -1.0])
    # The posterior-as-prior carries the full covariance: same density.
    assert jnp.allclose(prior.log_prob(z), post.log_prob(z), atol=1e-4)


def test_mixture_posterior_to_prior_preserves_modes():
    """MixturePosterior.to_prior() is a MixturePrior with matching density."""
    from photomancy.priors import MixturePrior

    means = jnp.array([[0.0, 0.0], [4.0, -3.0]])
    covs = jnp.stack([jnp.eye(2), 0.5 * jnp.eye(2)])
    log_ev = jnp.log(jnp.array([0.6, 0.4]))
    post = MixturePosterior(means=means, covs=covs, log_evidences=log_ev)

    prior = post.to_prior()
    assert isinstance(prior, MixturePrior)
    assert prior.ndim == 2
    z = jnp.array([0.5, -0.5])
    assert jnp.allclose(prior.log_prob(z), post.log_prob(z), atol=1e-4)


def test_sample_posterior_to_prior_via_clustering():
    """SamplePosterior.to_prior(k, key) clusters samples into a k-mode MixturePrior."""
    from photomancy.priors import MixturePrior

    k0, k1 = jax.random.split(jax.random.key(3))
    a = jax.random.normal(k0, (300, 2)) * 0.2
    b = jax.random.normal(k1, (300, 2)) * 0.2 + jnp.array([6.0, 6.0])
    samples = jnp.concatenate([a, b], axis=0)
    post = SamplePosterior(
        samples=samples, log_weights=jnp.zeros(600), evidence=jnp.asarray(jnp.nan)
    )

    prior = post.to_prior(2, key=jax.random.key(4))
    assert isinstance(prior, MixturePrior)
    # the two clusters survive as two modes with ~equal weight
    centers = jnp.sort(prior.means[:, 0])
    assert jnp.allclose(centers, jnp.array([0.0, 6.0]), atol=0.5)
    w = jax.nn.softmax(prior.log_weights)
    assert jnp.allclose(jnp.sort(w), jnp.array([0.5, 0.5]), atol=0.1)
