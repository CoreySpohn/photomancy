"""Unified posterior types for photomancy inference backends.

A fit returns one Posterior with a uniform query interface -- ``sample``,
``log_prob``, and ``evidence`` -- backed by a representation-specific subtype
(Gaussian, mixture, or samples). Consumers (the EIG / scheduling layer, model
comparison) read this interface and stay agnostic to which backend produced it.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal


class AbstractPosterior(eqx.Module):
    """The uniform posterior interface every backend returns.

    A posterior answers three queries regardless of its internal representation
    (Gaussian, evidence-weighted mixture, or weighted samples):

    - ``sample(key, n)`` draws ``n`` positions in flat parameter space,
    - ``log_prob(z)`` scores a flat position (may be unsupported for sample-based
      posteriors),
    - ``evidence`` is the scalar log marginal likelihood ``log Z``.

    ``evidence`` is an ``AbstractVar`` so a subtype may back it with a field (the
    Laplace ``log Z``) or a property (the mixture's ``logsumexp`` over modes). The
    EIG / scheduling layer and model comparison read this interface and stay
    agnostic to which backend produced the posterior.
    """

    evidence: AbstractVar[jnp.ndarray]

    @abstractmethod
    def sample(self, key, n):
        """Draw ``n`` samples in flat parameter space. Shape ``(n, d)``."""
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, z):
        """Log-density at flat position ``z``."""
        raise NotImplementedError


class GaussianPosterior(AbstractPosterior):
    """A Gaussian posterior over the flat parameter position.

    Produced by the Laplace / Pathfinder backends; ``sample`` and ``log_prob``
    are exact, and ``evidence`` is the (Laplace) log marginal likelihood.

    Args:
        mean: Posterior mean. Shape ``(d,)``.
        cov: Posterior covariance. Shape ``(d, d)``.
        evidence: Scalar log marginal likelihood (``log Z``).
    """

    mean: jnp.ndarray
    cov: jnp.ndarray
    evidence: jnp.ndarray

    def sample(self, key, n):
        """Draw ``n`` samples in flat parameter space. Shape ``(n, d)``."""
        return jax.random.multivariate_normal(key, self.mean, self.cov, (n,))

    def log_prob(self, z):
        """Exact Gaussian log-density at flat position ``z``."""
        return multivariate_normal.logpdf(z, self.mean, self.cov)


class MixturePosterior(AbstractPosterior):
    """An evidence-weighted Gaussian mixture posterior (the Laplace mixture).

    Each mode is a Gaussian; modes are weighted by ``softmax(log_evidences)`` and
    the total ``evidence`` marginalizes over modes via ``logsumexp(log_evidences)``.
    This is the substrate the analytic Fisher EIG consumes -- it carries the
    per-mode Gaussians (``means``, ``covs``).

    Args:
        means: Per-mode means. Shape ``(K, d)``.
        covs: Per-mode covariances. Shape ``(K, d, d)``.
        log_evidences: Per-mode log marginal likelihoods ``log Z_k``. Shape ``(K,)``.
    """

    means: jnp.ndarray
    covs: jnp.ndarray
    log_evidences: jnp.ndarray

    @property
    def evidence(self):
        """Total log marginal likelihood, ``logsumexp(log Z_k)``."""
        return logsumexp(self.log_evidences)

    @property
    def log_weights(self):
        """Normalized log mixture weights, ``log softmax(log Z_k)``. Shape ``(K,)``."""
        return self.log_evidences - logsumexp(self.log_evidences)

    def log_prob(self, z):
        """Mixture log-density at flat position ``z``."""
        comp_logp = jax.vmap(lambda m, c: multivariate_normal.logpdf(z, m, c))(
            self.means, self.covs
        )
        return logsumexp(self.log_weights + comp_logp)

    def sample(self, key, n):
        """Draw ``n`` samples: pick a mode by weight, then sample its Gaussian."""
        k_comp, k_draw = jax.random.split(key)
        comp = jax.random.categorical(k_comp, self.log_evidences, shape=(n,))
        keys = jax.random.split(k_draw, n)
        return jax.vmap(jax.random.multivariate_normal)(
            keys, self.means[comp], self.covs[comp]
        )


class SamplePosterior(AbstractPosterior):
    """A posterior represented by weighted samples (NUTS / SMC particles).

    ``sample`` resamples the stored particles by their weights with replacement;
    ``log_prob`` is unsupported (no closed-form density); ``evidence`` is the
    backend's log marginal likelihood -- the SMC tempering ``log Z`` for SMC, or
    ``NaN`` for plain MCMC samples that carry no evidence estimate.

    Args:
        samples: Particle positions in flat parameter space. Shape ``(n, d)``.
        log_weights: Per-particle log weights, normalized or not. Shape ``(n,)``.
        evidence: Scalar log marginal likelihood ``log Z`` (or ``NaN``).
    """

    samples: jnp.ndarray
    log_weights: jnp.ndarray
    evidence: jnp.ndarray

    def sample(self, key, n):
        """Resample ``n`` particles by weight, with replacement. Shape ``(n, d)``."""
        idx = jax.random.categorical(key, self.log_weights, shape=(n,))
        return self.samples[idx]

    def log_prob(self, z):
        """Unsupported: a weighted-sample posterior has no closed-form density."""
        raise NotImplementedError(
            "SamplePosterior has no closed-form density; refit a Gaussian to the "
            "samples or use a kernel density estimate to evaluate log_prob."
        )
