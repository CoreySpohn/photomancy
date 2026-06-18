"""Unified posterior types for photomancy inference backends.

A fit returns one Posterior with a uniform query interface -- ``sample``,
``log_prob``, and ``evidence`` -- backed by a representation-specific subtype
(Gaussian, mixture, or samples). Consumers (the EIG / scheduling layer, model
comparison) read this interface and stay agnostic to which backend produced it.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal


class GaussianPosterior(eqx.Module):
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


class MixturePosterior(eqx.Module):
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
