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
        """Draw ``n`` samples in the posterior's parameter space. Shape ``(n, d)``."""
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
    """A posterior represented by weighted samples (OFTI, grid_search, NUTS, SMC).

    Columns are named by ``param_names`` -- the space is whatever those names denote
    (physical for OFTI / grid_search, unconstrained ``z`` for NUTS / SMC). ``sample``
    resamples particles by weight (inverse-CDF SIR, ``O(n)`` memory); ``log_prob`` is
    unsupported; ``evidence`` is the backend's ``log Z`` (or ``NaN``).

    Args:
        samples: Particle rows. Shape ``(n, d)``.
        log_weights: Per-particle log weights, normalized or not. Shape ``(n,)``.
        evidence: Scalar log marginal likelihood ``log Z`` (or ``NaN``).
        param_names: Names of the ``d`` columns. Default ``()``.
    """

    samples: jnp.ndarray
    log_weights: jnp.ndarray
    evidence: jnp.ndarray
    param_names: tuple[str, ...] = eqx.field(static=True, default=())

    def _resample_idx(self, key, n):
        cdf = jnp.cumsum(jax.nn.softmax(self.log_weights))
        u = jax.random.uniform(key, (n,))
        return jnp.clip(
            jnp.searchsorted(cdf, u, side="right"), 0, self.samples.shape[0] - 1
        )

    def sample(self, key, n):
        """Resample ``n`` particles by weight (inverse-CDF SIR). Shape ``(n, d)``."""
        return self.samples[self._resample_idx(key, n)]

    def sample_dict(self, key, n):
        """Resample ``n`` particles and return ``{name: (n,)}`` by ``param_names``."""
        drawn = self.samples[self._resample_idx(key, n)]
        return {name: drawn[:, i] for i, name in enumerate(self.param_names)}

    def log_prob(self, z):
        """Unsupported: a weighted-sample posterior has no closed-form density."""
        raise NotImplementedError(
            "SamplePosterior has no closed-form density; cluster_to_mixture or a KDE."
        )


def cluster_to_mixture(posterior, k, *, key, iters=25, cov_floor=1e-6):
    """Cluster weighted samples into a ``k``-mode Gaussian ``MixturePosterior``.

    Weighted k-means (fixed iterations) assigns particles to ``k`` clusters; each mode
    is the cluster's weighted mean and covariance, weighted by its mass. The per-mode
    ``log_evidence`` is the log cluster mass, so ``softmax`` gives the mixture weights
    (what the EIG alias-breaking term needs). Operates in whatever coordinates the
    samples are in.

    Args:
        posterior: A :class:`SamplePosterior`.
        k: Number of mixture modes.
        key: PRNG key for k-means initialization.
        iters: k-means iterations. Default 25.
        cov_floor: Diagonal added to each covariance for conditioning. Default 1e-6.

    Returns:
        A :class:`MixturePosterior` with ``means (k, d)``, ``covs (k, d, d)``,
        ``log_evidences (k,)``.
    """
    x = posterior.samples
    w = jax.nn.softmax(posterior.log_weights)
    n, d = x.shape
    centers = x[jax.random.choice(key, n, (k,), replace=False)]

    def step(centers, _):
        sq = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        onehot = jax.nn.one_hot(jnp.argmin(sq, axis=1), k)
        wmass = onehot * w[:, None]
        mass = jnp.sum(wmass, axis=0)
        new = (wmass.T @ x) / jnp.maximum(mass[:, None], 1e-30)
        return jnp.where(mass[:, None] > 0, new, centers), None

    centers, _ = jax.lax.scan(step, centers, None, length=iters)

    sq = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    onehot = jax.nn.one_hot(jnp.argmin(sq, axis=1), k)
    wmass = onehot * w[:, None]
    mass = jnp.sum(wmass, axis=0)
    means = (wmass.T @ x) / jnp.maximum(mass[:, None], 1e-30)

    def _cov(mu_k, wcol):
        diff = x - mu_k
        return (diff * wcol[:, None]).T @ diff / jnp.maximum(jnp.sum(wcol), 1e-30)

    covs = jax.vmap(_cov)(means, wmass.T) + cov_floor * jnp.eye(d)[None]
    log_evidences = jnp.log(jnp.maximum(mass, 1e-30))
    return MixturePosterior(means=means, covs=covs, log_evidences=log_evidences)
