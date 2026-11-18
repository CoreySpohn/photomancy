"""Weight statistics and systematic resampling for JAX particle clouds."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = ["kish_ess", "normalize_log_weights", "systematic_indices"]


def _shift_log_weights(log_weights):
    log_weights = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
    peak = jnp.max(log_weights)
    no_support = ~jnp.isfinite(peak)
    shifted = jnp.where(no_support, -jnp.inf, log_weights - peak)
    return shifted, peak, no_support


def normalize_log_weights(log_weights):
    """Return ``(normalized, log_total, no_support)`` for a log-weight vector.

    Normalize relative to the largest log weight to retain precision at large
    common offsets. NaNs count as zero weight. A nonfinite maximum gives
    ``no_support=True``, all ``-inf`` normalized weights and a total of ``-inf``.
    """
    shifted, peak, no_support = _shift_log_weights(log_weights)
    log_sum = logsumexp(jnp.where(no_support, 0.0, shifted))
    return (
        shifted - log_sum,
        jnp.where(no_support, -jnp.inf, peak + log_sum),
        no_support,
    )


def kish_ess(log_weights):
    """Return Kish effective sample size, or zero without support."""
    shifted, _, no_support = _shift_log_weights(log_weights)
    weights = jnp.exp(shifted)
    sum_squares = jnp.sum(weights**2)
    return jnp.sum(weights) ** 2 / jnp.where(no_support, 1.0, sum_squares)


def systematic_indices(key, log_weights, n):
    """Draw ``n`` source indices with one random systematic offset.

    ``log_weights`` is a normalized log-weight vector with at least one finite
    entry. ``n`` is a static output size, independent of the input size.
    """
    cdf = jnp.cumsum(jnp.exp(log_weights))
    cdf = cdf / cdf[-1]
    grid = (jax.random.uniform(key) + jnp.arange(n)) / n
    # Rounding to one would let a trailing zero-weight bin receive a draw.
    grid = jnp.minimum(grid, jnp.nextafter(jnp.asarray(1.0, grid.dtype), 0.0))
    return jnp.clip(jnp.searchsorted(cdf, grid, side="right"), 0, cdf.shape[0] - 1)
