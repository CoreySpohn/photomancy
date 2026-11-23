"""Resampling of JAX particle clouds, with the key first.

``systematic_indices`` draws ``n`` indices from one random offset;
``categorical_indices`` draws independent indices with replacement. Both take
``(key, log_weights, ...)``.
"""

import math

import jax
import jax.numpy as jnp

__all__ = ["categorical_indices", "systematic_indices", "take_draws"]


def systematic_indices(key, log_weights, n):
    """Draw ``n`` source indices with one random systematic offset.

    ``log_weights`` is a normalized log-weight vector with at least one finite
    entry. ``n`` is a static output size, independent of the input size.

    Args:
        key: PRNG key for the single random offset.
        log_weights: Normalized log weights, shape ``(N,)``.
        n: Number of indices to draw.

    Returns:
        Integer indices into the ``N`` source draws, shape ``(n,)``.
    """
    cdf = jnp.cumsum(jnp.exp(log_weights))
    cdf = cdf / cdf[-1]
    grid = (jax.random.uniform(key) + jnp.arange(n)) / n
    # Rounding to one would let a trailing zero-weight bin receive a draw.
    grid = jnp.minimum(grid, jnp.nextafter(jnp.asarray(1.0, grid.dtype), 0.0))
    return jnp.clip(jnp.searchsorted(cdf, grid, side="right"), 0, cdf.shape[0] - 1)


def categorical_indices(key, log_weights, shape=()):
    """Draw indices of one slice with replacement, inverse CDF on the weights.

    ``log_weights`` is ``(N,)`` and normalized; memory is linear in ``N``. A slice
    with a NaN weight is drawn uniformly (the caller marks it). Returns ``shape``.

    Args:
        key: PRNG key.
        log_weights: Normalized log weights, shape ``(N,)``.
        shape: Shape of the returned index array.

    Returns:
        Integer indices into the ``N`` source draws, with shape ``shape``.
    """
    lw = jnp.asarray(log_weights)
    n = lw.shape[-1]
    safe = jnp.where(jnp.any(jnp.isnan(lw)), -math.log(n), lw)
    shape = tuple(shape)
    u = jax.random.uniform(key, (math.prod(shape),), jnp.float64)
    cdf = jnp.cumsum(jnp.exp(safe))
    hit = jnp.searchsorted(cdf / cdf[-1], u, side="right")
    return jnp.reshape(jnp.clip(hit, 0, n - 1), shape)


def take_draws(samples, idx):
    """Take the draws ``idx`` along the leading (draw) axis of every leaf.

    Args:
        samples: Pytree whose every leaf has a leading draw axis.
        idx: Integer indices of the draws to keep.

    Returns:
        The pytree with every leaf indexed by ``idx`` along its draw axis.
    """
    return jax.tree.map(lambda leaf: jnp.take(leaf, idx, axis=0), samples)
