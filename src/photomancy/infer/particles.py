"""Systematic resampling for JAX particle clouds."""

import jax
import jax.numpy as jnp

__all__ = ["systematic_indices"]


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
