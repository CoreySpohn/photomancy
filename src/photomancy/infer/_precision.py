"""Conversions at the float64 inference boundary."""

import jax.numpy as jnp


def _as_wide(value):
    """Convert an inference coordinate or a log-density operand to float64."""
    return jnp.asarray(
        value,
        jnp.float64,  # precision-ok: Fit coordinates and density sums use float64
    )
