"""Padding helpers and row capacities shared by the measurement data containers."""

import jax.numpy as jnp

MAX_REL_ASTROM = 64
MAX_STELLAR_ASTROM = 64
MAX_IMG = 64
MAX_CC_PTS = 200
MAX_RV = 256


def _pad_1d(arr, max_len, fill=0.0):
    """Pad a 1-D array to ``max_len``."""
    arr = jnp.asarray(arr)
    n = arr.shape[0]
    pad_width = max_len - n
    return jnp.pad(arr, (0, pad_width), constant_values=fill)


def _pad_2d(arr, max_rows, max_cols, fill=0.0):
    """Pad a 2-D array to ``(max_rows, max_cols)``."""
    arr = jnp.asarray(arr)
    pad_r = max_rows - arr.shape[0]
    pad_c = max_cols - arr.shape[1]
    return jnp.pad(arr, ((0, pad_r), (0, pad_c)), constant_values=fill)


def _valid_mask(n, max_len):
    """Boolean mask: True for the first ``n`` entries."""
    return jnp.arange(max_len) < n
