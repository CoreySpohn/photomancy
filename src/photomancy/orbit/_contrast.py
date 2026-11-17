"""Padded contrast-curve interpolation for likelihoods and reports."""

import jax
import jax.numpy as jnp

# Finite "undetectable" dMag0 sentinel (replaces -inf): the predicted flux ratio
# still underflows to 0 (no non-detection penalty), but staying finite keeps the
# interpolation gradient finite on padded grids.
_UNDETECTABLE_DMAG0 = -1e8


def interpolate_contrast_curve(alpha_pred, sep_grid, dmag0_grid, limit_grid=None):
    """Interpolate the dMag0 detection limit at each epoch, gradient-safely.

    The contrast-curve grids are padded (trailing sep=0, dmag0=-inf). A raw
    ``jnp.interp`` over that non-monotonic query grid divides by zero in its slopes
    and subtracts ``inf - inf`` -- it returns a finite value but a nan gradient
    w.r.t. the separation (and even the wrong value when the curve is padded), so a
    fit cannot initialize. The padded columns (identified by ``dmag0 == -inf``) are
    pushed just past the last real separation as a strictly increasing tail, and the
    undetectable limit is a finite sentinel. The real curve is left untouched, so the
    value is exact within the curve; out-of-curve separations map to the sentinel
    (``flux_ratio -> 0``, no penalty) with a finite gradient.

    Args:
        alpha_pred: Predicted separation per epoch. Shape ``(N,)``.
        sep_grid: Per-epoch separation grid (padded). Shape ``(N, K)``.
        dmag0_grid: Per-epoch dMag0 limit grid (padded with -inf). Shape ``(N, K)``.
        limit_grid: Optional noise levels on the same grid. Padded values
            and out-of-curve queries hold the nearest real edge value.

    Returns:
        The interpolated dMag0 limit per epoch, shape ``(N,)``. If limit_grid
        is supplied, return a tuple of dMag0 and noise levels, each ``(N,)``.
    """
    is_pad = ~jnp.isfinite(dmag0_grid)
    max_real = jnp.max(jnp.where(is_pad, -jnp.inf, sep_grid), axis=1, keepdims=True)
    max_real = jnp.where(jnp.isfinite(max_real), max_real, 0.0)
    pad_xp = max_real + (jnp.arange(sep_grid.shape[1]) + 1.0) * 1e-6
    xp = jnp.where(is_pad, pad_xp, sep_grid)
    fp = jnp.where(is_pad, _UNDETECTABLE_DMAG0, dmag0_grid)

    def _one(alpha_val, xp_row, fp_row):
        return jnp.interp(
            alpha_val,
            xp_row,
            fp_row,
            left=_UNDETECTABLE_DMAG0,
            right=_UNDETECTABLE_DMAG0,
        )

    dmag0 = jax.vmap(_one)(alpha_pred, xp, fp)
    if limit_grid is None:
        return dmag0
    real_sep = jnp.where(is_pad, -jnp.inf, sep_grid)
    last = jnp.argmax(real_sep, axis=1)[:, None]
    edge_limit = jnp.take_along_axis(limit_grid, last, axis=1)
    fp_limit = jnp.where(is_pad, edge_limit, limit_grid)
    limit = jax.vmap(jnp.interp)(alpha_pred, xp, fp_limit)
    return dmag0, limit
