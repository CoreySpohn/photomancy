"""Log-likelihood functions for orbit fitting -- pure JAX.

All functions are pure JAX with no sampler dependency. They accept predicted
observables (from the forward models) and data containers, returning a scalar
log-likelihood value. All are JIT-compilable and differentiable.

Static-shape masking
--------------------
All likelihoods respect the ``is_valid`` mask on data containers. Padded
entries (``is_valid == False``) contribute exactly zero to the sum.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

# Finite "undetectable" dMag0 sentinel (replaces -inf): the predicted flux ratio
# still underflows to 0 (no non-detection penalty), but staying finite keeps the
# interpolation gradient finite on padded grids.
_UNDETECTABLE_DMAG0 = -1e8


def _interp_dmag0(alpha_pred, sep_grid, dmag0_grid):
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

    Returns:
        The interpolated dMag0 limit per epoch. Shape ``(N,)``.
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

    return jax.vmap(_one)(alpha_pred, xp, fp)


def loglike_rv_marginalized(
    rv_obs, rv_model, rv_err, inst_ids, n_inst, jitters, is_valid
):
    """RV log-likelihood with analytically marginalized per-instrument offsets.

    Because the instrument design matrix is an indicator matrix, the analytical
    marginalization of zero-points gamma_j collapses to O(N) segment sums. This
    avoids sampling ``n_inst`` nuisance parameters.

    Args:
        rv_obs: Observed RVs (AU/day). Shape ``(N,)``.
        rv_model: Predicted RVs from forward model (AU/day). Shape ``(N,)``.
        rv_err: RV measurement uncertainties (AU/day). Shape ``(N,)``.
        inst_ids: Integer instrument index per obs, ``0..n_inst-1``.
            Shape ``(N,)``.
        n_inst: Number of distinct instruments. Integer.
        jitters: Per-instrument jitter terms (AU/day). Shape ``(n_inst,)``.
        is_valid: Boolean validity mask. Shape ``(N,)``.

    Returns:
        Scalar log-likelihood value.
    """
    # Mask invalid entries -- use jnp.where for gradient safety
    mask = is_valid  # boolean

    # Total variance: measurement + jitter
    var = rv_err**2 + jitters[inst_ids] ** 2
    w = jnp.where(mask, 1.0 / var, 0.0)  # masked weights
    r = rv_obs - rv_model

    # Segment sums collapse the matrix algebra to O(N)
    A = jax.ops.segment_sum(w, inst_ids, num_segments=n_inst)
    B = jax.ops.segment_sum(w * r, inst_ids, num_segments=n_inst)
    C = jax.ops.segment_sum(w * r**2, inst_ids, num_segments=n_inst)

    # Marginalized log-likelihood -- only count valid observations
    N_valid = jnp.sum(mask)
    return -0.5 * (
        jnp.sum(C - B**2 / jnp.where(A > 0, A, 1.0))
        + jnp.sum(jnp.where(A > 0, jnp.log(A), 0.0))
        + jnp.sum(jnp.where(mask, jnp.log(var), 0.0))
        + N_valid * jnp.log(2.0 * jnp.pi)
    )


def loglike_astrom(ra_pred, dec_pred, data):
    """Relative astrometry log-likelihood (bivariate Gaussian with correlation).

    Args:
        ra_pred: Predicted RA offsets (arcsec). Shape ``(N,)``.
        dec_pred: Predicted DEC offsets (arcsec). Shape ``(N,)``.
        data: An :class:`~photomancy.orbit.data.AstromData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    mask = data.is_valid  # boolean

    dx = data.ra - ra_pred
    dy = data.dec - dec_pred
    rho = data.corr
    sx = data.ra_err
    sy = data.dec_err

    # Bivariate Gaussian quadratic form
    one_minus_rho2 = jnp.clip(1.0 - rho**2, 1e-30)
    z = (
        dx**2 / sx**2 + dy**2 / sy**2 - 2.0 * rho * dx * dy / (sx * sy)
    ) / one_minus_rho2

    # Log normalization
    log_norm = jnp.log(sx * sy * jnp.sqrt(one_minus_rho2)) + jnp.log(2.0 * jnp.pi)

    # Gradient-safe masking: jnp.where prevents NaN grad leakage
    z_safe = jnp.where(mask, z, 0.0)
    log_norm_safe = jnp.where(mask, log_norm, 0.0)
    return -0.5 * jnp.sum(z_safe) - jnp.sum(log_norm_safe)


def loglike_null(alpha_pred, dMag_pred, data):
    """Non-detection log-likelihood via exact flux-space z-score.

    Uses :func:`jax.scipy.special.log_ndtr` for the cumulative normal
    log-probability, which is numerically stable at extreme z-scores.

    Sign convention: larger dMag = fainter. If a planet is brighter than
    the detection limit (``dMag_pred < dMag0_limit``), the flux ratio
    exceeds 1, producing a negative z-score and a heavy log-likelihood
    penalty.

    Planets outside the contrast curve receive zero penalty: the limit maps to a
    finite "undetectable" sentinel (see :func:`_interp_dmag0`) giving
    ``flux_ratio -> 0`` -> ``z = snr_thresh`` -> ``log_ndtr ~= 0``.

    Args:
        alpha_pred: Predicted angular separation (arcsec). Shape
            ``(N_epochs,)``.
        dMag_pred: Predicted delta-magnitude. Shape ``(N_epochs,)``.
        data: A :class:`~photomancy.orbit.data.NullData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    mask = data.is_valid  # boolean

    dMag0_limit = _interp_dmag0(alpha_pred, data.sep_grid, data.dmag0_grid)

    # Exact flux ratio: F_pred / F_limit
    # Clip the exponent to prevent float64 overflow (10**309 = inf)
    # when the sampler proposes an implausibly bright orbit.
    exp_arg = jnp.clip(-0.4 * (dMag_pred - dMag0_limit), max=200.0)
    flux_ratio = 10.0**exp_arg

    # z > 0: planet fainter than limit -> no penalty
    # z < 0: planet brighter -> should have been detected -> penalty
    z = data.snr_thresh * (1.0 - flux_ratio)

    return jnp.sum(jnp.where(mask, log_ndtr(z), 0.0))


def loglike_imaging(alpha_pred, dMag_pred, data):
    """Unified imaging log-likelihood for detection + null epochs.

    At **detection epochs** (``data.is_detected == True``): evaluates a
    Gaussian photometric likelihood comparing the predicted dMag to the
    measured ``dMag_obs +/- dMag_err``. This provides an unbounded parabolic
    penalty that prevents the optimizer from crushing Lambda to zero.

    At **null epochs** (``data.is_detected == False``): uses the existing
    flux-space z-score approach via :func:`loglike_null`.

    The two branches are combined with :func:`jnp.where` for branchless
    JIT compatibility.

    Args:
        alpha_pred: Predicted angular separation (arcsec). Shape
            ``(M,)``.
        dMag_pred: Predicted delta-magnitude. Shape ``(M,)``.
        data: An :class:`~photomancy.orbit.data.ImagingData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    mask = data.is_valid  # boolean

    # --- Null branch: flux-space z-score ---
    dMag0_limit = _interp_dmag0(alpha_pred, data.sep_grid, data.dmag0_grid)
    exp_arg = jnp.clip(-0.4 * (dMag_pred - dMag0_limit), max=200.0)
    flux_ratio = 10.0**exp_arg
    z = data.snr_thresh * (1.0 - flux_ratio)
    ll_null = log_ndtr(z)

    # --- Detection branch: Gaussian on measured dMag ---
    residual = dMag_pred - data.dmag_obs
    safe_err = jnp.where(data.dmag_err > 0, data.dmag_err, 1.0)
    ll_det = -0.5 * (residual / safe_err) ** 2 - jnp.log(
        safe_err * jnp.sqrt(2.0 * jnp.pi)
    )

    # --- Branchless selection ---
    ll = jnp.where(data.is_detected, ll_det, ll_null)
    return jnp.sum(jnp.where(mask, ll, 0.0))
