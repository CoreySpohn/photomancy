"""Data containers for orbit fitting observations.

All containers are :class:`equinox.Module` instances -- immutable, pytree-
compatible, and JIT-friendly. Field values are raw JAX arrays in natural
units (arcsec, m/s, days since J2000, dimensionless).

Static-shape padding
--------------------
For JIT-cache stability, data containers support padding to fixed
``MAX_*`` sizes via the :meth:`pad` classmethod. Padded entries have
``is_valid = False`` and are masked out in the likelihood functions.
"""

import equinox as eqx
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Static shape limits (for JIT-cacheable fitting)
# ---------------------------------------------------------------------------
MAX_ASTROM = 64
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


# ---------------------------------------------------------------------------
# RVData
# ---------------------------------------------------------------------------


class RVData(eqx.Module):
    """Radial velocity observations from one or more instruments.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        rv: Measured radial velocities (m/s). Shape ``(N,)``.
        rv_err: RV measurement uncertainties (m/s). Shape ``(N,)``.
        inst_ids: Integer instrument index per observation, ``0..n_inst-1``.
            Shape ``(N,)``.
        is_valid: Boolean validity mask. Shape ``(N,)``.
        n_inst: Number of distinct instruments (static, for ``segment_sum``).
    """

    times: jnp.ndarray
    rv: jnp.ndarray
    rv_err: jnp.ndarray
    inst_ids: jnp.ndarray
    is_valid: jnp.ndarray
    n_inst: int = eqx.field(static=True)

    @classmethod
    def pad(cls, *, times, rv, rv_err, inst_ids, n_inst, max_n=MAX_RV):
        """Create a padded RVData with ``is_valid`` mask."""
        n = jnp.asarray(times).shape[0]
        return cls(
            times=_pad_1d(times, max_n),
            rv=_pad_1d(rv, max_n),
            rv_err=_pad_1d(rv_err, max_n, fill=1.0),
            inst_ids=_pad_1d(inst_ids, max_n).astype(int),
            is_valid=_valid_mask(n, max_n),
            n_inst=n_inst,
        )

    @classmethod
    def zeros(cls, max_n=MAX_RV, n_inst=1):
        """Create an all-invalid placeholder (for model tracing)."""
        return cls(
            times=jnp.zeros(max_n),
            rv=jnp.zeros(max_n),
            rv_err=jnp.ones(max_n),
            inst_ids=jnp.zeros(max_n, dtype=int),
            is_valid=jnp.zeros(max_n, dtype=bool),
            n_inst=n_inst,
        )


# ---------------------------------------------------------------------------
# AstromData
# ---------------------------------------------------------------------------


class AstromData(eqx.Module):
    """Relative astrometry observations.

    Positions are measured in arcseconds relative to the host star.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        ra: Relative RA offset (arcsec). Shape ``(N,)``.
        dec: Relative DEC offset (arcsec). Shape ``(N,)``.
        ra_err: RA uncertainty (arcsec). Shape ``(N,)``.
        dec_err: DEC uncertainty (arcsec). Shape ``(N,)``.
        corr: RA/DEC correlation coefficient. Shape ``(N,)``.
        planet_id: Planet index per observation (for multi-planet systems).
            Shape ``(N,)``.
        is_valid: Boolean validity mask. Shape ``(N,)``.
    """

    times: jnp.ndarray
    ra: jnp.ndarray
    dec: jnp.ndarray
    ra_err: jnp.ndarray
    dec_err: jnp.ndarray
    corr: jnp.ndarray
    planet_id: jnp.ndarray
    is_valid: jnp.ndarray

    @classmethod
    def pad(cls, *, times, ra, dec, ra_err, dec_err, corr, planet_id, max_n=MAX_ASTROM):
        """Create a padded AstromData with ``is_valid`` mask."""
        n = jnp.asarray(times).shape[0]
        return cls(
            times=_pad_1d(times, max_n),
            ra=_pad_1d(ra, max_n),
            dec=_pad_1d(dec, max_n),
            ra_err=_pad_1d(ra_err, max_n, fill=1.0),
            dec_err=_pad_1d(dec_err, max_n, fill=1.0),
            corr=_pad_1d(corr, max_n),
            planet_id=_pad_1d(planet_id, max_n).astype(int),
            is_valid=_valid_mask(n, max_n),
        )

    @classmethod
    def zeros(cls, max_n=MAX_ASTROM):
        """Create an all-invalid placeholder (for model tracing)."""
        return cls(
            times=jnp.zeros(max_n),
            ra=jnp.zeros(max_n),
            dec=jnp.zeros(max_n),
            ra_err=jnp.ones(max_n),
            dec_err=jnp.ones(max_n),
            corr=jnp.zeros(max_n),
            planet_id=jnp.zeros(max_n, dtype=int),
            is_valid=jnp.zeros(max_n, dtype=bool),
        )


# ---------------------------------------------------------------------------
# NullData (legacy -- kept for backward compat)
# ---------------------------------------------------------------------------


class NullData(eqx.Module):
    """Non-detection (null) observation data.

    Stores the detection threshold as a dMag grid per epoch. Both Tier 1
    (static contrast curves) and Tier 2 (physics-aware ``dMag0Grid``) data
    are stored in the same format after pre-slicing.

    Args:
        epochs: Observation epochs (days). Shape ``(N_epochs,)``.
        sep_grid: Separation grid per epoch (arcsec), padded and monotonic.
            Shape ``(N_epochs, N_pts)``.
        dmag0_grid: Limiting dMag per epoch, padded with ``-jnp.inf`` for
            undetectable regions. Shape ``(N_epochs, N_pts)``.
        is_valid: Boolean validity mask. Shape ``(N_epochs,)``.
        snr_thresh: SNR detection threshold (static). Default 5.0.
    """

    epochs: jnp.ndarray
    sep_grid: jnp.ndarray
    dmag0_grid: jnp.ndarray
    is_valid: jnp.ndarray
    snr_thresh: float = eqx.field(static=True, default=5.0)

    @classmethod
    def from_contrast_curves(
        cls,
        epochs,
        sep_grids,
        contrast_grids,
        snr_thresh=5.0,
        max_n=MAX_IMG,
        max_pts=MAX_CC_PTS,
    ):
        """Create NullData from contrast curves (Tier 1).

        Converts contrast values to dMag space: ``dMag = -2.5 * log10(contrast)``.

        Args:
            epochs: Observation epochs. Shape ``(N_epochs,)``.
            sep_grids: Separation grids per epoch (arcsec).
                Shape ``(N_epochs, N_pts)``.
            contrast_grids: Contrast detection limits per epoch.
                Shape ``(N_epochs, N_pts)``.
            snr_thresh: SNR detection threshold.
            max_n: Padding size for epochs.
            max_pts: Padding size for contrast curve points.

        Returns:
            A ``NullData`` instance with the contrast curves converted to dMag.
        """
        epochs = jnp.asarray(epochs)
        sep_grids = jnp.asarray(sep_grids)
        dmag0_grid = -2.5 * jnp.log10(contrast_grids)
        n = epochs.shape[0]
        return cls(
            epochs=_pad_1d(epochs, max_n),
            sep_grid=_pad_2d(sep_grids, max_n, max_pts),
            dmag0_grid=_pad_2d(dmag0_grid, max_n, max_pts, fill=-jnp.inf),
            is_valid=_valid_mask(n, max_n),
            snr_thresh=snr_thresh,
        )

    @classmethod
    def pad(
        cls,
        *,
        epochs,
        sep_grid,
        dmag0_grid,
        snr_thresh=5.0,
        max_n=MAX_IMG,
        max_pts=MAX_CC_PTS,
    ):
        """Create a padded NullData with an ``is_valid`` mask."""
        n = jnp.asarray(epochs).shape[0]
        return cls(
            epochs=_pad_1d(epochs, max_n),
            sep_grid=_pad_2d(sep_grid, max_n, max_pts),
            dmag0_grid=_pad_2d(dmag0_grid, max_n, max_pts, fill=-jnp.inf),
            is_valid=_valid_mask(n, max_n),
            snr_thresh=snr_thresh,
        )

    @classmethod
    def zeros(cls, max_n=MAX_IMG, max_pts=MAX_CC_PTS, snr_thresh=5.0):
        """Create an all-invalid placeholder."""
        return cls(
            epochs=jnp.zeros(max_n),
            sep_grid=jnp.zeros((max_n, max_pts)),
            dmag0_grid=jnp.full((max_n, max_pts), -jnp.inf),
            is_valid=jnp.zeros(max_n, dtype=bool),
            snr_thresh=snr_thresh,
        )


# ---------------------------------------------------------------------------
# ImagingData
# ---------------------------------------------------------------------------


class ImagingData(eqx.Module):
    """Unified imaging data for detection and null epochs.

    Combines astrometric detection epochs (with measured brightness) and
    null detection epochs (with contrast-curve limits) into a single
    JIT-friendly container. The ``is_detected`` mask selects between
    a Gaussian photometric likelihood (detections) and the flux-space
    z-score non-detection likelihood (nulls) via ``jnp.where``.

    Args:
        epochs: Observation epochs (days). Shape ``(M,)``.
        sep_grid: Separation grid per epoch (arcsec), padded and monotonic.
            Shape ``(M, K)``.
        dmag0_grid: Limiting dMag per epoch, padded with ``-jnp.inf`` for
            undetectable regions. Shape ``(M, K)``.
        snr_thresh: SNR detection threshold (static). Default 5.0.
        is_detected: Boolean mask -- ``True`` for detection epochs,
            ``False`` for null epochs. Shape ``(M,)``.
        dmag_obs: Measured delta-magnitude at detection epochs. Set to 0.0
            for null epochs (unused). Shape ``(M,)``.
        dmag_err: Measurement uncertainty on dMag at detection epochs.
            Set to 1.0 for null epochs (unused). Shape ``(M,)``.
        is_valid: Boolean validity mask. Shape ``(M,)``.
    """

    epochs: jnp.ndarray
    sep_grid: jnp.ndarray
    dmag0_grid: jnp.ndarray
    is_detected: jnp.ndarray
    dmag_obs: jnp.ndarray
    dmag_err: jnp.ndarray
    is_valid: jnp.ndarray
    snr_thresh: float = eqx.field(static=True, default=5.0)

    @classmethod
    def from_detections_and_nulls(
        cls,
        det_epochs,
        det_dmag_obs,
        det_dmag_err,
        det_sep_grid,
        det_dmag0_grid,
        null_epochs,
        null_sep_grid,
        null_dmag0_grid,
        snr_thresh=5.0,
        max_n=MAX_IMG,
        max_pts=MAX_CC_PTS,
    ):
        """Build from separate detection and null arrays with padding.

        Concatenates detection and null epochs into a single container
        with the ``is_detected`` mask set accordingly, then pads to
        static shape.

        Args:
            det_epochs: Detection observation epochs (days). Shape ``(N_det,)``.
            det_dmag_obs: Measured dMag at detection epochs. Shape ``(N_det,)``.
            det_dmag_err: dMag uncertainty at detection epochs. Shape ``(N_det,)``.
            det_sep_grid: Contrast-curve separation grid for detection epochs.
                Shape ``(N_det, K)``.
            det_dmag0_grid: Contrast-curve dMag limit for detection epochs.
                Shape ``(N_det, K)``.
            null_epochs: Null observation epochs (days). Shape ``(N_null,)``.
            null_sep_grid: Contrast-curve separation grid for null epochs.
                Shape ``(N_null, K)``.
            null_dmag0_grid: Contrast-curve dMag limit for null epochs.
                Shape ``(N_null, K)``.
            snr_thresh: SNR detection threshold.
            max_n: Padding size for total epochs.
            max_pts: Padding size for contrast curve grid points.

        Returns:
            An ``ImagingData`` instance.
        """
        det_epochs = jnp.asarray(det_epochs)
        null_epochs = jnp.asarray(null_epochs)
        n_det = det_epochs.shape[0]
        n_null = null_epochs.shape[0]
        n_total = n_det + n_null

        epochs = jnp.concatenate([det_epochs, null_epochs])
        sep_grid = jnp.concatenate(
            [jnp.asarray(det_sep_grid), jnp.asarray(null_sep_grid)]
        )
        dmag0_grid = jnp.concatenate(
            [jnp.asarray(det_dmag0_grid), jnp.asarray(null_dmag0_grid)]
        )
        is_detected = jnp.concatenate(
            [jnp.ones(n_det, dtype=bool), jnp.zeros(n_null, dtype=bool)]
        )
        dmag_obs = jnp.concatenate([jnp.asarray(det_dmag_obs), jnp.zeros(n_null)])
        dmag_err = jnp.concatenate([jnp.asarray(det_dmag_err), jnp.ones(n_null)])

        # Pad sep_grid columns to max_pts

        return cls(
            epochs=_pad_1d(epochs, max_n),
            sep_grid=_pad_2d(sep_grid, max_n, max_pts),
            dmag0_grid=_pad_2d(dmag0_grid, max_n, max_pts, fill=-jnp.inf),
            is_detected=_pad_1d(is_detected, max_n).astype(bool),
            dmag_obs=_pad_1d(dmag_obs, max_n),
            dmag_err=_pad_1d(dmag_err, max_n, fill=1.0),
            is_valid=_valid_mask(n_total, max_n),
            snr_thresh=snr_thresh,
        )

    @classmethod
    def pad(
        cls,
        *,
        epochs,
        sep_grid,
        dmag0_grid,
        is_detected,
        dmag_obs,
        dmag_err,
        snr_thresh=5.0,
        max_n=MAX_IMG,
        max_pts=MAX_CC_PTS,
    ):
        """Create a padded ImagingData with an ``is_valid`` mask."""
        n = jnp.asarray(epochs).shape[0]
        return cls(
            epochs=_pad_1d(epochs, max_n),
            sep_grid=_pad_2d(sep_grid, max_n, max_pts),
            dmag0_grid=_pad_2d(dmag0_grid, max_n, max_pts, fill=-jnp.inf),
            is_detected=_pad_1d(is_detected, max_n).astype(bool),
            dmag_obs=_pad_1d(dmag_obs, max_n),
            dmag_err=_pad_1d(dmag_err, max_n, fill=1.0),
            is_valid=_valid_mask(n, max_n),
            snr_thresh=snr_thresh,
        )

    @classmethod
    def zeros(cls, max_n=MAX_IMG, max_pts=MAX_CC_PTS, snr_thresh=5.0):
        """Create an all-invalid placeholder (for model tracing)."""
        return cls(
            epochs=jnp.zeros(max_n),
            sep_grid=jnp.zeros((max_n, max_pts)),
            dmag0_grid=jnp.full((max_n, max_pts), -jnp.inf),
            is_detected=jnp.zeros(max_n, dtype=bool),
            dmag_obs=jnp.zeros(max_n),
            dmag_err=jnp.ones(max_n),
            is_valid=jnp.zeros(max_n, dtype=bool),
            snr_thresh=snr_thresh,
        )
