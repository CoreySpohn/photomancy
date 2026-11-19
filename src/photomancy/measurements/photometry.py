"""Photometry of a planetary system: contrasts, dMag detections and nulls.

Linear contrasts with upper limits (``PhotometryTerm``), single-planet dMag
detections with censored contrast-curve nulls (``DMagTerm``), the padded contrast-curve
interpolation both use, and the achromatic Lambertian prediction of a one-planet system.
Data times are days from the system's epoch.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import contrast_to_delta_mag, delta_mag_to_contrast
from hwoutils.model_paths import resolve
from jax.scipy.stats import norm
from optixstuff.spectral import Quadrature, Transmission
from orbix.system import System

from photomancy.infer.fit import AbstractTerm
from photomancy.measurements._observables import _band_contrasts
from photomancy.measurements._padding import (
    MAX_CC_PTS,
    MAX_IMG,
    _pad_1d,
    _pad_2d,
    _valid_mask,
)
from photomancy.measurements.report import ReportLaw, ReportPrediction

__all__ = [
    "DMagTerm",
    "ImagingData",
    "NullData",
    "PhotometryData",
    "PhotometryTerm",
    "interpolate_contrast_curve",
    "lambertian_dmag",
    "lambertian_from_area",
    "loglike_imaging",
    "loglike_null",
    "system_separation_dmag",
]


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
        dmag0_grid = contrast_to_delta_mag(contrast_grids)
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


class PhotometryData(eqx.Module):
    """Linear planet/star contrast measurements and upper limits.

    All fields have shape ``(N,)``. ``times`` uses the system's day frame;
    ``contrast`` and ``contrast_err`` are dimensionless linear flux ratios.
    ``band_id`` and ``planet_id`` index the term's bandpasses and system planets.
    ``is_upper_limit`` selects a Gaussian cumulative probability below the
    reported contrast; ``is_valid`` excludes padded rows.
    """

    times: jnp.ndarray
    contrast: jnp.ndarray
    contrast_err: jnp.ndarray
    band_id: jnp.ndarray
    planet_id: jnp.ndarray
    is_upper_limit: jnp.ndarray
    is_valid: jnp.ndarray

    @classmethod
    def pad(
        cls,
        *,
        times,
        contrast,
        contrast_err,
        band_id,
        planet_id,
        is_upper_limit,
        max_n=MAX_IMG,
    ):
        """Pad contrast observations, using unit errors on invalid rows."""
        n = jnp.asarray(times).shape[0]
        return cls(
            times=_pad_1d(times, max_n),
            contrast=_pad_1d(contrast, max_n),
            contrast_err=_pad_1d(contrast_err, max_n, fill=1.0),
            band_id=_pad_1d(band_id, max_n).astype(int),
            planet_id=_pad_1d(planet_id, max_n).astype(int),
            is_upper_limit=_pad_1d(is_upper_limit, max_n).astype(bool),
            is_valid=_valid_mask(n, max_n),
        )

    @classmethod
    def from_dmag(
        cls,
        *,
        times,
        dmag,
        dmag_err,
        band_id,
        planet_id=None,
        is_upper_limit=None,
        max_n=None,
    ):
        """Convert magnitude differences with first-order Gaussian errors.

        ``contrast = 10**(-0.4*dmag)`` and its standard deviation is
        ``0.4*ln(10)*contrast*dmag_err``. Upper-limit rows use that same
        conversion for the limit and its noise scale. Planet indices default
        to zero, limit flags to False, and ``max_n=None`` keeps all rows valid.
        """
        times = jnp.asarray(times)
        contrast = delta_mag_to_contrast(jnp.asarray(dmag))
        if planet_id is None:
            planet_id = jnp.zeros(times.shape, dtype=int)
        if is_upper_limit is None:
            is_upper_limit = jnp.zeros(times.shape, dtype=bool)
        values = dict(
            times=times,
            contrast=contrast,
            contrast_err=0.4 * jnp.log(10.0) * contrast * jnp.asarray(dmag_err),
            band_id=jnp.asarray(band_id, dtype=int),
            planet_id=jnp.asarray(planet_id, dtype=int),
            is_upper_limit=jnp.asarray(is_upper_limit, dtype=bool),
        )
        if max_n is not None:
            return cls.pad(**values, max_n=max_n)
        return cls(**values, is_valid=jnp.ones(times.shape, dtype=bool))


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
    # The artificial tail keeps interpolation well defined. A query beyond
    # the last real knot is undetectable even within that tail.
    dmag0 = jnp.where(alpha_pred > max_real[:, 0], _UNDETECTABLE_DMAG0, dmag0)
    if limit_grid is None:
        return dmag0
    real_sep = jnp.where(is_pad, -jnp.inf, sep_grid)
    last = jnp.argmax(real_sep, axis=1)[:, None]
    edge_limit = jnp.take_along_axis(limit_grid, last, axis=1)
    fp_limit = jnp.where(is_pad, edge_limit, limit_grid)
    limit = jax.vmap(jnp.interp)(alpha_pred, xp, fp_limit)
    return dmag0, limit


def lambertian_from_area(area_AU2):
    """Map Lambda = Ag Rp^2 to a unit-albedo model and radius in Earth radii.

    Reflected light identifies only the area product. Fix Ag=1 and put
    the sampled scale into radius; both returned arrays have shape (K,).
    """
    # Delay scene imports, keeping their module constants concrete under first JIT.
    with jax.ensure_compile_time_eval():
        from skyscapes.physical_model import LambertianPhysicalModel

    area = jnp.atleast_1d(area_AU2)
    return LambertianPhysicalModel(Ag=jnp.ones_like(area)), jnp.sqrt(area) / Rearth2AU


def lambertian_dmag(phase_rad, distance_AU, model, radius_Rearth):
    """Convert a Lambertian reflector's contrast to dMag at the orbit-fit floor."""
    contrast = model.contrast(
        phase_rad[None, :], distance_AU[None, :], None, radius_Rearth
    )[0]
    return contrast_to_delta_mag(jnp.maximum(contrast, 1e-300))


def system_separation_dmag(system, time_d, *, area_AU2=None):
    """Predict separation (arcsec) and dMag at days from a one-planet epoch.

    A scene supplies its Lambertian reflector and radius. An explicit
    geometric-albedo times radius squared (AU^2) supplies the reflector
    for a bare dynamical system or a known photometric scale.
    """
    bare = isinstance(system, System)
    count = system.orbits.a_AU.size if bare else system.n_planets
    if count != 1:
        raise ValueError(
            "Multi-planet null and imaging require a detection model per planet"
        )
    if area_AU2 is None:
        if bare:
            raise ValueError(
                "A dynamical system needs area_AU2 for Lambertian photometry"
            )
        with jax.ensure_compile_time_eval():
            from skyscapes.physical_model import LambertianPhysicalModel

        planet = system.planets[0]
        model = planet.physical_model
        if not isinstance(model, LambertianPhysicalModel):
            raise ValueError(
                "Imaging and null orbit terms require a Lambertian reflector"
            )
        radius_Rearth = planet.Rp_Rearth
    else:
        if jnp.size(area_AU2) != 1:
            raise ValueError("area_AU2 must specify one planet's photometric area")
        model, radius_Rearth = lambertian_from_area(jnp.reshape(area_AU2, (1,)))
    if not bare:
        radius_Rearth = jnp.where(system.valid, radius_Rearth, 0.0)
    ephemeris = system.solve(time_d) if bare else system.ephemeris(time_d)
    return ephemeris.separation_arcsec(1, 0), lambertian_dmag(
        ephemeris.phase_angle_rad(1, 0),
        ephemeris.distance_AU(1, 0),
        model,
        radius_Rearth,
    )


def _fixed_noise_null(alpha_pred, dMag_pred, data):
    """Score censored photometry in units of planet-free noise."""
    dmag0 = interpolate_contrast_curve(alpha_pred, data.sep_grid, data.dmag0_grid)
    # Clip the exponent when the sampler proposes an implausibly bright orbit.
    exp_arg = jnp.clip(-0.4 * (dMag_pred - dmag0), max=200.0)
    law = ReportLaw(threshold=data.snr_thresh)
    counts = law.detection_limit(1.0) * 10.0**exp_arg
    zero = jnp.zeros_like(counts)
    prediction = ReportPrediction(counts, jnp.ones_like(counts), zero, zero)
    return law.log_null_probability(1.0, prediction)


def loglike_null(alpha_pred, dMag_pred, data):
    """Non-detection log-likelihood via exact flux-space z-score.

    Uses the reporting law with fixed noise in flux space.

    Sign convention: larger dMag = fainter. If a planet is brighter than
    the detection limit (``dMag_pred < dMag0_limit``), the flux ratio
    exceeds 1, producing a negative z-score and a heavy log-likelihood
    penalty.

    Outside the contrast curve, the finite "undetectable" sentinel from
    :func:`interpolate_contrast_curve` gives a flux ratio near zero and a null
    probability near one.

    Args:
        alpha_pred: Predicted angular separation (arcsec). Shape
            ``(N_epochs,)``.
        dMag_pred: Predicted delta-magnitude. Shape ``(N_epochs,)``.
        data: A :class:`~photomancy.measurements.photometry.NullData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    return jnp.sum(
        jnp.where(data.is_valid, _fixed_noise_null(alpha_pred, dMag_pred, data), 0.0)
    )


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
        data: An :class:`~photomancy.measurements.photometry.ImagingData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    mask = data.is_valid  # boolean

    ll_null = _fixed_noise_null(alpha_pred, dMag_pred, data)

    # --- Detection branch: Gaussian on measured dMag ---
    residual = dMag_pred - data.dmag_obs
    safe_err = jnp.where(data.dmag_err > 0, data.dmag_err, 1.0)
    ll_det = -0.5 * (residual / safe_err) ** 2 - jnp.log(
        safe_err * jnp.sqrt(2.0 * jnp.pi)
    )

    # --- Branchless selection ---
    ll = jnp.where(data.is_detected, ll_det, ll_null)
    return jnp.sum(jnp.where(mask, ll, 0.0))


class PhotometryTerm(AbstractTerm):
    """Score linear contrasts using Gaussian detections or Gaussian upper limits.

    Build ``quadrature`` on the host with the system and filter curves. Its
    bands and ``bandpasses`` follow ``data.band_id`` order. The system at
    ``system_path`` (the empty path is the model itself) returns ``(W, K, T)`` band
    contrasts. Each row selects its own band, planet and epoch. An upper limit
    contributes ``log Phi((limit - prediction) / contrast_err)``. Invalid rows return
    zero. Data ``times`` are days from the system's epoch. Bands share one ephemeris.
    """

    data: PhotometryData
    quadrature: Quadrature
    bandpasses: tuple[Transmission, ...]
    system_path: str = eqx.field(static=True, default="")
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require a bandpass for contrast prediction."""
        if len(self.bandpasses) != len(self.quadrature.bands):
            raise ValueError("PhotometryTerm needs one bandpass per quadrature band")

    def _score(self, model):
        valid = self.data.is_valid
        data = PhotometryData(
            times=jnp.where(valid, self.data.times, 0.0),
            contrast=jnp.where(valid, self.data.contrast, 0.0),
            contrast_err=jnp.where(valid, self.data.contrast_err, 1.0),
            band_id=jnp.where(valid, self.data.band_id, 0),
            planet_id=jnp.where(valid, self.data.planet_id, 0),
            is_upper_limit=jnp.where(valid, self.data.is_upper_limit, False),
            is_valid=valid,
        )
        system = resolve(model, self.system_path)
        prediction = _band_contrasts(
            system,
            self.quadrature,
            self.bandpasses,
            data.times,
            data.band_id,
            data.planet_id,
        )
        prediction = jnp.where(valid, prediction, 0.0)
        z = (data.contrast - prediction) / data.contrast_err
        detection = -0.5 * z**2 - jnp.log(data.contrast_err * jnp.sqrt(2.0 * jnp.pi))
        rows = jnp.where(data.is_upper_limit, norm.logcdf(z), detection)
        return jnp.sum(jnp.where(valid, rows, 0.0)), prediction

    def __call__(self, model):
        """Return the scalar log-likelihood."""
        return self._score(model)[0]

    def predict(self, model):
        """Return each row's linear contrast, ``(T,)``."""
        return self._score(model)[1]

    def draw(self, key, model) -> Self:
        """Draw Gaussian contrasts, censoring at each stored upper limit.

        An upper-limit row retains its threshold and flag when the draw lies
        below the threshold. Above it, the row becomes a measured detection.
        Rows already measured as detections remain uncensored Gaussian draws.
        """
        data = self.data
        valid = data.is_valid
        sigma = jnp.where(valid, data.contrast_err, 1.0)
        draw = self.predict(model) + sigma * jax.random.normal(
            key, data.contrast.shape, dtype=jnp.float64
        )
        censored = data.is_upper_limit & (draw < data.contrast)
        contrast = jnp.where(censored, data.contrast, draw)
        sampled = eqx.tree_at(
            lambda d: (d.contrast, d.is_upper_limit),
            data,
            (
                jnp.where(valid, contrast, data.contrast),
                jnp.where(valid, censored, data.is_upper_limit),
            ),
        )
        return eqx.tree_at(lambda t: t.data, self, sampled)


class DMagTerm(AbstractTerm):
    """Score single-planet Gaussian dMag measurements and contrast-curve nulls.

    This term scores catalog-level measurements, not pixels (the pixel likelihood is
    :class:`photomancy.image.ImageTerm`). The prediction is a pair of vectors
    (separation in arcsec, dMag), zero on padded rows. Measured dMag is a Gaussian
    observable at detected epochs; null epochs use the censored flux likelihood of the
    supplied contrast curve. A data set of nulls alone is an all-False ``is_detected``.

    A scene supplies its Lambertian albedo and radius. For a bare dynamical system, or
    to override the scene, give the photometric area ``Ag * Rp**2`` (AU^2) as the
    ``area_AU2`` field or as a model leaf at ``area_path``.

    Attributes:
        data: Detection and null epochs with their contrast curves.
        system_path: Model path of the one-planet system; the empty path is the model.
        area_AU2: Fixed photometric area, or None.
        area_path: Model path of the photometric area, or None.
    """

    data: ImagingData
    system_path: str = eqx.field(static=True, default="")
    area_AU2: jax.Array | None = None
    area_path: str | None = eqx.field(static=True, default=None)
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require at most one source of the photometric area."""
        if self.area_AU2 is not None and self.area_path is not None:
            raise ValueError("Choose a fixed area_AU2 or an area_path, not both")

    def _area(self, model, area_AU2):
        if area_AU2 is not None:
            if self.area_AU2 is not None or self.area_path is not None:
                raise ValueError("Choose fixed or sampled photometric area")
            return area_AU2
        if self.area_path is not None:
            return resolve(model, self.area_path)
        return self.area_AU2

    def _prediction(self, model, area_AU2):
        valid = self.data.is_valid
        prediction = system_separation_dmag(
            resolve(model, self.system_path),
            jnp.where(valid, self.data.epochs, 0.0),
            area_AU2=self._area(model, area_AU2),
        )
        return tuple(jnp.where(valid, x, 0.0) for x in prediction)

    def __call__(self, model, *, area_AU2=None):
        """Return the combined dMag and null log likelihood.

        The ``area_AU2`` keyword is for a caller that samples the area as its own
        coordinate; a ``Fit`` calls with the model alone.
        """
        return loglike_imaging(*self._prediction(model, area_AU2), self.data)

    def predict(self, model, *, area_AU2=None):
        """Return (separation in arcsec, dMag) at each epoch, zero on padded rows."""
        return self._prediction(model, area_AU2)

    def draw(self, key, model, *, area_AU2=None) -> Self:
        """Copy with Gaussian dMag draws at the detected epochs.

        Null epochs are conditioning data and keep their contrast curves unchanged.
        """
        data = self.data
        mask = data.is_valid & data.is_detected
        sigma = jnp.where(mask, data.dmag_err, 1.0)
        dmag = self.predict(model, area_AU2=area_AU2)[1]
        draw = dmag + sigma * jax.random.normal(key, dmag.shape, dtype=jnp.float64)
        sampled = eqx.tree_at(
            lambda d: d.dmag_obs, data, jnp.where(mask, draw, data.dmag_obs)
        )
        return eqx.tree_at(lambda t: t.data, self, sampled)
