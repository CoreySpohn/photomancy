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
from optixstuff.spectral import Quadrature, Resolution, Transmission
from orbix.system import System

from photomancy.infer.fit import AbstractTerm
from photomancy.measurements._observables import (
    _band_contrasts,
    _band_table,
    _check_indices,
    _check_planets,
)
from photomancy.measurements._padding import (
    MAX_CC_PTS,
    _pad_1d,
    _pad_2d,
    _valid_mask,
)
from photomancy.measurements.report import ReportPrediction, ReportRule

__all__ = [
    "DMagData",
    "DMagTerm",
    "PhotometryData",
    "PhotometryTerm",
    "interpolate_contrast_curve",
    "lambertian_dmag",
    "lambertian_from_area",
    "loglike_dmag",
    "system_separation_dmag",
]


class DMagData(eqx.Module):
    """Single-planet dMag detections and contrast-curve nulls.

    Combines astrometric detection epochs (with measured brightness) and
    null detection epochs (with contrast-curve limits) into a single
    JIT-friendly container. The ``is_detected`` mask selects between
    a Gaussian photometric likelihood (detections) and the flux-space
    z-score non-detection likelihood (nulls) via ``jnp.where``.

    Attributes:
        time_d: Observation times (days). Shape ``(M,)``.
        sep_grid: Separation grid per epoch (arcsec), padded and monotonic.
            Shape ``(M, K)``.
        dmag0_grid: Limiting dMag per epoch, padded with ``-jnp.inf`` for
            undetectable regions. Shape ``(M, K)``.
        is_detected: Boolean mask -- ``True`` for detection epochs,
            ``False`` for null epochs. Shape ``(M,)``.
        dmag_obs: Measured delta-magnitude at detection epochs. Set to 0.0
            for null epochs (unused). Shape ``(M,)``.
        dmag_err: Measurement uncertainty on dMag at detection epochs.
            Set to 1.0 for null epochs (unused). Shape ``(M,)``.
        valid: Boolean validity mask. Shape ``(M,)``.
        snr_thresh: SNR detection threshold (static). Default 5.0.
    """

    time_d: jnp.ndarray
    sep_grid: jnp.ndarray
    dmag0_grid: jnp.ndarray
    is_detected: jnp.ndarray
    dmag_obs: jnp.ndarray
    dmag_err: jnp.ndarray
    valid: jnp.ndarray
    snr_thresh: float = eqx.field(static=True, default=5.0)

    @classmethod
    def from_detections_and_nulls(
        cls,
        det_time_d,
        det_dmag_obs,
        det_dmag_err,
        det_sep_grid,
        det_dmag0_grid,
        null_time_d,
        null_sep_grid,
        null_dmag0_grid,
        snr_thresh=5.0,
        *,
        max_n,
        max_pts=MAX_CC_PTS,
    ):
        """Build from separate detection and null arrays with padding.

        Concatenates detection and null epochs into a single container
        with the ``is_detected`` mask set accordingly, then pads to
        static shape.

        Args:
            det_time_d: Detection observation epochs (days). Shape ``(N_det,)``.
            det_dmag_obs: Measured dMag at detection epochs. Shape ``(N_det,)``.
            det_dmag_err: dMag uncertainty at detection epochs. Shape ``(N_det,)``.
            det_sep_grid: Contrast-curve separation grid for detection epochs.
                Shape ``(N_det, K)``.
            det_dmag0_grid: Contrast-curve dMag limit for detection epochs.
                Shape ``(N_det, K)``.
            null_time_d: Null observation epochs (days). Shape ``(N_null,)``.
            null_sep_grid: Contrast-curve separation grid for null epochs.
                Shape ``(N_null, K)``.
            null_dmag0_grid: Contrast-curve dMag limit for null epochs.
                Shape ``(N_null, K)``.
            snr_thresh: SNR detection threshold.
            max_n: Padding size for total epochs.
            max_pts: Padding size for contrast curve grid points.

        Returns:
            A ``DMagData`` instance.
        """
        det_time_d = jnp.asarray(det_time_d)
        null_time_d = jnp.asarray(null_time_d)
        n_det = det_time_d.shape[0]
        n_null = null_time_d.shape[0]
        n_total = n_det + n_null

        time_d = jnp.concatenate([det_time_d, null_time_d])
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
            time_d=_pad_1d(time_d, max_n),
            sep_grid=_pad_2d(sep_grid, max_n, max_pts),
            dmag0_grid=_pad_2d(dmag0_grid, max_n, max_pts, fill=-jnp.inf),
            is_detected=_pad_1d(is_detected, max_n).astype(bool),
            dmag_obs=_pad_1d(dmag_obs, max_n),
            dmag_err=_pad_1d(dmag_err, max_n, fill=1.0),
            valid=_valid_mask(n_total, max_n),
            snr_thresh=snr_thresh,
        )

    @classmethod
    def from_contrast_curves(
        cls,
        time_d,
        sep_grids,
        contrast_grids,
        snr_thresh=5.0,
        *,
        max_n,
        max_pts=MAX_CC_PTS,
    ):
        """Null epochs from contrast curves, with every epoch a non-detection.

        Converts each contrast limit to a dMag limit, ``dMag = -2.5 * log10(contrast)``.

        Args:
            time_d: Observation times (days). Shape ``(N_epochs,)``.
            sep_grids: Separation grids per epoch (arcsec).
                Shape ``(N_epochs, N_pts)``.
            contrast_grids: Contrast detection limits per epoch.
                Shape ``(N_epochs, N_pts)``.
            snr_thresh: SNR detection threshold.
            max_n: Padding size for epochs.
            max_pts: Padding size for contrast curve points.

        Returns:
            A ``DMagData`` instance with every epoch a null.
        """
        time_d = jnp.asarray(time_d)
        n = time_d.shape[0]
        return cls.pad(
            time_d=time_d,
            sep_grid=jnp.asarray(sep_grids),
            dmag0_grid=contrast_to_delta_mag(jnp.asarray(contrast_grids)),
            is_detected=jnp.zeros(n, dtype=bool),
            dmag_obs=jnp.zeros(n),
            dmag_err=jnp.ones(n),
            snr_thresh=snr_thresh,
            max_n=max_n,
            max_pts=max_pts,
        )

    @classmethod
    def pad(
        cls,
        *,
        time_d,
        sep_grid,
        dmag0_grid,
        is_detected,
        dmag_obs,
        dmag_err,
        snr_thresh=5.0,
        max_n,
        max_pts=MAX_CC_PTS,
    ):
        """Create a padded DMagData with an ``valid`` mask.

        Args:
            time_d: Observation epochs (days). Shape ``(M,)``.
            sep_grid: Separation grid per epoch (arcsec). Shape ``(M, K)``.
            dmag0_grid: Limiting dMag per epoch. Shape ``(M, K)``.
            is_detected: True for detection epochs, False for null epochs. Shape
                ``(M,)``.
            dmag_obs: Measured dMag at detection epochs. Shape ``(M,)``.
            dmag_err: dMag uncertainty at detection epochs. Shape ``(M,)``.
            snr_thresh: SNR detection threshold.
            max_n: Row capacity of the result.
            max_pts: Contrast-curve grid capacity of the result.

        Returns:
            The data padded to ``max_n`` rows and ``max_pts`` grid points, with
            ``valid``
            true for the first M rows.
        """
        n = jnp.asarray(time_d).shape[0]
        return cls(
            time_d=_pad_1d(time_d, max_n),
            sep_grid=_pad_2d(sep_grid, max_n, max_pts),
            dmag0_grid=_pad_2d(dmag0_grid, max_n, max_pts, fill=-jnp.inf),
            is_detected=_pad_1d(is_detected, max_n).astype(bool),
            dmag_obs=_pad_1d(dmag_obs, max_n),
            dmag_err=_pad_1d(dmag_err, max_n, fill=1.0),
            valid=_valid_mask(n, max_n),
            snr_thresh=snr_thresh,
        )

    @classmethod
    def zeros(cls, max_n, max_pts=MAX_CC_PTS, snr_thresh=5.0):
        """Create an all-invalid placeholder (for model tracing).

        Args:
            max_n: Row capacity of the placeholder.
            max_pts: Contrast-curve grid capacity of the placeholder.
            snr_thresh: SNR detection threshold.

        Returns:
            Data with every row invalid and every limit undetectable.
        """
        return cls(
            time_d=jnp.zeros(max_n),
            sep_grid=jnp.zeros((max_n, max_pts)),
            dmag0_grid=jnp.full((max_n, max_pts), -jnp.inf),
            is_detected=jnp.zeros(max_n, dtype=bool),
            dmag_obs=jnp.zeros(max_n),
            dmag_err=jnp.ones(max_n),
            valid=jnp.zeros(max_n, dtype=bool),
            snr_thresh=snr_thresh,
        )


class PhotometryData(eqx.Module):
    """Linear planet/star contrast measurements and upper limits.

    All fields have shape ``(N,)``. ``time_d`` uses the system's day frame;
    ``contrast`` and ``contrast_err`` are dimensionless linear flux ratios.
    ``band_id`` and ``planet`` index the term's transmissions and system planets.
    ``is_upper_limit`` selects a Gaussian cumulative probability below the
    reported contrast; ``valid`` excludes padded rows.

    Attributes:
        time_d: Observation epochs (days).
        contrast: Measured planet/star contrast (linear flux ratio).
        contrast_err: Contrast standard deviation, or the noise scale of an upper limit.
        band_id: Index into the term's transmissions, one per row.
        planet: Zero-based planet index into the system, one per row.
        is_upper_limit: True where the row is an upper limit on the contrast.
        valid: True for real rows, False for padding.
    """

    time_d: jnp.ndarray
    contrast: jnp.ndarray
    contrast_err: jnp.ndarray
    band_id: jnp.ndarray
    planet: jnp.ndarray
    is_upper_limit: jnp.ndarray
    valid: jnp.ndarray

    @classmethod
    def pad(
        cls,
        *,
        time_d,
        contrast,
        contrast_err,
        band_id,
        planet,
        is_upper_limit,
        max_n,
    ):
        """Pad contrast observations, using unit errors on invalid rows.

        Args:
            time_d: Observation epochs (days). Shape ``(N,)``.
            contrast: Measured linear contrasts. Shape ``(N,)``.
            contrast_err: Contrast uncertainties. Shape ``(N,)``. Padded rows hold 1.0.
            band_id: Band index per row. Shape ``(N,)``.
            planet: Zero-based planet index per row. Shape ``(N,)``.
            is_upper_limit: Upper-limit flag per row. Shape ``(N,)``.
            max_n: Row capacity of the result.

        Returns:
            The data padded to shape ``(max_n,)``, with ``valid`` true for the first N
            rows.
        """
        n = jnp.asarray(time_d).shape[0]
        return cls(
            time_d=_pad_1d(time_d, max_n),
            contrast=_pad_1d(contrast, max_n),
            contrast_err=_pad_1d(contrast_err, max_n, fill=1.0),
            band_id=_pad_1d(band_id, max_n).astype(int),
            planet=_pad_1d(planet, max_n).astype(int),
            is_upper_limit=_pad_1d(is_upper_limit, max_n).astype(bool),
            valid=_valid_mask(n, max_n),
        )

    @classmethod
    def from_dmag(
        cls,
        *,
        time_d,
        dmag,
        dmag_err,
        band_id,
        planet=None,
        is_upper_limit=None,
        max_n=None,
    ):
        """Convert magnitude differences with first-order Gaussian errors.

        ``contrast = 10**(-0.4*dmag)`` and its standard deviation is
        ``0.4*ln(10)*contrast*dmag_err``. Upper-limit rows use that same
        conversion for the limit and its noise scale. Planet indices default
        to zero, limit flags to False, and ``max_n=None`` keeps all rows valid.

        Args:
            time_d: Observation epochs (days), one per row.
            dmag: Magnitude difference between planet and star, one per row.
            dmag_err: Magnitude difference uncertainty, one per row.
            band_id: Band index per row.
            planet: Zero-based planet index per row. Defaults to zero.
            is_upper_limit: Upper-limit flag per row. Defaults to False.
            max_n: Row capacity to pad to. None keeps the rows as given, all valid.

        Returns:
            Linear-contrast data converted from the magnitude differences.
        """
        time_d = jnp.asarray(time_d)
        contrast = delta_mag_to_contrast(jnp.asarray(dmag))
        if planet is None:
            planet = jnp.zeros(time_d.shape, dtype=int)
        if is_upper_limit is None:
            is_upper_limit = jnp.zeros(time_d.shape, dtype=bool)
        values = dict(
            time_d=time_d,
            contrast=contrast,
            contrast_err=0.4 * jnp.log(10.0) * contrast * jnp.asarray(dmag_err),
            band_id=jnp.asarray(band_id, dtype=int),
            planet=jnp.asarray(planet, dtype=int),
            is_upper_limit=jnp.asarray(is_upper_limit, dtype=bool),
        )
        if max_n is not None:
            return cls.pad(**values, max_n=max_n)
        return cls(**values, valid=jnp.ones(time_d.shape, dtype=bool))


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


def lambertian_from_area(albedo_area_AU2):
    """Map Lambda = Ag Rp^2 to a unit-albedo model and radius in Earth radii.

    Reflected light identifies only the area product. Fix Ag=1 and put
    the sampled scale into radius; both returned arrays have shape (K,).

    Args:
        albedo_area_AU2: Geometric albedo times radius squared (AU^2), one value per
            planet.

    Returns:
        A tuple of a unit-albedo Lambertian physical model and the planet radius
        (Earth radii) that carries the area, each with shape ``(K,)``.
    """
    # Delay scene imports, keeping their module constants concrete under first JIT.
    with jax.ensure_compile_time_eval():
        from skyscapes.physical_model import LambertianPhysicalModel

    area = jnp.atleast_1d(albedo_area_AU2)
    return LambertianPhysicalModel(Ag=jnp.ones_like(area)), jnp.sqrt(area) / Rearth2AU


def lambertian_dmag(phase_rad, distance_AU, model, radius_Rearth, flux_scale=1.0):
    """Convert a Lambertian reflector's contrast to dMag at the orbit-fit floor.

    ``flux_scale`` is the planet's dimensionless factor on its contrast.

    Args:
        phase_rad: Phase angle (radians). Shape ``(T,)``.
        distance_AU: Planet-star distance (AU). Shape ``(T,)``.
        model: Lambertian physical model of the planet.
        radius_Rearth: Planet radius (Earth radii).
        flux_scale: Dimensionless factor on the planet's contrast.

    Returns:
        The magnitude difference between planet and star, shape ``(T,)``.
    """
    contrast = (
        flux_scale
        * model.contrast(phase_rad[None, :], distance_AU[None, :], None, radius_Rearth)[
            0
        ]
    )
    return contrast_to_delta_mag(jnp.maximum(contrast, 1e-300))


def system_separation_dmag(system, time_d, *, albedo_area_AU2=None):
    """Predict separation (arcsec) and dMag at days from a one-planet epoch.

    A scene supplies its Lambertian reflector and radius. An explicit
    geometric-albedo times radius squared (AU^2) supplies the reflector
    for a bare dynamical system or a known photometric scale. A scene's planet
    ``flux_scale`` multiplies the contrast in either case.

    Args:
        system: A scene with one planet, or a bare orbix ``System`` with one planet.
        time_d: Epochs in days from the system's epoch.
        albedo_area_AU2: Geometric albedo times radius squared (AU^2) of the planet.
            Required for a bare system; when given for a scene it replaces the scene's
            reflector.

    Returns:
        A tuple of the star-planet separation (arcsec) and the magnitude difference at
        each epoch, each with the shape of ``time_d``.
    """
    bare = isinstance(system, System)
    count = system.orbits.a_AU.size if bare else system.n_planets
    if count != 1:
        raise ValueError(
            "Multi-planet null and imaging require a detection model per planet"
        )
    if albedo_area_AU2 is None:
        if bare:
            raise ValueError(
                "A dynamical system needs albedo_area_AU2 for Lambertian photometry"
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
        if jnp.size(albedo_area_AU2) != 1:
            raise ValueError(
                "albedo_area_AU2 must specify one planet's photometric area"
            )
        model, radius_Rearth = lambertian_from_area(jnp.reshape(albedo_area_AU2, (1,)))
    flux_scale = 1.0
    if not bare:
        radius_Rearth = jnp.where(system.valid, radius_Rearth, 0.0)
        flux_scale = jnp.reshape(system.planets[0].flux_scale, ())
    ephemeris = system.solve(time_d) if bare else system.ephemeris(time_d)
    return ephemeris.separation_arcsec(1, 0), lambertian_dmag(
        ephemeris.phase_angle_rad(1, 0),
        ephemeris.distance_AU(1, 0),
        model,
        radius_Rearth,
        flux_scale,
    )


def _fixed_noise_null(alpha_pred, dMag_pred, data):
    """Score censored photometry in units of planet-free noise."""
    dmag0 = interpolate_contrast_curve(alpha_pred, data.sep_grid, data.dmag0_grid)
    # Clip the exponent when the sampler proposes an implausibly bright orbit.
    exp_arg = jnp.clip(-0.4 * (dMag_pred - dmag0), max=200.0)
    law = ReportRule(threshold=data.snr_thresh)
    counts = law.detection_limit(1.0) * 10.0**exp_arg
    zero = jnp.zeros_like(counts)
    prediction = ReportPrediction(counts, jnp.ones_like(counts), zero, zero)
    return law.log_null_probability(1.0, prediction)


def loglike_dmag(alpha_pred, dMag_pred, data):
    """Log-likelihood of dMag detections and contrast-curve nulls.

    At **detection epochs** (``data.is_detected == True``): evaluates a
    Gaussian photometric likelihood comparing the predicted dMag to the
    measured ``dMag_obs +/- dMag_err``. This provides an unbounded parabolic
    penalty that prevents the optimizer from crushing Lambda to zero.

    At **null epochs** (``data.is_detected == False``): uses the existing
    flux-space z-score of the predicted counts against the reporting rule's
    detection threshold, with fixed noise in flux space. A planet brighter than the
    limit (``dMag_pred < dMag0``) has a flux ratio above one and a heavy penalty;
    outside the contrast curve the finite "undetectable" sentinel of
    :func:`interpolate_contrast_curve` gives a null probability near one.

    The two branches are combined with :func:`jnp.where` for branchless
    JIT compatibility.

    Args:
        alpha_pred: Predicted angular separation (arcsec). Shape
            ``(M,)``.
        dMag_pred: Predicted delta-magnitude. Shape ``(M,)``.
        data: A :class:`~photomancy.measurements.photometry.DMagData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    mask = data.valid  # boolean

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
    bands and ``transmissions`` follow ``data.band_id`` order. The system at
    ``system_path`` (the empty path is the model itself) returns ``(W, K, T)`` band
    contrasts. Each row selects its own band, planet and epoch. An upper limit
    contributes ``log Phi((limit - prediction) / contrast_err)``. Invalid rows return
    zero. Data ``time_d`` is in days from the system's epoch. Bands share one ephemeris.

    Attributes:
        data: Linear contrast measurements and upper limits.
        quadrature: Host-built spectral quadrature, one band per transmission.
        transmissions: Filter response of each band, in ``data.band_id`` order.
        system_path: Model path of the system returning ``(W, K, T)`` band contrasts.
        resolution: Host-built reduction of the system's tabulated spectra onto
            ``quadrature``. The contrast integrates the star, planet and filter
            tables over each band; empty, it serves a system with no table, whose
            sources are sampled at the nodes.
        reads: Model paths a held build cached. Empty: the resolution depends on
            the system's static knot grids only, which no fit can free, and a
            replaced grid has no operator in it and is refused.
    """

    data: PhotometryData
    quadrature: Quadrature
    transmissions: tuple[Transmission, ...]
    system_path: str = eqx.field(static=True, default="")
    resolution: Resolution = eqx.field(default_factory=Resolution)
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require a transmission per band."""
        if len(self.transmissions) != len(self.quadrature.bands):
            raise ValueError(
                "PhotometryTerm needs one transmission per quadrature band"
            )

    def check(self, model) -> None:
        """Require every valid row's planet and band to index the system and quadrature.

        Args:
            model: The model holding the system at ``system_path``, with concrete
                values.

        Raises:
            ValueError: For a valid row whose planet is negative or at least the
                number of planets, or whose ``band_id`` is negative or at least the
                number of bands.
        """
        system = resolve(model, self.system_path)
        _check_planets(
            system,
            self.data.planet,
            self.data.valid,
            "PhotometryTerm",
            lambda: _band_table(
                system,
                self.quadrature,
                self.transmissions,
                jnp.zeros(1),
                self.resolution,
            ).shape[1],
        )
        _check_indices(
            self.data.band_id,
            self.data.valid,
            len(self.quadrature.bands),
            "PhotometryTerm",
            "band",
        )

    def _score(self, model):
        valid = self.data.valid
        data = PhotometryData(
            time_d=jnp.where(valid, self.data.time_d, 0.0),
            contrast=jnp.where(valid, self.data.contrast, 0.0),
            contrast_err=jnp.where(valid, self.data.contrast_err, 1.0),
            band_id=jnp.where(valid, self.data.band_id, 0),
            planet=jnp.where(valid, self.data.planet, 0),
            is_upper_limit=jnp.where(valid, self.data.is_upper_limit, False),
            valid=valid,
        )
        system = resolve(model, self.system_path)
        prediction = _band_contrasts(
            system,
            self.quadrature,
            self.transmissions,
            data.time_d,
            data.band_id,
            data.planet,
            self.resolution,
        )
        prediction = jnp.where(valid, prediction, 0.0)
        z = (data.contrast - prediction) / data.contrast_err
        detection = -0.5 * z**2 - jnp.log(data.contrast_err * jnp.sqrt(2.0 * jnp.pi))
        rows = jnp.where(data.is_upper_limit, norm.logcdf(z), detection)
        return jnp.sum(jnp.where(valid, rows, 0.0)), prediction

    def __call__(self, model):
        """Return the scalar log-likelihood.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            Scalar log-likelihood of the valid rows.
        """
        return self._score(model)[0]

    def predict(self, model):
        """Return each row's linear contrast, ``(T,)``.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            Predicted linear contrast (planet/star flux ratio) of each row's band,
            planet
            and epoch, zero on invalid rows. Shape ``(T,)``.
        """
        return self._score(model)[1]

    def draw(self, key, model) -> Self:
        """Draw Gaussian contrasts, censoring at each stored upper limit.

        An upper-limit row retains its threshold and flag when the draw lies
        below the threshold. Above it, the row becomes a measured detection.
        Rows already measured as detections remain uncensored Gaussian draws.

        Args:
            key: JAX random key.
            model: The model holding the system at ``system_path``.

        Returns:
            The term with contrasts drawn about the prediction, censored at stored upper
            limits.
        """
        data = self.data
        valid = data.valid
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
    to override the scene, the photometric area ``Ag * Rp**2`` (AU^2) is a model leaf at
    ``area_path`` (a held leaf for a fixed area, a free one to fit it).

    Attributes:
        data: Detection and null epochs with their contrast curves.
        system_path: Model path of the one-planet system; the empty path is the model.
        area_path: Model path of the photometric area, or None.
        reads: Model paths a held build cached. Empty: the term reads the model
            directly.
    """

    data: DMagData
    system_path: str = eqx.field(static=True, default="")
    area_path: str | None = eqx.field(static=True, default=None)
    reads: ClassVar[tuple[str, ...]] = ()

    def _prediction(self, model):
        valid = self.data.valid
        prediction = system_separation_dmag(
            resolve(model, self.system_path),
            jnp.where(valid, self.data.time_d, 0.0),
            albedo_area_AU2=None
            if self.area_path is None
            else resolve(model, self.area_path),
        )
        return tuple(jnp.where(valid, x, 0.0) for x in prediction)

    def __call__(self, model):
        """Return the combined dMag and null log likelihood.

        Args:
            model: The model holding the system at ``system_path`` and any
                ``area_path``.

        Returns:
            Scalar log-likelihood of the detection and null rows.
        """
        return loglike_dmag(*self._prediction(model), self.data)

    def predict(self, model):
        """Return (separation in arcsec, dMag) at each epoch, zero on padded rows.

        Args:
            model: The model holding the system at ``system_path`` and any
                ``area_path``.

        Returns:
            A tuple of separation (arcsec) and dMag, each with shape ``(M,)``.
        """
        return self._prediction(model)

    def draw(self, key, model) -> Self:
        """Copy with Gaussian dMag draws at the epochs of the data.

        A null epoch has no dMag to draw: the outcome of a planned observation is a
        detection or a non-detection, and the predictive for that outcome carries the
        probability of detection, which this term does not. A term with a valid null
        row refuses (:class:`~photomancy.measurements.ReportTerm` is the predictive
        for a detection outcome), so a planned non-detection never reports an
        information gain of exactly zero.

        Args:
            key: JAX random key.
            model: The model holding the system at ``system_path`` and any
                ``area_path``.

        Returns:
            The term with measured dMag drawn about the prediction at detection epochs.
        """
        data = self.data
        data = eqx.error_if(
            data,
            jnp.any(data.valid & ~data.is_detected),
            "DMagTerm.draw: a null epoch has no dMag to draw; use ReportTerm for the "
            "predictive of a detection outcome",
        )
        mask = data.valid & data.is_detected
        sigma = jnp.where(mask, data.dmag_err, 1.0)
        dmag = self.predict(model)[1]
        draw = dmag + sigma * jax.random.normal(key, dmag.shape, dtype=jnp.float64)
        sampled = eqx.tree_at(
            lambda d: d.dmag_obs, data, jnp.where(mask, draw, data.dmag_obs)
        )
        return eqx.tree_at(lambda t: t.data, self, sampled)
