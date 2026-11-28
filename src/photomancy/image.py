"""The pixel likelihood: one frame of electrons scored against the expected rates.

:class:`ImageTerm` is one frame of observed electrons. At a model it renders the
expected electron rates with :func:`coronagraphoto.expected`, hands them to the
detector's own ``log_prob`` (a Poisson mass for a photon-counting detector, the
moment-matched Gaussian otherwise; elementwise, normalized, constants included), masks
the pixels the term does not use and sums in float64. The same function drives
:func:`photomancy.detection.detect` (a matched-filter map) and
:func:`photomancy.detection.measure` (a measurement with its covariance), and it is a
fit term in ``Fit.terms``.

**One owner per fit.** The instrument, the exposure and the scene each have one owner,
either a field of the term or a node of the model. A fit that frees instrument leaves
(an optical path difference, a detector flat) keeps the instrument in the model, in an
:class:`ImageModel` or any record with the paths the term names, and the term holds
``instrument_path``; a fit that holds the instrument gives it to the term as a field and
never has two copies. The scene is always the model or a node of it: the default
``scene_path=""`` names the model itself, so ``Fit(scene, ...)`` and the orbit
posterior's model work unchanged. The exposure is the description of one observation,
so it is held by the term (a new exposure is a new term and does not recompile) or read
from the model by a path (when the roll or the time is free); exactly one of the two is
given.

**What the term reads.** :func:`coronagraphoto.expected` is a pure function and builds
nothing, and the held exposure is an :class:`optixstuff.Exposure`, which caches no
scene values, so the term declares no ``reads``.

**Scored pixels and source coverage.** :meth:`ImageTerm.on_support` stores the
detector pixels covered at a reference model. An explicit mask can select a subset.
:meth:`ImageTerm.check` raises if a scored pixel lacks coverage or a lit planet lies
outside the selected component's source domain at a band centre. ``Fit`` checks at
setup and at both prior endpoints of any free leaf that moves the field of view
(the wavelength nodes, the primary, the detector). If a proposal loses coverage
of a scored pixel, the image log likelihood is NaN.

**Static light beyond a speckle field.** A speckle field ends where its data end and
the drift beyond it is unknown, so by default the render zeroes those pixels and they
are not ``supported``, although the static products cover them.
``ImageTerm(..., static_beyond_speckle=True)`` is the analyst's assumption that there
is no drift beyond the field: the term renders
``coronagraphoto.expected(..., static_beyond_speckle=True)``, whose model there is the
static scene (the static leakage plus the field's stored samples, zero-filled, under
the field's own reconstruction), and it scores the pixels ``Rates.static_supported``
marks. The option is a statement about one field and the static map beside it: inside
a dark hole it understates the leakage by a factor of order one, and nothing here
checks that the static map beyond the field is bright enough. A fit, a
:class:`photomancy.detection.LikelihoodEstimate` and a
:class:`photomancy.detection.DetectionMap` name it in their ``assumptions``.

Count regime. The Poisson form needs integer counts, and a detector that is not
``photon_counting`` scores the moment-matched Gaussian, which is not exact at a few
electrons per pixel with read noise.
"""

from dataclasses import replace
from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from coronagraphoto import expected, planet_source_support
from hwoutils.leaves import is_traced
from hwoutils.model_paths import join, resolve
from hwoutils.precision import require_float64
from optixstuff import (
    ELECTRON_COUNT_QUANTITIES,
    AbstractDetector,
    Exposure,
    ExposureData,
    Instrument,
)
from optixstuff.detector import _floored
from optixstuff.grids import Image, Measure, SkyOrientation
from optixstuff.rates import floor_rate
from skyscapes.scene import Scene

from photomancy.infer._precision import _as_wide
from photomancy.infer.fit import AbstractTerm

__all__ = ["ImageForward", "ImageModel", "ImageTerm"]


def _masked_rates(rates, mask, *, preserve_floor=False, only_nonfinite=False):
    """Give excluded pixels zero source rates before evaluating detector moments.

    The detector laws handle zero mean and variance with safe operands. For the
    likelihood, preclip tolerated negative totals using the original frame peak
    (with nonfinite excluded components set to zero). This keeps the detector's
    floor decision for scored pixels after zeroing the excluded ones. Draws only
    replace nonfinite excluded inputs so finite rates keep their sampling behavior.
    """
    pixel_mask = mask[:, None, :, :] if rates.spectral else mask
    total = rates.total()
    if total.ndim != pixel_mask.ndim:
        return rates

    names = rates.component_names()
    values = tuple(getattr(rates, name) for name in names)
    keep = pixel_mask
    if preserve_floor:
        magnitude = sum(
            jnp.abs(jnp.where(pixel_mask | jnp.isfinite(value), value, 0))
            for value in values
        )
        peak = jnp.max(magnitude, axis=tuple(range(1, total.ndim)), keepdims=True)
        _, bad = floor_rate(total, peak)
        keep = pixel_mask & ~((total < 0) & ~bad)
    if only_nonfinite:
        keep = pixel_mask | jnp.isfinite(total)
    values = tuple(jnp.where(keep, value, 0) for value in values)

    variance = rates.variance
    if variance is not None:
        variance = _masked_rates(
            variance, mask, preserve_floor=preserve_floor, only_nonfinite=only_nonfinite
        )
    n_pix = rates.n_pix
    if n_pix is not None and n_pix.ndim == pixel_mask.ndim:
        keep_pixels = pixel_mask | jnp.isfinite(n_pix) if only_nonfinite else pixel_mask
        n_pix = jnp.where(keep_pixels, n_pix, 1)
    return eqx.tree_at(
        lambda r: (*[getattr(r, name) for name in names], r.variance, r.n_pix),
        rates,
        (*values, variance, n_pix),
        is_leaf=lambda value: value is None,
    )


def _same_angle(first, second):
    """Equal modulo 360 degrees, to the float32 resolution of the larger angle."""
    difference = abs((first - second + 180.0) % 360.0 - 180.0)
    return difference <= np.spacing(np.float32(max(abs(first), abs(second))))


def _check_image(image, instrument, exposure):
    """Refuse an ``Image`` that is not detector counts on this detector at this roll.

    The image must hold electrons per pixel (an integral measure). Its pixel
    geometry (shape, pitch, center, unit) must equal the detector's grid when the
    instrument is held. When the exposure is held with a concrete roll, the image
    must carry a sky orientation, and that orientation must be the roll's.
    """
    if (
        image.measure != Measure.integral()
        or image.quantity not in ELECTRON_COUNT_QUANTITIES
    ):
        raise ValueError(
            f"the observed Image holds {image.quantity!r} as {image.measure}, not "
            "electrons per pixel: score the detector counts observe returns"
        )
    if instrument is not None:
        detector = instrument.detector.grid
        if replace(image.grid, orientation=None) != detector:
            raise ValueError(
                f"the observed Image is on {image.grid}, not on the detector grid "
                f"{detector}: score data on the instrument's own pixels"
            )
    roll = None if exposure is None else exposure.pointing.roll_deg
    if roll is None or isinstance(roll, jax.core.Tracer):
        return
    have = image.grid.orientation
    if have is None:
        raise ValueError(
            "the observed Image carries no sky orientation (observe under a JAX "
            "transformation drops it), so its roll cannot be checked against the "
            "exposure's; pass Image.data when the roll is known to match"
        )
    want = SkyOrientation(roll)
    if have.parity != want.parity or not _same_angle(have.pa_y_deg, want.pa_y_deg):
        raise ValueError(
            f"the observed Image has sky orientation {have}, but the exposure's "
            f"roll gives {want}: the data and the term are at different rolls"
        )


def _python_bool(value, name):
    """``value`` as a Python bool: a bool, a NumPy bool or a concrete 0-d JAX bool.

    Raises:
        TypeError: For anything else (an int, a string, an array of booleans, a
            traced value).
    """
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if (
        isinstance(value, jax.Array)
        and not isinstance(value, jax.core.Tracer)
        and value.shape == ()
        and value.dtype == jnp.bool_
    ):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def _scored_flag(rates, static_beyond_speckle):
    """The ``(W, ny, nx)`` support a term scores, or ``None`` when the rates carry none.

    ``Rates.supported`` by default; ``Rates.static_supported`` under the option, read
    as ``supported`` when the record does not distinguish the two (``None``).
    """
    if static_beyond_speckle and rates.static_supported is not None:
        return rates.static_supported
    return rates.supported


@eqx.filter_jit
def _flags(scene, instrument, exposure):
    """The ``(W, ny, nx)`` support flags ``(supported, static_supported)``.

    The flags of the default render, which the option does not change. Each is all
    True when the rates carry no flag, and ``static_supported`` is ``supported`` when
    the record does not distinguish the two.
    """
    rates = expected(scene, instrument, exposure)
    strict = rates.supported
    if strict is None:
        strict = jnp.ones(rates.planet.shape, bool)
    static = rates.static_supported
    return strict, strict if static is None else static


def _lit_planets_outside(scene, instrument, exposure):
    """Number of lit planets outside the coronagraph's source domain at band centre."""
    if not scene.planets:
        return 0
    times = jnp.atleast_1d(exposure.time_d)
    ephemeris = scene.ephemeris(times)
    support = planet_source_support(scene, instrument, exposure)
    outside = np.zeros(support.shape[1], dtype=bool)
    for band, covered in zip(exposure.quadrature.bands, support, strict=True):
        flux = scene.planet_flux_densities(band.center_nm, times, ephemeris=ephemeris)
        outside |= np.any((np.asarray(flux) > 0) & ~covered, axis=1)
    return int(outside.sum())


def _held_or_resolved(model, held, path):
    return held if path is None else resolve(model, path)


def _render(
    model,
    *,
    scene_path,
    instrument,
    instrument_path,
    exposure,
    exposure_path,
    static_beyond_speckle,
):
    """The one place that turns a model into ``(detector, rates, plan)``.

    The scene is ``resolve(model, scene_path)`` (``""`` is the model itself); each of
    the instrument and the exposure is the held object or, with its path, the node of
    ``model`` at that path.
    """
    instrument = _held_or_resolved(model, instrument, instrument_path)
    exposure = _held_or_resolved(model, exposure, exposure_path)
    rates = expected(
        resolve(model, scene_path),
        instrument,
        exposure,
        static_beyond_speckle=static_beyond_speckle,
    )
    return instrument.detector, rates, exposure.plan


def _footprint_paths(instrument_path, exposure_path):
    """Model paths that move the field of view (band nodes, primary, detector)."""
    paths = []
    if exposure_path is not None:
        paths.append(join(exposure_path, "quadrature"))
    if instrument_path is not None:
        paths += [join(instrument_path, "primary"), join(instrument_path, "detector")]
    return tuple(paths)


def _assumptions(static_beyond_speckle):
    return ("static_beyond_speckle",) if static_beyond_speckle else ()


def _check_planets(scene, instrument, exposure):
    planets = _lit_planets_outside(scene, instrument, exposure)
    if planets:
        raise ValueError(
            f"{planets} lit planet(s) lie outside the coronagraph's source domain "
            "at a band centre. Start the fit inside the domain or hold them"
        )


def _check_sources(
    model, *, scene_path, instrument, instrument_path, exposure, exposure_path
):
    """Refuse a lit planet outside the source domain."""
    scene = resolve(model, scene_path)
    instrument = _held_or_resolved(model, instrument, instrument_path)
    exposure = _held_or_resolved(model, exposure, exposure_path)
    _check_planets(scene, instrument, exposure)


def _check_exclusive(instrument, instrument_path, exposure, exposure_path):
    if (instrument is None) == (instrument_path is None):
        raise ValueError("give exactly one of instrument and instrument_path")
    if (exposure is None) == (exposure_path is None):
        raise ValueError("give exactly one of exposure and exposure_path")
    if exposure is not None and exposure.plan is None:
        raise ValueError("the exposure needs a FramePlan: build it with plan=")


def _data_grid(grid):
    return replace(grid, orientation=None)


def _check_data_image(image, instrument, *, what):
    """Refuse data that is not integral electrons (or rates) on the detector grid."""
    if image.measure != Measure.integral():
        raise ValueError(
            f"{what}: the data holds {image.quantity!r} as {image.measure}, not an "
            "integral per pixel"
        )
    if "event" in image.leading_axes:
        raise ValueError(
            f"{what}: the data has an 'event' axis; build one term per event with "
            "ExposureData.take(index)"
        )
    if instrument is not None and _data_grid(image.grid) != _data_grid(
        instrument.detector.grid
    ):
        raise ValueError(
            f"{what}: the data is on {image.grid}, not on the detector grid "
            f"{instrument.detector.grid}: score data on the instrument's own pixels"
        )


def _from_exposure_data(data, instrument, exposure, exposure_path):
    """Electrons, the usable-pixel mask and the exposure a term holds, from data."""
    image = data.image
    if image.quantity not in ELECTRON_COUNT_QUANTITIES:
        raise ValueError(
            f"ImageTerm needs electrons, but the data holds {image.quantity!r}: "
            "score a rate image through ForwardTerm(ImageForward.from_exposure_data"
            "(data), MaskedLogLikelihood.from_exposure_data(data))"
        )
    _check_data_image(image, instrument, what="ImageTerm")
    if data.variance is not None:
        raise ValueError(
            "the data carries its own variance, but ImageTerm scores with the "
            "detector law as the noise model: use ForwardTerm(ImageForward"
            ".from_exposure_data(data), MaskedLogLikelihood.from_exposure_data(data))"
        )
    if exposure is not None:
        raise ValueError(
            "ExposureData supplies the exposure: drop exposure=, or name the "
            "model's exposure with exposure_path"
        )
    valid = data.valid
    if "draw" in image.leading_axes:
        if is_traced(valid):
            raise ValueError("valid must be concrete to compare it across draws")
        if not bool(jnp.all(valid == valid[:1])):
            raise ValueError("the draws have unequal valid masks: score them apart")
        valid = valid[0]
    return image.data, valid, data.exposure if exposure_path is None else None


def _check_score_shape(mask, observed, shape):
    if mask.shape != shape or observed.shape[-3:] != shape:
        raise ValueError(
            f"mask {mask.shape} and observed frames {observed.shape} "
            f"must match the render's {shape}"
        )


class _ImageScore(eqx.Module):
    """The log likelihood of one frame as a function of the expected rates.

    Holds what ``ImageTerm.__call__`` reads besides the rates: the frame, the mask,
    the detector and the frame plan, all arrays of the term and the model that do not
    move with a source amplitude. Calling it on the rates of a render is the term's
    value.

    Attributes:
        observed_e: The observed frame.
        mask: The fixed boolean mask shared by all frames.
        detector: The detector whose ``log_prob`` scores the frame.
        plan: The frame plan.
        static_beyond_speckle: Whether the beyond-speckle set is static.
    """

    observed_e: jax.Array
    mask: jax.Array
    detector: eqx.Module
    plan: eqx.Module
    static_beyond_speckle: bool = eqx.field(static=True)

    def __call__(self, rates) -> jax.Array:
        require_float64("image log likelihood")
        total = rates.total()
        _check_score_shape(self.mask, self.observed_e, total.shape)
        observed = jnp.where(self.mask, _as_wide(self.observed_e), 0.0)
        masked = _masked_rates(
            rates,
            self.mask,
            preserve_floor=type(self.detector).log_prob is AbstractDetector.log_prob,
        )
        log_prob = self.detector.log_prob(observed, masked, self.plan)
        value = jnp.sum(_as_wide(jnp.where(self.mask, log_prob, 0.0)))
        nan_rate = jnp.isnan(total)
        if rates.variance is not None:
            nan_rate = nan_rate | jnp.isnan(rates.variance.total())
        value = jnp.where(jnp.any(self.mask & nan_rate), jnp.nan, value)
        support = _scored_flag(rates, self.static_beyond_speckle)
        if support is None:
            return value
        return jnp.where(jnp.any(self.mask & ~support), jnp.nan, value)


class ImageModel(eqx.Module):
    """The model of a fit that frees instrument leaves as well as scene leaves.

    Paths are ``"scene.<path>"``, ``"instrument.<leaf>"`` and ``"exposures[i].<leaf>"``;
    pair it with ``ImageTerm(scene_path="scene", instrument_path="instrument")``.

    Attributes:
        scene: The sources.
        instrument: The instrument, with the seams a fit frees.
        exposures: Exposures that a term reads by path
            (``exposure_path="exposures[0]"``), for a fit that frees a roll or a time.
    """

    scene: Scene
    instrument: Instrument
    exposures: tuple[Exposure, ...] = ()


class ImageTerm(AbstractTerm):
    """Log likelihood of one frame of observed electrons under the expected rates.

    The observed frame may be given as the :class:`optixstuff.grids.Image` that
    :func:`coronagraphoto.observe` returns or as a bare array. An ``Image`` is checked
    at construction: it must hold electrons per pixel; with a held instrument its
    pixel grid must be the detector's grid; with a held exposure of concrete roll it
    must carry a sky orientation equal to the roll's (modulo 360 degrees, to the
    float32 resolution of the angle). Only the data are kept. When the instrument is
    read from the model (``instrument_path``) the detector is not known at
    construction and the grid is not checked; when the exposure is read from the
    model the roll is not checked.

    The observed frame may also be an :class:`optixstuff.ExposureData` whose image
    holds integral electrons, which supplies the exposure, so ``exposure`` is not
    passed. Its ``valid`` array is combined by logical and with ``mask`` (``mask`` is
    then optional). A variance is refused, because this term scores with the detector
    law as its noise model: score a rate image through ``ForwardTerm`` with
    :meth:`ImageForward.from_exposure_data` instead. A leading ``draw`` axis is
    scored jointly as further frames of the exposure, and the draws must share one
    ``valid``; an event axis is refused.

    Attributes:
        forward_fields: Forward differentiation uses ``instrument``, ``exposure``.
        reads: Model paths a held build cached. Empty: the term reads the model
            by path and holds no build.
        observed_e: Observed electrons, shape ``(*batch, W, ny, nx)``. Leading batch
            axes are further frames of the same exposure model (scored jointly).
        mask: Boolean array of shape ``(W, ny, nx)``, true at the pixels scored
            in every frame. Use :meth:`on_support` to select the pixels covered by
            a reference model. A proposal that loses coverage of any scored pixel
            gives a NaN log likelihood.
        instrument: A held :class:`optixstuff.Instrument`, or ``None``.
        exposure: A held :class:`optixstuff.Exposure` with a ``plan``, or ``None``.
        scene_path: Model path of the :class:`skyscapes.Scene`; ``""`` is the model
            itself.
        instrument_path: Model path of the instrument, or ``None`` when it is held.
        exposure_path: Model path of the exposure, or ``None`` when it is held.
            Exactly one of ``instrument`` and ``instrument_path``, and exactly one of
            ``exposure`` and ``exposure_path``, is given.
        static_beyond_speckle: The assumption that the speckle field does not drift
            beyond its data. False (the default) treats a pixel the speckle field
            does not cover as unknown: the model there is zero, ``check`` refuses a
            mask that scores it, and :meth:`on_support` leaves it out. True passes the
            option to :func:`coronagraphoto.expected`, so the model, ``predict``,
            ``residual``, ``fisher_root`` and ``draw`` there are the static scene: the
            static leakage plus the field's stored samples, zero-filled, under the
            field's own reconstruction, with no drift. ``supported_mask`` is then
            ``Rates.static_supported``, :meth:`on_support` stores it, and ``check``
            refuses only pixels no static product covers. A frame drawn under the
            option has no drift to recover beyond the field. The option changes no
            value for a mask inside the field, but it exists to admit a wider mask,
            and a wider mask scores a different data set: a log likelihood, an
            information matrix or an evidence obtained with it is not comparable with
            one from the default mask, and thresholds calibrated on one do not
            transfer to the other. The term reports the option in
            :attr:`assumptions`.
    """

    forward_fields = ("instrument", "exposure")
    reads: ClassVar[tuple[str, ...]] = ()
    observed_e: jax.Array
    mask: jax.Array
    instrument: Instrument | None = None
    exposure: Exposure | None = None
    scene_path: str = eqx.field(static=True, default="")
    instrument_path: str | None = eqx.field(static=True, default=None)
    exposure_path: str | None = eqx.field(static=True, default=None)
    static_beyond_speckle: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        observed_e,
        mask=None,
        instrument: Instrument | None = None,
        exposure: Exposure | None = None,
        scene_path: str = "",
        instrument_path: str | None = None,
        exposure_path: str | None = None,
        *,
        static_beyond_speckle: bool = False,
    ):
        """Store the frame, checking an ``Image`` against the held instrument.

        Args:
            observed_e: Observed electrons, an :class:`optixstuff.grids.Image`, an
                array of shape ``(*batch, W, ny, nx)`` or an
                :class:`optixstuff.ExposureData` (see the class docstring).
            mask: Boolean array of shape ``(W, ny, nx)`` shared by all frames. It is
                required for an array or an ``Image``; with ``ExposureData`` it is
                optional and combined by logical and with the data's ``valid``.
            instrument: The held instrument, or ``None`` with ``instrument_path``.
            exposure: The held exposure, or ``None`` with ``exposure_path``. Not
                accepted with ``ExposureData``, which supplies it.
            scene_path: Model path of the scene.
            instrument_path: Model path of the instrument, or ``None``.
            exposure_path: Model path of the exposure, or ``None``. With
                ``ExposureData`` it takes precedence over the data's exposure.
            static_beyond_speckle: Keep the static light where only the speckle
                field lacks data (see the class docstring). A Python, NumPy or
                concrete 0-d JAX boolean, stored as a Python bool.

        Raises:
            TypeError: When the mask or ``static_beyond_speckle`` is not boolean.
            ValueError: When the mask is None for an array or an ``Image``, the
                observed image has the wrong quantity, detector grid or sky
                orientation, or ``ExposureData`` has a variance, an event axis,
                draws with unequal ``valid``, or comes with an explicit ``exposure``.
        """
        data_valid = None
        if isinstance(observed_e, ExposureData):
            observed_e, data_valid, exposure = _from_exposure_data(
                observed_e, instrument, exposure, exposure_path
            )
        elif isinstance(observed_e, Image):
            _check_image(observed_e, instrument, exposure)
            observed_e = observed_e.data
        self.observed_e = jnp.asarray(observed_e)
        if mask is None and data_valid is None:
            raise ValueError(
                "give a boolean mask or use ImageTerm.on_support(model, ...)"
            )
        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.dtype != jnp.bool_:
                raise TypeError(f"mask must be boolean, got {mask.dtype}")
        if data_valid is not None:
            mask = data_valid if mask is None else mask & data_valid
        self.mask = mask
        self.instrument = instrument
        self.exposure = exposure
        self.scene_path = scene_path
        self.instrument_path = instrument_path
        self.exposure_path = exposure_path
        self.static_beyond_speckle = _python_bool(
            static_beyond_speckle, "static_beyond_speckle"
        )

    def __check_init__(self):
        """Require one source each for instrument and exposure, and a frame array."""
        if type(self.static_beyond_speckle) is not bool:
            raise TypeError(
                "static_beyond_speckle must be a Python bool, got "
                f"{type(self.static_beyond_speckle).__name__}"
            )
        _check_exclusive(
            self.instrument, self.instrument_path, self.exposure, self.exposure_path
        )
        if self.observed_e.ndim < 3:
            raise ValueError("observed_e must have shape (*batch, W, ny, nx)")

    @classmethod
    def on_support(cls, model, observed_e, **kwargs) -> Self:
        """Build a term on the pixels covered by a concrete reference model.

        Args:
            model: Reference model from which to read the support once at setup.
            observed_e: Observed electrons as an image or a frame array.
            **kwargs: Instrument, exposure, model paths and options for the term.

        Returns:
            A term whose mask is ``supported_mask(model)`` at construction.
        """
        if isinstance(observed_e, ExposureData):
            term = cls(observed_e, **kwargs)
        else:
            data = observed_e.data if isinstance(observed_e, Image) else observed_e
            term = cls(observed_e, jnp.ones(np.shape(data)[-3:], bool), **kwargs)
        mask = np.asarray(term.mask) & np.asarray(term.supported_mask(model))
        return eqx.tree_at(lambda t: t.mask, term, jnp.asarray(mask))

    @property
    def assumptions(self) -> tuple[str, ...]:
        """``("static_beyond_speckle",)`` when the option is set, else empty."""
        return _assumptions(self.static_beyond_speckle)

    def _instrument(self, model):
        return _held_or_resolved(model, self.instrument, self.instrument_path)

    def _exposure(self, model):
        return _held_or_resolved(model, self.exposure, self.exposure_path)

    def _rates(self, model):
        """Return ``(detector, rates, plan)`` at ``model``."""
        return _render(
            model,
            scene_path=self.scene_path,
            instrument=self.instrument,
            instrument_path=self.instrument_path,
            exposure=self.exposure,
            exposure_path=self.exposure_path,
            static_beyond_speckle=self.static_beyond_speckle,
        )

    @property
    def footprint_paths(self) -> tuple[str, ...]:
        """Model paths that move the field of view (band nodes, primary, detector)."""
        return _footprint_paths(self.instrument_path, self.exposure_path)

    def supported_mask(self, model) -> jax.Array:
        """Return the detector pixels the coronagraph data cover, shape ``(W, ny, nx)``.

        It is ``coronagraphoto.expected(...).supported`` at ``model``, or its
        ``static_supported`` when ``static_beyond_speckle`` is set, all True when the
        rates carry no flag. AND it into ``mask`` (it broadcasts over leading frame
        axes) to score only pixels the model can speak for; ``check`` passes then.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            Boolean array of shape ``(W, ny, nx)``, true where the coronagraph data
            cover the pixel.
        """
        strict, static = _flags(
            resolve(model, self.scene_path),
            self._instrument(model),
            self._exposure(model),
        )
        return static if self.static_beyond_speckle else strict

    def check(self, model) -> None:
        """Raise when the term would score light the coronagraph data do not hold.

        Call it where ``model`` is concrete. Fit checks the reference model and
        the prior endpoints of leaves that move the footprint. The support is
        ``Rates.supported`` by default and ``Rates.static_supported`` when
        ``static_beyond_speckle`` is set. The message
        gives two counts of scored pixels: those no static product covers, and those
        that lack only the speckle field (refused by default, scored under the
        option).

        Args:
            model: The concrete model PyTree holding the scene, and the instrument
                and exposure at their paths when they are not held by the term.

        Raises:
            ValueError: When the frame or mask shape disagrees with the exposure
                and detector, a scored pixel lacks support, or a lit planet lies
                outside the source domain at a band centre.
        """
        scene = resolve(model, self.scene_path)
        instrument = self._instrument(model)
        exposure = self._exposure(model)
        shape = (len(exposure.quadrature.bands), *instrument.detector.shape)
        _check_score_shape(self.mask, self.observed_e, shape)
        strict, static = (
            np.asarray(flag) for flag in _flags(scene, instrument, exposure)
        )
        scored = np.asarray(self.mask)
        no_static = int((scored & ~static).sum())
        speckle_only = int((scored & static & ~strict).sum())
        if self.static_beyond_speckle:
            bad, only = no_static, "scored as static light under the option"
        else:
            bad = no_static + speckle_only
            only = "refused unless static_beyond_speckle=True"
        if bad:
            raise ValueError(
                f"the mask scores {bad} pixel(s) the coronagraph data do not "
                f"cover: {no_static} that no static product covers and "
                f"{speckle_only} that lack only the speckle field ({only}); AND "
                "ImageTerm.supported_mask(model) into the mask or crop the "
                "detector"
            )
        _check_planets(scene, instrument, exposure)

    def __call__(self, model) -> jax.Array:
        """Return the scalar log likelihood, summed over scored pixels in float64.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            The scalar log likelihood, or NaN if a scored pixel lacks support.

        Raises:
            PrecisionError: If float64 is unavailable.
            ValueError: If the observed frame or mask has the wrong shape.
        """
        shape = (
            len(self._exposure(model).quadrature.bands),
            *self._instrument(model).detector.shape,
        )
        _check_score_shape(self.mask, self.observed_e, shape)
        require_float64("image log likelihood")
        rates, score = self.split(model)
        return score(rates)

    def split(self, model):
        """Return the expected rates at ``model`` and the function that scores them.

        ``score(rates)`` is ``__call__(model)`` for the rates of this render, so a
        caller that moves the rates linearly (``detect``) renders once and scores
        many times. ``score`` is an ``eqx.Module`` of arrays that do not depend on the
        model's source amplitudes (the frame, the mask, the detector and the plan).

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            ``(rates, score)``: the :class:`optixstuff.Rates` of the render and the
            callable ``rates -> scalar log likelihood``.
        """
        detector, rates, plan = self._rates(model)
        score = _ImageScore(
            self.observed_e, self.mask, detector, plan, self.static_beyond_speckle
        )
        return rates, score

    def predict(self, model) -> jax.Array:
        """Return the expected electron counts per pixel, shape ``(W, ny, nx)``.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            Expected electron counts per pixel, shape ``(W, ny, nx)``.
        """
        detector, rates, plan = self._rates(model)
        return detector.mean_and_variance(rates, plan)[0]

    def residual(self, model) -> jax.Array:
        """Return ``(observed - mean) / sqrt(variance)`` per pixel, NaN where unscored.

        The mean and variance are the detector's own
        (:meth:`optixstuff.AbstractDetector.mean_and_variance`) of the rates
        ``log_prob`` scores: the total rate clipped at zero, as
        :func:`optixstuff.rates.floor_rate` returns it. For a photon counter this is
        the Pearson residual, skewed where the mean is a few electrons. A pixel is NaN
        where it is not scored, where ``log_prob`` is ``-inf`` (a total below the
        floor's tolerance, or a count a zero mean cannot produce) and where the
        variance is zero. The NaN entries have a zero derivative, so the residual is
        safe to differentiate. Standalone whitening uses the common precision of
        the observations and detector moments. Fit residuals whiten in float64.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            Whitened residual with the shape of ``observed_e``, NaN where a pixel is not
            scored or has no residual.
        """
        detector, rates, plan = self._rates(model)
        observed = jnp.where(self.mask, self.observed_e, 0)
        preserve_floor = type(detector).log_prob is AbstractDetector.log_prob
        rates = _masked_rates(rates, self.mask, preserve_floor=preserve_floor)
        floored, bad = _floored(rates)
        mean, variance = detector.mean_and_variance(floored, plan)
        dtype = jnp.result_type(observed, mean, variance)
        mean, variance = mean.astype(dtype), variance.astype(dtype)
        if preserve_floor:
            # A residual needs the domain and moments, without a log density.
            defined = ~bad & jnp.isfinite(observed) & jnp.isfinite(mean)
            if detector.photon_counting:
                # gammaln(y + 1) has poles at negative integer counts.
                at_pole = (observed < 0) & (observed == jnp.floor(observed))
                defined &= ~at_pole & ((mean > 0) | ((mean == 0) & (observed == 0)))
            else:
                defined &= jnp.isfinite(variance)
        else:
            defined = jnp.isfinite(detector.log_prob(observed, rates, plan))
        ok = self.mask & defined & (variance > 0)
        whitened = (observed - mean) / jnp.sqrt(jnp.where(ok, variance, 1.0))
        return jnp.where(ok, whitened, jnp.nan)

    def _fit_residual(self, model):
        wide = eqx.tree_at(
            lambda term: term.observed_e, self, _as_wide(self.observed_e)
        )
        return wide.residual(model)

    def fisher_root(self, model) -> jax.Array:
        """Return the detector's Fisher root of the expected rates, flat and masked.

        The detector's own :meth:`optixstuff.AbstractDetector.fisher_root` (``2
        sqrt(mean)`` for a photon counter, the mean-and-variance stack otherwise) is
        repeated for every frame of ``observed_e`` and set to zero where ``mask`` is
        false, so ``J^T J`` of this vector is the expected information of
        ``__call__``. Masked entries are zeroed in place, not removed: the mask is a
        traced leaf under ``jit``, and a zero entry adds nothing to ``J^T J``.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            Flat vector holding the Fisher root of every pixel of every frame of
            ``observed_e``, zero where ``mask`` is false.
        """
        detector, rates, plan = self._rates(model)
        root = detector.fisher_root(_masked_rates(rates, self.mask), plan)
        parts = root if root.ndim == 4 else root[None]
        used = jnp.broadcast_to(self.mask, self.observed_e.shape)
        return jnp.ravel(jnp.where(used[..., None, :, :, :], parts, 0.0))

    def draw(self, key, model) -> Self:
        """Copy with frames drawn at ``model``; masked pixels keep their values.

        Args:
            key: PRNG key; one subkey is used per frame of ``observed_e``.
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            A copy of this term whose ``observed_e`` holds the drawn electron counts at
            scored pixels.
        """
        detector, rates, plan = self._rates(model)
        rates = _masked_rates(rates, self.mask, only_nonfinite=True)
        batch = self.observed_e.shape[:-3]
        n_frames = 1
        for size in batch:
            n_frames *= size
        keys = jax.random.split(key, n_frames)
        frames = jax.vmap(lambda k: detector.draw(k, rates, plan))(keys)
        frames = frames.reshape(*batch, *frames.shape[1:])
        observed = jnp.where(self.mask, frames, self.observed_e)
        return eqx.tree_at(lambda term: term.observed_e, self, observed)


_QUANTITIES = (*ELECTRON_COUNT_QUANTITIES, "electron/s")


class ImageForward(eqx.Module):
    """The expected detector image of a model, as the forward of a ``ForwardTerm``.

    It renders with the same function as :class:`ImageTerm` and returns the mean
    detector prediction, so a law that is not the detector's own (a calibrated rate
    image with its stated variance, a robust density) can score it::

        ForwardTerm(
            ImageForward.from_exposure_data(data, instrument_path="instrument"),
            MaskedLogLikelihood.from_exposure_data(data),
        )

    The instrument and the exposure each have one owner, a field of the forward or
    the node of the model at a path, as for :class:`ImageTerm`.

    Attributes:
        instrument: A held :class:`optixstuff.Instrument`, or ``None``.
        exposure: A held :class:`optixstuff.Exposure` with a ``plan``, or ``None``.
        scene_path: Model path of the :class:`skyscapes.Scene`; ``""`` is the model.
        instrument_path: Model path of the instrument, or ``None`` when it is held.
        exposure_path: Model path of the exposure, or ``None`` when it is held.
        quantity: ``"electrons"`` (or its singular spelling ``"electron"``, kept as
            given in :meth:`image`), the mean electrons accumulated over the plan (what
            :meth:`ImageTerm.predict` returns), or ``"electron/s"``, that mean divided
            by the plan's live time: the mean detector count rate, dark terms included.
        static_beyond_speckle: The option of :class:`ImageTerm`.
        reads: Model paths a held build cached; empty.
    """

    instrument: Instrument | None = None
    exposure: Exposure | None = None
    scene_path: str = eqx.field(static=True, default="")
    instrument_path: str | None = eqx.field(static=True, default=None)
    exposure_path: str | None = eqx.field(static=True, default=None)
    quantity: str = eqx.field(static=True, default="electrons")
    static_beyond_speckle: bool = eqx.field(static=True, default=False)
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require one source each for instrument and exposure and a known quantity."""
        _check_exclusive(
            self.instrument, self.instrument_path, self.exposure, self.exposure_path
        )
        if self.quantity not in _QUANTITIES:
            raise ValueError(
                f"quantity must be one of {_QUANTITIES}, got {self.quantity!r}"
            )

    @classmethod
    def from_exposure_data(
        cls,
        data: ExposureData,
        *,
        instrument: Instrument | None = None,
        instrument_path: str | None = None,
        exposure_path: str | None = None,
        scene_path: str = "",
        static_beyond_speckle: bool = False,
    ) -> Self:
        """Build the forward of a measured image: its quantity, grid and exposure.

        The quantity is the data's (``"electrons"`` or ``"electron/s"``). The data's
        exposure is held unless ``exposure_path`` is given, which takes precedence and
        stores nothing from the data.

        Args:
            data: Measured values with an integral measure and no event axis.
            instrument: The held instrument, whose detector grid the data must be on,
                or ``None`` with ``instrument_path``.
            instrument_path: Model path of the instrument, or ``None``.
            exposure_path: Model path of the exposure, or ``None``.
            scene_path: Model path of the scene.
            static_beyond_speckle: The option of :class:`ImageTerm`.

        Returns:
            The forward whose output is comparable with ``data.image.data``.

        Raises:
            ValueError: For another quantity, a measure that is not integral, an
                event axis, or a grid that is not the held detector's.
        """
        image = data.image
        if image.quantity not in _QUANTITIES:
            raise ValueError(
                f"the data holds {image.quantity!r}; ImageForward predicts one of "
                f"{_QUANTITIES}"
            )
        _check_data_image(image, instrument, what="ImageForward")
        return cls(
            instrument=instrument,
            exposure=data.exposure if exposure_path is None else None,
            scene_path=scene_path,
            instrument_path=instrument_path,
            exposure_path=exposure_path,
            quantity=image.quantity,
            static_beyond_speckle=static_beyond_speckle,
        )

    def rates(self, model):
        """Return the detector, rates and plan at ``model``, as ``ImageTerm`` renders.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held.

        Returns:
            ``(detector, rates, plan)``: the instrument's detector, the
            :class:`optixstuff.Rates` of the render and the exposure's frame plan.
        """
        return _render(
            model,
            scene_path=self.scene_path,
            instrument=self.instrument,
            instrument_path=self.instrument_path,
            exposure=self.exposure,
            exposure_path=self.exposure_path,
            static_beyond_speckle=self.static_beyond_speckle,
        )

    def __call__(self, model) -> jax.Array:
        """Return the predicted image, shape ``(W, ny, nx)``, in ``quantity``.

        Args:
            model: The model PyTree, as for :meth:`rates`.

        Returns:
            The mean detector prediction in ``quantity``.
        """
        detector, rates, plan = self.rates(model)
        mean = detector.mean_and_variance(rates, plan)[0]
        if self.quantity in ELECTRON_COUNT_QUANTITIES:
            return mean
        return mean / jnp.atleast_1d(plan.live_time_s)[:, None, None]

    def image(self, model) -> Image:
        """Return the prediction as a named :class:`optixstuff.grids.Image`.

        The grid is the detector's with no orientation, the axes are ``("band",)``,
        the measure is integral and the quantity is :attr:`quantity`.

        Args:
            model: The model PyTree, as for :meth:`rates`.

        Returns:
            The named prediction; its ``data`` is ``self(model)``.
        """
        instrument = _held_or_resolved(model, self.instrument, self.instrument_path)
        return Image(
            self(model),
            _data_grid(instrument.detector.grid),
            ("band",),
            Measure.integral(),
            self.quantity,
        )

    @property
    def footprint_paths(self) -> tuple[str, ...]:
        """Model paths that move the field of view (band nodes, primary, detector)."""
        return _footprint_paths(self.instrument_path, self.exposure_path)

    @property
    def assumptions(self) -> tuple[str, ...]:
        """``("static_beyond_speckle",)`` when the option is set, else empty."""
        return _assumptions(self.static_beyond_speckle)

    def check(self, model) -> None:
        """Raise when a lit planet lies outside the coronagraph's source domain.

        Call it where ``model`` is concrete. The forward scores no pixels, so the
        coverage of a mask is the likelihood's to state.

        Args:
            model: The concrete model PyTree, as for :meth:`rates`.

        Raises:
            ValueError: When a lit planet lies outside the coronagraph's source domain
                at a band centre.
        """
        _check_sources(
            model,
            scene_path=self.scene_path,
            instrument=self.instrument,
            instrument_path=self.instrument_path,
            exposure=self.exposure,
            exposure_path=self.exposure_path,
        )
