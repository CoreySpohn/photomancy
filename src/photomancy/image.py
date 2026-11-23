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
nothing, so a term with a plain exposure reads nothing. A built product that carries a
``reads`` tuple of scene paths (as ``coronagraphoto.ApertureRecord`` does) and is held
as the exposure has its entries prefixed with the scene path
(:func:`hwoutils.model_paths.join`) and declared in ``reads``.

**What the term refuses.** The coronagraph renders only what its data hold: a detector
pixel its tables do not cover is zero with ``Rates.supported`` False, and a planet
outside its source domain contributes zero. A likelihood scored there would compare
the data with a model that says nothing, so :meth:`ImageTerm.check` raises when the
mask scores such a pixel or a lit planet is outside the domain, and
:meth:`ImageTerm.supported_mask` is the boolean to AND into the mask. ``Fit`` calls
``check`` once at setup, and again at both ends of the prior of any free leaf that
moves the field of view (the wavelength nodes, the primary, the detector); nothing is
ever dropped inside ``__call__``.

Count regime. The Poisson form needs integer counts, and a detector that is not
``photon_counting`` scores the moment-matched Gaussian, which is not exact at a few
electrons per pixel with read noise.
"""

from dataclasses import replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from coronagraphoto import expected
from hwoutils.model_paths import join, resolve
from optixstuff import Exposure, Instrument, Rates
from optixstuff.grids import Image, Measure, SkyOrientation
from skyscapes.scene import Scene

from photomancy.infer.fit import AbstractTerm

__all__ = ["ImageModel", "ImageTerm"]


_COUNTS = ("electron", "electrons")


def _bool(x):
    return jnp.asarray(x, dtype=bool)


def _clipped(rates):
    """Rates holding only their total clipped at zero, the values ``log_prob`` scores.

    :func:`optixstuff.rates.floor_rate` returns the total clipped at zero whatever
    its tolerance; the tolerance only flags a total too negative to be integration
    error, which ``log_prob`` scores as ``-inf``.
    """
    variance = None if rates.variance is None else _clipped(rates.variance)
    return Rates(
        rates.bands,
        rates.carrier,
        planet=jnp.maximum(rates.total(), 0.0),
        n_pix=rates.n_pix,
        variance=variance,
        supported=rates.supported,
        static_supported=rates.static_supported,
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
    if image.measure != Measure.integral() or image.quantity not in _COUNTS:
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
    roll = None if exposure is None else exposure.roll_deg
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


@eqx.filter_jit
def _supported(scene, instrument, exposure):
    """The ``(W, ny, nx)`` read support of the expected rates, all True when unset."""
    rates = expected(scene, instrument, exposure)
    if rates.supported is None:
        return jnp.ones(rates.planet.shape, bool)
    return rates.supported


def _lit_planets_outside(scene, instrument, exposure):
    """Number of lit planets outside the coronagraph's source domain at band centre."""
    if not scene.planets:
        return 0
    times = jnp.atleast_1d(exposure.time_d)
    ephemeris = scene.ephemeris(times)
    east, north = scene.positions(times, ephemeris=ephemeris)[:, :, 0]
    x, y = SkyOrientation(exposure.roll_deg).sky_to_detector(east, north)
    frame = instrument.coronagraph.grid.frame
    outside = np.zeros(np.shape(x), bool)
    for band in exposure.quadrature.bands:
        centre = band.center_nm
        lod = frame.lod_to_arcsec(centre)
        psfs = instrument.coronagraph.source_psfs(centre, x / lod, y / lod)
        lit = scene.planet_flux_densities(centre, times, ephemeris=ephemeris)[:, 0] > 0
        outside |= np.asarray(~psfs.valid & lit)
    return int(outside.sum())


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

    Attributes:
        observed_e: Observed electrons, shape ``(*batch, W, ny, nx)``. Leading batch
            axes are further frames of the same exposure model (scored jointly).
        mask: Boolean, broadcastable to ``observed_e``; true where a pixel is scored.
            Observed values at masked pixels are never read; :meth:`check` refuses a
            mask that scores a pixel the coronagraph data do not cover. ``None``
            scores the supported pixels: those the render marks
            ``Rates.supported`` at the model being scored. The scored set then
            follows the model, so a fit that frees a leaf moving the field of view
            (:attr:`footprint_paths`) should give an explicit mask.
        instrument: A held :class:`optixstuff.Instrument`, or ``None``.
        exposure: A held :class:`optixstuff.Exposure` with a ``plan``, or ``None``.
        scene_path: Model path of the :class:`skyscapes.Scene`; ``""`` is the model
            itself.
        instrument_path: Model path of the instrument, or ``None`` when it is held.
        exposure_path: Model path of the exposure, or ``None`` when it is held.
            Exactly one of ``instrument`` and ``instrument_path``, and exactly one of
            ``exposure`` and ``exposure_path``, is given.
    """

    observed_e: jax.Array
    mask: jax.Array | None = None
    instrument: Instrument | None = None
    exposure: Exposure | None = None
    scene_path: str = eqx.field(static=True, default="")
    instrument_path: str | None = eqx.field(static=True, default=None)
    exposure_path: str | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        observed_e,
        mask=None,
        instrument: Instrument | None = None,
        exposure: Exposure | None = None,
        scene_path: str = "",
        instrument_path: str | None = None,
        exposure_path: str | None = None,
    ):
        """Store the frame, checking an ``Image`` against the held instrument.

        Args:
            observed_e: Observed electrons, an :class:`optixstuff.grids.Image` or an
                array of shape ``(*batch, W, ny, nx)``.
            mask: Boolean broadcastable to the frame, or ``None`` to score the
                supported pixels.
            instrument: The held instrument, or ``None`` with ``instrument_path``.
            exposure: The held exposure, or ``None`` with ``exposure_path``.
            scene_path: Model path of the scene.
            instrument_path: Model path of the instrument, or ``None``.
            exposure_path: Model path of the exposure, or ``None``.

        Raises:
            ValueError: When an ``Image`` does not hold electrons per pixel, is not
                on the held detector's grid, or lacks or contradicts the held
                exposure's roll.
        """
        if isinstance(observed_e, Image):
            _check_image(observed_e, instrument, exposure)
            observed_e = observed_e.data
        self.observed_e = jnp.asarray(observed_e)
        self.mask = None if mask is None else _bool(mask)
        self.instrument = instrument
        self.exposure = exposure
        self.scene_path = scene_path
        self.instrument_path = instrument_path
        self.exposure_path = exposure_path

    def __check_init__(self):
        """Require one source each for instrument and exposure, and a frame array."""
        if (self.instrument is None) == (self.instrument_path is None):
            raise ValueError("give exactly one of instrument and instrument_path")
        if (self.exposure is None) == (self.exposure_path is None):
            raise ValueError("give exactly one of exposure and exposure_path")
        if self.exposure is not None and self.exposure.plan is None:
            raise ValueError("the exposure needs a FramePlan: build it with plan=")
        if self.observed_e.ndim < 3:
            raise ValueError("observed_e must have shape (*batch, W, ny, nx)")

    @property
    def reads(self) -> tuple[str, ...]:
        """Scene paths a held built exposure cached, prefixed with the scene path."""
        held = getattr(self.exposure, "reads", ())
        return tuple(join(self.scene_path, path) for path in held)

    def _instrument(self, model):
        if self.instrument_path is None:
            return self.instrument
        return resolve(model, self.instrument_path)

    def _exposure(self, model):
        if self.exposure_path is None:
            return self.exposure
        return resolve(model, self.exposure_path)

    def _rates(self, model):
        """Return ``(detector, rates, plan)`` at ``model``."""
        instrument = self._instrument(model)
        exposure = self._exposure(model)
        rates = expected(resolve(model, self.scene_path), instrument, exposure)
        return instrument.detector, rates, exposure.plan

    def _used(self, rates) -> jax.Array:
        """The scored pixels: the mask, else the render's support, else every pixel."""
        if self.mask is not None:
            return self.mask
        if rates.supported is not None:
            return rates.supported
        return jnp.ones(jnp.shape(rates.total()), bool)

    @property
    def footprint_paths(self) -> tuple[str, ...]:
        """Model paths that move the field of view (band nodes, primary, detector)."""
        paths = []
        if self.exposure_path is not None:
            paths.append(join(self.exposure_path, "quadrature"))
        if self.instrument_path is not None:
            paths += [
                join(self.instrument_path, "primary"),
                join(self.instrument_path, "detector"),
            ]
        return tuple(paths)

    def supported_mask(self, model) -> jax.Array:
        """Return the detector pixels the coronagraph data cover, shape ``(W, ny, nx)``.

        It is ``coronagraphoto.expected(...).supported`` at ``model``, all True when
        the rates carry no flag. AND it into ``mask`` (it broadcasts over leading frame
        axes) to score only pixels the model can speak for; ``check`` passes then.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            Boolean array of shape ``(W, ny, nx)``, true where the coronagraph data
            cover the pixel.
        """
        scene = resolve(model, self.scene_path)
        return _supported(scene, self._instrument(model), self._exposure(model))

    def check(self, model) -> None:
        """Raise when the term would score light the coronagraph data do not hold.

        Call it where ``model`` is concrete (``Fit`` does, once, at setup).

        Args:
            model: The concrete model PyTree holding the scene, and the instrument
                and exposure at their paths when they are not held by the term.

        Raises:
            ValueError: Naming the count, when ``mask`` is true on a pixel that is not
                ``supported_mask(model)`` (a ``None`` mask scores only those), or when
                a lit planet of the scene is outside the coronagraph's source domain
                at band centre (it would render zero while the data may show it).
        """
        bad = 0
        if self.mask is not None:
            supported = np.asarray(self.supported_mask(model))
            scored = np.broadcast_to(np.asarray(self.mask), self.observed_e.shape)
            bad = int((scored & ~supported).reshape(-1, *supported.shape).any(0).sum())
        if bad:
            raise ValueError(
                f"the mask scores {bad} pixel(s) the coronagraph data do not cover "
                "(zero rate, unsupported); AND ImageTerm.supported_mask(model) into "
                "the mask or crop the detector"
            )
        planets = _lit_planets_outside(
            resolve(model, self.scene_path),
            self._instrument(model),
            self._exposure(model),
        )
        if planets:
            raise ValueError(
                f"{planets} lit planet(s) lie outside the coronagraph's source domain "
                "and render as zero; start the fit inside the domain or hold them"
            )

    def __call__(self, model) -> jax.Array:
        """Return the scalar log likelihood, summed over scored pixels in float64.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            The scalar log likelihood of the scored pixels.
        """
        detector, rates, plan = self._rates(model)
        used = self._used(rates)
        observed = jnp.where(used, self.observed_e, 0.0)
        log_prob = detector.log_prob(observed, rates, plan)
        return jnp.sum(jnp.where(used, log_prob, 0.0), dtype=jnp.float64)

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
        safe to differentiate.

        Args:
            model: The model PyTree holding the scene, and the instrument and exposure
                at their paths when they are not held by the term.

        Returns:
            Whitened residual with the shape of ``observed_e``, NaN where a pixel is not
            scored or has no residual.
        """
        detector, rates, plan = self._rates(model)
        used = self._used(rates)
        observed = jnp.where(used, self.observed_e, 0.0)
        mean, variance = detector.mean_and_variance(_clipped(rates), plan)
        defined = jnp.isfinite(detector.log_prob(observed, rates, plan))
        ok = used & defined & (variance > 0)
        whitened = (observed - mean) / jnp.sqrt(jnp.where(ok, variance, 1.0))
        return jnp.where(ok, whitened, jnp.nan)

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
        root = detector.fisher_root(rates, plan)
        parts = root if root.ndim == 4 else root[None]
        used = jnp.broadcast_to(self._used(rates), self.observed_e.shape)
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
        batch = self.observed_e.shape[:-3]
        n_frames = 1
        for size in batch:
            n_frames *= size
        keys = jax.random.split(key, n_frames)
        frames = jax.vmap(lambda k: detector.draw(k, rates, plan))(keys)
        frames = frames.reshape(*batch, *frames.shape[1:])
        observed = jnp.where(self._used(rates), frames, self.observed_e)
        return eqx.tree_at(lambda term: term.observed_e, self, observed)
