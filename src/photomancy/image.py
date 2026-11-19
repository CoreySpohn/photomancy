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

Count regime. The Poisson form needs integer counts, and a detector that is not
``photon_counting`` scores the moment-matched Gaussian, which is not exact at a few
electrons per pixel with read noise.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from coronagraphoto import expected
from hwoutils.model_paths import join, resolve
from optixstuff import Exposure, Instrument
from skyscapes.scene import Scene

from photomancy.infer.fit import AbstractTerm

__all__ = ["ImageModel", "ImageTerm"]


def _bool(x):
    return jnp.asarray(x, dtype=bool)


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

    Attributes:
        observed_e: Observed electrons, shape ``(*batch, W, ny, nx)``. Leading batch
            axes are further frames of the same exposure model (scored jointly).
        mask: Boolean, broadcastable to ``observed_e``; true where a pixel is scored.
            Observed values at masked pixels are never read.
        instrument: A held :class:`optixstuff.Instrument`, or ``None``.
        exposure: A held :class:`optixstuff.Exposure` with a ``plan``, or ``None``.
        scene_path: Model path of the :class:`skyscapes.Scene`; ``""`` is the model
            itself.
        instrument_path: Model path of the instrument, or ``None`` when it is held.
        exposure_path: Model path of the exposure, or ``None`` when it is held.
            Exactly one of ``instrument`` and ``instrument_path``, and exactly one of
            ``exposure`` and ``exposure_path``, is given.
    """

    observed_e: jax.Array = eqx.field(converter=jnp.asarray)
    mask: jax.Array = eqx.field(converter=_bool)
    instrument: Instrument | None = None
    exposure: Exposure | None = None
    scene_path: str = eqx.field(static=True, default="")
    instrument_path: str | None = eqx.field(static=True, default=None)
    exposure_path: str | None = eqx.field(static=True, default=None)

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

    def __call__(self, model) -> jax.Array:
        """Return the scalar log likelihood, summed over scored pixels in float64."""
        detector, rates, plan = self._rates(model)
        observed = jnp.where(self.mask, self.observed_e, 0.0)
        log_prob = detector.log_prob(observed, rates, plan)
        return jnp.sum(jnp.where(self.mask, log_prob, 0.0), dtype=jnp.float64)

    def predict(self, model) -> jax.Array:
        """Return the expected electron counts per pixel, shape ``(W, ny, nx)``."""
        detector, rates, plan = self._rates(model)
        return detector.mean_and_variance(rates, plan)[0]

    def draw(self, key, model) -> Self:
        """Copy with frames drawn at ``model``; masked pixels keep their values."""
        detector, rates, plan = self._rates(model)
        batch = self.observed_e.shape[:-3]
        n_frames = 1
        for size in batch:
            n_frames *= size
        keys = jax.random.split(key, n_frames)
        frames = jax.vmap(lambda k: detector.draw(k, rates, plan))(keys)
        frames = frames.reshape(*batch, *frames.shape[1:])
        observed = jnp.where(self.mask, frames, self.observed_e)
        return eqx.tree_at(lambda term: term.observed_e, self, observed)
