"""A log-density over a set of exposures that share one instrument.

The structure of a multi-exposure coronagraph fit (Ferrer-Chavez et al. 2026): an
instrument whose seams (optical path differences, a detector flat) are the same in
every exposure, a few nuisance parameters per exposure (the star position and flux),
and a scene (the companions). Each exposure that was taken contributes one term, the
likelihood of its image under a render function that maps ``(instrument, exposure,
scene)`` to a predicted image. Which parameters are fitted, in which order, is the
business of the staged backend, not of this module; the render function is injected,
so no optics library is imported here.

``ExposureModel`` is the parameter PyTree, ``ExposureTerm`` one exposure's render and
likelihood, and ``ExposureLogDensity`` the sum over terms, with the per-term
predictions as auxiliary output. ``restrict`` gives a view over some of the terms,
which is how a staged fit scores one exposure at a time (the reference star, then the
science target) against one shared parameter PyTree.

**The model of an image fit.** A term reads the model it is given and nothing else,
and a free path names a leaf of that model, so the model must contain every leaf a
fit frees. The model owns the scene and the instrument (one owner each per fit), and
:class:`photomancy.image.ImageTerm`, the pixel likelihood over
``coronagraphoto.expected``, reads them by path. An :class:`ExposureModel` is the
record that holds them: ``instrument``, ``nuisance`` (one PyTree of per-exposure
nuisance leaves such as a star position or flux; these are not
``optixstuff.Exposure`` objects, which an ``ImageTerm`` holds or reads from the model
by its own path) and ``scene``, so a path is ``"instrument.<leaf>"``,
``"nuisance[i].<leaf>"`` or ``"scene.<path>"``. A fit that frees scene leaves only may
use any record with a ``scene`` and an ``instrument``.

Neither shape lets a term build or cache anything from the model, so an image term
declares no ``reads``; a host build a term holds (an aperture record made from scene
values) declares the scene paths it read, prefixed with the scene's path in the model.
The likelihood builders of :mod:`photomancy.infer.likelihoods` are normalized, so an
:class:`ExposureTerm` built from one carries its constants.
"""

from collections.abc import Callable
from typing import Any, ClassVar, Self

import equinox as eqx

from photomancy.infer.fit import AbstractTerm
from photomancy.infer.terms import TermLogDensity

__all__ = ["ExposureLogDensity", "ExposureModel", "ExposureTerm"]


class ExposureModel(eqx.Module):
    """The parameters of a multi-exposure fit.

    Attributes:
        instrument: The shared instrument: the band's optical path with its seams, and
            the detector response. Whatever the render functions read from it.
        nuisance: One nuisance PyTree per exposure block (star position and flux,
            later drifts), a tuple. Several terms may read the same block.
        scene: The sources: the companions, with their fluxes (a companion fixed at
            flux zero belongs to a group no stage frees).
    """

    instrument: Any
    nuisance: tuple
    scene: Any


class ExposureTerm(AbstractTerm):
    """One exposure: a render function and the likelihood of its image.

    Attributes:
        render: ``(instrument, exposure, scene) -> image``, the injected forward
            model. An ``eqx.Module`` keeps its arrays (masks, grids) as traced
            leaves rather than compiled constants.
        likelihood: ``image -> scalar log-likelihood``, normally a builder from
            :mod:`photomancy.infer.likelihoods` holding the data, the per-pixel scale
            and the boolean mask. ``laplace_loglike`` with scale 1 is an L1 loss,
            ``huber_loglike`` with scale 1 and ``delta=1`` is a smooth-L1 loss, and a
            noise array as the scale gives the noise-weighted kinds.
        nuisance_index: Which entry of ``ExposureModel.nuisance`` this term reads.
    """

    render: Callable
    likelihood: Callable
    nuisance_index: int = eqx.field(static=True, default=0)
    reads: ClassVar[tuple[str, ...]] = ()

    def predict(self, model):
        """Render this exposure's image from ``model``."""
        return self.render(
            model.instrument, model.nuisance[self.nuisance_index], model.scene
        )

    def value_and_prediction(self, model):
        """Return the log-likelihood and the image, rendering once."""
        image = self.predict(model)
        return self.likelihood(image), image

    def __call__(self, model):
        """Return the scalar log-likelihood of the data at ``model``."""
        return self.likelihood(self.predict(model))

    def draw(self, key, model) -> Self:
        """Copy with a draw from an injected likelihood supporting ``sample``.

        The likelihood must implement ``sample(key, predicted)`` returning a
        likelihood containing the drawn image; arbitrary score callables cannot
        define a generative law.
        """
        if not hasattr(self.likelihood, "sample"):
            raise TypeError("image likelihood must implement sample(key, predicted)")
        likelihood = self.likelihood.sample(key, self.predict(model))
        return eqx.tree_at(lambda term: term.likelihood, self, likelihood)


class ExposureLogDensity(TermLogDensity):
    """Sum of exposure terms plus an optional prior, over an :class:`ExposureModel`.

    ``value_and_aux`` returns the total and the tuple of per-term predictions (in term
    order), so a caller that needs both the value and the images (a closed-form flux
    update, a progress callback) renders each exposure once.

    Attributes:
        terms: Tuple of :class:`ExposureTerm`.
        prior: ``model -> scalar`` log-prior, or ``None`` for a flat one.
    """

    terms: tuple[ExposureTerm, ...]
    prior: Callable | None = None
