"""Terms over a set of exposures that share one instrument.

The structure of a multi-exposure coronagraph fit (Ferrer-Chavez et al. 2026): an
instrument whose seams (optical path differences, a detector flat) are the same in
every exposure, a few nuisance parameters per exposure (the star position and flux),
and a scene (the companions). Each exposure that was taken contributes one term, the
likelihood of its image under a render function that maps ``(instrument, exposure,
scene)`` to a predicted image. Which parameters are fitted, in which order, is the
business of the staged backend, not of this module; the render function is injected,
so no optics library is imported here.

``ExposureModel`` is the parameter PyTree and ``ExposureTerm`` one exposure's render and
likelihood. The fit is ``run(Fit(ExposureModel(...), free, terms), MAP(...))``; a stage
that scores one exposure at a time (the reference star, then the science target)
gives ``MAPStage`` a ``Fit`` over a subset of the terms with the same free entries, and
a closed-form update reads the per-term predictions from before the step.

**The model of an image fit.** A term reads the model it is given and nothing else,
and a free path names a leaf of that model, so the model must contain every leaf a
fit frees. The model owns the scene and the instrument (one owner each per fit), and
:class:`photomancy.image.ImageTerm`, the pixel likelihood over
``coronagraphoto.expected``, reads them by path. An :class:`ExposureModel` is the
record for the render-function terms: ``instrument``, ``exposures`` (one PyTree of
per-exposure leaves such as a star position or flux) and ``scene``, so a path is
``"instrument.<leaf>"``, ``"exposures[i].<leaf>"`` or ``"scene.<path>"``. A fit of an
:class:`~photomancy.image.ImageTerm` uses :class:`photomancy.image.ImageModel`.

Neither record lets a term build or cache anything from the model, so an image term
declares no ``reads``; a host build a term holds (an aperture record made from scene
values) declares the scene paths it read, prefixed with the scene's path in the model.
The likelihood builders of :mod:`photomancy.infer.likelihoods` are normalized, so an
:class:`ExposureTerm` built from one carries its constants.
"""

from collections.abc import Callable
from typing import Any, ClassVar, Self

import equinox as eqx

from photomancy.infer.fit import AbstractTerm

__all__ = ["ExposureModel", "ExposureTerm"]


class ExposureModel(eqx.Module):
    """The parameters of a multi-exposure fit.

    Attributes:
        instrument: The shared instrument: the band's optical path with its seams, and
            the detector response. Whatever the render functions read from it.
        exposures: One nuisance PyTree per exposure block (star position and flux,
            later drifts), a tuple. Several terms may read the same block.
        scene: The sources: the companions, with their fluxes (a companion fixed at
            flux zero belongs to a group no stage frees).
    """

    instrument: Any
    exposures: tuple
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
        exposure_index: Which entry of ``ExposureModel.exposures`` this term reads.
        reads: Model paths a held build cached; empty, because the term holds no build.
    """

    render: Callable
    likelihood: Callable
    exposure_index: int = eqx.field(static=True, default=0)
    reads: ClassVar[tuple[str, ...]] = ()

    def predict(self, model):
        """Render this exposure's image from ``model``.

        Args:
            model: The :class:`ExposureModel` holding the instrument, the exposure
                blocks and the scene.

        Returns:
            The image that ``render`` produces for this term's exposure block.
        """
        return self.render(
            model.instrument, model.exposures[self.exposure_index], model.scene
        )

    def __call__(self, model):
        """Return the scalar log-likelihood of the data at ``model``.

        Args:
            model: The :class:`ExposureModel` to score.

        Returns:
            The scalar log-likelihood of the held data at the rendered image.
        """
        return self.likelihood(self.predict(model))

    def fisher_root(self, model):
        """Return the likelihood's Fisher root of the rendered image.

        Available when the likelihood is a :class:`MaskedLogLikelihood`; an arbitrary
        callable raises ``NotImplementedError``.

        Args:
            model: The :class:`ExposureModel` at which the image is rendered.

        Returns:
            The flat Fisher-root vector that the likelihood's ``fisher_root`` gives for
            the rendered image.
        """
        if not hasattr(self.likelihood, "fisher_root"):
            raise NotImplementedError(
                f"{type(self.likelihood).__name__} has no closed-form Fisher root"
            )
        return self.likelihood.fisher_root(self.predict(model))

    def draw(self, key, model) -> Self:
        """Copy with a draw from an injected likelihood supporting ``draw``.

        The likelihood must implement ``draw(key, predicted)`` returning a
        likelihood containing the drawn image; arbitrary score callables cannot
        define a generative law.

        Args:
            key: PRNG key for the draw.
            model: The :class:`ExposureModel` at which the image is rendered.

        Returns:
            A copy of this term whose likelihood holds the drawn image.
        """
        if not hasattr(self.likelihood, "draw"):
            raise TypeError("image likelihood must implement draw(key, predicted)")
        likelihood = self.likelihood.draw(key, self.predict(model))
        return eqx.tree_at(lambda term: term.likelihood, self, likelihood)
