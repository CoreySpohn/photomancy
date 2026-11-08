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
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx

from photomancy.core.model import AbstractLogDensity

__all__ = ["ExposureLogDensity", "ExposureModel", "ExposureTerm"]


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


class ExposureTerm(eqx.Module):
    """One exposure: a render function and the likelihood of its image.

    Attributes:
        render: ``(instrument, exposure, scene) -> image``, the injected forward
            model. An ``eqx.Module`` keeps its arrays (masks, grids) as traced
            leaves rather than compiled constants.
        likelihood: ``image -> scalar log-likelihood``, normally a builder from
            :mod:`photomancy.likelihoods` holding the data, the per-pixel scale and
            the boolean mask. ``laplace_loglike`` with scale 1 is an L1 loss,
            ``huber_loglike`` with scale 1 and ``delta=1`` is a smooth-L1 loss, and a
            noise array as the scale gives the noise-weighted kinds.
        exposure_index: Which entry of ``ExposureModel.exposures`` this term reads.
    """

    render: Callable
    likelihood: Callable
    exposure_index: int = eqx.field(static=True, default=0)

    def predict(self, model):
        """Render this exposure's image from ``model``."""
        return self.render(
            model.instrument, model.exposures[self.exposure_index], model.scene
        )

    def __call__(self, model):
        """Return ``(log_likelihood, predicted_image)`` at ``model``."""
        image = self.predict(model)
        return self.likelihood(image), image


class ExposureLogDensity(AbstractLogDensity):
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

    def __check_init__(self):
        """Require at least one term."""
        if len(self.terms) == 0:
            raise ValueError("an ExposureLogDensity needs at least one term")

    def value_and_aux(self, params):
        """Return ``(log_density, predictions)`` with one prediction per term."""
        results = [term(params) for term in self.terms]
        total = results[0][0]
        for value, _ in results[1:]:
            total = total + value
        images = [image for _, image in results]
        if self.prior is not None:
            total = total + self.prior(params)
        return total, tuple(images)

    def __call__(self, params):
        """Return the scalar log-density at ``params`` (an ``ExposureModel``)."""
        return self.value_and_aux(params)[0]

    def restrict(self, indices):
        """A view over the terms at ``indices``, with the same prior.

        Args:
            indices: Positions in ``terms``, a sequence of ints.

        Returns:
            An ``ExposureLogDensity`` over those terms, in the order given. Its
            predictions are indexed by position in the view, not in the original.
        """
        indices = tuple(indices)
        if not all(0 <= i < len(self.terms) for i in indices):
            raise IndexError(
                f"term indices {indices} out of range for {len(self.terms)} terms"
            )
        return ExposureLogDensity(
            terms=tuple(self.terms[i] for i in indices), prior=self.prior
        )
