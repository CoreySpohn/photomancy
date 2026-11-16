"""The term type of a fit and the sum of terms over a shared model PyTree.

A term holds some data and maps the model to ``(log likelihood, prediction)``. An
image is one kind of term (:mod:`photomancy.exposure`); measurements of a planetary
system that are not images are others (:mod:`photomancy.system_terms`).
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
from jaxtyping import PyTree, Scalar

from photomancy.core.model import AbstractLogDensity

__all__ = ["AbstractTerm", "TermLogDensity"]


class AbstractTerm(eqx.Module):
    """One data contribution evaluated against a shared model."""

    @abstractmethod
    def __call__(self, model) -> tuple[Scalar, PyTree]:
        """Return a scalar log-likelihood and its prediction PyTree."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, key, model) -> Self:
        """Copy this term with a draw at ``model``, preserving invalid entries."""
        raise NotImplementedError


class TermLogDensity(AbstractLogDensity):
    """Sum of terms plus an optional prior over a shared model.

    ``value_and_aux`` evaluates each term once and returns the total with the
    predictions in term order. ``restrict`` selects a view with the same prior.

    Attributes:
        terms: Non-empty tuple of likelihood terms.
        prior: ``model -> scalar`` log-prior, or ``None`` for a flat one.
    """

    terms: tuple[AbstractTerm, ...]
    prior: Callable | None = None

    def __check_init__(self):
        """Require at least one term."""
        if len(self.terms) == 0:
            raise ValueError(f"an {type(self).__name__} needs at least one term")

    def value_and_aux(self, params):
        """Return ``(log_density, predictions)`` with one prediction per term."""
        results = [term(params) for term in self.terms]
        total = results[0][0]
        for value, _ in results[1:]:
            total = total + value
        predictions = [prediction for _, prediction in results]
        if self.prior is not None:
            total = total + self.prior(params)
        return total, tuple(predictions)

    def __call__(self, params):
        """Return the scalar log-density at ``params``."""
        return self.value_and_aux(params)[0]

    def sample(self, key, model) -> Self:
        """Copy with independent draws, splitting the key in term order."""
        keys = jax.random.split(key, len(self.terms))
        terms = tuple(
            term.sample(subkey, model)
            for term, subkey in zip(self.terms, keys, strict=True)
        )
        return eqx.tree_at(lambda density: density.terms, self, terms)

    def restrict(self, indices):
        """A view over the terms at ``indices``, with the same prior.

        Args:
            indices: Positions in ``terms``, a sequence of ints.

        Returns:
            A log-density of the same type over those terms, in the order given. Its
            predictions are indexed by position in the view, not in the original.
        """
        indices = tuple(indices)
        if not all(0 <= i < len(self.terms) for i in indices):
            raise IndexError(
                f"term indices {indices} out of range for {len(self.terms)} terms"
            )
        return type(self)(terms=tuple(self.terms[i] for i in indices), prior=self.prior)
