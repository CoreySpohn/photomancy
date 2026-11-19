"""The sum of terms over a shared model PyTree, for the flat-vector backends.

The term type itself is :class:`photomancy.infer.fit.AbstractTerm`; a term
maps the model to a scalar log likelihood and predicts separately.
:class:`TermLogDensity` evaluates both for a backend that wants the value and the
predictions together.
"""

from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax

from photomancy.infer.fit import AbstractTerm
from photomancy.infer.model import AbstractLogDensity

__all__ = ["TermLogDensity"]


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
        values = [term(params) for term in self.terms]
        total = values[0]
        for value in values[1:]:
            total = total + value
        predictions = [term.predict(params) for term in self.terms]
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
            term.draw(subkey, model)
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
