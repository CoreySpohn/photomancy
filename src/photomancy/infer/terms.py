"""Generic terms for a :class:`photomancy.infer.fit.Fit`.

:class:`ForwardTerm` is the generic term of a fit whose model is rendered by one
function: the likelihood of a forward model's output.
"""

from collections.abc import Callable
from typing import ClassVar, Self

import equinox as eqx

from photomancy.infer.fit import AbstractTerm

__all__ = ["ForwardTerm"]


class ForwardTerm(AbstractTerm):
    """The likelihood of a forward model's output, as one term of a ``Fit``.

    ``predict(model)`` is ``forward(model)`` and the value is
    ``likelihood(forward(model))``. The term holds no build made from model values,
    so it reads no model path.

    Attributes:
        forward: ``model -> prediction``, an ``eqx.Module`` so that the arrays it
            holds (a grid, a mask, a table) stay leaves of the term and are traced
            instead of compiled in. A closure is refused.
        likelihood: ``prediction -> scalar`` normalized log likelihood, normally a
            builder from :mod:`photomancy.infer.likelihoods` that holds the data. Draws
            need one with a ``sample(key, predicted)`` method that returns the
            likelihood holding the drawn data.
    """

    forward: eqx.Module
    likelihood: Callable
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Refuse a forward that is not an ``eqx.Module``."""
        if not isinstance(self.forward, eqx.Module):
            raise TypeError(
                f"forward is a {type(self.forward).__name__}; make it an eqx.Module "
                "so the arrays it holds stay pytree leaves"
            )

    def predict(self, model):
        """Return the forward model's output at ``model``."""
        return self.forward(model)

    def __call__(self, model):
        """Return the scalar log likelihood of the data at ``model``."""
        return self.likelihood(self.forward(model))

    def draw(self, key, model) -> Self:
        """Copy holding data drawn from the likelihood at ``model``.

        Raises:
            TypeError: When the likelihood has no ``sample(key, predicted)`` method.
        """
        if not hasattr(self.likelihood, "sample"):
            raise TypeError("likelihood must implement sample(key, predicted)")
        likelihood = self.likelihood.sample(key, self.forward(model))
        return eqx.tree_at(lambda term: term.likelihood, self, likelihood)
