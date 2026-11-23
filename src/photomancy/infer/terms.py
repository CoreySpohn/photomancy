"""Generic terms for a :class:`photomancy.infer.fit.Fit`.

:class:`ForwardTerm` is the generic term of a fit whose model is rendered by one
function: the likelihood of a forward model's output.
"""

from typing import Self

import equinox as eqx

from photomancy.infer.fit import AbstractTerm

__all__ = ["ForwardTerm"]


class ForwardTerm(AbstractTerm):
    """The likelihood of a forward model's output, as one term of a ``Fit``.

    ``predict(model)`` is ``forward(model)`` and the value is
    ``likelihood(forward(model))``. A forward that holds a build made from model
    values (an exposure built from the scene, say) declares the model paths that
    build read in its own ``reads``; the term reports them, so ``Fit`` refuses to free
    a path the forward would not see change.

    Attributes:
        forward: ``model -> prediction``, an ``eqx.Module`` so that the arrays it
            holds (a grid, a mask, a table) stay leaves of the term and are traced
            instead of compiled in. A closure is refused.
        likelihood: ``prediction -> scalar`` normalized log likelihood, normally a
            builder from :mod:`photomancy.infer.likelihoods` that holds the data, an
            ``eqx.Module`` for the same reason as ``forward``. Draws need one with a
            ``draw(key, predicted)`` method that returns the likelihood holding the
            drawn data.
    """

    forward: eqx.Module
    likelihood: eqx.Module

    @property
    def reads(self) -> tuple[str, ...]:
        """Model paths the forward's held build read: its ``reads``, else none."""
        return tuple(getattr(self.forward, "reads", ()))

    def __check_init__(self):
        """Refuse a forward or likelihood that is not an ``eqx.Module``."""
        for name in ("forward", "likelihood"):
            value = getattr(self, name)
            if not isinstance(value, eqx.Module):
                raise TypeError(
                    f"{name} is a {type(value).__name__}; make it an eqx.Module "
                    "so the arrays it holds stay pytree leaves"
                )

    def predict(self, model):
        """Return the forward model's output at ``model``.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            The output of ``forward`` at ``model``.
        """
        return self.forward(model)

    def __call__(self, model):
        """Return the scalar log likelihood of the data at ``model``.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            Scalar log likelihood of the held data at ``model``.
        """
        return self.likelihood(self.forward(model))

    def fisher_root(self, model):
        """Return the likelihood's Fisher root of the forward model's output.

        Available when the likelihood is a :class:`MaskedLogLikelihood` (the Gaussian
        and the robust builders); any other likelihood raises ``NotImplementedError``.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            The likelihood's flat Fisher root of the forward output.
        """
        if not hasattr(self.likelihood, "fisher_root"):
            raise NotImplementedError(
                f"{type(self.likelihood).__name__} has no closed-form Fisher root"
            )
        return self.likelihood.fisher_root(self.forward(model))

    def draw(self, key, model) -> Self:
        """Copy holding data drawn from the likelihood at ``model``.

        Args:
            key: PRNG key for the data realization.
            model: The model at one draw, every leaf an array.

        Returns:
            A copy of the term whose likelihood holds the drawn data.

        Raises:
            TypeError: When the likelihood has no ``draw(key, predicted)`` method.
        """
        if not hasattr(self.likelihood, "draw"):
            raise TypeError("likelihood must implement draw(key, predicted)")
        likelihood = self.likelihood.draw(key, self.forward(model))
        return eqx.tree_at(lambda term: term.likelihood, self, likelihood)
