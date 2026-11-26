"""Generic terms for a :class:`photomancy.infer.fit.Fit`.

:class:`ForwardTerm` is the generic term of a fit whose model is rendered by one
function: the likelihood of a forward model's output.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.precision import require_float64

from photomancy.infer.fit import AbstractTerm
from photomancy.infer.likelihoods import (
    LowRankGaussianLogLikelihood,
    MaskedLogLikelihood,
)

__all__ = ["ForwardTerm"]

_BUILTIN_LIKELIHOODS = (MaskedLogLikelihood, LowRankGaussianLogLikelihood)


def _wide_leaf(x):
    if eqx.is_inexact_array(x):
        return x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)
    return x


def _score_prediction(likelihood, predicted):
    require_float64("ForwardTerm log likelihood")
    # The built-in densities widen their covariance and normalization together.
    # Custom scorers receive wide data as well as a wide prediction.
    likelihood = jax.tree.map(
        _wide_leaf,
        likelihood,
        is_leaf=lambda x: isinstance(x, _BUILTIN_LIKELIHOODS),
    )
    value = likelihood(jax.tree.map(_wide_leaf, predicted))
    return jnp.asarray(value, jnp.float64)


class ForwardTerm(AbstractTerm):
    """The likelihood of a forward model's output, as one term of a ``Fit``.

    ``predict(model)`` is ``forward(model)`` and the value is
    ``likelihood(forward(model))``. A forward that holds a build made from model
    values (an exposure built from the scene, say) declares the model paths that
    build read in its own ``reads``; the term reports them, so ``Fit`` refuses to free
    a path the forward would not see change.

    Attributes:
        forward_fields: Forward differentiation uses ``forward``.
        forward: ``model -> prediction``, an ``eqx.Module`` so that the arrays it
            holds (a grid, a mask, a table) stay leaves of the term and are traced
            instead of compiled in. A closure is refused.
        likelihood: ``prediction -> scalar`` normalized log likelihood, normally a
            builder from :mod:`photomancy.infer.likelihoods` that holds the data, an
            ``eqx.Module`` for the same reason as ``forward``. Draws need one with a
            ``draw(key, predicted)`` method that returns the likelihood holding the
            drawn data.
    """

    forward_fields = ("forward",)
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

    def split(self, model):
        """Return the forward's output at ``model`` and the likelihood that scores it.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            The raw prediction and its float64 scoring callable.
        """
        score = self.likelihood
        if not isinstance(score, _BUILTIN_LIKELIHOODS):
            score = eqx.Partial(_score_prediction, score)
        return self.forward(model), score

    def __call__(self, model):
        """Return the scalar log likelihood of the data at ``model``.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            Scalar log likelihood of the held data at ``model``.
        """
        require_float64("ForwardTerm log likelihood")
        return _score_prediction(self.likelihood, self.forward(model))

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

    def residual(self, model):
        """Return the likelihood's whitened residual of the forward model's output.

        Available when the likelihood has one (the Gaussian builder); any other
        likelihood raises ``NotImplementedError``.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            The whitened residual, shaped like the likelihood's data, NaN where an
            element is not used.
        """
        require_float64("ForwardTerm residual")
        if not hasattr(self.likelihood, "residual"):
            raise NotImplementedError(
                f"{type(self.likelihood).__name__} has no whitened residual"
            )
        likelihood = jax.tree.map(_wide_leaf, self.likelihood)
        predicted = jax.tree.map(_wide_leaf, self.forward(model))
        return likelihood.residual(predicted)

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
