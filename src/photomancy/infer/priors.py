"""Priors over the flat vector of fitted leaves, from ``priorities`` distributions.

A distribution from ``priorities`` is elementwise or has one event axis and scores
against ``d(value)``. A fit scores one flat vector ``z`` (the raveled fitted
leaves), so the product over leaves lives here: :class:`IndependentPrior` lays
distributions over consecutive blocks of ``z`` and sums their log densities. It
serves every backend through one interface: ``log_prob`` for the gradient and Laplace
backends, ``sample`` for initial particles, and ``icdf`` / ``cdf`` for
nested sampling.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from priorities import AbstractDistribution

__all__ = ["IndependentPrior", "flat_prior"]


class IndependentPrior(eqx.Module):
    """Independent distributions over consecutive blocks of the flat vector ``z``.

    Each distribution covers one block, in order. An elementwise distribution
    covers one entry per element of its batch, so its parameters are scalars (one
    entry) or have shape ``(k,)`` (``k`` entries). A distribution with an event axis
    of length ``d`` covers ``d`` entries and has no batch.

    Attributes:
        distributions: The distributions, in the order their blocks appear in ``z``.
    """

    distributions: tuple[AbstractDistribution, ...]

    def __check_init__(self):
        """Require every distribution to cover a one-dimensional block."""
        for dist in self.distributions:
            shape = dist.support()[0].shape
            if len(shape) > 1 or (dist.event_dim and len(shape) != 1):
                raise ValueError(
                    f"{type(dist).__name__} covers shape {shape}; a block of the "
                    "flat vector is a scalar or one axis."
                )

    def _shapes(self):
        return [dist.support()[0].shape for dist in self.distributions]

    @property
    def ndim(self) -> int:
        """Length of ``z``."""
        return sum(shape[0] if shape else 1 for shape in self._shapes())

    @property
    def has_icdf(self) -> bool:
        """Whether every distribution maps to and from the unit cube."""
        return all(dist.has_icdf for dist in self.distributions)

    def _split(self, vector):
        """Cut ``(..., ndim)`` into one block per distribution."""
        blocks, start = [], 0
        for shape in self._shapes():
            if shape:
                blocks.append(vector[..., start : start + shape[0]])
            else:
                blocks.append(vector[..., start])
            start += shape[0] if shape else 1
        return blocks

    def _join(self, blocks):
        blocks = [
            block if shape else block[..., None]
            for block, shape in zip(blocks, self._shapes(), strict=True)
        ]
        return jnp.concatenate(blocks, axis=-1)

    def log_prob(self, z):
        """Return the summed log density at ``z`` of shape ``(..., ndim)``."""
        total = 0.0
        blocks = zip(self.distributions, self._split(z), self._shapes(), strict=True)
        for dist, block, shape in blocks:
            value = dist.log_prob(block)
            # An elementwise distribution scores each entry of a (k,) block.
            per_entry = len(shape) == 1 and not dist.event_dim
            total = total + (value.sum(-1) if per_entry else value)
        return total

    def sample(self, key, shape=()):
        """Draw vectors of shape ``shape + (ndim,)``."""
        keys = jax.random.split(key, len(self.distributions))
        return self._join(
            [
                dist.sample(k, shape)
                for dist, k in zip(self.distributions, keys, strict=True)
            ]
        )

    def icdf(self, unit):
        """Map a point of the unit cube ``(..., ndim)`` to ``z``."""
        blocks = self._split(unit)
        return self._join(
            [dist.icdf(b) for dist, b in zip(self.distributions, blocks, strict=True)]
        )

    def cdf(self, z):
        """Map ``z`` of shape ``(..., ndim)`` back to the unit cube."""
        blocks = self._split(z)
        return self._join(
            [dist.cdf(b) for dist, b in zip(self.distributions, blocks, strict=True)]
        )


def flat_prior(prior):
    """Wrap a bare distribution as an :class:`IndependentPrior`.

    Anything else passes through. A fit accepts a prior over ``z`` as a single
    distribution (for example the ``MultivariateNormal`` a posterior converts to),
    as an :class:`IndependentPrior`, or as a callable on the recombined model.
    """
    return (
        IndependentPrior((prior,)) if isinstance(prior, AbstractDistribution) else prior
    )
