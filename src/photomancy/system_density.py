"""Chart-coordinate densities over systems built by pure functions."""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from priorities import AbstractDistribution

from photomancy.infer.fit import AbstractTerm
from photomancy.infer.model import AbstractLogDensity

__all__ = ["ScalarCoordinate", "SystemLogDensity"]


def _score_term(term, system, keywords):
    """``(value, prediction)`` of a term: a ``Fit`` term scores and predicts apart."""
    if isinstance(term, AbstractTerm):
        return term(system, **keywords), term.predict(system)
    return term(system, **keywords)


class ScalarCoordinate(eqx.Module):
    """Independent scalar(s) in physical units or their natural logarithm.

    The prior is a ``priorities`` distribution or a callable; either gives the log
    density against the physical differential. Nested sampling and prior draws
    need a distribution with an inverse CDF. Supply None when a joint system prior
    already includes this differential (e.g. a population prior per Earth mass).
    """

    prior: AbstractDistribution | Callable | None
    logarithmic: bool = True

    def log_prior(self, value):
        """Elementwise log density of the physical value(s)."""
        if isinstance(self.prior, AbstractDistribution):
            return self.prior.log_prob(value)
        return self.prior(value)

    def physical(self, x):
        """Decode sampler coordinates, elementwise for per-instrument arrays."""
        return jnp.exp(x) if self.logarithmic else x

    def log_volume(self, x):
        """Log determinant to the physical differential."""
        return jnp.sum(x) if self.logarithmic else 0.0


class SystemLogDensity(AbstractLogDensity):
    """Build once and sum a physical prior, scalar priors, volumes and terms.

    A custom data likelihood is an explicit callable in ``terms``. It returns
    a scalar score and an auxiliary prediction (which may be None). Supply
    each measurement once, including any joint or censored report data.

    With named scalar coordinates, params is a mapping containing their raw
    values. Pass decoded scalars as keyword arguments to the system builder
    via body_inputs, and to individual likelihoods via term_inputs (keyed by
    term index). Unlisted terms simply receive the system. The orbit volume
    callable and build both take params as their first argument.
    """

    build: Callable
    terms: tuple[Callable, ...]
    prior: Callable
    orbit_log_volume: Callable
    scalars: dict[str, ScalarCoordinate] = eqx.field(default_factory=dict)
    body_inputs: tuple[str, ...] = ()
    term_inputs: dict[int, tuple[str, ...]] = eqx.field(default_factory=dict)

    def __check_init__(self):
        """Check argument bindings before tracing a sampler."""
        for index in self.term_inputs:
            if not 0 <= index < len(self.terms):
                raise ValueError(f"Unknown term index {index}")
        for names in (self.body_inputs, *self.term_inputs.values()):
            if missing := set(names).difference(self.scalars):
                raise ValueError(f"Unknown scalar coordinates {missing}")

    def log_volume(self, params):
        """Full log Jacobian from sampler coordinates to the physical measure."""
        return self.orbit_log_volume(params) + sum(
            coord.log_volume(params[name]) for name, coord in self.scalars.items()
        )

    def _system_and_scalars(self, params):
        physical = {
            name: coord.physical(params[name]) for name, coord in self.scalars.items()
        }
        system = self.build(
            params, **{name: physical[name] for name in self.body_inputs}
        )
        return system, physical

    def build_system(self, params):
        """Build the system at these sampler coordinates for observable predictions."""
        return self._system_and_scalars(params)[0]

    def _terms(self, system, physical):
        results = tuple(
            _score_term(
                term,
                system,
                {name: physical[name] for name in self.term_inputs.get(i, ())},
            )
            for i, term in enumerate(self.terms)
        )
        values = tuple(value for value, _ in results)
        if any(jnp.ndim(value) != 0 for value in values):
            raise ValueError("Each term must return a scalar log-likelihood.")
        predictions = tuple(prediction for _, prediction in results)
        return values, predictions

    def _logprior(self, params, system, physical):
        scalar_prior = sum(
            jnp.sum(coord.log_prior(physical[name]))
            for name, coord in self.scalars.items()
            if coord.prior is not None
        )
        system_prior = self.prior(system)
        volume = self.log_volume(params)
        # A likelihood can be undefined on an excluded orbit (e.g. zero radius).
        excluded = (
            jnp.isneginf(system_prior)
            | jnp.isneginf(scalar_prior)
            | jnp.isneginf(volume)
        )
        return jnp.where(excluded, -jnp.inf, system_prior + scalar_prior + volume)

    def logprior(self, params):
        """Normalized-prior contribution in sampler coordinates, including volumes."""
        return self._logprior(params, *self._system_and_scalars(params))

    def loglikelihood(self, params):
        """Score data alone for nested sampling or tempering."""
        values, _ = self._terms(*self._system_and_scalars(params))
        return sum(values, jnp.asarray(0.0))

    def value_and_aux(self, params):
        """Return the joint and ordered tuples of term values and predictions."""
        system, physical = self._system_and_scalars(params)
        values, predictions = self._terms(system, physical)
        prior = self._logprior(params, system, physical)
        total = jnp.where(jnp.isneginf(prior), -jnp.inf, prior + sum(values))
        return total, {"term_values": values, "predictions": predictions}

    def __call__(self, params):
        """Return the scalar density in the supplied chart coordinates."""
        return self.value_and_aux(params)[0]
