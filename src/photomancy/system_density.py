"""Chart-coordinate densities over systems built by pure functions."""

from collections.abc import Callable

from photomancy.core.model import AbstractLogDensity
from photomancy.terms import AbstractTerm

__all__ = ["SystemLogDensity"]


class SystemLogDensity(AbstractLogDensity):
    """Sum likelihood terms, a system prior, and a chart's log volume.

    ``build(params)`` constructs the system from an unconstrained array PyTree.
    ``prior(system)`` is a scalar log density against the reference measure
    used by ``log_volume(params)``. Terms use the existing ``AbstractTerm``
    interface; an empty tuple gives a prior-only density.

    As in ``TermLogDensity``, callable fields retain any PyTree array leaves;
    ordinary functions are static under Equinox filtered transformations.
    """

    build: Callable
    terms: tuple[AbstractTerm, ...]
    prior: Callable
    log_volume: Callable

    def value_and_aux(self, params):
        """Return ``(value, {"term_values": values, "predictions": predictions})``.

        Both aux entries are tuples in term order (empty for zero terms).
        Prediction PyTrees are unchanged from each term's second output.
        Build once, then evaluate each term, the prior, and the volume once.
        """
        system = self.build(params)
        results = tuple(term(system) for term in self.terms)
        values = tuple(value for value, _ in results)
        predictions = tuple(prediction for _, prediction in results)
        total = self.prior(system) + self.log_volume(params)
        for value in values:
            total = total + value
        return total, {"term_values": values, "predictions": predictions}

    def __call__(self, params):
        """Return the scalar density in the supplied chart coordinates."""
        return self.value_and_aux(params)[0]
