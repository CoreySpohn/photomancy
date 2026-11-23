"""Score and predict in one call, and a multi-exposure record for forward terms."""

from collections.abc import Callable
from typing import Any

import equinox as eqx

from photomancy.measurements import RadialVelocityTerm


class ExposureRecord(eqx.Module):
    """A model of several exposures: one shared instrument, a block per exposure.

    Paths are ``"instrument.<leaf>"``, ``"exposures[i].<leaf>"`` and
    ``"scene.<path>"``, so a fit frees seams, per-exposure nuisances and sources
    of one record.
    """

    instrument: Any
    exposures: tuple
    scene: Any


class ExposureRender(eqx.Module):
    """Forward ``model -> render(instrument, exposures[index], scene)``.

    The forward of a ``ForwardTerm`` over an :class:`ExposureRecord`; ``render`` is
    the injected optics, ideally an ``eqx.Module`` so its arrays stay traced.
    """

    render: Callable
    index: int = eqx.field(static=True, default=0)

    def __call__(self, model):
        """Render exposure ``index`` of ``model``."""
        return self.render(model.instrument, model.exposures[self.index], model.scene)


def both(term, model, **keywords):
    """Return ``(log_likelihood, prediction)`` of a term at ``model``."""
    return term(model, **keywords), term.predict(model)


def sampled_jitter_rv(data):
    """An RV likelihood whose jitter is the sampled ``jitter_m_per_s`` coordinate.

    A caller-defined term for a density that binds the jitter by keyword: it builds
    the ``RadialVelocityTerm`` with that jitter and returns ``(value, prediction)``.
    """

    def term(system, *, jitter_m_per_s):
        rv = RadialVelocityTerm(data, jitters=jitter_m_per_s)
        return rv(system), rv.predict(system)

    return term
