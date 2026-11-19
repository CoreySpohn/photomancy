"""Score and predict in one call for the tests that compare both."""

from photomancy.measurements import RadialVelocityTerm


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
