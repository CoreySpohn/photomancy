"""Score and predict in one call for the tests that compare both."""

import jax.numpy as jnp

from photomancy.measurements import DMagTerm, ImagingData, RadialVelocityTerm


def both(term, model, **keywords):
    """Return ``(log_likelihood, prediction)`` of a term at ``model``."""
    return term(model, **keywords), term.predict(model)


def null_dmag_term(null):
    """A DMagTerm over the epochs of a ``NullData`` with every epoch a null."""
    return DMagTerm(
        ImagingData(
            epochs=null.epochs,
            sep_grid=null.sep_grid,
            dmag0_grid=null.dmag0_grid,
            is_detected=jnp.zeros_like(null.is_valid),
            dmag_obs=jnp.zeros_like(null.epochs),
            dmag_err=jnp.ones_like(null.epochs),
            is_valid=null.is_valid,
            snr_thresh=null.snr_thresh,
        )
    )


def sampled_jitter_rv(data):
    """An RV likelihood whose jitter is the sampled ``jitter_m_per_s`` coordinate.

    A caller-defined term for a density that binds the jitter by keyword: it builds
    the ``RadialVelocityTerm`` with that jitter and returns ``(value, prediction)``.
    """

    def term(system, *, jitter_m_per_s):
        rv = RadialVelocityTerm(data, jitters=jitter_m_per_s)
        return rv(system), rv.predict(system)

    return term
