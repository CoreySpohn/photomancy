"""Score and predict in one call for the tests that compare both."""

import jax.numpy as jnp

from photomancy.measurements import DMagTerm, ImagingData


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
