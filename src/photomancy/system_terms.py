"""Terms that score measurements of a planetary system other than images.

Each term holds a data container and calls the system object by method name, so any
object that reports the same observables can be the model. ``select`` maps the fitted
model to that object; the identity default makes a bare system a valid model, and a
fit with no image terms is then an orbit fit.
"""

from collections.abc import Callable

import jax.numpy as jnp

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.terms import AbstractTerm

__all__ = ["RelativeAstrometryTerm"]


def _identity(model):
    return model


class RelativeAstrometryTerm(AbstractTerm):
    """Score relative east/north offsets from a system's sky positions.

    Attributes:
        data: Relative astrometry in arcsec, with a planet index per epoch.
        select: Maps the model to an object whose ``positions(times)`` returns
            shape ``(2, K, T)`` in arcsec, rows east and north. The identity
            default accepts that object directly. Epochs use the same time
            origin as the system.

    Predictions have shape ``(2, T)``; invalid rows are returned as zeros.
    """

    data: RelativeAstromData
    select: Callable = _identity

    def __call__(self, model):
        """Return the log-likelihood and each row's planet at its own epoch."""
        valid = self.data.is_valid
        # Sanitize before propagation and division: masking a NaN result alone
        # does not prevent NaN derivatives from flowing through its operands.
        data = RelativeAstromData(
            times=jnp.where(valid, self.data.times, 0.0),
            ra=jnp.where(valid, self.data.ra, 0.0),
            dec=jnp.where(valid, self.data.dec, 0.0),
            ra_err=jnp.where(valid, self.data.ra_err, 1.0),
            dec_err=jnp.where(valid, self.data.dec_err, 1.0),
            corr=jnp.where(valid, self.data.corr, 0.0),
            planet_id=jnp.where(valid, self.data.planet_id, 0),
            is_valid=valid,
        )
        pos = self.select(model).positions(data.times)
        prediction = pos[:, data.planet_id, jnp.arange(data.times.shape[0])]
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        east, north = prediction
        return loglike_relative_astrom(east, north, data), prediction
