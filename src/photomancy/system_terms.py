"""Terms that score measurements of a planetary system other than images.

Each term holds a data container and calls the system object by method name, so any
object that reports the same observables can be the model. ``select`` maps the fitted
model to that object; the identity default makes a bare system a valid model, and a
fit with no image terms is then an orbit fit.

Data times are days from the selected system's epoch, including the Hipparcos
and Gaia reference times. Absolute dates must be converted on the host first.
"""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.stats import norm

from photomancy.orbit.data import (
    PhotometryData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
)
from photomancy.orbit.forward import _path_proper_motion
from photomancy.orbit.likelihoods import (
    loglike_pm_anomaly,
    loglike_relative_astrom,
    loglike_rv_marginalized,
    loglike_stellar_astrom,
)
from photomancy.terms import AbstractTerm

__all__ = [
    "PhotometryTerm",
    "ProperMotionAnomalyTerm",
    "RadialVelocityTerm",
    "RelativeAstrometryTerm",
    "StellarAstrometryTerm",
]


def _identity(model):
    return model


class RelativeAstrometryTerm(AbstractTerm):
    """Score relative east/north offsets from a system's sky positions.

    Attributes:
        data: Relative astrometry in arcsec, with a planet index per epoch.
        select: Maps the model to an object whose ``positions(time_d)`` returns
            shape ``(2, K, T)`` in arcsec, rows east and north. The identity
            default accepts that object directly. Data ``times`` are days from
            the system's epoch.

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
        pos = self.select(model).positions(time_d=data.times)
        prediction = pos[:, data.planet_id, jnp.arange(data.times.shape[0])]
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        east, north = prediction
        return loglike_relative_astrom(east, north, data), prediction


class StellarAstrometryTerm(AbstractTerm):
    """Score the stellar reflex path in arcsec with correlated Gaussian errors.

    ``select`` maps the model to a system whose ``star_reflex(time_d)`` returns
    east/north offsets of shape ``(2, T)``. The identity default accepts a system.
    Data ``times`` are days from the system's epoch.
    Invalid rows are sanitized before prediction and returned as zeros.
    """

    data: StellarAstromData
    select: Callable = _identity

    def __call__(self, model):
        """Return the log-likelihood and stellar east/north path, ``(2, T)``."""
        valid = self.data.is_valid
        data = StellarAstromData(
            times=jnp.where(valid, self.data.times, 0.0),
            ra=jnp.where(valid, self.data.ra, 0.0),
            dec=jnp.where(valid, self.data.dec, 0.0),
            ra_err=jnp.where(valid, self.data.ra_err, 1.0),
            dec_err=jnp.where(valid, self.data.dec_err, 1.0),
            corr=jnp.where(valid, self.data.corr, 0.0),
            is_valid=valid,
        )
        prediction = self.select(model).star_reflex(time_d=data.times)
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        return loglike_stellar_astrom(*prediction, data), prediction


class RadialVelocityTerm(AbstractTerm):
    """Score stellar RVs in m/s, marginalizing each instrument's constant offset.

    ``select`` maps the model to a system with ``star_radial_velocity(time_d)``
    in m/s, positive receding. ``jitters`` has shape ``(data.n_inst,)`` in m/s;
    None uses zeros. Predictions have shape ``(T,)`` with invalid rows zeroed.
    Data ``times`` are days from the system's epoch.
    """

    data: RVData
    select: Callable = _identity
    jitters: jnp.ndarray | None = None

    def __check_init__(self):
        """Require one jitter per instrument when supplied."""
        if self.jitters is not None and jnp.shape(self.jitters) != (self.data.n_inst,):
            raise ValueError("jitters must have shape (data.n_inst,)")

    def __call__(self, model):
        """Return the marginalized log-likelihood and stellar RVs, ``(T,)``."""
        valid = self.data.is_valid
        data = RVData(
            times=jnp.where(valid, self.data.times, 0.0),
            rv=jnp.where(valid, self.data.rv, 0.0),
            rv_err=jnp.where(valid, self.data.rv_err, 1.0),
            inst_ids=jnp.where(valid, self.data.inst_ids, 0),
            is_valid=valid,
            n_inst=self.data.n_inst,
        )
        prediction = self.select(model).star_radial_velocity(time_d=data.times)
        prediction = jnp.where(valid, prediction, 0.0)
        jitters = jnp.zeros(data.n_inst) if self.jitters is None else self.jitters
        value = loglike_rv_marginalized(
            data.rv,
            prediction,
            data.rv_err,
            data.inst_ids,
            data.n_inst,
            jitters,
            valid,
        )
        return value, prediction


class ProperMotionAnomalyTerm(AbstractTerm):
    """Score the Gaia-window reflex slope minus the Hipparcos-Gaia mean motion.

    ``select`` maps the model to a system with ``star_reflex(time_d)`` in arcsec.
    ``n_epochs`` is the static number of samples across the Gaia window (at
    least two), independent of the data container's sampling metadata. The
    prediction is an east/north vector in arcsec/day, zero when invalid.
    Data ``t_hip`` and ``t_gaia`` are days from the system's epoch.
    """

    data: PMAnomalyData
    select: Callable = _identity
    n_epochs: int = eqx.field(static=True, default=25)

    def __check_init__(self):
        """Require enough samples to define a slope."""
        if self.n_epochs < 2:
            raise ValueError("n_epochs must be at least two")

    def __call__(self, model):
        """Return the log-likelihood and proper-motion anomaly, ``(2,)``."""
        valid = self.data.is_valid
        data = PMAnomalyData(
            t_hip=jnp.where(valid, self.data.t_hip, 0.0),
            t_gaia=jnp.where(valid, self.data.t_gaia, 1.0),
            gaia_window=jnp.where(valid, self.data.gaia_window, 1.0),
            pm_anomaly=jnp.where(valid, self.data.pm_anomaly, 0.0),
            pm_anomaly_cov=jnp.where(valid, self.data.pm_anomaly_cov, jnp.eye(2)),
            is_valid=valid,
            n_epochs=self.data.n_epochs,
        )
        system = self.select(model)
        ts = jnp.linspace(
            data.t_gaia - 0.5 * data.gaia_window,
            data.t_gaia + 0.5 * data.gaia_window,
            self.n_epochs,
        )
        time_d = jnp.concatenate(
            (ts, jnp.atleast_1d(data.t_hip), jnp.atleast_1d(data.t_gaia))
        )
        ephemeris = system.ephemeris(time_d=time_d)
        path = system.star_reflex(time_d=time_d, ephemeris=ephemeris)
        mu_gaia = _path_proper_motion(ts, *path[:, : self.n_epochs])
        p_hip, p_gaia = path[:, -2], path[:, -1]
        mu_hg = (p_gaia - p_hip) / (data.t_gaia - data.t_hip)
        prediction = jnp.where(valid, mu_gaia - mu_hg, 0.0)
        return loglike_pm_anomaly(prediction, data), prediction


class PhotometryTerm(AbstractTerm):
    """Score linear contrasts using Gaussian detections or Gaussian upper limits.

    ``bandpasses`` is a tuple in ``data.band_id`` order. ``select`` maps the
    model to a system with ``band_contrasts(bandpass, time_d)`` of shape ``(K, T)``.
    Each row selects its own band, planet and epoch. An upper limit contributes
    ``log Phi((limit - prediction) / contrast_err)``. Invalid rows return zero.
    Data ``times`` are days from the system's epoch. Bands share one ephemeris.
    """

    data: PhotometryData
    bandpasses: tuple
    select: Callable = _identity

    def __check_init__(self):
        """Require a bandpass for contrast prediction."""
        if not self.bandpasses:
            raise ValueError("PhotometryTerm needs at least one bandpass")

    def __call__(self, model):
        """Return the log-likelihood and each row's linear contrast, ``(T,)``."""
        valid = self.data.is_valid
        data = PhotometryData(
            times=jnp.where(valid, self.data.times, 0.0),
            contrast=jnp.where(valid, self.data.contrast, 0.0),
            contrast_err=jnp.where(valid, self.data.contrast_err, 1.0),
            band_id=jnp.where(valid, self.data.band_id, 0),
            planet_id=jnp.where(valid, self.data.planet_id, 0),
            is_upper_limit=jnp.where(valid, self.data.is_upper_limit, False),
            is_valid=valid,
        )
        system = self.select(model)
        ephemeris = system.ephemeris(time_d=data.times)
        bands = jnp.stack(
            [
                system.band_contrasts(b, time_d=data.times, ephemeris=ephemeris)
                for b in self.bandpasses
            ]
        )
        prediction = bands[data.band_id, data.planet_id, jnp.arange(data.times.size)]
        prediction = jnp.where(valid, prediction, 0.0)
        z = (data.contrast - prediction) / data.contrast_err
        detection = -0.5 * z**2 - jnp.log(data.contrast_err * jnp.sqrt(2.0 * jnp.pi))
        rows = jnp.where(data.is_upper_limit, norm.logcdf(z), detection)
        return jnp.sum(jnp.where(valid, rows, 0.0)), prediction
