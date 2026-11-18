"""Terms that score measurements of a planetary system other than images.

Each term holds a data container and calls the system object by method name, so any
object that reports the same observables can be the model. ``select`` maps the fitted
model to that object; the identity default makes a bare system a valid model, and a
fit with no image terms is then an orbit fit.

Data times are days from the selected system's epoch, including the Hipparcos
and Gaia reference times. Absolute dates must be converted on the host first.
"""

from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.spectral import Quadrature, Transmission
from jax.scipy.stats import norm
from orbix.system import System

from photomancy.orbit.data import (
    PhotometryData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
)
from photomancy.orbit.forward import _proper_motion_anomaly
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


def _relative_positions(system, time_d, planet_id):
    """Read east/north offsets (2, T) for each row's planet and epoch."""
    if isinstance(system, System):
        ephemeris = system.solve(time_d)
        pos = jnp.stack(
            ephemeris.offset_arcsec(jnp.arange(1, system.orbits.a_AU.size + 1))
        )
        return pos[:, planet_id, jnp.arange(time_d.size)]
    pos = system.positions(time_d=time_d)
    return pos[:, planet_id, jnp.arange(time_d.size)]


def _band_contrasts(system, quadrature, bandpasses, time_d, band_id, planet_id):
    """Select each row's filter, planet and epoch from a single scene evaluation."""

    def response(nodes_nm):
        return jnp.stack(
            [curve(row) for curve, row in zip(bandpasses, nodes_nm, strict=True)]
        )

    bands = system.band_contrasts(quadrature, time_d, bandpass=response)
    return bands[band_id, planet_id, jnp.arange(time_d.size)]


def _sample_radec(key, term, prediction):
    """Sample correlated east/north errors without changing padded payloads."""
    data = term.data
    valid = data.is_valid
    eps = jax.random.normal(key, (2, data.times.size), dtype=jnp.float64)
    sx = jnp.where(valid, data.ra_err, 1.0)
    sy = jnp.where(valid, data.dec_err, 1.0)
    rho = jnp.where(valid, data.corr, 0.0)
    east = prediction[0] + sx * eps[0]
    north = prediction[1] + sy * (rho * eps[0] + jnp.sqrt(1 - rho**2) * eps[1])
    sampled = eqx.tree_at(
        lambda d: (d.ra, d.dec),
        data,
        (jnp.where(valid, east, data.ra), jnp.where(valid, north, data.dec)),
    )
    return eqx.tree_at(lambda t: t.data, term, sampled)


class RelativeAstrometryTerm(AbstractTerm):
    """Score relative east/north offsets from a system's sky positions.

    Attributes:
        data: Relative astrometry in arcsec, with a planet index per epoch.
        select: Maps the model to an object whose ``positions(time_d)`` returns
            shape ``(2, K, T)`` in arcsec, rows east and north. The identity
            default accepts that object directly. Data ``times`` are days from
            the system's epoch. A bare ``orbix.system.System`` is also accepted,
            using ``solve`` and its public ``offset_arcsec`` observable; planet
            indices then refer to zero-based Jacobi chain order.

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
        prediction = _relative_positions(self.select(model), data.times, data.planet_id)
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        east, north = prediction
        return loglike_relative_astrom(east, north, data), prediction

    def sample(self, key, model) -> Self:
        """Copy with correlated Gaussian positions at the same epochs."""
        return _sample_radec(key, self, self(model)[1])


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

    def sample(self, key, model) -> Self:
        """Copy with correlated Gaussian positions at the same epochs."""
        return _sample_radec(key, self, self(model)[1])


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

    def _jitters(self, jitter_m_per_s):
        jitters = self.jitters if jitter_m_per_s is None else jitter_m_per_s
        if jitter_m_per_s is not None and self.jitters is not None:
            raise ValueError("Choose fixed or sampled RV jitter")
        if jitters is None:
            jitters = jnp.zeros(self.data.n_inst)
        if jnp.shape(jitters) != (self.data.n_inst,):
            raise ValueError("jitters must have shape (data.n_inst,)")
        return jitters

    def __call__(self, model, *, jitter_m_per_s=None):
        """Score RVs with fixed or sampled per-instrument jitter (m/s)."""
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
        value = loglike_rv_marginalized(
            data.rv,
            prediction,
            data.rv_err,
            data.inst_ids,
            data.n_inst,
            self._jitters(jitter_m_per_s),
            valid,
        )
        return value, prediction

    def sample(self, key, model, *, jitter_m_per_s=None) -> Self:
        """Draw Gaussian RV noise with zero instrument offsets.

        The score marginalizes offsets against a flat measure and is therefore
        improper on raw RVs. Zero offsets choose a representative; its residual
        contrasts have exactly the marginalized Gaussian law. Jitter is added
        in quadrature, as in the score.
        """
        data = self.data
        valid = data.is_valid
        ids = jnp.where(valid, data.inst_ids, 0)
        jitter = self._jitters(jitter_m_per_s)
        sigma = jnp.sqrt(jnp.where(valid, data.rv_err, 1.0) ** 2 + jitter[ids] ** 2)
        draw = self(model)[1] + sigma * jax.random.normal(
            key, data.rv.shape, dtype=jnp.float64
        )
        sampled = eqx.tree_at(lambda d: d.rv, data, jnp.where(valid, draw, data.rv))
        return eqx.tree_at(lambda t: t.data, self, sampled)


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

        def reflex_path(time_d):
            ephemeris = system.ephemeris(time_d=time_d)
            return system.star_reflex(time_d=time_d, ephemeris=ephemeris)

        prediction = jnp.where(
            valid,
            _proper_motion_anomaly(
                data.t_hip, data.t_gaia, data.gaia_window, self.n_epochs, reflex_path
            ),
            0.0,
        )
        return loglike_pm_anomaly(prediction, data), prediction

    def sample(self, key, model) -> Self:
        """Copy with a full-covariance Gaussian proper-motion anomaly."""
        valid = self.data.is_valid
        covariance = jnp.where(valid, self.data.pm_anomaly_cov, jnp.eye(2))
        eps = jax.random.normal(key, (2,), dtype=jnp.float64)
        draw = self(model)[1] + jnp.linalg.cholesky(covariance) @ eps
        data = eqx.tree_at(
            lambda d: d.pm_anomaly,
            self.data,
            jnp.where(valid, draw, self.data.pm_anomaly),
        )
        return eqx.tree_at(lambda t: t.data, self, data)


class PhotometryTerm(AbstractTerm):
    """Score linear contrasts using Gaussian detections or Gaussian upper limits.

    Build ``quadrature`` on the host with the system and filter curves. Its
    bands and ``bandpasses`` follow ``data.band_id`` order. The selected
    system returns ``(W, K, T)`` band contrasts.
    Each row selects its own band, planet and epoch. An upper limit contributes
    ``log Phi((limit - prediction) / contrast_err)``. Invalid rows return zero.
    Data ``times`` are days from the system's epoch. Bands share one ephemeris.
    """

    data: PhotometryData
    quadrature: Quadrature
    bandpasses: tuple[Transmission, ...]
    select: Callable = _identity

    def __check_init__(self):
        """Require a bandpass for contrast prediction."""
        if len(self.bandpasses) != len(self.quadrature.bands):
            raise ValueError("PhotometryTerm needs one bandpass per quadrature band")

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
        prediction = _band_contrasts(
            system,
            self.quadrature,
            self.bandpasses,
            data.times,
            data.band_id,
            data.planet_id,
        )
        prediction = jnp.where(valid, prediction, 0.0)
        z = (data.contrast - prediction) / data.contrast_err
        detection = -0.5 * z**2 - jnp.log(data.contrast_err * jnp.sqrt(2.0 * jnp.pi))
        rows = jnp.where(data.is_upper_limit, norm.logcdf(z), detection)
        return jnp.sum(jnp.where(valid, rows, 0.0)), prediction

    def sample(self, key, model) -> Self:
        """Draw Gaussian contrasts, censoring at each stored upper limit.

        An upper-limit row retains its threshold and flag when the draw lies
        below the threshold. Above it, the row becomes a measured detection.
        Rows already measured as detections remain uncensored Gaussian draws.
        """
        data = self.data
        valid = data.is_valid
        sigma = jnp.where(valid, data.contrast_err, 1.0)
        draw = self(model)[1] + sigma * jax.random.normal(
            key, data.contrast.shape, dtype=jnp.float64
        )
        censored = data.is_upper_limit & (draw < data.contrast)
        contrast = jnp.where(censored, data.contrast, draw)
        sampled = eqx.tree_at(
            lambda d: (d.contrast, d.is_upper_limit),
            data,
            (
                jnp.where(valid, contrast, data.contrast),
                jnp.where(valid, censored, data.is_upper_limit),
            ),
        )
        return eqx.tree_at(lambda t: t.data, self, sampled)
