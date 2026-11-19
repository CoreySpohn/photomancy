"""Hipparcos-Gaia proper-motion anomaly: data container, kernel and term.

Epochs are days from the system's epoch. The anomaly is the Gaia-window slope of the
stellar reflex minus the Hipparcos-Gaia mean motion, in arcsec/day.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve

from photomancy.infer.fit import AbstractTerm
from photomancy.measurements._observables import _stellar_reflex

__all__ = ["ProperMotionAnomalyData", "ProperMotionAnomalyTerm", "loglike_pm_anomaly"]


class ProperMotionAnomalyData(eqx.Module):
    """Hipparcos-Gaia proper-motion-anomaly (PMa) observation.

    The anomaly is a single ``(RA, DEC)`` proper-motion difference (the Gaia
    instantaneous proper motion minus the long-baseline Hipparcos-to-Gaia mean), in
    arcsec/day, with a 2x2 covariance. Epochs are in the orbit's day frame. There is
    no padding (the observable is a fixed 2-vector); the mission windows feed the
    25-epoch path-fit forward.

    Args:
        t_hip: Hipparcos mean epoch (days). Scalar.
        t_gaia: Gaia mean epoch (days). Scalar.
        gaia_window: Gaia mission duration (days) for the proper-motion fit. Scalar.
        pm_anomaly: Observed anomaly ``(pm_ra, pm_dec)`` (arcsec/day). Shape ``(2,)``.
        pm_anomaly_cov: Anomaly covariance (arcsec/day)^2. Shape ``(2, 2)``.
        is_valid: Boolean validity flag. Scalar.
        n_epochs: Epochs sampled per mission window (static). Default 25.
    """

    t_hip: jnp.ndarray
    t_gaia: jnp.ndarray
    gaia_window: jnp.ndarray
    pm_anomaly: jnp.ndarray
    pm_anomaly_cov: jnp.ndarray
    is_valid: jnp.ndarray
    n_epochs: int = eqx.field(static=True, default=25)

    @classmethod
    def zeros(cls, n_epochs=25):
        """Create an all-invalid placeholder (for model tracing)."""
        return cls(
            t_hip=jnp.array(0.0),
            t_gaia=jnp.array(9041.0),  # ~24.75 yr, nonzero baseline
            gaia_window=jnp.array(1023.0),
            pm_anomaly=jnp.zeros(2),
            pm_anomaly_cov=jnp.eye(2),
            is_valid=jnp.array(False),
            n_epochs=n_epochs,
        )


def loglike_pm_anomaly(anom_pred, data):
    """Proper-motion-anomaly log-likelihood (2D Gaussian with full covariance).

    Args:
        anom_pred: Predicted anomaly ``(pm_ra, pm_dec)`` (arcsec/day). Shape ``(2,)``.
        data: A :class:`ProperMotionAnomalyData` instance.

    Returns:
        Scalar log-likelihood (zero when ``data.is_valid`` is False, for the
        placeholder channel during model tracing).
    """
    resid = data.pm_anomaly - anom_pred
    prec = jnp.linalg.inv(data.pm_anomaly_cov)
    _, logdet = jnp.linalg.slogdet(data.pm_anomaly_cov)
    ll = -0.5 * resid @ prec @ resid - 0.5 * logdet - jnp.log(2.0 * jnp.pi)
    return jnp.where(data.is_valid, ll, 0.0)


def _path_proper_motion(ts, ra, dec):
    """Least-squares east/north slopes of an angular path at day epochs."""
    dt = ts - jnp.mean(ts)
    denom = jnp.sum(dt**2)
    pm_ra = jnp.sum(dt * (ra - jnp.mean(ra))) / denom
    pm_dec = jnp.sum(dt * (dec - jnp.mean(dec))) / denom
    return jnp.stack([pm_ra, pm_dec])


def _proper_motion_anomaly(t_hip, t_gaia, window, n_epochs, reflex_path):
    """Gaia window slope minus Hipparcos-Gaia slope of an east/north path.

    reflex_path maps day epochs (N,) to angular offsets (2, N) in arcseconds.
    Both system terms and single-orbit predictions use this mission-window fit.
    """
    ts = jnp.linspace(t_gaia - 0.5 * window, t_gaia + 0.5 * window, n_epochs)
    time_d = jnp.concatenate((ts, jnp.atleast_1d(t_hip), jnp.atleast_1d(t_gaia)))
    path = reflex_path(time_d)
    mu_gaia = _path_proper_motion(ts, *path[:, :n_epochs])
    mu_hg = (path[:, -1] - path[:, -2]) / (t_gaia - t_hip)
    return mu_gaia - mu_hg


class ProperMotionAnomalyTerm(AbstractTerm):
    """Score the Gaia-window reflex slope minus the Hipparcos-Gaia mean motion.

    ``system_path`` names the model path of a system with ``star_reflex(time_d)`` in
    arcsec; the empty path is the model itself.
    ``data.n_epochs`` sets the number of samples across the Gaia window (at
    least two). The prediction is an east/north vector in arcsec/day,
    zero when invalid.
    Data ``t_hip`` and ``t_gaia`` are days from the system's epoch.
    """

    data: ProperMotionAnomalyData
    system_path: str = eqx.field(static=True, default="")
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require enough samples to define a slope."""
        if self.data.n_epochs < 2:
            raise ValueError("n_epochs must be at least two")

    def _score(self, model):
        valid = self.data.is_valid
        data = ProperMotionAnomalyData(
            t_hip=jnp.where(valid, self.data.t_hip, 0.0),
            t_gaia=jnp.where(valid, self.data.t_gaia, 1.0),
            gaia_window=jnp.where(valid, self.data.gaia_window, 1.0),
            pm_anomaly=jnp.where(valid, self.data.pm_anomaly, 0.0),
            pm_anomaly_cov=jnp.where(valid, self.data.pm_anomaly_cov, jnp.eye(2)),
            is_valid=valid,
            n_epochs=self.data.n_epochs,
        )
        system = resolve(model, self.system_path)

        def reflex_path(time_d):
            return _stellar_reflex(system, time_d)

        prediction = jnp.where(
            valid,
            _proper_motion_anomaly(
                data.t_hip, data.t_gaia, data.gaia_window, data.n_epochs, reflex_path
            ),
            0.0,
        )
        return loglike_pm_anomaly(prediction, data), prediction

    def __call__(self, model):
        """Return the scalar log-likelihood."""
        return self._score(model)[0]

    def predict(self, model):
        """Return the proper-motion anomaly, ``(2,)``."""
        return self._score(model)[1]

    def draw(self, key, model) -> Self:
        """Copy with a full-covariance Gaussian proper-motion anomaly."""
        valid = self.data.is_valid
        covariance = jnp.where(valid, self.data.pm_anomaly_cov, jnp.eye(2))
        eps = jax.random.normal(key, (2,), dtype=jnp.float64)
        draw = self.predict(model) + jnp.linalg.cholesky(covariance) @ eps
        data = eqx.tree_at(
            lambda d: d.pm_anomaly,
            self.data,
            jnp.where(valid, draw, self.data.pm_anomaly),
        )
        return eqx.tree_at(lambda t: t.data, self, data)
