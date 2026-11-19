"""Astrometry of a planetary system: relative and stellar-reflex positions on the sky.

Data containers, the bivariate-Gaussian kernel, the two terms, and the linear form of
the relative-astrometry term in the four sky constants of an orbit (one planet row).
Positions are east/north offsets in arcsec; data times are days from the system's epoch.

A term reads its system from the model at ``system_path`` (the empty path is the model
itself). The system is any object whose ``positions(time_d)`` (relative) or
``star_reflex(time_d)`` (stellar) returns east/north arcsec, such as a scene, or a bare
orbix ``System``. For several companions orbix propagates the supplied Jacobi chain:
each orbit refers to the interior center of mass, so the element-based model that sums
independent star-companion orbits gives different predictions.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve
from orbix.system import System, sky_design

from photomancy.infer.fit import AbstractLinearTerm, AbstractTerm, LinearForm
from photomancy.measurements._observables import (
    _relative_positions,
    _stellar_reflex,
)
from photomancy.measurements._padding import MAX_REL_ASTROM, _pad_1d, _valid_mask

__all__ = [
    "AstrometryData",
    "RelativeAstrometryTerm",
    "StellarAstrometryTerm",
    "loglike_relative_astrom",
    "loglike_stellar_astrom",
]


class AstrometryData(eqx.Module):
    """East/north astrometry in arcsec: a planet relative to its star, or the star.

    The relative term reads each row as the offset of planet ``planet_id`` from the
    host star; the stellar term reads each row as the reflex offset of the star about
    the system barycenter (one body, so ``planet_id`` is not read).

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        ra: East offset Delta alpha* = Delta RA cos Dec (arcsec).
            Shape ``(N,)``.
        dec: North offset Delta Dec (arcsec). Shape ``(N,)``.
        ra_err: RA uncertainty (arcsec). Shape ``(N,)``.
        dec_err: DEC uncertainty (arcsec). Shape ``(N,)``.
        corr: RA/DEC correlation coefficient. Shape ``(N,)``.
        planet_id: Planet index per observation (for multi-planet systems).
            Shape ``(N,)``.
        is_valid: Boolean validity mask. Shape ``(N,)``.
    """

    times: jnp.ndarray
    ra: jnp.ndarray
    dec: jnp.ndarray
    ra_err: jnp.ndarray
    dec_err: jnp.ndarray
    corr: jnp.ndarray
    planet_id: jnp.ndarray
    is_valid: jnp.ndarray

    @classmethod
    def from_sep_pa(
        cls, *, times, sep, sep_err, pa_deg, pa_err_deg, planet_id=None, max_n=None
    ):
        """Convert separation and position angle to east/north astrometry.

        Separation and its error are in arcsec. Position angle and its error
        are in degrees, east of north. Independent separation and angle errors
        are propagated to a first-order Cartesian covariance. Inputs have one
        entry per epoch; ``planet_id`` defaults to zero for every row.
        ``max_n=None`` keeps all rows valid; otherwise pad to ``max_n``.
        """
        times = jnp.asarray(times)
        sep, sep_err = jnp.asarray(sep), jnp.asarray(sep_err)
        pa = jnp.deg2rad(jnp.asarray(pa_deg))
        pa_err = jnp.deg2rad(jnp.asarray(pa_err_deg))
        sin_pa, cos_pa = jnp.sin(pa), jnp.cos(pa)
        sep_var, tangential_var = sep_err**2, (sep * pa_err) ** 2
        ra_err = jnp.sqrt(sin_pa**2 * sep_var + cos_pa**2 * tangential_var)
        dec_err = jnp.sqrt(cos_pa**2 * sep_var + sin_pa**2 * tangential_var)
        cov = sin_pa * cos_pa * (sep_var - tangential_var)
        if planet_id is None:
            planet_id = jnp.zeros(times.shape, dtype=int)
        values = dict(
            times=times,
            ra=sep * sin_pa,
            dec=sep * cos_pa,
            ra_err=ra_err,
            dec_err=dec_err,
            corr=cov / (ra_err * dec_err),
            planet_id=jnp.asarray(planet_id, dtype=int),
        )
        if max_n is not None:
            return cls.pad(**values, max_n=max_n)
        return cls(**values, is_valid=jnp.ones(times.shape, dtype=bool))

    @classmethod
    def pad(
        cls,
        *,
        times,
        ra,
        dec,
        ra_err,
        dec_err,
        corr,
        planet_id=None,
        max_n=MAX_REL_ASTROM,
    ):
        """Pad to ``max_n`` with an ``is_valid`` mask; ``planet_id`` defaults to 0."""
        n = jnp.asarray(times).shape[0]
        if planet_id is None:
            planet_id = jnp.zeros(n, dtype=int)
        return cls(
            times=_pad_1d(times, max_n),
            ra=_pad_1d(ra, max_n),
            dec=_pad_1d(dec, max_n),
            ra_err=_pad_1d(ra_err, max_n, fill=1.0),
            dec_err=_pad_1d(dec_err, max_n, fill=1.0),
            corr=_pad_1d(corr, max_n),
            planet_id=_pad_1d(planet_id, max_n).astype(int),
            is_valid=_valid_mask(n, max_n),
        )

    @classmethod
    def zeros(cls, max_n=MAX_REL_ASTROM):
        """Create an all-invalid placeholder (for model tracing)."""
        return cls(
            times=jnp.zeros(max_n),
            ra=jnp.zeros(max_n),
            dec=jnp.zeros(max_n),
            ra_err=jnp.ones(max_n),
            dec_err=jnp.ones(max_n),
            corr=jnp.zeros(max_n),
            planet_id=jnp.zeros(max_n, dtype=int),
            is_valid=jnp.zeros(max_n, dtype=bool),
        )


def _radec_bivariate_loglike(ra_pred, dec_pred, data):
    """Bivariate-Gaussian (with correlation) log-likelihood over RA/DEC offsets.

    Shared by relative and stellar astrometry: both compare predicted ``(ra, dec)``
    offsets to measured ones with per-epoch errors ``ra_err``/``dec_err`` and
    correlation ``corr``, masked by ``is_valid``.
    """
    mask = data.is_valid  # boolean

    dx = data.ra - ra_pred
    dy = data.dec - dec_pred
    rho = data.corr
    sx = data.ra_err
    sy = data.dec_err

    # Bivariate Gaussian quadratic form
    one_minus_rho2 = jnp.clip(1.0 - rho**2, 1e-30)
    z = (
        dx**2 / sx**2 + dy**2 / sy**2 - 2.0 * rho * dx * dy / (sx * sy)
    ) / one_minus_rho2

    # Log normalization
    log_norm = jnp.log(sx * sy * jnp.sqrt(one_minus_rho2)) + jnp.log(2.0 * jnp.pi)

    # Gradient-safe masking: jnp.where prevents NaN grad leakage
    z_safe = jnp.where(mask, z, 0.0)
    log_norm_safe = jnp.where(mask, log_norm, 0.0)
    return -0.5 * jnp.sum(z_safe) - jnp.sum(log_norm_safe)


def loglike_relative_astrom(ra_pred, dec_pred, data):
    """Relative astrometry log-likelihood (bivariate Gaussian with correlation).

    Args:
        ra_pred: Predicted RA offsets (arcsec). Shape ``(N,)``.
        dec_pred: Predicted DEC offsets (arcsec). Shape ``(N,)``.
        data: A :class:`AstrometryData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    return _radec_bivariate_loglike(ra_pred, dec_pred, data)


def loglike_stellar_astrom(ra_pred, dec_pred, data):
    """Stellar-reflex astrometry log-likelihood (bivariate Gaussian).

    Identical bivariate-Gaussian form as :func:`loglike_relative_astrom`, but the
    predictions and data are the star's reflex offsets rather than the planet's
    relative position.

    Args:
        ra_pred: Predicted stellar RA offsets (arcsec). Shape ``(N,)``.
        dec_pred: Predicted stellar DEC offsets (arcsec). Shape ``(N,)``.
        data: A :class:`AstrometryData` instance.

    Returns:
        Scalar log-likelihood value.
    """
    return _radec_bivariate_loglike(ra_pred, dec_pred, data)


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


def _dynamics(system):
    """The orbix ``System`` of a scene, or the system itself."""
    return system if isinstance(system, System) else system.dynamics


def _record_form(design, data):
    """Assemble a :class:`LinearForm` from an arcsec-per-AU design and the records."""
    valid = data.is_valid
    target = jnp.where(valid[:, None], jnp.stack([data.ra, data.dec], -1), 0.0)
    sx = jnp.where(valid, data.ra_err, 1.0)
    sy = jnp.where(valid, data.dec_err, 1.0)
    rho = jnp.where(valid, data.corr, 0.0)
    scale = 1.0 / (1.0 - rho**2)
    off = -scale * rho / (sx * sy)
    precision = jnp.stack(
        [jnp.stack([scale / sx**2, off], -1), jnp.stack([off, scale / sy**2], -1)], -2
    )
    return LinearForm(
        design=jnp.where(valid[:, None, None], design, 0.0),
        target=target,
        precision=jnp.where(valid[:, None, None], precision, 0.0),
    )


class RelativeAstrometryTerm(AbstractLinearTerm):
    """Score relative east/north offsets from a system's sky positions.

    Attributes:
        data: Relative astrometry in arcsec, with a planet index per epoch.
        system_path: Model path of the object whose ``positions(time_d)`` returns
            shape ``(2, K, T)`` in arcsec, rows east and north. The empty path is
            the model itself. Data ``times`` are days from the system's epoch. A bare
            ``orbix.system.System`` is also accepted, using ``solve`` and its public
            ``offset_arcsec`` observable; planet indices then refer to zero-based
            Jacobi chain order.

    Predictions have shape ``(2, T)``; invalid rows are returned as zeros.
    """

    data: AstrometryData
    system_path: str = eqx.field(static=True, default="")
    reads: ClassVar[tuple[str, ...]] = ()

    def _score(self, model):
        valid = self.data.is_valid
        # Sanitize before propagation and division: masking a NaN result alone
        # does not prevent NaN derivatives from flowing through its operands.
        data = AstrometryData(
            times=jnp.where(valid, self.data.times, 0.0),
            ra=jnp.where(valid, self.data.ra, 0.0),
            dec=jnp.where(valid, self.data.dec, 0.0),
            ra_err=jnp.where(valid, self.data.ra_err, 1.0),
            dec_err=jnp.where(valid, self.data.dec_err, 1.0),
            corr=jnp.where(valid, self.data.corr, 0.0),
            planet_id=jnp.where(valid, self.data.planet_id, 0),
            is_valid=valid,
        )
        system = resolve(model, self.system_path)
        prediction = _relative_positions(system, data.times, data.planet_id)
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        east, north = prediction
        return loglike_relative_astrom(east, north, data), prediction

    def __call__(self, model):
        """Return the scalar log-likelihood."""
        return self._score(model)[0]

    def predict(self, model):
        """Return each row's planet offset at its own epoch, ``(2, T)``."""
        return self._score(model)[1]

    def draw(self, key, model) -> Self:
        """Copy with correlated Gaussian positions at the same epochs."""
        return _sample_radec(key, self, self.predict(model))

    def linear_form(self, model, row=0) -> LinearForm:
        """The term as a Gaussian in the sky constants of orbit ``row``.

        Only ``n``, ``ex``, ``ey`` and the epoch of the row's orbit in ``model`` enter
        the design, so the form is exact for every value of the four constants at that
        shape, provided the records depend on the constants only through that orbit.
        That holds for a system with a single planet row, and the form refuses a system
        with more: the other rows' offsets, masses and mean motions move with the
        constants under a block that writes the star mass (``ThieleInnesBlock``). It
        also refuses a record of another planet, which one form cannot carry. The
        design is in arcsec per AU, the target and precision in the units of the data.

        Args:
            model: The model holding the system at ``system_path``.
            row: Zero-based Jacobi chain index of the planet.
        """
        dynamics = _dynamics(resolve(model, self.system_path))
        shape = jax.tree.map(lambda a: a[row], dynamics.orbits)
        valid = self.data.is_valid
        times = jnp.where(valid, self.data.times, shape.epoch_d)
        design = sky_design(shape, times) / dynamics.frame.dist_pc
        design = eqx.error_if(
            design,
            jnp.asarray(dynamics.orbits.a_AU.shape[-1] > 1),
            "linear_form: the system has more than one planet row, so the other rows "
            "move with the constants",
        )
        design = eqx.error_if(
            design,
            jnp.any(valid & (self.data.planet_id != row)),
            "linear_form: a valid record belongs to another planet",
        )
        return _record_form(design, self.data)


class StellarAstrometryTerm(AbstractTerm):
    """Score the stellar reflex path in arcsec with correlated Gaussian errors.

    ``system_path`` names the model path of a system whose ``star_reflex(time_d)``
    returns east/north offsets of shape ``(2, T)``; the empty path is the model itself.
    Data ``times`` are days from the system's epoch.
    Invalid rows are sanitized before prediction and returned as zeros.
    """

    data: AstrometryData
    system_path: str = eqx.field(static=True, default="")
    reads: ClassVar[tuple[str, ...]] = ()

    def _score(self, model):
        valid = self.data.is_valid
        data = AstrometryData(
            times=jnp.where(valid, self.data.times, 0.0),
            ra=jnp.where(valid, self.data.ra, 0.0),
            dec=jnp.where(valid, self.data.dec, 0.0),
            ra_err=jnp.where(valid, self.data.ra_err, 1.0),
            dec_err=jnp.where(valid, self.data.dec_err, 1.0),
            corr=jnp.where(valid, self.data.corr, 0.0),
            planet_id=self.data.planet_id,
            is_valid=valid,
        )
        system = resolve(model, self.system_path)
        prediction = _stellar_reflex(system, data.times)
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        return loglike_stellar_astrom(*prediction, data), prediction

    def __call__(self, model):
        """Return the scalar log-likelihood."""
        return self._score(model)[0]

    def predict(self, model):
        """Return the stellar east/north path, ``(2, T)``."""
        return self._score(model)[1]

    def draw(self, key, model) -> Self:
        """Copy with correlated Gaussian positions at the same epochs."""
        return _sample_radec(key, self, self.predict(model))
