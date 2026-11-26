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
from hwoutils.precision import require_float64
from orbix.system import System, sky_design

from photomancy.infer._precision import _as_wide
from photomancy.infer.fit import (
    AbstractLinearGaussianTerm,
    AbstractTerm,
    LinearGaussian,
)
from photomancy.measurements._observables import (
    _check_planets,
    _relative_positions,
    _sky_positions,
    _stellar_reflex,
)
from photomancy.measurements._padding import _pad_1d, _valid_mask

__all__ = [
    "AstrometryData",
    "RelativeAstrometryTerm",
    "StellarAstrometryTerm",
    "loglike_relative_astrom",
    "loglike_stellar_astrom",
]


class AstrometryData(eqx.Module):
    """East/north astrometry in arcsec: a planet relative to its star, or the star.

    The relative term reads each row as the offset of planet ``planet`` from the
    host star; the stellar term reads each row as the reflex offset of the star about
    the system barycenter (one body, so ``planet`` is not read).

    Attributes:
        time_d: Observation epochs (days). Shape ``(N,)``.
        east_arcsec: East offset Delta alpha* = Delta RA cos Dec (arcsec).
            Shape ``(N,)``.
        north_arcsec: North offset Delta Dec (arcsec). Shape ``(N,)``.
        east_err_arcsec: RA uncertainty (arcsec). Shape ``(N,)``.
        north_err_arcsec: DEC uncertainty (arcsec). Shape ``(N,)``.
        corr: RA/DEC correlation coefficient. Shape ``(N,)``.
        planet: Planet index per observation (for multi-planet systems).
            Shape ``(N,)``.
        valid: Boolean validity mask. Shape ``(N,)``.
    """

    time_d: jnp.ndarray
    east_arcsec: jnp.ndarray
    north_arcsec: jnp.ndarray
    east_err_arcsec: jnp.ndarray
    north_err_arcsec: jnp.ndarray
    corr: jnp.ndarray
    planet: jnp.ndarray
    valid: jnp.ndarray

    @classmethod
    def from_separation_pa(
        cls,
        *,
        time_d,
        separation_arcsec,
        separation_err_arcsec,
        pa_deg,
        pa_err_deg,
        planet=None,
        max_n=None,
    ):
        """Convert separation and position angle to east/north astrometry.

        Separation and its error are in arcsec. Position angle and its error
        are in degrees, east of north. Independent separation and angle errors
        are propagated to a first-order Cartesian covariance. Inputs have one
        entry per epoch; ``planet`` defaults to zero for every row.
        ``max_n=None`` keeps all rows valid; otherwise pad to ``max_n``.

        Args:
            time_d: Observation epochs in days from the system's epoch, one per row.
            separation_arcsec: Separation of the planet from its star (arcsec).
            separation_err_arcsec: Separation uncertainty (arcsec).
            pa_deg: Position angle, east of north (degrees).
            pa_err_deg: Position angle uncertainty (degrees).
            planet: Zero-based planet index per row. Defaults to zero for every row.
            max_n: Row capacity to pad to. None keeps the rows as given, all valid.

        Returns:
            East/north astrometry with the first-order propagated errors and
            correlation.
        """
        time_d = jnp.asarray(time_d)
        sep, sep_err = (
            jnp.asarray(separation_arcsec),
            jnp.asarray(separation_err_arcsec),
        )
        pa = jnp.deg2rad(jnp.asarray(pa_deg))
        pa_err = jnp.deg2rad(jnp.asarray(pa_err_deg))
        sin_pa, cos_pa = jnp.sin(pa), jnp.cos(pa)
        sep_var, tangential_var = sep_err**2, (sep * pa_err) ** 2
        east_err_arcsec = jnp.sqrt(sin_pa**2 * sep_var + cos_pa**2 * tangential_var)
        north_err_arcsec = jnp.sqrt(cos_pa**2 * sep_var + sin_pa**2 * tangential_var)
        cov = sin_pa * cos_pa * (sep_var - tangential_var)
        if planet is None:
            planet = jnp.zeros(time_d.shape, dtype=int)
        values = dict(
            time_d=time_d,
            east_arcsec=sep * sin_pa,
            north_arcsec=sep * cos_pa,
            east_err_arcsec=east_err_arcsec,
            north_err_arcsec=north_err_arcsec,
            corr=cov / (east_err_arcsec * north_err_arcsec),
            planet=jnp.asarray(planet, dtype=int),
        )
        if max_n is not None:
            return cls.pad(**values, max_n=max_n)
        return cls(**values, valid=jnp.ones(time_d.shape, dtype=bool))

    @classmethod
    def pad(
        cls,
        *,
        time_d,
        east_arcsec,
        north_arcsec,
        east_err_arcsec,
        north_err_arcsec,
        corr,
        planet=None,
        max_n,
    ):
        """Pad to ``max_n`` with an ``valid`` mask; ``planet`` defaults to 0.

        Args:
            time_d: Observation epochs in days from the system's epoch. Shape ``(N,)``.
            east_arcsec: East offsets (arcsec). Shape ``(N,)``.
            north_arcsec: North offsets (arcsec). Shape ``(N,)``.
            east_err_arcsec: East offset uncertainties (arcsec). Shape ``(N,)``. Padded
                rows hold 1.0.
            north_err_arcsec: North offset uncertainties (arcsec). Shape ``(N,)``.
                Padded
                rows hold 1.0.
            corr: East/north correlation coefficients. Shape ``(N,)``.
            planet: Zero-based planet index per row. Shape ``(N,)``. Defaults to zero.
            max_n: Row capacity of the result.

        Returns:
            The data padded to shape ``(max_n,)``, with ``valid`` true for the first N
            rows.
        """
        n = jnp.asarray(time_d).shape[0]
        if planet is None:
            planet = jnp.zeros(n, dtype=int)
        return cls(
            time_d=_pad_1d(time_d, max_n),
            east_arcsec=_pad_1d(east_arcsec, max_n),
            north_arcsec=_pad_1d(north_arcsec, max_n),
            east_err_arcsec=_pad_1d(east_err_arcsec, max_n, fill=1.0),
            north_err_arcsec=_pad_1d(north_err_arcsec, max_n, fill=1.0),
            corr=_pad_1d(corr, max_n),
            planet=_pad_1d(planet, max_n).astype(int),
            valid=_valid_mask(n, max_n),
        )

    @classmethod
    def zeros(cls, max_n):
        """Create an all-invalid placeholder (for model tracing).

        Args:
            max_n: Row capacity of the placeholder.

        Returns:
            Data of shape ``(max_n,)`` with every row invalid and unit errors.
        """
        return cls(
            time_d=jnp.zeros(max_n),
            east_arcsec=jnp.zeros(max_n),
            north_arcsec=jnp.zeros(max_n),
            east_err_arcsec=jnp.ones(max_n),
            north_err_arcsec=jnp.ones(max_n),
            corr=jnp.zeros(max_n),
            planet=jnp.zeros(max_n, dtype=int),
            valid=jnp.zeros(max_n, dtype=bool),
        )


def _radec_bivariate_loglike(ra_pred, dec_pred, data):
    """Bivariate-Gaussian (with correlation) log-likelihood over RA/DEC offsets.

    Shared by relative and stellar astrometry: both compare the predicted east and
    north offsets to measured ones with per-epoch errors ``east_err_arcsec`` and
    ``north_err_arcsec`` and correlation ``corr``, masked by ``valid``.
    """
    require_float64("Astrometry log likelihood")
    mask = data.valid  # boolean

    dx = _as_wide(data.east_arcsec) - _as_wide(ra_pred)
    dy = _as_wide(data.north_arcsec) - _as_wide(dec_pred)
    rho = _as_wide(data.corr)
    sx = _as_wide(data.east_err_arcsec)
    sy = _as_wide(data.north_err_arcsec)

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

    Raises:
        PrecisionError: If float64 is unavailable.
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

    Raises:
        PrecisionError: If float64 is unavailable.
    """
    return _radec_bivariate_loglike(ra_pred, dec_pred, data)


def _sample_radec(key, term, prediction):
    """Sample correlated east/north errors without changing padded payloads."""
    data = term.data
    valid = data.valid
    eps = jax.random.normal(
        key, (2, data.time_d.size), dtype=data.east_err_arcsec.dtype
    )
    sx = jnp.where(valid, data.east_err_arcsec, 1.0)
    sy = jnp.where(valid, data.north_err_arcsec, 1.0)
    rho = jnp.where(valid, data.corr, 0.0)
    east = prediction[0] + sx * eps[0]
    north = prediction[1] + sy * (rho * eps[0] + jnp.sqrt(1 - rho**2) * eps[1])
    sampled = eqx.tree_at(
        lambda d: (d.east_arcsec, d.north_arcsec),
        data,
        (
            jnp.where(valid, east, data.east_arcsec),
            jnp.where(valid, north, data.north_arcsec),
        ),
    )
    return eqx.tree_at(lambda t: t.data, term, sampled)


def _dynamics(system):
    """The orbix ``System`` of a scene, or the system itself."""
    return system if isinstance(system, System) else system.dynamics


def _record_form(design, data):
    """Assemble a :class:`LinearGaussian` from an arcsec-per-AU design and records."""
    valid = data.valid
    target = jnp.where(
        valid[:, None],
        jnp.stack([_as_wide(data.east_arcsec), _as_wide(data.north_arcsec)], -1),
        0.0,
    )
    sx = jnp.where(valid, _as_wide(data.east_err_arcsec), 1.0)
    sy = jnp.where(valid, _as_wide(data.north_err_arcsec), 1.0)
    rho = jnp.where(valid, _as_wide(data.corr), 0.0)
    scale = 1.0 / (1.0 - rho**2)
    off = -scale * rho / (sx * sy)
    precision = jnp.stack(
        [jnp.stack([scale / sx**2, off], -1), jnp.stack([off, scale / sy**2], -1)], -2
    )
    return LinearGaussian(
        design=jnp.where(valid[:, None, None], _as_wide(design), 0.0),
        target=target,
        precision=jnp.where(valid[:, None, None], precision, 0.0),
    )


class RelativeAstrometryTerm(AbstractLinearGaussianTerm):
    """Score relative east/north offsets from a system's sky positions.

    Attributes:
        data: Relative astrometry in arcsec, with a planet index per epoch.
        system_path: Model path of the object whose ``positions(time_d)`` returns
            shape ``(2, K, T)`` in arcsec, rows east and north. The empty path is
            the model itself. Data ``time_d`` is in days from the system's epoch. A bare
            ``orbix.system.System`` is also accepted, using ``solve`` and its public
            ``offset_arcsec`` observable; planet indices then refer to zero-based
            Jacobi chain order.
        forward_fields: Empty tuple because the term differentiates no stored fields.
        reads: Model paths a held build cached. Empty: the term reads the model
            directly.

    Predictions have shape ``(2, T)``; invalid rows are returned as zeros.
    """

    data: AstrometryData
    system_path: str = eqx.field(static=True, default="")
    forward_fields = ()
    reads: ClassVar[tuple[str, ...]] = ()

    def check(self, model) -> None:
        """Require every valid row's planet to index a planet of the system.

        Args:
            model: The model holding the system at ``system_path``, with concrete
                values.

        Raises:
            ValueError: For a valid row whose planet is negative or at least the
                number of planets.
        """
        system = resolve(model, self.system_path)
        _check_planets(
            system,
            self.data.planet,
            self.data.valid,
            "RelativeAstrometryTerm",
            lambda t: _sky_positions(system, t).shape[1],
            self.data.time_d,
        )

    def _prediction_and_data(self, model):
        valid = self.data.valid
        # Sanitize before propagation and division: masking a NaN result alone
        # does not prevent NaN derivatives from flowing through its operands.
        data = AstrometryData(
            time_d=jnp.where(valid, self.data.time_d, 0.0),
            east_arcsec=jnp.where(valid, self.data.east_arcsec, 0.0),
            north_arcsec=jnp.where(valid, self.data.north_arcsec, 0.0),
            east_err_arcsec=jnp.where(valid, self.data.east_err_arcsec, 1.0),
            north_err_arcsec=jnp.where(valid, self.data.north_err_arcsec, 1.0),
            corr=jnp.where(valid, self.data.corr, 0.0),
            planet=jnp.where(valid, self.data.planet, 0),
            valid=valid,
        )
        system = resolve(model, self.system_path)
        prediction = _relative_positions(system, data.time_d, data.planet)
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        return prediction, data

    def __call__(self, model):
        """Return the scalar log-likelihood.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            Scalar log-likelihood of the valid rows.

        Raises:
            PrecisionError: If float64 is unavailable.
        """
        require_float64("Relative astrometry log likelihood")
        prediction, data = self._prediction_and_data(model)
        return loglike_relative_astrom(*prediction, data)

    def predict(self, model):
        """Return each row's planet offset at its own epoch, ``(2, T)``.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            East/north offsets (arcsec) of each row's planet at its epoch, zero on
            invalid
            rows. Shape ``(2, T)``.
        """
        return self._prediction_and_data(model)[0]

    def draw(self, key, model) -> Self:
        """Copy with correlated Gaussian positions at the same epochs.

        Args:
            key: JAX random key.
            model: The model holding the system at ``system_path``.

        Returns:
            The term with east/north offsets drawn about the prediction, using the data
            errors and correlation. Invalid rows are unchanged.
        """
        return _sample_radec(key, self, self.predict(model))

    def linear_gaussian(self, model, planet=0) -> LinearGaussian:
        """The term as a Gaussian in the sky constants of orbit of ``planet``.

        Only ``n``, ``ex``, ``ey`` and the epoch of the planet's orbit in ``model``
        enter the design, so the form is exact for every value of the four constants at
        that shape, provided the records depend on the constants only through that
        orbit. That holds for a system with a single planet row, and the form refuses a
        system with more: the other rows' offsets, masses and mean motions move with the
        constants under a block that writes the star mass (``ThieleInnesBlock``). It
        also refuses a record of another planet, which one form cannot carry. The design
        is in arcsec per AU, the target and precision in the units of the data.

        Args:
            model: The model holding the system at ``system_path``.
            planet: Zero-based Jacobi chain index of the planet.

        Returns:
            The linear Gaussian form of the term in the four sky constants, with the
            design in arcsec per AU.

        Raises:
            PrecisionError: If float64 is unavailable.
        """
        require_float64("Astrometry linear Gaussian form")
        dynamics = _dynamics(resolve(model, self.system_path))
        shape = jax.tree.map(lambda a: a[planet], dynamics.orbits)
        valid = self.data.valid
        time_d = jnp.where(valid, self.data.time_d, shape.epoch_d)
        design = sky_design(shape, time_d) / dynamics.frame.dist_pc
        design = eqx.error_if(
            design,
            jnp.asarray(dynamics.orbits.a_AU.shape[-1] > 1),
            "linear_gaussian: the system has more than one planet row, so the other "
            "rows move with the constants",
        )
        design = eqx.error_if(
            design,
            jnp.any(valid & (self.data.planet != planet)),
            "linear_gaussian: a valid record belongs to another planet",
        )
        return _record_form(design, self.data)


class StellarAstrometryTerm(AbstractTerm):
    """Score the stellar reflex path in arcsec with correlated Gaussian errors.

    ``system_path`` names the model path of a system whose ``star_reflex(time_d)``
    returns east/north offsets of shape ``(2, T)``; the empty path is the model itself.
    Data ``time_d`` is in days from the system's epoch.
    Invalid rows are sanitized before prediction and returned as zeros.

    Attributes:
        data: Stellar reflex astrometry in arcsec.
        system_path: Model path of the system whose ``star_reflex(time_d)`` is read.
        forward_fields: Empty tuple because the term differentiates no stored fields.
        reads: Model paths a held build cached. Empty: the term reads the model
            directly.
    """

    data: AstrometryData
    system_path: str = eqx.field(static=True, default="")
    forward_fields = ()
    reads: ClassVar[tuple[str, ...]] = ()

    def _prediction_and_data(self, model):
        valid = self.data.valid
        data = AstrometryData(
            time_d=jnp.where(valid, self.data.time_d, 0.0),
            east_arcsec=jnp.where(valid, self.data.east_arcsec, 0.0),
            north_arcsec=jnp.where(valid, self.data.north_arcsec, 0.0),
            east_err_arcsec=jnp.where(valid, self.data.east_err_arcsec, 1.0),
            north_err_arcsec=jnp.where(valid, self.data.north_err_arcsec, 1.0),
            corr=jnp.where(valid, self.data.corr, 0.0),
            planet=self.data.planet,
            valid=valid,
        )
        system = resolve(model, self.system_path)
        prediction = _stellar_reflex(system, data.time_d)
        prediction = jnp.where(valid[None, :], prediction, 0.0)
        return prediction, data

    def __call__(self, model):
        """Return the scalar log-likelihood.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            Scalar log-likelihood of the valid rows.

        Raises:
            PrecisionError: If float64 is unavailable.
        """
        require_float64("Stellar astrometry log likelihood")
        prediction, data = self._prediction_and_data(model)
        return loglike_stellar_astrom(*prediction, data)

    def predict(self, model):
        """Return the stellar east/north path, ``(2, T)``.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            East/north reflex offsets (arcsec) at each row's epoch, zero on invalid
            rows.
            Shape ``(2, T)``.
        """
        return self._prediction_and_data(model)[0]

    def draw(self, key, model) -> Self:
        """Copy with correlated Gaussian positions at the same epochs.

        Args:
            key: JAX random key.
            model: The model holding the system at ``system_path``.

        Returns:
            The term with east/north offsets drawn about the prediction, using the data
            errors and correlation. Invalid rows are unchanged.
        """
        return _sample_radec(key, self, self.predict(model))
