"""Orbit and population priors against the orbit charts' reference measure.

``orbit_log_prior`` scores a physical orbit against the reference measure of the
orbit charts, for assembly with a chart volume in ``SystemLogDensity``.
``FactorizedOrbitPrior`` pairs that density with the quantiles nested sampling
needs. The eccentricity distributions are named here (uniform, Kipping 2013 Beta,
Rayleigh); the beta quantile is imported only when a Kipping eccentricity draw is
requested.

Generic distributions (uniform, normal, log-uniform, half-normal and the rest) come from
``priorities``; the product over the fitted leaves of a scene is
:class:`photomancy.infer.priors.IndependentPrior`.
"""

import math
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi
from jax.scipy.special import betainc
from jax.scipy.stats import beta
from priorities import LogUniform

ECC_ALPHA = 0.867
ECC_BETA = 3.03

RAYLEIGH_SCALES = {"rayleigh": 0.3, "vaneylen19": 0.049}
ECC_PRIOR_NAMES = ("disk", "kipping13", *RAYLEIGH_SCALES)


def _rayleigh_cdf(e, sigma):
    return -jnp.expm1(-0.5 * (e / sigma) ** 2)


def _rayleigh_icdf(unit, sigma, e_max):
    e = sigma * jnp.sqrt(-2.0 * jnp.log1p(-unit * _rayleigh_cdf(e_max, sigma)))
    return jnp.minimum(e, jnp.nextafter(jnp.asarray(e_max, dtype=e.dtype), 0.0))


def eccentricity_draws(key, shape, ecc_prior="kipping13"):
    """Draw eccentricities from a named law on [0, 1), without any sampler context.

    The laws are ``"disk"`` (uniform), ``"kipping13"`` (Beta(0.867, 3.03)),
    ``"rayleigh"`` (sigma 0.3) and ``"vaneylen19"`` (sigma 0.049), the Rayleigh laws
    conditioned on ``e < 1``.
    """
    if ecc_prior not in ECC_PRIOR_NAMES:
        raise ValueError(f"Unknown ecc_prior={ecc_prior!r}")
    if ecc_prior == "kipping13":
        return jax.random.beta(key, ECC_ALPHA, ECC_BETA, shape)
    unit = jax.random.uniform(key, shape)
    if ecc_prior == "disk":
        return unit
    return _rayleigh_icdf(unit, RAYLEIGH_SCALES[ecc_prior], 1.0)


def _rayleigh_log_area(e, sigma, e_max):
    """Truncated Rayleigh log density against e de, including its limit at zero."""
    return (
        -0.5 * (e / sigma) ** 2
        - 2 * jnp.log(sigma)
        - jnp.log(_rayleigh_cdf(e_max, sigma))
    )


def orbit_log_prior(
    orbit, *, period_range_d=(1.0, 100000.0), ecc_prior="kipping13", e_max=1.0
):
    """One-planet density against e de d(cos i) dW dw dM dln(a/AU).

    The period is log-uniform on its inclusive physical bounds at the orbit's
    actual total mass. Angles occupy one fundamental cell. Condition the named
    eccentricity law on e < e_max. At e=0 the Rayleigh laws have a finite
    area density; assign zero density at the singular point for the other laws.
    """
    e = orbit.e
    valid = (e >= 0) & (e < e_max) & (e_max <= 1)
    safe_e = jnp.where(valid & (e > 0), e, 0.5)
    if ecc_prior in RAYLEIGH_SCALES:
        area = _rayleigh_log_area(
            jnp.where(valid, e, 0.0), RAYLEIGH_SCALES[ecc_prior], e_max
        )
    elif ecc_prior == "disk":
        valid = valid & (e > 0)
        area = -jnp.log(safe_e) - jnp.log(e_max)
    elif ecc_prior == "kipping13":
        valid = valid & (e > 0)
        area = (
            beta.logpdf(safe_e, ECC_ALPHA, ECC_BETA)
            - jnp.log(safe_e)
            - jnp.log(betainc(ECC_ALPHA, ECC_BETA, jnp.asarray(e_max, dtype=e.dtype)))
        )
    else:
        raise ValueError(f"Unknown ecc_prior={ecc_prior!r}")
    low_d, high_d = map(jnp.asarray, period_range_d)
    valid &= (
        (low_d > 0)
        & (high_d > low_d)
        & jnp.isfinite(high_d)
        & (orbit.period_d >= low_d)
        & (orbit.period_d <= high_d)
    )
    return jnp.where(
        valid, area + jnp.log(1.5 / (2 * two_pi**3 * jnp.log(high_d / low_d))), -jnp.inf
    )


class FactorizedOrbitPrior(eqx.Module):
    """One-planet orbit prior with independent classical elements.

    ``get_orbit(system)`` selects the scalar orbit scored by the density.
    The same bounds and eccentricity law define its nested-sampling quantiles.
    """

    get_orbit: Callable
    period_range_d: tuple[float, float] = (1.0, 100000.0)
    ecc_prior: str = "kipping13"
    e_max: float = 1.0

    def __check_init__(self):
        """Validate the family and truncation before sampling."""
        low, high = self.period_range_d
        if not 0 < low < high or not math.isfinite(high):
            raise ValueError("Period bounds must be positive, finite and ordered.")
        if not 0 < self.e_max <= 1:
            raise ValueError("e_max must lie in (0, 1].")
        if self.ecc_prior not in ECC_PRIOR_NAMES:
            raise ValueError(f"Unknown ecc_prior={self.ecc_prior!r}")

    def __call__(self, system):
        """Score the selected orbit against the physical orbit measure."""
        return orbit_log_prior(
            self.get_orbit(system),
            period_range_d=self.period_range_d,
            ecc_prior=self.ecc_prior,
            e_max=self.e_max,
        )

    def period_icdf(self, unit):
        """Quantile of the log-uniform period in days."""
        return LogUniform(*self.period_range_d).icdf(unit)

    def eccentricity_icdf(self, unit):
        """Quantile of the named eccentricity law conditioned on e < e_max."""
        if self.ecc_prior == "disk":
            return unit * self.e_max
        if self.ecc_prior in RAYLEIGH_SCALES:
            return _rayleigh_icdf(unit, RAYLEIGH_SCALES[self.ecc_prior], self.e_max)
        from tensorflow_probability.substrates.jax.math import betaincinv

        alpha, beta = (jnp.asarray(x, dtype=unit.dtype) for x in (ECC_ALPHA, ECC_BETA))
        probability = unit * betainc(
            alpha, beta, jnp.asarray(self.e_max, dtype=unit.dtype)
        )
        return betaincinv(alpha, beta, probability)


class PopulationIntensity(Protocol):
    """Per-slot intensity against dnu d(mass/Mearth) and discrete type counting."""

    def log_intensity(self, system, stars):
        """Return per-slot log occurrence intensity in the stated measure."""

    def log_rate(self, stars):
        """Return log integrated occurrence rates per host."""


class PopulationPrior(eqx.Module):
    """One host and one valid planet, possibly padded, against dnu d(mass/Mearth).

    Subtract the host log occurrence rate once to condition on one planet.
    A sampler using ln(mass/Mearth) must add ln(mass/Mearth) to its log volume.
    The population supplies both the mass and orbit law.
    """

    population: PopulationIntensity
    stars: object

    def __call__(self, system):
        """Return the normalized one-planet mark density, or -inf off support."""
        (rate,) = self.population.log_rate(self.stars)
        values = self.population.log_intensity(system, self.stars)
        if values.ndim != 1:
            raise ValueError("PopulationPrior requires an unbatched system")
        return jnp.where(
            (system.valid.sum() == 1) & jnp.isfinite(rate),
            jnp.where(system.valid, values, 0.0).sum() - rate,
            -jnp.inf,
        )
