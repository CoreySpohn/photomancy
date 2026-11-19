"""Orbit and population priors against the orbit charts' reference measure.

``orbit_log_prior`` scores a physical orbit against the reference measure of the
orbit charts, for assembly with a chart volume in ``SystemLogDensity``.
``FactorizedOrbitPrior`` pairs that density with the quantiles nested sampling
needs. The eccentricity law is a ``priorities`` distribution, built by the
``skyscapes.population`` constructors (``kipping_eccentricity``,
``rayleigh_eccentricity``) or from ``priorities`` directly.

Generic distributions (uniform, normal, log-uniform, half-normal and the rest) come from
``priorities``; the product over the fitted leaves of a scene is
:class:`photomancy.infer.priors.IndependentPrior`.
"""

import math
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from hwoutils.constants import two_pi
from priorities import AbstractDistribution, LogUniform
from skyscapes.population import kipping_eccentricity


def orbit_log_prior(orbit, *, period_range_d=(1.0, 100000.0), eccentricity=None):
    """One-planet density against e de d(cos i) dW dw dM dln(a/AU).

    The period is log-uniform on its inclusive physical bounds at the orbit's
    actual total mass. Angles occupy one fundamental cell. Orbits with ``e >= 1``
    or ``e <= 0`` are off support; any further cut on the eccentricity is the
    distribution's own truncation.

    Args:
        orbit: The scored orbit.
        period_range_d: Inclusive period bounds in days.
        eccentricity: A ``priorities`` distribution of ``e`` (for example a
            ``skyscapes.population`` constructor). ``None`` is
            ``kipping_eccentricity()``.
    """
    if eccentricity is None:
        eccentricity = kipping_eccentricity()
    e = orbit.e
    valid = (e > 0) & (e < 1)
    safe_e = jnp.where(valid, e, 0.5)
    area = eccentricity.log_prob(safe_e) - jnp.log(safe_e)
    valid &= jnp.isfinite(area)
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
    The same period bounds and eccentricity distribution define its
    nested-sampling quantiles.
    """

    get_orbit: Callable
    period_range_d: tuple[float, float] = (1.0, 100000.0)
    eccentricity: AbstractDistribution = eqx.field(default_factory=kipping_eccentricity)

    def __check_init__(self):
        """Validate the period bounds before sampling."""
        low, high = self.period_range_d
        if not 0 < low < high or not math.isfinite(high):
            raise ValueError("Period bounds must be positive, finite and ordered.")

    def __call__(self, system):
        """Score the selected orbit against the physical orbit measure."""
        return orbit_log_prior(
            self.get_orbit(system),
            period_range_d=self.period_range_d,
            eccentricity=self.eccentricity,
        )

    def period_icdf(self, q):
        """Quantile of the log-uniform period in days."""
        return LogUniform(*self.period_range_d).icdf(q)

    def eccentricity_icdf(self, q):
        """Quantile of the eccentricity distribution."""
        return self.eccentricity.icdf(q)


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
