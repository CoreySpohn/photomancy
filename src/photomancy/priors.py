"""Unified prior layer: distributions over the fitted-leaf vector ``z``.

One ``AbstractPrior`` interface serves the fitted-leaf backends:
``log_prob`` for BlackJAX / Laplace (folded into the target logdensity), ``sample`` for
SMC's initial particles, and ``forward`` / ``inverse`` / ``log_prob`` for jaxns (via the
``_JaxnsPrior`` adapter in ``backends/nested.py``). Priors are over ``z`` (the raveled
fitted leaves) -- the same space ``build_scene_logdensity`` uses.

``orbit_log_prior`` instead scores a physical orbit against the orbit charts'
reference measure, for assembly with a chart volume in ``SystemLogDensity``.

Physical orbit and scalar priors pair normalized densities with their quantiles.
The beta quantile is imported only when a Kipping eccentricity draw is requested.
"""

import math
from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import betainc, erf, erfinv, logsumexp
from jax.scipy.stats import beta

ECC_ALPHA = 0.867
ECC_BETA = 3.03

_SQRT2 = 1.4142135623730951  # plain float: no import-time array, no precision lock

_MIXTURE_NO_CUBE = (
    "MixturePrior has no closed-form unit-cube transform; use it with a gradient / "
    "sample backend (Laplace / NUTS / SMC), not jaxns."
)


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


def _log_uniform_icdf(unit, bounds):
    low, high = bounds
    return jnp.exp(jnp.log(low) + unit * (jnp.log(high) - jnp.log(low)))


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


def log_uniform_prior(value, *, bounds):
    """Elementwise log density against d(value), on finite positive physical bounds."""
    low, high = map(jnp.asarray, bounds)
    valid = (
        (value >= low) & (value <= high) & (low > 0) & (high > low) & jnp.isfinite(high)
    )
    safe = jnp.where(valid, value, 1.0)
    return jnp.where(valid, -jnp.log(safe) - jnp.log(jnp.log(high / low)), -jnp.inf)


def half_normal_log_prior(value, *, scale):
    """Elementwise log density against d(value), e.g. RV jitter in m/s."""
    return jnp.where(
        (value >= 0) & (scale > 0) & jnp.isfinite(scale),
        0.5 * jnp.log(2 / jnp.pi) - jnp.log(scale) - 0.5 * (value / scale) ** 2,
        -jnp.inf,
    )


class LogUniformPrior(eqx.Module):
    """Independent log-uniform scalar in positive physical units."""

    bounds: tuple[float, float]

    def __check_init__(self):
        """Require a proper positive log-uniform interval."""
        low, high = self.bounds
        if not 0 < low < high or not math.isfinite(high):
            raise ValueError("Log-uniform bounds must be positive, finite and ordered.")

    def __call__(self, value):
        """Elementwise log density against the physical differential."""
        return log_uniform_prior(value, bounds=self.bounds)

    def icdf(self, unit):
        """Map a unit probability to its physical quantile."""
        return _log_uniform_icdf(unit, self.bounds)


class HalfNormalPrior(eqx.Module):
    """Independent nonnegative physical scalar, e.g. instrument jitter."""

    scale: float

    def __check_init__(self):
        """Require a proper half-normal scale."""
        if not self.scale > 0 or not math.isfinite(self.scale):
            raise ValueError("Half-normal scale must be positive and finite.")

    def __call__(self, value):
        """Elementwise log density against the physical differential."""
        return half_normal_log_prior(value, scale=self.scale)

    def icdf(self, unit):
        """Map a unit probability to its physical quantile."""
        return self.scale * _SQRT2 * erfinv(unit)


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
        return _log_uniform_icdf(unit, self.period_range_d)

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


def _std_normal_quantile(u):
    """Inverse standard-normal CDF on ``u`` in (0, 1)."""
    return _SQRT2 * erfinv(2.0 * u - 1.0)


def _std_normal_cdf(x):
    """Standard-normal CDF."""
    return 0.5 * (1.0 + erf(x / _SQRT2))


def _mvn_logpdf(z, mean, cholesky):
    """Multivariate-Normal log-density at ``z`` for ``mean`` + lower-Cholesky ``L``."""
    d = mean.shape[0]
    w = solve_triangular(cholesky, z - mean, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    return -0.5 * (jnp.sum(w**2) + d * jnp.log(2.0 * jnp.pi) + logdet)


class AbstractPrior(eqx.Module):
    """A prior over the fitted-leaf vector ``z`` of dimension ``ndim``.

    ``forward`` maps a unit-cube draw ``U`` in ``[0, 1]^ndim`` to ``z`` (the inverse-CDF
    transform jaxns needs); ``inverse`` is its inverse; ``log_prob`` is the log-density
    (for BlackJAX / Laplace). ``sample`` defaults to inverse-CDF sampling.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of fitted dimensions (the length of ``z``)."""

    @abstractmethod
    def forward(self, U):
        """Map a unit-cube draw ``U`` in ``[0, 1]^ndim`` to a parameter draw ``z``."""

    @abstractmethod
    def inverse(self, z):
        """Map ``z`` back to the unit cube ``[0, 1]^ndim``."""

    @abstractmethod
    def log_prob(self, z):
        """Prior log-density at ``z``."""

    def sample(self, key, n):
        """Draw ``n`` samples, shape ``(n, ndim)``, by transforming uniform draws."""
        U = jax.random.uniform(key, (n, self.ndim))
        return jax.vmap(self.forward)(U)


class Uniform(AbstractPrior):
    """Independent Uniform on each component; ``low`` / ``high`` are ``(ndim,)``."""

    low: jnp.ndarray
    high: jnp.ndarray

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.low.shape[0]

    def forward(self, U):
        """Affine map of the unit cube onto ``[low, high]``."""
        return self.low + (self.high - self.low) * U

    def inverse(self, z):
        """Map ``z`` back to the unit cube."""
        return (z - self.low) / (self.high - self.low)

    def log_prob(self, z):
        """Constant ``-sum(log(high-low))`` inside the box, ``-inf`` outside."""
        in_box = jnp.all((z >= self.low) & (z <= self.high))
        lp = -jnp.sum(jnp.log(self.high - self.low))
        return jnp.where(in_box, lp, -jnp.inf)


class Normal(AbstractPrior):
    """Independent Normal on each component; ``loc`` / ``scale`` are ``(ndim,)``."""

    loc: jnp.ndarray
    scale: jnp.ndarray

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.loc.shape[0]

    def forward(self, U):
        """Normal inverse-CDF of the unit cube."""
        return self.loc + self.scale * _std_normal_quantile(U)

    def inverse(self, z):
        """Normal CDF of ``z``."""
        return _std_normal_cdf((z - self.loc) / self.scale)

    def log_prob(self, z):
        """Gaussian log-density summed over components."""
        return jnp.sum(
            -0.5 * ((z - self.loc) / self.scale) ** 2
            - jnp.log(self.scale)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )


class LogNormal(AbstractPrior):
    """Independent LogNormal; ``loc`` / ``scale`` are the underlying Normal's params."""

    loc: jnp.ndarray
    scale: jnp.ndarray

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.loc.shape[0]

    def _normal(self):
        """The underlying Normal in log-space."""
        return Normal(loc=self.loc, scale=self.scale)

    def forward(self, U):
        """Exponentiate the underlying Normal's inverse-CDF."""
        return jnp.exp(self._normal().forward(U))

    def inverse(self, z):
        """Underlying Normal CDF of ``log z``."""
        return self._normal().inverse(jnp.log(z))

    def log_prob(self, z):
        """LogNormal log-density (Normal in log-space plus the ``-log z`` Jacobian)."""
        return self._normal().log_prob(jnp.log(z)) - jnp.sum(jnp.log(z))


class IndependentPrior(AbstractPrior):
    """Product of per-leaf priors, concatenated in z-order."""

    priors: tuple

    @property
    def ndim(self):
        """Total dimensions across the component priors."""
        return sum(p.ndim for p in self.priors)

    def _blocks(self, v):
        """Split a ``(ndim,)`` vector into per-prior slices."""
        out, i = [], 0
        for p in self.priors:
            out.append(v[i : i + p.ndim])
            i += p.ndim
        return out

    def forward(self, U):
        """Per-component ``forward``, concatenated."""
        return jnp.concatenate(
            [p.forward(b) for p, b in zip(self.priors, self._blocks(U), strict=True)]
        )

    def inverse(self, z):
        """Per-component ``inverse``, concatenated."""
        return jnp.concatenate(
            [p.inverse(b) for p, b in zip(self.priors, self._blocks(z), strict=True)]
        )

    def log_prob(self, z):
        """Sum of the component log-densities."""
        return sum(
            p.log_prob(b) for p, b in zip(self.priors, self._blocks(z), strict=True)
        )


class JointPrior(AbstractPrior):
    """A multivariate-Normal prior over ``z`` with ``cov = cholesky @ cholesky.T``.

    What ``to_prior()`` produces for a Gaussian (Laplace) posterior, so a fit's
    posterior carries its full covariance into the next fit's prior.
    """

    mean: jnp.ndarray
    cholesky: jnp.ndarray  # lower-triangular L

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.mean.shape[0]

    def forward(self, U):
        """``mean + L @ Phi^-1(U)`` -- affine of standard normals."""
        return self.mean + self.cholesky @ _std_normal_quantile(U)

    def inverse(self, z):
        """Standard-normal CDF of the whitened ``z``."""
        w = solve_triangular(self.cholesky, z - self.mean, lower=True)
        return _std_normal_cdf(w)

    def log_prob(self, z):
        """Multivariate-Normal log-density via the triangular solve."""
        return _mvn_logpdf(z, self.mean, self.cholesky)


class MixturePrior(AbstractPrior):
    """A Gaussian-mixture prior over ``z`` -- weighted modes, each an MVN.

    What ``MixturePosterior.to_prior()`` (and ``cluster_to_mixture`` for weighted
    samples) produce: it preserves multimodality across sequential epochs
    where a single Gaussian would collapse the modes. ``log_prob`` and ``sample`` are
    exact (serving the Laplace / NUTS / SMC family); ``forward`` / ``inverse`` -- the
    jaxns unit-cube transform -- are not implemented, as a multivariate mixture has no
    closed-form inverse-CDF (use a gradient / sample backend with a mixture prior).
    """

    means: jnp.ndarray  # (K, d)
    choleskys: jnp.ndarray  # (K, d, d), each lower-triangular
    log_weights: jnp.ndarray  # (K,), unnormalized log mixture weights

    @property
    def ndim(self):
        """Number of dimensions ``d``."""
        return self.means.shape[1]

    def log_prob(self, z):
        """Mixture log-density: ``logsumexp_k(log w_k + N(z | mode_k))``."""
        logw = self.log_weights - logsumexp(self.log_weights)
        comp = jax.vmap(lambda m, L: _mvn_logpdf(z, m, L))(self.means, self.choleskys)
        return logsumexp(logw + comp)

    def sample(self, key, n):
        """Draw ``n`` samples: pick a mode by weight, then sample its Gaussian."""
        k_comp, k_draw = jax.random.split(key)
        comp = jax.random.categorical(k_comp, self.log_weights, shape=(n,))
        keys = jax.random.split(k_draw, n)

        def draw(k, idx):
            std = jax.random.normal(k, (self.ndim,))
            return self.means[idx] + self.choleskys[idx] @ std

        return jax.vmap(draw)(keys, comp)

    def forward(self, U):
        """Not implemented -- a multivariate mixture has no closed-form inverse-CDF."""
        raise NotImplementedError(_MIXTURE_NO_CUBE)

    def inverse(self, z):
        """Not implemented -- a multivariate mixture has no closed-form inverse-CDF."""
        raise NotImplementedError(_MIXTURE_NO_CUBE)
