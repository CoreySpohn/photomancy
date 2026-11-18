"""GPU-parallel orbit grid-search (adaptive importance sampling).

Discovery tier of photomancy.orbit: enumerate orbits consistent with sparse data
and return weighted particles.
"""

import math
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G
from orbix.equations import period_to_sma
from orbix.system import EccentricityVectorChart, FrameOrbit

from photomancy.orbit._frame import (
    ORBIT_ROWS_CHART,
    orbit_rows,
    sky_offsets_arcsec,
)
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.inference import orbit_chart_log_volume
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.quasi_random import roberts_sequence
from photomancy.particles import normalize_log_weights
from photomancy.posterior import SamplePosterior


class ParamBounds(eqx.Module):
    """Box bounds over an ordered parameter set, on the unit cube."""

    low: jnp.ndarray
    high: jnp.ndarray
    names: tuple = eqx.field(static=True)

    def __check_init__(self):
        """Validate that low/high shapes agree with the number of names."""
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must share a shape")
        if self.low.shape[-1] != len(self.names):
            raise ValueError("bounds width must match number of names")
        if not bool(jnp.all(self.low <= self.high)):
            raise ValueError("each low bound must be <= its high bound")

    def scale(self, u):
        """Map unit-cube points ``(n, d)`` to physical box values ``(n, d)``."""
        return self.low + u * (self.high - self.low)


def chart_box_bounds(log_T_range=(0.0, 4.0), e_max=0.9):
    """Unit-cube box ``(logT, ex, ey, cos_i, W, tp_frac)`` of the eccentricity grid.

    ``(ex, ey)`` is ``(e cos w, e sin w)`` over the square ``[-e_max, e_max]^2``,
    which admits ``e >= 1`` in its corners for ``e_max > 1/sqrt(2)``.
    """
    names = ("logT", "ex", "ey", "cos_i", "W", "tp_frac")
    low = jnp.array([log_T_range[0], -e_max, -e_max, -1.0, 0.0, 0.0])
    high = jnp.array([log_T_range[1], e_max, e_max, 1.0, 2.0 * jnp.pi, 1.0])
    return ParamBounds(low=low, high=high, names=names)


class AbstractGridStrategy(eqx.Module):
    """Produces Stage-1 global samples and a Stage-2 refined proposal."""

    @abstractmethod
    def stage1(self, key, ndim, n):
        """Return ``(n, ndim)`` unit-cube samples for the global exploration stage."""

    @abstractmethod
    def stage2(self, key, survivors, n):
        """Return ``(samples, log_q)`` from a refined proposal around survivors."""


class AdaptiveImportanceSampler(AbstractGridStrategy):
    """Roberts global fill, Gaussian-mixture refinement around survivors.

    Attributes:
        n_modes: Number of Gaussian mixture components in Stage 2.
        jitter: Diagonal regularization added to the empirical covariance.

    Stage-2 draws may leave the unit cube; the target is zero there and
    ``grid_search`` assigns them zero weight, which keeps the weights exact
    importance weights of the mixture density ``log_q``.
    """

    n_modes: int = eqx.field(static=True, default=5)
    jitter: float = eqx.field(static=True, default=1e-6)

    def stage1(self, key, ndim, n):
        """Return ``(n, ndim)`` Roberts quasi-random points in the unit cube.

        Args:
            key: JAX PRNG key for a Cranley-Patterson rotation.
            ndim: Dimension of the parameter space.
            n: Number of points.

        Returns:
            Array of shape ``(n, ndim)`` with values in ``[0, 1)``.
        """
        return roberts_sequence(n, ndim, key=key)

    def stage2(self, key, survivors, n):
        """Gaussian-mixture proposal around the survivor set.

        Picks ``n_modes`` survivor centers (first rows, assumed sorted best-first),
        fits an isotropic-plus-empirical covariance to the full survivor set, then
        draws a balanced mixture. Samples are not clipped.

        Args:
            key: JAX PRNG key.
            survivors: Best unit-cube points from Stage 1, shape ``(m, d)``.
            n: Number of new samples to draw.

        Returns:
            Tuple ``(z, log_q)`` where ``z`` has shape ``(n, d)`` (it may leave
            the unit cube) and ``log_q`` has shape ``(n,)`` (mixture log-density).
        """
        if survivors.shape[0] < self.n_modes:
            raise ValueError(
                f"need at least n_modes ({self.n_modes}) survivors, "
                f"got {survivors.shape[0]}"
            )
        k_draw, k_pick = jax.random.split(key)
        m = self.n_modes
        d = survivors.shape[1]
        centers = survivors[:m]
        cov = jnp.cov(survivors, rowvar=False) + self.jitter * jnp.eye(d)
        chol = jnp.linalg.cholesky(cov)
        comp = jax.random.randint(k_pick, (n,), 0, m)
        eps = jax.random.normal(k_draw, (n, d))
        z = centers[comp] + eps @ chol.T
        diff = z[:, None, :] - centers[None, :, :]
        diff_flat = diff.reshape(n * m, d)
        sol_flat = jax.scipy.linalg.solve_triangular(chol, diff_flat.T, lower=True)
        maha = jnp.sum(sol_flat**2, axis=0).reshape(n, m)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        log_comp = -0.5 * (maha + logdet + d * jnp.log(2.0 * jnp.pi))
        log_q = jax.scipy.special.logsumexp(log_comp, axis=1) - jnp.log(m)
        return z, log_q


def batched_loglike(single_eval, phys, n_particles, chunk_size):
    """Evaluate ``single_eval`` over ``n_particles`` via vmap inside lax.scan.

    ``phys`` is a dict of ``(n_particles,)`` arrays. Chunking keeps peak memory
    at ``O(chunk_size)`` per step; the per-particle time series stay in registers.

    Args:
        single_eval: Callable ``phys_scalar -> scalar`` scoring one particle.
        phys: Dict of arrays, each of shape ``(n_particles, ...)``.
        n_particles: Total number of particles. Must be divisible by ``chunk_size``.
        chunk_size: Number of particles evaluated per scan step.

    Returns:
        Array of shape ``(n_particles,)`` containing log-likelihood values.
    """
    if n_particles % chunk_size != 0:
        raise ValueError(
            f"n_particles ({n_particles}) must be divisible by "
            f"chunk_size ({chunk_size})"
        )
    n_chunks = n_particles // chunk_size
    eval_chunk = jax.vmap(single_eval)

    def scan_body(carry, chunk):
        return carry, eval_chunk(chunk)

    reshaped = {
        k: v[: n_chunks * chunk_size].reshape((n_chunks, chunk_size, *v.shape[1:]))
        for k, v in phys.items()
    }
    _, ll = jax.lax.scan(scan_body, None, reshaped)
    return ll.reshape(-1)


_CHART = EccentricityVectorChart()


def _box_to_chart(p, star_mass_kg):
    """Chart coordinates of box points ``(..., 6)``, and whether each has e < 1.

    The orbit is built by orbix at epoch zero with ``tp = tp_frac * T``, then read
    back in the eccentricity-vector chart, so periapsis angle, node and mean
    longitude follow orbix's conventions. Points with ``e >= 1`` have no bound
    orbit; their coordinates are finite placeholders.
    """
    logT, ex, ey, cos_i, W, tp_frac = jnp.moveaxis(p, -1, 0)
    ecc = jnp.hypot(ex, ey)
    bound = ecc < 1.0
    safe = jnp.where(ecc > 0.0, ecc, 1.0)
    phase = 2.0 * jnp.pi * tp_frac
    orbit = FrameOrbit.from_classical_trig(
        a_AU=period_to_sma(10.0**logT, star_mass_kg),
        e=jnp.where(bound, ecc, 0.5),
        cos_i=cos_i,
        sin_i=jnp.sqrt(jnp.maximum(1.0 - cos_i**2, 0.0)),
        cos_W=jnp.cos(W),
        sin_W=jnp.sin(W),
        cos_w=jnp.where(ecc > 0.0, ex / safe, 1.0),
        sin_w=jnp.where(ecc > 0.0, ey / safe, 0.0),
        cos_M0=jnp.cos(phase),
        sin_M0=-jnp.sin(phase),
        mu_AU3_per_d2=G * star_mass_kg,
        epoch_d=0.0,
    )
    return _CHART.from_orbit(orbit), bound


def _box_log_jacobian(p, bounds):
    """Log determinant of the map from the unit cube to the chart coordinates.

    With ``ln a = (2/3) ln T`` at fixed mass, the disk map contributes
    ``(1 - e^2)^-2``, ``atanh(cos i)`` contributes ``1 / (1 - cos^2 i)``, and the
    mean longitude ``w - 2 pi tp_frac`` contributes ``2 pi``.
    """
    ecc2 = p[..., 1] ** 2 + p[..., 2] ** 2
    widths = jnp.sum(jnp.log(bounds.high - bounds.low))
    return (
        math.log(2.0 / 3.0 * math.log(10.0))
        - 2.0 * jnp.log1p(-ecc2)
        - jnp.log1p(-(p[..., 3] ** 2))
        + math.log(2.0 * math.pi)
        + widths
    )


def grid_search(
    data,
    *,
    star_mass_kg,
    dist_pc,
    sampler,
    log_T_range,
    e_max,
    n_particles=10**6,
    chunk_size=10**5,
    n_survivors=5000,
    key,
    epoch=None,
):
    """Two-stage importance-sampling grid search weighted in chart coordinates.

    The target is the likelihood times a flat density against the chart reference
    measure over the box ``(logT, ex, ey, cos_i, W, tp_frac)``, built through orbix
    and weighted by ``log prior + chart log volume + proposal Jacobian + log
    likelihood - log q``. The flat prior is normalized on its support, the disk
    ``e < min(e_max, 1)`` of the ``(ex, ey)`` square, so the evidence is the mean
    likelihood over the support. Draws outside the unit cube get zero weight.

    Args:
        data: Tuple of data containers (``RelativeAstromData`` is scored).
        star_mass_kg: Stellar mass (kg).
        dist_pc: Distance (parsec).
        sampler: An ``AbstractGridStrategy``.
        log_T_range: ``(log10_T_min, log10_T_max)`` in days.
        e_max: Radius of the eccentricity disk; the box is ``[-e_max, e_max]^2``.
        n_particles: Particles per stage; divisible by ``chunk_size``.
        chunk_size: Scan chunk size for memory control.
        n_survivors: Stage-1 particles passed to Stage 2.
        key: JAX PRNG key.
        epoch: The :class:`~hwoutils.epoch.Epoch` that the data times (and the
            returned ``tp``) are measured from; declared on the posterior.

    Returns:
        ``SamplePosterior`` over ``(a, e, cos_i, W, cos_w, sin_w, tp)`` with
        normalized log weights and the importance-sampling log evidence. It
        declares the chart ``ORBIT_ROWS_CHART`` and ``epoch``.
    """
    k1, k2 = jax.random.split(key)
    bounds = chart_box_bounds(log_T_range=log_T_range, e_max=e_max)
    astrom = next((d for d in data if isinstance(d, RelativeAstromData)), None)
    mu = G * star_mass_kg
    area = math.pi * min(e_max, 1.0) ** 2
    # Measure of the support: eccentricity area x d ln a x (d cos i dW d longitude).
    log_prior = -(
        math.log(area)
        + math.log(2.0 / 3.0 * math.log(10.0) * (log_T_range[1] - log_T_range[0]))
        + math.log(2.0 * (2.0 * math.pi) ** 2)
    )

    def states(u):
        p = bounds.scale(u)
        x, bound = _box_to_chart(p, star_mass_kg)
        ecc = jnp.hypot(p[..., 1], p[..., 2])
        support = bound & jnp.all((u >= 0.0) & (u < 1.0), axis=-1)
        support &= ecc < e_max
        return x, support, p

    def single_loglike(phys):
        x = phys["x"]
        if astrom is None:
            return 0.0
        orbit = _CHART.to_orbit(x, mu_AU3_per_d2=mu, epoch_d=0.0)
        ra, dec = sky_offsets_arcsec(orbit, astrom.times, dist_pc)
        return loglike_relative_astrom(ra, dec, astrom)

    def scored(u):
        x, support, p = states(u)
        ll = batched_loglike(single_loglike, {"x": x}, n_particles, chunk_size)
        ll = jnp.where(support & jnp.isfinite(ll), ll, -jnp.inf)
        return x, support, p, ll

    u1 = sampler.stage1(k1, len(bounds.names), n_particles)
    ll1 = scored(u1)[3]
    order = jnp.argsort(ll1)[::-1][:n_survivors]
    u2, log_q = sampler.stage2(k2, u1[order], n_particles)
    x2, support2, p2, ll2 = scored(u2)
    safe_x = jnp.where(support2[:, None], x2, 0.0)
    log_target = (
        log_prior
        + orbit_chart_log_volume(_CHART, safe_x)
        + _box_log_jacobian(jnp.where(support2[:, None], p2, 0.0), bounds)
        + ll2
    )
    log_w, log_total, _ = normalize_log_weights(
        jnp.where(support2 & jnp.isfinite(log_target), log_target - log_q, -jnp.inf)
    )
    orbits = _CHART.to_orbit(safe_x, mu_AU3_per_d2=mu, epoch_d=0.0)
    rows = orbit_rows(orbits)
    rows = rows.at[:, 6].set(jnp.mod(rows[:, 6], orbits.period_d))
    return SamplePosterior(
        samples=rows,
        log_weights=log_w,
        evidence=log_total - jnp.log(n_particles),
        param_names=("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp"),
        chart=ORBIT_ROWS_CHART,
        epoch=epoch,
    )
