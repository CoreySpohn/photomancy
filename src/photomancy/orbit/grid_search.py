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

from photomancy.orbit._frame import orbit_rows, sky_offsets_arcsec
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
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


class AbstractShapeParam(eqx.Module):
    """Maps an ordered unit-cube sample to a physical orbit-shape dict."""

    @abstractmethod
    def default_bounds(self, log_T_range, e_max) -> ParamBounds:
        """Return default ParamBounds for this parameterization."""

    @abstractmethod
    def to_physical(self, u, bounds, Ms) -> dict:
        """Convert unit-cube samples to a physical orbit-parameter dict."""


def chart_box_bounds(log_T_range=(0.0, 4.0), e_max=0.9):
    """Unit-cube box ``(logT, ex, ey, cos_i, W, tp_frac)`` of the eccentricity grid.

    ``(ex, ey)`` is ``(e cos w, e sin w)`` over the square ``[-e_max, e_max]^2``,
    which admits ``e >= 1`` in its corners for ``e_max > 1/sqrt(2)``.
    """
    names = ("logT", "ex", "ey", "cos_i", "W", "tp_frac")
    low = jnp.array([log_T_range[0], -e_max, -e_max, -1.0, 0.0, 0.0])
    high = jnp.array([log_T_range[1], e_max, e_max, 1.0, 2.0 * jnp.pi, 1.0])
    return ParamBounds(low=low, high=high, names=names)


class EccVectorShape(AbstractShapeParam):
    """Eccentricity-vector coordinate ``(logT, ex, ey, cos_i, W, tp_frac)``.

    Sampled names in order: ``logT, ex, ey, cos_i, W, tp_frac``.
    Derived quantities: ``T = 10**logT``, ``a = period_to_sma(T, Ms)``,
    ``e = hypot(ex, ey)``, ``cos_w = ex/e`` (1 when e=0), ``sin_w = ey/e``
    (0 when e=0), ``W`` in radians, ``tp = tp_frac * T``.
    """

    def default_bounds(self, log_T_range=(0.0, 4.0), e_max=0.9):
        """Return default ParamBounds for EccVectorShape.

        Args:
            log_T_range: ``(log10_T_min, log10_T_max)`` in days.
            e_max: Maximum eccentricity vector component magnitude.

        Returns:
            ParamBounds with six named parameters.
        """
        return chart_box_bounds(log_T_range, e_max)

    def to_physical(self, u, bounds, Ms):
        """Convert unit-cube samples to physical orbit parameters.

        Args:
            u: Unit-cube samples of shape ``(n, 6)``.
            bounds: ParamBounds returned by ``default_bounds``.
            Ms: Stellar mass in kg.

        Returns:
            Dict of physical parameter arrays, each of shape ``(n,)``.
            Keys: ``T, a, e, cos_i, W, cos_w, sin_w, tp``.
        """
        p = bounds.scale(u)
        logT, ex, ey, cos_i, W, tp_frac = (p[:, i] for i in range(6))
        T = 10.0**logT
        a = period_to_sma(T, Ms)
        e = jnp.hypot(ex, ey)
        safe_e = jnp.where(e > 0.0, e, 1.0)
        cos_w = jnp.where(e > 0.0, ex / safe_e, 1.0)
        sin_w = jnp.where(e > 0.0, ey / safe_e, 0.0)
        return {
            "T": T,
            "a": a,
            "e": e,
            "cos_i": cos_i,
            "W": W,
            "cos_w": cos_w,
            "sin_w": sin_w,
            "tp": tp_frac * T,
        }


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
        clip: Clip Stage-2 draws onto the unit cube. Clipping piles probability
            mass into atoms on the cube faces while ``log_q`` stays the unclipped
            mixture density, so the weights are not importance weights. With
            ``clip=False`` draws may leave the cube; the target is zero there and
            ``chart_grid_search`` assigns them zero weight, which is exact.
    """

    n_modes: int = eqx.field(static=True, default=5)
    jitter: float = eqx.field(static=True, default=1e-6)
    clip: bool = eqx.field(static=True, default=True)

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
        draws a balanced mixture. Samples are clipped to the unit cube when
        ``clip`` is set.

        Args:
            key: JAX PRNG key.
            survivors: Best unit-cube points from Stage 1, shape ``(m, d)``.
            n: Number of new samples to draw.

        Returns:
            Tuple ``(z, log_q)`` where ``z`` has shape ``(n, d)`` (clipped to
            ``[0, 1)`` when ``clip`` is set) and ``log_q`` has shape ``(n,)``
            (mixture log-density).
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
        if self.clip:
            z = jnp.clip(z, 0.0, 1.0 - 1e-7)
        diff = z[:, None, :] - centers[None, :, :]
        diff_flat = diff.reshape(n * m, d)
        sol_flat = jax.scipy.linalg.solve_triangular(chol, diff_flat.T, lower=True)
        maha = jnp.sum(sol_flat**2, axis=0).reshape(n, m)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        log_comp = -0.5 * (maha + logdet + d * jnp.log(2.0 * jnp.pi))
        log_q = jax.scipy.special.logsumexp(log_comp, axis=1) - jnp.log(m)
        return z, log_q


def build_evaluator(data, Ms, dist_pc, shape):
    """Return ``single_eval(phys) -> scalar log-likelihood`` over present data.

    ``data`` is a tuple of present data containers. v1 handles RelativeAstromData;
    absent observables contribute nothing. The ``if astrom is not None`` check
    runs at build time (static), not under JAX trace, so it is JAX-safe.

    Args:
        data: Tuple of data containers (e.g. RelativeAstromData).
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        shape: An AbstractShapeParam instance (unused here, reserved for Plan 2).

    Returns:
        A pure function ``single_eval(phys) -> scalar`` where ``phys`` is a
        dict of scalar orbit parameters (as returned by ``to_physical`` for
        a single particle).
    """
    astrom = next((d for d in data if isinstance(d, RelativeAstromData)), None)

    def single_eval(phys):
        ll = 0.0
        if astrom is not None:
            ra, dec = predict_relative_astrometry(
                astrom.times,
                phys["a"],
                phys["e"],
                phys["cos_i"],
                phys["W"],
                phys["cos_w"],
                phys["sin_w"],
                phys["tp"],
                Ms,
                dist_pc,
            )
            ll = ll + loglike_relative_astrom(ra, dec, astrom)
        return ll

    return single_eval


def batched_loglike(single_eval, phys, n_particles, chunk_size):
    """Evaluate ``single_eval`` over ``n_particles`` via vmap inside lax.scan.

    ``phys`` is a dict of ``(n_particles,)`` arrays. Chunking keeps peak memory
    at ``O(chunk_size)`` per step; the per-particle time series stay in registers.

    Args:
        single_eval: Callable ``phys_scalar -> scalar`` as from ``build_evaluator``.
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


def grid_search(
    data,
    *,
    Ms,
    dist_pc,
    shape,
    sampler,
    log_T_range,
    e_max,
    n_particles=10**6,
    chunk_size=10**5,
    n_survivors=5000,
    key,
):
    """Two-stage adaptive-importance-sampling orbit grid-search.

    Stage 1 fills the unit cube with quasi-random samples, evaluates the
    log-likelihood for each, and selects the top survivors. Stage 2 draws
    a refined Gaussian-mixture proposal around those survivors, reweights by
    ``ll - log_q``, and returns a normalized ParticlePosterior.

    Args:
        data: Tuple of data containers (e.g. RelativeAstromData).
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        shape: An AbstractShapeParam instance.
        sampler: An AbstractGridStrategy instance.
        log_T_range: ``(log10_T_min, log10_T_max)`` in days.
        e_max: Maximum eccentricity vector component magnitude.
        n_particles: Total particles per stage. Must be divisible by chunk_size.
        chunk_size: Scan chunk size for memory control.
        n_survivors: Number of Stage-1 top particles passed to Stage 2.
        key: JAX PRNG key.

    Returns:
        ParticlePosterior with physical shape-parameter rows and normalized
        log-importance weights.
    """
    k1, k2 = jax.random.split(key)
    bounds = shape.default_bounds(log_T_range=log_T_range, e_max=e_max)
    ndim = len(bounds.names)
    ev = build_evaluator(data, Ms=Ms, dist_pc=dist_pc, shape=shape)
    phys_names = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")

    def physical_rows(u):
        phys = shape.to_physical(u, bounds, Ms=Ms)
        return phys, jnp.stack([phys[n] for n in phys_names], axis=1)

    u1 = sampler.stage1(k1, ndim, n_particles)
    phys1, _ = physical_rows(u1)
    ll1 = batched_loglike(ev, phys1, n_particles, chunk_size)

    order = jnp.argsort(ll1)[::-1][:n_survivors]
    u2, log_q = sampler.stage2(k2, u1[order], n_particles)
    phys2, rows2 = physical_rows(u2)
    ll2 = batched_loglike(ev, phys2, n_particles, chunk_size)
    log_w, log_total, _ = normalize_log_weights(ll2 - log_q)
    log_Z = log_total - jnp.log(n_particles)

    return SamplePosterior(
        samples=rows2, log_weights=log_w, evidence=log_Z, param_names=phys_names
    )


_CHART = EccentricityVectorChart()


def _ecc_square_disk_area(half_width):
    """Area of ``[-s, s]^2`` intersected with the open unit disk, ``s = half_width``."""
    s = min(float(half_width), 1.0)
    if s <= 1.0 / math.sqrt(2.0):
        return 4.0 * s * s
    x0 = math.sqrt(1.0 - s * s)
    return 4.0 * (s * x0 + 0.5 * (math.asin(s) - math.asin(x0)))


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


def chart_grid_search(
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
    disk_support=False,
    key,
):
    """Two-stage importance-sampling grid search weighted in chart coordinates.

    The target is the likelihood times a flat density against the chart reference
    measure over the box ``(logT, ex, ey, cos_i, W, tp_frac)``, built through orbix
    and weighted by ``log prior + chart log volume + proposal Jacobian + log
    likelihood - log q``. At identical box points this equals ``grid_search``'s
    weight plus one constant. The flat prior is normalized on its support, so the
    evidence is the mean likelihood over the support.

    Defaults reproduce ``grid_search``, restricted to bound orbits (``e < 1``):
    ``(ex, ey)`` fills the square ``[-e_max, e_max]^2`` and a clipped sampler puts
    atoms on the cube faces. Both are corrected opt-in: ``disk_support=True``
    restricts to ``e < e_max`` and ``AdaptiveImportanceSampler(clip=False)`` draws
    unclipped Gaussian-mixture samples, giving draws outside the cube zero weight.

    Args:
        data: Tuple of data containers (``RelativeAstromData`` is scored).
        star_mass_kg: Stellar mass (kg).
        dist_pc: Distance (parsec).
        sampler: An ``AbstractGridStrategy``.
        log_T_range: ``(log10_T_min, log10_T_max)`` in days.
        e_max: Half width of the square (radius of the disk) of the ``(ex, ey)`` box.
        n_particles: Particles per stage; divisible by ``chunk_size``.
        chunk_size: Scan chunk size for memory control.
        n_survivors: Stage-1 particles passed to Stage 2.
        disk_support: Restrict the eccentricity support to the disk ``e < e_max``.
        key: JAX PRNG key.

    Returns:
        ``SamplePosterior`` over ``(a, e, cos_i, W, cos_w, sin_w, tp)`` with
        normalized log weights and the importance-sampling log evidence.
    """
    k1, k2 = jax.random.split(key)
    bounds = chart_box_bounds(log_T_range=log_T_range, e_max=e_max)
    astrom = next((d for d in data if isinstance(d, RelativeAstromData)), None)
    mu = G * star_mass_kg
    area = _ecc_square_disk_area(e_max)
    if disk_support:
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
        if disk_support:
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
    )
