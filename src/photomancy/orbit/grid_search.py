"""GPU-parallel orbit grid-search (adaptive importance sampling).

Discovery tier of photomancy.orbit: enumerate orbits consistent with sparse data
and return weighted particles.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from orbix.equations import period_to_sma
from orbix.utils.quasi_random import roberts_sequence

from photomancy.orbit.data import AstromData
from photomancy.orbit.forward import predict_astrometry
from photomancy.orbit.likelihoods import loglike_astrom


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
        names = ("logT", "ex", "ey", "cos_i", "W", "tp_frac")
        low = jnp.array([log_T_range[0], -e_max, -e_max, -1.0, 0.0, 0.0])
        high = jnp.array([log_T_range[1], e_max, e_max, 1.0, 2.0 * jnp.pi, 1.0])
        return ParamBounds(low=low, high=high, names=names)

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
        draws a balanced mixture. Samples are clipped to the unit cube.

        Args:
            key: JAX PRNG key.
            survivors: Best unit-cube points from Stage 1, shape ``(m, d)``.
            n: Number of new samples to draw.

        Returns:
            Tuple ``(z, log_q)`` where ``z`` has shape ``(n, d)`` clipped to
            ``[0, 1)`` and ``log_q`` has shape ``(n,)`` (mixture log-density).
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

    ``data`` is a tuple of present data containers. v1 handles AstromData;
    absent observables contribute nothing. The ``if astrom is not None`` check
    runs at build time (static), not under JAX trace, so it is JAX-safe.

    Args:
        data: Tuple of data containers (e.g. AstromData).
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        shape: An AbstractShapeParam instance (unused here, reserved for Plan 2).

    Returns:
        A pure function ``single_eval(phys) -> scalar`` where ``phys`` is a
        dict of scalar orbit parameters (as returned by ``to_physical`` for
        a single particle).
    """
    astrom = next((d for d in data if isinstance(d, AstromData)), None)

    def single_eval(phys):
        ll = 0.0
        if astrom is not None:
            ra, dec = predict_astrometry(
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
            ll = ll + loglike_astrom(ra, dec, astrom)
        return ll

    return single_eval


class ParticlePosterior(eqx.Module):
    """Weighted particle approximation to the orbit posterior.

    Attributes:
        particles: Physical-parameter rows, shape ``(n, d)``.
        log_weights: Normalized log-importance weights, shape ``(n,)``.
        param_names: Ordered parameter names corresponding to ``particles`` columns.
    """

    particles: jnp.ndarray
    log_weights: jnp.ndarray
    param_names: tuple = eqx.field(static=True)

    def sample(self, key, n=2000):
        """Inverse-CDF sampling-importance-resampling: draw ``n`` particles by weight.

        Uses a cumulative-weight inverse-CDF lookup, which costs
        ``O(n_particles + n)`` memory. (A ``jax.random.categorical`` draw would
        materialize an ``(n, n_particles)`` array and blow up for large particle
        counts.)

        Args:
            key: JAX PRNG key.
            n: Number of posterior draws to return.

        Returns:
            Dict mapping each name in ``param_names`` to a ``(n,)`` array.
        """
        cdf = jnp.cumsum(jax.nn.softmax(self.log_weights))
        u = jax.random.uniform(key, (n,))
        idx = jnp.clip(
            jnp.searchsorted(cdf, u, side="right"), 0, self.particles.shape[0] - 1
        )
        drawn = self.particles[idx]
        return {name: drawn[:, i] for i, name in enumerate(self.param_names)}


def batched_loglike(single_eval, phys, n_particles, chunk_size):
    """Evaluate ``single_eval`` over ``n_particles`` via vmap inside lax.scan.

    ``phys`` is a dict of ``(n_particles,)`` arrays. Chunking keeps peak memory
    at ``O(chunk_size)`` per step; the per-particle time series stay in registers.

    Args:
        single_eval: Callable ``phys_scalar -> scalar`` as from ``build_evaluator``.
        phys: Dict of arrays, each of shape ``(n_particles,)``.
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
        k: v[: n_chunks * chunk_size].reshape((n_chunks, chunk_size))
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
        data: Tuple of data containers (e.g. AstromData).
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
    log_w = ll2 - log_q
    log_w = log_w - jax.scipy.special.logsumexp(log_w)

    return ParticlePosterior(particles=rows2, log_weights=log_w, param_names=phys_names)
