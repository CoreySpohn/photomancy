"""MAP + Laplace approximation for orbit fitting.

Provides a fast alternative to MCMC by computing the maximum a posteriori
(MAP) point and approximating the posterior with a multivariate Gaussian
centered at the MAP.

JIT-cache architecture
----------------------
The module maintains a ``_MODEL_CACHE`` keyed by the *static* model
configuration (prior ranges, eccentricity prior, data-channel flags).
On first call, ``initialize_model`` traces the NumPyro model once,
extracting constraint transforms and building a compilable potential
function.  Subsequent calls with different data (but same shapes, via
static padding) reuse the cached XLA compilation -- dropping warm-start
time from ~8 s to < 500 ms.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax

from photomancy.backends.laplace import laplace_covariance
from photomancy.orbit._numpyro_bridge import (
    _get_or_build_cached,
    _init_dict_to_z_flat,
    _pad_orbit_data,
)
from photomancy.orbit.init import find_init, find_init_top_k
from photomancy.posterior import GaussianPosterior, MixturePosterior

# ---------------------------------------------------------------------------
# MAP optimizer (Static-Arg JIT)
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit, static_argnames=("potential_factory", "unflatten", "n_steps")
)
def optimize_map_static(
    potential_factory: Callable,
    unflatten: Callable,
    z_init: jnp.ndarray,
    model_args: tuple,
    *,
    n_steps: int = 500,
    lr: float = 0.01,
    max_grad_norm: float = 10.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find MAP estimate using a cached potential factory.

    This function is JIT-compiled *once* per model structure.  Runtime
    data (``model_args``) are passed as arguments, not closed over,
    allowing the XLA graph to be reused.

    Args:
        potential_factory: Cached factory ``f(*model_args) -> potential_fn``.
        unflatten: Cached function ``flat -> dict``.
        z_init: Initial flat parameters.
        model_args: Tuple of data arguments for the potential factory.
        n_steps: Number of Adam steps.
        lr: Adam learning rate.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        (z_map, trajectory): Final parameters and optimization trajectory.
    """
    potential_fn_constrained = potential_factory(*model_args)

    def loss_fn(z):
        return potential_fn_constrained(unflatten(z))

    grad_fn = jax.grad(loss_fn)
    opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))

    def step(carry, _):
        z, state = carry
        g = grad_fn(z)
        g = jnp.where(jnp.isnan(g), 0.0, g)  # guard boundary NaNs before the update
        updates, state = opt.update(g, state, z)
        z = optax.apply_updates(z, updates)
        return (z, state), z

    (z_map, _), trajectory = jax.lax.scan(
        step, (z_init, opt.init(z_init)), None, length=n_steps
    )
    return z_map, trajectory


# ---------------------------------------------------------------------------
# Fast Covariance (Fisher / JVP)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("potential_factory", "unflatten"))
def fisher_covariance_jvp(
    potential_factory: Callable,
    unflatten: Callable,
    z_map: jnp.ndarray,
    model_args: tuple,
    *,
    min_eigenvalue: float = 1.0,
) -> jnp.ndarray:
    """Eigenvalue-clamped inverse Hessian of the orbit potential at ``z_map``.

    Reconstructs the constrained potential from the cached factory, then delegates the
    HVP -> eigh -> clamp -> inverse to the shared engine helper
    :func:`photomancy.backends.laplace.laplace_covariance` (forward-over-reverse AD,
    O(D) Hessian-vector products).
    """
    potential_fn_constrained = potential_factory(*model_args)

    def neg_logdensity(z):
        return potential_fn_constrained(unflatten(z))

    return laplace_covariance(neg_logdensity, z_map, min_eigenvalue)


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def map_laplace_fit(
    Ms: float,
    dist_pc: float,
    *,
    rv_data: Any | None = None,
    relative_astrom_data: Any | None = None,
    null_data: Any | None = None,
    imaging_data: Any | None = None,
    log_P_range: tuple[float, float] = (1.0, 4.0),
    log_Mp_range: tuple[float, float] = (-2.0, 4.0),
    log_Rp_range: tuple[float, float] = (-5.0, -2.5),
    log_Ag_range: tuple[float, float] = (-2.0, 0.0),
    ecc_prior: str = "kipping13",
    jitter_scale: float = 1e-10,
    init_vals: dict | None = None,
    seed: int = 0,
    n_steps: int = 500,
    min_eigenvalue: float = 1.0,
) -> GaussianPosterior:
    """One-call MAP + Laplace fit for a single planet.

    Uses the model cache for JIT-compilation reuse. On first call,
    traces the NumPyro model once. Subsequent calls with the same
    model structure (prior ranges, data channels) reuse the cached
    compilation.

    Args:
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        rv_data: An :class:`~photomancy.orbit.data.RVData`, or ``None``.
        relative_astrom_data: An :class:`~photomancy.orbit.data.RelativeAstromData`, or ``None``.
        null_data: A :class:`~photomancy.orbit.data.NullData`, or ``None``.
        imaging_data: An :class:`~photomancy.orbit.data.ImagingData`, or ``None``.
        log_P_range: ``(min, max)`` for ``log10(period/days)`` prior.
        log_Mp_range: ``(min, max)`` for ``log10(mass/M_earth)`` prior.
        log_Rp_range: ``(min, max)`` for ``log10(Rp/AU)`` prior.
        log_Ag_range: ``(min, max)`` for ``log10(geometric albedo)``.
        ecc_prior: Eccentricity prior name.
        jitter_scale: Scale for HalfNormal jitter prior.
        init_vals: Init dict from :func:`find_init`. If ``None``,
            NumPyro's default random init is used.
        seed: PRNG seed for sampling.
        n_steps: Number of Adam optimiser steps.  Default 500.
        min_eigenvalue: Eigenvalue floor for Hessian regularisation.

    Returns:
        A :class:`~photomancy.posterior.GaussianPosterior` with the MAP mean, the
        Laplace covariance, and the Laplace log-evidence.
    """
    has_rv = rv_data is not None
    has_relative_astrom = relative_astrom_data is not None
    has_null = null_data is not None
    has_imaging = imaging_data is not None

    # 1. Get or build cached model
    cached = _get_or_build_cached(
        has_rv=has_rv,
        has_relative_astrom=has_relative_astrom,
        has_null=has_null,
        has_imaging=has_imaging,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
        seed=seed,
    )

    rv_data, relative_astrom_data, null_data, imaging_data = _pad_orbit_data(
        rv_data, relative_astrom_data, null_data, imaging_data
    )

    model_args = (Ms, dist_pc, rv_data, relative_astrom_data, null_data, imaging_data)

    # 3. Get initial z-vector
    if init_vals is not None:
        z_init_flat = _init_dict_to_z_flat(
            init_vals, cached["z_template"], cached["inv_transforms"]
        )
    else:
        # Use default from template
        z_init_flat, _ = jax.flatten_util.ravel_pytree(cached["z_template"])

    # 4. MAP Optimization (Static-Arg JIT)
    z_map, _trajectory = optimize_map_static(
        cached["potential_fn_factory"],
        cached["unflatten"],
        z_init_flat,
        model_args,
        n_steps=n_steps,
    )

    # 5. Covariance (Fisher / JVP)
    cov = fisher_covariance_jvp(
        cached["potential_fn_factory"],
        cached["unflatten"],
        z_map,
        model_args,
        min_eigenvalue=min_eigenvalue,
    )

    # Laplace log-evidence at the MAP (same convention as the mixture fit).
    p_fn = cached["potential_fn_factory"](*model_args)
    loss = p_fn(cached["unflatten"](z_map))
    d = z_map.shape[0]
    _, logdet = jnp.linalg.slogdet(cov)
    log_z = -loss + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * logdet

    return GaussianPosterior(mean=z_map, cov=cov, evidence=log_z)


# ---------------------------------------------------------------------------
# Multi-start MAP + Laplace mixture fit
# ---------------------------------------------------------------------------


def map_laplace_mixture_fit(
    Ms: float,
    dist_pc: float,
    *,
    rv_data: Any | None = None,
    relative_astrom_data: Any | None = None,
    null_data: Any | None = None,
    imaging_data: Any | None = None,
    log_P_range: tuple[float, float] = (1.0, 4.0),
    log_Mp_range: tuple[float, float] = (-2.0, 4.0),
    log_Rp_range: tuple[float, float] = (-5.0, -2.5),
    log_Ag_range: tuple[float, float] = (-2.0, 0.0),
    ecc_prior: str = "kipping13",
    jitter_scale: float = 1e-10,
    k: int = 5,
    init_list: list[dict] | None = None,
    use_top_k_init: bool = False,
    seed: int = 0,
    n_steps: int = 500,
    min_eigenvalue: float = 1.0,
) -> MixturePosterior:
    """Multi-start MAP + Laplace fit returning an evidence-weighted mixture.

    Uses the model cache for JIT-compilation reuse.

    Args:
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        rv_data: An :class:`~photomancy.orbit.data.RVData`, or ``None``.
        relative_astrom_data: An :class:`~photomancy.orbit.data.RelativeAstromData`, or ``None``.
        null_data: A :class:`~photomancy.orbit.data.NullData`, or ``None``.
        imaging_data: An :class:`~photomancy.orbit.data.ImagingData`, or ``None``.
        log_P_range: ``(min, max)`` for ``log10(period/days)`` prior.
        log_Mp_range: ``(min, max)`` for ``log10(mass/M_earth)`` prior.
        log_Rp_range: ``(min, max)`` for ``log10(Rp/AU)`` prior.
        log_Ag_range: ``(min, max)`` for ``log10(geometric albedo)``.
        ecc_prior: Eccentricity prior name.
        jitter_scale: Scale for HalfNormal jitter prior.
        k: Number of initial conditions to try.  Default 5.
        init_list: Optional list of K init dicts. If ``None``,
            initialisation is determined by ``use_top_k_init``.
        use_top_k_init: If ``True``, use :func:`find_init_top_k` to
            find the K globally-best TI maxima across the full period
            grid (better alias coverage).  If ``False`` (default),
            subdivide the period range into K sub-ranges.
        seed: PRNG seed.
        n_steps: Number of Adam optimiser steps per mode.
        min_eigenvalue: Eigenvalue floor for Hessian regularisation.

    Returns:
        A :class:`~photomancy.posterior.MixturePosterior` with one Gaussian mode per
        start, weighted by per-mode Laplace log-evidence.
    """
    has_rv = rv_data is not None
    has_relative_astrom = relative_astrom_data is not None
    has_null = null_data is not None
    has_imaging = imaging_data is not None

    # 1. Get or build cached model
    cached = _get_or_build_cached(
        has_rv=has_rv,
        has_relative_astrom=has_relative_astrom,
        has_null=has_null,
        has_imaging=has_imaging,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
        seed=seed,
    )

    rv_data, relative_astrom_data, null_data, imaging_data = _pad_orbit_data(
        rv_data, relative_astrom_data, null_data, imaging_data
    )

    # 3. Build model args
    model_args = (Ms, dist_pc, rv_data, relative_astrom_data, null_data, imaging_data)

    # 3. Get K initial conditions
    if init_list is None:
        if relative_astrom_data is None:
            raise ValueError(
                "relative_astrom_data is required for automatic init; "
                "pass init_list explicitly for non-astrometry fits."
            )

        if use_top_k_init:
            init_list = find_init_top_k(
                relative_astrom_data,
                Ms,
                dist_pc,
                k=k,
                log_T_range=log_P_range,
            )
        else:
            log_lo, log_hi = log_P_range
            edges = jnp.linspace(log_lo, log_hi, k + 1)
            init_list = []
            for j in range(k):
                sub_range = (float(edges[j]), float(edges[j + 1]))
                init_list.append(
                    find_init(
                        relative_astrom_data,
                        Ms,
                        dist_pc,
                        log_T_range=sub_range,
                        n_log_T=30,
                    )
                )

    # 4. Convert init dicts -> z-vectors
    z_inits = [
        _init_dict_to_z_flat(d, cached["z_template"], cached["inv_transforms"])
        for d in init_list
    ]

    # 5. Stack z_inits and vmap the optimization
    z_init_batch = jnp.stack(z_inits)  # (K, D)

    # We vmap over z_init_batch, broadcasting other args
    def fit_single(z_init):
        # Optimize
        z_map, _ = optimize_map_static(
            cached["potential_fn_factory"],
            cached["unflatten"],
            z_init,
            model_args,
            n_steps=n_steps,
        )
        # Covariance
        cov = fisher_covariance_jvp(
            cached["potential_fn_factory"],
            cached["unflatten"],
            z_map,
            model_args,
            min_eigenvalue=min_eigenvalue,
        )
        # Compute final loss (negative log posterior)
        # Reconstruct potential for evaluation
        p_fn = cached["potential_fn_factory"](*model_args)
        loss = p_fn(cached["unflatten"](z_map))

        return z_map, cov, loss

    # MAP over the batch of initial conditions
    z_maps, covs, losses = jax.vmap(fit_single)(z_init_batch)

    # Per-mode Laplace log-evidence (unnormalized log Z_k); MixturePosterior
    # derives the mode weights from these.
    d = z_maps.shape[-1]
    _, log_dets = jax.vmap(jnp.linalg.slogdet)(covs)
    log_ev = -losses + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * log_dets

    return MixturePosterior(means=z_maps, covs=covs, log_evidences=log_ev)
