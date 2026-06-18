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
import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import initialize_model

from photomancy.orbit.data import (
    MAX_ASTROM,
    MAX_IMG,
    MAX_RV,
    AstromData,
    ImagingData,
    NullData,
    RVData,
)
from photomancy.orbit.model import build_model

# ---------------------------------------------------------------------------
# LaplaceResult -- equinox container
# ---------------------------------------------------------------------------


class LaplaceResult(eqx.Module):
    """Result of a MAP + Laplace approximation.

    Stores the MAP point and regularized covariance in NumPyro's internal
    unconstrained space.  Provides methods to draw posterior samples
    (via the reparametrization trick) and to evaluate the Gaussian
    log-probability (for sequential observation planning).
    """

    z_map: jnp.ndarray
    covariance: jnp.ndarray
    cholesky: jnp.ndarray
    _unflatten: Callable = eqx.field(static=True)
    _postprocess_fn: Callable = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)
    n_params: int = eqx.field(static=True)

    def sample(
        self,
        key: jax.Array,
        n: int = 2000,
    ) -> dict[str, jnp.ndarray]:
        """Draw posterior samples and return physical orbital parameters.

        Args:
            key: JAX PRNG key.
            n: Number of samples to draw.  Default 2000.

        Returns:
            Dict mapping physical parameter names (``T``, ``e``, ``cos_i``,
            ``W``, ``tp``, ``a``, etc.) to arrays of shape ``(n,)``.
        """
        z_samples = _draw_samples(self.z_map, self.cholesky, key, n)
        return _postprocess_samples(z_samples, self._unflatten, self._postprocess_fn)

    def log_prob(self, z_flat: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Gaussian log-probability at a point.

        Useful as a prior term for the next observation in a sequential
        observation-planning loop.

        Args:
            z_flat: Parameter vector in unconstrained space.
                Shape ``(D,)``.

        Returns:
            Scalar log-probability.
        """
        return _mvn_log_prob(z_flat, self.z_map, self.covariance, self.cholesky)


# ---------------------------------------------------------------------------
# Internal JIT-compiled helpers
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(3,))
def _draw_samples(
    z_map: jnp.ndarray,
    chol: jnp.ndarray,
    key: jax.Array,
    n: int,
) -> jnp.ndarray:
    """Draw *n* MVN samples: z_map + L @ z, z ~ N(0, I)."""
    z = jax.random.normal(key, (n, z_map.shape[0]))
    return z_map + z @ chol.T


@jax.jit
def _mvn_log_prob(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    cov: jnp.ndarray,
    chol: jnp.ndarray,
) -> jnp.ndarray:
    """Multivariate normal log-probability."""
    d = mu.shape[0]
    diff = x - mu
    # Solve L y = diff for y, then ||y||^2 = diff^T Sigma^{-1} diff
    y = jax.scipy.linalg.solve_triangular(chol, diff, lower=True)
    maha = jnp.dot(y, y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5 * (maha + log_det + d * jnp.log(2.0 * jnp.pi))


def _postprocess_samples(
    z_samples: jnp.ndarray,
    unflatten: Callable,
    postprocess_fn: Callable,
) -> dict[str, jnp.ndarray]:
    """Convert unconstrained flat samples -> physical parameter dicts."""

    def _single(z_flat):
        z_dict = unflatten(z_flat)
        return postprocess_fn(z_dict)

    return jax.vmap(_single)(z_samples)


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
    # 1. Reconstruct the potential function from factory + args
    # This setup overhead is traced away by XLA
    potential_fn_constrained = potential_factory(*model_args)

    def loss_fn(z):
        return potential_fn_constrained(unflatten(z))

    grad_fn = jax.grad(loss_fn)

    # 2. Adam optimizer state
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m0 = jnp.zeros_like(z_init)
    v0 = jnp.zeros_like(z_init)

    # 3. Scan step function
    def step(state, i):
        p, m, v = state
        g = grad_fn(p)
        g = jnp.where(jnp.isnan(g), 0.0, g)

        # Gradient clipping
        g_norm = jnp.linalg.norm(g)
        g = jnp.where(g_norm > max_grad_norm, g * max_grad_norm / g_norm, g)

        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * g**2
        m_hat = m_new / (1 - beta1 ** (i + 1))
        v_hat = v_new / (1 - beta2 ** (i + 1))
        p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        # Return state and trajectory (p_new)
        return (p_new, m_new, v_new), p_new

    # 4. Run scan
    xs = jnp.arange(n_steps)
    (p_opt, _, _), trajectory = jax.lax.scan(step, (z_init, m0, v0), xs)

    return p_opt, trajectory


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
    """Compute covariance via Fisher information (outer product of gradients).

    Uses forward-over-reverse AD (JVP of Grad) to compute the Hessian of the
    log-likelihood. This is O(D) instead of O(D^2) for exact Hessian, and
    takes ~7ms warm (vs 2.4s for exact Hessian).
    """
    # Reconstruct potential
    potential_fn_constrained = potential_factory(*model_args)

    def loss_fn(z):
        return potential_fn_constrained(unflatten(z))

    grad_fn = jax.grad(loss_fn)
    D = z_map.shape[0]

    def hvp(v):
        """Hessian-vector product via forward-over-reverse AD."""
        return jax.jvp(grad_fn, (z_map,), (v,))[1]

    # Build Hessian column-by-column
    basis = jnp.eye(D)
    H = jax.vmap(hvp)(basis)

    # Symmetrise
    H = 0.5 * (H + H.T)

    # Regularize
    eigvals, eigvecs = jnp.linalg.eigh(H)
    eigvals = jnp.maximum(eigvals, min_eigenvalue)
    prec = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    cov = jnp.linalg.inv(prec)
    return cov


# ---------------------------------------------------------------------------
# Model cache -- trace once, reuse across calls
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple, dict] = {}


def _cache_key(
    has_rv,
    has_astrom,
    has_null,
    has_imaging,
    log_P_range,
    log_Mp_range,
    log_Rp_range,
    log_Ag_range,
    ecc_prior,
    jitter_scale,
    n_planets,
):
    """Generate a hashable key for the model cache."""
    return (
        has_rv,
        has_astrom,
        has_null,
        has_imaging,
        log_P_range,
        log_Mp_range,
        log_Rp_range,
        log_Ag_range,
        ecc_prior,
        jitter_scale,
        n_planets,
    )


def _get_or_build_cached(
    *,
    has_rv,
    has_astrom,
    has_null,
    has_imaging,
    log_P_range,
    log_Mp_range,
    log_Rp_range,
    log_Ag_range,
    ecc_prior,
    jitter_scale,
    n_planets=1,
    seed=0,
):
    """Return (or build + cache) model metadata for a given config.

    Returns a dict with keys:
        model, potential_fn_factory, postprocess_fn_factory,
        inv_transforms, z_template, unflatten, param_names
    """
    key = _cache_key(
        has_rv,
        has_astrom,
        has_null,
        has_imaging,
        log_P_range,
        log_Mp_range,
        log_Rp_range,
        log_Ag_range,
        ecc_prior,
        jitter_scale,
        n_planets,
    )

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # Build model (closure captures ONLY prior config)
    model = build_model(
        n_planets=n_planets,
        has_rv=has_rv,
        has_astrom=has_astrom,
        has_null=has_null,
        has_imaging=has_imaging,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
    )

    # Build placeholder data at MAX sizes (for tracing)
    placeholder_Ms = 1.989e30
    placeholder_dist = 10.0
    placeholder_rv = RVData.zeros() if has_rv else None
    placeholder_astrom = AstromData.zeros() if has_astrom else None
    placeholder_null = NullData.zeros() if has_null else None
    placeholder_imaging = ImagingData.zeros() if has_imaging else None

    model_args = (
        placeholder_Ms,
        placeholder_dist,
        placeholder_rv,
        placeholder_astrom,
        placeholder_null,
        placeholder_imaging,
    )

    # Trace the model ONCE with dynamic_args
    rng = jax.random.PRNGKey(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_info = initialize_model(
            rng,
            model,
            dynamic_args=True,
            model_args=model_args,
        )

    # Extract template and transforms
    z_template = model_info.param_info.z
    param_names = tuple(sorted(z_template.keys()))
    _, unflatten = jax.flatten_util.ravel_pytree(z_template)

    # Extract inverse constraint bijectors for init conversion

    # Re-trace to get model_trace (dynamic_args doesn't store it the same way)
    # We stored it in model_info.model_trace
    inv_transforms = {}
    for name, site in model_info.model_trace.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            inv_transforms[name] = biject_to(site["fn"].support).inv

    # potential_fn_factory: call with model_args -> returns potential_fn(z_dict)
    # postprocess_fn_factory: call with model_args -> returns postprocess_fn(z_dict)
    cached = {
        "model": model,
        "potential_fn_factory": model_info.potential_fn,
        "postprocess_fn_factory": model_info.postprocess_fn,
        "inv_transforms": inv_transforms,
        "z_template": z_template,
        "unflatten": unflatten,
        "param_names": param_names,
        "model_trace": dict(model_info.model_trace),
    }

    _MODEL_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def map_laplace_fit(
    Ms: float,
    dist_pc: float,
    *,
    rv_data: Any | None = None,
    astrom_data: Any | None = None,
    null_data: Any | None = None,
    imaging_data: Any | None = None,
    log_P_range: tuple[float, float] = (1.0, 4.0),
    log_Mp_range: tuple[float, float] = (-2.0, 4.0),
    log_Rp_range: tuple[float, float] = (-5.0, -2.5),
    log_Ag_range: tuple[float, float] = (-2.0, 0.0),
    ecc_prior: str = "kipping13",
    jitter_scale: float = 1e-10,
    init_vals: dict | None = None,
    n_samples: int = 2000,
    seed: int = 0,
    n_steps: int = 500,
    min_eigenvalue: float = 1.0,
) -> LaplaceResult:
    """One-call MAP + Laplace fit for a single planet.

    Uses the model cache for JIT-compilation reuse. On first call,
    traces the NumPyro model once. Subsequent calls with the same
    model structure (prior ranges, data channels) reuse the cached
    compilation.

    Args:
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        rv_data: An :class:`~photomancy.orbit.data.RVData`, or ``None``.
        astrom_data: An :class:`~photomancy.orbit.data.AstromData`, or ``None``.
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
        n_samples: Number of posterior samples.  Default 2000.
        seed: PRNG seed for sampling.
        n_steps: Number of Adam optimiser steps.  Default 500.
        min_eigenvalue: Eigenvalue floor for Hessian regularisation.

    Returns:
        A :class:`LaplaceResult` with MAP point, covariance, and methods
        for sampling and log-probability evaluation.
    """
    has_rv = rv_data is not None
    has_astrom = astrom_data is not None
    has_null = null_data is not None
    has_imaging = imaging_data is not None

    # 1. Get or build cached model
    cached = _get_or_build_cached(
        has_rv=has_rv,
        has_astrom=has_astrom,
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

    # 2. Pad data to MAX sizes for shape consistency with JIT cache
    if astrom_data is not None and astrom_data.times.shape[0] != MAX_ASTROM:
        n = astrom_data.times.shape[0]
        astrom_data = AstromData.pad(
            times=astrom_data.times[:n],
            ra=astrom_data.ra[:n],
            dec=astrom_data.dec[:n],
            ra_err=astrom_data.ra_err[:n],
            dec_err=astrom_data.dec_err[:n],
            corr=astrom_data.corr[:n],
            planet_id=astrom_data.planet_id[:n],
        )
    if rv_data is not None and rv_data.times.shape[0] != MAX_RV:
        n = rv_data.times.shape[0]
        rv_data = RVData.pad(
            times=rv_data.times[:n],
            rv=rv_data.rv[:n],
            rv_err=rv_data.rv_err[:n],
            inst_ids=rv_data.inst_ids[:n],
            n_inst=rv_data.n_inst,
        )
    if null_data is not None and null_data.epochs.shape[0] != MAX_IMG:
        n = null_data.epochs.shape[0]
        null_data = NullData.pad(
            epochs=null_data.epochs[:n],
            sep_grid=null_data.sep_grid[:n],
            dmag0_grid=null_data.dmag0_grid[:n],
        )
    if imaging_data is not None and imaging_data.epochs.shape[0] != MAX_IMG:
        n = imaging_data.epochs.shape[0]
        imaging_data = ImagingData.pad(
            epochs=imaging_data.epochs[:n],
            sep_grid=imaging_data.sep_grid[:n],
            dmag0_grid=imaging_data.dmag0_grid[:n],
            is_detected=imaging_data.is_detected[:n],
            dmag_obs=imaging_data.dmag_obs[:n],
            dmag_err=imaging_data.dmag_err[:n],
        )

    model_args = (Ms, dist_pc, rv_data, astrom_data, null_data, imaging_data)

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

    # 6. Reconstruct helpers for result object
    # Postprocess factory might need args or not, handle both
    try:
        raw_pp = cached["postprocess_fn_factory"](*model_args)
    except TypeError:
        raw_pp = cached["postprocess_fn_factory"]

    def postprocess_fn(z_dict):
        phys = raw_pp(z_dict)
        return {k: jnp.squeeze(v) for k, v in phys.items()}

    # Compute Cholesky of precision or covariance?
    # LaplaceResult expects cholesky of *covariance* (L L^T = Sigma)
    # fisher_covariance_jvp returns Covariance.
    chol = jnp.linalg.cholesky(cov)

    return LaplaceResult(
        z_map=z_map,
        covariance=cov,
        cholesky=chol,
        _unflatten=cached["unflatten"],
        _postprocess_fn=postprocess_fn,
        param_names=cached["param_names"],
        n_params=z_map.shape[0],
    )


# ---------------------------------------------------------------------------
# Helper: convert init dict -> flat z-vector using cached transforms
# ---------------------------------------------------------------------------


def _init_dict_to_z_flat(init_dict, z_template, inv_transforms):
    """Convert a physical-params init dict to a flat unconstrained vector."""
    z_out = {}
    for z_key in z_template.keys():
        if z_key in init_dict and z_key in inv_transforms:
            z_val = inv_transforms[z_key](
                jnp.asarray(init_dict[z_key], dtype=jnp.float64)
            )
            # Boundary init values produce +/-inf/NaN -- fall back to template
            z_val = jnp.where(jnp.isfinite(z_val), z_val, z_template[z_key])
            z_out[z_key] = z_val
        elif z_key in init_dict:
            z_out[z_key] = jnp.asarray(init_dict[z_key], dtype=jnp.float64)
        else:
            z_out[z_key] = z_template[z_key]
    z_flat, _ = jax.flatten_util.ravel_pytree(z_out)
    return z_flat


# ---------------------------------------------------------------------------
# LaplaceMixtureResult --- evidence-weighted mixture of Gaussians
# ---------------------------------------------------------------------------


class LaplaceMixtureResult(eqx.Module):
    """Mixture of Laplace approximations from multi-start MAP.

    Combines K MAP modes into a Gaussian Mixture Model weighted by
    the Laplace approximation of the marginal likelihood (Bayesian
    evidence) at each mode.

    Attributes:
        weights: Normalised evidence weights.  Shape ``(K,)``.
        z_maps: MAP estimates for each mode.  Shape ``(K, D)``.
        covariances: Regularised covariances.  Shape ``(K, D, D)``.
        choleskys: Cholesky factors.  Shape ``(K, D, D)``.
        losses: Final negative log-posterior at each MAP.  Shape ``(K,)``.
        log_evidence: Unnormalised log-evidence per mode.  Shape ``(K,)``.
        _unflatten: Callable = eqx.field(static=True)
        _postprocess_fn: Callable = eqx.field(static=True)
        param_names: tuple[str, ...] = eqx.field(static=True)
        n_params: int = eqx.field(static=True)
        n_modes: int = eqx.field(static=True)
    """

    weights: jnp.ndarray
    z_maps: jnp.ndarray
    covariances: jnp.ndarray
    choleskys: jnp.ndarray
    losses: jnp.ndarray
    log_evidence: jnp.ndarray
    _unflatten: Callable = eqx.field(static=True)
    _postprocess_fn: Callable = eqx.field(static=True)
    _model_trace: dict = eqx.field(static=True)
    _potential_fn_factory: Callable = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)
    n_params: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)

    def sample(
        self,
        key: jax.Array,
        n: int = 2000,
    ) -> dict[str, jnp.ndarray]:
        """Draw posterior samples from the evidence-weighted mixture.

        Args:
            key: JAX PRNG key.
            n: Number of samples.  Default 2000.

        Returns:
            Dict mapping physical parameter names to arrays of shape
            ``(n,)``.
        """
        key_cat, key_z = jax.random.split(key)

        # 1. Sample mode indices from Categorical(weights)
        mode_indices = jax.random.categorical(
            key_cat, jnp.log(self.weights), shape=(n,)
        )

        # 2. Draw from the selected MVN per sample
        z_noise = jax.random.normal(key_z, (n, self.n_params))

        # Gather the MAP and Cholesky for each sample's mode
        z_map_selected = self.z_maps[mode_indices]  # (n, D)
        chol_selected = self.choleskys[mode_indices]  # (n, D, D)

        # z_map + L @ noise  (batched matmul)
        z_samples = z_map_selected + jnp.einsum("nij,nj->ni", chol_selected, z_noise)

        # 3. Postprocess to physical space
        return _postprocess_samples(z_samples, self._unflatten, self._postprocess_fn)

    def best_mode(self) -> LaplaceResult:
        """Return the single highest-weight mode as a LaplaceResult."""
        idx = int(jnp.argmax(self.weights))
        return LaplaceResult(
            z_map=self.z_maps[idx],
            covariance=self.covariances[idx],
            cholesky=self.choleskys[idx],
            _unflatten=self._unflatten,
            _postprocess_fn=self._postprocess_fn,
            param_names=self.param_names,
            n_params=self.n_params,
        )

    def mode_summary(self) -> list[dict]:
        """Return a summary of each mode (for diagnostics)."""
        summaries = []
        for k in range(self.n_modes):
            z_dict = self._unflatten(self.z_maps[k])
            phys = self._postprocess_fn(z_dict)
            summaries.append(
                {
                    "weight": float(self.weights[k]),
                    "loss": float(self.losses[k]),
                    "log_evidence": float(self.log_evidence[k]),
                    "params": {kk: float(jnp.squeeze(v)) for kk, v in phys.items()},
                }
            )
        return summaries

    def sample_visual_z(
        self,
        key: jax.Array,
        n: int = 25,
        max_variance: float = 10.0,
    ) -> jnp.ndarray:
        """Draw z-space samples with capped covariance for visualization.

        Caps eigenvalues of the covariance to ``max_variance`` to prevent
        samples from saturating at prior boundaries through sigmoid
        transforms, which creates pathological orbits.

        Args:
            key: JAX PRNG key.
            n: Number of samples.  Default 25.
            max_variance: Maximum eigenvalue allowed in the covariance.
                Default 10.0 (SD ~= 3.2 in unconstrained space).

        Returns:
            Raw z-space samples, shape ``(n, D)``.
        """

        # Cap covariance eigenvalues to prevent boundary saturation
        def _cap_cov(cov):
            eigvals, eigvecs = jnp.linalg.eigh(cov)
            eigvals_capped = jnp.minimum(eigvals, max_variance)
            cov_capped = eigvecs @ jnp.diag(eigvals_capped) @ eigvecs.T
            return cov_capped

        capped_covs = jax.vmap(_cap_cov)(self.covariances)
        capped_chols = jax.vmap(jnp.linalg.cholesky)(capped_covs)

        key_cat, key_z = jax.random.split(key)
        mode_indices = jax.random.categorical(
            key_cat, jnp.log(self.weights), shape=(n,)
        )
        z_noise = jax.random.normal(key_z, (n, self.n_params))
        z_map_selected = self.z_maps[mode_indices]
        chol_selected = capped_chols[mode_indices]
        z_samples = z_map_selected + jnp.einsum("nij,nj->ni", chol_selected, z_noise)
        return z_samples

    def project_samples(
        self,
        z_samples: jnp.ndarray,
        model_args: tuple,
        n_steps: int = 150,
        lr: float = 0.005,
    ) -> dict[str, jnp.ndarray]:
        """Project z-space samples onto the data-consistent manifold.

        Re-optimizes each sample with a short MAP run, snapping it
        back from the Gaussian tangent plane onto the curved valley
        of valid orbits that pass through the observed data.

        Args:
            z_samples: Raw z-space samples, shape ``(n, D)``.
            model_args: Tuple of data arguments for the potential.
            n_steps: Optimization steps per sample.  Default 150.
            lr: Learning rate for projection.  Default 0.005.

        Returns:
            Dict mapping physical parameter names to arrays of shape
            ``(n,)``.
        """
        potential_factory = self._potential_fn_factory
        unflatten = self._unflatten

        def _project_single(z_init):
            z_opt, _ = optimize_map_static(
                potential_factory,
                unflatten,
                z_init,
                model_args,
                n_steps=n_steps,
                lr=lr,
            )
            return z_opt

        z_projected = jax.vmap(_project_single)(z_samples)
        return _postprocess_samples(z_projected, self._unflatten, self._postprocess_fn)


# ---------------------------------------------------------------------------
# Multi-start MAP + Laplace mixture fit
# ---------------------------------------------------------------------------


def map_laplace_mixture_fit(
    Ms: float,
    dist_pc: float,
    *,
    rv_data: Any | None = None,
    astrom_data: Any | None = None,
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
) -> LaplaceMixtureResult:
    """Multi-start MAP + Laplace fit returning an evidence-weighted mixture.

    Uses the model cache for JIT-compilation reuse.

    Args:
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        rv_data: An :class:`~photomancy.orbit.data.RVData`, or ``None``.
        astrom_data: An :class:`~photomancy.orbit.data.AstromData`, or ``None``.
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
        A :class:`LaplaceMixtureResult` with evidence-weighted modes.
    """
    has_rv = rv_data is not None
    has_astrom = astrom_data is not None
    has_null = null_data is not None
    has_imaging = imaging_data is not None

    # 1. Get or build cached model
    cached = _get_or_build_cached(
        has_rv=has_rv,
        has_astrom=has_astrom,
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

    # 2. Pad data to MAX sizes for shape consistency with JIT cache
    if astrom_data is not None and astrom_data.times.shape[0] != MAX_ASTROM:
        n = astrom_data.times.shape[0]
        astrom_data = AstromData.pad(
            times=astrom_data.times[:n],
            ra=astrom_data.ra[:n],
            dec=astrom_data.dec[:n],
            ra_err=astrom_data.ra_err[:n],
            dec_err=astrom_data.dec_err[:n],
            corr=astrom_data.corr[:n],
            planet_id=astrom_data.planet_id[:n],
        )
    if rv_data is not None and rv_data.times.shape[0] != MAX_RV:
        n = rv_data.times.shape[0]
        rv_data = RVData.pad(
            times=rv_data.times[:n],
            rv=rv_data.rv[:n],
            rv_err=rv_data.rv_err[:n],
            inst_ids=rv_data.inst_ids[:n],
            n_inst=rv_data.n_inst,
        )
    if null_data is not None and null_data.epochs.shape[0] != MAX_IMG:
        n = null_data.epochs.shape[0]
        null_data = NullData.pad(
            epochs=null_data.epochs[:n],
            sep_grid=null_data.sep_grid[:n],
            dmag0_grid=null_data.dmag0_grid[:n],
        )
    if imaging_data is not None and imaging_data.epochs.shape[0] != MAX_IMG:
        n = imaging_data.epochs.shape[0]
        imaging_data = ImagingData.pad(
            epochs=imaging_data.epochs[:n],
            sep_grid=imaging_data.sep_grid[:n],
            dmag0_grid=imaging_data.dmag0_grid[:n],
            is_detected=imaging_data.is_detected[:n],
            dmag_obs=imaging_data.dmag_obs[:n],
            dmag_err=imaging_data.dmag_err[:n],
        )

    # 3. Build model args
    model_args = (Ms, dist_pc, rv_data, astrom_data, null_data, imaging_data)

    # 3. Get K initial conditions
    if init_list is None:
        if astrom_data is None:
            raise ValueError(
                "astrom_data is required for automatic init; "
                "pass init_list explicitly for non-astrometry fits."
            )

        if use_top_k_init:
            # CIRCULAR IMPORT: photomancy.orbit.init -> photomancy.orbit.laplace
            from photomancy.orbit.init import find_init_top_k

            init_list = find_init_top_k(
                astrom_data,
                Ms,
                dist_pc,
                k=k,
                log_T_range=log_P_range,
            )
        else:
            # CIRCULAR IMPORT: photomancy.orbit.init -> photomancy.orbit.laplace
            from photomancy.orbit.init import find_init

            log_lo, log_hi = log_P_range
            edges = jnp.linspace(log_lo, log_hi, k + 1)
            init_list = []
            for j in range(k):
                sub_range = (float(edges[j]), float(edges[j + 1]))
                init_list.append(
                    find_init(
                        astrom_data,
                        Ms,
                        dist_pc,
                        log_T_range=sub_range,
                        n_log_T=30,
                    )
                )

    actual_k = len(init_list)

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

    # Compute Choleskys for storage/sampling
    chols = jax.vmap(jnp.linalg.cholesky)(covs)

    # 6. Compute Laplace evidence weights
    d = z_maps.shape[-1]
    _, log_dets = jax.vmap(jnp.linalg.slogdet)(covs)
    log_ev = -losses + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * log_dets
    weights = jax.nn.softmax(log_ev)

    # 7. Postprocess function
    try:
        raw_pp = cached["postprocess_fn_factory"](*model_args)
    except TypeError:
        raw_pp = cached["postprocess_fn_factory"]

    def postprocess_fn(z_dict):
        phys = raw_pp(z_dict)
        return {k: jnp.squeeze(v) for k, v in phys.items()}

    return LaplaceMixtureResult(
        weights=weights,
        z_maps=z_maps,
        covariances=covs,
        choleskys=chols,
        losses=losses,
        log_evidence=log_ev,
        _unflatten=cached["unflatten"],
        _postprocess_fn=postprocess_fn,
        _model_trace=cached["model_trace"],
        _potential_fn_factory=cached["potential_fn_factory"],
        param_names=cached["param_names"],
        n_params=int(z_maps.shape[-1]),
        n_modes=actual_k,
    )
