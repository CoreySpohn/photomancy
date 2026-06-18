"""Analytic Expected Information Gain for Bayesian Experimental Design.

Uses the Fisher Information Matrix to compute the effect of a hypothetical
observation WITHOUT re-running the optimizer.  The Laplace covariance update
is analytic:

    Sigma_new⁻¹ = Sigma_old⁻¹ + Jᵀ R⁻¹ J

where J is the forward-model Jacobian at the MAP and R is the measurement
noise covariance.  This replaces the O(seconds) optimizer with an O(mus)
matrix operation, enabling real-time BED evaluation.

The total EIG decomposes into:
    1. **Geometric refinement** -- how much covariance shrinks within each mode
    2. **Alias breaking** -- how much mixture weights collapse toward one mode

Both are computed analytically from the existing Laplace mixture result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from hwoutils.constants import G
from orbix.equations.orbit import (
    AB_matrices_reduced,
    mean_anomaly_tp,
    mean_motion,
)
from orbix.equations.phase import lambert_phase_exact
from orbix.equations.propagation import single_r
from orbix.kepler.core import diff_solve_trig

if TYPE_CHECKING:
    from photomancy.orbit.laplace import LaplaceMixtureResult


# ---------------------------------------------------------------------------
# Pure-JAX prediction from unconstrained z -- NO NumPyro dependency
# ---------------------------------------------------------------------------


def _predict_astrom_pure(
    z_flat: jnp.ndarray,
    t: float,
    unflatten: Callable,
    fwd_transforms: dict,
    Ms: float,
    dist_pc: float,
) -> jnp.ndarray:
    """Predict (RA, Dec) from unconstrained z at a single time.

    Uses constraint bijectors directly -- no NumPyro model trace.
    Returns shape ``(2,)`` array ``[RA, Dec]`` in arcsec.

    All physical params are squeezed to true scalars to ensure
    compatibility with the Kepler solver under double-vmap.
    """
    z_dict = unflatten(z_flat)

    # Apply forward (unconstrained -> constrained) transforms
    # Squeeze to scalar -- unflatten may produce shape (1,) values
    phys = {}
    for name, z_val in z_dict.items():
        if name in fwd_transforms:
            phys[name] = jnp.squeeze(fwd_transforms[name](z_val))
        else:
            phys[name] = jnp.squeeze(z_val)

    # Extract physical parameters (all scalar after squeeze)
    log_P = phys["log_P"]
    T = 10.0**log_P
    e = phys["e_raw"]  # after transform: physical eccentricity
    cos_i = phys["cos_i"]
    W = phys["W"]
    w = phys["w_raw"]  # after transform: physical omega
    M0 = phys["M0"]

    # Derived quantities
    a_cubed = G * Ms * T**2 / (4.0 * jnp.pi**2)
    a = a_cubed ** (1.0 / 3.0)
    tp = -M0 / (2.0 * jnp.pi / T)
    cos_w = jnp.cos(w)
    sin_w = jnp.sin(w)

    mu = G * Ms
    n = mean_motion(a, mu)
    # Use atleast_1d to create a proper 1-element time array
    M = mean_anomaly_tp(jnp.atleast_1d(t), n, tp)
    sinE, cosE = diff_solve_trig(M, e)

    sin_i = jnp.sqrt(1.0 - cos_i**2)
    sqrt_1me2 = jnp.sqrt(1.0 - e**2)
    sinW, cosW = jnp.sin(W), jnp.cos(W)

    A, B = AB_matrices_reduced(a, sqrt_1me2, sin_i, cos_i, sinW, cosW, sin_w, cos_w)
    r = single_r(A, B, e, sinE, cosE)

    ra = r[0, 0] / dist_pc
    dec = r[1, 0] / dist_pc
    return jnp.array([ra, dec])


def _predict_sep_dmag_pure(
    z_flat: jnp.ndarray,
    t: float,
    unflatten: Callable,
    fwd_transforms: dict,
    Ms: float,
    dist_pc: float,
    Lambda: float,
) -> jnp.ndarray:
    """Predict separation (arcsec) and dMag from unconstrained z.

    Uses the same orbital mechanics as ``_predict_astrom_pure`` plus
    the Lambert phase function to compute photometric observables.
    ``Lambda = Ag * Rp^2`` is provided as a known constant (the planet
    has already been detected once).

    Returns shape ``(2,)`` array ``[sep_arcsec, dMag]``.
    """
    z_dict = unflatten(z_flat)
    phys = {}
    for name, z_val in z_dict.items():
        if name in fwd_transforms:
            phys[name] = jnp.squeeze(fwd_transforms[name](z_val))
        else:
            phys[name] = jnp.squeeze(z_val)

    log_P = phys["log_P"]
    T = 10.0**log_P
    e = phys["e_raw"]
    cos_i = phys["cos_i"]
    W = phys["W"]
    w = phys["w_raw"]
    M0 = phys["M0"]

    a_cubed = G * Ms * T**2 / (4.0 * jnp.pi**2)
    a = a_cubed ** (1.0 / 3.0)
    tp = -M0 / (2.0 * jnp.pi / T)
    cos_w = jnp.cos(w)
    sin_w = jnp.sin(w)

    mu = G * Ms
    n = mean_motion(a, mu)
    M = mean_anomaly_tp(jnp.atleast_1d(t), n, tp)
    sinE, cosE = diff_solve_trig(M, e)

    sin_i = jnp.sqrt(1.0 - cos_i**2)
    sqrt_1me2 = jnp.sqrt(1.0 - e**2)
    sinW, cosW = jnp.sin(W), jnp.cos(W)

    A, B = AB_matrices_reduced(a, sqrt_1me2, sin_i, cos_i, sinW, cosW, sin_w, cos_w)
    r = single_r(A, B, e, sinE, cosE)

    # Angular separation on sky
    sky_sep_AU = jnp.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    sep_arcsec = sky_sep_AU / dist_pc

    # Phase angle
    r_mag = jnp.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2 + r[2, 0] ** 2)
    r_mag_safe = jnp.maximum(r_mag, 1e-30)
    cos_beta = -r[2, 0] / r_mag_safe
    sin_beta = jnp.sqrt(jnp.maximum(1.0 - cos_beta**2, 0.0))
    phase = lambert_phase_exact(cos_beta, sin_beta)

    # Contrast and dMag
    contrast = Lambda * phase / jnp.maximum(r_mag**2, 1e-60)
    dMag = -2.5 * jnp.log10(jnp.maximum(contrast, 1e-300))

    return jnp.array([sep_arcsec, dMag])


# ---------------------------------------------------------------------------
# Core EIG functions
# ---------------------------------------------------------------------------


def geometric_eig(
    cov_old: jnp.ndarray,
    jacobian: jnp.ndarray,
    obs_variance: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Information gain from covariance shrinkage at one mode.

    Args:
        cov_old: Current covariance matrix.  Shape ``(D, D)``.
        jacobian: Forward-model Jacobian ``dy/dz``.  Shape ``(n_obs, D)``
            where n_obs is the number of observables per epoch (e.g. 2
            for RA/Dec astrometry).
        obs_variance: Scalar or ``(n_obs,)`` measurement variances.

    Returns:
        Tuple of ``(eig, cov_new)`` where eig is the scalar information
        gain in nats and cov_new is the updated covariance ``(D, D)``.
    """
    prec_old = jnp.linalg.inv(cov_old)
    obs_var = jnp.atleast_1d(jnp.asarray(obs_variance))
    n_obs = jacobian.shape[0]
    # Broadcast scalar variance to all observables
    obs_var = jnp.broadcast_to(obs_var, (n_obs,))
    R_inv = jnp.diag(1.0 / obs_var)
    FIM = jacobian.T @ R_inv @ jacobian
    prec_new = prec_old + FIM
    cov_new = jnp.linalg.inv(prec_new)
    eig = 0.5 * (jnp.linalg.slogdet(cov_old)[1] - jnp.linalg.slogdet(cov_new)[1])
    return eig, cov_new


def alias_breaking_eig(
    weights: jnp.ndarray,
    y_preds: jnp.ndarray,
    obs_variance: float | jnp.ndarray,
) -> jnp.ndarray:
    """Information gain from prediction disagreement across modes.

    When modes predict very different observables for a candidate epoch,
    obtaining data at that epoch will dramatically shift the mixture
    weights -- breaking aliases.

    Args:
        weights: Mode weights.  Shape ``(K,)``.
        y_preds: Predictions at each mode.  Shape ``(K, n_obs)``.
        obs_variance: Scalar or ``(n_obs,)`` measurement variances.

    Returns:
        Scalar alias-breaking EIG in nats.
    """
    obs_var = jnp.atleast_1d(jnp.asarray(obs_variance))
    n_obs = y_preds.shape[-1]
    obs_var = jnp.broadcast_to(obs_var, (n_obs,))
    y_mean = jnp.sum(weights[:, None] * y_preds, axis=0)
    y_var = jnp.sum(weights[:, None] * (y_preds - y_mean[None, :]) ** 2, axis=0)
    return 0.5 * jnp.sum(jnp.log1p(y_var / obs_var))


def detectability_eig(
    weights: jnp.ndarray,
    det_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Information gain from detection disagreement across modes.

    When some modes predict the planet is detectable at a candidate
    epoch while others predict non-detection (inside IWA or fainter
    than the contrast curve), observing at that epoch is maximally
    informative -- regardless of the outcome.

    Uses the weighted variance of the binary detectability signal,
    normalized by the maximum Bernoulli variance (0.25).

    Args:
        weights: Mode weights.  Shape ``(K,)``.
        det_weights: Per-mode detectability (1.0 = detectable,
            0.0 = non-detectable).  Shape ``(K,)``.

    Returns:
        Scalar detection-disagreement EIG in nats.
    """
    p_det = jnp.sum(weights * det_weights)
    det_var = jnp.sum(weights * (det_weights - p_det) ** 2)
    return 0.5 * jnp.log1p(det_var / 0.25)


# ---------------------------------------------------------------------------
# Cached JIT-compiled EIG batch evaluator
# ---------------------------------------------------------------------------

# Module-level cache: keyed on (unflatten_id, Ms, dist_pc, has_imaging)
# so the JIT-compiled function is reused across calls with the
# same model structure.
_EIG_JIT_CACHE: dict[tuple, Callable] = {}


def _build_eig_batch_fn(
    unflatten: Callable,
    fwd_transforms: dict,
    Ms: float,
    dist_pc: float,
    *,
    Lambda: float | None = None,
    contrast_sep: jnp.ndarray | None = None,
    contrast_dmag: jnp.ndarray | None = None,
    iwa: float = 0.0,
) -> Callable:
    """Build and JIT-compile the batched EIG function once.

    When ``Lambda`` and ``contrast_sep``/``contrast_dmag`` are provided,
    the function also computes detectability per mode and includes
    detection-disagreement alias-breaking EIG.

    The returned function has signature::

        fn(candidate_epochs, z_maps, covs, weights, obs_var)
            -> (total, geom, alias, preds, det_weights, seps, dmags)

    All arguments are JAX arrays; the function is traced once and
    reused across calls.
    """
    has_imaging = Lambda is not None and contrast_sep is not None

    def _pred_one(z_flat, t):
        """Predict (RA, Dec) at time t for unconstrained params z."""
        return _predict_astrom_pure(z_flat, t, unflatten, fwd_transforms, Ms, dist_pc)

    def _pred_and_jac(z_flat, t):
        """Prediction + Jacobian at (z, t)."""
        y = _pred_one(z_flat, t)
        J = jax.jacrev(lambda z: _pred_one(z, t))(z_flat)
        return y, J

    if has_imaging:
        # Capture imaging params in closure -- static for this schedule
        _Lambda = Lambda
        _csep = jnp.asarray(contrast_sep)
        _cdmag = jnp.asarray(contrast_dmag)
        _iwa = iwa

        def _pred_detect(z_flat, t):
            """Predict (sep, dMag) for detectability check."""
            return _predict_sep_dmag_pure(
                z_flat,
                t,
                unflatten,
                fwd_transforms,
                Ms,
                dist_pc,
                _Lambda,
            )

    def _eig_single_candidate(t, z_maps, covs, weights, obs_var):
        """Compute EIG for one candidate epoch."""
        # vmap over K modes
        y_preds, jacobians = jax.vmap(lambda z_k: _pred_and_jac(z_k, t))(z_maps)

        # Geometric EIG per mode
        def _geom_one(cov_k, J_k):
            eig_k, _ = geometric_eig(cov_k, J_k, obs_var)
            return eig_k

        geom_eigs = jax.vmap(_geom_one)(covs, jacobians)

        if has_imaging:
            # Detectability predictions per mode
            det_preds = jax.vmap(lambda z_k: _pred_detect(z_k, t))(
                z_maps
            )  # (K, 2) = [sep, dMag]
            seps = det_preds[:, 0]
            dmags = det_preds[:, 1]

            # Check detectability: inside IWA or fainter than limit
            dmag_limit = jnp.interp(
                seps,
                _csep,
                _cdmag,
                left=jnp.inf,
                right=jnp.inf,
            )
            det_w = ((seps > _iwa) & (dmags < dmag_limit)).astype(jnp.float64)

            # Weight geometric EIG by detectability -- non-detectable
            # modes contribute no astrometric information
            weighted_geom = jnp.sum(weights * det_w * geom_eigs)

            # Alias-breaking: astrometric + detection disagreement
            alias_astrom = alias_breaking_eig(weights, y_preds, obs_var)
            alias_det = detectability_eig(weights, det_w)
            alias_val = alias_astrom + alias_det
        else:
            det_w = jnp.ones(z_maps.shape[0])
            seps = jnp.zeros(z_maps.shape[0])
            dmags = jnp.zeros(z_maps.shape[0])
            weighted_geom = jnp.sum(weights * geom_eigs)
            alias_val = alias_breaking_eig(weights, y_preds, obs_var)

        total = weighted_geom + alias_val
        return (
            total,
            weighted_geom,
            alias_val,
            y_preds,
            det_w,
            seps,
            dmags,
        )

    @jax.jit
    def _batch_eig(candidate_epochs, z_maps, covs, weights, obs_var):
        """Batched EIG over all candidate epochs."""
        return jax.vmap(
            lambda t: _eig_single_candidate(t, z_maps, covs, weights, obs_var)
        )(candidate_epochs)

    return _batch_eig


# ---------------------------------------------------------------------------
# High-level batch evaluation
# ---------------------------------------------------------------------------


def evaluate_candidates(
    mixture: LaplaceMixtureResult,
    candidate_epochs: jnp.ndarray,
    obs_variance: float | jnp.ndarray,
    Ms: float,
    dist_pc: float,
    *,
    Lambda: float | None = None,
    contrast_curve: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    iwa: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Evaluate EIG for a batch of candidate observation epochs.

    This is the main entry point for BED scheduling.  It vmaps over
    candidate epochs and, for each one, computes:

    1. Predictions at each MAP mode (pure JAX forward model)
    2. Jacobians at each MAP mode (via ``jax.jacrev``)
    3. Geometric EIG per mode (weighted by mixture weights)
    4. Alias-breaking EIG across modes
    5. Total EIG = weighted geometric + alias breaking

    **Imaging-aware mode** (optional): When ``Lambda`` and
    ``contrast_curve`` are provided, also computes:

    6. Separation and dMag at each mode via Lambert phase function
    7. Per-mode detectability (sep > IWA and dMag < contrast limit)
    8. Detection-weighted geometric EIG (non-detectable modes
       contribute no astrometric information)
    9. Detection-disagreement alias-breaking EIG

    The batch function is JIT-compiled **once** and cached for reuse.
    First call incurs ~1s compilation; subsequent calls with the same
    model structure run in ~ms.

    Args:
        mixture: A fitted ``LaplaceMixtureResult``.
        candidate_epochs: Array of candidate observation times (days).
            Shape ``(N_cand,)``.
        obs_variance: Measurement variance per observable.  Scalar
            (isotropic RA/Dec noise in arcsec^2) or ``(n_obs,)``
            for heteroscedastic noise.
        Ms: Stellar mass (kg).
        dist_pc: Distance (parsec).
        Lambda: Planet photometric area ``Ag * Rp^2`` (AU^2).  When
            provided with ``contrast_curve``, enables imaging-aware
            EIG.  Typically known from the initial detection.
        contrast_curve: Tuple of ``(sep_arcsec, dmag_limit)`` arrays
            defining the coronagraph contrast curve.  The detection
            threshold at a given separation is interpolated from
            these arrays.
        iwa: Inner working angle (arcsec).  Planets inside the IWA
            are not detectable.  Default 0.0 (no IWA constraint).

    Returns:
        Dict with keys:
            - ``"total_eig"``: Total EIG per candidate.  ``(N_cand,)``.
            - ``"geometric_eig"``: Weighted geometric EIG.  ``(N_cand,)``.
            - ``"alias_eig"``: Alias-breaking EIG.  ``(N_cand,)``.
            - ``"predictions"``: Mode predictions.  ``(N_cand, K, 2)``.
            - ``"detectability"``: Per-mode detection weight.
              ``(N_cand, K)``.  All 1.0 when imaging is off.
            - ``"separation"``: Predicted separation per mode.
              ``(N_cand, K)``.  All 0.0 when imaging is off.
            - ``"dMag"``: Predicted dMag per mode.
              ``(N_cand, K)``.  All 0.0 when imaging is off.
    """
    obs_var = jnp.atleast_1d(jnp.asarray(obs_variance))

    # Extract forward (unconstrained -> constrained) transforms
    # from the model trace stored in the mixture result.
    # Late import: numpyro is a heavy optional dependency only needed
    # at evaluation time, not when the module is imported.
    from numpyro.distributions.transforms import biject_to

    fwd_transforms = {}
    for name, site in mixture._model_trace.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            fwd_transforms[name] = biject_to(site["fn"].support)

    unflatten = mixture._unflatten

    has_imaging = Lambda is not None and contrast_curve is not None
    contrast_sep = contrast_curve[0] if has_imaging else None
    contrast_dmag = contrast_curve[1] if has_imaging else None

    # Cache key includes imaging config so we don't mix compiled fns
    cache_key = (
        id(unflatten),
        Ms,
        dist_pc,
        has_imaging,
        Lambda,
        iwa,
    )

    if cache_key not in _EIG_JIT_CACHE:
        _EIG_JIT_CACHE[cache_key] = _build_eig_batch_fn(
            unflatten,
            fwd_transforms,
            Ms,
            dist_pc,
            Lambda=Lambda,
            contrast_sep=contrast_sep,
            contrast_dmag=contrast_dmag,
            iwa=iwa,
        )

    batch_fn = _EIG_JIT_CACHE[cache_key]

    # All dynamic data passed as explicit arguments -- no re-tracing!
    total, geom, alias, preds, det_w, seps, dmags = batch_fn(
        candidate_epochs,
        mixture.z_maps,
        mixture.covariances,
        mixture.weights,
        obs_var,
    )

    return {
        "total_eig": total,
        "geometric_eig": geom,
        "alias_eig": alias,
        "predictions": preds,
        "detectability": det_w,
        "separation": seps,
        "dMag": dmags,
    }
