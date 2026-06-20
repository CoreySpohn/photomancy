"""Orbit-domain Expected Information Gain (a thin wrapper over the generic layer).

Supplies the orbit forward model (unconstrained ``z`` -> sky position via Kepler) and
the contrast-curve detectability, then delegates the analytic EIG to
:mod:`photomancy.eig`. The geometric / alias-breaking / detectability primitives are
re-exported from there. ``evaluate_orbit_candidates`` (a z-space MixturePosterior + an
OrbitProblem) schedules through this layer.
"""

from collections.abc import Callable

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

from photomancy.eig import (
    alias_breaking_eig,
    detectability_eig,
    geometric_eig,
)
from photomancy.eig import evaluate_candidates as _eval_candidates
from photomancy.posterior import MixturePosterior

__all__ = [
    "alias_breaking_eig",
    "detectability_eig",
    "evaluate_orbit_candidates",
    "geometric_eig",
]


# ---------------------------------------------------------------------------
# Pure-JAX prediction from unconstrained z -- the orbit forward model
# ---------------------------------------------------------------------------


def _predict_astrom_pure(
    z_flat: jnp.ndarray,
    t: float,
    unflatten: Callable,
    constrain: Callable,
    Ms: float,
    dist_pc: float,
) -> jnp.ndarray:
    """Predict (RA, Dec) from unconstrained z at a single time.

    ``constrain`` applies the forward bijectors (no NumPyro model trace), squeezing each
    param to a scalar for the Kepler solver under double-vmap. Returns shape ``(2,)``
    array ``[RA, Dec]`` in arcsec.
    """
    phys = constrain(unflatten(z_flat))

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
    constrain: Callable,
    Ms: float,
    dist_pc: float,
    Lambda: float,
) -> jnp.ndarray:
    """Predict separation (arcsec) and dMag from unconstrained z.

    Same orbital mechanics as ``_predict_astrom_pure`` plus the Lambert phase function.
    ``Lambda = Ag * Rp^2`` is a known constant (the planet was detected once). Returns
    shape ``(2,)`` array ``[sep_arcsec, dMag]``.
    """
    phys = constrain(unflatten(z_flat))

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
# High-level batch evaluation (delegates to photomancy.eig)
# ---------------------------------------------------------------------------


def _orbit_evaluate(
    means,
    covs,
    log_evidences,
    unflatten,
    constrain,
    candidate_epochs,
    obs_variance,
    Ms,
    dist_pc,
    *,
    Lambda,
    contrast_curve,
    iwa,
):
    """Shared orbit-EIG core: build forward + detectability, delegate, add sep/dMag."""
    posterior = MixturePosterior(means=means, covs=covs, log_evidences=log_evidences)

    def forward(z, t):
        return _predict_astrom_pure(z, t, unflatten, constrain, Ms, dist_pc)

    has_imaging = Lambda is not None and contrast_curve is not None
    detectable = None
    if has_imaging:
        csep = jnp.asarray(contrast_curve[0])
        cdmag = jnp.asarray(contrast_curve[1])

        def detectable(z, t):
            sep, dmag = _predict_sep_dmag_pure(
                z, t, unflatten, constrain, Ms, dist_pc, Lambda
            )
            dmag_limit = jnp.interp(sep, csep, cdmag, left=jnp.inf, right=jnp.inf)
            return ((sep > iwa) & (dmag < dmag_limit)).astype(jnp.float64)

    # id(unflatten) is a stable per-config identity: _MODEL_CACHE keeps one unflatten
    # per model config alive for the process lifetime, so the id never dangles or
    # aliases. A param-name key would be wrong here -- two configs with the same sites
    # but different prior ranges share param_names yet need distinct compiled kernels.
    cache_key = (id(unflatten), Ms, dist_pc, has_imaging, Lambda, iwa)
    epochs = jnp.asarray(candidate_epochs)
    res = _eval_candidates(
        posterior,
        epochs,
        forward,
        obs_variance,
        detectable=detectable,
        cache_key=cache_key,
    )

    if has_imaging:
        sep_dmag = jax.vmap(
            lambda t: jax.vmap(
                lambda z: _predict_sep_dmag_pure(
                    z, t, unflatten, constrain, Ms, dist_pc, Lambda
                )
            )(posterior.means)
        )(epochs)
        res["separation"] = sep_dmag[..., 0]
        res["dMag"] = sep_dmag[..., 1]
    else:
        res["separation"] = jnp.zeros_like(res["detectability"])
        res["dMag"] = jnp.zeros_like(res["detectability"])
    return res


def evaluate_orbit_candidates(
    posterior,
    problem,
    candidate_epochs,
    obs_variance,
    Ms,
    dist_pc,
    *,
    Lambda=None,
    contrast_curve=None,
    iwa=0.0,
):
    """EIG over candidate epochs for a generic z-space MixturePosterior + OrbitProblem.

    Lets OFTI / grid_search (via ``to_unconstrained`` -> ``cluster_to_mixture``) drive
    the same analytic EIG as the Laplace mixture. ``problem`` supplies the model
    ``unflatten`` and ``constrain``. Returns the per-candidate EIG dict.
    """
    return _orbit_evaluate(
        posterior.means,
        posterior.covs,
        posterior.log_evidences,
        problem.unflatten,
        problem.constrain,
        candidate_epochs,
        obs_variance,
        Ms,
        dist_pc,
        Lambda=Lambda,
        contrast_curve=contrast_curve,
        iwa=iwa,
    )
