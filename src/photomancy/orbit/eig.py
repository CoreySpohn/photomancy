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
from hwoutils.constants import Mearth2kg
from orbix.equations import period_to_sma

from photomancy.eig import (
    alias_breaking_eig,
    alias_breaking_eig_from_covariances,
    detectability_eig,
    geometric_eig,
)
from photomancy.eig import evaluate_candidates as _eval_candidates
from photomancy.orbit._contrast import interpolate_contrast_curve
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from photomancy.posterior import MixturePosterior

__all__ = [
    "alias_breaking_eig",
    "alias_breaking_eig_from_covariances",
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
    Mp = 10.0 ** phys["log_Mp"] * Mearth2kg if "log_Mp" in phys else 0.0
    a = period_to_sma(T, Ms + Mp)
    tp = -M0 / (2.0 * jnp.pi / T)
    cos_w = jnp.cos(w)
    sin_w = jnp.sin(w)

    ra, dec = predict_relative_astrometry(
        jnp.atleast_1d(t), a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc, Mp=Mp
    )
    return jnp.array([ra[0], dec[0]])


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

    Mp = 10.0 ** phys["log_Mp"] * Mearth2kg if "log_Mp" in phys else 0.0
    a = period_to_sma(T, Ms + Mp)
    tp = -M0 / (2.0 * jnp.pi / T)
    cos_w = jnp.cos(w)
    sin_w = jnp.sin(w)

    sep, dmag = predict_photometry(
        jnp.atleast_1d(t),
        a,
        e,
        cos_i,
        W,
        cos_w,
        sin_w,
        tp,
        Ms,
        Lambda,
        dist_pc,
        Mp=Mp,
    )
    return jnp.array([sep[0], dmag[0]])


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
    detectable_args = ()
    if has_imaging:
        detectable_args = tuple(map(jnp.asarray, contrast_curve))

        def detectable(z, t, csep, cdmag):
            sep, dmag = _predict_sep_dmag_pure(
                z, t, unflatten, constrain, Ms, dist_pc, Lambda
            )
            dmag_limit = interpolate_contrast_curve(
                sep[None], csep[None, :], cdmag[None, :]
            )[0]
            return ((sep > iwa) & (dmag < dmag_limit)).astype(sep.dtype)

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
        detectable_args=detectable_args,
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
    ``unflatten`` and ``constrain``. Returns the per-candidate EIG dict of
    :func:`photomancy.eig.evaluate_candidates`, an approximate score.

    Without ``Lambda`` and ``contrast_curve`` every epoch is assumed to yield
    reported relative astrometry with fixed measurement noise. With both, each mode
    is reported according to its mean orbit (outside ``iwa``, within the curve's
    separation range and brighter than its limit). This gives the per-mode reporting
    probability in :func:`photomancy.eig.evaluate_candidates`. The
    threshold is not applied within a mode, so selection that cuts through a mode's
    predicted position or brightness is not represented.
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
