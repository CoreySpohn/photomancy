"""Orbit-domain Expected Information Gain (a thin wrapper over the generic layer).

Supplies the orbit forward model (unconstrained ``z`` -> sky position via Kepler) and
the contrast-curve detectability, then delegates the analytic EIG to
:mod:`photomancy.eig`. The geometric / alias-breaking / detectability primitives are
re-exported from there. ``evaluate_candidates`` keeps its public signature and output,
now scheduling against any orbit Laplace mixture through the Posterior interface.
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
    "evaluate_candidates",
    "geometric_eig",
]


# ---------------------------------------------------------------------------
# Pure-JAX prediction from unconstrained z -- the orbit forward model
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
# High-level batch evaluation (delegates to photomancy.eig)
# ---------------------------------------------------------------------------


def evaluate_candidates(
    mixture,
    candidate_epochs,
    obs_variance,
    Ms,
    dist_pc,
    *,
    Lambda=None,
    contrast_curve=None,
    iwa=0.0,
):
    """Imaging-aware EIG over candidate epochs for an orbit Laplace mixture.

    Thin wrapper over :func:`photomancy.eig.evaluate_candidates`: builds the orbit
    forward model and (optionally) the contrast-curve detectability, then delegates.

    Args:
        mixture: A fitted ``LaplaceMixtureResult``.
        candidate_epochs: Candidate observation times (days). Shape ``(N,)``.
        obs_variance: Measurement variance per observable (scalar or ``(n_obs,)``).
        Ms: Stellar mass (kg).
        dist_pc: Distance (parsec).
        Lambda: Planet photometric area ``Ag * Rp^2`` (AU^2). With ``contrast_curve``,
            enables imaging-aware EIG.
        contrast_curve: ``(sep_arcsec, dmag_limit)`` arrays for the detection threshold.
        iwa: Inner working angle (arcsec); planets inside are not detectable.

    Returns:
        Dict with ``total_eig``, ``geometric_eig``, ``alias_eig``, ``predictions``,
        ``detectability`` (all per candidate), plus orbit ``separation`` and ``dMag``
        per mode.
    """
    from numpyro.distributions.transforms import biject_to

    fwd_transforms = {}
    for name, site in mixture._model_trace.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            fwd_transforms[name] = biject_to(site["fn"].support)
    unflatten = mixture._unflatten

    posterior = MixturePosterior(
        means=mixture.z_maps,
        covs=mixture.covariances,
        log_evidences=mixture.log_evidence,
    )

    def forward(z, t):
        return _predict_astrom_pure(z, t, unflatten, fwd_transforms, Ms, dist_pc)

    has_imaging = Lambda is not None and contrast_curve is not None
    detectable = None
    if has_imaging:
        csep = jnp.asarray(contrast_curve[0])
        cdmag = jnp.asarray(contrast_curve[1])

        def detectable(z, t):
            sep, dmag = _predict_sep_dmag_pure(
                z, t, unflatten, fwd_transforms, Ms, dist_pc, Lambda
            )
            dmag_limit = jnp.interp(sep, csep, cdmag, left=jnp.inf, right=jnp.inf)
            return ((sep > iwa) & (dmag < dmag_limit)).astype(jnp.float64)

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
                    z, t, unflatten, fwd_transforms, Ms, dist_pc, Lambda
                )
            )(posterior.means)
        )(epochs)
        res["separation"] = sep_dmag[..., 0]
        res["dMag"] = sep_dmag[..., 1]
    else:
        res["separation"] = jnp.zeros_like(res["detectability"])
        res["dMag"] = jnp.zeros_like(res["detectability"])
    return res
