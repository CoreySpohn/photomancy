"""NumPyro model + coordinate-map plumbing shared across the orbit fitters.

A leaf module (imports only orbit.model + orbix/numpyro/jax) so the fit,
inference-bridge, and EIG layers share one model trace cache, one data padding,
and one set of constraint helpers without importing each other's internals.

JIT-cache architecture: ``_MODEL_CACHE`` is keyed by the *static* model
configuration (prior ranges, eccentricity prior, data-channel flags). On first
call, the NumPyro model is traced once at the ``MAX_*`` shapes, extracting
constraint transforms and a compilable potential function. Subsequent calls with
different data (same shapes, via static padding) reuse the cached XLA compilation.
"""

import warnings

import jax
import jax.flatten_util
import jax.numpy as jnp
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import initialize_model

from photomancy.orbit.data import (
    ImagingData,
    NullData,
    OrbitData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
)
from photomancy.orbit.model import build_model

# ---------------------------------------------------------------------------
# Model cache -- trace once, reuse across calls
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple, dict] = {}


def _cache_key(
    has_rv,
    has_relative_astrom,
    has_stellar_astrom,
    has_pm_anomaly,
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
        has_relative_astrom,
        has_stellar_astrom,
        has_pm_anomaly,
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
    has_relative_astrom,
    has_stellar_astrom,
    has_pm_anomaly,
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
        has_relative_astrom,
        has_stellar_astrom,
        has_pm_anomaly,
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
        has_relative_astrom=has_relative_astrom,
        has_stellar_astrom=has_stellar_astrom,
        has_pm_anomaly=has_pm_anomaly,
        has_null=has_null,
        has_imaging=has_imaging,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
    )

    # Build a placeholder OrbitData at MAX sizes (for tracing the active channels)
    placeholder_Ms = 1.989e30
    placeholder_dist = 10.0
    placeholder_data = OrbitData(
        rv=RVData.zeros() if has_rv else None,
        relative_astrom=RelativeAstromData.zeros() if has_relative_astrom else None,
        stellar_astrom=StellarAstromData.zeros() if has_stellar_astrom else None,
        pm_anomaly=PMAnomalyData.zeros() if has_pm_anomaly else None,
        null=NullData.zeros() if has_null else None,
        imaging=ImagingData.zeros() if has_imaging else None,
    )

    model_args = (placeholder_Ms, placeholder_dist, placeholder_data)

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
# Coordinate maps -- init dict <-> flat z, and constraint bijectors
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


def make_constrain(fwd_transforms):
    """Build a ``raw-site dict -> physical-site dict`` closure from forward bijectors.

    Each named transform maps an unconstrained value to physical; sites without a
    transform pass through (squeezed). Shared by the orbit problem and the EIG Laplace
    path so both apply identical constraints.
    """

    def constrain(z_dict):
        return {
            name: jnp.squeeze(fwd_transforms[name](v))
            if name in fwd_transforms
            else jnp.squeeze(v)
            for name, v in z_dict.items()
        }

    return constrain


def _fwd_from_trace(model_trace):
    """Forward constraint bijectors per unobserved sample site, from a model trace."""
    return {
        name: biject_to(site["fn"].support)
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }
