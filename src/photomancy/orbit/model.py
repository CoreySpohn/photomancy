"""NumPyro model builder for orbit fitting.

Constructs a NumPyro model function from data containers. The model samples
orbital parameters using configurable eccentricity priors and evaluates
the pure-JAX likelihoods from :mod:`photomancy.orbit.likelihoods`.

Dynamic-args architecture
-------------------------
The model function returned by :func:`build_model` takes data containers
and physical parameters (``Ms``, ``dist_pc``) as **explicit arguments**
rather than capturing them in a closure. This allows the same model
function object to be reused across calls with different data, enabling
JIT compilation caching when combined with static-shape padding.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from hwoutils.constants import Mearth2kg
from orbix.equations import period_to_sma

from photomancy.orbit.likelihoods import (
    loglike_relative_astrom,
    loglike_imaging,
    loglike_null,
    loglike_rv_marginalized,
)
from photomancy.orbit.priors import sample_ecc_prior


def build_model(
    *,
    n_planets=1,
    has_rv=False,
    has_relative_astrom=False,
    has_null=False,
    has_imaging=False,
    log_P_range=(0.0, 5.0),
    log_Mp_range=(-2.0, 4.0),
    log_Rp_range=(-5.0, -2.5),
    log_Ag_range=(-2.0, 0.0),
    ecc_prior="kipping13",
    jitter_scale=1e-10,
):
    """Build a NumPyro model function for orbit fitting.

    The returned function accepts ``(Ms, dist_pc, rv_data, relative_astrom_data,
    null_data, imaging_data)`` as arguments (dynamic, for JIT caching).
    Prior configuration is captured in the closure (static).

    Args:
        n_planets: Number of planets to fit. Default 1.
        has_rv: Whether RV data will be provided.
        has_relative_astrom: Whether astrometry data will be provided.
        has_null: Whether null-detection data will be provided.
        has_imaging: Whether unified imaging data will be provided.
        log_P_range: (min, max) for log10(period/days) uniform prior.
        log_Mp_range: (min, max) for log10(mass/M_earth) uniform prior.
        log_Rp_range: (min, max) for log10(Rp/AU) uniform prior.
        log_Ag_range: (min, max) for log10(geometric albedo) prior.
        ecc_prior: Eccentricity prior name. One of ``"kipping13"`` (default),
            ``"rayleigh"``, ``"vaneylen19"``, or ``"disk"``.
        jitter_scale: Scale for HalfNormal jitter prior (AU/day).

    Returns:
        A callable ``model(Ms, dist_pc, rv_data, relative_astrom_data, null_data,
        imaging_data)`` suitable for ``numpyro.infer.MCMC`` and for use
        with ``dynamic_args=True`` in ``initialize_model``.
    """
    has_photometry = has_null or has_imaging

    if not any([has_rv, has_relative_astrom, has_null, has_imaging]):
        raise ValueError(
            "At least one of has_rv, has_relative_astrom, has_null, or has_imaging must be True."
        )

    def model(Ms, dist_pc, rv_data, relative_astrom_data, null_data, imaging_data):
        # CIRCULAR IMPORT: photomancy.orbit.forward -> photomancy.orbit.model
        from photomancy.orbit.forward import (
            predict_relative_astrometry,
            predict_photometry,
            predict_rv,
        )

        # ----------------------------------------------------------------
        # Per-planet orbital parameters via numpyro.plate
        # ----------------------------------------------------------------
        with numpyro.plate("planets", n_planets):
            # Period (log-uniform)
            log_P = numpyro.sample(
                "log_P", dist.Uniform(log_P_range[0], log_P_range[1])
            )
            T = numpyro.deterministic("T", 10.0**log_P)

            # Eccentricity + omega via selected prior
            e, cos_w, sin_w = sample_ecc_prior(ecc_prior)
            e = numpyro.deterministic("e", e)
            cos_w = numpyro.deterministic("cos_w", cos_w)
            sin_w = numpyro.deterministic("sin_w", sin_w)

            # Inclination (isotropic: uniform in cos_i)
            cos_i = numpyro.sample("cos_i", dist.Uniform(-1.0, 1.0))

            # Ascending node
            W = numpyro.sample("W", dist.Uniform(0.0, 2.0 * jnp.pi))

            # Phase / time of periapsis
            M0 = numpyro.sample("M0", dist.Uniform(0.0, 2.0 * jnp.pi))
            n_orb = 2.0 * jnp.pi / T
            tp = numpyro.deterministic("tp", -M0 / n_orb)

            # Semi-major axis from Kepler's 3rd law
            a = numpyro.deterministic("a", period_to_sma(T, Ms))

            # Photometric parameters
            if has_photometry:
                log_Rp = numpyro.sample(
                    "log_Rp",
                    dist.Uniform(log_Rp_range[0], log_Rp_range[1]),
                )
                Rp = numpyro.deterministic("Rp", 10.0**log_Rp)
                log_Ag = numpyro.sample(
                    "log_Ag",
                    dist.Uniform(log_Ag_range[0], log_Ag_range[1]),
                )
                Ag = numpyro.deterministic("Ag", 10.0**log_Ag)
                Lambda = numpyro.deterministic("Lambda", Ag * Rp**2)

        # ----------------------------------------------------------------
        # Mass parameters (if RV data present)
        # ----------------------------------------------------------------
        if has_rv:
            with numpyro.plate("planet_masses", n_planets):
                log_Mp = numpyro.sample(
                    "log_Mp",
                    dist.Uniform(log_Mp_range[0], log_Mp_range[1]),
                )
                Mp = numpyro.deterministic("Mp", 10.0**log_Mp * Mearth2kg)
                sin_i = jnp.sqrt(1.0 - cos_i**2)
                Mp_sini = numpyro.deterministic("Mp_sini", Mp * sin_i)

            with numpyro.plate("instruments", rv_data.n_inst):
                jitter = numpyro.sample("jitter", dist.HalfNormal(jitter_scale))

        # ----------------------------------------------------------------
        # Evaluate likelihoods
        # ----------------------------------------------------------------
        if n_planets == 1:
            T_s = jnp.squeeze(T)
            e_s = jnp.squeeze(e)
            cos_w_s = jnp.squeeze(cos_w)
            sin_w_s = jnp.squeeze(sin_w)
            cos_i_s = jnp.squeeze(cos_i)
            W_s = jnp.squeeze(W)
            tp_s = jnp.squeeze(tp)
            a_s = jnp.squeeze(a)

            ll_total = jnp.float64(0.0)

            # --- RV ---
            if has_rv:
                Mp_sini_s = jnp.squeeze(Mp_sini)
                rv_model = predict_rv(
                    rv_data.times,
                    T_s,
                    Ms,
                    Mp_sini_s,
                    e_s,
                    cos_w_s,
                    sin_w_s,
                    tp_s,
                )
                ll_rv_val = loglike_rv_marginalized(
                    rv_data.rv,
                    rv_model,
                    rv_data.rv_err,
                    rv_data.inst_ids,
                    rv_data.n_inst,
                    jitter,
                    rv_data.is_valid,
                )
                numpyro.factor("ll_rv", ll_rv_val)
                ll_total = ll_total + ll_rv_val

            # --- Astrometry ---
            if has_relative_astrom:
                ra_pred, dec_pred = predict_relative_astrometry(
                    relative_astrom_data.times,
                    a_s,
                    e_s,
                    cos_i_s,
                    W_s,
                    cos_w_s,
                    sin_w_s,
                    tp_s,
                    Ms,
                    dist_pc,
                )
                ll_relative_astrom_val = loglike_relative_astrom(ra_pred, dec_pred, relative_astrom_data)
                numpyro.factor("ll_relative_astrom", ll_relative_astrom_val)
                ll_total = ll_total + ll_relative_astrom_val

            # --- Null detections ---
            if has_null:
                Lambda_s = jnp.squeeze(Lambda)
                alpha_pred, dMag_pred = predict_photometry(
                    null_data.epochs,
                    a_s,
                    e_s,
                    cos_i_s,
                    W_s,
                    cos_w_s,
                    sin_w_s,
                    tp_s,
                    Ms,
                    Lambda_s,
                    dist_pc,
                )
                ll_null_val = loglike_null(alpha_pred, dMag_pred, null_data)
                numpyro.factor("ll_null", ll_null_val)
                ll_total = ll_total + ll_null_val

            # --- Imaging (unified detection + null) ---
            if has_imaging:
                Lambda_s = jnp.squeeze(Lambda)
                alpha_pred_img, dMag_pred_img = predict_photometry(
                    imaging_data.epochs,
                    a_s,
                    e_s,
                    cos_i_s,
                    W_s,
                    cos_w_s,
                    sin_w_s,
                    tp_s,
                    Ms,
                    Lambda_s,
                    dist_pc,
                )
                ll_img_val = loglike_imaging(
                    alpha_pred_img, dMag_pred_img, imaging_data
                )
                numpyro.factor("ll_imaging", ll_img_val)
                ll_total = ll_total + ll_img_val

            numpyro.deterministic("ll_total", ll_total)

        else:
            # Multi-planet: vmap forward models over planet axis
            # CIRCULAR IMPORT: photomancy.orbit.forward -> photomancy.orbit.model
            from photomancy.orbit.forward import predict_relative_astrometry as _pa
            from photomancy.orbit.forward import predict_rv as _prv

            def _single_planet_rv(T_p, Mp_sini_p, e_p, cos_w_p, sin_w_p, tp_p):
                return _prv(
                    rv_data.times,
                    T_p,
                    Ms,
                    Mp_sini_p,
                    e_p,
                    cos_w_p,
                    sin_w_p,
                    tp_p,
                )

            def _single_planet_astrom(a_p, e_p, cos_i_p, W_p, cos_w_p, sin_w_p, tp_p):
                return _pa(
                    relative_astrom_data.times,
                    a_p,
                    e_p,
                    cos_i_p,
                    W_p,
                    cos_w_p,
                    sin_w_p,
                    tp_p,
                    Ms,
                    dist_pc,
                )

            if has_rv:
                rv_per_planet = jax.vmap(_single_planet_rv)(
                    T, Mp_sini, e, cos_w, sin_w, tp
                )
                rv_model = jnp.sum(rv_per_planet, axis=0)
                ll_rv = loglike_rv_marginalized(
                    rv_data.rv,
                    rv_model,
                    rv_data.rv_err,
                    rv_data.inst_ids,
                    rv_data.n_inst,
                    jitter,
                    rv_data.is_valid,
                )
                numpyro.factor("ll_rv", ll_rv)

            if has_relative_astrom:
                ra_all, dec_all = jax.vmap(_single_planet_astrom)(
                    a, e, cos_i, W, cos_w, sin_w, tp
                )
                ra_pred = ra_all[
                    relative_astrom_data.planet_id,
                    jnp.arange(relative_astrom_data.times.shape[0]),
                ]
                dec_pred = dec_all[
                    relative_astrom_data.planet_id,
                    jnp.arange(relative_astrom_data.times.shape[0]),
                ]
                ll_relative_astrom = loglike_relative_astrom(ra_pred, dec_pred, relative_astrom_data)
                numpyro.factor("ll_relative_astrom", ll_relative_astrom)

    return model
