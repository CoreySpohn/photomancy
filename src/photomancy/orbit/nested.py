"""Nested sampling for the orbit model via NumPyro's jaxns wrapper.

The orbit fit is already a NumPyro model, so it gets nested-sampling evidence directly
from ``numpyro.contrib.nested_sampling`` (a thin jaxns wrapper) -- no model rewrite, no
hand-rolled prior layer. The headline output is the Bayesian evidence (``log Z``), which
turns an orbit fit into model comparison (planet count, period-alias resolution). The
result is returned as a photomancy :class:`~photomancy.posterior.SamplePosterior` in the
same physical columns OFTI / grid_search emit, so it drops into the EIG / clustering
machinery (``cluster_to_mixture``, ``to_unconstrained``).

NumPyro + jaxns are imported lazily inside the function: importing the wrapper pulls in
jaxns (which enables JAX x64 globally as a side effect), so ``import photomancy`` stays
light.
"""

import jax
import jax.numpy as jnp

from photomancy.orbit.data import OrbitData
from photomancy.orbit.model import build_model
from photomancy.posterior import SamplePosterior

# Physical columns, matching the OFTI / grid_search SamplePosterior convention so an
# orbit nested-sampling posterior is interchangeable with them downstream.
_ORBIT_PHYS_NAMES = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")


def orbit_nested_sampling(
    Ms,
    dist_pc,
    *,
    rv_data=None,
    relative_astrom_data=None,
    stellar_astrom_data=None,
    pm_anomaly_data=None,
    null_data=None,
    imaging_data=None,
    log_P_range=(1.0, 4.0),
    log_Mp_range=(-2.0, 4.0),
    log_Rp_range=(-5.0, -2.5),
    log_Ag_range=(-2.0, 0.0),
    ecc_prior="kipping13",
    jitter_scale=1e-10,
    num_live_points=None,
    max_samples=50000,
    num_samples=2000,
    key=None,
):
    """Nested-sample the orbit model -> ``SamplePosterior`` + evidence (``log Z``).

    Builds the NumPyro orbit model for the supplied data channels and runs it through
    ``numpyro.contrib.nested_sampling.NestedSampler`` (jaxns). The returned posterior
    carries equally-weighted physical samples (``a, e, cos_i, W, cos_w, sin_w, tp``) and
    the evidence; the evidence is the point of nested sampling -- a Bayes factor between
    two fits is orbit model comparison (e.g. planet present vs absent, alias A vs B).

    Args:
        Ms: Stellar mass (kg).
        dist_pc: Distance to the system (parsec).
        rv_data: An :class:`~photomancy.orbit.data.RVData`, or ``None``.
        relative_astrom_data: A
            :class:`~photomancy.orbit.data.RelativeAstromData`, or ``None``.
        stellar_astrom_data: A
            :class:`~photomancy.orbit.data.StellarAstromData`, or ``None``.
        pm_anomaly_data: A
            :class:`~photomancy.orbit.data.PMAnomalyData`, or ``None``.
        null_data: A :class:`~photomancy.orbit.data.NullData`, or ``None``.
        imaging_data: An :class:`~photomancy.orbit.data.ImagingData`, or ``None``.
        log_P_range: ``(min, max)`` for the ``log10(period/days)`` prior.
        log_Mp_range: ``(min, max)`` for the ``log10(mass/M_earth)`` prior.
        log_Rp_range: ``(min, max)`` for the ``log10(Rp/AU)`` prior.
        log_Ag_range: ``(min, max)`` for the ``log10(geometric albedo)`` prior.
        ecc_prior: Eccentricity prior name (``"kipping13"``, ``"rayleigh"``,
            ``"vaneylen19"``, ``"disk"``).
        jitter_scale: Scale for the HalfNormal RV-jitter prior.
        num_live_points: jaxns live points; ``None`` uses the jaxns default.
        max_samples: Termination cap on total nested-sampling samples.
        num_samples: Number of equally-weighted posterior samples to return.
        key: PRNG key (required).

    Returns:
        A :class:`~photomancy.posterior.SamplePosterior` with ``samples`` of shape
        ``(num_samples, 7)`` over ``_ORBIT_PHYS_NAMES``, uniform ``log_weights``, and
        ``evidence`` set to the nested-sampling ``log Z``.

    Raises:
        ValueError: If ``key`` is ``None``.
    """
    if key is None:
        raise ValueError("orbit_nested_sampling requires a PRNG key.")
    from numpyro.contrib.nested_sampling import NestedSampler

    model = build_model(
        n_planets=1,
        has_rv=rv_data is not None,
        has_relative_astrom=relative_astrom_data is not None,
        has_stellar_astrom=stellar_astrom_data is not None,
        has_pm_anomaly=pm_anomaly_data is not None,
        has_null=null_data is not None,
        has_imaging=imaging_data is not None,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
    )

    constructor_kwargs = {"max_samples": max_samples}
    if num_live_points is not None:
        constructor_kwargs["num_live_points"] = num_live_points

    run_key, sample_key = jax.random.split(key)
    ns = NestedSampler(model, constructor_kwargs=constructor_kwargs)
    data = OrbitData(
        rv=rv_data,
        relative_astrom=relative_astrom_data,
        stellar_astrom=stellar_astrom_data,
        pm_anomaly=pm_anomaly_data,
        null=null_data,
        imaging=imaging_data,
    )
    ns.run(run_key, Ms, dist_pc, data)

    draws = ns.get_samples(sample_key, num_samples=num_samples)
    # The model exposes the physical orbit parameters as deterministic sites, so the
    # equally-weighted draws give physical columns directly (squeeze the planet axis).
    samples = jnp.stack(
        [jnp.reshape(draws[name], (num_samples,)) for name in _ORBIT_PHYS_NAMES],
        axis=1,
    )
    # ``_results`` is the jaxns Results object; ``log_Z_mean`` is the evidence estimate.
    evidence = jnp.asarray(ns._results.log_Z_mean)

    return SamplePosterior(
        samples=samples,
        log_weights=jnp.zeros(num_samples),
        evidence=evidence,
        param_names=_ORBIT_PHYS_NAMES,
    )
