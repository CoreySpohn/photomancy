"""Bridge the NumPyro orbit model onto the generic photomancy backends.

``build_orbit_logdensity`` turns an orbit fitting problem (data + priors) into a
flat ``logdensity(z)`` over the unconstrained NumPyro position, plus the maps a
backend result needs: ``to_physical`` (flat z -> named physical parameters) and
``init_to_z`` (a blind-initializer dict -> flat z). Tracing the model once, caching
the potential, and the constraint transforms are reused from
``photomancy.orbit.laplace``. With this, orbit fitting runs through any generic
backend (Laplace, Laplace mixture, NUTS, SMC) and returns a uniform ``Posterior``.

The orbit-specific ``map_laplace_fit`` / ``map_laplace_mixture_fit`` remain the
high-throughput path (static-arg JIT reuses one compilation across many fits); this
bridge is the engine-agnostic path that closure-captures the (small, padded) data.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G

from photomancy.orbit._numpyro_bridge import (
    _get_or_build_cached,
    _init_dict_to_z_flat,
    make_constrain,
)
from photomancy.orbit.data import OrbitData
from photomancy.orbit.init import elements_to_sites
from photomancy.posterior import SamplePosterior


class OrbitProblem(eqx.Module):
    """An orbit fit reduced to a flat logdensity for the generic backends.

    Attributes:
        logdensity: ``z -> scalar`` log-density over the flat unconstrained position.
        to_physical: ``z -> dict`` mapping a flat position to named physical
            parameters (``T``, ``e``, ``a``, ...).
        init_to_z: ``init_dict -> z`` converting a blind-initializer dict (e.g. from
            ``photomancy.orbit.init.find_init``) to a flat unconstrained position.
        unflatten: ``z -> dict`` of raw (unconstrained) sample sites.
        constrain: ``raw-site dict -> physical-site dict`` applying each site's forward
            constraint bijector (the seam the EIG forward differentiates through).
        param_names: The unconstrained sample-site names, in flat order.
    """

    logdensity: Callable = eqx.field(static=True)
    to_physical: Callable = eqx.field(static=True)
    init_to_z: Callable = eqx.field(static=True)
    unflatten: Callable = eqx.field(static=True)
    constrain: Callable = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)


def build_orbit_logdensity(
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
    n_planets=1,
    seed=0,
):
    """Build a flat orbit ``logdensity`` (and its z<->physical maps) for a backend.

    Traces the NumPyro orbit model once (cached), pads the data to the model's
    ``MAX_*`` shapes, and binds the data into ``logdensity = -potential``. Feed the
    returned ``OrbitProblem.logdensity`` to any generic backend; seed multi-start /
    MCMC from ``init_to_z(find_init...(...))`` and map results back with
    ``to_physical``.

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
        ecc_prior: Eccentricity prior name (see
            :func:`~photomancy.orbit.model.build_model`).
        jitter_scale: Scale for the HalfNormal RV-jitter prior.
        n_planets: Number of planets to fit (the model vmaps over planets and
            indexes astrometry by ``planet_id``).
        seed: PRNG seed for the one-time model trace.

    Returns:
        An :class:`OrbitProblem`.
    """
    data = OrbitData(
        rv=rv_data,
        relative_astrom=relative_astrom_data,
        stellar_astrom=stellar_astrom_data,
        pm_anomaly=pm_anomaly_data,
        null=null_data,
        imaging=imaging_data,
    )
    cached = _get_or_build_cached(
        has_rv=data.rv is not None,
        has_relative_astrom=data.relative_astrom is not None,
        has_stellar_astrom=data.stellar_astrom is not None,
        has_pm_anomaly=data.pm_anomaly is not None,
        has_null=data.null is not None,
        has_imaging=data.imaging is not None,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
        n_planets=n_planets,
        seed=seed,
    )
    model_args = (Ms, dist_pc, data.padded())

    unflatten = cached["unflatten"]
    potential_fn = cached["potential_fn_factory"](*model_args)

    def logdensity(z):
        return -potential_fn(unflatten(z))

    try:
        raw_postprocess = cached["postprocess_fn_factory"](*model_args)
    except TypeError:
        raw_postprocess = cached["postprocess_fn_factory"]

    def to_physical(z):
        phys = raw_postprocess(unflatten(z))
        return {k: jnp.squeeze(v) for k, v in phys.items()}

    def init_to_z(init_dict):
        return _init_dict_to_z_flat(
            init_dict, cached["z_template"], cached["inv_transforms"]
        )

    # Forward (unconstrained -> physical) bijectors are the inverse of the cached
    # inverse transforms; the closure keeps them off the module as static array leaves.
    constrain = make_constrain(
        {name: t.inv for name, t in cached["inv_transforms"].items()}
    )

    return OrbitProblem(
        logdensity=logdensity,
        to_physical=to_physical,
        init_to_z=init_to_z,
        unflatten=unflatten,
        constrain=constrain,
        param_names=cached["param_names"],
    )


def to_unconstrained(posterior, problem, Ms):
    """Map a physical orbit SamplePosterior to the model's unconstrained z-space.

    OFTI / grid_search emit physical rows ``(a, e, cos_i, W, cos_w, sin_w, tp)``; the
    EIG needs the model's unconstrained ``z``. Converts ``a -> period`` (Kepler) then
    each row -> raw sites -> ``z`` via ``problem.init_to_z`` (vmapped, no Python loop).

    Args:
        posterior: A physical :class:`~photomancy.posterior.SamplePosterior` whose
            ``param_names`` include ``a, e, cos_i, W, cos_w, sin_w, tp``.
        problem: The :class:`OrbitProblem` defining the target ``z`` space.
        Ms: Stellar mass (kg), for the ``a -> period`` Kepler inversion.

    Returns:
        A :class:`~photomancy.posterior.SamplePosterior` in ``z`` space (``param_names``
        are ``problem.param_names``), preserving the input weights and evidence.
    """
    cols = {
        name: posterior.samples[:, i] for i, name in enumerate(posterior.param_names)
    }
    period = 2.0 * jnp.pi * jnp.sqrt(cols["a"] ** 3 / (G * Ms))

    def one(t, e, cos_i, w_node, cos_w, sin_w, tp):
        return problem.init_to_z(
            elements_to_sites(t, e, cos_i, w_node, cos_w, sin_w, tp, n_planets=1)
        )

    z = jax.vmap(one)(
        period,
        cols["e"],
        cols["cos_i"],
        cols["W"],
        cols["cos_w"],
        cols["sin_w"],
        cols["tp"],
    )
    return SamplePosterior(
        samples=z,
        log_weights=posterior.log_weights,
        evidence=posterior.evidence,
        param_names=problem.param_names,
    )
