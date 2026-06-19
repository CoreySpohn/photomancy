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
import jax.numpy as jnp

from photomancy.orbit.laplace import (
    _get_or_build_cached,
    _init_dict_to_z_flat,
    _pad_orbit_data,
)


class OrbitProblem(eqx.Module):
    """An orbit fit reduced to a flat logdensity for the generic backends.

    Attributes:
        logdensity: ``z -> scalar`` log-density over the flat unconstrained position.
        to_physical: ``z -> dict`` mapping a flat position to named physical
            parameters (``T``, ``e``, ``a``, ...).
        init_to_z: ``init_dict -> z`` converting a blind-initializer dict (e.g. from
            ``photomancy.orbit.init.find_init``) to a flat unconstrained position.
        param_names: The unconstrained sample-site names, in flat order.
    """

    logdensity: Callable = eqx.field(static=True)
    to_physical: Callable = eqx.field(static=True)
    init_to_z: Callable = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)


def build_orbit_logdensity(
    Ms,
    dist_pc,
    *,
    rv_data=None,
    astrom_data=None,
    null_data=None,
    imaging_data=None,
    log_P_range=(1.0, 4.0),
    log_Mp_range=(-2.0, 4.0),
    log_Rp_range=(-5.0, -2.5),
    log_Ag_range=(-2.0, 0.0),
    ecc_prior="kipping13",
    jitter_scale=1e-10,
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
        astrom_data: An :class:`~photomancy.orbit.data.AstromData`, or ``None``.
        null_data: A :class:`~photomancy.orbit.data.NullData`, or ``None``.
        imaging_data: An :class:`~photomancy.orbit.data.ImagingData`, or ``None``.
        log_P_range: ``(min, max)`` for the ``log10(period/days)`` prior.
        log_Mp_range: ``(min, max)`` for the ``log10(mass/M_earth)`` prior.
        log_Rp_range: ``(min, max)`` for the ``log10(Rp/AU)`` prior.
        log_Ag_range: ``(min, max)`` for the ``log10(geometric albedo)`` prior.
        ecc_prior: Eccentricity prior name (see
            :func:`~photomancy.orbit.model.build_model`).
        jitter_scale: Scale for the HalfNormal RV-jitter prior.
        seed: PRNG seed for the one-time model trace.

    Returns:
        An :class:`OrbitProblem`.
    """
    cached = _get_or_build_cached(
        has_rv=rv_data is not None,
        has_astrom=astrom_data is not None,
        has_null=null_data is not None,
        has_imaging=imaging_data is not None,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
        seed=seed,
    )
    rv_data, astrom_data, null_data, imaging_data = _pad_orbit_data(
        rv_data, astrom_data, null_data, imaging_data
    )
    model_args = (Ms, dist_pc, rv_data, astrom_data, null_data, imaging_data)

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

    return OrbitProblem(
        logdensity=logdensity,
        to_physical=to_physical,
        init_to_z=init_to_z,
        param_names=cached["param_names"],
    )
