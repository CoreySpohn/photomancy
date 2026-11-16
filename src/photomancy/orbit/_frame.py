"""Construct frame orbits from the fitting parameterizations."""

import jax.numpy as jnp
from orbix.system import FrameOrbit


def frame_from_tp(a, e, cos_i, W, cos_w, sin_w, tp, mu):
    """Build an AU/day orbit with periapsis at tp (days)."""
    return FrameOrbit.from_classical_trig(
        a_AU=a,
        e=e,
        cos_i=cos_i,
        sin_i=jnp.sqrt(1.0 - cos_i**2),
        cos_W=jnp.cos(W),
        sin_W=jnp.sin(W),
        cos_w=cos_w,
        sin_w=sin_w,
        cos_M0=1.0,
        sin_M0=0.0,
        mu_AU3_per_d2=mu,
        epoch_d=tp,
    )


def plane_from_motion(e, tp, n):
    """Build a unit-size orbit for dimensionless perifocal coordinates."""
    e, tp, n = jnp.broadcast_arrays(e, tp, n)
    return FrameOrbit(
        jnp.ones_like(e),
        n,
        e,
        jnp.zeros_like(e),
        jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (*e.shape, 4)),
        tp,
    )


def plane_from_phase(e, ln_period, tau, t_ref):
    """Build the TI plane with periastron phase tau relative to t_ref."""
    period = jnp.exp(ln_period)
    return plane_from_motion(e, t_ref + period * tau, 2 * jnp.pi / period)
