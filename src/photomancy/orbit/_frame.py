"""Construct frame orbits from the fitting parameterizations."""

import jax.numpy as jnp
from hwoutils.constants import G
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


def sky_offsets_arcsec(orbit, time_d, dist_pc):
    """East and north offsets in arcsec at ``time_d`` (days) for a chart orbit.

    One AU at one parsec subtends one arcsecond. The orbit's batch axes come
    first, then the time axes.
    """
    position, _ = orbit.at(time_d)
    return position[..., 0] / dist_pc, position[..., 1] / dist_pc


def relative_offsets_arcsec(times, a, e, cos_i, W, cos_w, sin_w, tp, Ms_kg, dist_pc):
    """East and north offsets (arcsec) of a test planet around a star of ``Ms_kg``.

    Classical elements with the periapsis time ``tp`` (days); ``a`` in AU.
    """
    orbit = frame_from_tp(a, e, cos_i, W, cos_w, sin_w, tp, G * Ms_kg)
    return sky_offsets_arcsec(orbit, times, dist_pc)


def orbit_rows(orbit):
    """Rows ``(a, e, cos_i, W, cos_w, sin_w, tp)`` of chart orbits, shape ``B + (7,)``.

    The time of periapsis is the epoch minus the epoch mean anomaly over the mean
    motion, so it lies within one period before the orbit's epoch. Classical
    angles are undefined at e=0 and cos_i=+-1; the row is for output only.
    """
    elements = orbit.to_classical()
    tp = orbit.epoch_d - elements["M0_rad"] / orbit.n_rad_per_d
    return jnp.stack(
        [
            elements["a_AU"],
            elements["e"],
            jnp.cos(elements["i_rad"]),
            elements["W_rad"],
            jnp.cos(elements["w_rad"]),
            jnp.sin(elements["w_rad"]),
            tp,
        ],
        axis=-1,
    )
