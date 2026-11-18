"""One-orbit predictions for tests, composed from orbix observables.

Each helper builds the orbit with ``frame_from_tp`` (orbix ``FrameOrbit``) at the total
mass ``Ms + Mp`` and reads the observable through orbix or the explicit system terms.
Positions and velocities use the right-handed (east, north, away) rows.
"""

import jax.numpy as jnp
from hwoutils.constants import G, two_pi
from orbix.system.observables import distance_AU, phase_angle_rad, separation_arcsec

from photomancy.orbit._frame import (
    frame_from_tp,
    plane_from_motion,
    relative_offsets_arcsec,
)
from photomancy.orbit._photometry import lambertian_dmag, lambertian_from_area
from photomancy.system_terms import _proper_motion_anomaly


def predict_relative_astrometry(
    times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc, *, Mp=0.0
):
    """East and north offsets (arcsec) of the planet from the star."""
    return relative_offsets_arcsec(
        times, a, e, cos_i, W, cos_w, sin_w, tp, Ms + Mp, dist_pc
    )


def predict_stellar_astrometry(
    times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Mp, dist_pc
):
    """East and north reflex offsets (arcsec) of the star about the barycenter."""
    ra, dec = predict_relative_astrometry(
        times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc, Mp=Mp
    )
    fraction = Mp / (Ms + Mp)
    return -fraction * ra, -fraction * dec


def predict_pm_anomaly(
    t_hip,
    t_gaia,
    gaia_window,
    n_epochs,
    a,
    e,
    cos_i,
    W,
    cos_w,
    sin_w,
    tp,
    Ms,
    Mp,
    dist_pc,
):
    """Hipparcos-Gaia proper-motion anomaly (arcsec/day) of the star."""

    def reflex_path(time_d):
        return jnp.stack(
            predict_stellar_astrometry(
                time_d, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Mp, dist_pc
            )
        )

    return _proper_motion_anomaly(t_hip, t_gaia, gaia_window, n_epochs, reflex_path)


def predict_rv(times, T, Ms, Mp_sini, e, cos_w, sin_w, tp, *, Mp=0.0):
    """Stellar radial velocity (AU/day) of a minimum-mass companion.

    The singularity-free form ``-a Mp_sini / (Ms + Mp) (sin_w Xdot + cos_w Ydot)``
    with dimensionless perifocal rates; ``Mp`` only enters the Kepler total mass.
    """
    mu = G * (Ms + Mp)
    a = (mu * (T / two_pi) ** 2) ** (1.0 / 3.0)
    orbit = plane_from_motion(e, tp, two_pi / T)
    _, rate = orbit.perifocal_coordinates(times)
    return -a * Mp_sini / (Ms + Mp) * (sin_w * rate[..., 0] + cos_w * rate[..., 1])


def predict_photometry(
    times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Lambda, dist_pc, *, Mp=0.0
):
    """Angular separation (arcsec) and Lambert dMag of the planet."""
    orbit = frame_from_tp(a, e, cos_i, W, cos_w, sin_w, tp, G * (Ms + Mp))
    r, _ = orbit.at(times)
    model, radius_Rearth = lambertian_from_area(Lambda)
    return separation_arcsec(r, dist_pc), lambertian_dmag(
        phase_angle_rad(r), distance_AU(r), model, radius_Rearth
    )
