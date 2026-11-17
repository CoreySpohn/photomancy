"""Forward models for orbit fitting -- pure JAX, no sampler dependency.

Each function takes scalar orbital parameters + a time array and returns
predicted observable quantities. Orbital kinematics come from
``orbix.system.FrameOrbit``, using the Normal->Disk parameterization
``(cos_w, sin_w)``.

All computations use natural units:
- Distances: AU
- Times: days
- Masses: kg
- Angles: radians (internal), arcsec (output where noted)
"""

import jax.numpy as jnp
from hwoutils.constants import G, two_pi
from hwoutils.conversions import contrast_to_delta_mag
from orbix.system.observables import distance_AU, phase_angle_rad, separation_arcsec

from photomancy.orbit._frame import frame_from_tp, plane_from_motion
from photomancy.orbit._photometry import lambertian_from_area


def predict_rv(times, T, Ms, Mp_sini, e, cos_w, sin_w, tp, *, Mp=0.0):
    """Predict radial velocity signal for a single planet.

    Uses the singularity-free RV formula with the Normal->Disk parameterization.
    The ``sqrt(1-e^2)`` factor in K cancels with the orbital velocity prefactor,
    giving a formula that is well-behaved at e=0. When the true mass ``Mp``
    is supplied, the amplitude uses ``Ms + Mp``. If only ``Mp sin i`` is
    available (an RV-only minimum-mass fit), omitting ``Mp`` uses the
    ``Mp << Ms`` approximation. Its fractional overestimate in K is
    ``(2/3)(Mp/Ms) + O((Mp/Ms)^2)``.

    With dimensionless perifocal rates (Xdot, Ydot), the velocity is
    ``-a * Mp_sini / (Ms + Mp) * (sin_w * Xdot + cos_w * Ydot)``.
    This minimum-mass expression needs no inclination.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        T: Orbital period (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        Mp_sini: Minimum planet mass, ``Mp * sin(i)`` (kg). Scalar.
        e: Eccentricity. Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        tp: Time of periapsis passage (days). Scalar.
        Mp: True planet mass (kg), if known. Defaults to zero in Kepler
            denominators for the minimum-mass approximation.

    Returns:
        rv_model: Predicted RV (AU/day). Shape ``(N,)``.
            Convert to m/s by multiplying by ``hwoutils.constants.AU2m / d2s``.
    """
    mu = G * (Ms + Mp)
    a = (mu * (T / two_pi) ** 2) ** (1.0 / 3.0)
    orbit = plane_from_motion(e, tp, two_pi / T)
    _, rate = orbit.perifocal_coordinates(times)
    rv_model = -a * Mp_sini / (Ms + Mp) * (sin_w * rate[..., 0] + cos_w * rate[..., 1])

    return rv_model


def predict_relative_astrometry(
    times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc, *, Mp=0.0
):
    """Predict relative astrometry (RA, DEC offsets) for a single planet.

    Projects the 3D position vector onto the sky plane and converts to
    angular separation in arcseconds. RA means Delta alpha* = Delta RA cos Dec
    (east); Dec is north. Raw positions and velocities use the right-handed
    (east, north, away) frame; dz/dt is positive when receding. The node angle
    is measured north through east.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        a: Semi-major axis (AU). Scalar.
        e: Eccentricity. Scalar.
        cos_i: Cosine of inclination. Scalar.
        W: Longitude of ascending node (radians). Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        tp: Time of periapsis passage (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        Mp: True planet mass (kg), if known. Default zero assumes ``Mp << Ms``.

    Returns:
        Tuple of ``(ra_pred, dec_pred)``, each shape ``(N,)`` in arcsec.
    """
    orbit = frame_from_tp(a, e, cos_i, W, cos_w, sin_w, tp, G * (Ms + Mp))
    r, _ = orbit.at(times)

    # Raw rows are (east, north, away); return (east, north) in arcsec.
    # At distance d_pc: angle_arcsec = separation_AU / d_pc
    # (since 1 AU at 1 pc = 1 arcsec by definition)
    ra_pred = r[..., 0] / dist_pc  # arcsec
    dec_pred = r[..., 1] / dist_pc  # arcsec

    return ra_pred, dec_pred


def predict_stellar_astrometry(
    times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Mp, dist_pc
):
    """Predict stellar-reflex astrometry (RA, DEC offsets of the star).

    The star orbits the system barycenter on a scaled, sign-flipped copy of the
    planet's relative orbit: its offset is the relative astrometry times the mass
    fraction ``Mp / (Ms + Mp)``, in the opposite direction. Unlike relative
    astrometry this depends on the planet mass, so RV (which constrains ``Mp sin i``)
    and stellar astrometry (which resolves ``i``) together yield the dynamical mass.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        a: Semi-major axis of the relative orbit (AU). Scalar.
        e: Eccentricity. Scalar.
        cos_i: Cosine of inclination. Scalar.
        W: Longitude of ascending node (radians). Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        tp: Time of periapsis passage (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        Mp: Planet mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.

    Returns:
        Tuple of ``(ra_pred, dec_pred)``, the star's reflex offset, each shape
        ``(N,)`` in arcsec.
    """
    ra_rel, dec_rel = predict_relative_astrometry(
        times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc, Mp=Mp
    )
    frac = Mp / (Ms + Mp)
    return -frac * ra_rel, -frac * dec_rel


def _reflex_proper_motion(
    t_center, window, n_epochs, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Mp, dist_pc
):
    """Least-squares proper motion of the stellar reflex over a mission window.

    Samples ``n_epochs`` epochs across ``[t_center - window/2, t_center + window/2]``,
    evaluates the reflex sky-path, and returns the linear slope (the proper motion a
    mission would report by fitting a moving photocenter). Octofitter's 25-epoch
    approximation to the smearing of the photocenter, without htof.
    """
    ts = jnp.linspace(t_center - 0.5 * window, t_center + 0.5 * window, n_epochs)
    ra, dec = predict_stellar_astrometry(
        ts, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Mp, dist_pc
    )
    return _path_proper_motion(ts, ra, dec)


def _path_proper_motion(ts, ra, dec):
    """Least-squares east/north slopes of an angular path at day epochs."""
    dt = ts - jnp.mean(ts)
    denom = jnp.sum(dt**2)
    pm_ra = jnp.sum(dt * (ra - jnp.mean(ra))) / denom
    pm_dec = jnp.sum(dt * (dec - jnp.mean(dec))) / denom
    return jnp.stack([pm_ra, pm_dec])


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
    """Predict the Hipparcos-Gaia proper-motion anomaly of the star (RA, DEC).

    The anomaly is the difference between the instantaneous Gaia proper motion (the
    slope of the reflex sky-path over the Gaia window) and the long-baseline
    Hipparcos-to-Gaia mean proper motion (the scaled positional difference). The
    barycenter proper motion cancels in the difference, so the anomaly is a pure
    reflex signal. Combined with the orbit shape (from relative astrometry or imaging),
    it constrains the planet mass. Uses Octofitter's 25-epoch path-fit approximation.

    Args:
        t_hip: Hipparcos mean epoch (days, in the orbit's time frame). Scalar.
        t_gaia: Gaia mean epoch (days, same frame). Scalar.
        gaia_window: Gaia mission duration (days) for the proper-motion fit. Scalar.
        n_epochs: Number of epochs to sample per mission window (static int).
        a: Semi-major axis of the relative orbit (AU). Scalar.
        e: Eccentricity. Scalar.
        cos_i: Cosine of inclination. Scalar.
        W: Longitude of ascending node (radians). Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        tp: Time of periapsis passage (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        Mp: Planet mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.

    Returns:
        The proper-motion anomaly ``(pm_ra, pm_dec)``, shape ``(2,)`` in arcsec/day.
    """

    def reflex_path(time_d):
        return jnp.stack(
            predict_stellar_astrometry(
                time_d, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Mp, dist_pc
            )
        )

    return _proper_motion_anomaly(t_hip, t_gaia, gaia_window, n_epochs, reflex_path)


def _proper_motion_anomaly(t_hip, t_gaia, window, n_epochs, reflex_path):
    """Gaia window slope minus Hipparcos-Gaia slope of an east/north path.

    reflex_path maps day epochs (N,) to angular offsets (2, N) in arcseconds.
    Both system terms and single-orbit predictions use this mission-window fit.
    """
    ts = jnp.linspace(t_gaia - 0.5 * window, t_gaia + 0.5 * window, n_epochs)
    time_d = jnp.concatenate((ts, jnp.atleast_1d(t_hip), jnp.atleast_1d(t_gaia)))
    path = reflex_path(time_d)
    mu_gaia = _path_proper_motion(ts, *path[:, :n_epochs])
    mu_hg = (path[:, -1] - path[:, -2]) / (t_gaia - t_hip)
    return mu_gaia - mu_hg


def predict_photometry(
    times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Lambda, dist_pc, *, Mp=0.0
):
    """Predict angular separation and delta-magnitude for a single planet.

    Computes the planet's on-sky separation (arcsec) and apparent brightness
    relative to the host star (dMag) using the Lambert phase function.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        a: Semi-major axis (AU). Scalar.
        e: Eccentricity. Scalar.
        cos_i: Cosine of inclination. Scalar.
        W: Longitude of ascending node (radians). Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        tp: Time of periapsis passage (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        Lambda: Photometric area ``Ag * Rp^2`` (AU^2). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        Mp: True planet mass (kg), if known. Default zero assumes ``Mp << Ms``.

    Returns:
        Tuple of ``(alpha_arcsec, dMag)``, each shape ``(N,)``.
            - ``alpha_arcsec``: Angular separation in arcsec.
            - ``dMag``: Delta-magnitude (larger = fainter).
    """
    orbit = frame_from_tp(a, e, cos_i, W, cos_w, sin_w, tp, G * (Ms + Mp))
    r, _ = orbit.at(times)

    alpha_arcsec = separation_arcsec(r, dist_pc)
    model, radius_Rearth = lambertian_from_area(Lambda)
    contrast = model.contrast(
        phase_angle_rad(r)[None, :], distance_AU(r)[None, :], None, radius_Rearth
    )[0]

    # Delta-magnitude
    dMag = contrast_to_delta_mag(jnp.maximum(contrast, 1e-300))

    return alpha_arcsec, dMag
