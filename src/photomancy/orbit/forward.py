"""Forward models for orbit fitting -- pure JAX, no sampler dependency.

Each function takes scalar orbital parameters + a time array and returns
predicted observable quantities. They call ``orbix.equations`` building blocks
directly, using the Normal->Disk parameterization ``(cos_w, sin_w)`` to avoid
all trig calls.

All computations use natural units:
- Distances: AU
- Times: days
- Masses: kg
- Angles: radians (internal), arcsec (output where noted)
"""

import jax.numpy as jnp
from hwoutils.constants import G, two_pi
from orbix.equations.orbit import AB_matrices_reduced, mean_anomaly_tp, mean_motion
from orbix.equations.phase import lambert_phase_exact
from orbix.equations.propagation import single_r
from orbix.kepler.core import diff_solve_trig

two_pi_G = two_pi * G


def predict_rv(times, T, Ms, Mp_sini, e, cos_w, sin_w, tp):
    """Predict radial velocity signal for a single planet.

    Uses the singularity-free RV formula with the Normal->Disk parameterization.
    The ``sqrt(1-e^2)`` factor in K cancels with the orbital velocity prefactor,
    giving a formula that is well-behaved at e=0.

    Args:
        times: Observation epochs (days). Shape ``(N,)``.
        T: Orbital period (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        Mp_sini: Minimum planet mass, ``Mp * sin(i)`` (kg). Scalar.
        e: Eccentricity. Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        tp: Time of periapsis passage (days). Scalar.

    Returns:
        rv_model: Predicted RV (AU/day). Shape ``(N,)``.
            Convert to m/s by multiplying by ``hwoutils.constants.AU2m / d2s``.
    """
    # Mean motion and mean anomaly
    n = two_pi / T
    M = mean_anomaly_tp(times, n, tp)

    # Solve Kepler's equation (differentiable)
    sinE, cosE = diff_solve_trig(M, e)

    # e-independent base velocity scale: K0 = K * sqrt(1-e^2)
    # K = (2piG/T)^(1/3) * Mp_sini / Ms^(2/3) / sqrt(1-e^2)
    # K0 = (2piG/T)^(1/3) * Mp_sini / Ms^(2/3)
    K0 = (two_pi_G / T) ** (1.0 / 3.0) * Mp_sini / Ms ** (2.0 / 3.0)

    # Singularity-free RV: cancellation of sqrt(1-e^2) factors
    sqrt_1me2 = jnp.sqrt(1.0 - e**2)
    denom = 1.0 - e * cosE
    rv_model = -(K0 / denom) * (sqrt_1me2 * cosE * cos_w - sinE * sin_w)

    return rv_model


def predict_astrometry(times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc):
    """Predict relative astrometry (RA, DEC offsets) for a single planet.

    Projects the 3D position vector onto the sky plane and converts to
    angular separation in arcseconds.

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

    Returns:
        Tuple of ``(ra_pred, dec_pred)``, each shape ``(N,)`` in arcsec.
    """
    # Derived quantities
    mu = G * Ms
    n = mean_motion(a, mu)
    M = mean_anomaly_tp(times, n, tp)
    sinE, cosE = diff_solve_trig(M, e)

    # Trig values for AB matrices
    sin_i = jnp.sqrt(1.0 - cos_i**2)
    sqrt_1me2 = jnp.sqrt(1.0 - e**2)
    sinW, cosW = jnp.sin(W), jnp.cos(W)

    # AB matrices in AU
    A, B = AB_matrices_reduced(a, sqrt_1me2, sin_i, cos_i, sinW, cosW, sin_w, cos_w)

    # Position vector: r = A*(cosE - e) + B*sinE, shape (3, N)
    r = single_r(A, B, e, sinE, cosE)

    # Project to sky plane (RA = x, DEC = y) and convert AU -> arcsec
    # At distance d_pc: angle_arcsec = separation_AU / d_pc
    # (since 1 AU at 1 pc = 1 arcsec by definition)
    ra_pred = r[0] / dist_pc  # arcsec
    dec_pred = r[1] / dist_pc  # arcsec

    return ra_pred, dec_pred


def predict_photometry(times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, Lambda, dist_pc):
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

    Returns:
        Tuple of ``(alpha_arcsec, dMag)``, each shape ``(N,)``.
            - ``alpha_arcsec``: Angular separation in arcsec.
            - ``dMag``: Delta-magnitude (larger = fainter).
    """
    # Orbital propagation
    mu = G * Ms
    n = mean_motion(a, mu)
    M = mean_anomaly_tp(times, n, tp)
    sinE, cosE = diff_solve_trig(M, e)

    # Trig values
    sin_i = jnp.sqrt(1.0 - cos_i**2)
    sqrt_1me2 = jnp.sqrt(1.0 - e**2)
    sinW, cosW = jnp.sin(W), jnp.cos(W)

    A, B = AB_matrices_reduced(a, sqrt_1me2, sin_i, cos_i, sinW, cosW, sin_w, cos_w)

    # Position vector in AU, shape (3, N)
    r = single_r(A, B, e, sinE, cosE)

    # Angular separation on the sky
    # Sky-plane distance: sqrt(x^2 + y^2), observer along z-axis
    sky_sep_AU = jnp.sqrt(r[0] ** 2 + r[1] ** 2)
    alpha_arcsec = sky_sep_AU / dist_pc

    # Phase angle: cos(beta) = -z/|r| (z toward observer, so phase angle is
    # angle between star-planet and star-observer vectors)
    r_mag = jnp.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    r_mag_safe = jnp.maximum(r_mag, 1e-30)
    cos_beta = -r[2] / r_mag_safe
    sin_beta = jnp.sqrt(jnp.maximum(1.0 - cos_beta**2, 0.0))

    # Lambert phase function
    phase = lambert_phase_exact(cos_beta, sin_beta)

    # Contrast: Fp/Fs = Lambda * phase / r^2
    # Lambda = Ag * Rp^2 (both in AU^2 when Rp is in AU)
    contrast = Lambda * phase / jnp.maximum(r_mag**2, 1e-60)

    # Delta-magnitude
    dMag = -2.5 * jnp.log10(jnp.maximum(contrast, 1e-300))

    return alpha_arcsec, dMag
