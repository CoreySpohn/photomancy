"""Independent NumPy sky oracle, without orbix or photomancy propagation."""

import numpy as np


def sky_position(times, *, T, a, e, cos_i, W, cos_w, sin_w, tp, dist_pc):
    """Return (east, north) arcsec from the independent 3D position."""
    return (
        position_au(
            times, T=T, a=a, e=e, cos_i=cos_i, W=W, cos_w=cos_w, sin_w=sin_w, tp=tp
        )[:2]
        / dist_pc
    )


def position_au(times, *, T, a, e, cos_i, W, cos_w, sin_w, tp):
    """Return right-handed (east, north, away) rows in AU.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186, Eqs. 6-9. Here Y includes sqrt(1-e^2), so F and G
    are the classical Thiele-Innes constants.
    """
    mean = (2 * np.pi * (np.asarray(times) - tp) / T + np.pi) % (2 * np.pi) - np.pi
    eccentric = mean.copy()
    for _ in range(50):
        step = (eccentric - e * np.sin(eccentric) - mean) / (1 - e * np.cos(eccentric))
        eccentric -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    else:
        raise AssertionError("Newton Kepler solve did not converge")
    X = np.cos(eccentric) - e
    Y = np.sqrt(1 - e**2) * np.sin(eccentric)
    cW, sW = np.cos(W), np.sin(W)
    A = a * (cW * cos_w - sW * sin_w * cos_i)
    B = a * (sW * cos_w + cW * sin_w * cos_i)
    F = a * (-cW * sin_w - sW * cos_w * cos_i)
    G = a * (-sW * sin_w + cW * cos_w * cos_i)
    sin_i = np.sqrt(1 - cos_i**2)
    C = a * sin_w * sin_i
    H = a * cos_w * sin_i
    x_north = A * X + F * Y
    y_east = B * X + G * Y
    z = C * X + H * Y
    return np.array([y_east, x_north, z])
