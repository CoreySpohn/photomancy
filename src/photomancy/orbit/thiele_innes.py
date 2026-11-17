r"""Thiele-Innes linear orbit fitter -- pure JAX.

Given trial values of ``(T, e, tp)`` and relative astrometry data, solves
for the optimal Thiele-Innes constants ``(A, B, F, G)`` via weighted
least squares. Since the astrometric model is *linear* in the TI constants
for fixed ``(T, e, tp)``, this is an O(N) exact solve -- no iterative
optimization needed.

The TI constants encode the 3D orientation (inclination, ascending node,
argument of periapsis) and the semi-major axis:

.. math::

    A = a (\\cos\\Omega\\cos\\omega - \\sin\\Omega\\sin\\omega\\cos i)

    B = a (\\sin\\Omega\\cos\\omega + \\cos\\Omega\\sin\\omega\\cos i)

    F = a\\sqrt{1-e^2} (-\\cos\\Omega\\sin\\omega - \\sin\\Omega\\cos\\omega\\cos i)

    G = a\\sqrt{1-e^2} (-\\sin\\Omega\\sin\\omega + \\cos\\Omega\\cos\\omega\\cos i)

These keep the classical letters: (A, F) multiply north and (B, G) east.
``FrameOrbit`` stores positions in the right-handed (east, north, away)
frame, with positive dz/dt receding. Its sky rows use (B, A) and (G, F).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, two_pi
from hwoutils.conversions import arcsec_to_au
from orbix.equations import period_to_sma, thiele_innes_to_elements
from orbix.system.linear import fit_linear_map

from photomancy.orbit._frame import frame_from_tp

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TIFitResult(eqx.Module):
    """Result of a Thiele-Innes linear fit.

    Attributes:
        A: TI constant A (AU). Scalar.
        B: TI constant B (AU). Scalar.
        F: TI constant F including ``sqrt(1 - e**2)`` (AU). Scalar.
        G_ti: TI constant G including ``sqrt(1 - e**2)`` (AU). Scalar.
            Named ``G_ti`` to avoid collision with the gravitational constant.
        a: Semi-major axis (AU). Scalar.
        e: Eccentricity (input). Scalar.
        cos_i: Cosine of inclination. Scalar.
        W: Longitude of ascending node (radians). Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        T: Orbital period (days, input). Scalar.
        tp: Time of periapsis passage (days, input). Scalar.
        chi2: Sum of squared weighted residuals. Scalar.
        log_likelihood: Gaussian log-likelihood of the fit. Scalar.
    """

    A: jnp.ndarray
    B: jnp.ndarray
    F: jnp.ndarray
    G_ti: jnp.ndarray
    a: jnp.ndarray
    e: jnp.ndarray
    cos_i: jnp.ndarray
    W: jnp.ndarray
    cos_w: jnp.ndarray
    sin_w: jnp.ndarray
    T: jnp.ndarray
    tp: jnp.ndarray
    chi2: jnp.ndarray
    log_likelihood: jnp.ndarray


def thiele_innes_fit(relative_astrom_data, T, e, tp, Ms, dist_pc):
    """Fit Thiele-Innes constants via linear least squares.

    Given trial ``(T, e, tp)`` and astrometry data, finds the optimal
    ``(A, B, F, G)`` orientation constants. This is exact and O(N) since
    the model is linear in these constants for fixed ``(T, e, tp)``.

    Args:
        relative_astrom_data: An
            :class:`~photomancy.orbit.data.RelativeAstromData` instance.
        T: Trial orbital period (days). Scalar.
        e: Trial eccentricity. Scalar.
        tp: Trial time of periapsis passage (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.

    Returns:
        A :class:`TIFitResult` with the best-fit TI constants, recovered
        orbital elements, and goodness-of-fit statistics.
    """
    data = relative_astrom_data
    mask = data.is_valid
    # The Keplerian size sets the trial mean motion; the fitted size is free.
    orbit_shape = frame_from_tp(period_to_sma(T, Ms), e, 1.0, 0.0, 1.0, 0.0, tp, G * Ms)
    block, _, chi2, _ = fit_linear_map(
        orbit_shape,
        data.times,
        arcsec_to_au(data.ra, dist_pc),
        arcsec_to_au(data.dec, dist_pc),
        (
            arcsec_to_au(data.ra_err, dist_pc),
            arcsec_to_au(data.dec_err, dist_pc),
            data.corr,
        ),
        mask=mask,
    )
    (B, G_ti), (A, F) = block
    a, cos_i, W, w = thiele_innes_to_elements(A, B, F, G_ti)
    scale = jnp.sqrt(1.0 - e**2)

    # Normalize in the observed angular units, masking before taking logarithms.
    sx = jnp.where(mask, data.ra_err, 1.0)
    sy = jnp.where(mask, data.dec_err, 1.0)
    corr = jnp.where(mask, data.corr, 0.0)
    log_norm = jnp.log(two_pi) + jnp.log(sx) + jnp.log(sy) + 0.5 * jnp.log1p(-(corr**2))
    ll = -0.5 * chi2 - jnp.sum(jnp.where(mask, log_norm, 0.0))

    return TIFitResult(
        A=A,
        B=B,
        F=F * scale,
        G_ti=G_ti * scale,
        a=a,
        e=e,
        cos_i=cos_i,
        W=W,
        cos_w=jnp.cos(w),
        sin_w=jnp.sin(w),
        T=T,
        tp=tp,
        chi2=chi2,
        log_likelihood=ll,
    )


def thiele_innes_grid_search(
    relative_astrom_data, Ms, dist_pc, log_T_grid, e_grid, n_tp=30
):
    """Search over (T, e, tp) grid with Thiele-Innes linearization.

    For each ``(T, e)`` pair, evaluates ``n_tp`` uniformly spaced
    tp values in ``[0, T]`` and returns the result with the best
    log-likelihood.

    The grid search is fully vectorized via ``jax.vmap`` for GPU
    acceleration.

    Args:
        relative_astrom_data: An
            :class:`~photomancy.orbit.data.RelativeAstromData` instance.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        log_T_grid: Array of log10(T/days) values to search.
        e_grid: Array of eccentricity values to search.
        n_tp: Number of tp grid points per period. Default 30.

    Returns:
        The :class:`TIFitResult` with the highest log-likelihood.
    """

    def _fit_single(log_T, e_val, tp_frac):
        """Fit at a single (T, e, tp) point."""
        T = 10.0**log_T
        tp = tp_frac * T
        return thiele_innes_fit(relative_astrom_data, T, e_val, tp, Ms, dist_pc)

    # Build the 3D grid
    tp_fracs = jnp.linspace(0.0, 1.0, n_tp, endpoint=False)

    # Create meshgrid of all combinations
    log_T_flat = jnp.repeat(jnp.asarray(log_T_grid), len(e_grid) * n_tp)
    e_flat = jnp.tile(
        jnp.repeat(jnp.asarray(e_grid), n_tp),
        len(log_T_grid),
    )
    tp_flat = jnp.tile(tp_fracs, len(log_T_grid) * len(e_grid))

    # Vectorize the fit over all grid points
    results = jax.vmap(_fit_single)(log_T_flat, e_flat, tp_flat)

    # Find the best log-likelihood
    best_idx = jnp.argmax(results.log_likelihood)

    # Extract the best result (tree_map to index into each leaf)
    best = jax.tree.map(lambda x: x[best_idx], results)

    return best
