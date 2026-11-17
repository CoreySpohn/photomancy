r"""Thiele-Innes linear orbit fitter -- pure JAX.

Given trial values of ``(T, e, tp)`` and relative astrometry data, solves
for the optimal Thiele-Innes constants ``(A, B, F, G)`` via ordinary
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
from orbix.equations import thiele_innes_to_elements

from photomancy.orbit._frame import plane_from_motion

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TIFitResult(eqx.Module):
    """Result of a Thiele-Innes linear fit.

    Attributes:
        A: TI constant A (AU). Scalar.
        B: TI constant B (AU). Scalar.
        F: TI constant F (AU). Scalar.
        G_ti: TI constant G (AU). Scalar. Named ``G_ti`` to avoid
            collision with the gravitational constant.
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


# ---------------------------------------------------------------------------
# Core linear solver
# ---------------------------------------------------------------------------


def _solve_abfg(X, Y, ra, dec, ra_err, dec_err, corr):
    """Solve for (A, B, F, G) via weighted least squares.

    The astrometric model (in arcsec) is::

        DEC(t) = (A*X(t) + F*Y(t)) / dist_pc
        RA(t)  = (B*X(t) + G*Y(t)) / dist_pc

    After absorbing dist_pc into (A, B, F, G), we can write this as two
    independent 2-parameter weighted linear regressions::

        DEC(t) = A'*X(t) + F'*Y(t)
        RA(t)  = B'*X(t) + G'*Y(t)

    where A' = A/dist_pc, etc. We solve for (A', F') and (B', G')
    separately, then multiply by dist_pc to recover (A, B, F, G) in AU.

    When RA/DEC are correlated (corr != 0), we use the full bivariate
    weighted least squares.

    Args:
        X: cosE - e, shape (N,).
        Y: sinE, shape (N,).
        ra: Observed RA offsets (arcsec), shape (N,).
        dec: Observed DEC offsets (arcsec), shape (N,).
        ra_err: RA uncertainties (arcsec), shape (N,).
        dec_err: DEC uncertainties (arcsec), shape (N,).
        corr: RA/DEC correlation coefficient, shape (N,).

    Returns:
        Tuple of (A', F', B', G') -- TI constants divided by dist_pc.
    """
    # For uncorrelated or weakly correlated data, RA and DEC decouple.
    # For the general case with correlations, we stack into a single
    # system and use the full inverse-covariance weighting.

    # Build the (2N, 2N) block-diagonal inverse-covariance matrix.
    # For the bivariate Gaussian at each point:
    #   C_i = [[sx^2, rho*sx*sy], [rho*sx*sy, sy^2]]
    #   C_i^{-1} = 1/(1-rho^2) * [[1/sx^2, -rho/(sx*sy)], [-rho/(sx*sy), 1/sy^2]]
    #
    # Rather than forming the full 2Nx2N matrix, we use the normal
    # equations directly: (D^T W D) beta = D^T W d

    one_minus_rho2 = jnp.maximum(1.0 - corr**2, 1e-30)

    # Weights for the normal equations (block-diagonal structure)
    w_xx = 1.0 / (dec_err**2 * one_minus_rho2)  # North self-weight
    w_yy = 1.0 / (ra_err**2 * one_minus_rho2)  # East self-weight
    w_xy = -corr / (ra_err * dec_err * one_minus_rho2)  # Cross-weight

    # Normal equation: D^T W D (4x4)
    # The (i,j) element is sum_k D_ki * W_kk' * D_k'j
    # With the block structure, we can compute this efficiently.

    # North-north block (weighted by w_xx)
    XwX = jnp.sum(w_xx * X * X)
    XwY = jnp.sum(w_xx * X * Y)
    YwY = jnp.sum(w_xx * Y * Y)

    # East-east block (weighted by w_yy)
    XvX = jnp.sum(w_yy * X * X)
    XvY = jnp.sum(w_yy * X * Y)
    YvY = jnp.sum(w_yy * Y * Y)

    # North-east cross block (weighted by w_xy)
    XcX = jnp.sum(w_xy * X * X)
    XcY = jnp.sum(w_xy * X * Y)
    YcX = jnp.sum(w_xy * Y * X)
    YcY = jnp.sum(w_xy * Y * Y)

    # 4x4 normal matrix
    # [A', F', B', G']
    normal_mat = jnp.array(
        [
            [XwX, XwY, XcX, XcY],
            [XwY, YwY, YcX, YcY],
            [XcX, XcY, XvX, XvY],
            [XcY, YcY, XvY, YvY],
        ]
    )

    # Right-hand side: D^T W d (4,)
    rhs = jnp.array(
        [
            jnp.sum(w_xx * X * dec) + jnp.sum(w_xy * X * ra),
            jnp.sum(w_xx * Y * dec) + jnp.sum(w_xy * Y * ra),
            jnp.sum(w_xy * X * dec) + jnp.sum(w_yy * X * ra),
            jnp.sum(w_xy * Y * dec) + jnp.sum(w_yy * Y * ra),
        ]
    )

    # Tikhonov regularization for rank-deficient systems (N=1,2 obs).
    # For well-determined systems the penalty is negligible.
    lambda_reg = 1e-4
    normal_mat_reg = normal_mat + lambda_reg * jnp.eye(4)

    # Solve the regularized 4x4 system
    params = jnp.linalg.solve(normal_mat_reg, rhs)

    return params[0], params[1], params[2], params[3]


def _compute_chi2_and_ll(X, Y, A_s, F_s, B_s, G_s, ra, dec, ra_err, dec_err, corr):
    """Compute chi^2 and log-likelihood for a TI fit.

    Args:
        X: Basis function cosE - e, shape (N,).
        Y: Basis function sinE, shape (N,).
        A_s: Scaled TI constant A / dist_pc.
        F_s: Scaled TI constant F / dist_pc.
        B_s: Scaled TI constant B / dist_pc.
        G_s: Scaled TI constant G / dist_pc.
        ra: Observed RA offsets (arcsec), shape (N,).
        dec: Observed DEC offsets (arcsec), shape (N,).
        ra_err: RA uncertainties (arcsec), shape (N,).
        dec_err: DEC uncertainties (arcsec), shape (N,).
        corr: RA/DEC correlation coefficients, shape (N,).

    Returns:
        Tuple of (chi2, log_likelihood).
    """
    dec_pred = A_s * X + F_s * Y
    ra_pred = B_s * X + G_s * Y

    dx = ra - ra_pred
    dy = dec - dec_pred

    one_minus_rho2 = jnp.maximum(1.0 - corr**2, 1e-30)
    z = (
        dx**2 / ra_err**2
        + dy**2 / dec_err**2
        - 2.0 * corr * dx * dy / (ra_err * dec_err)
    ) / one_minus_rho2

    chi2 = jnp.sum(z)
    log_norm = jnp.sum(
        jnp.log(ra_err * dec_err * jnp.sqrt(one_minus_rho2)) + jnp.log(2.0 * jnp.pi)
    )

    return chi2, -0.5 * chi2 - log_norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    orbit = plane_from_motion(e, tp, 2.0 * jnp.pi / T)
    xy, _ = orbit.perifocal_coordinates(relative_astrom_data.times)
    X = xy[..., 0]
    # F and G include sqrt(1-e^2), so retain the public unscaled Y basis.
    Y = xy[..., 1] / jnp.sqrt(1.0 - e**2)

    # Solve for (A', F', B', G') = (A/dist_pc, F/dist_pc, B/dist_pc, G/dist_pc)
    A_s, F_s, B_s, G_s = _solve_abfg(
        X,
        Y,
        relative_astrom_data.ra,
        relative_astrom_data.dec,
        relative_astrom_data.ra_err,
        relative_astrom_data.dec_err,
        relative_astrom_data.corr,
    )

    # Recover orbital elements from scaled constants
    scale = jnp.sqrt(1.0 - e**2)
    a_scaled, cos_i, W, w = thiele_innes_to_elements(A_s, B_s, F_s / scale, G_s / scale)
    cos_w, sin_w = jnp.cos(w), jnp.sin(w)

    # Convert a from arcsec-scale to AU
    a_AU = a_scaled * dist_pc

    # Compute chi^2 and log-likelihood
    chi2, ll = _compute_chi2_and_ll(
        X,
        Y,
        A_s,
        F_s,
        B_s,
        G_s,
        relative_astrom_data.ra,
        relative_astrom_data.dec,
        relative_astrom_data.ra_err,
        relative_astrom_data.dec_err,
        relative_astrom_data.corr,
    )

    # TI constants in AU
    A_AU = A_s * dist_pc
    F_AU = F_s * dist_pc
    B_AU = B_s * dist_pc
    G_AU = G_s * dist_pc

    return TIFitResult(
        A=A_AU,
        B=B_AU,
        F=F_AU,
        G_ti=G_AU,
        a=a_AU,
        e=e,
        cos_i=cos_i,
        W=W,
        cos_w=cos_w,
        sin_w=sin_w,
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
