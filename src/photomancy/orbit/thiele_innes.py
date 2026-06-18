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

These are the (x, y) components of the orbit's A and B position matrices
from :func:`orbix.equations.orbit.AB_matrices_reduced`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from orbix.equations.orbit import mean_anomaly_tp
from orbix.kepler.core import diff_solve_trig

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

    The astrometric model (in arcsec) is:

        RA(t)  = (A*X(t) + F*Y(t)) / dist_pc
        DEC(t) = (B*X(t) + G*Y(t)) / dist_pc

    After absorbing dist_pc into (A, B, F, G), we can write this as two
    independent 2-parameter weighted linear regressions:

        RA(t)  = A'*X(t) + F'*Y(t)
        DEC(t) = B'*X(t) + G'*Y(t)

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
    w_xx = 1.0 / (ra_err**2 * one_minus_rho2)  # RA self-weight
    w_yy = 1.0 / (dec_err**2 * one_minus_rho2)  # DEC self-weight
    w_xy = -corr / (ra_err * dec_err * one_minus_rho2)  # Cross-weight

    # Normal equation: D^T W D (4x4)
    # The (i,j) element is sum_k D_ki * W_kk' * D_k'j
    # With the block structure, we can compute this efficiently.

    # RA-RA block (weighted by w_xx)
    XwX = jnp.sum(w_xx * X * X)
    XwY = jnp.sum(w_xx * X * Y)
    YwY = jnp.sum(w_xx * Y * Y)

    # DEC-DEC block (weighted by w_yy)
    XvX = jnp.sum(w_yy * X * X)
    XvY = jnp.sum(w_yy * X * Y)
    YvY = jnp.sum(w_yy * Y * Y)

    # RA-DEC cross block (weighted by w_xy)
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
            jnp.sum(w_xx * X * ra) + jnp.sum(w_xy * X * dec),
            jnp.sum(w_xx * Y * ra) + jnp.sum(w_xy * Y * dec),
            jnp.sum(w_xy * X * ra) + jnp.sum(w_yy * X * dec),
            jnp.sum(w_xy * Y * ra) + jnp.sum(w_yy * Y * dec),
        ]
    )

    # Tikhonov regularization for rank-deficient systems (N=1,2 obs).
    # For well-determined systems the penalty is negligible.
    lambda_reg = 1e-4
    normal_mat_reg = normal_mat + lambda_reg * jnp.eye(4)

    # Solve the regularized 4x4 system
    params = jnp.linalg.solve(normal_mat_reg, rhs)

    return params[0], params[1], params[2], params[3]


def _abfg_to_elements(A_scaled, F_scaled, B_scaled, G_scaled, e):
    """Convert scaled TI constants to orbital elements.

    The TI constants relate to orbital elements as:
        A = a * (cosW*cosw - sinW*sinw*cosi)
        B = a * (sinW*cosw + cosW*sinw*cosi)
        F = a*sqrt(1-e^2) * (-cosW*sinw - sinW*cosw*cosi)
        G = a*sqrt(1-e^2) * (-sinW*sinw + cosW*cosw*cosi)

    From these we can recover (a, i, W, w) using the standard
    identities:
        A^2 + B^2 = a^2 * (cos^2w + sin^2w*cos^2i) = a^2 * alpha
        F^2 + G^2 = a^2(1-e^2) * (sin^2w + cos^2w*cos^2i) = a^2(1-e^2) * beta
        AF + BG = a^2sqrt(1-e^2) * (-cosw*sinw(1-cos^2i)) = a^2sqrt(1-e^2) * gamma
        AG - BF = a^2sqrt(1-e^2) * cosi

    The last identity gives cos(i) directly (up to sign of a^2sqrt(1-e^2)).

    Args:
        A_scaled: TI constant A / dist_pc (arcsec-scale). Scalar.
        F_scaled: TI constant F / dist_pc (arcsec-scale). Scalar.
        B_scaled: TI constant B / dist_pc (arcsec-scale). Scalar.
        G_scaled: TI constant G / dist_pc (arcsec-scale). Scalar.
        e: Eccentricity. Scalar.

    Returns:
        Tuple of (a_scaled, cos_i, W, cos_w, sin_w).
        a_scaled is in arcsec-scale (multiply by dist_pc for AU).
    """
    sqrt_1me2 = jnp.sqrt(1.0 - e**2)

    # Semi-major axis from the two quadratic invariants:
    # A^2 + B^2 + F^2 + G^2 = a^2(1 + cos^2i) + a^2(1-e^2)(1 + cos^2i) ... nope
    # Better approach: use the two independent sums.
    #
    # p^2 = A^2 + B^2 = a^2(cos^2w + sin^2w*cos^2i)
    # q^2 = F^2 + G^2 = a^2(1-e^2)(sin^2w + cos^2w*cos^2i)
    # And: AG - BF = a^2sqrt(1-e^2)*cosi

    p2 = A_scaled**2 + B_scaled**2
    q2 = F_scaled**2 + G_scaled**2

    # a^2 from: p^2 + q^2/(1-e^2) = a^2(1 + cos^2i)
    # and: (AG-BF)^2 = a⁴(1-e^2)*cos^2i
    # So a⁴(1-e^2)*cos^2i = (AG-BF)^2
    # And a^2(1+cos^2i) = p^2 + q^2/(1-e^2)
    #
    # Alternative direct formula:
    # a^2(cos^2w + sin^2w*cos^2i) = p^2
    # a^2(1-e^2)(sin^2w + cos^2w*cos^2i) = q^2
    # Sum: a^2[cos^2w + sin^2w*cos^2i + (1-e^2)(sin^2w + cos^2w*cos^2i)]
    #    = a^2[(cos^2w + (1-e^2)sin^2w) + cos^2i(sin^2w + (1-e^2)cos^2w)]
    #    = a^2[1 - e^2sin^2w + cos^2i(1 - e^2cos^2w)]
    # This is messy. The standard approach uses:
    #
    # d = AG - BF = a^2sqrt(1-e^2)*cosi
    # s = A^2 + B^2 + (F^2 + G^2)/(1-e^2)
    #   = a^2(cos^2w + sin^2w*cos^2i + sin^2w + cos^2w*cos^2i)
    #   = a^2(1 + cos^2i)
    #
    # Then a⁴(1-e^2)cos^2i = d^2
    # and  a^2(1+cos^2i) = s
    #
    # From these: let u = a^2*cos^2i  (unknown)
    # d^2 = (1-e^2)*u*a^2  ->  a^2 = d^2/((1-e^2)*u)
    # s = a^2 + u + a^2 - a^2  ... hmm, s = a^2(1+cos^2i) = a^2 + u
    # So a^2 = s - u
    # And d^2 = (1-e^2)*u*(s-u)
    # This is a quadratic in u:
    #   (1-e^2)*u^2 - (1-e^2)*s*u + d^2 = 0
    #   u = [(1-e^2)*s +/- sqrt((1-e^2)^2s^2 - 4(1-e^2)d^2)] / (2(1-e^2))
    #   u = [s +/- sqrt(s^2 - 4d^2/(1-e^2))] / 2

    d = A_scaled * G_scaled - B_scaled * F_scaled  # = a^2sqrt(1-e^2)*cosi / dist_pc^2
    s = p2 + q2 / jnp.maximum(1.0 - e**2, 1e-30)  # = a^2(1+cos^2i) / dist_pc^2

    # Quadratic for u = a^2*cos^2i / dist_pc^2
    discriminant = jnp.maximum(s**2 - 4.0 * d**2 / jnp.maximum(1.0 - e**2, 1e-30), 0.0)
    sqrt_disc = jnp.sqrt(discriminant)

    # Two solutions: u = (s +/- sqrt_disc) / 2
    # Since cos^2i <= 1, we need u <= a^2 = s - u, i.e., u <= s/2.
    # The minus root gives the smaller u -> cos^2i <= 1.
    u = (s - sqrt_disc) / 2.0

    # a^2 = s - u
    a2 = jnp.maximum(s - u, 1e-30)
    a_scaled = jnp.sqrt(a2)

    # cos_i from d = a^2sqrt(1-e^2)*cosi -> cosi = d / (a^2sqrt(1-e^2))
    cos_i = d / jnp.maximum(a2 * sqrt_1me2, 1e-30)
    cos_i = jnp.clip(cos_i, -1.0, 1.0)

    # Recover W and w from the TI constants.
    #
    # The four key identities (derived by expanding the TI definitions):
    #   A + G = (1+cosi) * cos(W+w)
    #   B - F = (1+cosi) * sin(W+w)
    #   A - G = (1-cosi) * cos(W-w)
    #   B + F = (1-cosi) * sin(W-w)
    #
    # These hold for the *unit* TI constants (a=1, e=0).  When e!=0,
    # F and G carry an extra sqrt(1-e^2) factor that must be divided out
    # before forming the sums.

    # Scale F, G by 1/sqrt(1-e^2) to put them on the same footing as A, B
    F_norm = F_scaled / jnp.maximum(sqrt_1me2, 1e-30)
    G_norm = G_scaled / jnp.maximum(sqrt_1me2, 1e-30)

    # W + w  from atan2(B-F_norm, A+G_norm)
    # W - w  from atan2(B+F_norm, A-G_norm)
    W_plus_w = jnp.arctan2(B_scaled - F_norm, A_scaled + G_norm)
    W_minus_w = jnp.arctan2(B_scaled + F_norm, A_scaled - G_norm)

    W = (W_plus_w + W_minus_w) / 2.0
    w = (W_plus_w - W_minus_w) / 2.0

    # Ensure W in [0, 2pi)
    W = W % (2.0 * jnp.pi)

    cos_w = jnp.cos(w)
    sin_w = jnp.sin(w)

    return a_scaled, cos_i, W, cos_w, sin_w


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
    ra_pred = A_s * X + F_s * Y
    dec_pred = B_s * X + G_s * Y

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


def thiele_innes_fit(astrom_data, T, e, tp, Ms, dist_pc):
    """Fit Thiele-Innes constants via linear least squares.

    Given trial ``(T, e, tp)`` and astrometry data, finds the optimal
    ``(A, B, F, G)`` orientation constants. This is exact and O(N) since
    the model is linear in these constants for fixed ``(T, e, tp)``.

    Args:
        astrom_data: An :class:`~photomancy.orbit.data.AstromData` instance.
        T: Trial orbital period (days). Scalar.
        e: Trial eccentricity. Scalar.
        tp: Trial time of periapsis passage (days). Scalar.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.

    Returns:
        A :class:`TIFitResult` with the best-fit TI constants, recovered
        orbital elements, and goodness-of-fit statistics.
    """
    # Compute basis functions X(t) = cosE(t) - e, Y(t) = sinE(t)
    # n = 2pi/T directly -- no need to go through Kepler III
    n = 2.0 * jnp.pi / T
    M = mean_anomaly_tp(astrom_data.times, n, tp)
    sinE, cosE = diff_solve_trig(M, e)

    X = cosE - e  # shape (N,)
    Y = sinE  # shape (N,)

    # Solve for (A', F', B', G') = (A/dist_pc, F/dist_pc, B/dist_pc, G/dist_pc)
    A_s, F_s, B_s, G_s = _solve_abfg(
        X,
        Y,
        astrom_data.ra,
        astrom_data.dec,
        astrom_data.ra_err,
        astrom_data.dec_err,
        astrom_data.corr,
    )

    # Recover orbital elements from scaled constants
    a_scaled, cos_i, W, cos_w, sin_w = _abfg_to_elements(A_s, F_s, B_s, G_s, e)

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
        astrom_data.ra,
        astrom_data.dec,
        astrom_data.ra_err,
        astrom_data.dec_err,
        astrom_data.corr,
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


def thiele_innes_grid_search(astrom_data, Ms, dist_pc, log_T_grid, e_grid, n_tp=30):
    """Search over (T, e, tp) grid with Thiele-Innes linearization.

    For each ``(T, e)`` pair, evaluates ``n_tp`` uniformly spaced
    tp values in ``[0, T]`` and returns the result with the best
    log-likelihood.

    The grid search is fully vectorized via ``jax.vmap`` for GPU
    acceleration.

    Args:
        astrom_data: An :class:`~photomancy.orbit.data.AstromData` instance.
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
        return thiele_innes_fit(astrom_data, T, e_val, tp, Ms, dist_pc)

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
