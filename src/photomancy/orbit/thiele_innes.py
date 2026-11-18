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
from orbix.system import (
    EccentricityVectorChart,
    FrameOrbit,
    ThieleInnesChart,
    disk_to_plane,
    line_of_sight_branch,
    periapsis_chart_shape,
)
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


# ---------------------------------------------------------------------------
# Chart route: epoch mean-longitude axes, free implied mass or fixed mass
# ---------------------------------------------------------------------------

_TI_CHART = ThieleInnesChart()

TI_CHART_PARAM_NAMES = ("ln_n", "ux", "uy", "A", "B", "F", "G")


class TIChartFit(eqx.Module):
    """Result of a conditional Thiele-Innes fit in ``ThieleInnesChart`` coordinates.

    The four constants lie on the stored epoch mean-longitude axes, not the
    periapsis axes: ``A, F`` are north and ``B, G`` east, so the sky block is
    ``[[B, G], [A, F]]``. Rotating it by the periapsis angle gives the classical
    constants (see ``FrameOrbit.linear_map``). The line-of-sight reflection is not
    constrained by relative astrometry, so both sheets are returned.

    Attributes:
        x: Chart coordinates ``(ln n, ux, uy, A, B, F, G)``, shape ``(7,)``.
        orbits: Orbits of the two sheets, batch shape ``(2,)``, epoch ``epoch_d``.
        branches: Line-of-sight sheet of each orbit, ``[1, -1]``.
        implied_mu_AU3_per_d2: ``n**2 a**3`` of the fitted orbit.
        log_volume: Chart log density against the reference measure extended by
            ``d ln mu`` (equal on both sheets).
        sky_block: Fitted ``[[B, G], [A, F]]`` in AU, shape ``(2, 2)``.
        covariance: Information-pseudoinverse covariance of ``(A, B, F, G)``, AU^2.
        rank: Numerical rank of the linear fit (4 when identified).
        chi2: Sum of squared whitened residuals.
        log_likelihood: Gaussian log-likelihood in the observed angular units.
    """

    x: jnp.ndarray
    orbits: FrameOrbit
    branches: jnp.ndarray
    implied_mu_AU3_per_d2: jnp.ndarray
    log_volume: jnp.ndarray
    sky_block: jnp.ndarray
    covariance: jnp.ndarray
    rank: jnp.ndarray
    chi2: jnp.ndarray
    log_likelihood: jnp.ndarray


def chart_thiele_innes_fit(data, dist_pc, ln_n, ex, ey, epoch_d=0.0):
    """Fit the Thiele-Innes constants at fixed shape, in chart coordinates.

    The shape is the mean motion and the eccentricity vector ``(ex, ey)`` at
    ``epoch_d`` (see ``periapsis_chart_shape``). The four constants are the
    weighted least-squares solution of orbix's masked QR/SVD solve, on the epoch
    mean-longitude axes; rank-deficient data give the minimum-norm solution. No
    penalty is applied.

    Args:
        data: ``RelativeAstromData`` (arcsec); invalid rows get zero weight.
        dist_pc: Distance to the system (parsec).
        ln_n: Natural log of the mean motion (radians per day).
        ex: Eccentricity-vector component along the epoch longitude axis f.
        ey: Component along g.
        epoch_d: Epoch (days) of the shape.

    Returns:
        A ``TIChartFit`` with both line-of-sight sheets mapped.
    """
    mask = data.is_valid
    shape = FrameOrbit(
        jnp.ones(()), jnp.exp(ln_n), ex, ey, jnp.array([1.0, 0.0, 0.0, 0.0]), epoch_d
    )
    block, covariance, chi2, rank = fit_linear_map(
        shape,
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
    u = disk_to_plane(jnp.stack([ex, ey]))
    x = jnp.concatenate([jnp.stack([ln_n]), u, jnp.stack([A, B, F, G_ti])])
    sheets = [
        ThieleInnesChart(branch=branch).to_orbit(x, mu_AU3_per_d2=1.0, epoch_d=epoch_d)
        for branch in (1, -1)
    ]
    orbits = jax.tree.map(lambda plus, minus: jnp.stack([plus, minus]), *sheets)
    sx = jnp.where(mask, data.ra_err, 1.0)
    sy = jnp.where(mask, data.dec_err, 1.0)
    corr = jnp.where(mask, data.corr, 0.0)
    log_norm = jnp.log(two_pi) + jnp.log(sx) + jnp.log(sy) + 0.5 * jnp.log1p(-(corr**2))
    return TIChartFit(
        x=x,
        orbits=orbits,
        branches=line_of_sight_branch(orbits),
        implied_mu_AU3_per_d2=_TI_CHART.implied_mu(x),
        log_volume=_TI_CHART.log_volume(x),
        sky_block=block,
        covariance=covariance,
        rank=rank,
        chi2=chi2,
        log_likelihood=-0.5 * chi2 - jnp.sum(jnp.where(mask, log_norm, 0.0)),
    )


def chart_thiele_innes_grid(data, dist_pc, log_T_grid, e_grid, n_tp=30, epoch_d=0.0):
    """``chart_thiele_innes_fit`` over a (log T, e, tp) grid, on one flat grid axis.

    Periods are ``10**log_T_grid`` days and periapsis times ``tp_frac * T`` for
    ``n_tp`` fractions of the period in ``[0, 1)``, as in ``thiele_innes_grid_search``.
    """
    log_T = jnp.asarray(log_T_grid)
    ecc = jnp.asarray(e_grid)
    tp_frac = jnp.linspace(0.0, 1.0, n_tp, endpoint=False)

    def fit_one(log_T_i, e_i, frac_i):
        T = 10.0**log_T_i
        shape = periapsis_chart_shape(T, e_i, frac_i * T, epoch_d)
        return chart_thiele_innes_fit(data, dist_pc, *shape, epoch_d=epoch_d)

    log_T_flat = jnp.repeat(log_T, len(ecc) * n_tp)
    e_flat = jnp.tile(jnp.repeat(ecc, n_tp), len(log_T))
    frac_flat = jnp.tile(tp_frac, len(log_T) * len(ecc))
    return jax.vmap(fit_one)(log_T_flat, e_flat, frac_flat)


def chart_thiele_innes_grid_search(
    data, dist_pc, log_T_grid, e_grid, n_tp=30, epoch_d=0.0
):
    """Best-likelihood ``TIChartFit`` over the grid of ``chart_thiele_innes_grid``."""
    results = chart_thiele_innes_grid(data, dist_pc, log_T_grid, e_grid, n_tp, epoch_d)
    best = jnp.argmax(results.log_likelihood)
    return jax.tree.map(lambda leaf: leaf[best], results)


_ECC_VECTOR_CHART = EccentricityVectorChart()


def fixed_mass_orbits(fit, star_mass_kg):
    """Orbits of a fit's two sheets resized to Kepler's third law at a fixed mass.

    Each orbit keeps its period, shape, orientation and epoch phase and takes the
    semimajor axis ``star_mass_kg`` implies for that period. The leading batch
    axes of ``fit`` are kept, with the sheet axis last.
    """
    orbits = fit.orbits
    return eqx.tree_at(
        lambda o: o.a_AU, orbits, period_to_sma(orbits.period_d, star_mass_kg)
    )


def fixed_mass_coordinates(fit, star_mass_kg):
    """Six-coordinate fixed-mass chart points of a fit's two line-of-sight sheets.

    The orbits of ``fixed_mass_orbits`` in ``EccentricityVectorChart`` coordinates
    ``(ln a, ux, uy, atanh(cos i), W, M0 + w)``, shape ``(..., 2, 6)`` with the
    sheet order of ``fit.branches``. The sky positions of the two sheets coincide;
    angles are in ``[-pi, pi)``.

    Args:
        fit: A ``TIChartFit`` (possibly batched).
        star_mass_kg: Stellar mass (kg).
    """
    return _ECC_VECTOR_CHART.from_orbit(fixed_mass_orbits(fit, star_mass_kg))
