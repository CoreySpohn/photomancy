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
from hwoutils.constants import two_pi
from hwoutils.conversions import arcsec_to_au
from orbix.equations import period_to_sma
from orbix.system import (
    EccentricityVectorChart,
    FrameOrbit,
    ThieleInnesChart,
    disk_to_plane,
    line_of_sight_branch,
    periapsis_chart_shape,
)
from orbix.system.linear import fit_linear_map

_TI_CHART = ThieleInnesChart()

TI_CHART_PARAM_NAMES = ("ln_n", "ux", "uy", "A", "B", "F", "G")


class TIFitResult(eqx.Module):
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


def thiele_innes_fit(data, dist_pc, ln_n, ex, ey, epoch_d=0.0):
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
        A ``TIFitResult`` with both line-of-sight sheets mapped.
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
    return TIFitResult(
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


def thiele_innes_grid(data, dist_pc, log_T_grid, e_grid, n_tp=30, epoch_d=0.0):
    """``thiele_innes_fit`` over a (log T, e, tp) grid, on one flat grid axis.

    Periods are ``10**log_T_grid`` days and periapsis times ``tp_frac * T`` for
    ``n_tp`` fractions of the period in ``[0, 1)``.
    """
    log_T = jnp.asarray(log_T_grid)
    ecc = jnp.asarray(e_grid)
    tp_frac = jnp.linspace(0.0, 1.0, n_tp, endpoint=False)

    def fit_one(log_T_i, e_i, frac_i):
        T = 10.0**log_T_i
        shape = periapsis_chart_shape(T, e_i, frac_i * T, epoch_d)
        return thiele_innes_fit(data, dist_pc, *shape, epoch_d=epoch_d)

    log_T_flat = jnp.repeat(log_T, len(ecc) * n_tp)
    e_flat = jnp.tile(jnp.repeat(ecc, n_tp), len(log_T))
    frac_flat = jnp.tile(tp_frac, len(log_T) * len(ecc))
    return jax.vmap(fit_one)(log_T_flat, e_flat, frac_flat)


def thiele_innes_grid_search(data, dist_pc, log_T_grid, e_grid, n_tp=30, epoch_d=0.0):
    """Best-likelihood ``TIFitResult`` over the grid of ``thiele_innes_grid``."""
    results = thiele_innes_grid(data, dist_pc, log_T_grid, e_grid, n_tp, epoch_d)
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
        fit: A ``TIFitResult`` (possibly batched).
        star_mass_kg: Stellar mass (kg).
    """
    return _ECC_VECTOR_CHART.from_orbit(fixed_mass_orbits(fit, star_mass_kg))
