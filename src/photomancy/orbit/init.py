"""Kepler-consistent initial positions for orbit fits, in a declared chart.

Relative astrometry fits the semimajor axis freely (the apparent ellipse fits at many
periods), but a fixed-mass orbit ties ``a = period_to_sma(T, Ms)`` by Kepler's third
law. Ranking candidates by the free-``a`` chi2 can pick a period whose Kepler-implied
orbit does not fit, so candidates are ranked by the log-likelihood of the
fixed-mass orbit.

Usage::

    from photomancy.orbit.init import find_init
    starts, loglike = find_init(relative_astrom_data, Ms_kg, dist_pc, k=4)
"""

import jax
import jax.numpy as jnp
from orbix.system import EccentricityVectorChart

from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.thiele_innes import fixed_mass_orbits, thiele_innes_grid


def find_init(
    relative_astrom_data,
    star_mass_kg,
    dist_pc,
    k=1,
    log_T_range=(1.0, 4.0),
    n_log_T=100,
    e_grid=None,
    n_tp=30,
    branch=1,
    chart=None,
    epoch_d=0.0,
):
    """Top-k Kepler-consistent starts in the coordinates of a declared chart.

    Runs the chart Thiele-Innes grid, resizes every candidate to Kepler's third law
    at ``star_mass_kg`` (see ``fixed_mass_orbits``) and ranks them by the
    relative-astrometry log-likelihood of that fixed-mass orbit. The starts are the
    resized orbits written in ``chart``, the same chart object a
    ``ChartOrbitProblem`` is built with (default ``EccentricityVectorChart()``:
    ``(ln a, ux, uy, atanh(cos i), W, M0 + w)``, regular at ``e = 0`` and
    unclamped). The two line-of-sight sheets have identical sky predictions;
    ``branch`` picks the reported one. Candidates with a non-finite score rank last.

    The data times and the chart's mean-longitude epoch are days from one
    :class:`~hwoutils.epoch.Epoch`; ``epoch_d`` is the day, from that epoch, at
    which the chart's angles are evaluated. A ``ChartOrbitProblem`` builds its
    orbits at day zero, so keep the default for its positions.

    Args:
        relative_astrom_data: ``RelativeAstromData`` (arcsec).
        star_mass_kg: Stellar mass (kg).
        dist_pc: Distance (parsec).
        k: Number of starts to return.
        log_T_range: ``(min, max)`` of log10(T/days) for the period grid.
        n_log_T: Number of period grid points.
        e_grid: Eccentricities to search; default ``[0, 0.1, ..., 0.8]``.
        n_tp: Periapsis-time grid points per period.
        branch: Line-of-sight sheet, +1 or -1.
        chart: An orbix chart instance to express the starts in; ``None`` selects
            ``EccentricityVectorChart()``.
        epoch_d: Chart epoch (days from the data's epoch). Default 0.

    Returns:
        ``(coordinates, log_likelihood)`` of shapes ``(k, chart.n_coords)`` and
        ``(k,)``, best first.
    """
    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    if e_grid is None:
        e_grid = jnp.linspace(0.0, 0.8, 9)
    if chart is None:
        chart = EccentricityVectorChart()
    data = relative_astrom_data
    log_T_grid = jnp.linspace(log_T_range[0], log_T_range[1], n_log_T)
    fits = thiele_innes_grid(data, dist_pc, log_T_grid, e_grid, n_tp, epoch_d)
    orbits = fixed_mass_orbits(fits, star_mass_kg)
    sheet = 0 if branch == 1 else 1

    def candidate_loglike(orbit):
        ra, dec = orbit.offsets_arcsec(data.times, dist_pc)
        return loglike_relative_astrom(ra, dec, data)

    ll = jax.vmap(candidate_loglike)(jax.tree.map(lambda leaf: leaf[:, 0], orbits))
    ll = jnp.where(jnp.isnan(ll), -jnp.inf, ll)
    top = jnp.argsort(ll)[-k:][::-1]
    chosen = jax.tree.map(lambda leaf: leaf[top, sheet], orbits)
    return chart.from_orbit(chosen), ll[top]
