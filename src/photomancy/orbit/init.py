"""MAP initialization for MCMC -- converts TI results to NumPyro init dicts.

Provides utilities to seed NUTS/HMC warmup from a Thiele-Innes grid search,
avoiding random initialization in complex posteriors.

The grid search is **Kepler-consistent**: Thiele-Innes fits the semimajor axis
freely (relative astrometry, so the apparent ellipse fits at many periods), but
the full model ties ``a = period_to_sma(T, Ms)`` via Kepler's third law. Ranking
candidates by the a-free TI chi2 can therefore pick a period whose Kepler-implied
orbit does not fit the data. We instead rank by the residual of the
Kepler-constrained orbit, so the chosen period is physically consistent with the
(known) stellar mass.

Usage::

    from photomancy.orbit.init import find_init
    init_vals = find_init(relative_astrom_data, Ms, dist_pc)
    kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
"""

import jax
import jax.numpy as jnp
from orbix.equations import period_to_sma

from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.thiele_innes import thiele_innes_fit


def elements_to_sites(T, e, cos_i, W, cos_w, sin_w, tp, n_planets=1):
    """Physical orbital elements -> NumPyro raw sample sites (clamped into support).

    Maps period / eccentricity / angles to the raw site names used by
    :func:`~photomancy.orbit.model.build_model`, clamped strictly inside each
    distribution's support; NaN inputs (common with very sparse data) fall back to
    prior-center defaults. Shared by the TI initializer and the OFTI / grid_search
    physical -> z bridge.

    Args:
        T: Orbital period (days).
        e: Eccentricity.
        cos_i: Cosine of inclination.
        W: Longitude of ascending node (rad).
        cos_w: Cosine of the argument of periapsis.
        sin_w: Sine of the argument of periapsis.
        tp: Time of periapsis (days).
        n_planets: Number of planets (the plate size). Default 1.

    Returns:
        Dict of raw site arrays of shape ``(n_planets,)``.
    """
    eps = 1e-6  # small offset to keep values inside open supports

    log_P = jnp.where(jnp.isnan(jnp.log10(T)), 2.5, jnp.log10(T))

    # e_raw clamped strictly inside Beta support (0, 1)
    e_raw = jnp.clip(jnp.where(jnp.isnan(e), 0.2, e), eps, 1.0 - eps)

    w = jnp.arctan2(
        jnp.where(jnp.isnan(sin_w), 0.0, sin_w),
        jnp.where(jnp.isnan(cos_w), 1.0, cos_w),
    ) % (2.0 * jnp.pi)
    w = jnp.clip(w, eps, 2.0 * jnp.pi - eps)

    cos_i = jnp.clip(jnp.where(jnp.isnan(cos_i), 0.0, cos_i), -1.0 + eps, 1.0 - eps)

    W = jnp.where(jnp.isnan(W), jnp.pi, W) % (2.0 * jnp.pi)
    W = jnp.clip(W, eps, 2.0 * jnp.pi - eps)

    # tp -> M0 = -2pi*tp / T, then mod 2pi
    T_safe = jnp.where(jnp.isnan(T), 1.0, T)
    M0 = (-jnp.where(jnp.isnan(tp), 0.0, tp) * 2.0 * jnp.pi / T_safe) % (2.0 * jnp.pi)
    M0 = jnp.clip(M0, eps, 2.0 * jnp.pi - eps)

    return {
        "log_P": jnp.full(n_planets, log_P),
        "e_raw": jnp.full(n_planets, e_raw),
        "w_raw": jnp.full(n_planets, w),
        "cos_i": jnp.full(n_planets, cos_i),
        "W": jnp.full(n_planets, W),
        "M0": jnp.full(n_planets, M0),
    }


def ti_to_init(ti_result, Ms, n_planets=1):
    """Convert a :class:`TIFitResult` to a NumPyro ``init_to_value`` dict.

    Thin wrapper over :func:`elements_to_sites` for the TI-recovered elements.

    Args:
        ti_result: A :class:`TIFitResult` from the TI fitter or grid search.
        Ms: Stellar mass (kg). Unused; kept for call-site compatibility.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        Dict mapping NumPyro sample site names to initial values, suitable
        for ``numpyro.infer.init_to_value(values=...)``.
    """
    return elements_to_sites(
        ti_result.T,
        ti_result.e,
        ti_result.cos_i,
        ti_result.W,
        ti_result.cos_w,
        ti_result.sin_w,
        ti_result.tp,
        n_planets,
    )


def _grid_results(
    relative_astrom_data, Ms, dist_pc, log_T_range, n_log_T, e_grid, n_tp
):
    """Vmapped Thiele-Innes fit over the full ``(log_T, e, tp)`` grid."""
    log_T_grid = jnp.linspace(log_T_range[0], log_T_range[1], n_log_T)
    tp_fracs = jnp.linspace(0.0, 1.0, n_tp, endpoint=False)

    def _fit(log_T, e_val, tp_frac):
        T = 10.0**log_T
        return thiele_innes_fit(
            relative_astrom_data, T, e_val, tp_frac * T, Ms, dist_pc
        )

    log_T_flat = jnp.repeat(log_T_grid, len(e_grid) * n_tp)
    e_flat = jnp.tile(jnp.repeat(jnp.asarray(e_grid), n_tp), n_log_T)
    tp_flat = jnp.tile(tp_fracs, n_log_T * len(e_grid))
    return jax.vmap(_fit)(log_T_flat, e_flat, tp_flat)


def _kepler_consistent_loglike(results, relative_astrom_data, Ms, dist_pc):
    """Log-likelihood of each grid candidate with ``a`` constrained to Kepler III.

    The Thiele-Innes fit reports a freely-fit ``a`` (apparent ellipse). Here we
    re-evaluate each candidate's orbit with ``a = period_to_sma(T, Ms)`` -- the
    constraint the full model imposes -- so the ranking favors periods whose
    Kepler-implied orbit actually fits the data. NaN candidates map to -inf.
    """
    a_kepler = period_to_sma(results.T, Ms)

    def _ll(a, e, cos_i, W, cos_w, sin_w, tp):
        ra, dec = predict_relative_astrometry(
            relative_astrom_data.times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc
        )
        return loglike_relative_astrom(ra, dec, relative_astrom_data)

    ll = jax.vmap(_ll)(
        a_kepler,
        results.e,
        results.cos_i,
        results.W,
        results.cos_w,
        results.sin_w,
        results.tp,
    )
    return jnp.where(jnp.isnan(ll), -jnp.inf, ll)


def find_init(
    relative_astrom_data,
    Ms,
    dist_pc,
    log_T_range=(1.0, 4.0),
    n_log_T=100,
    e_grid=None,
    n_tp=30,
    n_planets=1,
):
    """Find good MCMC initialization via a Kepler-consistent TI grid search.

    Performs a 3D grid search over ``(T, e, tp)`` using the linear Thiele-Innes
    fitter, ranks candidates by the **Kepler-constrained** residual (``a`` tied
    to the period via Kepler III), and converts the best into a NumPyro
    ``init_to_value`` dict. Ranking by the Kepler-constrained residual (rather
    than the a-free TI chi2) avoids periods that fit the apparent motion but
    imply an orbit size inconsistent with the stellar mass.

    Args:
        relative_astrom_data: An
            :class:`~photomancy.orbit.data.RelativeAstromData` instance.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        log_T_range: (min, max) log10(T/days) for the period grid.
            Default ``(1.0, 4.0)`` covers 10-10,000 days.
        n_log_T: Number of period grid points. Default 100.
        e_grid: Array of eccentricity values to search. Default is
            ``[0.0, 0.1, ..., 0.8]``.
        n_tp: Number of tp grid points per period. Default 30.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        A dict mapping NumPyro sample site names to initial values,
        suitable for ``numpyro.infer.init_to_value(values=...)``.

    Example::

        from numpyro.infer import MCMC, NUTS, init_to_value
        from photomancy.orbit.init import find_init
        from photomancy.orbit.model import build_model

        init_vals = find_init(relative_astrom_data, Ms, dist_pc)
        model = build_model(Ms, dist_pc, relative_astrom_data=relative_astrom_data)
        kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
        mcmc = MCMC(kernel, num_warmup=500, num_samples=2000)
        mcmc.run(jax.random.PRNGKey(0))
    """
    if e_grid is None:
        e_grid = jnp.linspace(0.0, 0.8, 9)

    results = _grid_results(
        relative_astrom_data, Ms, dist_pc, log_T_range, n_log_T, e_grid, n_tp
    )
    kep_ll = _kepler_consistent_loglike(results, relative_astrom_data, Ms, dist_pc)
    best = jax.tree.map(lambda x: x[jnp.argmax(kep_ll)], results)
    return ti_to_init(best, Ms, n_planets=n_planets)


def find_init_top_k(
    relative_astrom_data,
    Ms,
    dist_pc,
    k=5,
    log_T_range=(1.0, 4.0),
    n_log_T=100,
    e_grid=None,
    n_tp=30,
    n_planets=1,
):
    """Find the top-k Kepler-consistent MCMC initializations via TI grid search.

    Returns multiple initialization dicts for multi-chain MCMC / multi-start
    Laplace, seeding each from a different high-likelihood, Kepler-consistent
    candidate.

    Args:
        relative_astrom_data: An
            :class:`~photomancy.orbit.data.RelativeAstromData` instance.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        k: Number of top initializations to return. Default 5.
        log_T_range: (min, max) log10(T/days) for the period grid.
        n_log_T: Number of period grid points. Default 100.
        e_grid: Array of eccentricity values to search. Default is
            ``[0.0, 0.1, ..., 0.8]``.
        n_tp: Number of tp grid points per period. Default 30.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        A list of ``k`` dicts, each mapping NumPyro sample site names
        to initial values.
    """
    if e_grid is None:
        e_grid = jnp.linspace(0.0, 0.8, 9)

    results = _grid_results(
        relative_astrom_data, Ms, dist_pc, log_T_range, n_log_T, e_grid, n_tp
    )
    kep_ll = _kepler_consistent_loglike(results, relative_astrom_data, Ms, dist_pc)

    top_k_indices = jnp.argsort(kep_ll)[-k:][::-1]
    init_dicts = []
    for idx in top_k_indices:
        single = jax.tree.map(lambda x, i=idx: x[i], results)
        init_dicts.append(ti_to_init(single, Ms, n_planets=n_planets))
    return init_dicts
