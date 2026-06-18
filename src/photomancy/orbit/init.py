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
    init_vals = find_init(astrom_data, Ms, dist_pc)
    kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
"""

import jax
import jax.numpy as jnp
from orbix.equations import period_to_sma

from photomancy.orbit.forward import predict_astrometry
from photomancy.orbit.likelihoods import loglike_astrom
from photomancy.orbit.thiele_innes import thiele_innes_fit


def ti_to_init(ti_result, Ms, n_planets=1):
    """Convert a :class:`TIFitResult` to a NumPyro ``init_to_value`` dict.

    Maps the TI-recovered orbital elements back to the raw NumPyro
    parameter names used by :func:`~photomancy.orbit.model.build_model`.

    Values are clamped to lie strictly within the support of each
    NumPyro distribution (e.g. ``e_raw`` away from 0 for Beta priors).
    Any NaN values (common with very sparse data) are replaced with
    prior-center defaults before clamping.

    Args:
        ti_result: A :class:`TIFitResult` from the TI fitter or grid search.
        Ms: Stellar mass (kg). Needed only for logging/validation.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        Dict mapping NumPyro sample site names to initial values, suitable
        for ``numpyro.infer.init_to_value(values=...)``.
    """
    eps = 1e-6  # small offset to keep values inside open supports

    # Period -> log10(T)
    log_P = jnp.log10(ti_result.T)
    log_P = jnp.where(jnp.isnan(log_P), 2.5, log_P)

    # Eccentricity -> e_raw.  Clamp to (eps, 1-eps) so it lies strictly
    # inside Beta(a,b) support (0,1).  e=0 from a circular-orbit grid
    # point would give -inf log-prob under Beta.
    e_raw = jnp.where(jnp.isnan(ti_result.e), 0.2, ti_result.e)
    e_raw = jnp.clip(e_raw, eps, 1.0 - eps)

    # Argument of periapsis -> w_raw in (eps, 2pi-eps)
    sin_w = jnp.where(jnp.isnan(ti_result.sin_w), 0.0, ti_result.sin_w)
    cos_w = jnp.where(jnp.isnan(ti_result.cos_w), 1.0, ti_result.cos_w)
    w = jnp.arctan2(sin_w, cos_w) % (2.0 * jnp.pi)
    w = jnp.clip(w, eps, 2.0 * jnp.pi - eps)

    # cos_i -- clamp to (-1+eps, 1-eps) for Uniform(-1,1)
    cos_i = jnp.where(jnp.isnan(ti_result.cos_i), 0.0, ti_result.cos_i)
    cos_i = jnp.clip(cos_i, -1.0 + eps, 1.0 - eps)

    # Longitude of ascending node -- clamp to (eps, 2pi-eps)
    W_val = jnp.where(jnp.isnan(ti_result.W), jnp.pi, ti_result.W)
    W = W_val % (2.0 * jnp.pi)
    W = jnp.clip(W, eps, 2.0 * jnp.pi - eps)

    # Convert tp -> M0: M0 = n * (0 - tp) = -2pi*tp / T, then mod 2pi
    tp_val = jnp.where(jnp.isnan(ti_result.tp), 0.0, ti_result.tp)
    T_safe = jnp.where(jnp.isnan(ti_result.T), 1.0, ti_result.T)
    M0 = (-tp_val * 2.0 * jnp.pi / T_safe) % (2.0 * jnp.pi)
    M0 = jnp.clip(M0, eps, 2.0 * jnp.pi - eps)

    # Pack into plate-shaped arrays (shape = (n_planets,))
    return {
        "log_P": jnp.full(n_planets, log_P),
        "e_raw": jnp.full(n_planets, e_raw),
        "w_raw": jnp.full(n_planets, w),
        "cos_i": jnp.full(n_planets, cos_i),
        "W": jnp.full(n_planets, W),
        "M0": jnp.full(n_planets, M0),
    }


def _grid_results(astrom_data, Ms, dist_pc, log_T_range, n_log_T, e_grid, n_tp):
    """Vmapped Thiele-Innes fit over the full ``(log_T, e, tp)`` grid."""
    log_T_grid = jnp.linspace(log_T_range[0], log_T_range[1], n_log_T)
    tp_fracs = jnp.linspace(0.0, 1.0, n_tp, endpoint=False)

    def _fit(log_T, e_val, tp_frac):
        T = 10.0**log_T
        return thiele_innes_fit(astrom_data, T, e_val, tp_frac * T, Ms, dist_pc)

    log_T_flat = jnp.repeat(log_T_grid, len(e_grid) * n_tp)
    e_flat = jnp.tile(jnp.repeat(jnp.asarray(e_grid), n_tp), n_log_T)
    tp_flat = jnp.tile(tp_fracs, n_log_T * len(e_grid))
    return jax.vmap(_fit)(log_T_flat, e_flat, tp_flat)


def _kepler_consistent_loglike(results, astrom_data, Ms, dist_pc):
    """Log-likelihood of each grid candidate with ``a`` constrained to Kepler III.

    The Thiele-Innes fit reports a freely-fit ``a`` (apparent ellipse). Here we
    re-evaluate each candidate's orbit with ``a = period_to_sma(T, Ms)`` -- the
    constraint the full model imposes -- so the ranking favors periods whose
    Kepler-implied orbit actually fits the data. NaN candidates map to -inf.
    """
    a_kepler = period_to_sma(results.T, Ms)

    def _ll(a, e, cos_i, W, cos_w, sin_w, tp):
        ra, dec = predict_astrometry(
            astrom_data.times, a, e, cos_i, W, cos_w, sin_w, tp, Ms, dist_pc
        )
        return loglike_astrom(ra, dec, astrom_data)

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
    astrom_data,
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
        astrom_data: An :class:`~photomancy.orbit.data.AstromData` instance.
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

        init_vals = find_init(astrom_data, Ms, dist_pc)
        model = build_model(Ms, dist_pc, astrom_data=astrom_data)
        kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
        mcmc = MCMC(kernel, num_warmup=500, num_samples=2000)
        mcmc.run(jax.random.PRNGKey(0))
    """
    if e_grid is None:
        e_grid = jnp.linspace(0.0, 0.8, 9)

    results = _grid_results(
        astrom_data, Ms, dist_pc, log_T_range, n_log_T, e_grid, n_tp
    )
    kep_ll = _kepler_consistent_loglike(results, astrom_data, Ms, dist_pc)
    best = jax.tree.map(lambda x: x[jnp.argmax(kep_ll)], results)
    return ti_to_init(best, Ms, n_planets=n_planets)


def find_init_top_k(
    astrom_data,
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
        astrom_data: An :class:`~photomancy.orbit.data.AstromData` instance.
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
        astrom_data, Ms, dist_pc, log_T_range, n_log_T, e_grid, n_tp
    )
    kep_ll = _kepler_consistent_loglike(results, astrom_data, Ms, dist_pc)

    top_k_indices = jnp.argsort(kep_ll)[-k:][::-1]
    init_dicts = []
    for idx in top_k_indices:
        single = jax.tree.map(lambda x, i=idx: x[i], results)
        init_dicts.append(ti_to_init(single, Ms, n_planets=n_planets))
    return init_dicts
