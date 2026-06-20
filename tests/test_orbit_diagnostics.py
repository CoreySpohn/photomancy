"""Tests for orbit posterior diagnostics (physical samples, mode summaries)."""

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402

from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.diagnostics import mode_summary, sample_physical  # noqa: E402
from photomancy.orbit.forward import predict_relative_astrometry  # noqa: E402
from photomancy.orbit.inference import build_orbit_logdensity  # noqa: E402
from photomancy.orbit.laplace import map_laplace_mixture_fit  # noqa: E402

MSUN_KG = 1.989e30
DIST_PC = 10.0
TRUE_T = 1096.0
LOG_P_RANGE = (float(np.log10(600.0)), float(np.log10(1600.0)))


def _make_astrom(n_obs=5, seed=42):
    a = float(period_to_sma(TRUE_T, MSUN_KG))
    tp = -1.5 / (2.0 * jnp.pi / TRUE_T)
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 3.0 * TRUE_T, n_obs))
    err = 5.0e-3
    ra, dec = predict_relative_astrometry(
        jnp.asarray(times),
        a,
        0.15,
        0.5,
        2.3,
        jnp.cos(0.8),
        jnp.sin(0.8),
        tp,
        MSUN_KG,
        DIST_PC,
    )
    return RelativeAstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(np.asarray(ra) + rng.normal(0.0, err, n_obs)),
        dec=jnp.asarray(np.asarray(dec) + rng.normal(0.0, err, n_obs)),
        ra_err=jnp.full(n_obs, err),
        dec_err=jnp.full(n_obs, err),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


def _fit_posterior_and_problem():
    astrom = _make_astrom(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = map_laplace_mixture_fit(
            MSUN_KG, DIST_PC, relative_astrom_data=astrom, log_P_range=LOG_P_RANGE, k=3
        )
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, relative_astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    return post, problem


def test_sample_physical_shapes_and_keys():
    """sample_physical draws physical orbital parameters of shape (n,)."""
    post, problem = _fit_posterior_and_problem()
    phys = sample_physical(post, problem, jax.random.key(0), 50)
    assert "T" in phys and "e" in phys
    assert phys["T"].shape == (50,)
    assert jnp.all(jnp.isfinite(phys["T"]))


def test_mode_summary_structure():
    """mode_summary returns one record per mode with weight/log_evidence/params."""
    post, problem = _fit_posterior_and_problem()
    summary = mode_summary(post, problem)
    assert len(summary) == post.n_modes
    assert set(summary[0]) == {"weight", "log_evidence", "params"}
    assert "T" in summary[0]["params"]
