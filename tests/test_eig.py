"""Tests for the analytic Fisher EIG (Bayesian experimental design)."""

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402

from photomancy.orbit.data import AstromData  # noqa: E402
from photomancy.orbit.eig import (  # noqa: E402
    alias_breaking_eig,
    detectability_eig,
    evaluate_candidates,
    geometric_eig,
)
from photomancy.orbit.forward import predict_astrometry  # noqa: E402
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
    ra, dec = predict_astrometry(
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
    return AstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(np.asarray(ra) + rng.normal(0.0, err, n_obs)),
        dec=jnp.asarray(np.asarray(dec) + rng.normal(0.0, err, n_obs)),
        ra_err=jnp.full(n_obs, err),
        dec_err=jnp.full(n_obs, err),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


def test_geometric_eig_matches_analytic_gaussian_update():
    """The log-det shrinkage equals the 1-D conjugate-Gaussian information gain."""
    eig, cov_new = geometric_eig(jnp.array([[4.0]]), jnp.array([[1.0]]), 1.0)
    # prec_new = 1/4 + 1 -> cov_new = 0.8; eig = 0.5 * log(4 / 0.8) = 0.5 * log 5
    assert jnp.allclose(eig, 0.5 * jnp.log(5.0))
    assert float(cov_new[0, 0]) < 4.0

    eig_precise, _ = geometric_eig(jnp.array([[4.0]]), jnp.array([[1.0]]), 0.1)
    assert eig_precise > eig  # a more precise measurement is more informative


def test_alias_breaking_eig_zero_when_modes_agree():
    """Zero when modes predict the same observable; positive when they differ."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(
        alias_breaking_eig(weights, jnp.array([[1.0], [1.0]]), 1.0), 0.0
    )
    assert alias_breaking_eig(weights, jnp.array([[1.0], [5.0]]), 1.0) > 0.0


def test_detectability_eig_zero_when_modes_agree():
    """No information when all modes agree on detectability; positive when split."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(detectability_eig(weights, jnp.array([1.0, 1.0])), 0.0)
    assert detectability_eig(weights, jnp.array([1.0, 0.0])) > 0.0


def test_evaluate_candidates_returns_finite_nonnegative_eig():
    """evaluate_candidates scores a batch of epochs against a real mixture posterior."""
    astrom = _make_astrom(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixture = map_laplace_mixture_fit(
            MSUN_KG, DIST_PC, astrom_data=astrom, log_P_range=LOG_P_RANGE, k=3
        )
    candidates = jnp.linspace(0.0, 3.0 * TRUE_T, 8)
    result = evaluate_candidates(mixture, candidates, (5.0e-3) ** 2, MSUN_KG, DIST_PC)

    assert result["total_eig"].shape == (8,)
    assert jnp.all(jnp.isfinite(result["total_eig"]))
    assert jnp.all(result["total_eig"] >= -1e-6)
