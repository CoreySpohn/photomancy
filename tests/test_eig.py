"""Tests for the analytic Fisher EIG (Bayesian experimental design)."""

import jax.numpy as jnp

from photomancy.orbit.eig import (
    alias_breaking_eig,
    detectability_eig,
    geometric_eig,
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
