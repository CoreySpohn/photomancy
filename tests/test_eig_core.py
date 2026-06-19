"""Tests for the domain-agnostic EIG layer."""

import jax.numpy as jnp

from photomancy.eig import (
    alias_breaking_eig,
    detectability_eig,
    evaluate_candidates,
    geometric_eig,
)
from photomancy.posterior import MixturePosterior


def _two_mode_posterior():
    return MixturePosterior(
        means=jnp.array([[1.0], [2.0]]),
        covs=jnp.array([[[1.0]], [[1.0]]]),
        log_evidences=jnp.array([0.0, 0.0]),
    )


def test_geometric_eig_matches_analytic_gaussian_update():
    """The log-det shrinkage equals the 1-D conjugate-Gaussian information gain."""
    eig, cov_new = geometric_eig(jnp.array([[4.0]]), jnp.array([[1.0]]), 1.0)
    assert jnp.allclose(eig, 0.5 * jnp.log(5.0))
    assert float(cov_new[0, 0]) < 4.0


def test_alias_breaking_eig_zero_when_modes_agree():
    """Zero when modes predict the same observable; positive when they differ."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(
        alias_breaking_eig(weights, jnp.array([[1.0], [1.0]]), 1.0), 0.0
    )
    assert alias_breaking_eig(weights, jnp.array([[1.0], [5.0]]), 1.0) > 0.0


def test_detectability_eig_zero_when_modes_agree():
    """Zero when all modes agree on detectability; positive when split."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(detectability_eig(weights, jnp.array([1.0, 1.0])), 0.0)
    assert detectability_eig(weights, jnp.array([1.0, 0.0])) > 0.0


def test_evaluate_candidates_linear_forward_more_informative_scores_higher():
    """A larger Jacobian and wider mode disagreement give more information."""
    posterior = _two_mode_posterior()

    def forward(z, c):
        return jnp.array([c * z[0]])

    res = evaluate_candidates(posterior, jnp.array([0.5, 2.0]), forward, 1.0)
    assert res["total_eig"].shape == (2,)
    assert jnp.all(jnp.isfinite(res["total_eig"]))
    assert jnp.all(res["total_eig"] >= -1e-6)
    assert res["total_eig"][1] > res["total_eig"][0]
    assert res["predictions"].shape == (2, 2, 1)


def test_evaluate_candidates_detectable_splits_modes():
    """A detectable callback that splits the modes drives the detectability output."""
    posterior = _two_mode_posterior()

    def forward(z, c):
        return jnp.array([c * z[0]])

    def detectable(z, c):
        return jnp.where(z[0] < 1.5, 1.0, 0.0)  # mode 0 detectable, mode 1 not

    res = evaluate_candidates(
        posterior, jnp.array([1.0]), forward, 1.0, detectable=detectable
    )
    assert res["detectability"].shape == (1, 2)
    assert jnp.allclose(res["detectability"][0], jnp.array([1.0, 0.0]))
    assert jnp.isfinite(res["total_eig"][0])
