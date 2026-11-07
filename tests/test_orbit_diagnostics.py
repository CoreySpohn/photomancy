"""Tests for orbit posterior diagnostics (physical samples, mode summaries)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from photomancy.orbit.diagnostics import mode_summary, sample_physical  # noqa: E402
from photomancy.orbit.eig import evaluate_orbit_candidates  # noqa: E402

# Keep all consumers of the fitted mixture in one xdist loadfile group.
MSUN_KG = 1.989e30
DIST_PC = 10.0
TRUE_T = 1096.0


# Medium: fitting and differentiating an orbit mixture includes a cold compile.
@pytest.mark.medium
def test_evaluate_orbit_candidates_returns_finite_nonnegative_eig(fitted_orbit_mixture):
    """Fails if scoring a fitted orbit mixture loses finite, nonnegative EIG."""
    post, problem = fitted_orbit_mixture
    candidates = jnp.linspace(0.0, 3.0 * TRUE_T, 3)
    result = evaluate_orbit_candidates(
        post, problem, candidates, (5.0e-3) ** 2, MSUN_KG, DIST_PC
    )

    assert result["total_eig"].shape == (3,)
    assert jnp.all(jnp.isfinite(result["total_eig"]))
    assert jnp.all(result["total_eig"] >= -1e-6)


# Medium: this consumer can pay for the shared orbit fixture's cold compile.
@pytest.mark.medium
def test_sample_physical_shapes_and_keys(fitted_orbit_mixture):
    """Fails if physical sampling loses period/eccentricity columns or valid periods."""
    post, problem = fitted_orbit_mixture
    phys = sample_physical(post, problem, jax.random.key(0), 50)
    assert "T" in phys and "e" in phys
    assert phys["T"].shape == (50,)
    assert jnp.all(jnp.isfinite(phys["T"]))


# Medium: this consumer can pay for the shared orbit fixture's cold compile.
@pytest.mark.medium
def test_mode_summary_structure(fitted_orbit_mixture):
    """Fails if a fitted mode loses its weight, evidence or physical parameters."""
    post, problem = fitted_orbit_mixture
    summary = mode_summary(post, problem)
    assert len(summary) == post.n_modes
    assert set(summary[0]) == {"weight", "log_evidence", "params"}
    assert "T" in summary[0]["params"]
