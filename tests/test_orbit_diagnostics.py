"""Tests for orbit posterior diagnostics (physical samples, mode summaries)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from orbit_fit import chart_problem  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402
from orbix.system import ClassicalChart  # noqa: E402
from sky_reference import sky_position  # noqa: E402

from photomancy.measurements import RelativeAstromData  # noqa: E402
from photomancy.orbit.diagnostics import mode_summary, sample_physical  # noqa: E402
from photomancy.orbit.eig import evaluate_orbit_candidates  # noqa: E402
from photomancy.orbit.init import find_init  # noqa: E402
from photomancy.orbit.laplace import map_laplace_mixture_fit  # noqa: E402

# Keep all consumers of the fitted mixture in one xdist loadfile group.
MSUN_KG = 1.989e30
DIST_PC = 10.0
TRUE_T = 1096.0


@pytest.fixture(scope="module")
def fitted_orbit_mixture():
    """One small, fitted three-component posterior for EIG and diagnostic checks.

    These checks consume a fitted posterior; blind initialization and period
    recovery have dedicated tests. Three blind starts exercise the mixture path.
    """
    mass, distance, period, error = 1.989e30, 10.0, 1096.0, 5e-3
    log_range = (float(np.log10(600.0)), float(np.log10(1600.0)))
    times = period * np.array([0.0, 0.23, 0.71, 1.17, 1.89])
    ra, dec = sky_position(
        times,
        T=period,
        a=float(period_to_sma(period, mass)),
        e=0.15,
        cos_i=0.5,
        W=2.3,
        cos_w=np.cos(0.8),
        sin_w=np.sin(0.8),
        tp=-1.5 * period / (2.0 * np.pi),
        dist_pc=distance,
    )
    noise = np.random.default_rng(0).normal(0.0, error, (2, 5))
    data = RelativeAstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(ra + noise[0]),
        dec=jnp.asarray(dec + noise[1]),
        ra_err=jnp.full(5, error),
        dec_err=jnp.full(5, error),
        corr=jnp.zeros(5),
        planet_id=jnp.zeros(5, dtype=int),
        is_valid=jnp.ones(5, dtype=bool),
    )
    chart = ClassicalChart()
    starts, _ = find_init(data, mass, distance, k=3, log_T_range=log_range, chart=chart)
    problem = chart_problem(
        data,
        {"orbit": starts[0]},
        star_mass_kg=mass,
        dist_pc=distance,
        chart=chart,
        period_range_d=(600.0, 1600.0),
    )
    init = jnp.stack([problem.flatten({"orbit": start}) for start in starts])
    post = map_laplace_mixture_fit(problem, init, n_steps=100)
    return post, problem


# Medium: fitting and differentiating an orbit mixture includes a cold compile.
@pytest.mark.medium
def test_evaluate_orbit_candidates_returns_finite_nonnegative_eig(
    fitted_orbit_mixture,
):
    """Fails if scoring a fitted orbit mixture loses finite, nonnegative EIG."""
    post, problem = fitted_orbit_mixture
    candidates = jnp.linspace(0.0, 3.0 * TRUE_T, 3)
    result = evaluate_orbit_candidates(post, problem, candidates, (5.0e-3) ** 2)

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
