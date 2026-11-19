"""Regression coverage for total-mass propagation of fitted orbits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg
from kinematics import predict_photometry, predict_relative_astrometry
from orbit_fit import mass_problem
from orbix.equations import period_to_sma
from system_helpers import classical_coordinates, ecc_law

from photomancy.infer.legacy_posterior import MixturePosterior
from photomancy.orbit.diagnostics import orbits_from_samples
from photomancy.orbit.eig import evaluate_orbit_candidates
from photomancy.orbit.nested import orbit_nested_sampling

jax.config.update("jax_enable_x64", True)

ORBIT_NAMES = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")
MASS_RATIO = 0.1
MASS_MEARTH = MASS_RATIO * Msun2kg / Mearth2kg
PERIOD_D = 400.0


def _problem(mass_bounds=(1e3, 1e6), period_range_d=(10.0, 1e4)):
    a = period_to_sma(PERIOD_D, Msun2kg * (1.0 + MASS_RATIO))
    tp = -0.8 * PERIOD_D / (2 * jnp.pi)
    orbit = classical_coordinates(a, 0.35, 0.4, 1.1, 0.7, tp, PERIOD_D)
    return mass_problem(
        (),
        orbit,
        MASS_MEARTH,
        star_mass_kg=Msun2kg,
        dist_pc=10.0,
        period_range_d=period_range_d,
        mass_bounds=mass_bounds,
        eccentricity=ecc_law("disk"),
    )


@pytest.fixture
def massive_orbit():
    """A 400-day orbit with true mass ratio 0.1 at chart coordinates."""
    problem = _problem()
    z = problem.flatten(problem.position_template)
    return problem, z, problem.to_physical(z)


@pytest.mark.parametrize("observable", ["astrometry", "photometry"])
def test_eig_predictions_match_model_at_known_mass(massive_orbit, observable):
    """Fails if EIG drops the planet mass from the orbit size or dynamics.

    The reference is the independent single-orbit kinematics at total mass
    ``Ms + Mp`` with mass ratio 0.1, to rtol 1e-12.
    """
    problem, z, phys = massive_orbit
    t, Lambda = 173.0, 1e-9
    args = tuple(phys[k] for k in ORBIT_NAMES)
    mixture = problem.declare(
        MixturePosterior(
            means=z[None], covs=0.01 * jnp.eye(z.size)[None], log_evidences=jnp.zeros(1)
        )
    )
    curve = (jnp.array([0.0, 10.0]), jnp.array([80.0, 80.0]))
    result = evaluate_orbit_candidates(
        mixture,
        problem,
        jnp.array([t]),
        1e-6,
        contrast_curve=curve,
        area_AU2=Lambda,
    )
    if observable == "astrometry":
        expected = predict_relative_astrometry(
            jnp.array([t]), *args, Msun2kg, 10.0, Mp=phys["Mp"]
        )
        actual = result["predictions"][0, 0]
    else:
        expected = predict_photometry(
            jnp.array([t]), *args, Msun2kg, Lambda, 10.0, Mp=phys["Mp"]
        )
        actual = jnp.stack((result["separation"][0, 0], result["dMag"][0, 0]))
    np.testing.assert_allclose(
        actual, jnp.asarray(expected).squeeze(), rtol=1e-12, atol=0
    )


def test_posterior_orbit_preserves_fitted_semimajor_axis(massive_orbit):
    """Posterior tracks reconstruct the model's a using the sampled mass."""
    _, _, phys = massive_orbit
    samples = {k: jnp.atleast_1d(v) for k, v in phys.items()}
    orbit = orbits_from_samples(samples, Msun2kg, reference_day=0.0)
    np.testing.assert_allclose(orbit.a_AU, samples["a"], rtol=1e-12, atol=0)


def test_chart_coordinates_round_trip_total_mass_period(massive_orbit):
    """A 400-day massive orbit retains its period and mass through the chart."""
    _, _, phys = massive_orbit
    assert float(phys["T"]) == pytest.approx(PERIOD_D, rel=1e-9, abs=0)
    assert float(phys["Mp"]) == pytest.approx(MASS_MEARTH * Mearth2kg, rel=1e-12, abs=0)


# Medium: The real nested-sampler mass-column regression needs a cold sampler compile.
@pytest.mark.medium
def test_nested_posterior_carries_true_mass_for_period_inversion():
    """Nested rows keep the mass needed to invert their derived semimajor axis."""
    problem = _problem(
        mass_bounds=(MASS_MEARTH * (1 - 1e-3), MASS_MEARTH * (1 + 1e-3)),
        period_range_d=(399.0, 401.0),
    )
    post = orbit_nested_sampling(
        problem,
        num_live_points=16,
        max_samples=2000,
        num_samples=16,
        key=jax.random.key(12),
    )
    recovered = jax.vmap(problem.to_physical)(post.samples)
    assert bool(jnp.all((recovered["T"] >= 399.0) & (recovered["T"] <= 401.0)))
    assert bool(
        jnp.all(jnp.abs(recovered["Mp"] / (MASS_MEARTH * Mearth2kg) - 1) < 2e-3)
    )
    reference = period_to_sma(recovered["T"], Msun2kg + recovered["Mp"])
    np.testing.assert_allclose(recovered["a"], reference, rtol=1e-12, atol=0)
