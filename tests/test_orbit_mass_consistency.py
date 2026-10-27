"""Regression coverage for total-mass propagation of fitted orbits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg

from photomancy.orbit.data import RVData
from photomancy.orbit.diagnostics import orbits_from_samples
from photomancy.orbit.eig import _predict_astrom_pure, _predict_sep_dmag_pure
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from photomancy.orbit.inference import build_orbit_logdensity, to_unconstrained
from photomancy.orbit.nested import orbit_nested_sampling
from photomancy.posterior import SamplePosterior

jax.config.update("jax_enable_x64", True)

ORBIT_NAMES = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")


@pytest.fixture
def massive_orbit():
    """A fitted 400-day orbit with true mass ratio 0.1."""
    problem = build_orbit_logdensity(
        Msun2kg,
        10.0,
        rv_data=RVData.zeros(),
        ecc_prior="disk",
        log_Mp_range=(3.0, 6.0),
    )
    z = problem.init_to_z(
        {
            "log_P": jnp.array([jnp.log10(400.0)]),
            "log_Mp": jnp.array([jnp.log10(0.1 * Msun2kg / Mearth2kg)]),
            "e_raw": jnp.array([0.35]),
            "cos_i": jnp.array([0.4]),
            "W": jnp.array([1.1]),
            "w_raw": jnp.array([0.7]),
            "M0": jnp.array([0.8]),
        }
    )
    return problem, z, problem.to_physical(z)


@pytest.mark.parametrize("observable", ["astrometry", "photometry"])
def test_eig_predictions_match_model_at_known_mass(massive_orbit, observable):
    """EIG uses the fitted orbit size and dynamics at mass ratio 0.1."""
    problem, z, phys = massive_orbit
    t, Lambda = 173.0, 1e-9
    args = tuple(phys[k] for k in ORBIT_NAMES)
    if observable == "astrometry":
        expected = predict_relative_astrometry(
            jnp.array([t]), *args, Msun2kg, 10.0, Mp=phys["Mp"]
        )
        actual = _predict_astrom_pure(
            z, t, problem.unflatten, problem.constrain, Msun2kg, 10.0
        )
    else:
        expected = predict_photometry(
            jnp.array([t]), *args, Msun2kg, Lambda, 10.0, Mp=phys["Mp"]
        )
        actual = _predict_sep_dmag_pure(
            z, t, problem.unflatten, problem.constrain, Msun2kg, 10.0, Lambda
        )
    np.testing.assert_allclose(
        actual, jnp.asarray(expected).squeeze(), rtol=1e-12, atol=0
    )


def test_posterior_orbit_preserves_fitted_semimajor_axis(massive_orbit):
    """Posterior tracks reconstruct the model's a using the sampled mass."""
    _, _, phys = massive_orbit
    samples = {k: jnp.atleast_1d(v) for k, v in phys.items()}
    orbit = orbits_from_samples(samples, Msun2kg)
    np.testing.assert_allclose(orbit.a_AU, samples["a"], rtol=1e-12, atol=0)


def test_to_unconstrained_round_trips_total_mass_period(massive_orbit):
    """A 400-day massive orbit retains its period and mass through z-space."""
    problem, _, phys = massive_orbit
    names = (*ORBIT_NAMES, "Mp")
    post = SamplePosterior(
        samples=jnp.array([[phys[k] for k in names]]),
        log_weights=jnp.zeros(1),
        evidence=jnp.array(2.0),
        param_names=names,
    )
    zpost = to_unconstrained(post, problem, Msun2kg)
    recovered = problem.to_physical(zpost.samples[0])
    assert float(recovered["T"]) == pytest.approx(400.0, rel=1e-9, abs=0)
    assert float(recovered["Mp"]) == pytest.approx(float(phys["Mp"]), rel=1e-12, abs=0)
    np.testing.assert_array_equal(zpost.log_weights, post.log_weights)
    assert zpost.evidence == post.evidence


def test_nested_posterior_carries_true_mass_for_period_inversion():
    """Nested rows keep the mass needed to invert their derived semimajor axis."""
    log_mass = float(jnp.log10(0.1 * Msun2kg / Mearth2kg))
    post = orbit_nested_sampling(
        Msun2kg,
        10.0,
        rv_data=RVData.zeros(),
        ecc_prior="disk",
        log_P_range=(float(jnp.log10(399.0)), float(jnp.log10(401.0))),
        log_Mp_range=(log_mass - 1e-3, log_mass + 1e-3),
        num_live_points=16,
        max_samples=2000,
        num_samples=16,
        key=jax.random.key(12),
    )
    assert post.param_names == (*ORBIT_NAMES, "Mp")
    problem = build_orbit_logdensity(
        Msun2kg,
        10.0,
        rv_data=RVData.zeros(),
        ecc_prior="disk",
        log_Mp_range=(log_mass - 1e-3, log_mass + 1e-3),
    )
    zpost = to_unconstrained(post, problem, Msun2kg)
    recovered = jax.vmap(problem.to_physical)(zpost.samples)
    assert bool(jnp.all((recovered["T"] >= 399.0) & (recovered["T"] <= 401.0)))
    np.testing.assert_allclose(recovered["a"], post.samples[:, 0], rtol=1e-12, atol=0)
