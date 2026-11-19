"""Tests for the analytic Fisher EIG (Bayesian experimental design)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg, Rearth2AU, two_pi
from kinematics import predict_relative_astrometry
from orbit_fit import chart_problem, classical_coordinates, ecc_law, scale_problem
from orbix.equations import period_a, period_to_sma

from photomancy import RelativeAstrometryTerm
from photomancy.infer.legacy_posterior import MixturePosterior
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.eig import (
    alias_breaking_eig,
    detectability_eig,
    evaluate_orbit_candidates,
    geometric_eig,
)

DIST_PC = 10.0
TRUE_T = 1096.0
PERIOD_RANGE_D = (600.0, 1600.0)


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


@pytest.fixture(scope="module")
def imaging_problem():
    """One mode on an eccentric orbit, for imaging decisions at a few epochs."""
    astrom = RelativeAstromData.pad(
        times=[0.0, 30.0],
        ra=[0.10, 0.13],
        dec=[0.10, 0.07],
        ra_err=[0.01, 0.01],
        dec_err=[0.01, 0.01],
        corr=[0.0, 0.0],
        planet_id=[0, 0],
        max_n=2,
    )
    tp = -TRUE_T / two_pi
    orbit = classical_coordinates(
        float(period_to_sma(TRUE_T, Msun2kg)), 0.4, 0.95, 2.3, 0.8, tp, TRUE_T
    )
    problem = scale_problem(
        (RelativeAstrometryTerm(astrom),),
        orbit,
        star_mass_kg=Msun2kg,
        dist_pc=DIST_PC,
        period_range_d=PERIOD_RANGE_D,
    )
    # Radius and albedo whose product gives Lambda = 0.3 Rearth^2.
    position = dict(
        problem.position_template,
        radius_AU=jnp.log(Rearth2AU),
        albedo=jnp.log(0.3),
    )
    mean = problem.flatten(position)
    mix = problem.declare(
        MixturePosterior(
            means=mean[None, :],
            covs=0.01 * jnp.eye(mean.size)[None, :, :],
            log_evidences=jnp.zeros(1),
        )
    )
    return problem, mix, jnp.array([tp, tp + TRUE_T / 2])


@pytest.mark.medium(reason="cold chart density and EIG compile, 8 s cold")
def test_orbit_eig_reports_no_detection_outside_contrast_curve(imaging_problem):
    """Periapsis and apoapsis place the same orbit on opposite sides of the grid."""
    problem, mix, epochs = imaging_problem
    sep_grid = jnp.array([0.15, 0.25])
    result = evaluate_orbit_candidates(
        mix,
        problem,
        epochs,
        1e-5,
        contrast_curve=(sep_grid, jnp.array([50.0, 50.0])),
    )
    assert 0 < result["separation"][0, 0] < sep_grid[0]
    assert result["separation"][1, 0] > sep_grid[-1]
    assert jnp.all(result["dMag"] < 50.0)
    np.testing.assert_array_equal(result["detectability"], jnp.zeros((2, 1)))
    np.testing.assert_array_equal(result["total_eig"], jnp.zeros(2))


@pytest.mark.medium(reason="second EIG evaluation on the shared problem")
def test_orbit_eig_uses_current_contrast_curve_on_repeat_call(imaging_problem):
    """Changing only the magnitude limits changes the score on a repeat call."""
    problem, mix, epochs = imaging_problem
    sep_grid = jnp.array([0.05, 0.4])
    deep, shallow = [
        evaluate_orbit_candidates(
            mix,
            problem,
            epochs,
            1e-5,
            contrast_curve=(sep_grid, jnp.full(2, depth)),
        )
        for depth in (50.0, 1.0)
    ]
    assert jnp.all(
        (sep_grid[0] < deep["separation"]) & (deep["separation"] < sep_grid[-1])
    )
    assert jnp.all((1.0 < deep["dMag"]) & (deep["dMag"] < 50.0))
    np.testing.assert_array_equal(deep["detectability"], jnp.ones((2, 1)))
    np.testing.assert_array_equal(shallow["detectability"], jnp.zeros((2, 1)))
    assert jnp.all(deep["total_eig"] > 0.0)
    np.testing.assert_array_equal(shallow["total_eig"], jnp.zeros(2))


@pytest.mark.medium(reason="differentiate the complete chart-to-astrometry map")
def test_chart_eig_and_mode_predictions():
    """Fails if candidate scoring drops a photometric scale or swaps mode/epoch axes."""
    from orbix.system import EccentricityVectorChart
    from test_system_priors import _orbits, _scene
    from test_terms import _data

    from photomancy import RelativeAstrometryTerm, ScalarCoordinate
    from photomancy.orbit.diagnostics import (
        mode_observables,
        mode_summary,
        orbits_from_samples,
    )
    from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity

    chart = EccentricityVectorChart()
    orbit = jax.tree.map(lambda x: x[1], _orbits(17.0))
    term = RelativeAstrometryTerm(_data())
    density = build_chart_logdensity(
        chart,
        _scene,
        lambda system: 0.0,
        star_mass_kg=Msun2kg,
        terms=(term,),
        scalars={
            "albedo": ScalarCoordinate(None),
            "mass_Mearth": ScalarCoordinate(None),
            "radius_AU": ScalarCoordinate(None),
            "jitter_m_per_s": ScalarCoordinate(None),
        },
        body_inputs=("albedo", "radius_AU"),
    )
    params = {
        "orbit": chart.from_orbit(orbit),
        "mass_Mearth": jnp.log(17.0),
        "jitter_m_per_s": jnp.log(jnp.array([0.4, 1.6])),
    }
    bright_params = dict(params, albedo=jnp.log(0.8), radius_AU=jnp.log(Rearth2AU))
    problem = ChartOrbitProblem(density, bright_params)
    bright = problem.flatten(bright_params)
    faint = problem.flatten(
        dict(params, albedo=jnp.log(0.08), radius_AU=jnp.log(0.5 * Rearth2AU))
    )
    means = jnp.stack((bright, faint))
    posterior = problem.declare(
        MixturePosterior(
            means=means,
            covs=jnp.tile(0.01 * jnp.eye(bright.size), (2, 1, 1)),
            log_evidences=jnp.zeros(2),
        )
    )
    times = term.data.times
    summary = mode_summary(posterior, problem)
    for mode, area_AU2 in zip(
        summary, (0.8 * Rearth2AU**2, 0.08 * (0.5 * Rearth2AU) ** 2), strict=True
    ):
        physical = mode["params"]
        decoded = orbits_from_samples(physical, Msun2kg, reference_day=0.0)
        np.testing.assert_allclose(
            decoded.at(times)[0], orbit.at(times)[0], rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(physical["Lambda"], area_AU2, rtol=1e-12, atol=0)
        np.testing.assert_allclose(physical["jitter_m_per_s"], [0.4, 1.6], rtol=1e-12)
    aux = mode_observables(posterior, problem)
    result = evaluate_orbit_candidates(
        posterior,
        problem,
        times,
        1e-5,
        contrast_curve=(jnp.array([0.0, 1.0]), jnp.array([80.0, 80.0])),
    )
    # Mode diagnostics carry (K, 2, T), whereas EIG carries (T, K, 2).
    np.testing.assert_allclose(
        result["predictions"],
        aux["predictions"][0].transpose(2, 0, 1),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result["dMag"][:, 1] - result["dMag"][:, 0],
        2.5 * np.log10(40.0),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        result["detectability"], jnp.ones_like(result["dMag"])
    )


@pytest.mark.medium(reason="OFTI, clustering and the EIG forward, 10 s cold")
def test_ofti_to_eig_end_to_end():
    """OFTI -> chart coordinates -> cluster_to_mixture -> EIG runs and is finite.

    Fails if an OFTI orbit row cannot be re-expressed in the fit chart (the bridge
    that replaced the unconstrained-space conversion) or the declared mixture is
    refused by the EIG.
    """
    from orbix.system import EccentricityVectorChart

    from photomancy.infer.legacy_posterior import SamplePosterior, cluster_to_mixture
    from photomancy.orbit.diagnostics import orbits_from_samples
    from photomancy.orbit.ofti import ofti

    a = float(period_to_sma(TRUE_T, Msun2kg))
    tp = -1.5 / (two_pi / TRUE_T)
    times = jnp.linspace(0.0, 0.08 * TRUE_T, 4)  # short arc -> OFTI's regime
    ra, dec = predict_relative_astrometry(
        times, a, 0.15, 0.5, 2.3, jnp.cos(0.8), jnp.sin(0.8), tp, Msun2kg, DIST_PC
    )
    err = 5.0e-3
    astrom = RelativeAstromData(
        times=times,
        ra=ra,
        dec=dec,
        ra_err=jnp.full(4, err),
        dec_err=jnp.full(4, err),
        corr=jnp.zeros(4),
        planet_id=jnp.zeros(4, dtype=int),
        is_valid=jnp.ones(4, dtype=bool),
    )
    chart = EccentricityVectorChart()
    problem = chart_problem(
        astrom,
        {"orbit": jnp.zeros(chart.n_coords)},
        star_mass_kg=Msun2kg,
        dist_pc=DIST_PC,
        chart=chart,
        period_range_d=PERIOD_RANGE_D,
    )
    samp = ofti(
        astrom,
        Ms=Msun2kg,
        dist_pc=DIST_PC,
        key=jax.random.key(0),
        n_accept=1000,
        eccentricity=ecc_law("disk"),
        log_P_range=tuple(float(np.log10(x)) for x in PERIOD_RANGE_D),
        batch=50000,
        max_batches=300,
    )
    rows = dict(zip(samp.param_names, samp.samples.T, strict=True))
    rows["T"] = period_a(rows["a"], G * Msun2kg)
    orbits = orbits_from_samples(rows, Msun2kg, reference_day=0.0)
    coords = jax.vmap(chart.from_orbit)(orbits)
    zpost = SamplePosterior(
        samples=coords,
        log_weights=samp.log_weights,
        evidence=jnp.asarray(jnp.nan),
        param_names=problem.param_names,
    )
    mix = problem.declare(cluster_to_mixture(zpost, 4, key=jax.random.key(1)))
    epochs = jnp.linspace(0.0, 3.0 * TRUE_T, 12)
    res = evaluate_orbit_candidates(mix, problem, epochs, (5.0e-3) ** 2)
    assert res["total_eig"].shape == (12,)
    assert jnp.all(jnp.isfinite(res["total_eig"]))
