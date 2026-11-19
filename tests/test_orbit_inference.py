"""Tests for the chart orbit problem and the generic backends."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hwoutils.constants import G  # noqa: E402
from hwoutils.epoch import Epoch  # noqa: E402
from kinematics import predict_relative_astrometry  # noqa: E402
from orbit_fit import (  # noqa: E402
    _first_orbit,
    chart_problem,
    classical_coordinates,
    ecc_law,
)
from orbix.equations import period_a, period_to_sma  # noqa: E402
from orbix.system import (  # noqa: E402
    Body,
    ClassicalChart,
    EccentricityVectorChart,
    SkyFrame,
    System,
)

from photomancy.infer.legacy_posterior import MixturePosterior  # noqa: E402
from photomancy.measurements import RelativeAstromData  # noqa: E402
from photomancy.orbit.init import find_init  # noqa: E402
from photomancy.orbit.laplace import map_laplace_mixture_fit  # noqa: E402

MSUN_KG = 1.989e30
DIST_PC = 10.0
TRUE_T = 1096.0
PERIOD_RANGE_D = (600.0, 1600.0)
LOG_P_RANGE = tuple(float(jnp.log10(x)) for x in PERIOD_RANGE_D)
ASTROM_ERR = 5.0e-3


def _make_astrom(n_obs, seed=42):
    """Synthetic relative astrometry for a single planet at TRUE_T."""
    a_true = float(period_to_sma(TRUE_T, MSUN_KG))
    tp_true = -1.5 / (2.0 * jnp.pi / TRUE_T)
    np.random.seed(seed)  # noqa: NPY002 - Preserve seeded fixture stream.
    times = np.sort(
        np.random.uniform(0, 3 * TRUE_T, 20)  # noqa: NPY002 - Preserve seeded times.
    )[:n_obs]
    ra_true, dec_true = predict_relative_astrometry(
        jnp.array(times),
        a_true,
        0.15,
        0.5,
        2.3,
        jnp.cos(0.8),
        jnp.sin(0.8),
        tp_true,
        MSUN_KG,
        DIST_PC,
    )
    ra = jnp.array(
        np.array(ra_true) + np.random.randn(n_obs) * ASTROM_ERR  # noqa: NPY002 - Preserve seeded noise.
    )
    dec = jnp.array(
        np.array(dec_true) + np.random.randn(n_obs) * ASTROM_ERR  # noqa: NPY002 - Preserve seeded noise.
    )
    return RelativeAstromData(
        times=jnp.array(times),
        ra=ra,
        dec=dec,
        ra_err=jnp.full(n_obs, ASTROM_ERR),
        dec_err=jnp.full(n_obs, ASTROM_ERR),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


@pytest.fixture(scope="module")
def orbit_problem_and_inits():
    """Share one chart problem and a small blind grid across backend consumers.

    Initializer ranking has dedicated regressions; these consumers check the
    chart problem and backend. Keep two starts and the original period recovery bound.
    """
    astrom = _make_astrom(5)
    # Keep the blind grid inside the prior's period window.
    lo, hi = LOG_P_RANGE
    init_range = (lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
    starts, _ = find_init(
        astrom,
        MSUN_KG,
        DIST_PC,
        k=2,
        log_T_range=init_range,
        n_log_T=32,
        e_grid=jnp.array([0.0, 0.15, 0.4]),
        n_tp=12,
        chart=ClassicalChart(),
    )
    problem = chart_problem(
        astrom,
        {"orbit": starts[0]},
        star_mass_kg=MSUN_KG,
        dist_pc=DIST_PC,
        chart=ClassicalChart(),
        period_range_d=PERIOD_RANGE_D,
    )
    return problem, starts


# Medium: Building and evaluating the chart problem includes the first trace.
@pytest.mark.medium(reason="first chart density compile, 6 s cold")
def test_chart_problem_maps_between_flat_chart_and_physical(orbit_problem_and_inits):
    """The problem yields a finite flat logdensity, round-trips and decodes the period.

    Fails if flatten and unflatten disagree, the density is not finite at a blind
    start, the decoded period is off the start's semi-major axis by more than
    5 percent, or the decoded inclination leaves (-1, 1).
    """
    problem, starts = orbit_problem_and_inits
    z_init = problem.flatten({"orbit": starts[0]})

    assert jnp.isfinite(problem.logdensity(z_init))
    assert set(problem.unflatten(z_init)) == {"orbit"}
    np.testing.assert_allclose(
        problem.flatten(problem.unflatten(z_init)), z_init, rtol=0, atol=0
    )

    phys = problem.to_physical(z_init)
    init_T = float(period_a(jnp.exp(starts[0][0]), G * MSUN_KG))
    assert abs(float(phys["T"]) - init_T) / init_T < 0.05
    assert -1.0 < float(phys["cos_i"]) < 1.0
    assert len(problem.param_names) > 0


# Medium: Blind initialization and a multi-start orbit fit exceed the unit budget.
@pytest.mark.medium(reason="multi-start Laplace fit, 8 s cold")
def test_orbit_fit_through_generic_backend_recovers_period(orbit_problem_and_inits):
    """Orbit fitting run through the generic LaplaceMixtureBackend recovers T.

    The chart density plus a Thiele-Innes seeded multi-start Laplace mixture
    recover the true period within three Laplace standard deviations on sparse
    (n=5) astrometry.
    """
    problem, starts = orbit_problem_and_inits
    inits = jnp.stack([problem.flatten({"orbit": start}) for start in starts])

    post = map_laplace_mixture_fit(problem, inits, n_steps=100)

    assert isinstance(post, MixturePosterior)
    problem.require(post)
    best = jnp.argmax(post.log_evidences)
    t_fit = float(problem.to_physical(post.means[best])["T"])
    draws = jax.random.multivariate_normal(
        jax.random.key(0), post.means[best], post.covs[best], (2000,)
    )
    t_std = float(jnp.std(jax.vmap(problem.to_physical)(draws)["T"]))
    # Sparse (n=5) astrometry: the MAP sits about one posterior sigma from the
    # truth (spike: T = 1210 +/- 111 d against 1096 d), so bound by 3 sigma.
    assert 0 < t_std < 0.3 * TRUE_T
    assert abs(t_fit - TRUE_T) < 3 * t_std, f"t_fit={t_fit:.1f} vs {TRUE_T}"
    assert abs(t_fit - TRUE_T) / TRUE_T < 0.25


@pytest.mark.slow
def test_orbit_nested_sampling_recovers_period_and_evidence():
    """Nested sampling (jaxns) on the chart density -> SamplePosterior + log Z."""
    from photomancy import RelativeAstrometryTerm
    from photomancy.infer.legacy_posterior import SamplePosterior
    from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity
    from photomancy.orbit.nested import orbit_nested_sampling
    from photomancy.priors import FactorizedOrbitPrior

    astrom = _make_astrom(6)
    chart = EccentricityVectorChart()

    def build(orbit):
        return System(
            Body(1.0),
            Body(jnp.zeros(1)),
            jax.tree.map(lambda x: x[None], orbit),
            SkyFrame(DIST_PC, Epoch()),
        )

    density = build_chart_logdensity(
        chart,
        build,
        FactorizedOrbitPrior(_first_orbit, PERIOD_RANGE_D, ecc_law("kipping13")),
        star_mass_kg=MSUN_KG,
        terms=(RelativeAstrometryTerm(astrom),),
    )
    problem = ChartOrbitProblem(density, {"orbit": jnp.zeros(chart.n_coords)})
    post = orbit_nested_sampling(
        problem,
        max_samples=30000,
        num_samples=1500,
        key=jax.random.key(0),
    )
    assert isinstance(post, SamplePosterior)
    problem.require(post)
    assert post.samples.shape == (1500, chart.n_coords)
    assert bool(jnp.isfinite(post.evidence))
    # With normalized priors, Z cannot exceed the zero-residual Gaussian peak.
    # There is no analytic orbit log Z target to compare within sampling error.
    loglike_upper = -2 * astrom.times.size * jnp.log(jnp.sqrt(2 * jnp.pi) * ASTROM_ERR)
    assert float(post.evidence) <= float(loglike_upper)

    periods = jax.vmap(problem.to_physical)(post.samples)["T"]
    period_median = float(jnp.median(periods))
    period_std = float(jnp.std(periods))
    period_error = abs(period_median - TRUE_T)
    # Recover within 3 posterior standard deviations (seed 0: z = 1.82).
    # Six epochs cover less than half an orbit; also keep a loose absolute guard.
    assert 0 < period_std < float("inf")
    assert period_error < 3 * period_std, (
        f"period={period_median:.3f}, truth={TRUE_T}, std={period_std:.3f}, "
        f"z={period_error / period_std:.3f}"
    )
    assert period_error / TRUE_T < 0.25


@pytest.mark.parametrize("use_x64", [False, True])
def test_chart_density_respects_precision(use_x64):
    """The chart density follows JAX precision and stays finite in both modes.

    Fails if a hard-coded dtype promotes the log density in float32 mode or
    loses precision in float64 mode.
    """
    with jax.enable_x64(use_x64):
        astrom = _make_astrom(3)
        a = float(period_to_sma(TRUE_T, MSUN_KG))
        coords = classical_coordinates(a, 0.15, 0.5, 2.3, 0.8, -100.0, TRUE_T)
        problem = chart_problem(
            astrom,
            {"orbit": coords},
            star_mass_kg=MSUN_KG,
            dist_pc=DIST_PC,
            chart=ClassicalChart(),
            period_range_d=PERIOD_RANGE_D,
        )
        z = problem.flatten({"orbit": coords})
        ll = problem.logdensity(z)
        assert ll.dtype == (jnp.float64 if use_x64 else jnp.float32)
        assert jnp.isfinite(ll)
        assert z.dtype == ll.dtype
