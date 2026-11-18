"""Physical-prior normalization and sampler pullbacks at synthetic orbits.

All pointwise log-density comparisons use absolute tolerance 1e-10, the
float64 floor for the small sums here. Support and masks are exact.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Mearth2kg, Msun2kg, Rearth2AU, two_pi
from hwoutils.epoch import Epoch
from orbix.equations import period_to_sma
from orbix.system import (
    Body,
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    SkyFrame,
    System,
)
from scipy import integrate, stats
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet
from skyscapes.scene.system import System as Scene

from photomancy import RadialVelocityTerm, SystemLogDensity
from photomancy.orbit.data import RVData
from photomancy.orbit.inference import (
    build_chart_logdensity,
    build_numpyro_logdensity,
    orbit_chart_log_volume,
)
from photomancy.orbit.priors import ecc_distribution
from photomancy.orbit.updater import FixedMassTarget
from photomancy.priors import (
    PopulationPrior,
    half_normal_log_prior,
    log_uniform_prior,
    orbit_log_prior,
)
from photomancy.system_density import ScalarCoordinate

ATOL = 1e-10
BOUNDS = {
    "mass_Mearth": (0.01, 1e4),
    "radius_AU": (1e-5, 10**-2.5),
    "albedo": (0.01, 1.0),
}


def _check(label, actual, expected):
    error = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
    print(f"{label}: max_error={error:.4g} atol={ATOL:g}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


def _orbits(mass_Mearth=0.0):
    mass_kg = Msun2kg + Mearth2kg * mass_Mearth
    period_d = jnp.array([75.0, 320.0, 1130.0, 3800.0])
    x = jnp.stack(
        [
            jnp.log(period_to_sma(period_d, mass_kg)),
            jnp.array([-3.1, -1.6, -0.2, 1.4]),
            jnp.array([1.0, 0.2, -0.4, -1.2]),
            jnp.array([0.2, 1.7, 3.4, 5.6]),
            jnp.array([0.6, 2.1, 3.1, 4.4]),
            jnp.array([0.7, 1.6, 4.2, 5.7]),
        ],
        axis=-1,
    )
    return ClassicalChart().to_orbit(x, mu_AU3_per_d2=G * mass_kg)


def test_eccentricity_sites_and_conditional_cdfs():
    """The polar area factor and truncation constant affect different integrals."""
    orbits = _orbits()
    constant = np.log(1.5 / (2 * np.log(1e5) * two_pi**3))
    one = jax.tree.map(lambda x: x[0], orbits)
    cap = 0.73
    for name, marginal in (
        ("disk", stats.uniform()),
        ("rayleigh", stats.rayleigh(scale=0.3)),
        ("vaneylen19", stats.rayleigh(scale=0.049)),
        ("kipping13", stats.beta(0.867, 3.03)),
    ):
        law = partial(orbit_log_prior, ecc_prior=name)
        _check(
            name,
            law(orbits),
            ecc_distribution(name).log_prob(orbits.e) - jnp.log(orbits.e) + constant,
        )

        def density(e, law=law):
            orbit = eqx.tree_at(lambda o: (o.ex, o.ey), one, (e, 0.0))
            return law(orbit, e_max=cap)

        score = jax.jit(density)
        for end in (0.17, cap):
            area, _ = integrate.quad(
                lambda e, score=score: e * np.exp(score(e) - constant),
                0,
                end,
                epsabs=1e-12,
                epsrel=1e-12,
            )
            _check(f"{name} CDF", area, marginal.cdf(end) / marginal.cdf(cap))
        assert np.isneginf(score(cap))
        assert np.isfinite(score(0.0)) == (name in ("rayleigh", "vaneylen19"))


def test_scalar_cdfs_and_support():
    """Integrate in physical units; check a partial CDF as well as total mass."""
    for name, (lo, hi) in BOUNDS.items():
        law = jax.jit(partial(log_uniform_prior, bounds=(lo, hi)))
        for end in (np.sqrt(lo * hi), hi):
            area, _ = integrate.quad(
                lambda x, law=law: np.exp(law(x)), lo, end, epsabs=1e-12, epsrel=1e-12
            )
            _check(name, area, stats.loguniform.cdf(end, lo, hi))
        assert np.all(np.isneginf(law(jnp.array([0.0, lo / 2, hi * 2]))))
    sigma = 0.8
    for end in (0.37, np.inf):
        area, _ = integrate.quad(
            lambda x: np.exp(half_normal_log_prior(x, scale=sigma)),
            0,
            end,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        _check("jitter CDF", area, stats.halfnorm.cdf(end, scale=sigma))
    assert np.isneginf(half_normal_log_prior(-0.1, scale=sigma))


def test_independent_nuisance_inputs():
    """Two terms consume different instrument coordinates on one shared model."""

    def residual(model, *, offset):
        return -0.5 * (model + offset) ** 2, model + offset

    def noise(model, *, sigma):
        return -jnp.log(sigma), sigma

    offset_prior = jax.scipy.stats.norm.logpdf
    sigma_prior = partial(half_normal_log_prior, scale=1.5)
    density = SystemLogDensity(
        lambda p: p["signal"],
        (residual, noise),
        lambda model: 0.0,
        lambda p: 0.0,
        scalars={
            "offset": ScalarCoordinate(offset_prior, logarithmic=False),
            "sigma": ScalarCoordinate(sigma_prior),
        },
        term_inputs={0: ("offset",), 1: ("sigma",)},
    )
    params = dict(signal=0.4, offset=-0.7, sigma=np.log(2.0))
    value, aux = density.value_and_aux(params)
    _check("nuisance predictions", aux["predictions"], [-0.3, 2.0])
    _check("nuisance log volume", density.log_volume(params), np.log(2.0))
    _check(
        "nuisance joint",
        value,
        -0.5 * 0.3**2 + stats.norm.logpdf(-0.7) + stats.halfnorm.logpdf(2.0, scale=1.5),
    )


def test_fixed_mass_reference_across_charts():
    """Remove both charts' Jacobians at identical orbits, including truncation."""
    target = FixedMassTarget(e_max=0.7)
    orbits = _orbits()
    classical = orbits.to_classical()
    old = jax.vmap(target.from_physical)(
        jnp.stack(
            [
                orbits.a_AU,
                orbits.e,
                jnp.cos(classical["i_rad"]),
                classical["W_rad"],
                classical["w_rad"],
                classical["M0_rad"],
            ],
            axis=-1,
        )
    )
    reference = jax.vmap(target.log_prior)(old) - jnp.log(2 * orbits.e)
    assert np.isneginf(reference[-1])
    for chart in (
        ClassicalChart(),
        EccentricityVectorChart(),
        CartesianChart(scale_AU=2.1),
        FrameChart(),
        FrameChart(gauge_sd=None),
    ):
        density = build_chart_logdensity(
            chart,
            lambda o: o,
            partial(orbit_log_prior, period_range_d=(50, 5000), e_max=0.7),
            star_mass_kg=Msun2kg,
        )
        params = {"orbit": chart.from_orbit(orbits)}
        values, aux = eqx.filter_jit(eqx.filter_vmap(density.value_and_aux))(params)
        assert aux == {"term_values": (), "predictions": ()}
        assert np.isneginf(values[-1])
        physical = values - jax.vmap(density.log_volume)(params)
        _check(type(chart).__name__, physical[:-1], reference[:-1])
        if isinstance(chart, (ClassicalChart, EccentricityVectorChart)):
            outside = params["orbit"][0].at[-1].add(two_pi)
            assert np.isneginf(density({"orbit": outside}))


def test_frame_gauge_integral():
    """Integrate r^3 dr dS3 over both quaternion signs at zero disk offset."""
    for sd in (None, 0.1, 0.9):
        chart = FrameChart(gauge_sd=sd)
        volume = jax.jit(
            lambda r, chart=chart: orbit_chart_log_volume(
                chart, jnp.zeros(7).at[3].set(r)
            )
        )
        area, _ = integrate.quad(
            lambda r, volume=volume: 2 * np.pi**2 * r**3 * np.exp(volume(r)),
            0,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        _check(f"Frame gauge sd={sd}", area / (8 * np.pi**2), 1.0)


def _scene(orbit, *, mass_Mearth, radius_AU, albedo):
    dynamics = System(
        Body(1.0),
        Body(jnp.atleast_1d(mass_Mearth * Mearth2kg / Msun2kg)),
        jax.tree.map(lambda x: x[None], orbit),
        SkyFrame(10.0, Epoch()),
    )
    return Scene(
        FlatStar(1.0),
        (
            Planet(
                Rp_Rearth=jnp.atleast_1d(radius_AU / Rearth2AU),
                physical_model=LambertianPhysicalModel(Ag=jnp.atleast_1d(albedo)),
            ),
        ),
        dynamics,
    )


def _orbit_sites(orbit):
    """Constrained orbit sites at the same period, phase and orientation."""
    elem = orbit.to_classical()
    return dict(
        log_P=jnp.log10(orbit.period_d),
        e_raw=orbit.e,
        cos_i=jnp.cos(elem["i_rad"]),
        W=elem["W_rad"],
        w_raw=elem["w_rad"],
        M0=elem["M0_rad"],
    )


def _scale_coordinates(jitter_scale):
    """Physical measures for the three body scales and instrument jitters."""
    scalars = {
        name: ScalarCoordinate(partial(log_uniform_prior, bounds=bounds))
        for name, bounds in BOUNDS.items()
    }
    scalars["jitter_m_per_s"] = ScalarCoordinate(
        partial(half_normal_log_prior, scale=jitter_scale)
    )
    return scalars


@pytest.mark.medium
def test_old_unconstrained_joint_with_sampled_jitter():
    """Differentiate the full map to the actual unconstrained NumPyro sites."""
    chart = CartesianChart(scale_AU=1.6)
    data = RVData(
        times=jnp.array([0.0, 31.0, 77.0, 103.0, 0.0]),
        rv=jnp.array([0.4, -0.2, 0.1, 0.3, 0.0]),
        rv_err=jnp.ones(5),
        inst_ids=jnp.array([0, 1, 0, 1, 0]),
        n_inst=2,
        is_valid=jnp.array([True, True, True, True, False]),
    )
    old = build_numpyro_logdensity(
        Msun2kg,
        10.0,
        rv_data=data,
        sample_planet_scale=True,
        ecc_prior="disk",
        jitter_scale=0.8,
    )
    scalars = _scale_coordinates(0.8)
    poisoned = eqx.tree_at(
        lambda d: (d.rv, d.rv_err, d.times, d.inst_ids),
        data,
        (
            data.rv.at[-1].set(jnp.nan),
            data.rv_err.at[-1].set(jnp.nan),
            data.times.at[-1].set(jnp.nan),
            data.inst_ids.at[-1].set(99),
        ),
    )
    density = build_chart_logdensity(
        chart,
        _scene,
        lambda s: orbit_log_prior(
            s.dynamics.orbits, period_range_d=(10, 1e4), ecc_prior="disk"
        )[0],
        star_mass_kg=Msun2kg,
        terms=(RadialVelocityTerm(poisoned),),
        scalars=scalars,
        body_inputs=("radius_AU", "albedo"),
        term_inputs={0: ("jitter_m_per_s",)},
    )

    def unpack(x):
        return dict(
            orbit=x[:6],
            mass_Mearth=x[6],
            radius_AU=x[7],
            albedo=x[8],
            jitter_m_per_s=x[9:],
        )

    def old_coords(x):
        orbit = chart.to_orbit(
            x[:6], mu_AU3_per_d2=G * (Msun2kg + Mearth2kg * jnp.exp(x[6]))
        )
        sites = dict(
            **_orbit_sites(orbit),
            log_Mp=x[6] / jnp.log(10),
            log_Rp=x[7] / jnp.log(10),
            log_Ag=x[8] / jnp.log(10),
            jitter=jnp.exp(x[9:]),
        )
        return old.init_to_z(jax.tree.map(jnp.atleast_1d, sites))

    mass = jnp.array([0.5, 2.0, 8.0, 13.0])
    points = jnp.column_stack(
        (
            chart.from_orbit(_orbits(mass)),
            jnp.log(mass),
            jnp.log(jnp.linspace(0.0001, 0.001, 4)),
            jnp.log(jnp.linspace(0.1, 0.7, 4)),
            jnp.log(jnp.linspace(0.1, 0.6, 4)),
            jnp.log(jnp.linspace(1.0, 0.3, 4)),
        )
    )

    def reference(x):
        return (
            old.logdensity(old_coords(x))
            + jnp.linalg.slogdet(jax.jacfwd(old_coords)(x))[1]
        )

    def compare(x):
        value, aux = density.value_and_aux(unpack(x))
        return value, reference(x), aux["predictions"][0][-1]

    value, old_value, masked = eqx.filter_jit(jax.vmap(compare))(points)
    np.testing.assert_array_equal(masked, 0.0)
    _check("full old z pullback", value, old_value)
    x = points[1]
    new_grad = jax.jit(jax.grad(lambda x: density(unpack(x))))(x)
    old_grad = jax.jit(jax.grad(reference))(x)
    np.testing.assert_allclose(new_grad, old_grad, rtol=1e-9, atol=ATOL)
    print(f"pulled-back gradient: max_error={np.max(np.abs(new_grad - old_grad)):.4g}")
    assert np.isneginf(eqx.filter_jit(density)(unpack(x.at[:6].set(0.0))))


def test_fixed_count_population_intensity():
    """Check dnu dMearth and dnu dln(Mearth) against the same synthetic marks."""
    from skyscapes.population import (
        HostStars,
        OccurrencePopulation,
        PlanetTypes,
        PopulationConfig,
        occurrence_table_from_grid,
    )

    host = HostStars([7], [1.0], [1.0], [10.0], FlatStar(jnp.ones(1), jnp.zeros(1)))
    types = PlanetTypes(
        ("Mars",),
        jnp.zeros(1),
        jnp.full(1, 100.0),
        jnp.zeros(1),
        jnp.full(1, 100.0),
        jnp.ones(1),
        jnp.zeros(1, bool),
        jnp.array([400.0, 800.0]),
        jnp.full((1, 2), 0.3),
    )
    grid = occurrence_table_from_grid([[2.5]], [0.1, 8.0], [0.2, 20.0], [[0]])
    population = OccurrencePopulation(
        grid, types, PopulationConfig(e_min=0.01, e_max=0.9), Epoch()
    )
    prior = PopulationPrior(population, host)
    build = partial(population.mark_build, host, label=0)
    # Use the population's reader/builder units on every side of the comparison.
    mass = jnp.array([0.5, 2.0, 8.0, 13.0])
    orbits = _orbits(mass)
    scenes = eqx.filter_jit(jax.vmap(build))(orbits, mass)
    expected = (
        eqx.filter_jit(population.log_intensity)(scenes, host)[:, 0]
        - population.log_rate(host)[0]
    )
    assert np.isfinite(expected).all()
    _check(
        "population per Earth mass",
        eqx.filter_jit(eqx.filter_vmap(prior))(scenes),
        expected,
    )
    chart = ClassicalChart()
    density = build_chart_logdensity(
        chart,
        build,
        prior,
        star_mass_kg=Msun2kg,
        scalars={"mass_Mearth": ScalarCoordinate(None)},
    )
    params = {"orbit": chart.from_orbit(orbits), "mass_Mearth": jnp.log(mass)}
    values = eqx.filter_jit(eqx.filter_vmap(density))(params)
    _check(
        "population per log mass",
        values - chart.log_volume(params["orbit"]),
        expected + jnp.log(mass),
    )
    scene = jax.tree.map(lambda x: x[0], scenes)
    packed = Scene.from_systems([scene], capacity=2)
    padded = jax.tree.map(lambda x: x[0], packed)
    _check("one valid population slot", eqx.filter_jit(prior)(padded), expected[0])
    absent = eqx.tree_at(lambda s: s.valid, scene, jnp.array([False]))
    assert np.isneginf(eqx.filter_jit(prior)(absent))
    assert np.isneginf(
        eqx.filter_jit(density)(
            {"orbit": params["orbit"][0], "mass_Mearth": jnp.log(25.0)}
        )
    )
