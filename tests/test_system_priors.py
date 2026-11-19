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
from hwoutils.constants import Msun2kg, two_pi
from hwoutils.epoch import Epoch
from orbix.system import (
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
)
from priorities import HalfNormal, LogUniform, TruncatedDistribution
from scipy import integrate, stats
from skyscapes.scene import FlatStar, Scene
from system_helpers import BOUNDS, _orbits, _scale_coordinates, _scene, ecc_law

from photomancy import RadialVelocityTerm, SystemLogDensity
from photomancy.measurements import RVData
from photomancy.orbit.inference import (
    build_chart_logdensity,
    orbit_chart_log_volume,
)
from photomancy.orbit.updater import FixedMassTarget
from photomancy.priors import (
    LOG_HAAR,
    PopulationPrior,
    orbit_log_prior,
)
from photomancy.system_density import ScalarCoordinate

ATOL = 1e-10


def _check(label, actual, expected):
    error = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
    print(f"{label}: max_error={error:.4g} atol={ATOL:g}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


# Frozen eccentricity-site log densities of the retired NumPyro laws at _orbits().e.
_ECC_LOG_PROB = {
    "disk": [0.0, 0.0, 0.0, 0.0],
    "rayleigh": [
        -0.7425684773214338,
        0.4711525969262529,
        0.4878499848627348,
        -1.3835927839369297,
    ],
    "vaneylen19": [
        2.500834850831562,
        -1.6282954465785795,
        -36.967316382786876,
        -128.19500123055093,
    ],
    "kipping13": [
        1.1792472661367226,
        0.7144769078510184,
        -0.2575328219006827,
        -2.409595213299103,
    ],
}
# Frozen log density and its gradient at the second point of the sampled-jitter test:
# the retired unconstrained NumPyro joint plus the log determinant of the map from the
# chart point to its unconstrained sites.
_OLD_JOINT = [
    -11.983164735105953,
    -15.124466212443133,
    -18.04801104334017,
    -21.734194578606996,
]
_OLD_GRAD = [
    -4.540111035731708,
    -1.400878134936334,
    9.818543780126888,
    2.424568770699403,
    -1.5065999591547925,
    5.900001510120951,
    -0.027998079921538157,
    4.6066979547215587e-17,
    5.94982916365241e-17,
    0.8227612236659537,
    -0.23563430122799972,
]


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
        law = partial(orbit_log_prior, eccentricity=ecc_law(name))
        _check(
            name,
            law(orbits),
            jnp.asarray(_ECC_LOG_PROB[name]) - jnp.log(orbits.e) + constant,
        )

        capped = partial(
            orbit_log_prior, eccentricity=TruncatedDistribution(ecc_law(name), high=cap)
        )

        def density(e, capped=capped):
            orbit = eqx.tree_at(lambda o: (o.ex, o.ey), one, (e, 0.0))
            return capped(orbit)

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
        assert np.isneginf(score(cap + 1e-9))
        assert np.isneginf(score(0.0))


def test_scalar_cdfs_and_support():
    """Integrate in physical units; check a partial CDF as well as total mass."""
    for name, (lo, hi) in BOUNDS.items():
        law = jax.jit(lambda x, lo=lo, hi=hi: LogUniform(lo, hi).log_prob(x))
        for end in (np.sqrt(lo * hi), hi):
            area, _ = integrate.quad(
                lambda x, law=law: np.exp(law(x)), lo, end, epsabs=1e-12, epsrel=1e-12
            )
            _check(name, area, stats.loguniform.cdf(end, lo, hi))
        assert np.all(np.isneginf(law(jnp.array([0.0, lo / 2, hi * 2]))))
    sigma = 0.8
    for end in (0.37, np.inf):
        area, _ = integrate.quad(
            lambda x: np.exp(HalfNormal(sigma).log_prob(x)),
            0,
            end,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        _check("jitter CDF", area, stats.halfnorm.cdf(end, scale=sigma))
    assert np.isneginf(HalfNormal(sigma).log_prob(-0.1))


def test_independent_nuisance_inputs():
    """Two terms consume different instrument coordinates on one shared model."""

    def residual(model, *, offset):
        return -0.5 * (model + offset) ** 2, model + offset

    def noise(model, *, sigma):
        return -jnp.log(sigma), sigma

    offset_prior = jax.scipy.stats.norm.logpdf
    sigma_prior = HalfNormal(1.5)
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
    """Remove each chart's Jacobian at identical orbits, including truncation.

    The reference is the closed-form fixed-mass element density against
    e de d(cos i) dW dw dM d(ln a): the truncated Beta law of e over e, 1/2 for
    cos i, 1/(2 pi) per angle and 1/ln(a_hi/a_lo), written with SciPy and no
    photomancy code. The updater's fixed-mass target is one more chart of it.
    """
    target = FixedMassTarget(e_max=0.7)
    orbits = _orbits()
    e = np.asarray(orbits.e)
    ln_a_range = 2.0 / 3.0 * np.log(5000.0 / 50.0)
    reference = (
        stats.beta.logpdf(e, 0.867, 3.03)
        - stats.beta.logcdf(0.7, 0.867, 3.03)
        - np.log(e)
        - np.log(2.0)
        - 3.0 * np.log(two_pi)
        - np.log(ln_a_range)
    )
    reference = np.where(e < 0.7, reference, -np.inf)
    assert np.isneginf(reference[-1])
    point = EccentricityVectorChart().from_orbit(orbits)
    fixed = jax.vmap(target.log_prior)(point)
    assert np.isneginf(fixed[-1])
    _check(
        "FixedMassTarget",
        fixed[:-1]
        - jax.vmap(EccentricityVectorChart().log_volume)(point)[:-1]
        - LOG_HAAR,
        reference[:-1],
    )
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
            partial(
                orbit_log_prior,
                period_range_d=(50, 5000),
                eccentricity=TruncatedDistribution(ecc_law("kipping13"), high=0.7),
            ),
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


@pytest.mark.medium
def test_old_unconstrained_joint_with_sampled_jitter():
    """Fails if the pulled-back joint or gradient departs from the frozen reference."""
    chart = CartesianChart(scale_AU=1.6)
    data = RVData(
        times=jnp.array([0.0, 31.0, 77.0, 103.0, 0.0]),
        rv=jnp.array([0.4, -0.2, 0.1, 0.3, 0.0]),
        rv_err=jnp.ones(5),
        inst_ids=jnp.array([0, 1, 0, 1, 0]),
        n_inst=2,
        is_valid=jnp.array([True, True, True, True, False]),
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
            s.dynamics.orbits, period_range_d=(10, 1e4), eccentricity=ecc_law("disk")
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

    def compare(x):
        value, aux = density.value_and_aux(unpack(x))
        return value, aux["predictions"][0][-1]

    value, masked = eqx.filter_jit(jax.vmap(compare))(points)
    old_value = jnp.asarray(_OLD_JOINT)
    np.testing.assert_array_equal(masked, 0.0)
    _check("full old z pullback", value, old_value)
    x = points[1]
    new_grad = jax.jit(jax.grad(lambda x: density(unpack(x))))(x)
    old_grad = jnp.asarray(_OLD_GRAD)
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
        kipping_eccentricity,
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
        grid, types, PopulationConfig(eccentricity=kipping_eccentricity(0.9)), Epoch()
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
        values - chart.log_volume(params["orbit"]) - LOG_HAAR,
        expected + jnp.log(mass),
    )
    scene = jax.tree.map(lambda x: x[0], scenes)
    packed = Scene.from_scenes([scene], capacity=2)
    padded = jax.tree.map(lambda x: x[0], packed)
    _check("one valid population slot", eqx.filter_jit(prior)(padded), expected[0])
    absent = eqx.tree_at(lambda s: s.valid, scene, jnp.array([False]))
    assert np.isneginf(eqx.filter_jit(prior)(absent))
    assert np.isneginf(
        eqx.filter_jit(density)(
            {"orbit": params["orbit"][0], "mass_Mearth": jnp.log(25.0)}
        )
    )
