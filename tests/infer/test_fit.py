"""Fit: density against the old system density, refusals, round trip, orbit block.

The golden file holds ``SystemLogDensity`` values from the base commit (photomancy
stack-v3 with its orbix) at chart coordinates of three systems. The new ``Fit`` writes
the same orbit through an ``OrbitBlock`` on a real ``Scene``, so the two densities agree
up to named changes of coordinates (the offset is zero to rounding).
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2, Mearth2kg, Msun2kg, Rearth2AU
from hwoutils.model_paths import resolve
from optixstuff.spectral import Band, Quadrature, Transmission
from orbix.system import ClassicalChart, OrbitBlock, OrbitLaw
from priorities import LogUniform, Normal, TruncatedDistribution, Uniform
from priorities.testing import check_distribution
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.population import kipping_eccentricity
from skyscapes.scene import FlatStar, Planet, Scene

from photomancy import (
    PhotometryTerm,
    ProperMotionAnomalyTerm,
    RadialVelocityTerm,
    RelativeAstrometryTerm,
    StellarAstrometryTerm,
)
from photomancy.infer import AbstractTerm, Fit
from photomancy.measurements import (
    AstrometryData,
    PhotometryData,
    ProperMotionAnomalyData,
    RadialVelocityData,
)

jax.config.update("jax_enable_x64", True)

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "data" / "fit_density_golden.json").read_text()
)
TIMES = np.array([31.0, 120.0, 290.0, 470.0])
PERIOD_RANGE = (600.0, 1600.0)
MASS_BOUNDS = (1e3, 1e6)
RADIUS_BOUNDS = (1e-5, 10**-2.5)
ALBEDO_BOUNDS = (0.01, 1.0)
EARTH_MASS_MSUN = Mearth2kg / Msun2kg


def _relative():
    return RelativeAstrometryTerm(
        AstrometryData.from_separation_pa(
            time_d=TIMES,
            separation_arcsec=[0.13, 0.11, 0.09, 0.15],
            separation_err_arcsec=[0.004] * 4,
            pa_deg=[45.0, 120.0, 210.0, 310.0],
            pa_err_deg=[1.0] * 4,
            max_n=4,
        )
    )


def _stellar():
    return StellarAstrometryTerm(
        AstrometryData.pad(
            time_d=TIMES,
            east_arcsec=[1e-4, -2e-4, 3e-4, 2e-4],
            north_arcsec=[-1e-4, 1e-4, 2e-4, -2e-4],
            east_err_arcsec=[5e-5] * 4,
            north_err_arcsec=[8e-5] * 4,
            corr=[0.2, -0.3, 0.1, 0.0],
            max_n=4,
        )
    )


def _rv():
    return RadialVelocityTerm(
        RadialVelocityData.pad(
            time_d=TIMES,
            rv=[2.0, -4.0, 8.0, 1.0],
            rv_err=[1.5] * 4,
            inst_ids=[0, 1, 0, 1],
            n_inst=3,
            max_n=4,
        ),
        jitters=jnp.array([0.5, 0.8, 1.0]),
    )


def _pma():
    return ProperMotionAnomalyTerm(
        ProperMotionAnomalyData(
            t_hip=jnp.array(0.0),
            t_gaia=jnp.array(9000.0),
            gaia_window=jnp.array(1023.0),
            pm_anomaly=jnp.array([2e-7, -3e-7]),
            pm_anomaly_cov=jnp.array([[4e-14, 1e-14], [1e-14, 9e-14]]),
            valid=jnp.array(True),
        )
    )


def _photometry():
    band = Band(550.0, 650.0)
    return PhotometryTerm(
        PhotometryData.from_dmag(
            time_d=TIMES,
            dmag=[25.0, 25.2, 24.5, 26.0],
            dmag_err=[0.2] * 4,
            band_id=[0] * 4,
            is_upper_limit=[False, True, False, True],
            max_n=4,
        ),
        Quadrature.build((band,)),
        (Transmission.from_band(band),),
    )


def _scene(planet_mass_Msun=0.0, Rp_Rearth=1.0, albedo=0.3):
    planet = Planet(
        Rp_Rearth=jnp.array([Rp_Rearth]),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([albedo])),
    )
    return Scene.from_classical(
        FlatStar(1.0),
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=planet_mass_Msun,
        dist_pc=10.0,
        a_AU=1.5,
        e=0.3,
        i_rad=0.8,
        W_rad=1.1,
        w_rad=0.7,
        M0_rad=0.4,
        order=(0,),
    )


def _block(law=None, chart=None):
    law = law or OrbitLaw(
        period_d=LogUniform(*PERIOD_RANGE), eccentricity=kipping_eccentricity()
    )
    return OrbitBlock(chart or ClassicalChart(), law, rows=(0,), system="dynamics")


COORDS = _block().coordinate_names


def _orbit_map(mass=0.0):
    """The golden block's chart and the chart-coordinate bounds its law maps."""
    block, model = _block(), _scene(planet_mass_Msun=mass)
    return block.chart, block.chart_support(model)


def _coords(point, mass=0.0):
    """Block coordinates of a golden point, given in chart coordinates."""
    chart, support = _orbit_map(mass)
    y = chart.unconstrain(jnp.asarray(point)[None, :], support)
    return {name: y[:, k] for k, name in enumerate(COORDS)}


def _chart_coords(block_coords, mass=0.0):
    """Chart coordinates ``(..., 6)`` and ``log |dx/dy|`` of block coordinates."""
    chart, support = _orbit_map(mass)
    y = jnp.stack([jnp.asarray(block_coords[n]) for n in COORDS], axis=-1)
    x, log_jac = chart.constrain(y, support)
    return x[..., 0, :], log_jac[..., 0]


def _chart_jacobian(x, mass=0.0):
    """Old density minus new for an orbit entry: ``-log |dx/dy|`` of the block map."""
    return -float(_chart_coords(x["dynamics.orbits"], mass)[1])


def _check_against_old(fit, golden, entries, jacobian, *, constant=None):
    """New density equals old plus a change-of-variable term and one constant.

    ``entries(i)`` gives the new draw ``x`` of point ``i`` and ``jacobian(x)`` the log
    of ``d(old coordinate) / d(new coordinate)`` inverted (old = new + jacobian).
    """
    prior, like, total = [], [], []
    for i, old in enumerate(golden["values"]):
        x = entries(i)
        prior.append(old["logprior"] - float(fit.log_prior(x)) - jacobian(x))
        like.append(float(fit.log_likelihood(x)) / old["loglike"] - 1.0)
        total.append(old["total"] - float(fit.log_density(x)) - jacobian(x))
    prior, like, total = map(np.asarray, (prior, like, total))
    # The likelihood is the same arithmetic: relative agreement to rounding.
    np.testing.assert_allclose(like, 0.0, atol=1e-12)
    # The prior differs by one constant per system (zero, once the Jacobian is in).
    np.testing.assert_allclose(prior, prior[0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(total, prior, rtol=0, atol=1e-9 + 1e-12 * 1e5)
    if constant is not None:
        np.testing.assert_allclose(prior[0], constant, rtol=0, atol=1e-9)
    return float(prior[0])


@pytest.fixture(scope="module")
def offsets():
    """The constant by which each golden prior exceeds the new normalized prior."""
    return {}


def test_golden_fixed_mass_astrometry(offsets):
    """Fixed mass, relative astrometry: same density up to the chart constant."""
    fit = Fit(_scene(), {"dynamics.orbits": _block()}, (_relative(),))
    golden = GOLDEN["g1"]
    offsets["g1"] = _check_against_old(
        fit,
        golden,
        lambda i: {"dynamics.orbits": _coords(golden["points"][i])},
        _chart_jacobian,
    )


def test_golden_free_mass_kinematics(offsets):
    """Free planet mass with stellar astrometry, RV and proper-motion anomaly.

    The old coordinate is ``ln`` of the mass in Earth masses and the new leaf is
    the mass in solar masses, so old = new + log(mass_Msun).
    """
    law = LogUniform(MASS_BOUNDS[0] * EARTH_MASS_MSUN, MASS_BOUNDS[1] * EARTH_MASS_MSUN)
    model = _scene(planet_mass_Msun=0.1)
    fit = Fit(
        model,
        {
            "dynamics.planets.mass_Msun": jax.tree.map(jnp.asarray, law),
            "dynamics.orbits": _block(),
        },
        (_stellar(), _rv(), _pma()),
    )
    golden = GOLDEN["g2"]

    def entries(i):
        mass = jnp.exp(golden["ln_mass_Mearth"][i]) * EARTH_MASS_MSUN
        return {
            "dynamics.planets.mass_Msun": jnp.array([mass]),
            "dynamics.orbits": _coords(golden["points"][i], float(mass)),
        }

    offsets["g2"] = _check_against_old(
        fit,
        golden,
        entries,
        lambda x: (
            float(jnp.log(x["dynamics.planets.mass_Msun"][0]))
            + _chart_jacobian(x, float(x["dynamics.planets.mass_Msun"][0]))
        ),
    )


def test_golden_free_radius_albedo_photometry(offsets):
    """Free radius and albedo with astrometry and photometry on scene leaves."""
    rp = LogUniform(RADIUS_BOUNDS[0] / Rearth2AU, RADIUS_BOUNDS[1] / Rearth2AU)
    fit = Fit(
        _scene(),
        {
            "planets[0].Rp_Rearth": rp,
            "planets[0].physical_model.Ag": LogUniform(*ALBEDO_BOUNDS),
            "dynamics.orbits": _block(),
        },
        (_relative(), _photometry()),
    )
    golden = GOLDEN["g3"]

    def entries(i):
        return {
            "planets[0].Rp_Rearth": jnp.exp(golden["ln_radius_AU"][i])
            / Rearth2AU
            * jnp.ones(1),
            "planets[0].physical_model.Ag": jnp.exp(golden["ln_albedo"][i])
            * jnp.ones(1),
            "dynamics.orbits": _coords(golden["points"][i]),
        }

    def jacobian(x):
        return float(
            jnp.log(x["planets[0].Rp_Rearth"][0])
            + jnp.log(x["planets[0].physical_model.Ag"][0])
        ) + _chart_jacobian(x)

    offsets["g3"] = _check_against_old(fit, golden, entries, jacobian)


def test_golden_prior_offset_is_zero(offsets):
    """The old chart prior was already normalized: no constant separates the two.

    Both densities are against the same measure, so the offsets are zero to
    rounding on every golden system.
    """
    values = np.array([offsets[k] for k in ("g1", "g2", "g3")])
    np.testing.assert_allclose(values, 0.0, rtol=0, atol=1e-9)


# ----- refusals -----------------------------------------------------------------


def _simple_fit(terms=()):
    free = {"planets[0].Rp_Rearth": LogUniform(0.5, 4.0)}
    return Fit(_scene(), free, terms)


class _ReadsRadius(AbstractTerm):
    """A term whose build holds the planet radius."""

    reads = ("planets[0].Rp_Rearth",)

    def __call__(self, model):
        return jnp.zeros(())

    def draw(self, key, model):
        return self


def test_refuses_a_free_key_that_meets_term_reads():
    """A free path inside a term's read set is refused, naming path and term."""
    with pytest.raises(ValueError, match=r"Rp_Rearth.*_ReadsRadius"):
        _simple_fit(terms=(_ReadsRadius(),))
    # a free parent of the read path overlaps too
    with pytest.raises(ValueError, match="planets"):
        Fit(_scene(), {"planets[0]": LogUniform(0.5, 4.0)}, (_ReadsRadius(),))
    # no free key: the same term is fine
    Fit(_scene(), {}, (_ReadsRadius(),))


def test_refuses_a_block_write_that_meets_term_reads():
    """A block whose write overlaps a read is refused, as a free leaf is."""

    class Orbits(AbstractTerm):
        reads = property(lambda self: ("dynamics.orbits",))

        def __call__(self, model):
            return jnp.zeros(())

        def draw(self, key, model):
            return self

    with pytest.raises(ValueError, match=r"dynamics\.orbits.*Orbits\.reads"):
        Fit(_scene(), {"dynamics.orbits": _block()}, (Orbits(),))


def test_refuses_two_writers():
    """A leaf law on a path a block writes, and a block not keyed by its node."""
    from orbix.system import ThieleInnesBlock

    block = ThieleInnesBlock(
        _block().law, Uniform(0.9, 1.3), rows=(0,), system="dynamics"
    )
    with pytest.raises(ValueError, match="two writers"):
        Fit(
            _scene(),
            {
                "dynamics.orbits": block,
                "dynamics.star.mass_Msun": Uniform(0.9, 1.3),
            },
        )
    with pytest.raises(ValueError, match="must be keyed by the path it owns"):
        Fit(_scene(), {"dynamics.orbit": _block()})


def test_refuses_unknown_paths_and_bad_entries():
    """A path the model lacks, a node that is not an array leaf, a wrong kind."""
    with pytest.raises(ValueError, match="unknown model path"):
        Fit(_scene(), {"planets[0].radius": LogUniform(0.5, 4.0)})
    with pytest.raises(ValueError, match="not one array leaf"):
        Fit(_scene(), {"planets[0]": LogUniform(0.5, 4.0)})
    with pytest.raises(TypeError, match="distribution or a block"):
        Fit(_scene(), {"planets[0].Rp_Rearth": 3.0})
    with pytest.raises(ValueError, match="parameter shape"):
        Fit(
            _scene(), {"planets[0].Rp_Rearth": LogUniform(jnp.ones(3), 2 * jnp.ones(3))}
        )


def test_refuses_an_index_into_an_array_leaf():
    """``leaf[0]`` into an array is not a path: refused as an unknown path."""
    with pytest.raises(ValueError, match="unknown model path"):
        Fit(_scene(), {"planets[0].Rp_Rearth[0]": LogUniform(0.5, 4.0)})


def test_refuses_a_changed_free_set():
    """with_priors keeps the free set and the layout of a block.

    A block of another chart is refused (``same_layout``). A law whose bounds move is
    a swap here, as is a law with the same bounds and another shape: whether stored
    draws survive is the posterior's question (``same_map``), not the fit's.
    """
    fit = Fit(
        _scene(),
        {"planets[0].Rp_Rearth": LogUniform(0.5, 4.0), "dynamics.orbits": _block()},
        (_relative(),),
    )
    with pytest.raises(ValueError, match="cannot change the free set"):
        fit.with_priors({"planets[0].physical_model.Ag": Uniform(0.0, 1.0)})
    with pytest.raises(ValueError, match="same layout"):
        fit.with_priors(
            {
                "dynamics.orbits": _block(
                    chart=__import__("orbix").system.EccentricityVectorChart()
                )
            }
        )
    with pytest.raises(ValueError, match="priorities distribution"):
        fit.with_priors({"planets[0].Rp_Rearth": _block()})
    swapped = fit.with_priors({"planets[0].Rp_Rearth": LogUniform(1.0, 2.0)})
    assert swapped.free["planets[0].Rp_Rearth"].low == 1.0
    moved = OrbitLaw(period_d=LogUniform(100.0, 3000.0), eccentricity=Uniform(0.0, 1.0))
    wide = fit.with_priors({"dynamics.orbits": _block(moved)})
    assert not fit.free["dynamics.orbits"].same_map(wide.free["dynamics.orbits"])
    reshaped = OrbitLaw(
        period_d=LogUniform(*PERIOD_RANGE), eccentricity=Uniform(0.0, 1.0)
    )
    assert (
        fit.with_priors({"dynamics.orbits": _block(reshaped)}).digest() != fit.digest()
    )


def test_refuses_a_term_that_is_not_an_abstract_term():
    """A term on the old protocol cannot enter a fit: TypeError names the index."""
    with pytest.raises(TypeError, match=r"terms\[1\] is a _LegacyTerm"):
        _simple_fit(terms=(_ReadsNothing(), _LegacyTerm()))


class _LegacyTerm(eqx.Module):
    def __call__(self, model):
        return jnp.zeros(()), None


class _ReadsNothing(AbstractTerm):
    reads = ()

    def __call__(self, model):
        return jnp.zeros(())

    def draw(self, key, model):
        return self


class _RadiusReading(AbstractTerm):
    """A caller-defined Gaussian reading of the planet radius, holding its own value."""

    observed: jax.Array
    reads = ()

    def __call__(self, model):
        return -0.5 * ((model.planets[0].Rp_Rearth[0] - self.observed) / 0.3) ** 2

    def draw(self, key, model):
        return self


def test_a_caller_defined_term_adds_only_its_own_value():
    """Fails if a caller-defined term gains a hidden factor or two terms share data.

    At three radii the density of a fit with one caller-defined term is its prior
    plus that term's value at the draw's model, to 1e-12. Two fits whose terms hold
    different observations each get their own term's value, and their densities
    differ by more than 1e-6, so a term's data never leaks into another fit.
    """
    x = {"planets[0].Rp_Rearth": jnp.array([[1.3], [2.2], [3.1]])}
    densities = []
    for observed in (1.0, 2.5):
        term = _RadiusReading(jnp.asarray(observed))
        fit = _simple_fit(terms=(term,))
        own = jax.vmap(lambda xi, fit=fit, term=term: term(fit.model_at(xi)))(x)
        density = fit.log_density(x)
        np.testing.assert_allclose(density, fit.log_prior(x) + own, rtol=0, atol=1e-12)
        densities.append(density)
    assert np.min(np.abs(densities[0] - densities[1])) > 1e-6


def test_refuses_without_x64():
    """Construction on the host refuses when float64 is off."""
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            Fit(_scene(), {})
    finally:
        jax.config.update("jax_enable_x64", True)


# ----- round trip, leaf laws, draws ---------------------------------------------


def _mixed_fit():
    return Fit(
        _scene(planet_mass_Msun=1e-4),
        {
            "planets[0].Rp_Rearth": TruncatedDistribution(Normal(2.0, 0.7), low=0.5),
            "planets[0].physical_model.Ag": Uniform(0.05, 0.9),
            "dynamics.frame.dist_pc": TruncatedDistribution(Normal(10.0, 3.0), low=2.0),
            "dynamics.orbits": _block(),
        },
        (_relative(), _stellar()),
    )


@pytest.mark.medium(reason="mixed fit compile, 7 s cold")
def test_real_round_trip_and_density():
    """to_real and from_real invert each other; real_log_prob adds the volume."""
    fit = _mixed_fit()
    x = fit.sample_prior(jax.random.key(0), (5,))
    u = fit.to_real(x)
    back, volume = fit.from_real(u)
    for key in x:
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12),
            x[key],
            back[key],
        )
    assert volume.shape == (5,)
    np.testing.assert_allclose(
        fit.real_log_prob(u), fit.log_density(x) + volume, rtol=1e-12
    )
    # a block entry is the identity map with zero volume
    assert all(
        np.array_equal(u["dynamics.orbits"][k], x["dynamics.orbits"][k]) for k in COORDS
    )
    # the unconstrained density integrates to the prior: importance check on a leaf
    one = jax.tree.map(lambda a: a[0], x)
    grad = jax.grad(lambda v: fit.real_log_prob(v))(fit.to_real(one))
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad))


@pytest.mark.parametrize(
    "law",
    [
        TruncatedDistribution(Normal(2.0, 0.7), low=0.5),
        Uniform(0.05, 0.9),
        LogUniform(0.5, 4.0),
        TruncatedDistribution(Normal(10.0, 3.0), low=2.0),
    ],
)
def test_leaf_laws_are_normalized(law):
    """Every leaf entry of the fits above passes the priorities distribution check."""
    check_distribution(law, jax.random.key(1), n_draws=20_000)


@pytest.mark.medium(reason="mixed fit batch compile, 6 s cold")
def test_density_is_normalized_prior_draws_agree():
    """Prior draws have finite log_prior and the batch shapes hold."""
    fit = _mixed_fit()
    x = fit.sample_prior(jax.random.key(3), (4, 2))
    assert x["planets[0].Rp_Rearth"].shape == (4, 2, 1)
    assert x["dynamics.orbits"][COORDS[0]].shape == (4, 2, 1)
    lp = fit.log_prior(x)
    assert lp.shape == (4, 2) and np.all(np.isfinite(lp))
    one = jax.tree.map(lambda a: a[1, 1], x)
    np.testing.assert_allclose(fit.log_prior(one), lp[1, 1], rtol=1e-13)
    model = fit.model_at(one)
    assert resolve(model, "planets[0].Rp_Rearth").shape == (1,)
    cloud = jax.vmap(jax.vmap(fit.model_at))(x)
    assert resolve(cloud, "dynamics.orbits").a_AU.shape == (4, 2, 1)
    off = fit.log_density({**one, "planets[0].Rp_Rearth": jnp.array([0.1])})
    assert off == -jnp.inf


def test_prior_draws_do_not_depend_on_the_insertion_order_of_free():
    """A pytree round trip sorts the free keys; the draws must not change."""
    free = {
        "planets[0].physical_model.Ag": Uniform(0.05, 0.9),
        "dynamics.frame.dist_pc": Uniform(5.0, 15.0),
        "planets[0].Rp_Rearth": LogUniform(0.5, 4.0),
    }
    fit = Fit(_scene(), free)
    leaves, treedef = jax.tree.flatten(fit)
    crossed = jax.tree.unflatten(treedef, leaves)
    assert list(crossed.free) != list(fit.free)
    key = jax.random.key(11)
    for name, value in fit.sample_prior(key, (3,)).items():
        np.testing.assert_array_equal(value, crossed.sample_prior(key, (3,))[name])


def test_from_real_volume_has_the_batch_shape_for_a_block_only_fit():
    """A fit with only block entries still returns one volume per draw."""
    fit = Fit(_scene(), {"dynamics.orbits": _block()})
    u = fit.sample_prior(jax.random.key(0), (3,))
    _, volume = fit.from_real(fit.to_real(u))
    assert volume.shape == (3,)


def test_the_block_chain_is_traced_once_per_density():
    """log_density runs each block's forward once per density."""
    calls = []

    class Counting(OrbitBlock):
        def forward(self, x, model):
            calls.append(1)
            return super().forward(x, model)

    block = Counting(
        ClassicalChart(),
        OrbitLaw(
            period_d=LogUniform(*PERIOD_RANGE), eccentricity=kipping_eccentricity()
        ),
        rows=(0,),
        system="dynamics",
    )
    fit = Fit(_scene(), {"dynamics.orbits": block}, (_relative(),))
    x = fit.sample_prior(jax.random.key(0))
    calls.clear()
    fit.log_density(x)
    assert len(calls) == 1


# ----- an orbit fit through a block on a real Scene ------------------------------


def test_orbit_fit_recovers_the_truth_on_a_scene():
    """The density peaks near the generating orbit and sample_prior feeds it."""
    scene = _scene(planet_mass_Msun=0.0)
    truth = scene.dynamics.orbits
    times = jnp.array(TIMES)
    east, north = scene.positions(times)[:, 0, jnp.arange(4)]
    data = AstrometryData.pad(
        time_d=times,
        east_arcsec=east,
        north_arcsec=north,
        east_err_arcsec=jnp.full(4, 1e-4),
        north_err_arcsec=jnp.full(4, 1e-4),
        corr=jnp.zeros(4),
        planet=jnp.zeros(4, dtype=int),
        max_n=4,
    )
    block = _block()
    term = RelativeAstrometryTerm(data)
    fit = Fit(scene, {"dynamics.orbits": block}, (term,))
    x_true = {"dynamics.orbits": block.inverse(scene)}
    x_other = {"dynamics.orbits": {k: v + 0.3 for k, v in block.inverse(scene).items()}}
    assert fit.log_likelihood(x_true) > fit.log_likelihood(x_other) + 100.0
    built = fit.model_at(x_true).dynamics.orbits
    np.testing.assert_allclose(built.a_AU, truth.a_AU, rtol=1e-10)
    np.testing.assert_allclose(
        term.predict(fit.model_at(x_true)), [east, north][:], atol=1e-12
    )
    draws = fit.sample_prior(jax.random.key(7), (8,))
    assert np.all(np.isfinite(fit.log_density(draws)))
    assert np.all(fit.log_density(draws) < fit.log_density(x_true))


# ----- an image fit over an ExposureModel -----------------------------------------


def test_exposure_fit_over_an_exposure_model():
    """An image fit frees instrument and scene leaves of one ExposureModel."""
    from photomancy.exposure import ExposureModel, ExposureTerm
    from photomancy.infer.likelihoods import laplace_loglike

    class Render(eqx.Module):
        def __call__(self, instrument, exposure, scene):
            return instrument * scene * jnp.ones((3, 3)) + exposure

    model = ExposureModel(
        instrument=jnp.array(1.0),
        exposures=(jnp.array(0.1),),
        scene=jnp.array(2.0),
    )
    data = 2.1 * jnp.ones((3, 3))
    term = ExposureTerm(Render(), laplace_loglike(data, 1.0), 0)
    fit = Fit(
        model,
        {
            "instrument": LogUniform(0.5, 2.0),
            "exposures[0]": Uniform(-1.0, 1.0),
        },
        (term,),
    )
    x = {"instrument": jnp.array(1.0), "exposures[0]": jnp.array(0.1)}
    assert np.isfinite(fit.log_density(x))
    assert fit.log_likelihood(x) > fit.log_likelihood(
        {**x, "instrument": jnp.array(1.6)}
    )
    grad = jax.grad(lambda v: fit.real_log_prob(v))(fit.to_real(x))
    assert all(np.isfinite(g) for g in jax.tree.leaves(grad))
    assert fit.model_at(x).exposures[0] == 0.1
    with pytest.raises(ValueError, match="scene"):
        Fit(model, {"scene.nothing": LogUniform(0.5, 2.0)}, (term,))


# ----- block and prior-swap behaviour on a Scene ---------------------------------


def _two_planet_scene():
    def planet():
        return Planet(jnp.ones(1), LambertianPhysicalModel(jnp.array([0.3])))

    return Scene.from_classical(
        FlatStar(1.0),
        (planet(), planet()),
        star_mass_Msun=1.1,
        planet_mass_Msun=jnp.array([1e3, 3e2]) * Mearth2kg / Msun2kg,
        dist_pc=10.0,
        a_AU=jnp.array([1.0, 3.0]),
        e=jnp.array([0.1, 0.2]),
        i_rad=jnp.array([0.7, 0.9]),
        W_rad=jnp.array([0.5, 1.5]),
        w_rad=jnp.array([1.0, 2.0]),
        M0_rad=jnp.array([2.0, 0.3]),
    )


def test_thiele_innes_fit_exposes_the_written_star_mass_in_model_at():
    """The star mass a Thiele-Innes block writes is the enclosed mass minus planets.

    Truth source: ``n^2 a^3 / GM`` of the built row, minus the planet mass, computed
    here from the model's own orbit.
    Fails if: ``model_at`` drops the block's second write, or applies it before the
    orbit it implies.
    """
    from orbix.system import ThieleInnesBlock

    scene = _two_planet_scene()
    block = ThieleInnesBlock(
        OrbitLaw(period_d=LogUniform(*PERIOD_RANGE), eccentricity=Uniform(0.05, 0.8)),
        Uniform(0.9, 1.3),
        rows=(0,),
        system="dynamics",
    )
    fit = Fit(scene, {"dynamics.orbits": block})
    x = fit.sample_prior(jax.random.key(3))
    model = fit.model_at(x)
    star_mass = float(model.dynamics.star.mass_Msun)
    assert star_mass != float(scene.dynamics.star.mass_Msun)
    assert 0.9 <= star_mass <= 1.3
    orbit = model.dynamics.orbits
    gm = float(orbit.n_rad_per_d[0] ** 2 * orbit.a_AU[0] ** 3)
    inner = float(model.dynamics.planets.mass_Msun[0])
    np.testing.assert_allclose(gm / GM_SUN_AU3_PER_D2 - inner, star_mass, rtol=1e-10)
    held_gm = float(orbit.n_rad_per_d[1] ** 2 * orbit.a_AU[1] ** 3)
    total = star_mass + float(jnp.sum(model.dynamics.planets.mass_Msun))
    np.testing.assert_allclose(held_gm / GM_SUN_AU3_PER_D2, total, rtol=1e-10)


def test_a_free_star_mass_rates_the_held_row_and_the_freed_row():
    """A freed star mass moves the mean motion of the held row, not its geometry.

    Truth source: ``orbix.equations.mean_motion`` at ``enclosed_mu`` of the drawn star
    mass and the planet masses.
    Fails if: a held row keeps the mean motion of the base star mass.
    """
    from orbix.equations import mean_motion
    from orbix.system import enclosed_mu

    scene = _two_planet_scene()
    free = {
        "dynamics.star.mass_Msun": Uniform(0.7, 1.6),
        "dynamics.orbits": OrbitBlock(
            ClassicalChart(),
            OrbitLaw(Uniform(0.05, 0.8), a_AU=LogUniform(0.8, 3.0)),
            rows=(0,),
            system="dynamics",
        ),
    }
    fit = Fit(scene, free)
    x = fit.sample_prior(jax.random.key(5))
    model = fit.model_at(x)
    mass = model.dynamics.star.mass_Msun
    assert float(mass) != float(scene.dynamics.star.mass_Msun)
    mu = enclosed_mu(mass, model.dynamics.planets.mass_Msun)
    orbit = model.dynamics.orbits
    np.testing.assert_allclose(
        orbit.n_rad_per_d, mean_motion(orbit.a_AU, mu), rtol=1e-12
    )
    held = scene.dynamics.orbits
    np.testing.assert_array_equal(orbit.a_AU[1], held.a_AU[1])
    assert float(orbit.n_rad_per_d[1]) != float(held.n_rad_per_d[1])


def _mean_ratio(old_fit, new_fit, n=40000):
    """Mean and standard error of ``pi_new / pi_old`` over draws of ``old_fit``."""
    x = old_fit.sample_prior(jax.random.key(11), (n,))
    ratio = np.exp(np.asarray(new_fit.log_prior(x) - old_fit.log_prior(x)))
    return ratio.mean(), ratio.std() / np.sqrt(n)


@pytest.mark.parametrize("kind", ["leaf", "orbit"])
def test_a_prior_swap_ratio_averages_to_one_under_the_old_prior(kind):
    """``E_old[pi_new / pi_old] = 1`` when the new support lies inside the old.

    The orbit case swaps the eccentricity law at fixed bounds (a bound change is a
    different map and is refused).

    Truth source: the identity itself, for normalized densities; the tolerance is four
    Monte Carlo standard errors of the sample mean (seeded draws).
    Fails if: a leaf law or an orbit block's density is not normalized, or the swap
    changes a constant the ratio should cancel.
    """
    if kind == "leaf":
        fit = Fit(_scene(), {"planets[0].Rp_Rearth": LogUniform(0.5, 4.0)})
        new = fit.with_priors({"planets[0].Rp_Rearth": LogUniform(1.0, 2.0)})
    else:
        flat = OrbitLaw(
            period_d=LogUniform(*PERIOD_RANGE), eccentricity=Uniform(0.0, 1.0)
        )
        fit = Fit(_scene(), {"dynamics.orbits": _block(flat)})
        new = fit.with_priors({"dynamics.orbits": _block()})
    mean, se = _mean_ratio(fit, new)
    assert abs(mean - 1.0) < 4.0 * se


def _kepler_residual(model):
    """``|n^2 a^3 / (G M_enclosed) - 1|`` per row of the model's orbits."""
    from orbix.system import enclosed_mu

    system = model.dynamics if hasattr(model, "dynamics") else model
    mu = enclosed_mu(system.star.mass_Msun, system.planets.mass_Msun)
    orbit = system.orbits
    return jnp.abs(orbit.n_rad_per_d**2 * orbit.a_AU**3 / mu - 1.0)


def _orbit_leaf_laws(system):
    """A leaf law on each tied field of the stored orbits, shaped like the leaf."""
    orbit = system.orbits
    return {
        "a_AU": LogUniform(jnp.ones_like(orbit.a_AU), 100 * jnp.ones_like(orbit.a_AU)),
        "ex": Normal(jnp.zeros_like(orbit.ex), 2 * jnp.ones_like(orbit.ex)),
        "n_rad_per_d": LogUniform(
            1e-3 * jnp.ones_like(orbit.a_AU), 1e-1 * jnp.ones_like(orbit.a_AU)
        ),
        "quat": Normal(jnp.zeros_like(orbit.quat), jnp.ones_like(orbit.quat)),
    }


@pytest.mark.parametrize("prefix", ["dynamics", ""], ids=["scene", "root-system"])
def test_refuses_a_leaf_law_inside_a_stored_orbit_or_a_lone_mass(prefix):
    """Fails if a fit can return orbits that break Kepler's third law.

    A stored orbit's fields are tied to each other and to the masses. Before the
    refusal, a log-uniform law on ``a_AU`` with no block gave ``n^2 a^3`` of 7.9e-4 to
    23 AU^3/d^2 against the system's 2.96e-4 (``n^2 a^3 / (G M)`` from 9.0 to 2.6e5 on
    a root System), a Normal(0, 2) on ``ex`` gave ``ex`` of 2.6, and a log-normal star
    mass with no orbit block gave ``n^2 a^3 / (G M)`` from 0.77 to 1.22. Each leaf
    inside the orbits and each mass leaf alone is refused, with the System under a
    Scene and as the model itself. Truth: definitional.
    """
    scene = _two_planet_scene()
    model = scene if prefix else scene.dynamics
    system = scene.dynamics
    path = lambda p: f"{prefix}.{p}" if prefix else p  # noqa: E731
    for name, law in _orbit_leaf_laws(system).items():
        with pytest.raises(ValueError, match=r"inside .*orbits.*block"):
            Fit(model, {path(f"orbits.{name}"): law})
    planets = system.planets.mass_Msun
    for key, law in (
        ("star.mass_Msun", Uniform(0.7, 1.6)),
        ("planets.mass_Msun", Uniform(0.5 * planets, 1.5 * planets)),
    ):
        with pytest.raises(ValueError, match=r"source of .*orbits.*block"):
            Fit(model, {path(key): law})


@pytest.mark.parametrize("prefix", ["dynamics", ""], ids=["scene", "root-system"])
def test_a_free_mass_with_an_orbit_block_keeps_every_row_keplerian(prefix):
    """Fails if a mass leaf beside an orbit block leaves a row off Kepler's third law.

    The star mass and the planet masses, each free beside an ``OrbitBlock`` that frees
    row 0 and holds row 1, at 100 ``sample_prior`` draws: ``|n^2 a^3 / (G M) - 1|`` at
    most 1e-12 on both rows, with ``M`` the enclosed mass of each row (measured
    4.4e-16 for a free star mass). A ``ThieleInnesBlock`` alone writes the orbits and
    the star mass, and is accepted. Truth: Kepler's third law. Tolerance basis:
    ``exact-algebra``.
    """
    from orbix.system import ThieleInnesBlock

    scene = _two_planet_scene()
    model = scene if prefix else scene.dynamics
    path = lambda p: f"{prefix}.{p}" if prefix else p  # noqa: E731
    block = OrbitBlock(
        ClassicalChart(),
        OrbitLaw(Uniform(0.05, 0.8), a_AU=LogUniform(0.8, 3.0)),
        rows=(0,),
        system=prefix,
    )
    planets = scene.dynamics.planets.mass_Msun
    for key, law in (
        ("star.mass_Msun", Uniform(0.7, 1.6)),
        ("planets.mass_Msun", Uniform(0.5 * planets, 1.5 * planets)),
    ):
        fit = Fit(model, {path(key): law, path("orbits"): block})
        x = fit.sample_prior(jax.random.key(7), (100,))
        residual = jax.vmap(lambda one, f=fit: _kepler_residual(f.model_at(one)))(x)
        assert float(jnp.max(residual)) <= 1e-12, key
    ti = ThieleInnesBlock(block.law, Uniform(0.9, 1.3), rows=(0,), system=prefix)
    Fit(model, {path("orbits"): ti})


def _walled_fits():
    """The prior-recovery orbit sets: Cartesian, Thiele-Innes and the classical chart.

    Log-uniform a on 1 to 100 AU and uniform e on 0 to 0.95; the Thiele-Innes block
    adds a uniform star mass on 0.8 to 1.2.
    """
    from orbix.system import CartesianChart, ThieleInnesBlock

    law = OrbitLaw(Uniform(0.0, 0.95), a_AU=LogUniform(1.0, 100.0))
    scene = _scene()
    orbit = lambda chart: OrbitBlock(chart, law, rows=(0,), system="dynamics")  # noqa: E731
    ti = ThieleInnesBlock(law, Uniform(0.8, 1.2), rows=(0,), system="dynamics")
    return {
        "cartesian": (
            Fit(scene, {"dynamics.orbits": orbit(CartesianChart())}),
            "a_AU, e",
        ),
        "thiele-innes": (Fit(scene, {"dynamics.orbits": ti}), "a_AU, star_mass_Msun"),
        "classical": (Fit(scene, {"dynamics.orbits": orbit(ClassicalChart())}), None),
    }


def test_gradient_samplers_refuse_a_fit_with_walls():
    """Fails if NUTS or MCLMC samples a density with walls, or refuses one without.

    Against a wall every rejected step is flagged as a divergence (75 and 99.9 percent
    of transitions on these two sets), so the samplers refuse before sampling and name
    the entry and every wall. The same fits still draw from the prior and map the unit
    cube for nested sampling, and the classical fit, which maps both bounds, is not
    refused. Truth: definitional. No sampling.
    """
    from photomancy.infer import MCLMC, NUTS, run
    from photomancy.infer.backends._flat import require_no_walls

    for name, (fit, walls) in _walled_fits().items():
        if walls is None:
            assert fit.walls == {}
            require_no_walls(fit, "NUTS")
            continue
        assert fit.walls == {"dynamics.orbits": tuple(walls.split(", "))}, name
        for backend in (NUTS(n_warmup=10, n_draws=10), MCLMC(n_warmup=10, n_draws=10)):
            label = type(backend).__name__
            with pytest.raises(
                ValueError, match=rf"{label} needs .*'dynamics.orbits'.*{walls}"
            ):
                run(fit, backend, jax.random.key(0))
        x = fit.sample_prior(jax.random.key(1), (4,))
        assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(x))
        unit = jax.tree.map(lambda a: a + 0.5, fit.unit_template())
        assert all(
            bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(fit.from_unit(unit))
        )
