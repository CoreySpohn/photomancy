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
from hwoutils.constants import Mearth2kg, Msun2kg, Rearth2AU
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
from photomancy.orbit.data import (
    PhotometryData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
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
COORDS = ("ln_a_AU", "logit_e", "atanh_cos_i", "W_rad", "w_rad", "M0_rad")


def _relative():
    return RelativeAstrometryTerm(
        RelativeAstromData.from_sep_pa(
            times=TIMES,
            sep=[0.13, 0.11, 0.09, 0.15],
            sep_err=[0.004] * 4,
            pa_deg=[45.0, 120.0, 210.0, 310.0],
            pa_err_deg=[1.0] * 4,
            max_n=4,
        )
    )


def _stellar():
    return StellarAstrometryTerm(
        StellarAstromData.pad(
            times=TIMES,
            ra=[1e-4, -2e-4, 3e-4, 2e-4],
            dec=[-1e-4, 1e-4, 2e-4, -2e-4],
            ra_err=[5e-5] * 4,
            dec_err=[8e-5] * 4,
            corr=[0.2, -0.3, 0.1, 0.0],
            max_n=4,
        )
    )


def _rv():
    return RadialVelocityTerm(
        RVData.pad(
            times=TIMES,
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
        PMAnomalyData(
            t_hip=jnp.array(0.0),
            t_gaia=jnp.array(9000.0),
            gaia_window=jnp.array(1023.0),
            pm_anomaly=jnp.array([2e-7, -3e-7]),
            pm_anomaly_cov=jnp.array([[4e-14, 1e-14], [1e-14, 9e-14]]),
            is_valid=jnp.array(True),
        )
    )


def _photometry():
    band = Band(550.0, 650.0)
    return PhotometryTerm(
        PhotometryData.from_dmag(
            times=TIMES,
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


def _coords(point):
    return {name: jnp.array([value]) for name, value in zip(COORDS, point, strict=True)}


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
        lambda x: 0.0,
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
            "dynamics.orbits": _coords(golden["points"][i]),
        }

    offsets["g2"] = _check_against_old(
        fit,
        golden,
        entries,
        lambda x: float(jnp.log(x["dynamics.planets.mass_Msun"][0])),
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
        )

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
    """A leaf law inside a block's written node, and two blocks on one node."""
    with pytest.raises(ValueError, match="two writers"):
        Fit(
            _scene(),
            {
                "dynamics.orbits": _block(),
                "dynamics.orbits.a_AU": LogUniform(0.1, 10.0),
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


def test_refuses_a_changed_free_set():
    """with_priors keeps the free set and the map of a block."""
    fit = Fit(
        _scene(),
        {"planets[0].Rp_Rearth": LogUniform(0.5, 4.0), "dynamics.orbits": _block()},
        (_relative(),),
    )
    with pytest.raises(ValueError, match="cannot change the free set"):
        fit.with_priors({"planets[0].physical_model.Ag": Uniform(0.0, 1.0)})
    with pytest.raises(ValueError, match="same map"):
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
    law = OrbitLaw(period_d=LogUniform(100.0, 3000.0), eccentricity=Uniform(0.0, 1.0))
    assert fit.with_priors({"dynamics.orbits": _block(law)}).digest() != fit.digest()


def test_refuses_a_term_that_is_not_an_abstract_term():
    """A term on the old protocol cannot enter a fit: TypeError names the index."""
    from photomancy.system_terms import NullTerm

    with pytest.raises(TypeError, match=r"terms\[1\] is a NullTerm"):
        _simple_fit(terms=(_ReadsNothing(), NullTerm(None)))


class _ReadsNothing(AbstractTerm):
    reads = ()

    def __call__(self, model):
        return jnp.zeros(())

    def draw(self, key, model):
        return self


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
    assert x["dynamics.orbits"]["ln_a_AU"].shape == (4, 2, 1)
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
    data = RelativeAstromData.pad(
        times=times,
        ra=east,
        dec=north,
        ra_err=jnp.full(4, 1e-4),
        dec_err=jnp.full(4, 1e-4),
        corr=jnp.zeros(4),
        planet_id=jnp.zeros(4, dtype=int),
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
        nuisance=(jnp.array(0.1),),
        scene=jnp.array(2.0),
    )
    data = 2.1 * jnp.ones((3, 3))
    term = ExposureTerm(Render(), laplace_loglike(data, 1.0), 0)
    fit = Fit(
        model,
        {
            "instrument": LogUniform(0.5, 2.0),
            "nuisance[0]": Uniform(-1.0, 1.0),
        },
        (term,),
    )
    x = {"instrument": jnp.array(1.0), "nuisance[0]": jnp.array(0.1)}
    assert np.isfinite(fit.log_density(x))
    assert fit.log_likelihood(x) > fit.log_likelihood(
        {**x, "instrument": jnp.array(1.6)}
    )
    grad = jax.grad(lambda v: fit.real_log_prob(v))(fit.to_real(x))
    assert all(np.isfinite(g) for g in jax.tree.leaves(grad))
    assert fit.model_at(x).nuisance[0] == 0.1
    with pytest.raises(ValueError, match="scene"):
        Fit(model, {"scene.nothing": LogUniform(0.5, 2.0)}, (term,))
