"""Fit: density against the old system density, refusals, round trip, orbit block.

The golden file holds ``SystemLogDensity`` values from the base commit (photomancy
stack-v3 with its orbix) at chart coordinates of three systems. The new ``Fit`` writes
the same orbit through an ``OrbitBlock`` on a real ``Scene``, so the two densities agree
up to named changes of coordinates (the offset is zero to rounding).
"""

import enum
import json
from pathlib import Path
from typing import ClassVar

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.blocks import AbstractBlock
from hwoutils.constants import GM_SUN_AU3_PER_D2, Mearth2kg, Msun2kg, Rearth2AU
from hwoutils.model_paths import resolve
from hwoutils.precision import PrecisionError
from hwoutils.testing import ulps
from optixstuff.spectral import Band, Quadrature, Transmission
from orbix.system import ClassicalChart, OrbitBlock, OrbitLaw
from priorities import LogUniform, Normal, TruncatedDistribution, Uniform
from priorities.testing import check_distribution
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.population import kipping_eccentricity
from skyscapes.scene import FlatStar, Planet, Scene
from term_utils import NamedElement, named_model

from photomancy import (
    PhotometryTerm,
    ProperMotionAnomalyTerm,
    RadialVelocityTerm,
    RelativeAstrometryTerm,
    StellarAstrometryTerm,
)
from photomancy.detection import detect, measure
from photomancy.infer import MAP, AbstractTerm, Fit, ForwardTerm, Laplace, run
from photomancy.infer.likelihoods import gaussian_loglike
from photomancy.measurements import (
    AstrometryData,
    PhotometryData,
    ProperMotionAnomalyData,
    RadialVelocityData,
)

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
            instruments=("a", "b", "c"),
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_refuses_a_free_key_that_meets_term_reads():
    """A free path inside a term's read set is refused, naming path and term."""
    with pytest.raises(ValueError, match=r"Rp_Rearth.*_ReadsRadius"):
        _simple_fit(terms=(_ReadsRadius(),))
    # a free parent of the read path overlaps too
    with pytest.raises(ValueError, match="planets"):
        Fit(_scene(), {"planets[0]": LogUniform(0.5, 4.0)}, (_ReadsRadius(),))
    # no free key: the same term is fine
    Fit(_scene(), {}, (_ReadsRadius(),))


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_refuses_an_index_into_an_array_leaf():
    """``leaf[0]`` into an array is not a path: refused as an unknown path."""
    with pytest.raises(ValueError, match="unknown model path"):
        Fit(_scene(), {"planets[0].Rp_Rearth[0]": LogUniform(0.5, 4.0)})


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float32_only(reason="Fit must refuse a float32 process")
def test_refuses_without_x64():
    """Construction on the host refuses when float64 is off."""
    with pytest.raises(PrecisionError, match="enable_x64"):
        Fit(_scene(), {})


@pytest.fixture(scope="module")
def mixed_draws():
    """One mixed leaf/orbit fit and draws with two distinct batch extents."""
    fit = Fit(
        _scene(planet_mass_Msun=1e-4),
        {
            "planets[0].Rp_Rearth": TruncatedDistribution(Normal(2.0, 0.7), low=0.5),
            "planets[0].physical_model.Ag": Uniform(0.05, 0.9),
            "dynamics.frame.dist_pc": TruncatedDistribution(Normal(10.0, 3.0), low=2.0),
            "dynamics.orbits": _block(),
        },
        (_relative(), _stellar()),
    )
    draws = eqx.filter_jit(fit.sample_prior)(jax.random.key(0), (2, 3))
    return fit, draws


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(
    reason="traces the real-coordinate density and gradient on a mixed orbit and "
    "leaf fit, 21 s projected on CI"
)
def test_real_round_trip_and_density(mixed_draws):
    """to_real and from_real invert each other; real_log_prob adds the volume."""
    fit, x = mixed_draws
    u = fit.to_real(x)
    back, volume = fit.from_real(u)
    for key in x:
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12),
            x[key],
            back[key],
        )
    assert volume.shape == (2, 3)
    np.testing.assert_allclose(
        fit.real_log_prob(u), fit.log_density(x) + volume, rtol=1e-12
    )
    # a block entry is the identity map with zero volume
    assert all(
        np.array_equal(u["dynamics.orbits"][k], x["dynamics.orbits"][k]) for k in COORDS
    )
    # the unconstrained density integrates to the prior: importance check on a leaf
    one = jax.tree.map(lambda a: a[0, 0], x)
    grad = jax.grad(lambda v: fit.real_log_prob(v))(fit.to_real(one))
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad))


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.parametrize(
    "law",
    [
        pytest.param(
            TruncatedDistribution(Normal(2.0, 0.7), low=0.5),
            marks=pytest.mark.slow(
                reason="checks distribution normalization and tail round trips with "
                "20000 prior draws, 8 s projected on CI"
            ),
        ),
        Uniform(0.05, 0.9),
        LogUniform(0.5, 4.0),
        TruncatedDistribution(Normal(10.0, 3.0), low=2.0),
    ],
)
def test_leaf_laws_are_normalized(law):
    """Every leaf entry of the fits above passes the priorities distribution check.

    A truncated law's unit round trip is checked against its own floating-point
    bound instead of the generic relative tolerance (see
    ``_cut_round_trip_excess``); every other check is the generic one.
    """
    truncated = isinstance(law, TruncatedDistribution)
    check_distribution(
        law,
        jax.random.key(1),
        n_draws=20_000,
        unit_rtol=np.inf if truncated else 1e-11,
    )
    if truncated:
        assert float(jnp.max(_cut_round_trip_excess(law))) <= 0.0


def _cut_round_trip_excess(law, ulps=4.0, cut_ulps=32.0):
    """Error of ``cdf(icdf(u))`` beyond its floating-point bound, on the tail grid.

    Tolerance basis: floating-point. The grid and the ``ulps`` allowance for the
    representation of ``u`` and of the value are those of
    ``priorities.testing.icdf_round_trip``. A lower cut adds a term that the
    generic check divides by ``u`` and so cannot absorb: the quantile evaluates
    the base inverse at ``F(low) + u * mass`` and the distribution function
    subtracts ``F(low)`` again, so the round trip carries the rounding of that sum
    and the base ``ndtr``/``ndtri`` errors near the cut, each a few units in the
    last place of ``F(low)``, divided by ``mass``. Relative to a ``u`` of 1e-13
    that is 1e-5 for the cut at 2 of ``Normal(10, 3)``, and its size changes with
    the platform's math library (on an x86-64 Linux CPU the error exceeds the
    generic floor by 1.4 and 3.0 units of ``eps * F(low) / mass`` for the two
    laws here). ``cut_ulps`` allows 32 such units. A quantile whose cut is off by
    1e-9 relative still exceeds this bound by 8e-12 or more.
    """
    eps = jnp.finfo(jnp.zeros(()).dtype).eps
    lower = jnp.logspace(-30.0, jnp.log10(0.5), 61)
    upper = 1 - jnp.logspace(jnp.log10(1e4 * eps), jnp.log10(0.5), 25)
    unit = jnp.concatenate([lower, upper])
    value = law.icdf(unit)
    below = law.base_dist.cdf(law.low)
    mass = 1.0 - below if law.high is None else law.base_dist.cdf(law.high) - below
    floor = ulps * eps * (unit + jnp.abs(value) * jnp.exp(law.log_prob(value)))
    bound = floor + cut_ulps * eps * below / mass
    return jnp.abs(law.cdf(value) - unit) - bound


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.medium(reason="mixed fit batch compile, 6 s cold")
def test_density_is_normalized_prior_draws_agree(mixed_draws):
    """Prior draws have finite log_prior and the batch shapes hold."""
    fit, x = mixed_draws
    assert x["planets[0].Rp_Rearth"].shape == (2, 3, 1)
    assert x["dynamics.orbits"][COORDS[0]].shape == (2, 3, 1)
    lp = fit.log_prior(x)
    assert lp.shape == (2, 3) and np.all(np.isfinite(lp))
    one = jax.tree.map(lambda a: a[1, 1], x)
    np.testing.assert_allclose(fit.log_prior(one), lp[1, 1], rtol=1e-13)
    model = fit.model_at(one)
    assert resolve(model, "planets[0].Rp_Rearth").shape == (1,)
    cloud = jax.vmap(jax.vmap(fit.model_at))(x)
    assert resolve(cloud, "dynamics.orbits").a_AU.shape == (2, 3, 1)
    off = fit.log_density({**one, "planets[0].Rp_Rearth": jnp.array([0.1])})
    assert off == -jnp.inf


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(
    reason="compiles prior draws and coordinate volume on a periodic orbit, 6 s "
    "projected on CI"
)
def test_from_real_volume_has_the_batch_shape_for_a_block_only_fit():
    """A fit with only block entries still returns one volume per draw."""
    fit = Fit(_scene(), {"dynamics.orbits": _block()})
    u = fit.sample_prior(jax.random.key(0), (3,))
    _, volume = fit.from_real(fit.to_real(u))
    assert volume.shape == (3,)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(
    reason="compiles and counts the block chain across several density entry "
    "points, 5 s projected on CI"
)
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(
    reason="runs orbit inference over the astrometric scene and compares the "
    "recovered physical coordinates, 12 s projected on CI"
)
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.parametrize(
    "kind",
    [
        "leaf",
        pytest.param(
            "orbit",
            marks=pytest.mark.slow(
                reason="runs 40000 prior draws and checks the density ratio "
                "normalization, 6 s projected on CI"
            ),
        ),
    ],
)
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.parametrize("prefix", ["dynamics", ""], ids=["scene", "root-system"])
@pytest.mark.slow(
    reason="compiles joint mass and orbit writes on a two-planet model, "
    "8 s per case projected on CI"
)
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


@pytest.fixture(scope="module")
def walled_fits():
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_gradient_samplers_refuse_a_fit_with_walls(walled_fits):
    """Fails if NUTS or MCLMC samples a density with walls, or refuses one without.

    Against a wall every rejected step is flagged as a divergence (75 and 99.9 percent
    of transitions on these two sets), so the samplers refuse before sampling and name
    the entry and every wall. The same fits still draw from the prior and map the unit
    cube for nested sampling, and the classical fit, which maps both bounds, is not
    refused. Truth: definitional. No sampling.
    """
    from photomancy.infer import MCLMC, NUTS, run
    from photomancy.infer.backends._flat import require_no_walls

    for name, (fit, walls) in walled_fits.items():
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


def _stack_two(fit):
    """Two copies of ``fit`` stacked along a new leading world axis."""
    arrays, static = eqx.partition(fit, eqx.is_array)
    return eqx.combine(jax.tree.map(lambda a: jnp.stack([a, a]), arrays), static)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_walls_are_evaluated_once_and_survive_stacking_jit_and_edits(walled_fits):
    """Fails if a stacked, jitted or edited fit reports walls its source did not have.

    Inside ``filter_jit`` or ``filter_vmap`` every law parameter is traced, and a block
    asked for its walls there counts a traced unmapped edge as a wall. The fit
    evaluates the walls when it is built on the host, so a copy under either
    transformation, and an edit made under ``filter_vmap``, reports the eager walls.
    Truth: the eager ``Fit.walls`` of each table fit and of a frame-chart fit (none).
    """
    from orbix.system import FrameChart

    fits = {k: f for k, (f, _) in walled_fits.items()}
    fits["frame"] = Fit(_scene(), {"dynamics.orbits": _block(chart=FrameChart())})
    assert fits["frame"].walls == {}
    for name, fit in fits.items():
        seen = {}

        @eqx.filter_jit
        def jitted(f, seen=seen):
            seen["jit"] = f.walls
            return jnp.zeros(())

        @eqx.filter_vmap
        def mapped(f, seen=seen):
            seen["vmap"] = f.walls
            seen["edit"] = f.append_term(_ReadsNothing()).walls
            return jnp.zeros(())

        jitted(fit)
        mapped(_stack_two(fit))
        assert seen == {"jit": fit.walls, "vmap": fit.walls, "edit": fit.walls}, name


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_walls_follow_a_law_edited_in_place_and_survive_replace(walled_fits):
    """Fails if an edit that skips ``__init__`` leaves ``Fit.walls`` stale.

    ``eqx.tree_at`` (and ``tree_deserialise_leaves``) rebuild a fit without
    ``__init__``. Raising the eccentricity law's lower edge from 0 to 0.2 on the frame
    chart, whose disk ignores the lower edge, adds a wall on ``e``; lowering it back
    removes it; swapping the classical block for the Cartesian one adds walls on
    ``a_AU`` and ``e``. NUTS follows: it refuses the walled edit before sampling.
    A dynamic fit argument and a captured edited fit are both refused.
    ``dataclasses.replace`` builds a fit with the same walls. Truth: the walls of a
    fit built afresh from the edited entries. No sampling.
    """
    import dataclasses

    from orbix.system import FrameChart

    from photomancy.infer import NUTS, run

    a_law = LogUniform(1.0, 100.0)
    frame = Fit(
        _scene(),
        {
            "dynamics.orbits": OrbitBlock(
                FrameChart(),
                OrbitLaw(Uniform(0.0, 0.95), a_AU=a_law),
                rows=(0,),
                system="dynamics",
            )
        },
    )
    assert frame.walls == {}
    law_e = lambda f: f.free["dynamics.orbits"].law.eccentricity  # noqa: E731
    raised = eqx.tree_at(law_e, frame, Uniform(0.2, 0.95))
    assert raised.walls == {"dynamics.orbits": ("e",)}
    assert raised.walls == Fit(raised.model, raised.free, raised.terms).walls
    nuts = NUTS(n_warmup=10, n_draws=10)
    key = jax.random.key(0)
    with pytest.raises(ValueError, match=r"NUTS needs .*'dynamics.orbits'.*e"):
        run(raised, nuts, key)
    with pytest.raises(ValueError, match=r"NUTS needs .*'dynamics.orbits'.*e"):
        eqx.filter_jit(lambda k: run(raised, nuts, k))(key)

    from photomancy.infer.backends._flat import require_no_walls

    def admit(fit, key):
        return require_no_walls(fit, "NUTS", key)

    compiled = eqx.filter_jit(admit)
    with pytest.raises(
        eqx.EquinoxRuntimeError, match=r"NUTS needs .*'dynamics.orbits'.*e"
    ):
        jax.block_until_ready(compiled(raised, key))
    lowered = eqx.tree_at(law_e, raised, Uniform(0.0, 0.95))
    assert lowered.walls == {}
    np.testing.assert_array_equal(
        jax.random.key_data(compiled(lowered, key)), jax.random.key_data(key)
    )
    fits = walled_fits
    classical, cartesian = fits["classical"][0], fits["cartesian"][0]
    swapped = eqx.tree_at(
        lambda f: f.free["dynamics.orbits"],
        classical,
        cartesian.free["dynamics.orbits"],
    )
    assert swapped.walls == {"dynamics.orbits": ("a_AU", "e")}
    for fit in (frame, raised, cartesian):
        replaced = dataclasses.replace(fit, terms=(_ReadsNothing(),))
        assert replaced.walls == fit.walls
        assert replaced.terms == (_ReadsNothing(),)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(reason="two-world NUTS compile, about 560 programs, 37 s cold")
def test_nuts_maps_over_a_stack_of_wall_free_fits_and_refuses_a_walled_stack(
    walled_fits,
):
    """Fails if a stack of wall-free fits is refused, or a stack of walled fits runs.

    ``eqx.filter_vmap(run)`` over two stacked worlds is the documented way to run a
    stack. A frame-chart fit has no walls and runs; the Cartesian table fit has walls
    on ``a_AU`` and ``e`` and is refused before sampling. Truth: definitional; the
    run is checked for finite draws only.
    """
    from orbix.system import FrameChart

    from photomancy.infer import NUTS, run

    nuts = NUTS(n_warmup=10, n_draws=10, n_chains=1)
    keys = jax.random.split(jax.random.key(0), 2)
    frame = Fit(_scene(), {"dynamics.orbits": _block(chart=FrameChart())})
    out = eqx.filter_vmap(lambda f, k: run(f, nuts, k))(_stack_two(frame), keys)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(out.samples))
    cartesian = walled_fits["cartesian"][0]
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match=r"NUTS needs .*'dynamics.orbits'.*a_AU, e",
    ):
        eqx.filter_jit(eqx.filter_vmap(lambda f, k: run(f, nuts, k)))(
            _stack_two(cartesian), keys
        ).log_weights.block_until_ready()


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_refuses_a_static_detector_field_as_static_structure():
    """Fails if the pixel scale is reported as a node to free through a block.

    The pixel scale and shape are static: the detector grid, every resampling operator
    onto it and every support mask are built from them on the host. The refusal says
    so instead of pointing at a block that cannot exist. Truth: definitional.
    """
    from optixstuff.detector import IdealDetector

    class _Instrument(eqx.Module):
        detector: IdealDetector

    class _Model(eqx.Module):
        instrument: _Instrument

    model = _Model(_Instrument(IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8))))
    with pytest.raises(ValueError, match="static field that fixes structure"):
        Fit(model, {"instrument.detector.pixel_scale_arcsec": Normal(0.01, 0.001)})
    with pytest.raises(ValueError, match="free the node through a block"):
        Fit(model, {"instrument.detector": Normal(0.01, 0.001)})


class _Leaves(eqx.Module):
    count: jax.Array
    value: jax.Array
    rate: float = 2.0
    pixel_scale: float = eqx.field(static=True, default=0.1)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_free_accepts_a_float_leaf_and_refuses_a_static_field_and_an_integer_leaf():
    """Freeing a static field, an integer leaf and a float leaf: refuse, refuse, accept.

    Truth source: definitional (the leaf predicate ``hwoutils.leaves.is_param``: a
    floating or complex array may be free; integer arrays and static fields fix
    structure). A non-static Python float is converted by ``Fit`` to a float array and
    is then free. Fails if the integer array is accepted or a float leaf, array or
    converted Python number, is refused.
    """
    model = _Leaves(
        count=jnp.asarray(3, jnp.int32), value=jnp.asarray(1.5, jnp.float64)
    )
    with pytest.raises(ValueError, match="pixel_scale"):
        Fit(model, {"pixel_scale": Normal(0.1, 0.01)})
    with pytest.raises(ValueError, match=r"count.*not a floating leaf"):
        Fit(model, {"count": Normal(3.0, 1.0)})
    assert "value" in Fit(model, {"value": Normal(1.5, 0.1)}).free
    assert "rate" in Fit(model, {"rate": Normal(2.0, 0.1)}).free


class _Curve(eqx.Module):
    table: object
    value: jax.Array
    rate: float
    weak: jax.Array


class _Holder(eqx.Module):
    curve: _Curve


def _holder(value, rate, weak):
    import interpax

    x = jnp.linspace(0.0, 1.0, 5)
    table = interpax.Interpolator1D(x, x**2, method="cubic")
    return table, _Holder(_Curve(table, value, rate, weak))


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_fit_converts_numpy_python_and_weak_leaves_to_strong_arrays():
    """The model a user builds is converted at the door; ``fit.model`` is the result.

    Truth source: definitional (``hwoutils.leaves.canonical``: NumPy scalars, Python
    numbers and weakly typed arrays become strong JAX arrays in the active dtype;
    ``hwoutils.leaves.check_tree`` lists the leaves that break the rule). Fails if
    ``Fit`` refuses the model or ``check_tree(fit.model)`` is not empty, if a leaf of
    ``fit.model`` is not a strong float64 array, or if the model passed in was changed.
    """
    from hwoutils.leaves import check_tree

    weak = jnp.asarray(3.0)
    assert weak.weak_type
    _, model = _holder(np.float64(2.0), 4.0, weak)
    fit = Fit(model, {"curve.value": Normal(2.0, 0.1), "curve.rate": Normal(4.0, 0.1)})
    assert check_tree(fit.model) == ()
    for name, want in (("value", 2.0), ("rate", 4.0), ("weak", 3.0)):
        leaf = getattr(fit.model.curve, name)
        assert isinstance(leaf, jax.Array) and not leaf.weak_type
        assert leaf.dtype == jnp.float64 and float(leaf) == want
    assert isinstance(model.curve.value, np.float64) and model.curve.rate == 4.0


class _Mode(enum.IntEnum):
    A = 0
    B = 1


class _Counted(eqx.Module):
    value: jax.Array
    n: int
    flag: bool
    rate: float
    mode: _Mode = _Mode.B


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_fit_keeps_python_int_and_bool_scalars_as_compile_time_constants():
    """A Python ``int`` or ``bool`` field survives ``Fit`` and drives a jitted model.

    Truth source: Equinox convention (under ``eqx.filter_jit`` a non-array leaf is a
    compile-time constant, so ``jnp.zeros(model.n)`` and ``if model.flag`` trace; an
    array would raise a concretization error), and ``hwoutils.leaves.canonical`` for
    the float field. Fails if ``n`` or ``flag`` leaves ``Fit`` as an array, the Python
    float is not a strong float64 array, the jitted ``jnp.zeros(model.n)`` raises,
    or freeing ``n`` or ``flag`` is accepted.
    """
    model = _Counted(jnp.asarray(1.0), 3, True, 2.0)
    fit = Fit(model, {"value": Normal(1.0, 0.1)})
    assert type(fit.model.n) is int and fit.model.n == 3
    assert type(fit.model.flag) is bool and fit.model.flag is True
    assert fit.model.mode is _Mode.B
    rate = fit.model.rate
    assert isinstance(rate, jax.Array) and rate.dtype == jnp.float64
    assert not rate.weak_type

    @eqx.filter_jit
    def backend(m):
        out = jnp.zeros(m.n) + m.value + m.rate
        out = out if m.flag else -out
        return out if m.mode == _Mode.B else 2.0 * out

    assert backend(fit.model).shape == (3,)
    for name in ("n", "flag"):
        with pytest.raises(ValueError, match=rf"{name}.*static field"):
            Fit(model, {name: Normal(1.0, 0.1)})


class _FloatWritingBlock(AbstractBlock):
    """Owns ``a`` and writes a float32 strong value and a weak Python float."""

    owns = "a"
    reads = ()
    writes = ("a", "c")
    coordinate_names = ("u",)
    unit_size = 1

    def forward(self, x, model):
        return {"a": jnp.asarray(x["u"], jnp.float32), "c": 2.5}

    def inverse(self, model):
        return {"u": model.a}

    def log_prob(self, x, model):
        return -0.5 * x["u"] ** 2

    def sample(self, key, model):
        return {"u": jax.random.normal(key)}

    def from_unit(self, c, model):
        return {"u": c[0] - 0.5}

    def same_layout(self, other):
        return isinstance(other, _FloatWritingBlock)

    def same_map(self, other):
        return self.same_layout(other)


class _DtypeProbeBlock(_FloatWritingBlock):
    """Owns ``b``, reads the written paths and records their dtypes when called."""

    owns = "b"
    reads = ("a", "c")
    writes = ("b",)
    seen = None

    def _record(self, model):
        self.seen.append(
            tuple((leaf.dtype, leaf.weak_type) for leaf in (model.a, model.c))
        )

    def forward(self, x, model):
        return {"b": x["u"]}

    def log_prob(self, x, model):
        self._record(model)
        return -0.5 * x["u"] ** 2

    def sample(self, key, model):
        self._record(model)
        return {"u": jax.random.normal(key)}

    def from_unit(self, c, model):
        self._record(model)
        return {"u": c[0] - 0.5}


class _Written(eqx.Module):
    a: jax.Array
    b: jax.Array
    c: jax.Array


class _Payload(eqx.Module):
    fields: dict
    label: str = eqx.field(static=True)


class _PayloadBlock(_FloatWritingBlock):
    writes = ("a",)

    def forward(self, x, model):
        fields = {"value": 2 * x["u"], "optional": 1.5, "added": 3.0, "count": 1.5}
        return {"a": _Payload(fields, "updated")}

    def inverse(self, model):
        return {"u": model.a.fields["value"] / 2}


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_block_node_replacement_matches_destination_by_path():
    """Updated nodes keep matched array widths and carry their own static metadata.

    For the stored value v = 2u, the score -v**2 / 2 has derivative -4u.
    At u = 1/4 every multiply, cast and derivative is exact at either width.
    """
    node = _Payload(
        {
            "value": jnp.zeros((), jnp.float32),
            "optional": None,
            "removed": 2.0,
            "count": 1,
        },
        "original",
    )
    fit = Fit(_Written(node, jnp.zeros(()), jnp.zeros(())), {"a": _PayloadBlock()})

    def write(u):
        return fit.model_at({"a": {"u": u}}).a

    u = jnp.asarray(0.25, jnp.float64)
    updated = write(u)
    assert updated.label == "updated"
    assert updated.fields.keys() == {"value", "optional", "added", "count"}
    assert updated.fields["value"].dtype == jnp.float32
    for name, expected in (("optional", 1.5), ("added", 3.0), ("count", 1.5)):
        leaf = updated.fields[name]
        assert leaf.dtype == jnp.float64 and not leaf.weak_type and leaf == expected
    gradient = jax.grad(lambda u: -0.5 * write(u).fields["value"] ** 2)(u)
    assert gradient.dtype == jnp.float64 and gradient == -1.0


@pytest.mark.float64_only(reason="curvature and MAP use float64 coordinates")
@pytest.mark.parametrize(
    "operation", ["observed", "fisher", "MAP", "detect", "measure", "Laplace"]
)
def test_curvature_checks_layer_installed_by_block(operation):
    """A block installs a narrow response only when a draw is written."""

    class InstallLayer(_PayloadBlock):
        layer: eqx.nn.Linear

        def forward(self, x, model):
            fields = {"value": x["u"], "optional": self.layer}
            return {"a": _Payload(fields, "installed")}

    def predict(model):
        return model.a.fields["optional"](jnp.reshape(model.b, (1,)))

    node = _Payload({"value": jnp.float64(0), "optional": None}, "empty")
    model = _Written(node, jnp.float64(0), jnp.float64(0))
    key = jax.random.key(0)
    block = InstallLayer(eqx.nn.Linear(1, 1, dtype=jnp.float32, key=key))
    term = ForwardTerm(eqx.Partial(predict), gaussian_loglike(jnp.zeros(1), 1.0))
    fit = Fit(model, {"a": block, "b": Normal(0.0, 1.0)}, (term,))
    start = {"a": {"u": jnp.float64(0)}, "b": jnp.float64(0)}
    path = r"model\.a\.fields\['optional'\]\.weight"
    if operation == "MAP":
        with pytest.warns(UserWarning, match=rf"unscaled.*{path}"):
            post = run(fit, MAP(n_steps=1, init=start), key)
        assert "model.a.fields['optional'].weight" in post.info.scale_fallback_reason
        assert np.isfinite(post.log_target).all()
        return
    name = "Fit.information" if operation in ("observed", "fisher") else operation
    with pytest.raises(PrecisionError, match=rf"{name}.*{path}"):
        if operation == "detect":
            detect(fit, {"a": {"u": jnp.zeros(1)}}, amplitude="b")
        elif operation == "measure":
            measure(fit, start)
        elif operation == "Laplace":
            run(fit, Laplace(init=start), key)
        else:
            fit.information(start, kind=operation)


class _LayerBlock(_FloatWritingBlock):
    writes = ("a",)

    def forward(self, x, model):
        weight = jnp.reshape(x["u"], (1, 1))
        return {"a": eqx.tree_at(lambda layer: layer.weight, model.a, weight)}

    def inverse(self, model):
        return {"u": model.a.weight[0, 0]}


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_block_node_replacement_preserves_foreign_layer_weight_dtype():
    """A whole-layer write assigns the sampled weight at its destination width."""
    layer = eqx.nn.Linear(1, 1, key=jax.random.key(0), dtype=jnp.float32)
    fit = Fit(_Written(layer, jnp.zeros(()), jnp.zeros(())), {"a": _LayerBlock()})
    updated = fit.model_at({"a": {"u": jnp.asarray(0.25, jnp.float64)}}).a
    assert updated.weight.dtype == jnp.float32 and not updated.weight.weak_type
    np.testing.assert_array_equal(updated.weight, [[0.25]])
    np.testing.assert_array_equal(updated.bias, layer.bias)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_values_written_by_a_block_enter_the_model_as_strong_float_arrays(dtype):
    """Each block sees the dtypes of the leaves the preceding block replaced.

    The block writes a float32 array and a Python float at either width. Both writes
    must be strongly typed through density evaluation, prior sampling, the unit
    cube and model reconstruction.
    """
    seen = []
    probe = _DtypeProbeBlock()
    object.__setattr__(probe, "seen", seen)
    model = _Written(jnp.asarray(0.0, dtype), jnp.asarray(0.0), jnp.asarray(0.0, dtype))
    fit = Fit(model, {"a": _FloatWritingBlock(), "b": probe})
    x = {"a": {"u": jnp.asarray(0.25)}, "b": {"u": jnp.asarray(0.5)}}
    fit.log_density(x)
    prior = fit.sample_prior(jax.random.key(0))
    unit = fit.from_unit({"a": jnp.asarray([0.6]), "b": jnp.asarray([0.7])})
    for record in (prior, unit, fit.draw_schema()):
        assert all(s.dtype == jnp.float64 for s in jax.tree.leaves(record))
    assert len(seen) == 3
    assert all(row == ((dtype, False), (dtype, False)) for row in seen), seen
    at = fit.model_at(x)
    for leaf in (at.a, at.c):
        assert isinstance(leaf, jax.Array) and leaf.dtype == dtype
        assert not leaf.weak_type


class _InverseProbeBlock(_FloatWritingBlock):
    """A block with a different map that records what ``inverse`` sees."""

    seen = None

    def inverse(self, model):
        self.seen.append((model.a.dtype, model.a.weak_type))
        return {"u": model.a}

    def same_map(self, other):
        return False


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_a_support_swap_maps_block_writes_through_the_conversion():
    """The prior swap of a posterior re-expresses a draw through converted writes.

    Truth source: definitional (``hwoutils.leaves.canonical``), as for the fit's own
    write paths: the old block writes a float32 value, and the new block's ``inverse``
    inside the swap's Jacobian must see it as a strong float64. Fails if the swap
    assigns the block's values unconverted (the log Jacobian then runs at float32
    precision).
    """
    from photomancy.infer.posterior import _support_swap

    seen = []
    probe = _InverseProbeBlock()
    object.__setattr__(probe, "seen", seen)
    model = _Written(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))
    old = Fit(model, {"a": _FloatWritingBlock()})
    new = Fit(model, {"a": probe})
    _support_swap(old, new, ("a",))({"a": {"u": jnp.asarray(0.3)}})
    assert seen and all(row == (jnp.float64, False) for row in seen), seen


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_fit_refuses_what_the_conversion_missed(monkeypatch):
    """With the conversion removed the postcondition refuses a NumPy leaf by path.

    Truth source: definitional (``hwoutils.leaves.require_tree``, the check ``Fit``
    keeps after converting). ``canonical`` is replaced by the identity, so the NumPy
    scalar reaches the check. Fails if the model is accepted or the path
    ``curve.value`` is not named.
    """
    import photomancy.infer.fit as fit_module

    monkeypatch.setattr(fit_module, "canonical", lambda tree: tree)
    _, model = _holder(np.float64(2.0), 4.0, jnp.asarray(3.0, jnp.float64))
    with pytest.raises(TypeError, match=r"curve\.value"):
        Fit(model, {})


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_fit_enters_an_owned_module_but_not_a_foreign_one():
    """The interpolator inside an owned Module is kept as the same object.

    Truth source: definitional (``canonical`` and ``require_tree`` skip third-party
    Modules such as ``interpax`` interpolators and enter owned or user Modules).
    Fails if the model holding an interpolator is refused or the interpolator in
    ``fit.model`` is not the object that was passed in.
    """
    table, model = _holder(
        jnp.asarray(2.0, jnp.float64), 4.0, jnp.asarray(3.0, jnp.float64)
    )
    fit = Fit(model, {"curve.value": Normal(2.0, 0.1)})
    assert "curve.value" in fit.free
    assert fit.model.curve.table is table


class _NameReadingTerm(_ReadsRadius):
    reads = ("els[first]",)


class _NameFootprintTerm(_ReadsRadius):
    reads = ()

    @property
    def footprint_paths(self):
        return ("els[first]",)


class _IndexedLatentTerm(_ReadsRadius):
    reads = ()
    marginalizes = ("planets[b]",)


class _NamedWritingBlock(_FloatWritingBlock):
    owns: str = eqx.field(static=True, default="els[0]")
    outputs: tuple[str, ...] = eqx.field(static=True, default=("els[0].x",))

    @property
    def writes(self):
        return ("els[first]",)

    def forward(self, x, model):
        return {path: x["u"] for path in self.outputs}

    def inverse(self, model):
        return {"u": model.els[0].x}


class _DependentBlock(_FloatWritingBlock):
    owns = "els[second]"
    reads = ("els[0].x",)
    writes = ("els[1]",)

    def forward(self, x, model):
        return {"els[second].x": model.els[0].x + x["u"]}

    def inverse(self, model):
        return {"u": model.els[1].x - model.els[0].x}


class _TiedElement(NamedElement):
    tied_fields: ClassVar = {"derived": ("x",)}
    derived: NamedElement


class _FirstX(eqx.Module):
    def __call__(self, model):
        return model.els[0].x


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_named_and_numeric_free_paths_constrain_the_same_model():
    """One model and two spellings give the same density and model, at zero ULPs."""
    model = named_model()
    law = Normal(0.0, 1.0)
    named, numbered = "els[first].x", "els[0].x"
    # Reverse insertion order to exercise the lexicographic coordinate order.
    term = ForwardTerm(_FirstX(), gaussian_loglike(1.0, 1.0))
    by_name = Fit(model, {"els[third].x": law, named: law}, (term,))
    by_index = Fit(model, {numbered: law, "els[2].x": law}, (term,))
    aliases = {named: numbered, "els[third].x": "els[2].x"}
    key = jax.random.key(3)
    x_named, x_index = by_name.sample_prior(key), by_index.sample_prior(key)
    assert x_named.keys() == aliases.keys()
    assert x_index.keys() == set(aliases.values())
    assert by_name.canonical_keys == aliases
    assert tuple(by_name.canonical_keys) == tuple(aliases)
    by_name.canonical_keys.clear()
    assert by_name.canonical_keys == aliases
    assert by_name.flat_layout() == (((named, None), ("els[third].x", None)), (1, 1))
    assert tuple(by_name.draw_schema()) == (named, "els[third].x")
    z = jnp.asarray([0.25, 0.75])
    actual = by_name.real_log_prob(by_name.unravel(z))
    expected = by_index.real_log_prob(by_index.unravel(z))
    assert jnp.abs(actual - expected) <= ulps(0, expected)

    for actual, expected in zip(
        by_name.model_at(x_named).els, by_index.model_at(x_index).els, strict=True
    ):
        assert jnp.abs(actual.x - expected.x) <= ulps(0, expected.x)

    # The mixed keys sort in the opposite order from their canonical paths.
    indexed_third = "els[2].x"
    mixed = Fit(model, {named: law, indexed_third: law})
    assert mixed.flat_layout() == (((indexed_third, None), (named, None)), (1, 1))
    assert tuple(mixed.draw_schema()) == (indexed_third, named)
    x_mixed = mixed.sample_prior(key)
    assert jnp.abs(x_mixed[indexed_third] - x_index[numbered]) <= ulps(
        0, x_index[numbered]
    )
    assert jnp.abs(x_mixed[named] - x_index[indexed_third]) <= ulps(
        0, x_index[indexed_third]
    )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_numeric_free_path_preserves_released_digest_layout_and_draw():
    """Exact reference values recorded from the released numeric-path fixture."""
    fit = _simple_fit()
    path = "planets[0].Rp_Rearth"
    assert (
        fit.digest()
        == "f041dc38aef7b6b4bd6d8502e051c1ef55619a574abce2e9a473e1031ed7ea34"
    )
    assert fit.flat_layout() == (((path, None),), (1,))
    draw = fit.sample_prior(jax.random.key(0))
    assert draw.keys() == {path}
    np.testing.assert_array_equal(draw[path], [0.6959669742207993])
    pair = {**fit.free, "planets[0].physical_model.Ag": Uniform(0.0, 1.0)}
    reversed_pair = dict(reversed(pair.items()))
    assert Fit(fit.model, pair).digest() == Fit(fit.model, reversed_pair).digest()


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_free_aliases_are_two_writers():
    """Exactly one writer may select the first element's x leaf."""
    law = Normal(0.0, 1.0)
    with pytest.raises(ValueError, match=r"two writers.*els\[0\]\.x.*els\[first\]\.x"):
        Fit(named_model(), {"els[0].x": law, "els[first].x": law})


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_block_ownership_and_overlap_follow_canonical_paths():
    """One node owns its descendants, and its two spellings select that same node."""
    model = named_model()
    block = _NamedWritingBlock(owns="els[first]")
    with pytest.raises(ValueError, match="two writers"):
        Fit(model, {"els[first]": block, "els[0].x": Normal(0.0, 1.0)})

    fit = Fit(
        model, {"els[second]": _DependentBlock(), "els[first]": _NamedWritingBlock()}
    )
    assert fit.block_order == ("els[first]", "els[second]")
    assert fit.canonical_keys == {"els[first]": "els[0]", "els[second]": "els[1]"}
    at = fit.model_at({"els[first]": {"u": 4.0}, "els[second]": {"u": 5.0}})
    assert float(at.els[1].x) == 9.0


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_named_term_reads_overlap_an_indexed_free_leaf():
    """The cached first element contains its x leaf, including after term edits."""
    model, free = named_model(), {"els[0].x": Normal(0.0, 1.0)}
    initial = Fit(model, free, (_IndexedLatentTerm(),))
    term = _NameReadingTerm()
    for make in (
        lambda: Fit(model, free, (term,)),
        lambda: initial.append_term(term),
        lambda: initial.replace_term(0, term),
    ):
        with pytest.raises(ValueError, match=r"els\[0\]\.x.*reads.*els\[first\]"):
            make()


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_block_forward_respects_instance_write_paths():
    """Only descendants of the first element may change, with no alias duplicates."""
    model = named_model()
    x = {"els[first]": {"u": jnp.asarray(4.0)}}
    fit = Fit(model, {"els[first]": _NamedWritingBlock()})
    assert float(fit.model_at(x).els[0].x) == 4.0
    for outputs, message in (
        (("els[1].x",), r"els\[1\]\.x.*declared writes"),
        (("els[0].x", "els[first].x"), r"els\[0\]\.x.*els\[first\]\.x"),
    ):
        fit = Fit(model, {"els[first]": _NamedWritingBlock(outputs=outputs)})
        with pytest.raises(ValueError, match=message):
            eqx.filter_eval_shape(fit.model_at, x)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_named_ancestor_enforces_its_relative_tied_fields():
    """The declared derived subtree forbids writing just one of its leaves."""
    model = named_model()
    first = model.els[0]
    tied = _TiedElement(first.name, first.x, NamedElement("derived", first.x))
    model = eqx.tree_at(lambda m: m.els[0], model, tied)
    with pytest.raises(ValueError, match=r"els\[first\]\.derived\.x.*tied together"):
        Fit(model, {"els[first].derived.x": Normal(0.0, 1.0)})


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_path_shaped_latent_identifier_does_not_bind_to_model():
    """A marginalized latent can have any identifier, including an absent path."""
    term = _IndexedLatentTerm()
    fit = Fit(named_model(), {"els[0].x": Normal(0.0, 1.0)}, (term,))
    assert fit.terms[0].marginalizes == ("planets[b]",)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_named_footprint_checks_indexed_leaf_prior_endpoints():
    """An unbounded x prior moves the first element's footprint and is refused."""
    with pytest.raises(ValueError, match=r"els\[0\]\.x.*unbounded end"):
        Fit(named_model(), {"els[0].x": Normal(0.0, 1.0)}, (_NameFootprintTerm(),))
