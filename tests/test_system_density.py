"""Chart pullbacks of the existing one-planet relative-astrometry joint.

Two chart structures share four regular orbits and three synthetic observations.
No existing test checks a built-system density or its reference-measure prior.
"""

from unittest.mock import Mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg
from hwoutils.epoch import Epoch
from numpyro.infer.util import log_density
from orbix.equations import period_to_sma
from orbix.system import Body, ClassicalChart, EccentricityVectorChart, SkyFrame, System
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene.planet import Planet
from skyscapes.scene.star import FlatStar
from skyscapes.scene.system import System as SceneSystem

from photomancy import RelativeAstrometryTerm, SystemLogDensity
from photomancy.orbit.data import OrbitData, RelativeAstromData
from photomancy.orbit.model import build_model
from photomancy.priors import orbit_log_prior


def _floor(*terms):
    """Float64 accumulation floor specified for comparisons of log densities."""
    return max(
        1e-10, 64 * np.finfo(float).eps * sum(np.sum(1 + np.abs(t)) for t in terms)
    )


def _prior(system):
    """Score the scalar orbit owned by the one-planet system."""
    return orbit_log_prior(jax.tree.map(lambda x: x[0], system.orbits))


@pytest.fixture(scope="module")
def fits():
    """Share two chart structures, their pullbacks, and four physical points."""
    data = RelativeAstromData(
        times=jnp.array([0.0, 83.0, 211.0]),
        ra=jnp.array([0.08, -0.03, 0.11]),
        dec=jnp.array([-0.02, 0.06, 0.04]),
        ra_err=jnp.array([0.03, 0.025, 0.04]),
        dec_err=jnp.array([0.02, 0.035, 0.03]),
        corr=jnp.array([0.2, -0.3, 0.1]),
        planet_id=jnp.zeros(3, dtype=int),
        is_valid=jnp.ones(3, dtype=bool),
    )
    periods = jnp.array([120.0, 480.0, 1200.0, 4000.0])
    eccentricities = jnp.array([0.06, 0.25, 0.55, 0.8])
    classical = jnp.stack(
        [
            jnp.log(period_to_sma(periods, Msun2kg)),
            jnp.log(eccentricities / (1 - eccentricities)),
            jnp.arctanh(jnp.array([0.8, 0.2, -0.4, -0.85])),
            jnp.array([0.7, 2.1, 3.4, 5.2]),
            jnp.array([0.5, 1.3, 2.8, 4.5]),
            jnp.array([0.4, 2.0, 4.0, 5.0]),
        ],
        axis=-1,
    )
    reference_orbits = ClassicalChart().to_orbit(classical, mu_AU3_per_d2=G * Msun2kg)
    model = build_model(has_relative_astrom=True)
    records = []
    for chart in (ClassicalChart(), EccentricityVectorChart()):

        def build(params, chart=chart):
            return System(
                Body(1.0),
                Body(jnp.zeros(1)),
                chart.to_orbit(params[None, :], mu_AU3_per_d2=G * Msun2kg),
                SkyFrame(10.0, Epoch(2451545.0)),
            )

        density = SystemLogDensity(
            build, (RelativeAstrometryTerm(data),), _prior, chart.log_volume
        )
        points = chart.from_orbit(reference_orbits)

        def sites(x, build=build):
            orbit = jax.tree.map(lambda leaf: leaf[0], build(x).orbits)
            elements = orbit.to_classical()
            return jnp.stack(
                [
                    jnp.log10(orbit.period_d),
                    elements["e"],
                    jnp.cos(elements["i_rad"]),
                    elements["W_rad"],
                    elements["w_rad"],
                    elements["M0_rad"],
                ]
            )

        def compare(x, sites=sites, density=density):
            z = sites(x)
            params = {
                name: z[i : i + 1]
                for i, name in enumerate(
                    ("log_P", "e_raw", "cos_i", "W", "w_raw", "M0")
                )
            }
            joint, trace = log_density(
                model, (Msun2kg, 10.0, OrbitData(relative_astrom=data)), {}, params
            )
            determinant = jnp.linalg.slogdet(jax.jacfwd(sites)(x))[1]
            value, aux = density.value_and_aux(x)
            contributions = jnp.stack(
                [
                    site["fn"].log_prob(site["value"]).sum()
                    for site in trace.values()
                    if site["type"] == "sample"
                ]
            )
            return (
                value,
                joint + determinant,
                density.log_volume(x),
                aux["term_values"][0],
                contributions,
            )

        comparison = eqx.filter_jit(jax.vmap(compare))(points)
        jax.block_until_ready(comparison)
        records.append((chart, density, points, comparison))
    return records


def test_model_parity(fits):
    """Fails if the prior, likelihood, or chart determinant changes the joint.

    Subtract one constant at the first orbit. Absolute tolerance is
    max(1e-10, 64 eps sum(1+abs(term))), including both points' joint
    contributions and volumes: a floor for float64 arithmetic accumulation.
    """
    for chart, _, _, (value, reference, volume, likelihood, contributions) in fits:
        delta = np.asarray(value - reference)
        errors = np.abs(delta - delta[0])
        tolerances = [
            _floor(
                contributions[i],
                contributions[0],
                volume[i],
                volume[0],
                likelihood[i],
                likelihood[0],
            )
            for i in range(len(delta))
        ]
        assert np.all(errors <= tolerances)
        print(
            f"{type(chart).__name__}: points={len(delta)}, "
            f"max_error={errors.max():.17g}, tolerance={max(tolerances):.17g}, "
            f"anchor={delta[0]:.17g}"
        )


def test_chart_independence(fits):
    """Fails if removing log_volume leaves a chart-dependent physical density.

    Use max(1e-10, 64 eps sum(1+abs(term))) for the two independently
    constructed systems' likelihoods, priors and volume arithmetic.
    """
    left, right = (record[3] for record in fits)
    for i in range(len(left[0])):
        a, b = left[0][i] - left[2][i], right[0][i] - right[2][i]
        np.testing.assert_allclose(
            a, b, rtol=0, atol=_floor(*(x[i] for x in left), *(x[i] for x in right))
        )


def test_support_and_masks(fits):
    """Fails if support leaks or padded NaNs affect the value or its gradient.

    Masked and clean-padding scores/gradients use the 64 eps accumulation
    floor (at least 1e-10); excluded support and zero predictions are exact.
    """
    _, density, points, _ = fits[1]
    x = points[1]
    prior_only = SystemLogDensity(density.build, (), density.prior, density.log_volume)
    for log_a in (-8.0, 8.0):
        assert jnp.isneginf(prior_only(x.at[0].set(log_a)))
    assert jnp.isneginf(prior_only(x.at[1:3].set(0.0)))
    orbit = jax.tree.map(lambda leaf: leaf[0], density.build(x).orbits)
    for e in (1.0, 1.2):
        invalid = eqx.tree_at(
            lambda o: (o.ex, o.ey), orbit, (jnp.asarray(e), jnp.asarray(0.0))
        )
        assert jnp.isneginf(orbit_log_prior(invalid))

    data = density.terms[0].data
    valid = jnp.array([True, True, False])
    clean = eqx.tree_at(lambda d: d.is_valid, data, valid)
    poisoned = jax.tree.map(
        lambda leaf: (
            leaf.at[-1].set(jnp.nan)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            else leaf
        ),
        clean,
    )
    poisoned = eqx.tree_at(lambda d: d.planet_id, poisoned, jnp.array([0, 0, 999]))
    score = eqx.filter_jit(
        jax.value_and_grad(lambda x, term: term(density.build(x))[0])
    )
    clean_value, clean_grad = score(x, RelativeAstrometryTerm(clean))
    bad_value, bad_grad = score(x, RelativeAstrometryTerm(poisoned))
    np.testing.assert_allclose(bad_value, clean_value, rtol=0, atol=_floor(clean_value))
    np.testing.assert_allclose(bad_grad, clean_grad, rtol=0, atol=_floor(clean_grad))
    assert np.isfinite(bad_grad).all()
    prediction = RelativeAstrometryTerm(poisoned)(density.build(x))[1]
    np.testing.assert_array_equal(prediction[:, -1], 0.0)
    all_masked = eqx.tree_at(lambda d: d.is_valid, poisoned, jnp.zeros(3, dtype=bool))
    value, gradient = score(x, RelativeAstrometryTerm(all_masked))
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(x))


def test_aux_jit_gradient_and_prior_only(fits):
    """Fails if assembly repeats evaluation, loses aux, or breaks differentiation.

    Eager/JIT and aux use the 64 eps accumulation floor, at least 1e-10.
    Central differences use h=2e-6 in each dimensionless chart coordinate;
    rtol=2e-6, atol=2e-6 allow O(h^2) truncation and O(eps/h) cancellation.
    """
    for _, density, points, _ in fits:
        x = points[1]
        value, aux = density.value_and_aux(x)
        np.testing.assert_allclose(density(x), value, rtol=0, atol=_floor(value))
        term_value, prediction = density.terms[0](density.build(x))
        np.testing.assert_array_equal(aux["term_values"], (term_value,))
        np.testing.assert_array_equal(aux["predictions"][0], prediction)
        compiled = eqx.filter_jit(density)
        np.testing.assert_allclose(compiled(x), value, rtol=0, atol=_floor(value))
        gradient = eqx.filter_jit(eqx.filter_grad(density))(x)
        steps = np.eye(6) * 2e-6
        finite_diff = np.array(
            [(compiled(x + h) - compiled(x - h)) / 4e-6 for h in steps]
        )
        np.testing.assert_allclose(gradient, finite_diff, rtol=2e-6, atol=2e-6)
        prior_only = SystemLogDensity(
            density.build, (), density.prior, density.log_volume
        )
        prior_value, prior_aux = prior_only.value_and_aux(x)
        np.testing.assert_allclose(
            prior_value, value - term_value, rtol=0, atol=_floor(value, term_value)
        )
        assert prior_aux == {"term_values": (), "predictions": ()}

    second_term = eqx.tree_at(
        lambda t: t.data.ra, density.terms[0], density.terms[0].data.ra + 0.01
    )
    counted = SystemLogDensity(
        Mock(wraps=lambda params: density.build(params["orbit"])),
        (Mock(wraps=density.terms[0]), Mock(wraps=second_term)),
        Mock(wraps=density.prior),
        Mock(wraps=lambda params: density.log_volume(params["orbit"])),
    )
    total, aux = counted.value_and_aux({"orbit": x})
    second_value, second_prediction = second_term(density.build(x))
    np.testing.assert_allclose(
        total, value + second_value, rtol=0, atol=_floor(value, second_value)
    )
    np.testing.assert_array_equal(aux["term_values"], (term_value, second_value))
    np.testing.assert_array_equal(aux["predictions"], (prediction, second_prediction))
    for part in (
        counted.build,
        *counted.terms,
        counted.prior,
        counted.orbit_log_volume,
    ):
        assert part.call_count == 1


def test_bare_system_matches_scene(fits):
    """Fails if bare-system body indexing or sky units differ from the scene.

    Prediction atol=64 eps max(1,abs(prediction)) arcsec covers roundoff in
    shared solve/observable paths; scores use the 1e-10/64 eps density floor.
    """
    _, density, points, _ = fits[0]
    system = density.build(points[1])
    scene = SceneSystem(
        FlatStar(1.0),
        (
            Planet(
                Rp_Rearth=jnp.ones(1),
                physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
            ),
        ),
        system,
    )
    bare_value, bare_prediction = density.terms[0](system)
    scene_value, scene_prediction = density.terms[0](scene)
    np.testing.assert_allclose(
        bare_prediction,
        scene_prediction,
        rtol=0,
        atol=64 * np.finfo(float).eps * max(1, np.max(np.abs(scene_prediction))),
    )
    np.testing.assert_allclose(
        bare_value, scene_value, rtol=0, atol=_floor(bare_value, scene_value)
    )


def test_tempering_split_shares_system_and_scalar_units():
    """The prior includes every Jacobian. Only the data score is tempered."""
    from photomancy.system_density import ScalarCoordinate

    build = Mock(side_effect=lambda params, mass: mass + params["orbit"])
    term = Mock(side_effect=lambda model, jitter: (model / jitter, None))
    prior = Mock(side_effect=lambda model: -model)
    volume = Mock(return_value=0.75)
    density = SystemLogDensity(
        build,
        (term,),
        prior,
        volume,
        {
            "mass": ScalarCoordinate(lambda mass: -mass),
            "jitter": ScalarCoordinate(lambda jitter: -jitter),
        },
        ("mass",),
        {0: ("jitter",)},
    )
    p = {"orbit": 1.0, "mass": jnp.log(2.0), "jitter": jnp.log(3.0)}
    ll = density.loglikelihood(p)
    assert term.call_count == build.call_count == 1
    assert prior.call_count == volume.call_count == 0
    lp = density.logprior(p)
    assert term.call_count == 1
    np.testing.assert_allclose(ll, 1.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(lp, -8.0 + 0.75 + np.log(6.0), atol=1e-14, rtol=0)
    np.testing.assert_allclose(density(p), ll + lp, atol=1e-14, rtol=0)


@pytest.fixture(scope="module")
def sampling_problem():
    """One synthetic measurement of a planet's log semimajor axis."""
    from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity
    from photomancy.priors import FactorizedOrbitPrior

    def orbit_of(system):
        return jax.tree.map(lambda leaf: leaf[0], system.orbits)

    def build(orbit):
        return System(
            Body(1.0),
            Body(jnp.zeros(1)),
            jax.tree.map(lambda leaf: leaf[None], orbit),
            SkyFrame(10.0, Epoch()),
        )

    def measurement(system):
        log_a = jnp.log(orbit_of(system).a_AU)
        return -0.5 * ((log_a - 0.35) / 0.8) ** 2, log_a

    # Broad orbital data and two draws exercise each transition and decoder;
    # they are not an estimate of posterior moments.
    position = {"orbit": jnp.array([0.25, -1.0, -0.2, 2.0, 2.1, 3.0])}
    density = build_chart_logdensity(
        ClassicalChart(),
        build,
        FactorizedOrbitPrior(orbit_of, ecc_prior="disk"),
        terms=(measurement,),
        star_mass_kg=Msun2kg,
    )
    return ChartOrbitProblem(density, position)


@pytest.mark.medium(reason="Exercises a compiled backend transition on one orbit.")
@pytest.mark.parametrize(
    "engine", ["map", "nuts", "mclmc", "pathfinder", "pathfinder_mixture", "smc"]
)
def test_chart_backend_step(sampling_problem, engine):
    """A fit or transition must move from its start to a finite chart density."""
    import optax

    from photomancy.backends import (
        MAPBackend,
        MCLMCBackend,
        NUTSBackend,
        PathfinderBackend,
        PathfinderMixtureBackend,
        SMCBackend,
    )
    from photomancy.backends.staged import FitStage

    problem = sampling_problem
    z = problem.flatten(problem.position_template)
    key = jax.random.key(14)
    if engine == "smc":
        posterior = SMCBackend(step_size=0.05, n_mcmc_steps=1).run_chart(
            problem,
            n_particles=8,
            key=key,
        )
        rows = posterior.samples
        assert np.isfinite(posterior.evidence)
    elif engine == "map":
        backend = MAPBackend(
            "orbit", {"orbit": optax.adam(0.01)}, (FitStage(1, ("orbit",)),)
        )
        post = backend.run(problem.logdensity, z)
        rows = post.params[None, :]
        assert post.log_density > problem.logdensity(z)
    elif engine == "pathfinder_mixture":
        starts = jnp.stack((z, z.at[0].add(0.1)))
        post = PathfinderMixtureBackend(maxiter=6, maxcor=3).run(
            problem.logdensity, starts, key
        )
        rows = post.means
        # An ELBO draw outside the chart cell scores -inf, so a short path can
        # give -inf; NaN would mean a broken fit.
        assert not np.any(np.isnan(post.log_evidences))
    elif engine == "pathfinder":
        post = PathfinderBackend(maxiter=6, maxcor=3).run(problem.logdensity, z, key)
        rows = post.mean[None, :]
        assert np.all(np.linalg.eigvalsh(post.cov) > 0)
    else:
        backend = (
            NUTSBackend(n_warmup=10, n_samples=2)
            if engine == "nuts"
            else MCLMCBackend(n_tune=100, n_samples=2)
        )
        rows = backend.run(problem.logdensity, z, key).samples
        assert not np.array_equal(rows[0], rows[1])
    assert rows.shape[-1] == len(problem.param_names)
    assert np.any(np.asarray(rows) != z)
    assert np.all(np.isfinite(jax.vmap(problem.logdensity)(rows)))
