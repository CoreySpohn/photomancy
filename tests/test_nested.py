"""Nested-sampling backend (jaxns): Bayesian evidence + the scene-model adapter."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from hwoutils.constants import Msun2kg
from jaxctx import ScopedDict
from jaxctx import transform as transform_model
from orbix.system import EccentricityVectorChart, FrameChart
from priorities import HalfNormal, LogUniform, TruncatedDistribution
from system_helpers import BOUNDS, _floor, _new_term, _scene, ecc_law

from photomancy import RadialVelocityTerm
from photomancy.infer.backends import JaxnsBackend, build_scene_nested_model
from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity
from photomancy.orbit.nested import (
    OrbitPriorTransform,
    build_chart_nested_model,
    orbit_nested_sampling,
)
from photomancy.orbit.system_density import ScalarCoordinate
from photomancy.priors import FactorizedOrbitPrior

# jaxns requires x64; make the standalone file independent of other tests.
jax.config.update("jax_enable_x64", True)

tfpd = tfp.distributions


class _Toy(eqx.Module):
    theta: jnp.ndarray
    label: str = eqx.field(static=True)


@pytest.mark.slow
def test_jaxns_recovers_analytic_evidence():
    """JaxnsBackend computes the conjugate-Gaussian log Z (mirrors SMC-vs-analytic)."""
    from jaxns.model import Model
    from jaxns.priors import Prior

    data, lik_s, pri_s = 2.0, 0.5, 1.0
    # data = x + noise -> data ~ N(0, sqrt(pri_s^2 + lik_s^2)); analytic evidence below
    analytic = float(tfpd.Normal(0.0, jnp.sqrt(pri_s**2 + lik_s**2)).log_prob(data))

    def prior_model():
        x = Prior(tfpd.Normal(0.0, pri_s), name="x").realise()
        return tfpd.Normal(data, lik_s).log_prob(x)

    model = Model(prior_model=prior_model)
    post = JaxnsBackend(max_samples=2e4).run(model, key=jax.random.key(0))

    assert bool(jnp.isfinite(post.evidence))
    assert abs(float(post.evidence) - analytic) < 0.5  # nested-sampling Z is stochastic
    assert post.samples.shape[1] == 1


@pytest.mark.slow
def test_build_scene_nested_model_recovers_and_gives_evidence():
    """Scene adapter: a distribution over z; recovers truth + log Z."""
    from priorities import Uniform

    truth = jnp.array([0.3, -0.4])
    sigma = 0.05
    data = truth  # noiseless for a deterministic recovery check
    scene = _Toy(theta=jnp.zeros(2), label="t")

    def forward(s):
        return s.theta

    def likelihood(pred):
        return -0.5 * jnp.sum((pred - data) ** 2 / sigma**2)

    def fit_leaves(s):
        return [s.theta]

    prior = Uniform(-jnp.ones(2), jnp.ones(2))  # Uniform(-1, 1) over z
    model, unravel = build_scene_nested_model(
        scene, forward, likelihood, fit_leaves=fit_leaves, prior=prior
    )
    post = JaxnsBackend(max_samples=4e4).run(model, key=jax.random.key(0))

    assert bool(jnp.isfinite(post.evidence))
    assert post.samples.shape[1] == 2
    # z-space samples reconstruct a scene through the returned unravel.
    assert unravel(post.samples[0]).theta.shape == (2,)
    # posterior mean recovers the truth.
    w = jax.nn.softmax(post.log_weights)
    mean = jnp.sum(post.samples * w[:, None], axis=0)
    assert jnp.allclose(mean, truth, atol=0.06)


@pytest.mark.slow
def test_jaxns_prior_adapter_recovers_analytic_evidence_tfp_free():
    """A ``priorities`` Normal prior, through the adapter, recovers log Z (no tfp).

    The conjugate-Gaussian check of test_jaxns_recovers_analytic_evidence, but the prior
    is a ``priorities.Normal`` wrapped by the adapter inside
    ``build_scene_nested_model`` -- so the whole prior path is tfp-free.
    """
    from priorities import Normal

    data, lik_s, pri_s = 2.0, 0.5, 1.0
    log2pi = jnp.log(2.0 * jnp.pi)
    # data ~ N(0, sqrt(pri_s^2 + lik_s^2)); analytic evidence (closed form, no tfp).
    var = pri_s**2 + lik_s**2
    analytic = float(-0.5 * (data**2 / var + jnp.log(var) + log2pi))

    scene = _Toy(theta=jnp.zeros(1), label="t")

    def forward(s):
        return s.theta

    def likelihood(pred):
        return -0.5 * ((pred[0] - data) / lik_s) ** 2 - jnp.log(lik_s) - 0.5 * log2pi

    model, _ = build_scene_nested_model(
        scene,
        forward,
        likelihood,
        fit_leaves=lambda s: [s.theta],
        prior=Normal(loc=jnp.zeros(1), scale=jnp.full((1,), pri_s)),
    )
    post = JaxnsBackend(max_samples=2e4).run(model, key=jax.random.key(0))

    assert bool(jnp.isfinite(post.evidence))
    assert abs(float(post.evidence) - analytic) < 0.5  # nested-sampling Z is stochastic


def _scene_orbit(scene):
    return jax.tree.map(lambda leaf: leaf[0], scene.dynamics.orbits)


def _nested_problem(chart, ecc_prior, rv=None, e_max=1.0):
    law = ecc_law(ecc_prior)
    if e_max < 1.0:
        law = TruncatedDistribution(law, high=e_max)
    prior = FactorizedOrbitPrior(_scene_orbit, (10.0, 10000.0), law)
    scalars = {
        name: ScalarCoordinate(LogUniform(*bounds)) for name, bounds in BOUNDS.items()
    }
    template = dict.fromkeys(scalars, 0.0)
    if rv is not None:
        scalars["jitter_m_per_s"] = ScalarCoordinate(HalfNormal(0.8), logarithmic=False)
        template["jitter_m_per_s"] = jnp.zeros(rv.n_inst)
    density = build_chart_logdensity(
        chart,
        _scene,
        prior,
        star_mass_kg=Msun2kg,
        terms=() if rv is None else (RadialVelocityTerm(rv),),
        scalars=scalars,
        body_inputs=("radius_AU", "albedo"),
        term_inputs={} if rv is None else {0: ("jitter_m_per_s",)},
    )
    return ChartOrbitProblem(density, dict(orbit=jnp.zeros(chart.n_coords), **template))


# Frozen from the retired unit-cube joint at the four cube points of
# test_chart_nested_cube_matches_frozen_reference: the physical values (T, a, e, cos i,
# W, cos w, sin w, tp, Ag, Mp, Rp and three jitters) and the log joint, per
# eccentricity law.
_FROZEN_CUBE = {
    "kipping13": (
        [
            [
                19.952623149688797,
                0.1439695303888295,
                0.033308432799780935,
                -0.5780000000000001,
                1.5582299561805373,
                0.4595798606214879,
                0.8881364488135445,
                -5.686497597661308,
                0.04405548635065534,
                3.945772180651261e25,
                0.0001851398894205272,
                0.37304087050120843,
                0.4149255456643127,
                0.4579802574804078,
            ],
            [
                112.20184543019641,
                0.4553638281032702,
                0.12171237530049483,
                -0.07799999999999996,
                3.129026282975434,
                -0.8881364488135447,
                0.4595798606214876,
                -60.02798730515508,
                0.13931568029453034,
                1.2477627218987342e27,
                0.000780728440137907,
                0.6862442149345948,
                0.7414868102528348,
                0.8005134630099596,
            ],
            [
                446.6835921509635,
                1.1473617294931335,
                0.22003514924308704,
                0.32200000000000006,
                4.385663344411352,
                -0.7115356772092855,
                -0.7026499697988491,
                -328.3124402309582,
                0.3499451670283573,
                1.9775706437440563e28,
                0.002468880104906209,
                1.0461012567862369,
                1.1404352296643614,
                1.2539926913067518,
            ],
            [
                2041.7379446695295,
                3.1496898019426016,
                0.38499817517421236,
                0.762,
                5.76796411199086,
                0.5568756164881876,
                -0.8305958991958129,
                -1949.859737159401,
                0.012705741052085419,
                9.465248197897579e23,
                3.912911271268532e-05,
                0.0894218265678464,
                0.12686378380307492,
                0.16458586711453094,
            ],
        ],
        [
            -13.029116522725396,
            -75.34873243348285,
            -13932.502965778825,
            -10.096336755588077,
        ],
    ),
    "rayleigh": (
        [
            [
                19.952623149688797,
                0.1439695303888295,
                0.1625142210557195,
                -0.5780000000000001,
                1.5582299561805373,
                0.4595798606214879,
                0.8881364488135445,
                -5.686497597661308,
                0.04405548635065534,
                3.945772180651261e25,
                0.0001851398894205272,
                0.37304087050120843,
                0.4149255456643127,
                0.4579802574804078,
            ],
            [
                112.20184543019641,
                0.4553638281032702,
                0.29605992998347913,
                -0.07799999999999996,
                3.129026282975434,
                -0.8881364488135447,
                0.4595798606214876,
                -60.02798730515508,
                0.13931568029453034,
                1.2477627218987342e27,
                0.000780728440137907,
                0.6862442149345948,
                0.7414868102528348,
                0.8005134630099596,
            ],
            [
                446.6835921509635,
                1.1473617294931335,
                0.39772987284371303,
                0.32200000000000006,
                4.385663344411352,
                -0.7115356772092855,
                -0.7026499697988491,
                -328.3124402309582,
                0.3499451670283573,
                1.9775706437440563e28,
                0.002468880104906209,
                1.0461012567862369,
                1.1404352296643614,
                1.2539926913067518,
            ],
            [
                2041.7379446695295,
                3.1496898019426016,
                0.5415028411774943,
                0.762,
                5.76796411199086,
                0.5568756164881876,
                -0.8305958991958129,
                -1949.859737159401,
                0.012705741052085419,
                9.465248197897579e23,
                3.912911271268532e-05,
                0.0894218265678464,
                0.12686378380307492,
                0.16458586711453094,
            ],
        ],
        [
            -12.980714078999998,
            -101.67107140486027,
            -10553.886405356985,
            -10.094966537888713,
        ],
    ),
    "vaneylen19": (
        [
            [
                19.952623149688797,
                0.1439695303888295,
                0.02659942674362741,
                -0.5780000000000001,
                1.5582299561805373,
                0.4595798606214879,
                0.8881364488135445,
                -5.686497597661308,
                0.04405548635065534,
                3.945772180651261e25,
                0.0001851398894205272,
                0.37304087050120843,
                0.4149255456643127,
                0.4579802574804078,
            ],
            [
                112.20184543019641,
                0.4553638281032702,
                0.04847733931752582,
                -0.07799999999999996,
                3.129026282975434,
                -0.8881364488135447,
                0.4595798606214876,
                -60.02798730515508,
                0.13931568029453034,
                1.2477627218987342e27,
                0.000780728440137907,
                0.6862442149345948,
                0.7414868102528348,
                0.8005134630099596,
            ],
            [
                446.6835921509635,
                1.1473617294931335,
                0.06516475664247773,
                0.32200000000000006,
                4.385663344411352,
                -0.7115356772092855,
                -0.7026499697988491,
                -328.3124402309582,
                0.3499451670283573,
                1.9775706437440563e28,
                0.002468880104906209,
                1.0461012567862369,
                1.1404352296643614,
                1.2539926913067518,
            ],
            [
                2041.7379446695295,
                3.1496898019426016,
                0.0888797083847093,
                0.762,
                5.76796411199086,
                0.5568756164881876,
                -0.8305958991958129,
                -1949.859737159401,
                0.012705741052085419,
                9.465248197897579e23,
                3.912911271268532e-05,
                0.0894218265678464,
                0.12686378380307492,
                0.16458586711453094,
            ],
        ],
        [
            -13.027060372066533,
            -62.712349500332174,
            -17515.238952347838,
            -10.097087012049194,
        ],
    ),
    "disk": (
        [
            [
                19.952623149688797,
                0.1439695303888295,
                0.13699999999999998,
                -0.5780000000000001,
                1.5582299561805373,
                0.4595798606214879,
                0.8881364488135445,
                -5.686497597661308,
                0.04405548635065534,
                3.945772180651261e25,
                0.0001851398894205272,
                0.37304087050120843,
                0.4149255456643127,
                0.4579802574804078,
            ],
            [
                112.20184543019641,
                0.4553638281032702,
                0.387,
                -0.07799999999999996,
                3.129026282975434,
                -0.8881364488135447,
                0.4595798606214876,
                -60.02798730515508,
                0.13931568029453034,
                1.2477627218987342e27,
                0.000780728440137907,
                0.6862442149345948,
                0.7414868102528348,
                0.8005134630099596,
            ],
            [
                446.6835921509635,
                1.1473617294931335,
                0.5870000000000001,
                0.32200000000000006,
                4.385663344411352,
                -0.7115356772092855,
                -0.7026499697988491,
                -328.3124402309582,
                0.3499451670283573,
                1.9775706437440563e28,
                0.002468880104906209,
                1.0461012567862369,
                1.1404352296643614,
                1.2539926913067518,
            ],
            [
                2041.7379446695295,
                3.1496898019426016,
                0.807,
                0.762,
                5.76796411199086,
                0.5568756164881876,
                -0.8305958991958129,
                -1949.859737159401,
                0.012705741052085419,
                9.465248197897579e23,
                3.912911271268532e-05,
                0.0894218265678464,
                0.12686378380307492,
                0.16458586711453094,
            ],
        ],
        [
            -13.003439483915155,
            -105.17278871012685,
            -9583.69260121713,
            -10.09242120192119,
        ],
    ),
}


@pytest.mark.medium(reason="Differentiates both chart and normalized priors on a cube.")
@pytest.mark.parametrize("name", ["kipping13", "rayleigh", "vaneylen19", "disk"])
def test_chart_nested_cube_matches_frozen_reference(name):
    """Compare physical systems, likelihoods and normalized cube pullbacks separately.

    Four regular phases, inclinations, periods, eccentricities and body scales;
    all three instrument jitter components vary. Physical values use relative
    float64 tolerance 1e-12. The two log-density differences use the per-point
    accumulation floor max(1e-10, 64 eps sum(1 + abs(contributions))).
    """
    rv = _new_term("rv").data
    problem = _nested_problem(EccentricityVectorChart(), name, rv)
    transform = OrbitPriorTransform(problem)
    nested_model = transform_model(
        build_chart_nested_model(problem).prior_model, base_dtype=jnp.float64
    )
    physical = ("T", "a", "e", "cos_i", "W", "cos_w", "sin_w", "tp", "Ag", "Mp", "Rp")

    def compare(unit):
        realized = nested_model.apply(None, {"U": ScopedDict({"unit": unit})})
        z = realized.collections["X"]["z"]
        decoded = problem.to_physical(z)
        actual = jnp.stack([decoded[k] for k in physical])
        actual = jnp.concatenate((actual, decoded["jitter_m_per_s"]))
        likelihood = realized.fn_val
        logdet = jnp.linalg.slogdet(jax.jacfwd(transform.forward)(unit))[1]
        return actual, likelihood, problem.logdensity(z), logdet

    points = (
        jnp.arange(12) * 0.037 + jnp.array([0.08, 0.33, 0.53, 0.75])[:, None]
    ) % 0.94 + 0.02
    actual, likelihood, joint, logdet = eqx.filter_jit(jax.vmap(compare))(points)
    expected, reference = (jnp.asarray(x) for x in _FROZEN_CUBE[name])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    tolerances = np.array(
        [_floor(*row) for row in zip(reference, likelihood, joint, logdet, strict=True)]
    )
    like_err = np.abs(likelihood - reference)
    cube_err = np.abs(joint + logdet - reference)
    assert np.all(like_err <= tolerances)
    assert np.all(cube_err <= tolerances)
    print(
        f"{name}: physical_rel={np.max(np.abs(actual / expected - 1)):.3g}, "
        f"loglikelihood={like_err.max():.3g}, pulled_joint={cube_err.max():.3g}, "
        f"max_floor={tolerances.max():.3g}"
    )


@pytest.mark.parametrize("sd", [None, 0.9])
def test_chart_prior_draw_includes_frame_gauge(sd):
    """Both quaternion signs and the sampled radial density normalize on the cube."""
    problem = _nested_problem(FrameChart(gauge_sd=sd), "disk", e_max=0.7)
    transform = OrbitPriorTransform(problem)
    unit = jnp.linspace(0.11, 0.73, transform.ndim)

    def check(unit):
        z = transform.forward(unit)
        jac = jax.jacfwd(transform.forward)(unit)
        return problem.unflatten(z)["orbit"][3:], problem.logprior(
            z
        ) + jnp.linalg.slogdet(jac)[1]

    quat, cube_density = eqx.filter_jit(jax.vmap(check))(
        jnp.stack((unit.at[-1].set(0.12), unit.at[-1].set(0.62)))
    )
    np.testing.assert_allclose(quat[0], -quat[1], rtol=1e-12)
    np.testing.assert_allclose(cube_density, 0, atol=1e-10)


@pytest.mark.medium(reason="Runs a synthetic one-planet nested sampler.")
def test_prior_only_chart_evidence():
    """A constant likelihood integrates to one under the factorized physical prior."""
    problem = _nested_problem(EccentricityVectorChart(), "disk")
    post = orbit_nested_sampling(
        problem,
        num_live_points=8,
        max_samples=256,
        num_samples=16,
        key=jax.random.key(31),
    )
    error = float(post.diagnostics["log_evidence_error"])
    print(f"prior only: logZ={float(post.evidence):.7g} +/- {error:.7g}")
    assert abs(float(post.evidence)) <= 4 * error + 1e-10
    assert post.param_names == problem.param_names
    problem.require(post)
    assert post.chart == "EccentricityVectorChart"
    assert np.isfinite(jax.vmap(problem.logprior)(post.samples)).all()
