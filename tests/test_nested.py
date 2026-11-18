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
from numpyro.handlers import reparam
from numpyro.infer.util import log_density
from orbix.system import EccentricityVectorChart, FrameChart
from test_system_density import _floor
from test_system_priors import BOUNDS, _scene
from test_terms import _new_term

from photomancy import RadialVelocityTerm, ScalarCoordinate
from photomancy.backends import JaxnsBackend, build_scene_nested_model
from photomancy.backends.nested import build_chart_nested_model
from photomancy.orbit.data import OrbitData
from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity
from photomancy.orbit.model import build_model
from photomancy.orbit.nested import (
    OrbitPriorTransform,
    _UnitCubeReparam,
    orbit_nested_sampling,
)
from photomancy.priors import FactorizedOrbitPrior, HalfNormalPrior, LogUniformPrior

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
    """Scene adapter: a photomancy AbstractPrior over z; recovers truth + log Z."""
    from photomancy.priors import Uniform

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

    prior = Uniform(low=-jnp.ones(2), high=jnp.ones(2))  # Uniform(-1, 1) over z
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
    """A photomancy Normal prior, via the _JaxnsPrior adapter, recovers log Z (no tfp).

    The conjugate-Gaussian check of test_jaxns_recovers_analytic_evidence, but the prior
    is our own ``photomancy.priors.Normal`` wrapped by the adapter inside
    ``build_scene_nested_model`` -- so the whole prior path is tfp-free.
    """
    from photomancy.priors import Normal

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
    prior = FactorizedOrbitPrior(_scene_orbit, (10.0, 10000.0), ecc_prior, e_max)
    scalars = {
        name: ScalarCoordinate(LogUniformPrior(bounds))
        for name, bounds in BOUNDS.items()
    }
    template = dict.fromkeys(scalars, 0.0)
    if rv is not None:
        scalars["jitter_m_per_s"] = ScalarCoordinate(
            HalfNormalPrior(0.8), logarithmic=False
        )
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


@pytest.mark.medium(reason="Differentiates both chart and normalized priors on a cube.")
@pytest.mark.parametrize("name", ["kipping13", "rayleigh", "vaneylen19", "disk"])
def test_chart_nested_cube_matches_numpyro(name):
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
    old = build_model(
        has_rv=True,
        sample_planet_scale=True,
        jitter_scale=0.8,
        ecc_prior=name,
        log_P_range=(1.0, 4.0),
    )
    names = (
        "log_P",
        "e_raw",
        "w_raw",
        "cos_i",
        "W",
        "M0",
        "log_Ag",
        "jitter",
        "log_Mp",
        "log_Rp",
    )
    old = reparam(old, config=dict.fromkeys(names, _UnitCubeReparam()))
    physical = ("T", "a", "e", "cos_i", "W", "cos_w", "sin_w", "tp", "Ag", "Mp", "Rp")

    def compare(unit):
        parts = [*unit[:7], unit[7:10], *unit[10:]]
        cube = {
            f"{k}_base": jnp.atleast_1d(v) for k, v in zip(names, parts, strict=True)
        }
        reference, trace = log_density(old, (Msun2kg, 10.0, OrbitData(rv=rv)), {}, cube)
        realized = nested_model.apply(None, {"U": ScopedDict({"unit": unit})})
        z = realized.collections["X"]["z"]
        decoded = problem.to_physical(z)
        actual = jnp.stack([decoded[k] for k in physical])
        expected = jnp.stack([trace[k]["value"].squeeze() for k in physical])
        actual = jnp.concatenate((actual, decoded["jitter_m_per_s"]))
        expected = jnp.concatenate((expected, trace["jitter"]["value"]))
        likelihood = realized.fn_val
        logdet = jnp.linalg.slogdet(jax.jacfwd(transform.forward)(unit))[1]
        return actual, expected, likelihood, reference, problem.logdensity(z), logdet

    points = (
        jnp.arange(12) * 0.037 + jnp.array([0.08, 0.33, 0.53, 0.75])[:, None]
    ) % 0.94 + 0.02
    actual, expected, likelihood, reference, joint, logdet = eqx.filter_jit(
        jax.vmap(compare)
    )(points)
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
    assert np.isfinite(jax.vmap(problem.logprior)(post.samples)).all()
