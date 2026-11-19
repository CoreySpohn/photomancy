"""Custom-term inference on the chart nested route and the generic backend."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from jaxctx import ScopedDict
from orbix.system import EccentricityVectorChart
from priorities import LogUniform
from test_nested import _scene_orbit
from test_system_priors import BOUNDS, _scene

from photomancy import ScalarCoordinate
from photomancy.infer.backends import NUTSBackend
from photomancy.infer.posterior import SamplePosterior
from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity
from photomancy.orbit.nested import OrbitPriorTransform, orbit_nested_sampling
from photomancy.priors import FactorizedOrbitPrior

jax.config.update("jax_enable_x64", True)

from jaxns.algorithm.depth import _depth_condition_reached  # noqa: E402
from jaxns.algorithm.initialisation import _build_init_state  # noqa: E402
from jaxns.core import NestedSampler  # noqa: E402

_MEAN = jnp.array([-3.8, -0.8])


def _gaussian_term(mean, sigma):
    """Independent Gaussian observations of log10 radius (AU) and log10 albedo."""

    def term(scene, *, radius_AU, albedo):
        x = jnp.array([jnp.log10(radius_AU), jnp.log10(albedo)])
        return -0.5 * jnp.sum(((x - mean) / sigma) ** 2), None

    return term


def _problem(sigma):
    """Sampleable chart problem with the custom Gaussian term on the two scales."""
    prior = FactorizedOrbitPrior(_scene_orbit, (10.0, 10000.0), "disk", 1.0)
    scalars = {
        name: ScalarCoordinate(LogUniform(*bounds)) for name, bounds in BOUNDS.items()
    }
    chart = EccentricityVectorChart()
    density = build_chart_logdensity(
        chart,
        _scene,
        prior,
        star_mass_kg=Msun2kg,
        terms=(_gaussian_term(_MEAN, sigma),),
        scalars=scalars,
        body_inputs=("radius_AU", "albedo"),
        term_inputs={0: ("radius_AU", "albedo")},
    )
    template = {"orbit": jnp.zeros(chart.n_coords)} | dict.fromkeys(scalars, 0.0)
    return ChartOrbitProblem(density, template)


@pytest.fixture
def capped_nested_run(monkeypatch):
    """Isolate diagnostics with a real jaxns root state of eight stored samples.

    The full sampler is covered by the Gaussian-posterior test below; fixed unit
    roots avoid compiling the constrained sampler.
    """
    problem = _problem(jnp.ones(2))
    ndim = OrbitPriorTransform(problem).ndim
    captured = {}

    def capture(self, *, key, args):
        ramp = jnp.linspace(0.1, 0.9, 8, dtype=jnp.float32)
        unit = ramp[:, None] * jnp.ones((8, ndim), dtype=jnp.float32)
        loglike = -0.5 * ((jnp.linspace(0.1, 0.9, 8) - 0.5) / 0.3) ** 2
        state = _build_init_state(
            self.model,
            args,
            None,
            ScopedDict({"unit": unit}),
            loglike,
            jnp.ones(8, dtype=jnp.int64),
            sample_capacity=8,
            num_phantom=0,
        )
        state = replace(state, termination_reason=jnp.array(1))
        captured.update(state=state, condition=self.depth_condition)
        return state

    monkeypatch.setattr(NestedSampler, "run", capture)
    post = orbit_nested_sampling(
        problem,
        num_live_points=8,
        max_samples=8,
        num_samples=16,
        evidence_tolerance=0.9,
        key=jax.random.key(0),
    )
    return post, captured["state"], captured["condition"]


def test_nested_convergence_evaluates_evidence_at_storage_cap(capped_nested_run):
    """Fails for the termination-code-only convergence mutant.

    Covers a cap reached together with a satisfied evidence criterion (the
    capped-run test below covers a cap before convergence). Eight roots suffice.
    """
    post, state, condition = capped_nested_run
    assert int(state.num_samples) == 8
    assert int(post.diagnostics["termination_reason"]) != 0
    assert bool(_depth_condition_reached(state, condition))
    assert bool(post.diagnostics["converged"])


def test_nested_weight_ess_uses_importance_weights_before_resampling(capped_nested_run):
    """Fails for the ESS-after-resampling mutant with uniform posterior weights.

    The reported ESS equals the raw Kish ESS of the importance weights (1e-13).
    """
    post, state, _ = capped_nested_run
    log_importance_weights = state.to_result().trim().log_dp
    expected = 1.0 / jnp.sum(jax.nn.softmax(log_importance_weights) ** 2)
    np.testing.assert_allclose(post.diagnostics["weight_ess"], expected, rtol=1e-13)
    assert float(expected) < 8.0
    assert float(expected) != pytest.approx(len(post.samples))


@pytest.mark.medium(reason="NUTS and nested on one nine-dimensional chart, 25 s cold")
def test_custom_nested_and_generic_backend_agree_with_gaussian_posterior():
    """Fails if nested drops the custom term, the scale columns or uses other priors.

    The two observed parameters have Gaussian posteriors with negligible
    truncation. 1024 MCMC draws resolve 0.125 posterior sigma at four standard
    errors for independent draws; batch means account for chain correlation.
    Nested mean errors use the smaller of weight ESS and the distinct resampled
    count.
    """
    sigma = jnp.array([0.18, 0.13])
    problem = _problem(sigma)
    transform = OrbitPriorTransform(problem)
    mcmc_key, nested_key, init_key = jax.random.split(jax.random.key(0), 3)
    init = transform.sample(init_key, 1)[0]
    mcmc = problem.run(NUTSBackend(n_warmup=512, n_samples=1024), init, mcmc_key)
    nested = orbit_nested_sampling(
        problem,
        num_live_points=64,
        max_samples=6000,
        num_samples=1024,
        evidence_tolerance=1e-3,
        key=nested_key,
    )
    assert isinstance(nested, SamplePosterior)
    assert nested.param_names == problem.param_names
    problem.require(nested)

    def log10_scales(z):
        scales = jax.vmap(problem.unflatten)(z)
        return np.column_stack([scales["radius_AU"], scales["albedo"]]) / np.log(10.0)

    phys = jax.vmap(problem.to_physical)(nested.samples)
    np.testing.assert_allclose(phys["Lambda"], phys["Ag"] * phys["Rp"] ** 2)
    mcmc_x, nested_x = log10_scales(mcmc.samples), log10_scales(nested.samples)
    mcmc_se = mcmc_x.reshape(16, 64, 2).mean(axis=1).std(axis=0, ddof=1) / 4.0
    diag = nested.diagnostics
    nested_neff = min(float(diag["weight_ess"]), int(diag["num_distinct"]), 1024)
    nested_se = nested_x.std(axis=0, ddof=1) / np.sqrt(nested_neff)
    assert np.all(
        np.abs(mcmc_x.mean(axis=0) - nested_x.mean(axis=0))
        < 4 * np.sqrt(mcmc_se**2 + nested_se**2)
    )
    assert np.all(np.abs(mcmc_x.mean(axis=0) - _MEAN) < 4 * mcmc_se)
    assert np.all(np.abs(nested_x.mean(axis=0) - _MEAN) < 4 * nested_se)
    assert bool(diag["converged"])
    assert diag["log_evidence"] == nested.evidence
    # Z = product(sqrt(2*pi)*sigma/prior width in log10), far from either boundary.
    analytic_logz = jnp.sum(
        jnp.log(jnp.sqrt(2 * jnp.pi) * sigma / jnp.array([2.5, 2.0]))
    )
    assert abs(float(nested.evidence - analytic_logz)) < 4 * float(
        diag["log_evidence_error"]
    )
    assert float(diag["log_evidence_error"]) > 0
    assert int(diag["num_iterations"]) > 0
    assert (
        int(diag["num_distinct"])
        == np.unique(np.asarray(nested.samples), axis=0).shape[0]
    )


@pytest.mark.medium(reason="two capped nested runs, 10 s cold")
def test_nested_retains_scale_columns_and_reports_stopping_reason():
    """Fails if nested loses scale columns or calls a storage cap convergence.

    A wide Gaussian (sigma 100) is flat over the prior, so its evidence is the
    normalized prior (log Z = 0); a cap of eight stored samples must report no
    convergence and a nonzero termination reason.
    """
    flat = _problem(jnp.full(2, 100.0))
    post = orbit_nested_sampling(
        flat,
        num_live_points=8,
        max_samples=512,
        num_samples=32,
        key=jax.random.key(0),
    )
    assert post.param_names == flat.param_names
    assert {"radius_AU", "albedo", "mass_Mearth"} <= set(post.param_names)
    assert abs(float(post.evidence)) < (
        4 * float(post.diagnostics["log_evidence_error"]) + 1e-3
    )
    capped = orbit_nested_sampling(
        _problem(jnp.ones(2)),
        num_live_points=8,
        max_samples=8,
        num_samples=32,
        key=jax.random.key(0),
    )
    assert not bool(capped.diagnostics["converged"])
    assert int(capped.diagnostics["termination_reason"]) != 0
    assert int(capped.diagnostics["num_distinct"]) <= 8
