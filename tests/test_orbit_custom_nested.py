"""End-to-end custom likelihood inference across independent sampler routes."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from hwoutils.constants import Msun2kg
from jaxctx import ScopedDict, transform

from photomancy.backends import NUTSBackend
from photomancy.orbit import _numpyro_bridge
from photomancy.orbit import inference as inference_module
from photomancy.orbit import nested as nested_module
from photomancy.orbit.data import NullData, RelativeAstromData
from photomancy.orbit.inference import build_orbit_logdensity
from photomancy.orbit.nested import orbit_nested_sampling
from photomancy.posterior import SamplePosterior

jax.config.update("jax_enable_x64", True)

from jaxns.algorithm.depth import _depth_condition_reached  # noqa: E402
from jaxns.algorithm.initialisation import _build_init_state  # noqa: E402
from jaxns.core import NestedSampler  # noqa: E402


def _toy_loglike(params):
    """Independent Gaussian observations of log radius and log albedo."""
    data = params["custom_data"]
    x = jnp.array([jnp.log10(params["Rp"]), jnp.log10(params["Ag"])])
    return -0.5 * jnp.sum(((x - data["mean"]) / data["sigma"]) ** 2)


@pytest.fixture
def capped_nested_run(monkeypatch):
    """Isolate diagnostics with a one-site model and a real jaxns root state.

    The full orbit model is covered by the sampler integration test below;
    fixed radius roots avoid compiling the unrelated constrained sampler.
    """

    def build_diagnostic_model(**config):
        def model(Ms, dist_pc, data):
            log_rp = numpyro.sample("log_Rp", dist.Uniform(-5.0, -2.5).expand((1,)))
            params = {"Rp": 10.0**log_rp, "Ag": jnp.full(1, 0.2)}
            params["Lambda"] = params["Ag"] * params["Rp"] ** 2
            for name in ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp"):
                numpyro.deterministic(name, jnp.zeros(1))
            for name, value in params.items():
                numpyro.deterministic(name, value)
            numpyro.factor(
                "ll_custom",
                config["custom_loglike"](
                    {k: jnp.squeeze(v) for k, v in params.items()}
                    | {"custom_data": config["custom_data"]}
                ),
            )

        return model

    monkeypatch.setattr(_numpyro_bridge, "build_model", build_diagnostic_model)
    # The old independent path must receive the same reduced model as well.
    monkeypatch.setattr(
        nested_module, "build_model", build_diagnostic_model, raising=False
    )
    captured = {}

    def capture(self, *, key, args):
        unit = jnp.linspace(0.1, 0.9, 8, dtype=jnp.float32)[:, None]
        log_radius = -5.0 + 2.5 * unit[:, 0].astype(jnp.float64)
        loglike = -0.5 * ((log_radius + 3.8) ** 2 + (jnp.log10(0.2) + 0.8) ** 2)
        state = _build_init_state(
            self.model,
            args,
            None,
            ScopedDict({"log_Rp_base": unit}),
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
        Msun2kg,
        10.0,
        custom_loglike=_toy_loglike,
        custom_data={"mean": jnp.array([-3.8, -0.8]), "sigma": jnp.ones(2)},
        ecc_prior="disk",
        num_live_points=8,
        max_samples=8,
        num_samples=16,
        evidence_tolerance=0.9,
        key=jax.random.key(0),
    )
    return post, captured["state"], captured["condition"]


def test_nested_convergence_evaluates_evidence_at_storage_cap(capped_nested_run):
    """Fails for the termination-code-only convergence mutant.

    The existing capped test covers a cap before convergence, not a simultaneous
    cap and satisfied evidence criterion. Eight roots suffice for this boundary.
    """
    post, state, condition = capped_nested_run
    assert int(state.num_samples) == 8
    assert int(post.diagnostics["termination_reason"]) != 0
    assert bool(_depth_condition_reached(state, condition))
    assert bool(post.diagnostics["converged"])


def test_nested_weight_ess_uses_importance_weights_before_resampling(capped_nested_run):
    """Fails for the ESS-after-resampling mutant with uniform posterior weights.

    Replaces the range-only ESS guard with equality to the raw Kish ESS.
    """
    post, state, _ = capped_nested_run
    log_importance_weights = state.to_result().trim().log_dp
    expected = 1.0 / jnp.sum(jax.nn.softmax(log_importance_weights) ** 2)
    np.testing.assert_allclose(post.diagnostics["weight_ess"], expected, rtol=1e-13)
    assert float(expected) < 8.0
    assert float(expected) != pytest.approx(len(post.samples))


def test_nested_and_logdensity_share_model_construction(monkeypatch):
    """Fails for the independent-construction mutant even when densities agree.

    Also retains the 7e-15 density agreement guard after the cube Jacobian.
    Stops before sampling so construction has a unit-test compile budget.
    """
    constructed = []
    build = _numpyro_bridge._build_orbit_model

    def record_build(*args, **kwargs):
        kwargs.setdefault("seed", 0)
        constructed.append((args, kwargs))
        return build(*args, **kwargs)

    captured = {}

    class ConstructionCaptured(Exception):
        pass

    def capture_run(self, *, key, args):
        captured.update(model=self.model, args=args)
        raise ConstructionCaptured

    monkeypatch.setattr(
        inference_module, "_build_orbit_model", record_build, raising=False
    )
    monkeypatch.setattr(
        nested_module, "_build_orbit_model", record_build, raising=False
    )
    monkeypatch.setattr(NestedSampler, "run", capture_run)
    config = dict(
        custom_loglike=_toy_loglike,
        custom_data={"mean": jnp.array([-3.8, -0.8]), "sigma": jnp.ones(2)},
        sample_planet_scale=True,
        log_P_range=(2.0, 3.0),
        log_Mp_range=(-1.0, 2.0),
        log_Rp_range=(-4.5, -3.0),
        log_Ag_range=(-1.8, -0.1),
        ecc_prior="disk",
    )
    problem = build_orbit_logdensity(Msun2kg, 10.0, **config)
    with pytest.raises(ConstructionCaptured):
        orbit_nested_sampling(Msun2kg, 10.0, **config, key=jax.random.key(0))
    assert len(constructed) == 2
    assert constructed[0] == constructed[1]
    z = jnp.linspace(-0.7, 0.8, len(problem.param_names))
    unit = {
        f"{name}_base": jax.nn.sigmoid(raw).reshape(-1)
        for name, raw in problem.unflatten(z).items()
    }
    cube_loglike = (
        transform(captured["model"].prior_model, base_dtype=jnp.float64)
        .apply(None, {"U": ScopedDict(unit)}, *captured["args"])
        .fn_val
    )
    # All priors in this disk-eccentricity model are uniform: their widths
    # cancel the support transforms, leaving the logistic cube Jacobian.
    log_jac = jnp.sum(jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z))
    np.testing.assert_allclose(
        problem.logdensity(z), cube_loglike + log_jac, rtol=0.0, atol=7e-15
    )


@pytest.mark.medium
def test_custom_nested_and_generic_backend_agree_with_gaussian_posterior():
    """Fails if nested drops custom terms, scale columns, or uses different priors.

    No existing test compares the custom nested and logdensity routes. The two
    observed parameters have Gaussian posteriors with negligible truncation.
    1024 MCMC draws resolve 0.125 posterior sigma at four standard errors for
    independent draws; batch means account for chain correlation. Nested mean
    errors use the smaller of weight ESS and the distinct resampled count.
    """
    data = {"mean": jnp.array([-3.8, -0.8]), "sigma": jnp.array([0.18, 0.13])}
    config = dict(custom_loglike=_toy_loglike, custom_data=data, ecc_prior="disk")
    problem = build_orbit_logdensity(Msun2kg, 10.0, **config)
    init = problem.init_to_z({})
    mcmc_key, nested_key = jax.random.split(jax.random.key(0))
    mcmc = NUTSBackend(n_warmup=512, n_samples=1024).run(
        problem.logdensity, init, key=mcmc_key
    )
    nested = orbit_nested_sampling(
        Msun2kg,
        10.0,
        **config,
        num_live_points=64,
        max_samples=6000,
        num_samples=1024,
        evidence_tolerance=1e-3,
        key=nested_key,
    )
    assert isinstance(nested, SamplePosterior)
    assert {"Mp", "Rp", "Ag", "Lambda"} <= set(nested.param_names)
    cols = {name: nested.samples[:, i] for i, name in enumerate(nested.param_names)}
    np.testing.assert_allclose(cols["Lambda"], cols["Ag"] * cols["Rp"] ** 2)
    phys = jax.vmap(problem.to_physical)(mcmc.samples)
    mcmc_x = np.log10(np.column_stack([phys["Rp"], phys["Ag"]]))
    nested_x = np.log10(np.column_stack([cols["Rp"], cols["Ag"]]))
    mcmc_se = mcmc_x.reshape(16, 64, 2).mean(axis=1).std(axis=0, ddof=1) / 4.0
    diag = nested.diagnostics
    nested_neff = min(float(diag["weight_ess"]), int(diag["num_distinct"]), 1024)
    nested_se = nested_x.std(axis=0, ddof=1) / np.sqrt(nested_neff)
    assert np.all(
        np.abs(mcmc_x.mean(axis=0) - nested_x.mean(axis=0))
        < 4 * np.sqrt(mcmc_se**2 + nested_se**2)
    )
    assert np.all(np.abs(mcmc_x.mean(axis=0) - data["mean"]) < 4 * mcmc_se)
    assert np.all(np.abs(nested_x.mean(axis=0) - data["mean"]) < 4 * nested_se)
    assert bool(diag["converged"])
    assert diag["log_evidence"] == nested.evidence
    # Z = product(sqrt(2*pi)*sigma/prior width), far from either boundary.
    analytic_logz = jnp.sum(
        jnp.log(jnp.sqrt(2 * jnp.pi) * data["sigma"] / jnp.array([2.5, 2.0]))
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


@pytest.mark.medium
@pytest.mark.parametrize("mode", ["prior", "photometry", "capped"])
def test_nested_retains_photometric_columns_and_reports_stopping_reason(mode):
    """Fails if nested loses scales or calls a storage cap evidence convergence.

    Existing nested tests require built-in data and do not check diagnostics.
    """
    configs = dict(
        prior=dict(sample_planet_scale=True),
        photometry=dict(null_data=NullData.zeros()),
        capped=dict(
            custom_loglike=_toy_loglike,
            custom_data={"mean": jnp.array([-3.8, -0.8]), "sigma": jnp.ones(2)},
            null_data=NullData.zeros(),
            allow_overlap=True,
        ),
    )
    post = orbit_nested_sampling(
        Msun2kg,
        10.0,
        **configs[mode],
        ecc_prior="disk",
        num_live_points=8,
        max_samples=8 if mode == "capped" else 512,
        num_samples=32,
        key=jax.random.key(0),
    )
    assert {"Rp", "Ag", "Lambda"} <= set(post.param_names)
    if mode != "photometry":
        assert "Mp" in post.param_names
    if mode != "capped":
        # A flat likelihood can reach the storage cap before the tail cutoff;
        # its evidence must still be consistent with the normalized prior.
        assert abs(float(post.evidence)) < (
            4 * float(post.diagnostics["log_evidence_error"]) + 1e-3
        )
    else:
        assert not bool(post.diagnostics["converged"])
        assert int(post.diagnostics["termination_reason"]) != 0
        assert int(post.diagnostics["num_distinct"]) <= 8


@pytest.mark.parametrize("channel", ["relative_astrom", "null"])
def test_nested_custom_overlap_is_rejected(channel):
    """Fails if the direct nested model path bypasses the overlap safety guard.

    Astrometric and photometric channels cover the two caller data families.
    """
    data = (
        RelativeAstromData.zeros() if channel == "relative_astrom" else NullData.zeros()
    )
    with pytest.raises(ValueError, match="allow_overlap=True"):
        orbit_nested_sampling(
            Msun2kg,
            10.0,
            **{f"{channel}_data": data},
            custom_loglike=_toy_loglike,
            key=jax.random.key(0),
        )
