"""End-to-end custom likelihood inference across independent sampler routes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg

from photomancy.backends import NUTSBackend
from photomancy.orbit.data import NullData, RelativeAstromData
from photomancy.orbit.inference import build_orbit_logdensity
from photomancy.orbit.nested import orbit_nested_sampling
from photomancy.posterior import SamplePosterior

jax.config.update("jax_enable_x64", True)


def _toy_loglike(params):
    """Independent Gaussian observations of log radius and log albedo."""
    data = params["custom_data"]
    x = jnp.array([jnp.log10(params["Rp"]), jnp.log10(params["Ag"])])
    return -0.5 * jnp.sum(((x - data["mean"]) / data["sigma"]) ** 2)


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
    assert 1 < float(diag["weight_ess"]) <= 6000
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
