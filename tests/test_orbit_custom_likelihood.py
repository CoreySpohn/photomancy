"""Custom orbit likelihoods and independent planet-scale priors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
from scipy.stats import kstest

from photomancy.orbit.data import NullData, OrbitData, RelativeAstromData
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from photomancy.orbit.inference import build_orbit_logdensity
from photomancy.orbit.likelihoods import loglike_null, loglike_relative_astrom
from photomancy.orbit.model import build_model

jax.config.update("jax_enable_x64", True)


def test_planet_scale_prior_only_is_log_uniform():
    """Fails if data-free scales are rejected or their logarithms are not uniform.

    With 1024 draws, the KS 1% cutoff detects a CDF error of about 0.051.
    No existing test covers prior-only orbit models.
    """
    ranges = {"log_Rp": (-4.8, -3.1), "log_Ag": (-1.7, -0.2)}
    model = build_model(
        sample_planet_scale=True,
        log_Rp_range=ranges["log_Rp"],
        log_Ag_range=ranges["log_Ag"],
    )
    draws = Predictive(model, num_samples=1024)(
        jax.random.key(0), Msun2kg, 10.0, OrbitData()
    )
    for name, (low, high) in ranges.items():
        uniform = (np.asarray(draws[name]).ravel() - low) / (high - low)
        assert kstest(uniform, "uniform").pvalue > 0.01
    np.testing.assert_allclose(draws["Lambda"], draws["Ag"] * draws["Rp"] ** 2)
    assert np.all(draws["Mp"] > 0)
    np.testing.assert_array_equal(draws["ll_total"], 0.0)


@pytest.mark.parametrize(
    "channel",
    ["rv", "relative_astrom", "stellar_astrom", "pm_anomaly", "null", "imaging"],
)
def test_custom_overlap_requires_explicit_opt_in(channel):
    """Fails if any built-in channel can silently double count custom data.

    Six distinct channels share this construction guard; no previous guard exists.
    """
    config = {f"has_{channel}": True, "custom_loglike": lambda p: 0.0}
    with pytest.raises(ValueError, match="allow_overlap=True"):
        build_model(**config)
    build_model(**config, allow_overlap=True)


def test_custom_multiplanet_callback_runs_once_and_requires_scalar():
    """Fails if the whole-data factor is repeated per planet or accepts a vector.

    Existing multi-planet tests only exercise built-in forward models.
    """
    model = build_model(
        n_planets=2,
        custom_data={"scale": jnp.array([2.0, 3.0])},
        custom_loglike=lambda p: -jnp.sum(p["Ag"] * p["custom_data"]["scale"]),
    )
    sites = trace(seed(model, rng_seed=0)).get_trace(Msun2kg, 10.0, OrbitData())
    expected = -jnp.sum(sites["Ag"]["value"] * jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(sites["ll_custom"]["fn"].log_factor, expected)
    bad_model = build_model(custom_loglike=lambda p: jnp.ones(1))
    with pytest.raises(ValueError, match="scalar"):
        trace(seed(bad_model, rng_seed=0)).get_trace(Msun2kg, 10.0, OrbitData())


def _custom_null(params):
    """Use the same forward predictions and flux-space null law as the model."""
    data = params["custom_data"]
    alpha, dmag = predict_photometry(
        data.epochs,
        *(params[k] for k in ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")),
        params["Ms"],
        params["Lambda"],
        params["dist_pc"],
        Mp=params["Mp"],
    )
    return loglike_null(alpha, dmag, data)


@pytest.mark.medium
def test_custom_null_matches_builtin_logdensity():
    """Fails if custom physical parameters, prior factors or null terms differ.

    No existing test compares a caller-defined likelihood with a built-in law.
    """
    data = NullData.pad(
        epochs=jnp.array([0.0, 100.0]),
        sep_grid=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
        dmag0_grid=jnp.full((2, 2), 23.0),
    )
    config = dict(sample_planet_scale=True, ecc_prior="disk")
    builtin = build_orbit_logdensity(Msun2kg, 10.0, null_data=data, **config)
    custom = build_orbit_logdensity(
        Msun2kg, 10.0, custom_loglike=_custom_null, custom_data=data, **config
    )
    assert builtin.param_names == custom.param_names
    positions = jax.random.normal(jax.random.key(0), (8, len(custom.param_names)))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(custom.logdensity))(positions),
        jax.jit(jax.vmap(builtin.logdensity))(positions),
        rtol=0.0,
        atol=1e-10,
    )


def _custom_astrom(params):
    """Average sky positions at three exposure nodes before evaluating the law."""
    data, offsets = params["custom_data"]
    ra, dec = predict_relative_astrometry(
        (data.times[:, None] + offsets).ravel(),
        *(params[k] for k in ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")),
        params["Ms"],
        params["dist_pc"],
        Mp=params["Mp"],
    )
    return loglike_relative_astrom(
        ra.reshape(-1, 3).mean(axis=1), dec.reshape(-1, 3).mean(axis=1), data
    )


@pytest.mark.medium
def test_custom_astrometry_is_prior_plus_custom_and_data_are_isolated():
    """Fails if absent astrometry adds a term or custom closures share stale data.

    Existing bridge tests only cover instantaneous built-in astrometry.
    """
    data = RelativeAstromData(
        times=jnp.array([100.0]),
        ra=jnp.array([0.03]),
        dec=jnp.array([-0.02]),
        ra_err=jnp.array([0.02]),
        dec_err=jnp.array([0.02]),
        corr=jnp.zeros(1),
        planet_id=jnp.zeros(1, dtype=int),
        is_valid=jnp.ones(1, dtype=bool),
    )
    bound_data = (data, jnp.array([-10.0, 0.0, 10.0]))
    config = dict(ecc_prior="disk", sample_planet_scale=True)
    prior = build_orbit_logdensity(Msun2kg, 10.0, **config)
    custom = build_orbit_logdensity(
        Msun2kg,
        10.0,
        relative_astrom_data=None,
        custom_loglike=_custom_astrom,
        custom_data=bound_data,
        **config,
    )
    z = jax.random.normal(jax.random.key(0), (len(custom.param_names),))
    params = dict(
        custom.to_physical(z), Ms=Msun2kg, dist_pc=10.0, custom_data=bound_data
    )
    np.testing.assert_allclose(
        custom.logdensity(z),
        prior.logdensity(z) + _custom_astrom(params),
        rtol=0.0,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="allow_overlap=True"):
        build_orbit_logdensity(
            Msun2kg, 10.0, relative_astrom_data=data, custom_loglike=_custom_astrom
        )
    overlap = build_orbit_logdensity(
        Msun2kg,
        10.0,
        relative_astrom_data=data,
        custom_loglike=_custom_astrom,
        custom_data=bound_data,
        allow_overlap=True,
        **config,
    )
    builtin = build_orbit_logdensity(Msun2kg, 10.0, relative_astrom_data=data, **config)
    np.testing.assert_allclose(
        overlap.logdensity(z),
        builtin.logdensity(z) + _custom_astrom(params),
        rtol=0.0,
        atol=1e-12,
    )
    shifted_data = (data, bound_data[1] + 50.0)
    other = build_orbit_logdensity(
        Msun2kg,
        10.0,
        custom_loglike=_custom_astrom,
        custom_data=shifted_data,
        **config,
    )
    np.testing.assert_allclose(
        other.logdensity(z),
        prior.logdensity(z) + _custom_astrom(dict(params, custom_data=shifted_data)),
        rtol=0.0,
        atol=1e-12,
    )
    assert abs(float(other.logdensity(z) - custom.logdensity(z))) > 1e-6
    np.testing.assert_allclose(
        custom.logdensity(z),
        prior.logdensity(z) + _custom_astrom(params),
        rtol=0.0,
        atol=1e-12,
    )
