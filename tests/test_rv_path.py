"""Regression coverage for RV units, masks, priors, and orbital dynamics."""

import jax
import jax.numpy as jnp
import pytest
from hwoutils.constants import AU2m, Mearth2kg, Msun2kg, d2s
from numpyro import handlers

from photomancy.orbit.data import OrbitData, RVData
from photomancy.orbit.forward import predict_rv
from photomancy.orbit.model import build_model

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n_planets", [1, 2])
def test_rv_likelihood_prefers_true_amplitude_in_m_per_s(n_planets):
    """A 10 m/s signal favors its true amplitude over half or twice that mass."""
    times = jnp.linspace(0.0, 800.0, 60)
    periods = jnp.array([300.0, 170.0])[:n_planets]
    masses = jnp.array([100.0, 15.0])[:n_planets]
    clean = sum(
        predict_rv(times, T, Msun2kg, Mp * Mearth2kg, 0.2, 1.0, 0.0, 0.0)
        for T, Mp in zip(periods, masses, strict=True)
    ) * (AU2m / d2s)
    assert 8.0 < float(jnp.max(jnp.abs(clean))) < 15.0
    rv = RVData(
        times=times,
        rv=clean + 17.0,
        rv_err=jnp.full(times.shape, 0.2),
        inst_ids=jnp.zeros(times.shape, dtype=int),
        is_valid=jnp.ones(times.shape, dtype=bool),
        n_inst=1,
    )
    model = build_model(has_rv=True, n_planets=n_planets, ecc_prior="disk")

    def likelihood(scale):
        params = {
            "log_P": jnp.log10(periods),
            "log_Mp": jnp.log10(masses * scale),
            "e_raw": jnp.full(n_planets, 0.2),
            "w_raw": jnp.zeros(n_planets),
            "cos_i": jnp.zeros(n_planets),
            "W": jnp.zeros(n_planets),
            "M0": jnp.zeros(n_planets),
            "jitter": jnp.array([0.01]),
        }
        trace = handlers.trace(
            handlers.substitute(handlers.seed(model, 0), data=params)
        ).get_trace(Msun2kg, 10.0, OrbitData(rv=rv))
        return float(trace["ll_rv"]["fn"].log_factor)

    assert likelihood(1.0) > likelihood(0.5)
    assert likelihood(1.0) > likelihood(2.0)


def test_cached_rv_jitter_matches_instrument_count():
    """Cached templates retain a separate jitter for each instrument."""
    from photomancy.orbit.inference import build_orbit_logdensity

    for n_inst in (1, 2, 1):
        data = RVData(
            times=jnp.array([0.0, 10.0]),
            rv=jnp.array([3.0, -2.0]),
            rv_err=jnp.ones(2),
            inst_ids=jnp.array([0, n_inst - 1]),
            is_valid=jnp.ones(2, dtype=bool),
            n_inst=n_inst,
        )
        problem = build_orbit_logdensity(Msun2kg, 10.0, rv_data=data)
        physical = problem.to_physical(problem.init_to_z({}))
        assert jnp.atleast_1d(physical["jitter"]).shape == (n_inst,)


@pytest.mark.parametrize(
    "channel", ["rv", "relative_astrom", "stellar_astrom", "null", "imaging"]
)
def test_repadding_preserves_channel_validity_mask(channel):
    """Growing any observation channel preserves invalid rows and static shape."""
    import equinox as eqx

    from photomancy.orbit.data import (
        ImagingData,
        NullData,
        RelativeAstromData,
        StellarAstromData,
    )

    classes = {
        "rv": RVData,
        "relative_astrom": RelativeAstromData,
        "stellar_astrom": StellarAstromData,
        "null": NullData,
        "imaging": ImagingData,
    }
    data = classes[channel].zeros(max_n=2)
    data = eqx.tree_at(lambda d: d.is_valid, data, jnp.array([True, False]))
    padded = OrbitData(**{channel: data}).padded()
    result = getattr(padded, channel)
    assert result.is_valid.shape[0] > 2
    assert int(result.is_valid.sum()) == 1
    assert bool(result.is_valid[0])
    assert not bool(result.is_valid[1])
    assert getattr(padded.padded(), channel) is result


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_invalid_rv_rows_have_finite_likelihood_and_gradient(invalid_value):
    """Invalid observations, predictions, errors, and jitters cannot poison sums."""
    from photomancy.orbit.likelihoods import loglike_rv_marginalized

    def evaluate(obs, model, err, jitter):
        return loglike_rv_marginalized(
            obs,
            model,
            err,
            jnp.array([0, 1]),
            2,
            jitter,
            jnp.array([True, False]),
        )

    args = (
        jnp.array([3.0, invalid_value]),
        jnp.array([1.0, invalid_value]),
        jnp.array([0.5, invalid_value]),
        jnp.array([0.2, invalid_value]),
    )
    value, gradients = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1, 2, 3)))(
        *args
    )
    expected = loglike_rv_marginalized(
        args[0][:1],
        args[1][:1],
        args[2][:1],
        jnp.array([0]),
        1,
        args[3][:1],
        jnp.array([True]),
    )
    assert jnp.isfinite(value)
    assert float(value) == pytest.approx(float(expected), abs=1e-14)
    for gradient in gradients:
        assert jnp.all(jnp.isfinite(gradient))
        assert float(gradient[1]) == 0.0


@pytest.mark.parametrize("name", ["rayleigh", "vaneylen19"])
def test_eccentricity_prior_draws_and_density_stay_below_one(name):
    """Both direct draws and log densities exclude unbound eccentricities."""
    from photomancy.orbit.priors import ecc_distribution

    prior = ecc_distribution(name)
    draws = prior.sample(jax.random.key(47), (100_000,))
    assert bool(jnp.all((draws >= 0.0) & (draws < 1.0)))
    assert float(prior.log_prob(1.01)) == -jnp.inf
    assert float(prior.log_prob(1.0)) == -jnp.inf
    assert float(prior.log_prob(-0.01)) == -jnp.inf


def test_rayleigh_prior_has_truncation_normalization():
    """The interior Rayleigh density includes the probability removed above 1."""
    import numpyro.distributions as dist

    from photomancy.orbit.priors import ecc_distribution

    e = jnp.array([0.1, 0.3, 0.8])
    base = dist.Weibull(scale=0.3 * jnp.sqrt(2.0), concentration=2.0)
    expected = base.log_prob(e) - jnp.log1p(-jnp.exp(-1.0 / (2 * 0.3**2)))
    assert jnp.allclose(ecc_distribution("rayleigh").log_prob(e), expected, rtol=1e-12)


def test_single_rv_observation_integrates_to_one():
    """Integrating the sole observation's flat offset leaves log likelihood zero."""
    from photomancy.orbit.likelihoods import loglike_rv_marginalized

    result = loglike_rv_marginalized(
        jnp.array([7.0]),
        jnp.array([-2.0]),
        jnp.ones(1),
        jnp.zeros(1, dtype=int),
        1,
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
    )
    assert float(result) == 0.0


@pytest.mark.parametrize("n_inst", [2, 3])
def test_rv_offset_marginalization_matches_quadrature(n_inst):
    """Three observations match integrating both populated instrument offsets."""
    import numpy as np
    from scipy.integrate import quad

    from photomancy.orbit.likelihoods import loglike_rv_marginalized

    obs = np.array([1.4, -0.3, 2.1])
    model = np.array([0.2, -0.7, 1.0])
    err = np.array([0.7, 0.8, 0.9])
    ids = np.array([0, 0, 1])
    jitters = np.array([0.3, 0.4, 0.5])[:n_inst]
    variance = err**2 + jitters[ids] ** 2
    # The double integral factorizes because the offsets have disjoint rows.
    integral = 1.0
    for instrument in (0, 1):
        selected = ids == instrument

        def gaussian(offset, selected=selected):
            residual = obs[selected] - model[selected] - offset
            return np.exp(-0.5 * np.sum(residual**2 / variance[selected])) / np.sqrt(
                np.prod(2 * np.pi * variance[selected])
            )

        integral *= quad(gaussian, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
    result = loglike_rv_marginalized(
        jnp.asarray(obs),
        jnp.asarray(model),
        jnp.asarray(err),
        jnp.asarray(ids),
        n_inst,
        jnp.asarray(jitters),
        jnp.ones(3, dtype=bool),
    )
    assert float(jnp.exp(result)) == pytest.approx(integral, rel=1e-6)
