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
