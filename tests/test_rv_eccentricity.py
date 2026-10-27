"""Bounded eccentricity priors retain the nested sampler's unit-cube interface."""

import jax
import jax.numpy as jnp
import pytest
from hwoutils.constants import Msun2kg
from numpyro import handlers

from photomancy.orbit.data import OrbitData, RVData
from photomancy.orbit.model import build_model
from photomancy.orbit.nested import _UnitCubeReparam
from photomancy.orbit.priors import ecc_distribution

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name,sigma", [("rayleigh", 0.3), ("vaneylen19", 0.049)])
def test_truncated_eccentricity_prior_supports_nested_inverse_cdf(name, sigma):
    """The unit-cube transform has the same normalized prior as direct draws."""
    q = jnp.array([0.1, 0.5, 0.99])
    normalizer = -jnp.expm1(-1.0 / (2.0 * sigma**2))
    expected = sigma * jnp.sqrt(-2.0 * jnp.log1p(-q * normalizer))
    assert jnp.allclose(ecc_distribution(name).icdf(q), expected, rtol=1e-12)
    model = build_model(has_rv=True, ecc_prior=name)
    reparametrized = handlers.reparam(model, config={"e_raw": _UnitCubeReparam()})
    trace = handlers.trace(handlers.seed(reparametrized, 12)).get_trace(
        Msun2kg, 10.0, OrbitData(rv=RVData.zeros(max_n=2))
    )
    assert bool(jnp.all((trace["e"]["value"] >= 0) & (trace["e"]["value"] < 1)))
