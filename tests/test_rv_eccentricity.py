"""Bounded eccentricity priors retain the nested sampler's unit-cube interface."""

import jax
import jax.numpy as jnp
import pytest
from skyscapes.population import rayleigh_eccentricity

from photomancy.priors import FactorizedOrbitPrior

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name,sigma", [("rayleigh", 0.3), ("vaneylen19", 0.049)])
def test_truncated_eccentricity_prior_supports_nested_inverse_cdf(name, sigma):
    """Fails if the unit-cube quantile or the direct draws leave the truncated law.

    The quantile is the closed-form truncated Rayleigh inverse CDF to rtol 1e-12;
    direct draws lie in [0, 1).
    """
    q = jnp.array([0.1, 0.5, 0.99])
    normalizer = -jnp.expm1(-1.0 / (2.0 * sigma**2))
    expected = sigma * jnp.sqrt(-2.0 * jnp.log1p(-q * normalizer))
    prior = FactorizedOrbitPrior(
        lambda system: system, eccentricity=rayleigh_eccentricity(sigma, 1.0)
    )
    assert jnp.allclose(prior.eccentricity_icdf(q), expected, rtol=1e-12)
    draws = prior.eccentricity.sample(jax.random.key(0), (64,))
    assert bool(jnp.all((draws >= 0) & (draws < 1)))
