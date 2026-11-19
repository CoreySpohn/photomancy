"""The product prior over the flat vector of fitted leaves."""

import jax
import jax.numpy as jnp
import pytest
from priorities import (
    HalfNormal,
    Mixture,
    MultivariateNormal,
    Normal,
    Uniform,
)

from photomancy.infer.priors import IndependentPrior, flat_prior

jax.config.update("jax_enable_x64", True)


def _prior():
    """Scalar, batched and event distributions over a five-entry vector."""
    return IndependentPrior(
        (
            HalfNormal(0.8),
            Uniform(jnp.array([-2.0, 1.0]), jnp.array([3.0, 5.0])),
            MultivariateNormal(
                jnp.array([1.0, -2.0]), jnp.array([[1.0, 0.0], [0.5, 0.7]])
            ),
        )
    )


def test_blocks_sum_and_round_trip():
    """Each distribution scores its own block; the inverse CDF inverts per block."""
    prior = _prior()
    assert prior.ndim == 5
    assert prior.has_icdf
    z = jnp.array([0.4, 0.5, 2.0, 0.3, -1.5])
    half_normal, uniform, mvn = prior.distributions
    expected = half_normal.log_prob(z[0]) + uniform.log_prob(z[1:3]).sum()
    expected = expected + mvn.log_prob(z[3:])
    assert jnp.allclose(prior.log_prob(z), expected)
    assert prior.log_prob(z.at[1].set(10.0)) == -jnp.inf
    unit = jnp.array([0.2, 0.25, 0.75, 0.6, 0.9])
    assert jnp.allclose(prior.icdf(unit)[1:3], jnp.array([-0.75, 4.0]))
    assert jnp.allclose(prior.cdf(prior.icdf(unit)), unit, atol=1e-8)


def test_batches_and_draws():
    """Leading axes pass through the density, the draws and the inverse CDF."""
    prior = _prior()
    draws = prior.sample(jax.random.key(0), (7,))
    assert draws.shape == (7, 5)
    assert prior.log_prob(draws).shape == (7,)
    assert jnp.allclose(prior.log_prob(draws)[3], prior.log_prob(draws[3]))
    assert jnp.allclose(prior.icdf(prior.cdf(draws)), draws, atol=1e-8)
    assert prior.sample(jax.random.key(1)).shape == (5,)


def test_bare_distribution_is_wrapped_and_mixture_has_no_icdf():
    """One distribution over z becomes a one-block prior; a mixture has no icdf."""
    components = MultivariateNormal(
        jnp.array([[0.0, 0.0], [5.0, 5.0]]), jnp.stack([jnp.eye(2), jnp.eye(2)])
    )
    mixture = Mixture(jnp.log(jnp.array([0.7, 0.3])), components)
    prior = flat_prior(mixture)
    assert isinstance(prior, IndependentPrior)
    assert prior.ndim == 2
    assert not prior.has_icdf
    z = jnp.array([0.3, -0.2])
    assert jnp.allclose(prior.log_prob(z), mixture.log_prob(z))
    callable_prior = jnp.sum
    assert flat_prior(callable_prior) is callable_prior


def test_matrix_shaped_distribution_is_rejected():
    """A block is a scalar or one axis, so a matrix of entries has no place in z."""
    with pytest.raises(ValueError, match="flat vector"):
        IndependentPrior((Normal(jnp.zeros((2, 3)), 1.0),))
