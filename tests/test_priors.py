"""photomancy.priors -- the unified prior layer (tfp-free, over the fitted-leaf z)."""

import jax
import jax.numpy as jnp

from photomancy.priors import IndependentPrior, JointPrior, LogNormal, Normal, Uniform

jax.config.update("jax_enable_x64", True)


def test_uniform():
    """Uniform forward/inverse round-trip + log_prob inside and outside the box."""
    p = Uniform(low=jnp.array([-2.0, 1.0]), high=jnp.array([3.0, 5.0]))
    assert p.ndim == 2
    U = jnp.array([0.25, 0.75])
    z = p.forward(U)
    assert jnp.allclose(z, jnp.array([-2.0 + 5.0 * 0.25, 1.0 + 4.0 * 0.75]))
    assert jnp.allclose(p.inverse(z), U)
    assert jnp.allclose(p.log_prob(z), -(jnp.log(5.0) + jnp.log(4.0)))
    assert p.log_prob(jnp.array([10.0, 2.0])) == -jnp.inf  # outside the box


def test_normal():
    """Normal median, inverse-CDF round-trip, and analytic log_prob."""
    p = Normal(loc=jnp.array([1.0]), scale=jnp.array([2.0]))
    assert jnp.allclose(p.forward(jnp.array([0.5])), jnp.array([1.0]))  # median = loc
    U = jnp.array([0.3])
    assert jnp.allclose(p.inverse(p.forward(U)), U, atol=1e-8)
    z = jnp.array([1.5])
    expect = -0.5 * ((1.5 - 1.0) / 2.0) ** 2 - jnp.log(2.0) - 0.5 * jnp.log(2 * jnp.pi)
    assert jnp.allclose(p.log_prob(z), expect)


def test_lognormal():
    """LogNormal stays positive, round-trips, and log_prob at the median."""
    p = LogNormal(loc=jnp.array([0.0]), scale=jnp.array([1.0]))
    z = p.forward(jnp.array([0.7]))
    assert bool(jnp.all(z > 0.0))
    assert jnp.allclose(p.inverse(z), jnp.array([0.7]), atol=1e-8)
    # log-density at the median (z=1, underlying normal at 0): -0.5*log(2pi) - log(1)
    assert jnp.allclose(p.log_prob(jnp.array([1.0])), -0.5 * jnp.log(2 * jnp.pi))


def test_independent():
    """IndependentPrior sums component log_probs and round-trips per block."""
    p = IndependentPrior(
        (
            Uniform(jnp.array([0.0]), jnp.array([1.0])),
            Normal(jnp.array([0.0]), jnp.array([1.0])),
        )
    )
    assert p.ndim == 2
    z = jnp.array([0.5, 0.3])
    expect = p.priors[0].log_prob(z[:1]) + p.priors[1].log_prob(z[1:])
    assert jnp.allclose(p.log_prob(z), expect)
    U = jnp.array([0.2, 0.8])
    assert jnp.allclose(p.inverse(p.forward(U)), U, atol=1e-8)


def test_joint_mvn():
    """JointPrior matches the analytic MVN density, sample moments, and round-trips."""
    mean = jnp.array([1.0, -2.0])
    cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    p = JointPrior(mean=mean, cholesky=jnp.linalg.cholesky(cov))
    assert p.ndim == 2
    z = jnp.array([0.5, -1.0])
    diff = z - mean
    expect = -0.5 * (
        diff @ jnp.linalg.inv(cov) @ diff
        + 2 * jnp.log(2 * jnp.pi)
        + jnp.log(jnp.linalg.det(cov))
    )
    assert jnp.allclose(p.log_prob(z), expect, atol=1e-8)
    s = p.sample(jax.random.key(0), 40000)
    assert jnp.allclose(jnp.mean(s, axis=0), mean, atol=0.05)
    assert jnp.allclose(jnp.cov(s.T), cov, atol=0.1)
    U = jnp.array([0.2, 0.7])
    assert jnp.allclose(p.inverse(p.forward(U)), U, atol=1e-8)
