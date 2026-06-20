"""Nested-sampling backend (jaxns): Bayesian evidence + the scene-model adapter."""

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from photomancy.backends import JaxnsBackend, build_scene_nested_model

tfpd = tfp.distributions


class _Toy(eqx.Module):
    theta: jnp.ndarray
    label: str = eqx.field(static=True)


def test_jaxns_recovers_analytic_evidence():
    """JaxnsBackend computes the conjugate-Gaussian log Z (mirrors SMC-vs-analytic)."""
    from jaxns.framework.model import Model
    from jaxns.framework.prior import Prior

    data, lik_s, pri_s = 2.0, 0.5, 1.0
    # data = x + noise -> data ~ N(0, sqrt(pri_s^2 + lik_s^2)); analytic evidence below
    analytic = float(tfpd.Normal(0.0, jnp.sqrt(pri_s**2 + lik_s**2)).log_prob(data))

    def prior_model():
        x = yield Prior(tfpd.Normal(0.0, pri_s), name="x")
        return x

    def log_likelihood(x):
        return tfpd.Normal(data, lik_s).log_prob(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    post = JaxnsBackend(max_samples=2e4).run(model, key=jax.random.key(0))

    assert bool(jnp.isfinite(post.evidence))
    assert abs(float(post.evidence) - analytic) < 0.5  # nested-sampling Z is stochastic
    assert post.samples.shape[1] == 1


def test_build_scene_nested_model_recovers_and_gives_evidence():
    """Scene adapter: a photomancy AbstractPrior over z; recovers truth + log Z."""
    from photomancy.priors import Uniform

    truth = jnp.array([0.3, -0.4])
    sigma = 0.05
    data = truth  # noiseless for a deterministic recovery check
    scene = _Toy(theta=jnp.zeros(2), label="t")

    def forward(s):
        return s.theta

    def likelihood(pred):
        return -0.5 * jnp.sum((pred - data) ** 2 / sigma**2)

    def fit_leaves(s):
        return [s.theta]

    prior = Uniform(low=-jnp.ones(2), high=jnp.ones(2))  # Uniform(-1, 1) over z
    model, unravel = build_scene_nested_model(
        scene, forward, likelihood, fit_leaves=fit_leaves, prior=prior
    )
    post = JaxnsBackend(max_samples=4e4).run(model, key=jax.random.key(0))

    assert bool(jnp.isfinite(post.evidence))
    assert post.samples.shape[1] == 2
    # z-space samples reconstruct a scene through the returned unravel.
    assert unravel(post.samples[0]).theta.shape == (2,)
    # posterior mean recovers the truth.
    w = jax.nn.softmax(post.log_weights)
    mean = jnp.sum(post.samples * w[:, None], axis=0)
    assert jnp.allclose(mean, truth, atol=0.06)


def test_jaxns_prior_adapter_recovers_analytic_evidence_tfp_free():
    """A photomancy Normal prior, via the _JaxnsPrior adapter, recovers log Z (no tfp).

    The conjugate-Gaussian check of test_jaxns_recovers_analytic_evidence, but the prior
    is our own ``photomancy.priors.Normal`` wrapped by the adapter inside
    ``build_scene_nested_model`` -- so the whole prior path is tfp-free.
    """
    from photomancy.priors import Normal

    data, lik_s, pri_s = 2.0, 0.5, 1.0
    log2pi = jnp.log(2.0 * jnp.pi)
    # data ~ N(0, sqrt(pri_s^2 + lik_s^2)); analytic evidence (closed form, no tfp).
    var = pri_s**2 + lik_s**2
    analytic = float(-0.5 * (data**2 / var + jnp.log(var) + log2pi))

    scene = _Toy(theta=jnp.zeros(1), label="t")

    def forward(s):
        return s.theta

    def likelihood(pred):
        return -0.5 * ((pred[0] - data) / lik_s) ** 2 - jnp.log(lik_s) - 0.5 * log2pi

    model, _ = build_scene_nested_model(
        scene,
        forward,
        likelihood,
        fit_leaves=lambda s: [s.theta],
        prior=Normal(loc=jnp.zeros(1), scale=jnp.full((1,), pri_s)),
    )
    post = JaxnsBackend(max_samples=2e4).run(model, key=jax.random.key(0))

    assert bool(jnp.isfinite(post.evidence))
    assert abs(float(post.evidence) - analytic) < 0.5  # nested-sampling Z is stochastic
