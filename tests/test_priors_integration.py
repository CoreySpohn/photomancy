"""Integration: one AbstractPrior across backends, and posterior-as-prior updating.

These exercise the P1 payoff: one photomancy prior drives both the Laplace
(``log_prob``) and jaxns (``_JaxnsPrior``) paths, and a Gaussian posterior folds into
the next fit's prior (``to_prior``) so information accumulates across data batches.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.backends import (
    JaxnsBackend,
    LaplaceBackend,
    build_scene_nested_model,
)
from photomancy.core import build_scene_logdensity
from photomancy.priors import IndependentPrior, Normal

jax.config.update("jax_enable_x64", True)


class _Toy(eqx.Module):
    theta: jnp.ndarray
    label: str = eqx.field(static=True)


def test_same_prior_drives_laplace_and_jaxns_consistently():
    """One IndependentPrior drives a Laplace fit and a jaxns fit (adapter)."""
    data = jnp.array([0.5])
    sigma = 0.3
    scene = _Toy(theta=jnp.zeros(1), label="t")

    def forward(s):
        return s.theta

    def likelihood(pred):
        return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

    prior = IndependentPrior((Normal(loc=jnp.zeros(1), scale=jnp.full((1,), 5.0)),))

    # Laplace path: the prior is folded in as log_prob(z).
    logdensity, z0, _ = build_scene_logdensity(scene, forward, likelihood, prior)
    lap = LaplaceBackend().run(logdensity, z0)

    # jaxns path: the SAME prior, via the _JaxnsPrior adapter.
    model, _ = build_scene_nested_model(
        scene, forward, likelihood, fit_leaves=lambda s: [s.theta], prior=prior
    )
    ns = JaxnsBackend(max_samples=3e4).run(model, key=jax.random.key(0))
    ns_mean = jnp.sum(ns.samples * jax.nn.softmax(ns.log_weights)[:, None], axis=0)

    # broad prior -> both recover the data, and the two backends agree.
    assert jnp.allclose(lap.mean, data, atol=0.05)
    assert jnp.allclose(ns_mean, lap.mean, atol=0.05)


def test_sequential_updating_tightens_posterior():
    """A Laplace posterior -> to_prior() -> next fit beats either data batch alone."""
    sigma = 0.4
    d1 = jnp.array([0.5])
    d2 = jnp.array([0.7])
    scene = _Toy(theta=jnp.zeros(1), label="t")

    def forward(s):
        return s.theta

    def lik(data):
        def likelihood(pred):
            return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

        return likelihood

    broad = IndependentPrior((Normal(loc=jnp.zeros(1), scale=jnp.full((1,), 5.0)),))

    def fit(prior, data):
        ld, z0, _ = build_scene_logdensity(scene, forward, lik(data), prior)
        return LaplaceBackend().run(ld, z0)

    p1 = fit(broad, d1)  # batch 1 alone
    p2_alone = fit(broad, d2)  # batch 2 alone
    p_seq = fit(p1.to_prior(), d2)  # batch 1 posterior as prior, then batch 2

    v1 = float(p1.cov[0, 0])
    v2 = float(p2_alone.cov[0, 0])
    v_seq = float(p_seq.cov[0, 0])
    # information accumulates: the sequential posterior is tighter than either batch.
    assert v_seq < v1
    assert v_seq < v2
    # and its mean sits between the two batch estimates.
    assert d1[0] - 0.05 < float(p_seq.mean[0]) < d2[0] + 0.05


def test_mixture_prior_from_samples_drives_a_scene_fit():
    """A bimodal SamplePosterior -> to_prior() -> MixturePrior drives a scene fit."""
    from photomancy.posterior import SamplePosterior

    k0, k1 = jax.random.split(jax.random.key(7))
    a = jax.random.normal(k0, (300, 1)) * 0.3 - 3.0
    b = jax.random.normal(k1, (300, 1)) * 0.3 + 3.0
    samples = jnp.concatenate([a, b], axis=0)
    post = SamplePosterior(
        samples=samples, log_weights=jnp.zeros(600), evidence=jnp.asarray(jnp.nan)
    )
    prior = post.to_prior(2, key=jax.random.key(8))  # bimodal MixturePrior over z

    scene = _Toy(theta=jnp.zeros(1), label="t")

    def forward(s):
        return s.theta

    def flat_likelihood(pred):
        return 0.0  # isolate the prior's bimodality in the logdensity

    logdensity, _z0, _ = build_scene_logdensity(scene, forward, flat_likelihood, prior)
    # the two prior modes survive into the logdensity (mode > between-modes).
    assert float(logdensity(jnp.array([3.0]))) > float(logdensity(jnp.array([0.0])))
    assert float(logdensity(jnp.array([-3.0]))) > float(logdensity(jnp.array([0.0])))
