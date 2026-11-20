"""Integration: a posterior carries into the next fit, by a new term or as a prior.

Two routes for information to accumulate across data batches on a one-parameter
model: ``Posterior.append_term`` folds a new observation into a stored cloud, and
``cluster_to_mixture`` turns a cloud into the law of the next fit's leaf.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priorities import Mixture, Normal, Uniform

jax.config.update("jax_enable_x64", True)

from photomancy.infer import (  # noqa: E402
    Fit,
    ForwardTerm,
    Grid,
    Laplace,
    cluster_to_mixture,
    run,
    weighted_cov,
    weighted_mean,
)
from photomancy.infer.backends.grid import midpoint  # noqa: E402
from photomancy.infer.likelihoods import gaussian_loglike  # noqa: E402

SIGMA = 0.4


class _Toy(eqx.Module):
    theta: jnp.ndarray


class _Identity(eqx.Module):
    """Forward ``model -> theta``."""

    def __call__(self, model):
        return model.theta


def _term(data):
    return ForwardTerm(_Identity(), gaussian_loglike(data, SIGMA))


def _fit(*data):
    prior = {"theta": Normal(jnp.zeros(1), jnp.full((1,), 5.0))}
    return Fit(_Toy(jnp.zeros(1)), prior, tuple(_term(d) for d in data))


def _moments(post):
    theta = post.samples["theta"]
    return (
        float(weighted_mean(post.log_weights, theta)[0]),
        float(weighted_cov(post.log_weights, theta)[0, 0]),
    )


# Medium: three cold compiles (two fits and the appended term) in one test.
@pytest.mark.medium
def test_appending_a_term_tightens_the_posterior():
    """A cloud plus one more observation beats the cloud alone and matches the refit.

    Fails if the appended term replaces the old likelihood instead of adding to it, is
    dropped, or is evaluated off the stored draws (the variance would stay at one
    batch, or the mean would sit at one batch estimate).
    """
    d1, d2 = jnp.array([0.5]), jnp.array([0.7])
    laplace = Laplace(n_draws=4000)
    p1 = run(_fit(d1), laplace, jax.random.key(0))
    p_seq = p1.append_term(_term(d2), min_n_eff=500, max_khat=0.7)
    joint = run(_fit(d1, d2), laplace, jax.random.key(2))

    _, v1 = _moments(p1)
    m_seq, v_seq = _moments(p_seq)
    # information accumulates: the sequential posterior is tighter than the first batch.
    assert v_seq < 0.75 * v1
    # its mean sits between the two batch estimates, and it agrees with the joint fit.
    assert d1[0] - 0.05 < m_seq < d2[0] + 0.05
    m_joint, v_joint = _moments(joint)
    assert abs(m_seq - m_joint) < 0.02
    np.testing.assert_allclose(v_seq, v_joint, rtol=0.1)


def test_mixture_from_a_cloud_is_the_law_of_the_next_leaf():
    """A bimodal cloud clustered into a Mixture drives a fit through its leaf law.

    Fails if the mixture loses a mode or its weights (a mode of the density would
    drop below the gap between modes), or does not serve as a ``(1,)`` leaf law.
    """

    class _TwoBumps(eqx.Module):
        def __call__(self, model):
            return model.theta

    class _Bumps(eqx.Module):
        def __call__(self, pred):
            z = (pred - jnp.array([-3.0, 3.0])) / 0.3
            return jax.scipy.special.logsumexp(-0.5 * z**2)

    fit = Fit(
        _Toy(jnp.zeros(1)),
        {"theta": Uniform(jnp.full((1,), -6.0), jnp.full((1,), 6.0))},
        (ForwardTerm(_TwoBumps(), _Bumps()),),
    )
    cloud = run(fit, Grid({"theta": midpoint(-6.0, 6.0, 1200)}), jax.random.key(0))
    law = cluster_to_mixture(cloud, 2, jax.random.key(1))
    assert isinstance(law, Mixture)
    np.testing.assert_allclose(
        np.sort(np.asarray(law.component_distribution.loc).ravel()),
        [-3.0, 3.0],
        atol=0.05,
    )
    np.testing.assert_allclose(jax.nn.softmax(law.mixing_logits), [0.5, 0.5], atol=0.02)

    # With no data the next fit's density is the law itself: both modes survive.
    nxt = Fit(_Toy(jnp.zeros(1)), {"theta": law})

    def density(x):
        return float(nxt.log_density({"theta": jnp.array([x])}))

    assert density(3.0) > density(0.0)
    assert density(-3.0) > density(0.0)
