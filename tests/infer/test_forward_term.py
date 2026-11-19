"""ForwardTerm: a likelihood of the forward of a model, as a term of a Fit."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priorities import Normal

jax.config.update("jax_enable_x64", True)

from photomancy.infer import AbstractTerm, Fit, ForwardTerm  # noqa: E402
from photomancy.infer.likelihoods import (  # noqa: E402
    gaussian_loglike,
    huber_loglike,
)


class _Model(eqx.Module):
    mu: jax.Array


class _Scale(eqx.Module):
    """Forward ``model -> weights * mu``; the weights are a pytree leaf."""

    weights: jax.Array

    def __call__(self, model):
        return self.weights * model.mu


class _Drawable(eqx.Module):
    """A likelihood that can draw: its data is the prediction plus a fixed offset."""

    data: jax.Array

    def __call__(self, predicted):
        return -0.5 * jnp.sum((predicted - self.data) ** 2)

    def sample(self, key, predicted):
        """Return a likelihood holding data drawn around ``predicted``."""
        return _Drawable(predicted + jax.random.normal(key, predicted.shape))


def _term(likelihood=None):
    data = jnp.array([1.0, 2.0, 3.0])
    return ForwardTerm(
        _Scale(jnp.array([1.0, 2.0, 0.5])),
        gaussian_loglike(data, 0.5) if likelihood is None else likelihood,
    )


def test_is_a_term_that_reads_nothing():
    """A ForwardTerm is an AbstractTerm with no cached build."""
    term = _term()
    assert isinstance(term, AbstractTerm)
    assert term.reads == ()


def test_value_and_prediction_follow_the_forward():
    """The value is the likelihood of the forward and predict returns the forward.

    Fails if the term scores anything but ``likelihood(forward(model))``.
    """
    model = _Model(jnp.array([0.9, 1.1, 5.0]))
    term = _term()
    predicted = term.forward(model)
    np.testing.assert_array_equal(term.predict(model), predicted)
    assert float(term(model)) == float(term.likelihood(predicted))


def test_forward_arrays_are_pytree_leaves():
    """The forward weights are leaves of the term, not baked constants.

    Fails if a closure hides the weights from the pytree.
    """
    leaves = jax.tree.leaves(eqx.filter(_term(), eqx.is_array))
    assert any(np.array_equal(leaf, [1.0, 2.0, 0.5]) for leaf in leaves)


def test_refuses_a_forward_that_is_not_a_module():
    """A closure forward is refused on the host with the reason."""
    with pytest.raises(TypeError, match=r"eqx\.Module"):
        ForwardTerm(lambda model: model.mu, gaussian_loglike(jnp.zeros(3), 1.0))


def test_draw_replaces_only_the_data():
    """Draw returns a term whose likelihood holds a draw around the prediction."""
    model = _Model(jnp.array([0.9, 1.1, 5.0]))
    term = ForwardTerm(_Scale(jnp.ones(3)), _Drawable(jnp.zeros(3)))
    drawn = term.draw(jax.random.key(3), model)
    assert isinstance(drawn, ForwardTerm)
    np.testing.assert_array_equal(drawn.forward.weights, term.forward.weights)
    expected = model.mu + jax.random.normal(jax.random.key(3), (3,))
    np.testing.assert_allclose(drawn.likelihood.data, expected, rtol=1e-12)


def test_draw_needs_a_likelihood_that_can_sample():
    """A likelihood with no sample method cannot define a draw."""
    with pytest.raises(TypeError, match="sample"):
        _term().draw(jax.random.key(0), _Model(jnp.ones(3)))


def test_fit_density_adds_the_prior_to_the_term_value():
    """A Fit over the term is the prior plus the term value, whatever the likelihood.

    Fails if ``Fit`` does not call the term as given (the ``mask=`` and ``likelihood``
    path of the retired Gaussian builder lives here now).
    """
    obs = jnp.array([1.0, 2.0, 3.0])
    sigma = jnp.array([0.1, 0.2, 0.3])
    mask = np.array([True, True, False])
    x = {"mu": jnp.array([0.5, 2.5, 2.0])}
    prior = Normal(jnp.zeros(3), jnp.ones(3) * 10.0)
    model = _Model(jnp.zeros(3))
    for likelihood in (
        gaussian_loglike(obs, sigma),
        huber_loglike(obs, sigma, mask),
    ):
        fit = Fit(model, {"mu": prior}, (ForwardTerm(_Scale(jnp.ones(3)), likelihood),))
        expected = float(prior.log_prob(x["mu"]).sum()) + float(likelihood(x["mu"]))
        np.testing.assert_allclose(float(fit.log_density(x)), expected, rtol=1e-12)
