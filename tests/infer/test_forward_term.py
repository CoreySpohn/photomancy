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

    def draw(self, key, predicted):
        """Return a likelihood holding data drawn around ``predicted``."""
        return _Drawable(predicted + jax.random.normal(key, predicted.shape))


def _term(likelihood=None):
    data = jnp.array([1.0, 2.0, 3.0])
    return ForwardTerm(
        _Scale(jnp.array([1.0, 2.0, 0.5])),
        gaussian_loglike(data, 0.5) if likelihood is None else likelihood,
    )


class _Built(eqx.Module):
    """A forward holding a build that cached the model's ``mu``."""

    weights: jax.Array
    reads: tuple[str, ...] = eqx.field(static=True, default=("mu",))

    def __call__(self, model):
        return self.weights


def test_reads_are_the_forwards_reads_and_a_fit_refuses_freeing_them():
    """The term reports what its forward's build read; ``Fit`` refuses freeing it.

    Truth source: the forward's own ``reads`` tuple.
    Fails if: the term hides a held build's reads (a class-level empty tuple), so a
    fit frees a path whose change the cached build never sees.
    """
    term = _term()
    assert isinstance(term, AbstractTerm)
    assert term.reads == ()
    built = ForwardTerm(_Built(jnp.ones(3)), gaussian_loglike(jnp.zeros(3), 1.0))
    assert built.reads == ("mu",)
    law = Normal(jnp.zeros(3), jnp.ones(3))
    with pytest.raises(ValueError, match="reads"):
        Fit(_Model(jnp.ones(3)), {"mu": law}, (built,))


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


def test_residuals_come_from_the_likelihood_and_a_fit_names_a_term_without_one():
    """``Fit.residuals`` gives each term's residual at ``model_at(x)``.

    Truth source: the Gaussian builder's residual of the forward at the model of
    ``x`` (``(data - weights * mu) / sigma``), computed by hand.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: ``Fit.residuals`` evaluates at the base model instead of ``x``, drops or
    reorders a term, or a term whose likelihood has no residual (a robust law) is
    passed over silently.
    """
    weights = jnp.array([1.0, 2.0, 0.5])
    data = jnp.array([1.0, 2.0, 3.0])
    first = ForwardTerm(_Scale(weights), gaussian_loglike(data, 0.5))
    second = ForwardTerm(_Scale(weights), gaussian_loglike(2 * data, 0.25))
    law = Normal(jnp.zeros(3), jnp.ones(3))
    fit = Fit(_Model(jnp.zeros(3)), {"mu": law}, (first, second))
    mu = jnp.array([0.9, 1.1, 5.0])
    out = fit.residuals({"mu": mu})
    assert len(out) == 2
    np.testing.assert_allclose(out[0], (data - weights * mu) / 0.5, rtol=1e-12)
    np.testing.assert_allclose(out[1], (2 * data - weights * mu) / 0.25, rtol=1e-12)
    robust = ForwardTerm(_Scale(weights), huber_loglike(data, 0.5))
    with pytest.raises(NotImplementedError, match=r"terms\[1\]"):
        fit.replace_term(1, robust).residuals({"mu": mu})
    with pytest.raises(NotImplementedError, match="residual"):
        ForwardTerm(_Scale(weights), _Drawable(data)).residual(_Model(mu))


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
    """A likelihood with no draw method cannot define a draw."""
    with pytest.raises(TypeError, match="draw"):
        _term().draw(jax.random.key(0), _Model(jnp.ones(3)))


def test_a_closure_likelihood_is_refused():
    """A lambda likelihood is a static leaf that recompiles per instance: refuse it.

    Fails if ``ForwardTerm`` accepts a likelihood that is not an ``eqx.Module``.
    """
    with pytest.raises(TypeError, match="likelihood is a function"):
        _term(lambda predicted: -0.5 * jnp.sum(predicted**2))


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
