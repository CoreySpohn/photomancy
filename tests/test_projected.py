"""Projected astrometric loss: analytic partials against automatic differentiation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.orbit.projected import (
    astrometric_loss,
    chart_astrometric_loss,
    make_astrometric_evaluator,
    make_chart_astrometric_evaluator,
)

jax.config.update("jax_enable_x64", True)


def inputs(orbits=3, epochs=19):
    """Nonunit geometry and unequal residuals expose coordinate scaling errors."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(orbits, 7))
    features[:, 0] = rng.uniform(0.01, 0.98, orbits)
    features[:, 2] = rng.uniform(0.1, 3, orbits)
    return tuple(
        jnp.asarray(x)
        for x in (
            features,
            np.linspace(-0.4, 1.2, epochs),
            rng.normal(size=(orbits, 2, epochs)),
            0.3,
        )
    )


@pytest.mark.parametrize("shape", [(1, 1), (3, 19), (19, 3)])
def test_all_feature_derivatives(shape):
    """All seven analytic partials agree with AD to float64 reduction error."""
    args = inputs(*shape)
    reference = jax.jit(jax.value_and_grad(astrometric_loss))(*args)
    actual = jax.jit(make_astrometric_evaluator())(*args)
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)


def test_higher_derivatives_and_finite_difference():
    """Compare HVPs and an independent finite-difference anchor."""
    features, *args = inputs()
    evaluate = make_astrometric_evaluator()
    tangent = jnp.linspace(-0.3, 0.7, features.size).reshape(features.shape)

    def analytic(x):
        return evaluate(x, *args)[1]

    ordinary = jax.grad(lambda x: astrometric_loss(x, *args))
    got = jax.jvp(analytic, (features,), (tangent,))[1]
    want = jax.jvp(ordinary, (features,), (tangent,))[1]
    np.testing.assert_allclose(got, want, rtol=2e-11, atol=2e-10)
    step = 1e-5
    derivative = (
        astrometric_loss(features + step * tangent, *args)
        - astrometric_loss(features - step * tangent, *args)
    ) / (2 * step)
    np.testing.assert_allclose(
        derivative, jnp.sum(analytic(features) * tangent), rtol=2e-8, atol=2e-8
    )


def test_circular_closed_form():
    """e=0 reduces to a rotated circle, independent of a Kepler implementation."""
    features = jnp.array([[0.0, 0.2, 1.7, 2.0, -0.4, 0.3, 1.1]])
    times = jnp.linspace(-1, 1, 11)
    observed = jnp.zeros((1, 2, 11))
    phase = 1.7 * (times - 0.2)
    prediction = features[0, 3:5, None] * jnp.cos(phase)
    prediction += features[0, 5:7, None] * jnp.sin(phase)
    actual, _ = make_astrometric_evaluator()(features, times, observed, jnp.array(0.2))
    np.testing.assert_allclose(
        actual, 0.5 * jnp.sum((prediction / 0.2) ** 2), rtol=2e-15
    )


def _to_chart_features(legacy, epoch_d):
    """Perifocal features to epoch-frame features (same orbits, same prediction)."""
    e, tp_d, n, ax, ay, bx, by = legacy.T
    mean_anom = n * (epoch_d - tp_d)
    ex, ey = e * jnp.cos(mean_anom), -e * jnp.sin(mean_anom)
    phi = jnp.arctan2(ey, ex)
    c, s = jnp.cos(phi), jnp.sin(phi)
    # block_chart = block_peri @ R(phi): perifocal axes sit at -phi from stored axes.
    a_new = jnp.stack([ax * c - bx * s, ay * c - by * s])
    b_new = jnp.stack([ax * s + bx * c, ay * s + by * c])
    return jnp.stack([ex, ey, n, a_new[0], a_new[1], b_new[0], b_new[1]], axis=1)


@pytest.mark.parametrize("epoch_d", [0.0, 0.35])
def test_chart_loss_and_gradient_match_perifocal_loss(epoch_d):
    """Fails if the epoch-frame loss or its pulled-back gradient drift from perifocal.

    Three orbits, 19 epochs, e in (0.01, 0.98), at two reference epochs. The loss
    at the mapped features matches the perifocal loss to rtol 1e-12, and the chart
    gradient pulled back through the feature map (vjp) equals the analytic
    perifocal gradient to rtol 1e-9 (forward map conditioning, 7 columns).
    """
    legacy, times, observed, sigma = inputs(3, 19)
    chart = _to_chart_features(legacy, epoch_d)
    want = astrometric_loss(legacy, times, observed, sigma)
    got = chart_astrometric_loss(chart, times, observed, sigma, epoch_d=epoch_d)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    value, grad_chart = make_chart_astrometric_evaluator()(
        chart, times, observed, sigma, epoch_d=epoch_d
    )
    np.testing.assert_allclose(value, want, rtol=1e-12, atol=1e-12)
    _, pullback = jax.vjp(lambda f: _to_chart_features(f, epoch_d), legacy)
    pulled = pullback(grad_chart)[0]
    _, grad_legacy = make_astrometric_evaluator()(legacy, times, observed, sigma)
    np.testing.assert_allclose(pulled, grad_legacy, rtol=1e-9, atol=1e-9)
