"""Masked, normalized likelihood builders against closed forms."""

import math

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy import integrate, stats

from photomancy.core import build_gaussian_fit
from photomancy.likelihoods import (
    gaussian_loglike,
    huber_loglike,
    laplace_loglike,
    student_t_loglike,
)

BUILDERS = {
    "gaussian": gaussian_loglike,
    "laplace": laplace_loglike,
    "huber": huber_loglike,
    "student_t": student_t_loglike,
}


def _problem():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 5))
    sigma = rng.uniform(0.5, 2.0, size=(4, 5))
    predicted = data + sigma * rng.standard_t(3, size=(4, 5))
    mask = rng.uniform(size=(4, 5)) < 0.7
    return data, sigma, predicted, mask


class _ToyScene(eqx.Module):
    mu: jnp.ndarray


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize(
    ("name", "logpdf"),
    [
        ("gaussian", lambda x, loc, scale: stats.norm.logpdf(x, loc, scale)),
        ("laplace", lambda x, loc, scale: stats.laplace.logpdf(x, loc, scale)),
        ("student_t", lambda x, loc, scale: stats.t.logpdf(x, 4.0, loc, scale)),
    ],
)
def test_closed_forms_match_scipy_logpdf_sums(name, logpdf):
    """Value equals the scipy logpdf sum over used elements, per-element sigma.

    Tolerance basis: floating-point (rtol 1e-12, float64).
    Fails if: a normalization constant, the -log(sigma) term, the sigma-as-scale
    convention or the mask handling is wrong.
    """
    data, sigma, predicted, mask = _problem()
    ours = BUILDERS[name](data, sigma, mask)(predicted)
    expected = logpdf(predicted, data, sigma)[mask].sum()
    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=0.0)
    assert mask.sum() < mask.size


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_scalar_sigma_and_no_mask_match_scipy():
    """A scalar sigma with no mask sums every element (Student-t, nu=2.5).

    Tolerance basis: floating-point (rtol 1e-12).
    Fails if: scalar sigma does not broadcast or the default mask drops elements.
    """
    data, _, predicted, _ = _problem()
    ours = student_t_loglike(data, 0.7, nu=2.5)(predicted)
    expected = stats.t.logpdf(predicted, 2.5, loc=data, scale=0.7).sum()
    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=0.0)


def _huber_at(delta, sigma=1.0, data=0.0):
    like = huber_loglike(jnp.asarray([data]), sigma, delta=delta)
    return lambda x: like(jnp.asarray([x]))


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("delta", [0.5, 1.345, 3.0])
def test_huber_pieces_and_smooth_join(delta):
    """Huber is quadratic inside delta, linear outside, and C1 at the join.

    Tolerance basis: floating-point (atol 1e-12 on values, 1e-9 on the second
    difference, whose step 1e-3 amplifies round-off by 1e6 times 1e-16).
    Fails if: the tail intercept (delta^2 / 2) or slope is wrong, so the value or the
    first derivative jumps at |z| = delta.
    """
    f = _huber_at(delta)
    df = jax.grad(f)
    h = 1e-3
    for sign in (1.0, -1.0):
        edge = sign * delta
        np.testing.assert_allclose(
            f(edge - sign * 1e-9), f(edge + sign * 1e-9), atol=1e-8
        )
        np.testing.assert_allclose(
            df(edge - sign * 1e-9), df(edge + sign * 1e-9), atol=1e-8
        )
        np.testing.assert_allclose(df(edge), -sign * delta, atol=1e-12)
    inside = 0.4 * delta
    second_in = (f(inside + h) - 2 * f(inside) + f(inside - h)) / h**2
    np.testing.assert_allclose(second_in, -1.0, atol=1e-6)
    outside = 2.0 * delta
    second_out = (f(outside + h) - 2 * f(outside) + f(outside - h)) / h**2
    np.testing.assert_allclose(second_out, 0.0, atol=1e-6)
    np.testing.assert_allclose(df(outside), -delta, atol=1e-12)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("delta", [0.5, 1.345, 3.0])
def test_huber_density_integrates_to_one(delta):
    """exp(huber_loglike) integrates to 1 over the data axis for scale sigma.

    Tolerance basis: quadrature (scipy.integrate.quad, split at the kinks, tails to
    infinity), rtol 1e-9.
    Fails if: the erf-and-tail normalizer or the -log(sigma) term is wrong.
    """
    sigma, center = 2.0, 0.5
    like = huber_loglike(jnp.asarray([center]), sigma, delta=delta)

    def density(x):
        return math.exp(float(like(jnp.asarray([x]))))

    a, b = center - sigma * delta, center + sigma * delta
    total = sum(
        integrate.quad(density, lo, hi, epsabs=0, epsrel=1e-12)[0]
        for lo, hi in [(-np.inf, a), (a, b), (b, np.inf)]
    )
    np.testing.assert_allclose(total, 1.0, rtol=1e-9, atol=0.0)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("name", list(BUILDERS))
def test_masked_elements_are_inert_even_when_nan(name):
    """NaN data, sigma and predicted at masked elements leave value and gradient finite.

    The value equals the likelihood built on the used subset, the gradient is exactly
    zero at masked elements, and changing a masked element changes nothing.
    Tolerance basis: exact-algebra (rtol 1e-14).
    Fails if: a masked element leaks into the sum or the gradient (NaN * 0), or the
    normalization counts masked elements.
    """
    data, sigma, predicted, mask = _problem()
    build = BUILDERS[name]
    clean = build(data[mask], sigma[mask])(predicted[mask])

    bad_data = np.where(mask, data, np.nan)
    bad_sigma = np.where(mask, sigma, np.nan)
    bad_pred = jnp.asarray(np.where(mask, predicted, np.nan))
    like = build(bad_data, bad_sigma, mask)
    value, grad = jax.value_and_grad(like)(bad_pred)
    np.testing.assert_allclose(value, clean, rtol=1e-14, atol=0.0)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.asarray(grad)[~mask] == 0.0)

    moved = jnp.asarray(np.where(mask, predicted, 1e30))
    assert float(like(moved)) == float(value)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("name", list(BUILDERS))
def test_jit_equals_eager(name):
    """filter_jit of the Module likelihood equals the eager value.

    Tolerance basis: floating-point (4 ULP; XLA fusion differs across versions).
    Fails if: the Module leaves are not traced consistently (a baked or stale mask,
    data or normalization).
    """
    data, sigma, predicted, mask = _problem()
    like = BUILDERS[name](data, sigma, mask)
    eager = like(predicted)
    jitted = eqx.filter_jit(lambda lk, p: lk(p))(like, jnp.asarray(predicted))
    np.testing.assert_array_max_ulp(np.asarray(eager), np.asarray(jitted), maxulp=4)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("name", list(BUILDERS))
def test_gradients_match_finite_differences(name):
    """Forward and reverse derivatives to order 2 agree with finite differences.

    Residuals are chosen away from the Laplace kink at 0 and the Huber join at
    1.345, in both the Huber core and tails.
    Tolerance basis: finite difference (eps 1e-4, atol and rtol 1e-4, float64).
    Fails if: the hand-written kernels or normalization differ from the density.
    """
    data = np.array([0.0, 0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 1.5, 0.8])
    predicted = jnp.asarray([0.4, -0.7, 3.0, -4.0])
    like = BUILDERS[name](data, sigma)
    with jax.enable_x64():
        check_grads(
            like,
            (predicted,),
            order=2,
            modes=("fwd", "rev"),
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )


@pytest.mark.parametrize("name", ["huber", "student_t"])
def test_shape_parameters_must_be_positive(name):
    """A non-positive delta or nu raises at build time.

    Fails if: an invalid shape parameter silently builds an improper density.
    """
    key = "delta" if name == "huber" else "nu"
    with pytest.raises(ValueError, match=key):
        BUILDERS[name](np.zeros(2), 1.0, **{key: 0.0})


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_build_gaussian_fit_default_is_unchanged_and_accepts_a_likelihood():
    """The default fit is the bare Gaussian sum; likelihood= and mask= swap it in.

    Tolerance basis: exact-algebra (the default value is compared bitwise).
    Fails if: the default path changed (numbers or ordering), the builder is not
    called with the noise sigma and mask, or mask is silently ignored without a
    likelihood.
    """
    obs = jnp.array([1.0, 2.0, 3.0])
    sigma = jnp.array([0.1, 0.2, 0.3])
    scene = _ToyScene(mu=jnp.zeros(3))
    kwargs = {
        "fit_leaves": lambda s: [s.mu],
        "noise_sigma": sigma,
        "forward": lambda s: s.mu,
    }
    z = jnp.array([0.5, 2.5, 2.0])

    ld, _, _ = build_gaussian_fit(scene, obs, **kwargs)
    assert float(ld(z)) == float(-0.5 * jnp.sum((z - obs) ** 2 * (1.0 / sigma**2)))

    mask = np.array([True, True, False])
    robust, _, _ = build_gaussian_fit(
        scene, obs, **kwargs, likelihood=huber_loglike, mask=mask
    )
    expected = huber_loglike(obs, sigma, mask)(z)
    assert float(robust(z)) == float(expected)

    with pytest.raises(ValueError, match="mask requires"):
        build_gaussian_fit(scene, obs, **kwargs, mask=mask)
