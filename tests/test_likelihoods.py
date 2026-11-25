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

from photomancy.infer.likelihoods import (
    LowRankGaussianLogLikelihood,
    gaussian_loglike,
    huber_loglike,
    laplace_loglike,
    low_rank_gaussian_loglike,
    residual_basis,
    student_t_loglike,
)
from photomancy.infer.terms import ForwardTerm

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


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_gaussian_prediction_broadcasts_only_to_the_observation():
    """The two observations determine the number of Gaussian densities scored.

    Truth source: two unit Gaussians at their means give -log(2 pi).
    Tolerance basis: exact-algebra with an absolute float64 tolerance of 1e-9.
    Fails if: a column prediction expands the two observations to four.
    """
    data = jnp.array([1.0, 2.0])
    score = gaussian_loglike(data, 1.0)
    np.testing.assert_allclose(score(data), -math.log(2 * math.pi), rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        score(1.0), -math.log(2 * math.pi) - 0.5, rtol=0, atol=1e-9
    )
    with pytest.raises(ValueError):
        score(data[:, None])


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
@pytest.mark.parametrize("name", [*BUILDERS, "low_rank"])
def test_masked_elements_are_inert_even_when_nan(name):
    """NaN data, sigma and predicted at masked elements leave value and gradient finite.

    The value equals the likelihood built on the used subset, the gradient is exactly
    zero at masked elements, and changing a masked element changes nothing.
    Tolerance basis: exact-algebra (rtol 1e-14).
    Fails if: a masked element leaks into the sum or the gradient (NaN * 0), or the
    normalization counts masked elements.
    """
    data, sigma, predicted, mask = _problem()
    bad_data = np.where(mask, data, np.nan)
    bad_sigma = np.where(mask, sigma, np.nan)
    bad_pred = jnp.asarray(np.where(mask, predicted, np.nan))
    if name == "low_rank":
        basis = np.random.default_rng(0).normal(size=(data.size, 2))
        clean = low_rank_gaussian_loglike(data[mask], sigma[mask], basis[mask.ravel()])(
            predicted[mask]
        )
        basis[~mask.ravel()] = np.nan
        like = low_rank_gaussian_loglike(bad_data, bad_sigma, basis, mask)
        np.testing.assert_array_equal(like.solve(bad_pred)[~mask], 0)
        np.testing.assert_array_equal(like.posterior_term(bad_pred)[~mask], 0)
    else:
        build = BUILDERS[name]
        clean = build(data[mask], sigma[mask])(predicted[mask])
        like = build(bad_data, bad_sigma, mask)
    value, grad = jax.value_and_grad(like)(bad_pred)
    np.testing.assert_allclose(value, clean, rtol=1e-14, atol=0.0)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.asarray(grad)[~mask] == 0.0)

    moved = jnp.asarray(np.where(mask, predicted, 1e30))
    assert float(like(moved)) == float(value)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("name", [*BUILDERS, "low_rank"])
def test_jit_equals_eager(name):
    """filter_jit of the Module likelihood equals the eager value.

    Tolerance basis: floating-point (4 ULP; XLA fusion differs across versions).
    Fails if: the Module leaves are not traced consistently (a baked or stale mask,
    data or normalization).
    """
    data, sigma, predicted, mask = _problem()
    if name == "low_rank":
        basis = np.random.default_rng(0).normal(size=(data.size, 2))
        like = low_rank_gaussian_loglike(data, sigma, basis, mask)
    else:
        like = BUILDERS[name](data, sigma, mask)
    eager = like(predicted)
    evaluate = eqx.filter_jit(lambda lk, p: lk(p))
    jitted = evaluate(like, jnp.asarray(predicted))
    np.testing.assert_array_max_ulp(np.asarray(eager), np.asarray(jitted), maxulp=4)
    if name == "low_rank":
        changed = low_rank_gaussian_loglike(data + 1, sigma * 2, basis * 3, ~mask)
        np.testing.assert_array_max_ulp(
            np.asarray(changed(predicted)),
            np.asarray(evaluate(changed, jnp.asarray(predicted))),
            maxulp=4,
        )
        assert all(eqx.is_array(leaf) for leaf in jax.tree.leaves(like))
        assert not jax.make_jaxpr(lambda lk, p: lk(p))(like, predicted).consts


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("name", [*BUILDERS, "low_rank"])
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
    if name == "low_rank":
        basis = np.random.default_rng(0).normal(size=(data.size, 2))
        like = low_rank_gaussian_loglike(data, sigma, basis)
        check_grads(
            like.solve,
            (predicted,),
            order=2,
            modes=("fwd", "rev"),
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )
        check_grads(
            like.posterior_term,
            (predicted,),
            order=2,
            modes=("fwd", "rev"),
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )
    else:
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


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_gaussian_residual_recovers_the_injected_offsets_and_robust_kinds_refuse():
    """The Gaussian residual is ``(data - predicted) / sigma``; robust kinds raise.

    Truth source: data built as ``predicted + sigma * k`` with known offsets ``k``,
    so the residual must return ``k`` (sign and scale), NaN off the mask.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: the residual is signed the wrong way, divides by a variance instead of
    a scale, leaks a masked element, or a robust or low-rank density reports a
    residual that has no unit-variance null.
    """
    _, sigma, predicted, mask = _problem()
    k = np.random.default_rng(5).normal(size=predicted.shape)
    data = predicted + sigma * k
    out = np.asarray(gaussian_loglike(data, sigma, mask).residual(predicted))
    np.testing.assert_allclose(out[mask], k[mask], rtol=1e-12)
    assert np.all(np.isnan(out[~mask]))
    for name in ("laplace", "huber", "student_t"):
        with pytest.raises(NotImplementedError, match="unit-variance"):
            BUILDERS[name](data, sigma, mask).residual(predicted)
    assert not hasattr(
        low_rank_gaussian_loglike(data, sigma, np.zeros((20, 1))), "residual"
    )


@pytest.mark.parametrize("name", ["huber", "student_t"])
def test_shape_parameters_must_be_positive(name):
    """A non-positive delta or nu raises at build time.

    Fails if: an invalid shape parameter silently builds an improper density.
    """
    key = "delta" if name == "huber" else "nu"
    with pytest.raises(ValueError, match=key):
        BUILDERS[name](np.zeros(2), 1.0, **{key: 0.0})


def test_low_rank_dense_density_and_conditional_mean():
    """Fails if the determinant, inverse covariance, or conditional mean is wrong."""
    data, sigma, predicted, mask = (v[:2] for v in _problem())
    basis = np.random.default_rng(0).normal(size=(data.size, 3))
    like = LowRankGaussianLogLikelihood(data, sigma, basis, mask)
    used_basis = basis[mask.ravel()]
    cov = np.diag(sigma[mask] ** 2) + used_basis @ used_basis.T
    expected = stats.multivariate_normal.logpdf(
        data[mask], mean=predicted[mask], cov=cov
    )
    np.testing.assert_allclose(like(predicted), expected, rtol=1e-10)
    solved = np.linalg.solve(cov, data[mask] - predicted[mask])
    np.testing.assert_allclose(like.solve(data - predicted)[mask], solved, rtol=1e-10)
    np.testing.assert_allclose(
        like.posterior_term(predicted)[mask],
        used_basis @ used_basis.T @ solved,
        rtol=1e-10,
    )


def test_low_rank_zero_is_diagonal_gaussian():
    """Fails if rank zero changes the Gaussian density, precision, or residual mean."""
    data, sigma, predicted, mask = (v[:2] for v in _problem())
    data, sigma, predicted = (
        np.where(mask, v, np.nan) for v in (data, sigma, predicted)
    )
    like = low_rank_gaussian_loglike(data, sigma, np.zeros((data.size, 0)), mask)
    assert like(predicted) == gaussian_loglike(data, sigma, mask)(predicted)
    np.testing.assert_array_equal(like.posterior_term(predicted), np.zeros(data.shape))
    np.testing.assert_allclose(
        like.solve(predicted), np.where(mask, predicted / sigma**2, 0), rtol=1e-14
    )


@pytest.mark.parametrize(
    ("data", "basis"),
    [([2e4, 0], [[1e4], [0]]), ([0, 0], [[1e4, 1e4], [0, 0]])],
)
def test_low_rank_float32_large_aligned_residual(data, basis):
    """Fails if float32 factorization loses the prior or the quadratic cancels.

    The two identical large modes make the float32 Gram matrix singular without
    host double precision. An aligned residual makes subtractive Woodbury lose
    the order-one quadratic. The covariance here is diagonal, giving an oracle.
    """
    data = np.array(data, dtype=np.float32)
    basis = np.array(basis, dtype=np.float32)
    sigma = np.ones(2, dtype=np.float32)
    with jax.enable_x64(False):
        like = low_rank_gaussian_loglike(data, sigma, basis)
        value = like(np.zeros_like(data))
    variance = 1 + np.sum(basis.astype(float) ** 2, axis=1)
    expected = stats.norm.logpdf(data, scale=np.sqrt(variance)).sum()
    np.testing.assert_allclose(value, expected, rtol=1e-6)


def test_low_rank_amplitude_statistics():
    """Fails if nuisance covariance cannot remove a confounded amplitude bias.

    Two images have different noise and templates but share one amplitude. This
    is a fixed injection, not a Monte Carlo coverage test.
    """
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(40, 3)))
    basis = 100 * q[:, :2]
    templates = (q[:, 0] + q[:, 2], q[:, 0] + 0.8 * q[:, 2])
    truth = 3.0
    sigmas = (0.05, 0.08)
    residuals = tuple(
        truth * p + 4 * q[:, 0] - 2 * q[:, 1] + rng.normal(size=40) * sigma
        for p, sigma in zip(templates, sigmas, strict=True)
    )
    for rank in (2, 0):
        numerator, precision = 0.0, 0.0
        for template, residual, sigma in zip(templates, residuals, sigmas, strict=True):
            like = low_rank_gaussian_loglike(residual, sigma, basis[:, :rank])
            numerator += template @ like.solve(residual)
            precision += template @ like.solve(template)
        amplitude, error = numerator / precision, 1 / jnp.sqrt(precision)
        if rank:
            assert abs(amplitude - truth) < error
        else:
            assert abs(amplitude - truth) > 10 * error


def test_residual_basis_uncentered_scaling_and_mask():
    """Fails if SVD is centered, scaled by the wrong count, truncated, or unmasked."""
    residuals = np.array([[[1.0, 2.0], [4.0, np.nan]], [[3.0, 4.0], [7.0, np.nan]]])
    mask = np.array([[True, True], [True, False]])
    matrix = residuals[:, mask].T
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    for rank in (0, 1, 2):
        basis, values = residual_basis(residuals, rank, mask)
        np.testing.assert_allclose(values, singular, rtol=1e-12)
        expected = left[:, :rank] * singular[:rank] / np.sqrt(2)
        # Column signs are arbitrary; each retained mode must keep its own scale.
        np.testing.assert_allclose(abs(basis[mask.ravel()]), abs(expected), rtol=1e-12)
        np.testing.assert_array_equal(basis[~mask.ravel()], np.zeros((1, rank)))
        if rank == 2:
            np.testing.assert_allclose(
                basis[mask.ravel()] @ basis[mask.ravel()].T,
                matrix @ matrix.T / 2,
                rtol=1e-12,
            )
    unmasked, values = residual_basis(residuals[:, mask], 2)
    np.testing.assert_allclose(unmasked @ unmasked.T, matrix @ matrix.T / 2, rtol=1e-12)
    np.testing.assert_allclose(values, singular, rtol=1e-12)


@pytest.mark.parametrize(
    ("residuals", "rank", "mask", "message"),
    [
        (np.ones((2, 3)), -1, None, "between zero and n"),
        (np.ones((2, 3)), 3, None, "between zero and n"),
        (np.ones((2, 3)), 2, [True, False, False], "used pixels"),
        (np.array([[1.0, np.nan]]), 0, None, "finite on the mask"),
        (np.array([[1.0, np.inf]]), 1, None, "finite on the mask"),
        (np.empty((0, 3)), 0, None, "at least one"),
    ],
)
def test_residual_basis_rejects_invalid_inputs(residuals, rank, mask, message):
    """Fails if invalid ranks or non-finite used residuals produce a silent basis."""
    with pytest.raises(ValueError, match=message):
        residual_basis(residuals, rank, mask)


class _Mu(eqx.Module):
    """Forward ``scene -> scene.mu``."""

    def __call__(self, scene):
        return scene.mu


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_forward_term_scores_the_builder_on_the_forward():
    """A ForwardTerm is the builder applied to the forward, mask and sigma included.

    Tolerance basis: exact-algebra (the values are compared bitwise).
    Fails if: the term changes the builder's number or ordering, or drops the mask or
    the per-element sigma the builder was given.
    """
    obs = jnp.array([1.0, 2.0, 3.0])
    sigma = jnp.array([0.1, 0.2, 0.3])
    scene = _ToyScene(mu=jnp.array([0.5, 2.5, 2.0]))
    mask = np.array([True, True, False])

    for builder in (gaussian_loglike, huber_loglike):
        for use in (None, mask):
            likelihood = builder(obs, sigma, use)
            term = ForwardTerm(_Mu(), likelihood)
            assert float(term(scene)) == float(likelihood(scene.mu))
    masked = ForwardTerm(_Mu(), huber_loglike(obs, sigma, mask))
    moved = _ToyScene(mu=scene.mu.at[2].set(1e6))
    assert float(masked(moved)) == float(masked(scene))
