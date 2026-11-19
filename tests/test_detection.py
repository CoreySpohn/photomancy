"""Detection: amplitude profiling, the likelihood-ratio map and the curvature block."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import toy_coronagraph as toy
from image_scene import (
    Placed,
    make_detector,
    make_exposure,
    make_instrument,
    make_scene,
)

from photomancy.detection import (
    curvature_block,
    detect,
    likelihood_ratio_map,
    measure,
    profile_amplitude,
)
from photomancy.image import ImageModel, ImageTerm
from photomancy.infer.likelihoods import (
    gaussian_loglike,
    huber_loglike,
    laplace_loglike,
)
from photomancy.postproc.core.detection import PSFTemplateFilter
from photomancy.postproc.templates.base import ArrayTemplateProvider


def _problem(seed=1, n=200, shift=0.7):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    template = rng.normal(size=n)
    sigma = rng.uniform(0.5, 2.0, size=n)
    data = base + shift * template + sigma * rng.normal(size=n)
    return data, base, template, sigma


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_gaussian_profile_equals_the_closed_form():
    """``a_hat = t'Wr / t'Wt``, ``2 dlogL = (t'Wr)**2 / t'Wt``, curvature ``t'Wt``.

    Truth source: the weighted least-squares solution for ``r = data - base`` and
    ``W = 1 / sigma**2``, from NumPy.
    Tolerance basis: exact-algebra (rtol 1e-12; a quadratic needs one Newton step).
    Fails if: the amplitude, the statistic or the curvature uses the wrong weights, or
    the derivative of the likelihood is taken with respect to the wrong argument.
    """
    data, base, template, sigma = _problem()
    w = 1.0 / sigma**2
    r = data - base
    tw = np.sum(template * w * template)
    want = np.sum(template * w * r) / tw
    a_hat, two_delta, curvature = profile_amplitude(
        gaussian_loglike(data, sigma), base, template
    )
    np.testing.assert_allclose(a_hat, want, rtol=1e-12)
    np.testing.assert_allclose(two_delta, want**2 * tw, rtol=1e-12)
    np.testing.assert_allclose(curvature, tw, rtol=1e-12)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_huber_profile_matches_a_bisection_optimum():
    """Newton on the Huber likelihood reaches the optimum of a bisection search.

    Truth source: bisection on the sign of the exact derivative of
    ``a -> huber_loglike(base + a t)`` over [-10, 10], 200 halvings, with five gross
    outliers in the data so some residuals sit in the Laplace tails.
    Tolerance basis: floating-point (atol 1e-9 on ``a_hat``; Newton on a
    piecewise quadratic converges in a finite number of steps).
    Fails if: a tail pixel's zero curvature breaks the step, or fewer steps than the
    default are enough only for the Gaussian.
    """
    data, base, template, sigma = _problem()
    data[:5] += 8.0
    like = huber_loglike(data, sigma)
    grad = jax.grad(lambda a: like(base + a * template))
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if grad(mid) > 0 else (lo, mid)
    a_hat, two_delta, curvature = profile_amplitude(like, base, template)
    np.testing.assert_allclose(a_hat, 0.5 * (lo + hi), atol=1e-9)
    assert float(two_delta) > 0.0 and float(curvature) > 0.0


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_zero_template_gives_zero_without_nan():
    """A zero template returns ``a_hat = 0``, no statistic, no curvature, no NaN.

    Truth source: the definition (the likelihood does not depend on ``a``).
    Tolerance basis: exact-algebra (equality).
    Fails if: the Newton step divides by a zero curvature.
    """
    data, base, _, sigma = _problem()
    for like in (gaussian_loglike(data, sigma), huber_loglike(data, sigma)):
        out = profile_amplitude(like, base, jnp.zeros_like(base))
        assert [float(v) for v in out] == [0.0, 0.0, 0.0]


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_laplace_likelihood_is_rejected():
    """The Laplace likelihood, piecewise linear in the prediction, raises a ValueError.

    Truth source: its second derivative is zero almost everywhere.
    Tolerance basis: exact-algebra (the exception type and message).
    Fails if: Newton silently returns ``a_hat = 0`` for a likelihood it cannot use.
    """
    data, base, template, sigma = _problem()
    like = laplace_loglike(data, sigma)
    with pytest.raises(ValueError, match="Laplace"):
        profile_amplitude(like, base, template)
    with pytest.raises(ValueError, match="Laplace"):
        likelihood_ratio_map(((like, base, lambda x: template),), jnp.zeros(1))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_masked_nan_does_not_leak():
    """NaN in the data, base and template at masked pixels leaves the fit unchanged.

    Truth source: the closed form over the used pixels only.
    Tolerance basis: exact-algebra (rtol 1e-12, and every output finite).
    Fails if: a masked pixel's NaN reaches the value or a derivative (as it does when
    the derivative is taken in reverse mode through a NaN template).
    """
    data, base, template, sigma = _problem()
    mask = np.arange(data.size) % 5 != 0
    w = 1.0 / sigma**2
    r = data - base
    want = np.sum((template * w * r)[mask]) / np.sum((template * w * template)[mask])

    def poisoned(a):
        a = np.array(a)
        a[~mask] = np.nan
        return a

    for build in (gaussian_loglike, huber_loglike):
        like = build(poisoned(data), sigma, mask)
        a_hat, two_delta, curvature = profile_amplitude(
            like, poisoned(base), poisoned(template)
        )
        assert all(np.isfinite(float(v)) for v in (a_hat, two_delta, curvature))
        if build is gaussian_loglike:
            np.testing.assert_allclose(a_hat, want, rtol=1e-12)
    out = likelihood_ratio_map(
        ((like, poisoned(base), lambda x: poisoned(template) * x),), jnp.ones(3)
    )
    assert np.all(np.isfinite(out.z)) and np.all(np.isfinite(out.a_hat))


class _PointTemplate(eqx.Module):
    """Unit-flux toy companion at a detector-pixel position."""

    instrument: toy.Instrument

    def __call__(self, position):
        return toy.point_image(self.instrument, position)


def _toy_scene(flux_scale=2.0):
    instrument = toy.build_instrument()
    star = toy.Star(jnp.asarray([0.3, -0.2]), jnp.asarray(1.0e-3))
    unocculted = eqx.tree_at(
        lambda i: i.occulter, instrument, jnp.ones_like(instrument.occulter)
    )
    peak = float(jnp.max(toy.point_image(unocculted, jnp.zeros(2))))
    flux = flux_scale * 1.0e-3 / peak
    base = toy.star_only_image(instrument, star)
    c = (toy.N_PAD - 1) / 2.0
    yy, xx = np.mgrid[: toy.N_PAD, : toy.N_PAD] - c
    mask = jnp.asarray(np.hypot(xx, yy) >= 3.0)
    sigma = 0.05 * float(jnp.median(base))
    grid = np.array(
        [(x, y) for x in range(-8, 9) for y in range(-8, 9) if np.hypot(x, y) >= 4.0],
        dtype=float,
    )
    return instrument, base, flux, mask, sigma, grid


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_noise_free_companion_is_found_at_its_pixel_with_its_flux():
    """On a noise-free toy companion the map peaks at the true pixel at the true flux.

    Truth source: the injected position and flux; with ``data = base + f t(x0)`` the
    residual equals ``f t(x0)`` and, by Cauchy-Schwarz, ``z(x)`` is largest at ``x0``.
    Tolerance basis: exact-algebra (rtol 1e-9 on the amplitude, in float64).
    Fails if: the signed root, the shared template or the position axis is mixed up.
    """
    instrument, base, flux, mask, sigma, grid = _toy_scene()
    truth = jnp.asarray([6.0, -5.0])
    data = base + flux * toy.point_image(instrument, truth)
    like = gaussian_loglike(data, sigma, mask)
    out = likelihood_ratio_map(
        ((like, base, _PointTemplate(instrument)),), jnp.asarray(grid)
    )
    best = int(jnp.argmax(out.z))
    np.testing.assert_array_equal(grid[best], [6.0, -5.0])
    np.testing.assert_allclose(out.a_hat[best], flux, rtol=1e-9)
    assert float(out.z[best]) > 5.0
    np.testing.assert_allclose(
        out.z[best], out.a_hat[best] / out.sigma[best], rtol=1e-9
    )


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_batched_map_equals_the_unbatched_map():
    """``batch_size`` changes the memory use, not the result.

    Truth source: the same map with one position at a time.
    Tolerance basis: floating-point (1e-9 of each output's largest magnitude: a batch
    is vmapped, which may reorder the pixel sums by an ulp, and the residual
    ``data - base`` subtracts a star 1e3 times brighter than the companion, so
    bitwise equality is not promised; 7 does not divide the grid, so the remainder
    batch is exercised).
    Fails if: batching drops or misorders positions, or changes the statistic.
    """
    instrument, base, flux, mask, sigma, grid = _toy_scene()
    data = base + flux * toy.point_image(instrument, jnp.asarray([6.0, -5.0]))
    terms = ((gaussian_loglike(data, sigma, mask), base, _PointTemplate(instrument)),)
    plain = likelihood_ratio_map(terms, jnp.asarray(grid))
    for batch in (7, 16):
        batched = likelihood_ratio_map(terms, jnp.asarray(grid), batch_size=batch)
        for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(batched), strict=True):
            np.testing.assert_allclose(
                a, b, rtol=0.0, atol=1e-9 * float(jnp.max(jnp.abs(a)))
            )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_two_terms_share_one_amplitude_like_the_summed_closed_form():
    """Two exposures give ``a_hat = sum_k t'W_k r_k / sum_k t'W_k t``.

    Truth source: NumPy weighted least squares over both exposures at one template.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: the terms are profiled separately or the amplitudes are averaged.
    """
    d1, b1, t, s1 = _problem(seed=2)
    d2, b2, _, s2 = _problem(seed=3)
    terms = (
        (gaussian_loglike(d1, s1), b1, lambda x: x * t),
        (gaussian_loglike(d2, s2), b2, lambda x: x * t),
    )
    out = likelihood_ratio_map(terms, jnp.asarray([1.0, 2.0]))
    num = np.sum(t * (d1 - b1) / s1**2) + np.sum(t * (d2 - b2) / s2**2)
    den = np.sum(t * t / s1**2) + np.sum(t * t / s2**2)
    # template x * t: a_hat(x) = want / x and sigma(x) = 1 / (x sqrt(den))
    np.testing.assert_allclose(
        out.a_hat, (num / den) / np.array([1.0, 2.0]), rtol=1e-12
    )
    np.testing.assert_allclose(
        out.sigma, 1.0 / (np.array([1.0, 2.0]) * np.sqrt(den)), rtol=1e-12
    )
    np.testing.assert_allclose(out.z, num / np.sqrt(den), rtol=1e-12)


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
def test_white_noise_gives_unit_variance_z():
    """Over 2000 seeded noise-only draws, the standard deviation of ``z`` is 1.

    Truth source: for Gaussian noise ``z = t'Wr / sqrt(t'Wt)`` is N(0, 1) exactly.
    Tolerance basis: monte-carlo (|sd - 1| < 0.06, four times
    ``sqrt(2 / 1999)``, so a correct implementation passes for any seed).
    Fails if: the statistic is scaled by the wrong curvature or the noise weights.
    """
    instrument, base, _, mask, sigma, _ = _toy_scene()
    position = jnp.asarray([[6.0, -5.0]])
    noise = np.random.default_rng(7).normal(size=(2000, *base.shape)) * sigma

    def z_of(data):
        like = gaussian_loglike(data, sigma, mask)
        out = likelihood_ratio_map(
            ((like, base, _PointTemplate(instrument)),), position
        )
        return out.z[0]

    z = np.asarray(jax.vmap(z_of)(jnp.asarray(base + noise)))
    assert abs(np.std(z, ddof=1) - 1.0) < 0.06
    assert abs(np.mean(z)) < 4.0 / np.sqrt(2000)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_curvature_block_matches_closed_form_and_finite_difference():
    """The block is ``-d2 logp`` over the chosen leaves, the rest held fixed.

    Truth source: for a quadratic ``-0.5 (p - m)' A (p - m)`` over two leaves the block
    is ``A``, whatever the held leaf is; for a cubic term, a central difference of the
    gradient.
    Tolerance basis: floating-point (rtol 1e-12 for the quadratic; atol 1e-6 for
    the central difference with step 1e-4, whose error is of order step**2).
    Fails if: leaves are ordered differently from the selector, the held leaf is
    differentiated, or the matrix is not symmetric.
    """
    a_mat = np.array([[3.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 1.0]])

    class P(eqx.Module):
        u: jnp.ndarray
        v: jnp.ndarray
        held: jnp.ndarray

    params = P(jnp.array([0.4, -1.0]), jnp.array(0.3), jnp.array([5.0, 7.0]))

    def quadratic(p):
        x = jnp.concatenate([p.u, p.v[None]]) - jnp.array([1.0, 2.0, 3.0])
        return -0.5 * x @ a_mat @ x + 0.1 * jnp.sum(p.held)

    where = lambda p: (p.u, p.v)  # noqa: E731
    block = curvature_block(quadratic, params, where)
    np.testing.assert_allclose(block, a_mat, rtol=1e-12)

    def cubic(p):
        return quadratic(p) - p.u[0] ** 3 * p.v + jnp.sum(p.held) * p.u[1] ** 2

    block = curvature_block(cubic, params, where)
    flat = np.array([0.4, -1.0, 0.3])

    def grad_at(x):
        p = P(jnp.asarray(x[:2]), jnp.asarray(x[2]), params.held)
        g = jax.grad(cubic)(p)
        return np.concatenate([np.asarray(g.u), [float(g.v)]])

    step = 1e-4
    fd = np.stack(
        [
            -(grad_at(flat + step * e) - grad_at(flat - step * e)) / (2 * step)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(block, fd, atol=1e-6)
    np.testing.assert_array_equal(block, block.T)
    single = curvature_block(quadratic, params, lambda p: p.v)
    np.testing.assert_allclose(single, [[a_mat[2, 2]]], rtol=1e-12)


# ----- detect and measure over an ImageTerm ---------------------------------------

_VARIANCE = 100.0
_PATHS = {"scene_path": "scene", "instrument_path": "instrument"}
_PAS = jnp.arange(0.0, 360.0, 30.0)
_TRUE_PA, _TRUE_AMP = 150.0, 3.0


def _image_problem(kind, seed=11):
    """A model, an ImageTerm over one frame, and the noise-free mean at the truth.

    The frame is the expected counts of a planet of amplitude 3 (units of contrast
    1e-10) at position angle 150 degrees plus noise from the detector's own law.
    """
    detector = make_detector(
        kind, **({"variance_e2": _VARIANCE} if kind == "constant" else {})
    )
    instrument = make_instrument(detector)
    scene = make_scene(contrast=0.0)
    exposure = make_exposure(scene, instrument)
    model = ImageModel(scene, instrument)
    shape = (1, *detector.shape)
    blank = ImageTerm(
        jnp.zeros(shape),
        jnp.ones(shape, bool),
        exposure=exposure,
        **_PATHS,
    )
    truth = Placed()(model, _TRUE_PA, _TRUE_AMP)
    mean = np.asarray(blank.predict(truth))
    rng = np.random.default_rng(seed)
    if kind == "constant":
        observed = mean + np.sqrt(_VARIANCE) * rng.normal(size=mean.shape)
    else:
        observed = rng.poisson(mean).astype(float)
    mask = np.ones(shape, bool)
    mask[..., :3, :] = False
    mask[..., 10:13, 10:13] = False
    term = ImageTerm(
        jnp.asarray(observed), jnp.asarray(mask), exposure=exposure, **_PATHS
    )
    return model, term, mean, mask


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.medium(
    reason="forward-mode scan through the renderer compiles for each of three calls"
)
def test_detect_equals_the_linear_matched_filter_on_a_synthetic_frame():
    """``detect`` is the whitened matched filter for a linear model, Gaussian noise.

    Truth source: NumPy ``a_hat = t'Wr / t'Wt``, ``z = t'Wr / sqrt(t'Wt)``,
    ``sigma = 1 / sqrt(t'Wt)`` over the scored pixels, with ``r = data - m0``, ``m0``
    the expected counts without the planet and ``t`` the expected counts of a unit
    planet at the position (the model is linear in the planet's contrast).
    Tolerance basis: exact-algebra (rtol 1e-9; one Newton step is exact for a
    quadratic, and the data are about 1e3 counts so the residual keeps 12 digits).
    Fails if: the statistic uses the wrong noise, the mask, the position axis, or the
    Newton step is not exact for a Gaussian term.
    """
    model, term, _, mask = _image_problem("constant")
    place = Placed()
    m0 = np.asarray(term.predict(place(model, 0.0, 0.0)))
    resid = np.asarray(term.observed_e) - m0
    want_a, want_z, want_s = [], [], []
    for pa in np.asarray(_PAS):
        t = np.asarray(term.predict(place(model, pa, 1.0))) - m0
        w = mask / _VARIANCE
        twt = np.sum(w * t * t)
        twr = np.sum(w * t * resid)
        want_a.append(twr / twt)
        want_z.append(twr / np.sqrt(twt))
        want_s.append(1.0 / np.sqrt(twt))
    out = detect(term, model, place, _PAS)
    np.testing.assert_allclose(out.a_hat, want_a, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out.z, want_z, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out.sigma, want_s, rtol=1e-9)
    assert int(jnp.argmax(out.z)) == int(np.argmax(np.asarray(_PAS) == _TRUE_PA))
    assert float(out.z.max()) > 5.0
    batched = detect(term, model, place, _PAS, batch_size=5)
    np.testing.assert_allclose(batched.z, out.z, rtol=1e-12, atol=1e-12)


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
def test_detect_on_poisson_counts_recovers_the_planet():
    """On Poisson counts the map peaks at the true angle and the amplitude is close.

    The amplitude is within four formal standard errors of the truth, and Newton has
    converged by four steps.

    Truth source: the injected amplitude and position angle.
    Tolerance basis: monte-carlo (one seeded realization, four formal sigma; a
    correct implementation passes for any seed with probability above 0.9999) and
    floating-point (rtol 1e-6 between 4 and 8 Newton steps).
    Fails if: the Poisson term is not the one profiled, or Newton diverges.
    """
    model, term, _, _ = _image_problem("poisson")
    place = Placed()
    out = detect(term, model, place, _PAS)
    best = int(jnp.argmax(out.z))
    assert float(_PAS[best]) == _TRUE_PA
    assert abs(float(out.a_hat[best]) - _TRUE_AMP) < 4.0 * float(out.sigma[best])
    more = detect(term, model, place, _PAS, n_newton=8)
    np.testing.assert_allclose(out.a_hat, more.a_hat, rtol=1e-6, atol=1e-9)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.medium(
    reason="second-order autodiff through the renderer compiles the Hessian"
)
def test_measure_covariance_matches_the_curvature_block_and_the_fisher_matrix():
    """The covariance is the inverse curvature block, the inverse Fisher at the truth.

    At noise-free truth the covariance of amplitude and position angle is the inverse
    Fisher matrix ``J'WJ``.

    Truth source: ``curvature_block`` over a record of (amplitude, position angle);
    and NumPy ``J'WJ`` with ``J = [dm/da, dm/dpa]`` from the expected counts,
    ``dm/dpa`` by a central difference of step 0.01 degree. With zero residual the
    Hessian equals ``J'WJ``.
    Tolerance basis: floating-point (rtol 1e-8 against the block, same Hessian; rtol
    1e-4 against the Fisher matrix, the central difference error, about
    (0.01 / 7.6)**2).
    Fails if: the covariance uses another curvature, the position enters the wrong
    scalar, or the Newton iteration does not reach the maximum (value check, atol
    1e-6).
    """
    model, term, mean, mask = _image_problem("constant")
    place = Placed()
    clean = eqx.tree_at(lambda t: t.observed_e, term, jnp.asarray(mean))
    rows = measure(clean, model, place, jnp.asarray(_TRUE_PA + 2.0), n_newton=10)
    assert rows.names == ("amplitude", "position/0")
    np.testing.assert_allclose(rows.value, [_TRUE_AMP, _TRUE_PA], atol=1e-6)

    class Source(eqx.Module):
        amplitude: jax.Array
        position: jax.Array

    def loglike(src):
        return clean(place(model, src.position, src.amplitude))

    truth = Source(jnp.asarray(_TRUE_AMP), jnp.asarray(_TRUE_PA))
    block = curvature_block(loglike, truth, lambda s: (s.amplitude, s.position))
    np.testing.assert_allclose(rows.cov, np.linalg.inv(block), rtol=1e-8)
    np.testing.assert_allclose(rows.sigma, np.sqrt(np.diag(rows.cov)))

    h = 0.01
    pred = lambda pa, a: np.asarray(clean.predict(place(model, pa, a)))  # noqa: E731
    d_amp = pred(_TRUE_PA, 1.0) - pred(_TRUE_PA, 0.0)
    d_pa = (pred(_TRUE_PA + h, _TRUE_AMP) - pred(_TRUE_PA - h, _TRUE_AMP)) / (2 * h)
    jac = np.stack([d_amp[mask], d_pa[mask]], axis=1)
    fisher = jac.T @ jac / _VARIANCE
    np.testing.assert_allclose(rows.cov, np.linalg.inv(fisher), rtol=1e-4)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_psf_template_filter_amplitude_equals_the_linear_profile_amplitude():
    """``PSFTemplateFilter`` matches the linear amplitude profile.

    Truth source: ``profile_amplitude(gaussian_loglike(patch, sigma), b, t - tbar)``
    with ``b`` the weighted patch mean and ``tbar`` the weighted template mean, which
    is the generalized least squares amplitude of ``y = c + a t`` (the filter's
    definition).
    Both the inverse-variance and the unweighted forms are checked, at an integer
    position so the patch extraction is exact.
    Tolerance basis: exact-algebra (rtol 1e-9; spline interpolation at an integer
    position reproduces the pixels to rounding).
    Fails if: the filter drops the weighting or the background, or the linear profile
    uses a different normalization.
    """
    rng = np.random.default_rng(21)
    size, k = 25, 9
    image = rng.normal(size=(size, size)) + 3.0
    variance = rng.uniform(0.5, 2.0, size=(size, size))
    yy, xx = np.mgrid[:k, :k] - (k - 1) / 2
    stamp = np.exp(-(yy**2 + xx**2) / 6.0) + 0.05 * rng.normal(size=(k, k))
    position = jnp.asarray([12.0, 12.0])
    window = (slice(8, 17), slice(8, 17))
    patch, var_patch = image[window], variance[window]
    provider = ArrayTemplateProvider(stamp)
    for weighted in (True, False):
        flt = PSFTemplateFilter(
            provider, noise_variance=jnp.asarray(variance) if weighted else None
        ).bind(position)
        got = flt.evaluate(jnp.asarray(image), position)
        w = 1.0 / var_patch if weighted else np.ones_like(patch)
        b = np.sum(w * patch) / np.sum(w)
        t_zm = stamp - np.sum(w * stamp) / np.sum(w)
        sigma = np.sqrt(var_patch) if weighted else np.ones_like(patch)
        want, _, _ = profile_amplitude(
            gaussian_loglike(patch, sigma), np.full_like(patch, b), t_zm
        )
        np.testing.assert_allclose(got, want, rtol=1e-9)
