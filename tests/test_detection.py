"""Detection: amplitude profiling, the likelihood-ratio map and the curvature block."""

from collections.abc import Callable

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import toy_coronagraph as toy
from hwoutils.memory import MemoryLimit
from image_scene import (
    make_detector,
    make_exposure,
    make_instrument,
    make_scene,
)
from priorities import Normal, Uniform
from scipy import stats
from skyscapes.physical_model import GridPhysicalModel

import photomancy.detection as detection
from photomancy.detection import (
    _correlation_inverse,
    curvature_block,
    detect,
    likelihood_ratio_map,
    measure,
    profile_amplitude,
)
from photomancy.image import ImageTerm
from photomancy.infer import ForwardTerm
from photomancy.infer.fit import AbstractTerm, Fit
from photomancy.infer.likelihoods import (
    gaussian_loglike,
    huber_loglike,
    laplace_loglike,
)
from photomancy.orbit.sheets import SheetSum
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
@pytest.mark.parametrize("budget", [{}, {"n_newton": 1}], ids=["default", "one-step"])
def test_gaussian_profile_equals_the_closed_form(budget):
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
    a_hat, two_delta, curvature, resolved = profile_amplitude(
        gaussian_loglike(data, sigma), base, template, **budget
    )
    assert bool(resolved)
    np.testing.assert_allclose(a_hat, want, rtol=1e-12)
    np.testing.assert_allclose(two_delta, want**2 * tw, rtol=1e-12)
    np.testing.assert_allclose(curvature, tw, rtol=1e-12)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_poisson_profile_stays_in_the_domain():
    """A count of one against a null mean of ten favors an amplitude of -9.

    Truth source: the Poisson score vanishes at mu = k. Twice the log ratio
    is 2 (log(1/10) - 1 + 10).
    Tolerance basis: rtol 1e-6 allows for iterative convergence in float64.
    Fails if: the first Newton step goes to -90 and leaves the log domain.
    """

    def like(mu):
        return jnp.sum(jnp.log(mu) - mu)

    a_hat, gain, _, resolved = profile_amplitude(like, jnp.array([10.0]), jnp.ones(1))
    np.testing.assert_allclose(a_hat, -9.0, rtol=1e-6)
    np.testing.assert_allclose(gain, 2 * (np.log(0.1) + 9), rtol=1e-6)
    assert bool(resolved)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_huber_profile_reaches_the_peak_from_a_linear_tail():
    """A single observation ten sigma above the null has positive gain.

    Truth source: the maximum of the Huber log likelihood on a 100,000 point
    NumPy grid over [0, 20], including the null.
    Tolerance basis: rtol 1e-4 on twice the gain. The nearest grid point is
    within 0.00011 sigma of the peak in its unit-curvature quadratic core.
    Fails if: the zero curvature at the null leaves the gain at zero.
    """
    delta = 1.345
    residual = np.abs(10.0 - np.linspace(0.0, 20.0, 100_000))
    loss = np.where(
        residual <= delta, 0.5 * residual**2, delta * residual - 0.5 * delta**2
    )
    like = huber_loglike(jnp.array([10.0]), 1.0, delta=delta)
    _, gain, _, resolved = profile_amplitude(like, jnp.zeros(1), jnp.ones(1))
    np.testing.assert_allclose(gain, 2 * (loss[0] - loss.min()), rtol=1e-4)
    assert bool(resolved)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e6])
def test_huber_profile_does_not_depend_on_the_template_scale(scale):
    """A source ten sigma above the null is found at any template scale.

    Truth source: the Huber loss is linear beyond delta, so the peak is at
    a = 10 / scale with twice the gain 2 (10 delta - delta^2 / 2).
    Tolerance basis: rtol 1e-6 allows for iterative convergence in float64.
    Fails if: a rejected overshoot is discarded rather than bracketed, or the
    expansion cannot reach an amplitude of 1e7 units.
    """
    delta = 1.345
    like = huber_loglike(jnp.array([10.0]), 1.0, delta=delta)
    a_hat, gain, _, resolved = profile_amplitude(like, jnp.zeros(1), jnp.full(1, scale))
    np.testing.assert_allclose(a_hat * scale, 10.0, rtol=1e-6)
    np.testing.assert_allclose(gain, 2 * (10 * delta - 0.5 * delta**2), rtol=1e-6)
    assert bool(resolved)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("null", [300.0, 1000.0])
def test_poisson_profile_survives_a_far_domain_overshoot(null):
    """A count of one against a large null mean is found by bracketing.

    Truth source: the Poisson score vanishes at mu = k, so a = 1 - null and
    twice the gain is 2 (log(1/null) - 1 + null).
    Tolerance basis: rtol 1e-6 allows for iterative convergence in float64.
    Fails if: halving alone cannot return from outside the log domain and the
    profile is left unresolved at zero gain.
    """

    def like(mu):
        return jnp.sum(jnp.log(mu) - mu)

    a_hat, gain, _, resolved = profile_amplitude(like, jnp.array([null]), jnp.ones(1))
    np.testing.assert_allclose(a_hat, 1 - null, rtol=1e-6)
    np.testing.assert_allclose(gain, 2 * (np.log(1 / null) - 1 + null), rtol=1e-6)
    assert bool(resolved)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_bright_poisson_source_resolves_at_the_default_budget():
    """One hundred counts against a null mean of ten resolve by default.

    Truth source: with k = 100 and unit template, a = 90 and twice the gain is
    2 [100 log(10) - 90].
    Tolerance basis: rtol 1e-9, the profile converges to rounding.
    Fails if: the default iteration budget leaves a bright source unresolved
    (a = 71.5 after four iterations).
    """
    k = 100.0

    def like(mu):
        return jnp.sum(k * jnp.log(mu) - mu)

    a_hat, gain, _, resolved = profile_amplitude(like, jnp.array([10.0]), jnp.ones(1))
    np.testing.assert_allclose(a_hat, 90.0, rtol=1e-9)
    np.testing.assert_allclose(gain, 2 * (k * np.log(10.0) - 90.0), rtol=1e-9)
    assert bool(resolved)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_closed_bracket_resolves_a_flat_huber_peak():
    """Opposite Huber tail slopes cancel throughout the peak.

    Truth source: the losses for observations 8 and 12 with delta = 1 sum to
    3 throughout (9, 11). The null loss is 19. The curvature on the plateau is zero.
    Tolerance basis: exact plateau bounds and rtol 1e-12 for the gain in float64.
    Fails if: a closed bracket needs positive curvature to resolve.
    """
    like = huber_loglike(jnp.array([8.0, 12.0]), 1.0, delta=1.0)
    a_hat, gain, curvature, resolved = profile_amplitude(
        like, jnp.zeros(2), jnp.ones(2)
    )
    assert 9 < float(a_hat) < 11
    np.testing.assert_allclose(gain, 2 * (19 - 3), rtol=1e-12)
    assert float(curvature) == 0 and bool(resolved)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_only_roundoff_losses_have_zero_significance():
    """A resolved gain can round slightly below zero.

    Truth source: a negative likelihood ratio statistic has no real square
    root. A one-ulp loss at unit scale is within subtraction roundoff.
    Tolerance basis: exact zeros and NaNs for one ulp, a one-nat loss,
    and a zero gain with the resolution flag clear.
    Fails if: a material loss or an unresolved null produces zero significance.
    """
    gain = jnp.array([-np.finfo(float).eps, -1.0, 0.0, 0.0])
    resolved = jnp.array([True, True, True, False])
    _, z, sigma = detection._summarize(jnp.ones(4), gain, jnp.zeros(4), resolved)
    np.testing.assert_array_equal(z, [0.0, np.nan, 0.0, np.nan])
    assert np.all(np.isposinf(sigma))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_unresolved_huber_profile_has_nan_significance():
    """One Newton step crosses two Huber core boundaries and misses the peak.

    Truth source: at zero only the first residual is in the quadratic core.
    Newton moves to 0.5 + 2 delta, where the three tail slopes sum to -delta.
    Tolerance basis: exact boolean and NaN at a point with nonzero slope.
    Fails if: exhausting the budget marks the profile resolved or gives a
    numeric map significance.
    """
    like = huber_loglike(jnp.array([0.5, 1.6, 10.0]), 1.0)
    base, template = jnp.zeros(3), jnp.ones(3)
    _, _, _, resolved = profile_amplitude(like, base, template, n_newton=1)
    assert not bool(resolved)
    found = likelihood_ratio_map(
        ((like, base, lambda x: x * template),), jnp.ones(1), n_newton=1
    )
    assert np.isnan(float(found.z[0]))


@vv.case("scene-fits", "code-verification", basis="mutation")
def test_float32_profile_three_sigma_off_a_large_objective_is_unresolved():
    """A float32 quadratic objective of size 1e7 left 3 standard errors off its peak.

    Truth source: for -1e7 - (a - 3)^2 / 2 at a = 0 the slope over the root
    curvature is exactly 3 standard errors.
    Tolerance basis: mutation; the float32 rounding allowance is
    sqrt(eps * 1e7) = 1.09 standard errors here, and the summation bound
    (8 eps per reduction level) would admit 3.09 and resolve this point.
    Fails if: the resolution tolerance admits an amplitude error of several
    standard errors on a large objective.
    """
    with jax.enable_x64(False):

        def objective(a):
            return jnp.float32(-1e7) - 0.5 * (a - 3) ** 2

        _, _, curvature, resolved = detection._profile(objective, 0, jnp.float32)
    assert float(curvature) == 1.0
    assert not bool(resolved)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize(
    ("build", "expected_z"),
    [
        (gaussian_loglike, np.array([10.05, 1.5]) / np.sqrt(1.01)),
        (
            huber_loglike,
            [
                np.sqrt(2 * 1.345 * 10 - 1.345**2 + 0.25 - 0.25 / 1.01),
                0.5 + 0.1 * 1.345,
            ],
        ),
    ],
)
def test_float32_map_resolves_source_on_bright_background(build, expected_z):
    """Two pixels have excess counts of 10 and 0.5 background standard deviations.

    Truth source: the Gaussian z is the projection onto the unit-norm template.
    At the source the Huber fit leaves both residuals in the quadratic core.
    At the neighbor the bright pixel stays in the linear tail, so it adds a
    constant 0.1 * delta to the slope from the other pixel.
    Tolerance basis: rounding a 1e4 count prediction costs about 1e-5 sigma.
    Fails if: roundoff in the fitted slope makes either significance NaN.
    """
    with jax.enable_x64(False):
        base = jnp.full((1, 2), 1e4)
        like = build(base + jnp.array([[1000.0, 50.0]]), 100.0)

        def template(position):
            return jnp.where(jnp.arange(2)[None, :] == position, 1.0, 0.1)

        found = likelihood_ratio_map(
            ((like, base, template),), jnp.arange(2), batch_size=None
        )
        np.testing.assert_allclose(found.z, expected_z, rtol=0, atol=2e-5)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_jitted_vmap_profiles_match_scalar_profiles_bitwise():
    """One compiled batch handles eight mixed core and tail observations.

    Truth source: a Python loop over scalar amplitude profiles of the same data.
    Tolerance basis: bitwise equality for one pixel with a unit template and
    exactly representable Huber transition, with no pixel reduction to reorder.
    Fails if: batching mixes brackets, changes a result, or retraces for new data.
    """
    traces = 0

    def profile(data):
        nonlocal traces
        traces += 1
        like = huber_loglike(data, 1.0, delta=1.0)
        return profile_amplitude(like, jnp.asarray(0.0), jnp.asarray(1.0))

    data = jnp.array([-12.0, -6.5, -0.5, 0.0, 0.25, 2.0, 8.0, 10.5])
    expected = tuple(
        np.stack(col) for col in zip(*(profile(y) for y in data), strict=True)
    )
    traces = 0
    batch = eqx.filter_jit(jax.vmap(profile))
    for order in (slice(None), slice(None, None, -1)):
        for got, want in zip(batch(data[order]), expected, strict=True):
            assert np.asarray(got).tobytes() == want[order].tobytes()
    assert traces == 1


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
    a_hat, two_delta, curvature, resolved = profile_amplitude(like, base, template)
    np.testing.assert_allclose(a_hat, 0.5 * (lo + hi), atol=1e-9)
    assert bool(resolved)
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
        a_hat, two_delta, curvature, resolved = profile_amplitude(
            like, base, jnp.zeros_like(base)
        )
        assert (float(a_hat), float(two_delta), float(curvature)) == (0.0, 0.0, 0.0)
        assert bool(resolved)


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
        a_hat, two_delta, curvature, resolved = profile_amplitude(
            like, poisoned(base), poisoned(template)
        )
        assert bool(resolved)
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
    """``batch_size`` changes the memory use, not the result; ``None`` is all at once.

    The default is one position at a time (``batch_size=1``) and ``None`` runs every
    position at once, as everywhere else in the package.

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
    plain = likelihood_ratio_map(terms, jnp.asarray(grid), batch_size=1)
    for a, b in zip(
        jax.tree.leaves(plain),
        jax.tree.leaves(likelihood_ratio_map(terms, jnp.asarray(grid))),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    for batch in (7, 16, None):
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
_PAS = jnp.arange(0.0, 360.0, 30.0)
_TRUE_PA, _TRUE_AMP = 150.0, 3.0
_UNIT_CONTRAST = 1e-10


class _Sky(eqx.Module):
    """A point source: amplitude in units of 1e-10 contrast, angle in degrees."""

    amplitude: jax.Array
    pa_deg: jax.Array


_SCENE_BUILDS = []  # one entry per time a term builds and renders the scene


class _SkyImage(AbstractTerm):
    """An ImageTerm on a scene rebuilt from the model's amplitude and position angle.

    Stands in for a sky-position block, which does not exist yet: the model leaves are
    the two numbers and the scene is derived from them.
    """

    image: ImageTerm
    unit_model: GridPhysicalModel = eqx.field(
        default_factory=lambda: make_scene(contrast=1.0).planets[0].physical_model
    )
    separation_arcsec: float = eqx.field(static=True, default=0.05)
    reads = ()

    def _scene(self, model):
        _SCENE_BUILDS.append(1)
        # The planet table is built once on the host; only its values vary here.
        grid = self.unit_model.contrast_grid * model.amplitude * _UNIT_CONTRAST
        planet_model = eqx.tree_at(lambda m: m.contrast_grid, self.unit_model, grid)
        return make_scene(
            sky_pa_deg=model.pa_deg,
            separation_arcsec=self.separation_arcsec,
            model=planet_model,
        )

    def __call__(self, model):
        return self.image(self._scene(model))

    def predict(self, model):
        return self.image.predict(self._scene(model))

    def fisher_root(self, model):
        return self.image.fisher_root(self._scene(model))

    def split(self, model):
        return self.image.split(self._scene(model))

    def draw(self, key, model):
        raise NotImplementedError


class _NoSplit(_SkyImage):
    """The same term without ``split``: ``detect`` renders at every Newton step."""

    def split(self, model):
        return AbstractTerm.split(self, model)


def _sky(amplitude, pa_deg):
    return _Sky(jnp.asarray(amplitude, jnp.float64), jnp.asarray(pa_deg, jnp.float64))


def _fit(term):
    free = {"amplitude": Uniform(0.0, 10.0), "pa_deg": Uniform(0.0, 360.0)}
    return Fit(_sky(0.0, 0.0), free, (term,))


def _image_problem(kind, seed=11):
    """A fit with both coordinates free, and the noise-free mean at the truth.

    The frame is the expected counts of a planet of amplitude 3 (units of contrast
    1e-10) at position angle 150 degrees plus noise from the detector's own law.
    """
    detector = make_detector(
        kind, **({"variance_e2": _VARIANCE} if kind == "constant" else {})
    )
    instrument = make_instrument(detector)
    exposure = make_exposure(make_scene(contrast=0.0), instrument)
    shape = (1, *detector.shape)
    blank = ImageTerm(
        jnp.zeros(shape),
        jnp.ones(shape, bool),
        instrument=instrument,
        exposure=exposure,
    )
    mean = np.asarray(_SkyImage(blank).predict(_sky(_TRUE_AMP, _TRUE_PA)))
    rng = np.random.default_rng(seed)
    if kind == "constant":
        observed = mean + np.sqrt(_VARIANCE) * rng.normal(size=mean.shape)
    else:
        observed = rng.poisson(mean).astype(float)
    mask = np.ones(shape, bool)
    mask[..., :3, :] = False
    mask[..., 10:13, 10:13] = False
    # The detector is as wide as the toy tables: score only supported pixels.
    mask &= np.asarray(blank.supported_mask(make_scene(contrast=0.0)))
    image = ImageTerm(
        jnp.asarray(observed),
        jnp.asarray(mask),
        instrument=instrument,
        exposure=exposure,
    )
    return _fit(_SkyImage(image)), mean, mask


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
    Fails if: the statistic uses the wrong noise, the mask, the trial axis, or the
    Newton step is not exact for a Gaussian term. The batched run is held to the
    same 1e-9: a batch size changes the vectorization and so the summation order
    of the scored-pixel sums, which agree only to the 12 digits the residual keeps,
    so a relative 1e-12 is below that resolution (one platform differed by 3.7e-10).
    """
    fit, _, mask = _image_problem("constant")
    sky = fit.terms[0]
    resid = np.asarray(sky.image.observed_e) - np.asarray(sky.predict(_sky(0.0, 0.0)))
    want_a, want_z, want_s = [], [], []
    for pa in np.asarray(_PAS):
        m0 = np.asarray(sky.predict(_sky(0.0, pa)))
        t = np.asarray(sky.predict(_sky(1.0, pa))) - m0
        w = mask / _VARIANCE
        twt = np.sum(w * t * t)
        twr = np.sum(w * t * resid)
        want_a.append(twr / twt)
        want_z.append(twr / np.sqrt(twt))
        want_s.append(1.0 / np.sqrt(twt))
    out = detect(fit, {"pa_deg": _PAS}, amplitude="amplitude")
    np.testing.assert_allclose(out.a_hat, want_a, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out.z, want_z, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out.sigma, want_s, rtol=1e-9)
    assert int(jnp.argmax(out.z)) == int(np.argmax(np.asarray(_PAS) == _TRUE_PA))
    assert float(out.z.max()) > 5.0
    batched = detect(fit, {"pa_deg": _PAS}, amplitude="amplitude", batch_size=5)
    np.testing.assert_allclose(batched.z, out.z, rtol=1e-9, atol=1e-9)


def test_detect_refuses_a_bad_amplitude_path_or_trial_keys():
    """The amplitude must be a free one-element leaf law; trials cover the rest."""
    fit, _, _ = _image_problem("constant")
    with pytest.raises(ValueError, match="free leaf-law"):
        detect(fit, {"pa_deg": _PAS}, amplitude="flux")
    with pytest.raises(ValueError, match="keys"):
        detect(fit, {"amplitude": _PAS}, amplitude="amplitude")
    with pytest.raises(ValueError, match="keys"):
        measure(fit, {"amplitude": jnp.asarray(1.0)})


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.medium(reason="two Newton depths through the renderer, 5 s cold")
def test_detect_on_poisson_counts_recovers_the_planet_and_renders_once():
    """On Poisson counts the map peaks at the true angle, and ``detect`` renders once.

    The amplitude is within four formal standard errors of the truth, and Newton has
    converged by four steps. The render-once path (a term that offers ``split``) gives
    the map the nested-derivative path gives (the same term without ``split``), and it
    builds the scene twice (the render with its tangent, and the render of the
    linearity check) where the nested path builds it at every derivative.

    Truth source: the injected amplitude and position angle; for the parity, the
    nested-derivative path that ``detect`` takes for a term without ``split``,
    which is today's code; the scene builds are counted while ``detect`` traces.
    Tolerance basis: monte-carlo (one seeded realization, four formal sigma; a
    correct implementation passes for any seed with probability above 0.9999),
    floating-point (rtol 1e-6 between 4 and 8 Newton steps; rtol 1e-9 between the two
    paths, which differ only in the order of the arithmetic).
    Fails if: the Poisson term is not the one profiled, Newton diverges, the
    render-once path scores a different likelihood (the scored set, the detector law,
    the tangent) or renders more than once.
    """
    fit, _, _ = _image_problem("poisson")
    out = detect(fit, {"pa_deg": _PAS}, amplitude="amplitude")
    best = int(jnp.argmax(out.z))
    assert float(_PAS[best]) == _TRUE_PA
    assert abs(float(out.a_hat[best]) - _TRUE_AMP) < 4.0 * float(out.sigma[best])
    more = detect(fit, {"pa_deg": _PAS}, amplitude="amplitude", n_newton=8)
    np.testing.assert_allclose(out.a_hat, more.a_hat, rtol=1e-6, atol=1e-9)

    nested = _fit(_NoSplit(fit.terms[0].image))
    trials = {"pa_deg": _PAS[:7]}  # a new shape, so both paths trace afresh
    del _SCENE_BUILDS[:]
    once = detect(fit, trials, amplitude="amplitude")
    builds_once = len(_SCENE_BUILDS)
    del _SCENE_BUILDS[:]
    other = detect(nested, trials, amplitude="amplitude")
    builds_nested = len(_SCENE_BUILDS)
    # the host-side shape check of ``split`` builds the scene once, the trace twice
    # (the render with its tangent, and the render of the linearity check)
    assert builds_once == 3 and builds_nested >= 4, (builds_once, builds_nested)
    for name in ("a_hat", "z", "sigma"):
        np.testing.assert_allclose(
            getattr(once, name), getattr(other, name), rtol=1e-9, atol=1e-12
        )


_NULL_FRAMES = 1600
_NULL_PAS = jnp.arange(0.0, 360.0, 45.0)
# 3.2 lambda/D at 550 nm for the 6 m aperture; the toy field reaches 3.6.
_NULL_SEPARATION = 0.06


class _PoissonLoglike(eqx.Module):
    """``sum k log(mu) - mu`` over the masked pixels of one frame of counts."""

    counts: jax.Array
    mask: jax.Array

    def __call__(self, prediction):
        safe = jnp.where(self.mask, prediction, 1.0)
        return jnp.sum(jnp.where(self.mask, self.counts * jnp.log(safe) - safe, 0.0))


class _Templates(eqx.Module):
    """Unit-amplitude planet images, one per trial position index."""

    images: jax.Array

    def __call__(self, index):
        return self.images[index]


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.slow(reason="12800 null trials, past the 2000-draw unit limit")
def test_detect_false_alarms_match_the_poisson_null():
    """On planet-free Poisson frames ``z`` and the pull are unit normals.

    1600 frames of the toy scene with no planet, drawn from the detector's own law
    (NumPy Poisson of the expected counts, about 1160 electrons per pixel,
    ``default_rng(0)``), and ``detect`` at 8 position angles 45 degrees apart at
    3.2 lambda/D, outside 3 lambda/D (template overlap below 1e-5, so the trials
    are independent). ``likelihood_ratio_map`` scores the same frames with a
    hand-written Poisson log-likelihood over the same base and templates.

    Truth source: the null by construction; under it the signed root of the
    likelihood ratio is a unit normal for each trial, and so is the pull
    ``a_hat / sigma``; the maximum of 8 independent trials exceeds
    ``t8 = Phi^-1(0.99 ** (1 / 8))`` in one map in 100. ``z`` is a difference of
    log-likelihoods and does not see the curvature: a wrong curvature (and so a
    wrong ``sigma``) leaves it unchanged, which only the pull detects. The map is
    the same profile of the same likelihood up to a constant.
    Tolerance basis: monte-carlo. Over n = 12800 trials, mean within ``4 / sqrt(n)``
    (0.035) and standard deviations within ``4 sqrt(2 / (n - 1))`` (0.050, so a 5
    percent scale error fails); the count above 3 sigma and the count of maps above
    ``t8`` inside their binomial bands at 0.01 two sided (p = 1.35e-3 of 12800 and
    p = 0.01 of 1600). A correct implementation passes for a random seed with
    probability above 0.97. Map against ``detect``: floating-point (atol 1e-9 on
    ``a_hat``, whose standard error is about 0.5; rtol 1e-10 on ``sigma``; atol
    1e-7 on ``z**2``, the rounding of a difference of log-likelihoods near 3e6,
    measured at 2.4e-9 on 480 trials).
    Fails if: the statistic is scaled by the wrong factor (a missing factor 2 in the
    likelihood ratio gives a standard deviation of 0.71), the curvature is (doubled
    it gives a pull standard deviation of 1.41, with ``z`` unchanged to 1e-9), the
    Newton profile does not reach the maximum, the trials are not scored against
    the planet-free model, or the map and ``detect`` profile different likelihoods.
    """
    detector = make_detector("poisson")
    instrument = make_instrument(detector)
    empty = make_scene(contrast=0.0)
    exposure = make_exposure(empty, instrument)
    shape = (1, *detector.shape)
    blank = ImageTerm.on_support(
        empty, jnp.zeros(shape), instrument=instrument, exposure=exposure
    )
    sky = _SkyImage(blank, separation_arcsec=_NULL_SEPARATION)
    mean = np.asarray(sky.predict(_sky(0.0, 0.0)))
    rng = np.random.default_rng(0)
    frames = rng.poisson(mean, size=(_NULL_FRAMES, *mean.shape)).astype(float)
    mask = blank.mask
    image = ImageTerm(frames[0], mask, instrument=instrument, exposure=exposure)
    fit = _fit(_SkyImage(image, separation_arcsec=_NULL_SEPARATION))
    templates = _Templates(
        jnp.stack([sky.predict(_sky(1.0, pa)) - mean for pa in _NULL_PAS])
    )

    def scored(frame):
        one = eqx.tree_at(lambda f: f.terms[0].image.observed_e, fit, frame)
        found = detect(one, {"pa_deg": _NULL_PAS}, amplitude="amplitude")
        loglike = _PoissonLoglike(frame, mask)
        mapped = likelihood_ratio_map(
            ((loglike, mean, templates),), jnp.arange(_NULL_PAS.size)
        )
        return found, mapped

    found, mapped = jax.vmap(scored)(jnp.asarray(frames))
    z = np.asarray(found.z)
    pull = np.asarray(found.a_hat / found.sigma)
    assert not np.isnan(z).any()
    assert not np.isnan(pull).any()
    n = z.size
    band = 4.0 * np.sqrt(2.0 / (n - 1))
    assert abs(z.mean()) < 4.0 / np.sqrt(n)
    assert abs(z.std(ddof=1) - 1.0) < band
    assert abs(pull.std(ddof=1) - 1.0) < band
    low, high = stats.binom.ppf([0.005, 0.995], n, stats.norm.sf(3.0))
    assert low <= int((z > 3.0).sum()) <= high
    t8 = stats.norm.isf(1.0 - 0.99 ** (1.0 / _NULL_PAS.size))
    low, high = stats.binom.ppf([0.005, 0.995], _NULL_FRAMES, 0.01)
    assert low <= int((z.max(axis=1) > t8).sum()) <= high
    np.testing.assert_allclose(mapped.a_hat, found.a_hat, rtol=0, atol=1e-9)
    np.testing.assert_allclose(mapped.sigma, found.sigma, rtol=1e-10)
    np.testing.assert_allclose(mapped.z**2, z**2, rtol=0, atol=1e-7)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.medium(
    reason="second-order autodiff through the renderer compiles the Hessian"
)
def test_measure_covariance_matches_the_curvature_block_and_the_fisher_matrix():
    """The covariance is the inverse curvature block, the inverse Fisher at the truth.

    At noise-free truth the covariance of amplitude and position angle is the inverse
    Fisher matrix ``J'WJ``. The start is the detection at the offset position, as in
    use.

    Truth source: ``curvature_block`` over a record of (amplitude, position angle);
    and NumPy ``J'WJ`` with ``J = [dm/da, dm/dpa]`` from the expected counts,
    ``dm/dpa`` by a central difference of step 0.01 degree. With zero residual the
    Hessian equals ``J'WJ``.
    Tolerance basis: floating-point (rtol 1e-8 against the block, same Hessian; rtol
    1e-4 against the Fisher matrix, the central difference error, about
    (0.01 / 7.6)**2).
    Fails if: the covariance uses another curvature, a coordinate enters the wrong
    row, or the Newton iteration does not reach the maximum (value check, atol 1e-6).
    """
    fit, mean, mask = _image_problem("constant")
    sky = fit.terms[0]
    clean_image = eqx.tree_at(lambda t: t.observed_e, sky.image, jnp.asarray(mean))
    clean = _SkyImage(clean_image)
    fit = _fit(clean)
    start_pa = jnp.asarray([_TRUE_PA + 2.0])
    peak = detect(fit, {"pa_deg": start_pa}, amplitude="amplitude")
    start = {"amplitude": peak.a_hat[0], "pa_deg": start_pa[0]}
    est = measure(fit, start, n_newton=10)
    np.testing.assert_allclose(est.value["amplitude"], _TRUE_AMP, atol=1e-6)
    np.testing.assert_allclose(est.value["pa_deg"], _TRUE_PA, atol=1e-6)

    block = curvature_block(
        clean, _sky(_TRUE_AMP, _TRUE_PA), lambda s: (s.amplitude, s.pa_deg)
    )
    np.testing.assert_allclose(est.cov, np.linalg.inv(block), rtol=1e-8)
    np.testing.assert_allclose(est.sigma, np.sqrt(np.diag(est.cov)))
    assert bool(est.regular) and bool(est.converged) and bool(est.finite)
    assert bool(est.positive_definite) and int(est.n_steps) > 0
    root = np.sqrt(np.diag(np.asarray(block)))
    np.testing.assert_allclose(
        est.min_eigenvalue,
        np.linalg.eigvalsh(np.asarray(block) / np.outer(root, root))[0],
    )
    np.testing.assert_allclose(
        est.log_likelihood, float(fit.log_likelihood(est.value)), rtol=1e-12
    )
    np.testing.assert_allclose(np.diag(est.corr), 1.0, rtol=1e-12)

    h = 0.01
    pred = lambda pa, a: np.asarray(clean.predict(_sky(a, pa)))  # noqa: E731
    d_amp = pred(_TRUE_PA, 1.0) - pred(_TRUE_PA, 0.0)
    d_pa = (pred(_TRUE_PA + h, _TRUE_AMP) - pred(_TRUE_PA - h, _TRUE_AMP)) / (2 * h)
    jac = np.stack([d_amp[mask], d_pa[mask]], axis=1)
    fisher = jac.T @ jac / _VARIANCE
    np.testing.assert_allclose(est.cov, np.linalg.inv(fisher), rtol=1e-4)


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
        want, _, _, _ = profile_amplitude(
            gaussian_loglike(patch, sigma), np.full_like(patch, b), t_zm
        )
        np.testing.assert_allclose(got, want, rtol=1e-9)


def _spd(n, seed, cond):
    """A random symmetric positive definite matrix of the given condition number."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.geomspace(1.0, cond, n)) @ q.T


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("span", [1e6, 1e8])
def test_correlation_form_inverse_recovers_a_known_covariance_at_condition_1e12(span):
    """The inverse and the flag survive a graded matrix (condition 1e12 and 1e16).

    The information is ``D^-1 K D^-1`` with ``K`` of condition 5 and ``D`` spanning
    ``span``, so the matrix has condition number near ``span**2`` and its covariance is
    exactly ``D K^-1 D``. It is positive definite by construction.

    Truth source: ``K^-1`` from NumPy on the well conditioned ``K``; the smallest
    eigenvalue of ``K`` in correlation form, which is the one the flag reports (it does
    not depend on ``span``).
    Tolerance basis: floating-point (rtol 1e-10 in the scaled covariance, the
    correlation matrix and the eigenvalue).
    Fails if: the diagonal scaling is dropped or applied once instead of twice, the
    correlation matrix is not normalized, or a positive definite graded matrix is
    reported as not regular.
    """
    scale = np.geomspace(1.0, span, 6)
    k = _spd(6, seed=2, cond=5.0)
    info = k / np.outer(scale, scale)
    assert np.linalg.cond(info) > 0.5 * span**2
    cov, corr, regular, min_eig = _correlation_inverse(jnp.asarray(info))
    want = np.linalg.inv(k)
    diag = np.sqrt(np.diag(k))
    np.testing.assert_allclose(
        np.asarray(cov) / np.outer(scale, scale), want, rtol=1e-10
    )
    assert bool(regular)
    np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        corr, want / np.sqrt(np.outer(np.diag(want), np.diag(want))), rtol=1e-10
    )
    np.testing.assert_allclose(
        min_eig, np.linalg.eigvalsh(k / np.outer(diag, diag))[0], rtol=1e-10
    )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_regular_is_false_at_a_saddle_and_true_at_a_maximum():
    """The flag is the sign of the information's eigenvalues, with finite output.

    Truth source: the eigenvalues of ``diag(2, -3)`` (a saddle of the log density, so
    information of mixed sign) and of ``diag(2, 3)``, in correlation form (the
    diagonal scaled out, so ``diag(1, -1)`` and the identity).
    Tolerance basis: exact-algebra (the covariance is ``diag(1/2, -1/3)`` to
    rounding; the smallest eigenvalue is exact).
    Fails if: a saddle reports a covariance without the flag, or the inversion turns
    the indefinite matrix into NaN or inf.
    """
    cov, _, regular, min_eig = _correlation_inverse(jnp.diag(jnp.array([2.0, -3.0])))
    assert not bool(regular)
    assert float(min_eig) == pytest.approx(-1.0, rel=1e-12)  # diag(1, -1)
    np.testing.assert_allclose(cov, np.diag([0.5, -1.0 / 3.0]), rtol=1e-12)
    assert np.all(np.isfinite(cov))
    cov, _, regular, min_eig = _correlation_inverse(jnp.diag(jnp.array([2.0, 3.0])))
    assert bool(regular) and float(min_eig) == pytest.approx(1.0, rel=1e-12)


class _Identity(eqx.Module):
    def __call__(self, model):
        return model.p


class _Saddle(eqx.Module):
    """``log L = -0.5 (p0**2 - p1**2)``, a saddle at the origin."""

    def __call__(self, predicted):
        return -0.5 * (predicted[0] ** 2 - predicted[1] ** 2)


class _P(eqx.Module):
    p: jax.Array


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_measure_flags_a_saddle_and_stays_there():
    """``measure`` at a stationary saddle stays, reports ``regular == False``.

    Truth source: the Hessian of the quadratic ``-0.5 (p0**2 - p1**2)`` is
    ``diag(-1, 1)``, so the information is ``diag(1, -1)`` and the covariance
    ``diag(1, -1)`` (the second width is NaN); the gradient at the origin is zero.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: the flag is missing, a step leaves the stationary point, the gradient
    norm or Newton decrement is not zero there, or the width is not NaN.
    """
    term = ForwardTerm(_Identity(), _Saddle())
    fit = Fit(_P(jnp.zeros(2)), {"p": Uniform(-5.0, 5.0)}, (term,))
    start = {"p": jnp.zeros(2)}
    est = measure(fit, start)
    assert not bool(est.regular)
    assert float(est.grad_norm) == 0.0
    assert float(est.newton_decrement) == pytest.approx(0.0, abs=1e-12)
    assert float(est.min_eigenvalue) == pytest.approx(-1.0)
    np.testing.assert_allclose(est.value["p"], start["p"], rtol=1e-12)
    np.testing.assert_allclose(est.cov, np.diag([1.0, -1.0]), rtol=1e-12)
    assert np.isnan(float(est.sigma[1])) and float(est.sigma[0]) == pytest.approx(1.0)
    assert est.assumptions == ()


class _PseudoHuber(eqx.Module):
    """``log L = -sqrt(1 + p**2)``: positive curvature, a far Newton step too long."""

    def __call__(self, predicted):
        return -jnp.sum(jnp.sqrt(1.0 + predicted**2))


class _Bump(eqx.Module):
    """``log L = exp(-p**2 / 2) - 1``: negative curvature beyond ``|p| = 1``."""

    def __call__(self, predicted):
        return jnp.sum(jnp.exp(-0.5 * predicted**2) - 1.0)


class _CountedLikelihood(eqx.Module):
    likelihood: eqx.Module
    record: Callable = eqx.field(static=True)

    def __call__(self, predicted):
        jax.debug.callback(self.record, predicted, ordered=True)
        return self.likelihood(predicted)


def _counted_measure(likelihood, prior, point, n_newton, *, n_steps=0):
    calls = []
    term = ForwardTerm(_Identity(), _CountedLikelihood(likelihood, calls.append))
    fit = Fit(_P(jnp.zeros_like(point)), {"p": prior}, (term,))
    estimate = measure(fit, {"p": point}, n_steps=n_steps, n_newton=n_newton)
    jax.effects_barrier()
    count = len(calls)
    info = fit.information(
        estimate.value, kind="observed", include_prior=False, batch_size=1
    )
    jax.effects_barrier()
    return count, estimate, info


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("n_newton", [1, 4])
def test_measure_stops_evaluating_after_the_first_newton_refusal(n_newton):
    """At a stationary maximum, the count is independent of the Newton budget.

    Truth source: ordered runtime callbacks and the zero of the pseudo-Huber score.
    Base a103c3f evaluated the likelihood 15 times for one step and 51 for four.
    Tolerance basis: exact callback counts and flags, rtol 1e-10 for curvature.
    Fails if: Newton rescans a refused point, evaluates a rounded-away trial, or
    recomputes the final identity-coordinate Hessian.
    """
    calls, estimate, info = _counted_measure(
        _PseudoHuber(), Normal(0.0, 10.0), jnp.asarray(0.0), n_newton
    )
    assert calls == 5
    np.testing.assert_allclose(estimate.value["p"], 0.0, rtol=1e-10)
    np.testing.assert_allclose(info, [[1.0]], rtol=1e-10)
    np.testing.assert_allclose(estimate.cov, [[1.0]], rtol=1e-10)
    assert bool(estimate.regular) and bool(estimate.converged)
    assert int(estimate.n_steps) == int(estimate.n_newton) == 0
    assert not bool(estimate.stopped_early)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize(
    "n_newton, calls, point, curvature, variance, moved, regular",
    [
        (1, 9, -0.4999999999999978, 0.7155417527999346, 1.3975424859373649, 1, False),
        (8, 32, 7.450580596923828e-09, 1.0, 1.0, 4, True),
    ],
)
def test_measure_reuses_newton_curvature_only_at_the_final_point(
    n_newton, calls, point, curvature, variance, moved, regular
):
    """Newton keeps the backtracked steps and reuses curvature only after a refusal.

    Truth source: the pseudo-Huber likelihood starting at 2 and a103c3f, which
    evaluated it 9 times with one step and 69 times with eight.
    Tolerance basis: exact counts and flags, rtol 1e-10 against the base values.
    Fails if: a backtracked step changes, a refusal is retried, or the Hessian from
    before an accepted step is used at the new point.
    """
    count, estimate, info = _counted_measure(
        _PseudoHuber(), Normal(0.0, 10.0), jnp.asarray(2.0), n_newton
    )
    assert count == calls
    np.testing.assert_allclose(estimate.value["p"], point, rtol=1e-10)
    np.testing.assert_allclose(info, [[curvature]], rtol=1e-10)
    np.testing.assert_allclose(estimate.cov, [[variance]], rtol=1e-10)
    assert bool(estimate.regular) is bool(estimate.converged) is regular
    assert int(estimate.n_steps) == 0 and int(estimate.n_newton) == moved
    assert not bool(estimate.stopped_early)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_measure_builds_stored_curvature_for_a_bounded_fit():
    """A refused Newton step on bounded support still needs the stored Hessian.

    Truth source: the saddle has stored information diag(1, -1), whereas the
    support map has derivative 2.5 at the origin. Base a103c3f evaluated the
    likelihood 56 times for the same budget of four steps.
    Tolerance basis: exact counts and flags, rtol 1e-10 for curvature.
    Fails if: polishing retries the refusal or substitutes the real-coordinate Hessian.
    """
    calls, estimate, info = _counted_measure(
        _Saddle(), Uniform(-5.0, 5.0), jnp.zeros(2), 4
    )
    assert calls == 8
    np.testing.assert_allclose(estimate.value["p"], [0.0, 0.0], rtol=1e-10)
    np.testing.assert_allclose(info, np.diag([1.0, -1.0]), rtol=1e-10)
    np.testing.assert_allclose(estimate.cov, np.diag([1.0, -1.0]), rtol=1e-10)
    assert not bool(estimate.regular) and bool(estimate.converged)
    assert int(estimate.n_steps) == int(estimate.n_newton) == 0
    assert not bool(estimate.stopped_early)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_measure_run_with_both_stages_uses_fewer_evaluations():
    """A complete measure run keeps the estimate with fewer likelihood calls.

    Truth source: a103c3f evaluated this pseudo-Huber likelihood 53 times with both
    the L-BFGS and Newton budgets set to four. The L-BFGS stage is unchanged.
    Tolerance basis: exact counts and flags, rtol 1e-10 against the base values.
    Fails if: the Newton polish does redundant work or changes the final estimate.
    """
    calls, estimate, info = _counted_measure(
        _PseudoHuber(), Normal(0.0, 10.0), jnp.asarray(2.0), 4, n_steps=4
    )
    assert calls == 28
    np.testing.assert_allclose(estimate.value["p"], 6.8579414972368136e-09, rtol=1e-10)
    np.testing.assert_allclose(info, [[1.0]], rtol=1e-10)
    np.testing.assert_allclose(estimate.cov, [[1.0]], rtol=1e-10)
    assert bool(estimate.regular) and bool(estimate.converged)
    assert int(estimate.n_steps) == 4 and int(estimate.n_newton) == 1
    assert not bool(estimate.stopped_early)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_the_newton_polish_accepts_a_step_only_where_the_likelihood_rises():
    """A full Newton step that overshoots is backtracked; an indefinite point stays.

    Truth source: the model. For ``log L = -sqrt(1 + p**2)`` at ``p = 2`` the Hessian of
    the negative is ``(1 + p**2)**-1.5 > 0``, and the full Newton step goes to
    ``p = -8``, where the log likelihood is -8.06 against -2.24 at the start. Halving
    twice gives ``p = -0.5`` (log L -1.118), which raises it, so the polish moves there
    and then toward the peak at 0. For ``log L = exp(-p**2 / 2) - 1`` at ``p = 2`` the
    Hessian of the negative is ``(1 - p**2) exp(-p**2 / 2) < 0`` (indefinite), so the
    polish does not move and reports no step, where a proposal built on the identity
    would take the gradient step.
    Tolerance basis: exact-algebra (the log likelihood must rise at every accepted
    step; the point must be unchanged when refused).
    Fails if a proposal is accepted without the likelihood check (the first case ends at
    -8.06, below its start), if the backtracking is dropped (the first case then never
    moves), or if an indefinite Hessian is followed.
    """
    far = Fit(
        _P(jnp.zeros(())),
        {"p": Normal(0.0, 10.0)},
        (ForwardTerm(_Identity(), _PseudoHuber()),),
    )
    start = far.to_real({"p": jnp.asarray(2.0)})
    ones = jax.tree.map(jnp.ones_like, start)
    for n in (1, 4):
        moved, count, _ = detection._newton(far, start, ones, n, 1)
        x, _ = far.from_real(moved)
        assert int(count) >= 1
        assert float(far.log_likelihood(x)) > -np.sqrt(5.0) + 0.5
    x, _ = far.from_real(detection._newton(far, start, ones, 1, 1)[0])
    assert float(x["p"]) == pytest.approx(-0.5, abs=1e-12)
    assert float(far.log_likelihood(x)) == pytest.approx(-np.sqrt(1.25), rel=1e-12)

    saddle = Fit(
        _P(jnp.zeros(())),
        {"p": Normal(0.0, 10.0)},
        (ForwardTerm(_Identity(), _Bump()),),
    )
    start = saddle.to_real({"p": jnp.asarray(2.0)})
    moved, count, _ = detection._newton(
        saddle, start, jax.tree.map(jnp.ones_like, start), 4, 1
    )
    assert int(count) == 0
    np.testing.assert_array_equal(moved["p"], start["p"])


class _TwoBumps(eqx.Module):
    """Log of a 70/30 mixture of unit-height-free bumps at -2 and +2, width 0.5."""

    def __call__(self, predicted):
        bump = lambda c: jnp.exp(-0.5 * ((predicted - c) / 0.5) ** 2)  # noqa: E731
        return jnp.log(0.7 * bump(-2.0) + 0.3 * bump(2.0))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_measure_is_not_converged_when_the_last_iterate_is_below_the_best_reached(
    monkeypatch,
):
    """A last iterate on a lower peak than the trace reached is not converged.

    Truth source: the model. ``log(0.7 N(p; -2) + 0.3 N(p; 2))`` has two peaks whose
    heights differ by ``ln(7/3) = 0.85`` nat. The run from ``p = -1.9`` reaches the high
    one and is converged. A spy then replaces the optimizer's result by ``p = 2``, the
    lower peak, standing for a last line-search step that rose: the gradient there is
    zero and the information positive, so only the comparison with the trace's best
    (and with the start's) can say the run did not end where it was.
    Tolerance basis: exact-algebra (flags; the gap is 0.85 against the 1e-7 stall
    tolerance).
    Fails if ``converged`` looks only at the gradient and the decrement.
    """
    term = ForwardTerm(_Identity(), _TwoBumps())
    fit = Fit(_P(jnp.zeros(())), {"p": Normal(0.0, 10.0)}, (term,))
    start = {"p": jnp.asarray(-1.9)}
    good = measure(fit, start)
    assert bool(good.converged) and bool(good.regular)
    assert float(good.value["p"]) == pytest.approx(-2.0, abs=1e-6)

    real = detection.map_fit

    def moved(*args, **kwargs):
        out = real(*args, **kwargs)
        flat = jax.tree.map(lambda a: jnp.full_like(a, 2.0), out.params)
        return eqx.tree_at(lambda r: r.params, out, flat)

    monkeypatch.setattr(detection, "map_fit", moved)
    bad = measure(fit, start)
    assert float(bad.value["p"]) == pytest.approx(2.0, abs=1e-6)
    assert float(bad.newton_decrement) < 1e-6 and bool(bad.positive_definite)
    assert bool(bad.finite) and not bool(bad.converged) and not bool(bad.regular)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_measure_needs_signed_support_for_a_faint_source_and_sizes_itself():
    """Positive support is not regular at a negative estimate; signed support is.

    The model is ``(a, a * b / 360)`` observed with unit noise, ``b`` a position
    stand-in. Truth source: the closed-form maximum. With the data ``(-3, 0)`` the
    unconstrained maximum is ``a = -3``, ``b = 0``; a law on ``(0, 5)`` puts the
    maximum on the edge ``a -> 0`` where ``b`` drops out of the likelihood (a flat
    direction; the stored-coordinate gradient is ``-3``, a Newton decrement of 4.5
    nats, above the 1 nat limit). With data ``(0, 0)`` and
    a start at ``a = 0`` the gradient is exactly zero and the information is
    ``diag(1, 0)``: a flat direction with ``grad_norm == 0``.
    Tolerance basis: exact-algebra (atol 1e-6 on the signed maximum, exact zeros on the
    flat point).
    Fails if: a positive law reports a regular maximum at its edge, a signed law does
    not reach the maximum, the flat point is called regular, or ``measure`` does not
    refuse beyond ``max_bytes`` (``MemoryLimit``) before evaluating the model.
    """
    free_signed = {"amplitude": Normal(0.0, 10.0), "pa_deg": Uniform(-200.0, 200.0)}
    free_positive = {"amplitude": Uniform(0.0, 5.0), "pa_deg": Uniform(-200.0, 200.0)}
    start = {"amplitude": jnp.asarray(0.5), "pa_deg": jnp.asarray(10.0)}
    data = gaussian_loglike(jnp.array([-3.0, 0.0]), 1.0)
    term = ForwardTerm(_Line(), data)
    signed = measure(Fit(_sky(0.0, 0.0), free_signed, (term,)), start)
    np.testing.assert_allclose(signed.value["amplitude"], -3.0, atol=1e-6)
    assert bool(signed.regular)
    positive = measure(Fit(_sky(0.0, 0.0), free_positive, (term,)), start)
    assert not bool(positive.regular)
    assert float(positive.value["amplitude"]) < 1e-6

    flat_data = gaussian_loglike(jnp.zeros(2), 1.0)
    flat_fit = Fit(_sky(0.0, 0.0), free_signed, (ForwardTerm(_Line(), flat_data),))
    flat = measure(
        flat_fit, {"amplitude": jnp.asarray(0.0), "pa_deg": jnp.asarray(10.0)}
    )
    assert float(flat.grad_norm) == 0.0
    assert not bool(flat.regular)
    assert float(flat.min_eigenvalue) <= 0.0

    del _FORWARD_CALLS[:]
    counting = Fit(
        _sky(0.0, 0.0), free_signed, (ForwardTerm(_CountingLine(), flat_data),)
    )
    with pytest.raises(MemoryLimit):
        measure(counting, start, max_bytes=10)
    assert _FORWARD_CALLS == []  # refused before any evaluation of the model


class _Square(eqx.Module):
    """``model -> (a + 0.4 a**2, a * pa_deg / 360)``, nonlinear in ``a``."""

    def __call__(self, model):
        return jnp.stack(
            [
                model.amplitude + 0.4 * model.amplitude**2,
                model.amplitude * model.pa_deg / 360.0,
            ]
        )


class _Root(eqx.Module):
    """``model -> (a sqrt(1 + a), 0)``: not finite below ``a = -1``."""

    def __call__(self, model):
        a = model.amplitude
        return jnp.stack([a * jnp.sqrt(1.0 + a), 0.0 * a])


class _SquareNoSplit(ForwardTerm):
    """The same forward term without ``split``: ``detect`` renders at every step."""

    def split(self, model):
        return AbstractTerm.split(self, model)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_render_once_detect_holds_only_for_a_prediction_linear_in_the_amplitude():
    """``detect`` checks the render-once path and answers a nonlinear forward.

    The forward is ``(a + 0.4 a**2, a * pa / 360)`` observed with unit noise at
    ``(1, 0)``; at ``pa = 0`` the likelihood is maximal at
    ``a = (sqrt(2.6) - 1) / 0.8 = 0.766``. The render-once path alone takes the
    tangent at ``a = 0``, whose slope is one, and finds ``a = 1``; the linearity check
    (the render at ``a = 1`` is 1.4 against the linear 1.0) sends the trial to the
    Newton path through the fit's likelihood, which finds the root. A linear forward
    keeps the render-once answer, equal to the Newton path's to rounding. The same
    holds under ``filter_jit``, where every trial takes the Newton path. A render that
    is not finite at the amplitude found (``a sqrt(1 + a)`` against data -2 gives
    ``a = -2`` from the tangent) is a departure, not a masked pixel: Newton finds the
    likelihood's maximum at the forward's minimum, ``a = -2/3``.

    Truth source: the closed-form root of ``a + 0.4 a**2 = 1``; the stationary point
    of ``a sqrt(1 + a)``, ``a = -2/3``.
    Tolerance basis: exact-algebra (atol 1e-6 for the roots, atol 1e-9 between the two
    paths on the linear forward).
    Fails if: the fallback is not taken for a term without ``split``, the linearity
    check is removed (``a_hat`` then comes back as 1.0), it is skipped under a
    transformation, a non-finite render passes it, or it rejects a linear forward.
    """
    data = gaussian_loglike(jnp.array([1.0, 0.0]), 1.0)
    free = {"amplitude": Uniform(-5.0, 5.0), "pa_deg": Uniform(-200.0, 200.0)}
    trials = {"pa_deg": jnp.zeros(1)}
    root = (np.sqrt(2.6) - 1.0) / 0.8
    nested = Fit(_sky(0.0, 0.0), free, (_SquareNoSplit(_Square(), data),))
    once = Fit(_sky(0.0, 0.0), free, (ForwardTerm(_Square(), data),))
    found = detect(nested, trials, amplitude="amplitude", n_newton=30)
    np.testing.assert_allclose(found.a_hat, root, atol=1e-6)
    checked = detect(once, trials, amplitude="amplitude", n_newton=30)
    np.testing.assert_allclose(checked.a_hat, root, atol=1e-6)
    np.testing.assert_allclose(checked.z, found.z, atol=1e-6)
    jitted = eqx.filter_jit(
        lambda f, t: detect(f, t, amplitude="amplitude", n_newton=30)
    )(once, trials)
    np.testing.assert_allclose(jitted.a_hat, root, atol=1e-6)
    low = Fit(
        _sky(0.0, 0.0),
        free,
        (ForwardTerm(_Root(), gaussian_loglike(jnp.array([-2.0, 0.0]), 1.0)),),
    )
    edge = detect(low, trials, amplitude="amplitude", n_newton=30)
    np.testing.assert_allclose(edge.a_hat, -2.0 / 3.0, atol=1e-6)
    linear_data = gaussian_loglike(jnp.array([2.0, 0.0]), 1.0)
    line = Fit(_sky(0.0, 0.0), free, (ForwardTerm(_Line(), linear_data),))
    line_nested = Fit(_sky(0.0, 0.0), free, (_SquareNoSplit(_Line(), linear_data),))
    np.testing.assert_allclose(
        detect(line, trials, amplitude="amplitude").a_hat,
        detect(line_nested, trials, amplitude="amplitude").a_hat,
        atol=1e-9,
    )


_FORWARD_CALLS = []  # one entry per trace or call of ``_CountingLine``


class _CountingLine(eqx.Module):
    """``_Line`` that records each time it is evaluated or traced."""

    def __call__(self, model):
        _FORWARD_CALLS.append(1)
        return jnp.stack([model.amplitude, model.amplitude * model.pa_deg / 360.0])


class _Line(eqx.Module):
    """``model -> (amplitude, amplitude * pa_deg / 360)``."""

    def __call__(self, model):
        return jnp.stack([model.amplitude, model.amplitude * model.pa_deg / 360.0])


class _Assuming(ForwardTerm):
    """A forward term that names the assumptions it is built with."""

    named: tuple[str, ...] = eqx.field(static=True, default=())

    @property
    def assumptions(self):
        return self.named


def test_the_detection_results_copy_the_fit_assumptions():
    """Fails if ``detect``, ``measure`` or a ``SheetSum`` loses the fit's assumptions.

    Truth source: the names the terms declare; ``Fit.assumptions`` is their sorted
    union (a name two terms share appears once), and a ``SheetSum`` reports the
    union of the terms it holds. Tolerance basis: exact (tuples of strings). A
    cheap Gaussian forward term stands in for an image term, since the record is
    generic over term types.
    """
    data = gaussian_loglike(jnp.array([1.0, 0.5]), 1.0)
    terms = (
        _Assuming(_Line(), data, ("no_drift", "flat_field")),
        _Assuming(_Line(), data, ("no_drift",)),
    )
    free = {"amplitude": Uniform(-5.0, 5.0), "pa_deg": Uniform(0.0, 360.0)}
    fit = Fit(_sky(0.0, 0.0), free, terms)
    assert fit.assumptions == ("flat_field", "no_drift")
    trials = {"pa_deg": jnp.array([90.0, 180.0])}
    assert detect(fit, trials, amplitude="amplitude").assumptions == fit.assumptions
    start = {"amplitude": jnp.asarray(1.0), "pa_deg": jnp.asarray(180.0)}
    assert measure(fit, start, n_newton=1).assumptions == fit.assumptions
    assert Fit(_sky(0.0, 0.0), free, (ForwardTerm(_Line(), data),)).assumptions == ()
    sheets = SheetSum(terms, orbit_path="orbits")
    assert sheets.assumptions == ("flat_field", "no_drift")

    class _Stack(eqx.Module):
        quat: jax.Array

    class _SkyWithOrbit(_Sky):
        orbits: _Stack

    sky = _sky(0.0, 0.0)
    held = _SkyWithOrbit(sky.amplitude, sky.pa_deg, _Stack(jnp.zeros((1, 4))))
    assert Fit(held, free, (sheets,)).assumptions == sheets.assumptions


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.medium(reason="forward-mode tangents through the renderer, two kinds")
@pytest.mark.parametrize("kind", ["poisson", "constant"])
def test_fisher_of_an_image_fit_equals_the_closed_form_on_a_linear_mean(kind):
    """``Fit.information(kind="fisher")`` over an ``ImageTerm`` is ``sum t**2 / var``.

    The mean is ``m0 + a t`` in the amplitude ``a``, so the information of ``a`` is
    ``sum_used t**2 / m`` for a Poisson law (variance equal to the mean) and
    ``sum_used t**2 / V`` for a Gaussian of fixed variance ``V``. Pixels the mask
    removes carry none. The position angle's entries are checked against a central
    difference of the rendered mean.

    Truth source: ``t = predict(a=1) - predict(a=0)`` and the mean from ``predict``
    (the renderer, not ``fisher_root``), NumPy sums over the unmasked pixels; the
    position-angle Jacobian by a central difference of step 0.01 degree.
    Tolerance basis: exact-algebra for the amplitude entry (rtol 1e-10); floating-point
    for the entries with the position angle (rtol 1e-4, the central difference error).
    Fails if: the mask is dropped from the root, the root is the square root of the
    mean (half the Poisson information), or the variance enters twice.
    """
    fit, _, mask = _image_problem(kind)
    sky = fit.terms[0]
    point = _sky(_TRUE_AMP, _TRUE_PA)
    x = {"amplitude": point.amplitude, "pa_deg": point.pa_deg}
    mean = np.asarray(sky.predict(point))
    t = np.asarray(sky.predict(_sky(1.0, _TRUE_PA))) - np.asarray(
        sky.predict(_sky(0.0, _TRUE_PA))
    )
    variance = mean if kind == "poisson" else _VARIANCE
    got = np.asarray(fit.information(x, kind="fisher", include_prior=False))
    # Only scored pixels have a positive mean; the rest are 0 / 0 and carry no weight.
    w = np.divide(mask, variance, out=np.zeros(mask.shape), where=mask)
    np.testing.assert_allclose(got[0, 0], np.sum(w * t * t), rtol=1e-10)
    h = 0.01
    pred = lambda pa: np.asarray(sky.predict(_sky(_TRUE_AMP, pa)))  # noqa: E731
    d_pa = (pred(_TRUE_PA + h) - pred(_TRUE_PA - h)) / (2 * h)
    np.testing.assert_allclose(got[0, 1], np.sum(w * t * d_pa), rtol=1e-4)
    np.testing.assert_allclose(got[1, 1], np.sum(w * d_pa * d_pa), rtol=1e-4)
    with_prior = fit.information(x, kind="fisher", include_prior=True)
    assert np.all(np.isfinite(with_prior))
