"""The pixel likelihood ``ImageTerm`` over ``coronagraphoto.expected``."""

from dataclasses import replace

import coronagraphoto as cp
import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import canonical
from hwoutils.model_paths import resolve
from hwoutils.precision import PrecisionError
from hwoutils.testing import ulps
from image_scene import (
    BANDS,
    GaussianCoronagraph,
    make_data,
    make_detector,
    make_exposure,
    make_instrument,
    make_scene,
)
from optixstuff import (
    AbstractSpeckleField,
    Detector,
    ExposureData,
    Rates,
    stack_exposure_data,
)
from optixstuff.coronagraph import MultiBandCoronagraph
from optixstuff.exposure import FramePlan
from optixstuff.grids import Image, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Quadrature
from priorities import LogUniform, Normal
from scipy import stats
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar

from photomancy import detection
from photomancy.image import ImageForward, ImageModel, ImageTerm, _ImageScore
from photomancy.infer import MAP, Fit, run
from photomancy.infer.likelihoods import MaskedLogLikelihood, gaussian_loglike
from photomancy.infer.terms import ForwardTerm
from photomancy.orbit.sheets import SheetSum

_PATHS = {"scene_path": "scene", "instrument_path": "instrument"}


def _setup(kind="poisson", contrast=1e-10):
    instrument = make_instrument(make_detector(kind))
    scene = make_scene(contrast=contrast, sky_pa_deg=40.0)
    exposure = make_exposure(scene, instrument)
    return ImageModel(scene, instrument), exposure


def _scored(model, exposure, observed):
    """Score the pixels covered by the toy coronagraph at the reference model."""
    return ImageTerm.on_support(model, observed, exposure=exposure, **_PATHS)


def _mean(model, exposure):
    shape = (1, *model.instrument.detector.shape)
    return ImageTerm(
        jnp.zeros(shape), jnp.ones(shape, bool), exposure=exposure, **_PATHS
    ).predict(model)


@pytest.fixture(scope="module")
def poisson_image():
    """The shared Poisson scene, exposure and noiseless image."""
    model, exposure = _setup("poisson")
    return model, exposure, _mean(model, exposure)


@pytest.mark.float32_only(reason="image log densities require float64")
def test_image_log_density_refuses_float32():
    """Both the direct term and the reusable scorer refuse float32 reductions."""
    model, exposure = _setup()
    shape = (1, *model.instrument.detector.shape)
    term = ImageTerm(
        jnp.zeros(shape), jnp.ones(shape, bool), exposure=exposure, **_PATHS
    )
    with pytest.raises(PrecisionError, match=r"image log likelihood.*Enable x64"):
        term(model)
    rates, score = term.split(model)
    with pytest.raises(PrecisionError, match=r"image log likelihood.*Enable x64"):
        score(rates)


@pytest.mark.float64_only(reason="float64 image log-density reduction")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_poisson_value_is_the_summed_pmf_of_the_expected_counts(poisson_image):
    """A photon-counting detector scores the Poisson pmf of ``predict`` per pixel.

    Truth source: ``scipy.stats.poisson.logpmf`` at the expected counts, summed over the
    scored pixels only (a masked pixel holds NaN and must not reach the value).
    Tolerance basis: exact-algebra (rtol 1e-12; the same sum in float64).
    Fails if: the mask is applied after the sum, a NaN at a masked pixel leaks, or the
    density is not normalized.
    """
    model, exposure, mean = poisson_image
    mean = np.asarray(mean)
    counts = np.random.default_rng(3).poisson(mean).astype(float)
    mask = np.zeros(counts.shape, bool)
    mask[..., 4:-4, 4:-4] = True
    observed = np.where(mask, counts, np.nan)
    term = ImageTerm(observed, mask, exposure=exposure, **_PATHS)
    want = stats.poisson.logpmf(counts[mask], mean[mask]).sum()
    np.testing.assert_allclose(term(model), want, rtol=1e-12)
    grad = eqx.filter_grad(term)(model)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(grad))


@pytest.mark.float64_only(reason="float64 image log-density reduction")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_gaussian_value_is_the_summed_density_of_the_detector_moments():
    """A read-noise detector scores the Gaussian of its own mean and variance.

    Truth source: ``scipy.stats.norm.logpdf`` with the detector's
    ``mean_and_variance`` at the model's expected rates, summed over covered pixels.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: the variance is dropped or the log-determinant constant is missing.
    """
    model, exposure = _setup("read")
    detector = model.instrument.detector
    probe = _scored(model, exposure, jnp.zeros((1, *detector.shape)))
    _, rates, plan = probe._rates(model)
    mean, variance = (np.asarray(x) for x in detector.mean_and_variance(rates, plan))
    observed = mean + np.sqrt(variance) * np.random.default_rng(4).normal(
        size=mean.shape
    )
    term = eqx.tree_at(lambda t: t.observed_e, probe, observed)
    want = stats.norm.logpdf(observed, mean, np.sqrt(variance))[
        np.asarray(term.mask)
    ].sum()
    np.testing.assert_allclose(term(model), want, rtol=1e-12)


@pytest.mark.float64_only(reason="Fit residuals and image log densities require x64")
def test_subpixel_count_difference_survives_a_float32_detector(monkeypatch):
    """One pixel differs by one electron, a quarter of a float32 spacing.

    A Gaussian detector with variance three gives residual 1/sqrt(3) and
    log density -1/6 - log(6 pi)/2. Four double-precision roundings cover the
    square root, division, logarithm and addition. Standalone counts
    retain their precision, and a whole-spacing offset tests Fit's
    whitening precision when all detector and observation inputs are narrow.
    """
    model, exposure = _setup("constant")
    detector = canonical(make_detector("constant", variance_e2=3), precision="float32")
    plan = FramePlan(jnp.float32(1), 1)
    shape = (1, *detector.shape)
    rates = Rates(BANDS, "electron", planet=jnp.full(shape, 2**25, jnp.float32))
    monkeypatch.setattr(
        ImageTerm, "_rates", lambda self, model: (detector, rates, plan)
    )
    pixel = (0, shape[1] // 2, shape[2] // 2)
    mask = jnp.zeros(shape, bool).at[pixel].set(True)
    mean, variance = detector.mean_and_variance(rates, plan)
    assert mean.dtype == variance.dtype == jnp.float32
    observed = mean.astype(jnp.float64) + 1
    assert observed[pixel].astype(jnp.float32) == mean[pixel]
    term = ImageTerm(observed, mask, exposure=exposure, **_PATHS)
    np.testing.assert_array_max_ulp(term(model), -1 / 6 - np.log(6 * np.pi) / 2, 4)
    np.testing.assert_array_max_ulp(term.residual(model)[pixel], 1 / np.sqrt(3), 4)

    narrow = eqx.tree_at(lambda t: t.observed_e, term, mean + jnp.float32(4))
    assert narrow.residual(model).dtype == jnp.float32
    fit = Fit(model, {_CONTRAST: LogUniform(1e-12, 1e-8)}, (narrow,))
    point = {_CONTRAST: _contrast(model)}
    (residual,) = fit.residuals(point)
    assert residual.dtype == jnp.float64
    np.testing.assert_array_max_ulp(residual[pixel], 4 / np.sqrt(3), 4)
    with jax.enable_x64(False), pytest.raises(PrecisionError, match=r"Fit.residuals"):
        fit.residuals(point)


@pytest.mark.float64_only(reason="float64 image log-density reduction")
@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("kind", ["constant", "read", "poisson"])
def test_excluded_nonfinite_rates_do_not_enter_the_score_gradient(kind):
    """Only the first of two pixels contributes to the score and its gradient.

    Truth source: the gradient with finite rates at the excluded pixel and the
    same scored rates. With unit mean and shot variance rates and zero counts,
    the Poisson derivative with respect to live time is -1. The unit-variance
    Gaussian derivative is minus the live time. For read noise of 3 electrons,
    the score is ``-(time**2 / (time + 9) + log(2*pi*(time + 9))) / 2``, whose
    derivative at one second is -0.145.
    Tolerance basis: floating-point (rtol 1e-12 for float64 gradients).
    Fails if: a likelihood derivative at the excluded pixel enters the reverse
    pass, or a NaN rate at a scored pixel gives a finite score.
    """
    mask = jnp.array([[[True, False]]])
    detector = make_detector(kind)

    def score(time, mean_rate, variance_rate):
        variance = Rates(BANDS, "electron", thermal=variance_rate.reshape(mask.shape))
        rates = Rates(
            BANDS,
            "electron",
            star_leakage=mean_rate.reshape(mask.shape),
            variance=variance,
            supported=mask,
        )
        return _ImageScore(
            jnp.zeros(mask.shape), mask, detector, FramePlan(time, 1), False
        )(rates)

    evaluate = jax.jit(jax.vmap(jax.value_and_grad(score)))
    values, gradients = evaluate(
        jnp.ones(6),
        jnp.array([[1, 2], [1, jnp.nan], [1, jnp.inf], [1, 2], [jnp.nan, 2], [1, 2]]),
        jnp.array([[1, 2], [1, 2], [1, 2], [1, jnp.nan], [1, 2], [jnp.nan, 2]]),
    )
    assert np.isfinite(values[:4]).all()
    np.testing.assert_array_equal(values[1:4], np.broadcast_to(values[0], (3,)))
    assert np.isfinite(gradients[:4]).all()
    np.testing.assert_allclose(gradients[1:4], gradients[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        gradients[0], -0.145 if kind == "read" else -1.0, rtol=1e-12
    )
    assert np.isnan(values[4:]).all()


@pytest.mark.float64_only(reason="float64 image log-density reduction")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_exclusion_preserves_the_finite_detector_floor():
    """The detector uses the finite frame peak to classify negative totals.

    Truth source: the excluded sources cancel in the total and contribute 1000
    to the peak of the sum of absolute component rates. The tolerance is 1e-4
    times that peak. At zero observed counts the Poisson score for a rate
    clipped to zero is zero, and a rate below the tolerance gives minus infinity.
    Tolerance basis: exact-algebra (exact zero and negative infinity).
    Fails if: masking reduces the floor tolerance and rejects a scored rate
    that the original finite frame would clip to zero.
    """
    mask = jnp.array([[[True, False]]])
    score = _ImageScore(
        jnp.zeros(mask.shape), mask, make_detector("poisson"), FramePlan(1, 1), False
    )

    def rates(first):
        return Rates(
            BANDS,
            "electron",
            planet=jnp.array([[[first, 500.0]]]),
            star_leakage=jnp.array([[[0.0, -500.0]]]),
        )

    assert score(rates(-0.05)) == 0.0
    assert np.isneginf(score(rates(-0.2)))


_READ_NOISE_E = {"poisson": 0.0, "read": 3.0}


def _contrast(model):
    return model.scene.planets[0].physical_model.contrast_grid


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("kind", ["poisson", "read"])
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_residual_is_the_offset_in_detector_standard_deviations(kind):
    """The residual returns the injected offsets in units of the detector's sigma.

    Truth source: two frames built as ``mean + k * sqrt(variance)`` with known
    ``k``, the variance typed from primitives: the shot noise equals the mean (no
    gain excess, dark or clock-induced charge), plus the read noise squared once
    for the single frame (3 electrons for ``read``, none for the photon counter,
    whose residual is then the Pearson residual). ``Fit.residuals`` must give the
    same array at ``x``.
    Tolerance basis: floating-point (rtol 1e-12, and an absolute floor of 8 ulp of
    the largest ``mean / sigma``, the rounding of ``mean + k sigma`` before it is
    subtracted again).
    Fails if: the residual is signed the wrong way, whitened by ``sqrt(mean)`` (the
    Poisson part only) where the detector adds read noise, mixes the frames of a
    batch, or a masked pixel is not NaN.
    """
    model, exposure = _setup(kind)
    detector = model.instrument.detector
    read = _READ_NOISE_E[kind]
    assert float(detector.read_noise_e_rms) == read
    assert float(detector.excess_noise_factor_sq) == 1.0
    assert float(detector.dark_current_e_per_pixel_s) == 0.0
    assert float(detector.clock_induced_charge_e_per_pixel_frame) == 0.0
    assert exposure.plan.n_frames == 1
    mean = np.asarray(_mean(model, exposure))
    variance = mean + read**2
    k = np.random.default_rng(6).normal(size=(2, *mean.shape))
    observed = mean + k * np.sqrt(variance)
    term = _scored(model, exposure, observed)
    mask = np.broadcast_to(np.asarray(term.mask), observed.shape)
    out = np.asarray(term.residual(model))
    assert out.shape == observed.shape
    scored = np.asarray(term.mask)
    floor = 8 * np.finfo(float).eps * np.max(mean[scored] / np.sqrt(variance[scored]))
    np.testing.assert_allclose(out[mask], k[mask], rtol=1e-12, atol=floor)
    assert np.all(np.isnan(out[~mask])) and not mask.all()
    fit = Fit(model, {_CONTRAST: LogUniform(1e-12, 1e-8)}, (term,))
    x = {_CONTRAST: model.scene.planets[0].physical_model.contrast_grid}
    (from_fit,) = fit.residuals(x)
    np.testing.assert_array_equal(from_fit, out)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_residual_reads_the_floored_rates_and_is_safe_to_differentiate():
    """The residual uses the rates ``log_prob`` scores and has a finite Jacobian.

    A planet of negative contrast drives the total rate at its peak pixel just
    below zero, inside the detector's floor tolerance (half of 1e-4 of the frame
    peak), where ``log_prob`` scores a rate of zero.
    Truth source: there the mean is 0 and the variance is the read noise squared
    (9, typed from primitives), so the residual of 2 electrons is ``2 / 3``; the
    Jacobian at the noiseless observation is minus the planet counts divided
    by the noise, since the residual numerator is zero. Tolerance basis: one
    division for the floored residual, and eight operations at the full mean
    over noise scale for two rendered means, subtraction and whitening.
    Fails if: the residual whitens the unfloored (negative) mean, a masked or
    undefined pixel puts NaN into ``jax.jacfwd``, or a photon-counter pixel whose
    total is a tenth of the dark rate below zero (``log_prob`` is ``-inf``, the
    variance zero) puts NaN into the reverse-mode gradient of the chi-square.
    """
    model, exposure = _setup("read")
    term = _scored(model, exposure, _mean(model, exposure))
    grid = model.scene.planets[0].physical_model.contrast_grid
    dark = np.asarray(_mean(eqx.tree_at(_contrast, model, 0.0 * grid), exposure))
    lit = np.asarray(_mean(model, exposure))
    planet = np.where(np.asarray(term.mask), lit - dark, 0.0)
    peak = np.unravel_index(np.argmax(planet), planet.shape)
    eps = 0.5e-4
    scale = -dark[peak] * (1 + eps) / (planet[peak] * (1 - eps))
    negative = eqx.tree_at(_contrast, model, scale * grid)
    observed = np.zeros(planet.shape)
    observed[peak] = 2.0
    out = np.asarray(
        eqx.tree_at(lambda t: t.observed_e, term, observed).residual(negative)
    )
    np.testing.assert_allclose(
        out[peak], 2.0 / 3.0, rtol=float(ulps(1, jnp.ones_like(out[peak])))
    )

    def residual_at(s):
        return term.residual(eqx.tree_at(_contrast, model, s * grid))

    jac = np.asarray(jax.jacfwd(residual_at)(1.0))
    assert np.all(np.isfinite(jac))
    noise = np.sqrt(lit + 9.0)
    used = np.asarray(term.mask)
    assert np.all(np.isnan(np.asarray(residual_at(1.0))[~used]))
    np.testing.assert_allclose(
        jac[used],
        -planet[used] / noise[used],
        rtol=0,
        atol=float(ulps(8, jnp.max(jnp.asarray(lit[used] / noise[used])))),
    )

    counter, counter_exposure = _setup("poisson")
    counted = _scored(counter, counter_exposure, _mean(counter, counter_exposure))
    below = -dark[peak] * 1.1 / planet[peak]

    def chi_square(s):
        r = counted.residual(eqx.tree_at(_contrast, counter, s * grid))
        return jnp.sum(jnp.where(jnp.isnan(r), 0.0, r) ** 2)

    undefined = counted.residual(eqx.tree_at(_contrast, counter, below * grid))
    assert np.isnan(np.asarray(undefined)[peak])
    assert np.isfinite(jax.grad(chi_square)(below))


@pytest.mark.parametrize("kind", ["poisson", "read", "custom"])
def test_residual_domain_agrees_with_detector_density(kind, monkeypatch, precision):
    """Positive variance and finite detector log density define a residual.

    A separate shot-variance rate tests both rate floors, even for a counter
    with a zero mean and positive variance. A custom two-point readout assigns
    equal probability to the two values one standard deviation from its mean.
    Membership of the density domain is exact. The default detectors exercise
    adjacent values at the float32 floor with a peak of 13, where the product
    rounds. Fails if the residual or score uses a working-dtype threshold.
    """

    class TwoPoint(Detector):
        def log_prob(self, observed, rates, plan):
            mean, variance = self.mean_and_variance(rates, plan)
            return jnp.where(
                (observed - mean) ** 2 == variance, -jnp.log(2.0), -jnp.inf
            )

    detector = (
        TwoPoint(pixel_scale_arcsec=1.0, shape=(1, 8))
        if kind == "custom"
        else make_detector(kind)
    )
    observed = jnp.asarray([[[0, 1, 2, 2, 13, 2, -1, -0.5]]])
    mean = jnp.asarray([[[0, 0, 1, 1, 13, -1, 1, 1]]], dtype=precision)
    variance = jnp.asarray([[[1, 1, 1, 1, 1, 1, 1, 1]]], dtype=precision)
    if kind == "custom":
        mean = mean.at[0, 0, 4].set(12)
        variance = variance.at[0, 0, 3].set(-1)
    if kind == "poisson":
        observed = observed.at[0, 0, 5].set(0)
    plan = FramePlan(1.0, 1)
    term = ImageTerm(
        observed,
        jnp.ones(observed.shape, bool),
        instrument_path="det",
        exposure_path="exp",
    )
    monkeypatch.setattr(
        ImageTerm, "_rates", lambda self, rates: (detector, rates, plan)
    )

    def residual_valid(rates):
        return jnp.isfinite(term.residual(rates))

    floor = np.float32(-0.0013)
    for rate, admitted in (
        (floor, True),
        (np.nextafter(floor, np.float32(-np.inf)), False),
    ):
        rates = Rates(
            BANDS,
            "electron",
            planet=mean.at[0, 0, 5].set(rate if kind != "custom" else -1),
            variance=Rates(BANDS, "electron", planet=variance),
        )
        # At a floored zero the Poisson density admits only a zero count.
        # The readout with nonzero read noise can also admit a positive count.
        expected = {
            "poisson": [True, False, True, True, True, admitted, False, True],
            "read": [True, True, True, True, True, admitted, True, True],
            "custom": [False, True, True, False, True, False, False, False],
        }[kind]
        for evaluate in (residual_valid, eqx.filter_jit(residual_valid)):
            np.testing.assert_array_equal(evaluate(rates).ravel(), expected)
        if kind == "custom" or precision == np.float64:
            np.testing.assert_array_equal(
                jnp.isfinite(detector.log_prob(observed, rates, plan)).ravel(), expected
            )


@pytest.mark.float64_only(reason="float64 image log-density reduction")
def test_the_value_peaks_at_the_true_model(poisson_image):
    """Noise-free counts are likelier at the truth than at 5 percent off."""
    model, exposure, mean = poisson_image
    term = _scored(model, exposure, jnp.round(mean))
    off = _setup("poisson", contrast=1.05e-10)[0]
    assert term(model) > term(off)


@pytest.mark.slow(
    reason="compiles a Poisson image simulation and checks the count statistics, 5 "
    "s projected on CI"
)
def test_draw_gives_counts_consistent_with_the_expectation_and_keeps_masked_pixels(
    poisson_image,
):
    """Drawn counts are integers about the mean; masked pixels keep their stored value.

    Tolerance: with over 350 scored Poisson pixels the standard errors of the
    standard deviation and mean are below 0.04 and 0.06. The bounds are 0.2 and 0.25.
    """
    model, exposure, mean = poisson_image
    supported = _scored(model, exposure, mean).mask
    mask = supported & jnp.ones(mean.shape, bool).at[0, :3, :].set(False)
    term = ImageTerm(jnp.full(mean.shape, -7.0), mask, exposure=exposure, **_PATHS)
    drawn = term.draw(jax.random.PRNGKey(5), model)
    out = np.asarray(drawn.observed_e)
    assert np.all(out[0, :3] == -7.0)
    kept = np.asarray(mask)
    assert np.all(out[kept] == np.round(out[kept]))
    z = (out[kept] - np.asarray(mean)[kept]) / np.sqrt(np.asarray(mean)[kept])
    assert abs(z.std() - 1.0) < 0.2 and abs(z.mean()) < 0.25


@pytest.mark.float64_only(reason="float64 image log-density reduction")
@pytest.mark.slow(
    reason="compiles scoring and Poisson draws on a batch of images, 9 s projected "
    "on CI"
)
def test_batched_frames_are_scored_jointly(poisson_image):
    """A batch of two frames scores as the sum of the two single frames."""
    model, exposure, mean = poisson_image
    mean = np.asarray(mean)
    rng = np.random.default_rng(6)
    frames = rng.poisson(np.broadcast_to(mean, (2, *mean.shape))).astype(float)
    both = _scored(model, exposure, frames)
    singles = [_scored(model, exposure, f) for f in frames]
    np.testing.assert_allclose(
        both(model), sum(float(t(model)) for t in singles), rtol=1e-12
    )
    drawn = both.draw(jax.random.PRNGKey(0), model).observed_e
    assert drawn.shape == frames.shape
    assert not np.array_equal(drawn[0], drawn[1])


@pytest.mark.float64_only(reason="float64 image log-density reduction")
def test_exposure_comes_from_the_model_by_path_or_the_term_not_both(poisson_image):
    """A path exposure equals a held one; giving both or neither fails."""
    model, exposure, mean = poisson_image
    observed = jnp.round(mean)
    held = _scored(model, exposure, observed)
    by_path = ImageTerm(observed, held.mask, exposure_path="exposures[0]", **_PATHS)
    with_exposure = ImageModel(model.scene, model.instrument, (exposure,))
    np.testing.assert_allclose(by_path(with_exposure), held(model), rtol=1e-12)
    with pytest.raises(ValueError, match="exactly one"):
        ImageTerm(observed, jnp.ones(observed.shape, bool), **_PATHS)
    with pytest.raises(ValueError, match="exactly one"):
        ImageTerm(
            observed,
            jnp.ones(observed.shape, bool),
            exposure=exposure,
            exposure_path="exposures[0]",
            **_PATHS,
        )
    with pytest.raises(ValueError, match="FramePlan"):
        ImageTerm(
            observed,
            jnp.ones(observed.shape, bool),
            exposure=eqx.tree_at(
                lambda e: e.plan, exposure, None, is_leaf=lambda x: x is None
            ),
            **_PATHS,
        )


@pytest.mark.float64_only(reason="float64 image log-density reduction")
def test_the_term_runs_on_a_scene_rooted_model_with_a_held_instrument(poisson_image):
    """The model may be the scene itself, with the instrument held by the term.

    Truth source: the same frame scored through an ``ImageModel`` with both paths.
    Tolerance basis: exact (rtol 1e-12; the same renderer on the same leaves).
    Fails if: the empty scene path does not name the model, or the held instrument is
    ignored (``Fit(scene, ...)`` is the model of every orbit posterior).
    """
    model, exposure, mean = poisson_image
    observed = jnp.round(mean)
    mask = _scored(model, exposure, observed).mask
    rooted = ImageTerm(observed, mask, instrument=model.instrument, exposure=exposure)
    held = _scored(model, exposure, observed)
    np.testing.assert_allclose(rooted(model.scene), held(model), rtol=1e-12)
    with pytest.raises(ValueError, match="instrument"):
        ImageTerm(observed, mask, exposure=exposure)
    with pytest.raises(ValueError, match="instrument"):
        ImageTerm(
            observed,
            mask,
            instrument=model.instrument,
            instrument_path="instrument",
            exposure=exposure,
        )


def test_the_term_declares_no_reads():
    """The held exposure caches no scene values, so the term declares none.

    The exposure builder holds the knots of the tabulated sources it resolved as
    static data, so there is nothing to refuse a free leaf for. The truth is the
    empty tuple itself, so there is no tolerance.
    """
    model, exposure = _setup("poisson")
    observed = jnp.zeros((1, *model.instrument.detector.shape))
    assert exposure.resolution.rules
    assert _scored(model, exposure, observed).reads == ()


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_an_observed_image_is_checked_against_the_held_detector_and_roll():
    """An ``Image`` from ``observe`` scores as its data; foreign data are refused.

    Truth source: the same term built from the bare ``Image.data`` array, and, for
    ``on_support``, the explicit mask ``supported_mask(model)`` (the toy detector is
    wider than its tables, 69 percent supported).
    Tolerance basis: exact (the same pixels and the same renderer, compared bit for
    bit); angles equal modulo 360 to the float32 resolution of the angle.
    Fails if: the term drops the data of an ``Image``; accepts data on another
    pitch, at another roll (also with the instrument read from the model), with no
    orientation where the roll is known, or in another quantity or measure; refuses
    a roll of 360 against 0 or a float32 copy of the roll; or ``on_support`` scores
    anything but the supported pixels (all True is refused by ``Fit``).
    """
    model, exposure = _setup("poisson")
    instrument = model.instrument
    image = cp.observe(jax.random.PRNGKey(2), model.scene, instrument, exposure)
    mask = _scored(model, exposure, image.data).mask
    assert not bool(jnp.all(mask))
    from_image = ImageTerm(image, mask, instrument=instrument, exposure=exposure)
    from_array = ImageTerm(image.data, mask, instrument=instrument, exposure=exposure)
    assert float(from_image(model.scene)) == float(from_array(model.scene))
    supported = ImageTerm.on_support(
        model.scene, image, instrument=instrument, exposure=exposure
    )
    np.testing.assert_array_equal(
        supported.mask, from_array.supported_mask(model.scene)
    )
    law = {"planets[0].physical_model.contrast_grid": LogUniform(1e-12, 1e-8)}
    fit = Fit(model.scene, law, (supported,))
    x = {k: resolve(model.scene, k) for k in law}
    assert float(fit.log_likelihood(x)) == float(from_array(model.scene))
    grid = image.grid
    coarse = PixelGrid.regular(
        grid.shape, 2.0 * grid.step, grid.unit, orientation=grid.orientation
    )
    with pytest.raises(ValueError, match="detector grid"):
        ImageTerm(
            replace(image, grid=coarse), mask, instrument=instrument, exposure=exposure
        )
    turned = replace(grid, orientation=SkyOrientation(grid.orientation.pa_y_deg + 30))
    with pytest.raises(ValueError, match="roll"):
        ImageTerm(
            replace(image, grid=turned), mask, instrument=instrument, exposure=exposure
        )
    with pytest.raises(ValueError, match="roll"):
        ImageTerm(replace(image, grid=turned), mask, exposure=exposure, **_PATHS)
    bare = replace(grid, orientation=None)
    with pytest.raises(ValueError, match="no sky orientation"):
        ImageTerm(
            replace(image, grid=bare), mask, instrument=instrument, exposure=exposure
        )
    for quantity, measure in (("electrons/s", image.measure), ("electrons", None)):
        measure = Measure.intensive() if measure is None else measure
        foreign = replace(image, quantity=quantity, measure=measure)
        with pytest.raises(ValueError, match="electrons per pixel"):
            ImageTerm(foreign, mask, instrument=instrument, exposure=exposure)
    full_turn = replace(grid, orientation=SkyOrientation(360.0))
    ImageTerm(
        replace(image, grid=full_turn), mask, instrument=instrument, exposure=exposure
    )
    rolled = eqx.tree_at(lambda e: e.pointing.roll_deg, exposure, jnp.asarray(30.1))
    single = replace(grid, orientation=SkyOrientation(float(np.float32(30.1))))
    ImageTerm(replace(image, grid=single), mask, instrument=instrument, exposure=rolled)
    unchecked = ImageTerm(
        replace(image, grid=coarse), mask, exposure=exposure, **_PATHS
    )
    assert float(unchecked(model)) == float(
        ImageTerm(image.data, mask, exposure=exposure, **_PATHS)(model)
    )


_CONTRAST = "scene.planets[0].physical_model.contrast_grid"


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_an_image_fit_frees_a_scene_leaf_and_gradients_flow(poisson_image):
    """``Fit`` accepts the term over a scene leaf: finite density and gradient."""
    model, exposure, mean = poisson_image
    term = _scored(model, exposure, jnp.round(mean))
    fit = Fit(model, {_CONTRAST: LogUniform(1e-12, 1e-8)}, (term,))
    x = {_CONTRAST: model.scene.planets[0].physical_model.contrast_grid}
    assert np.isfinite(fit.log_density(x))
    grad = jax.grad(fit.real_log_prob)(fit.to_real(x))
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad))
    worse = {_CONTRAST: 1.3 * x[_CONTRAST]}
    assert fit.log_likelihood(x) > fit.log_likelihood(worse)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_a_scene_rooted_fit_frees_a_leaf_of_the_scene(poisson_image):
    """A held-instrument term on ``Fit(scene, ...)`` scores like the record model."""
    model, exposure, mean = poisson_image
    observed = jnp.round(mean)
    mask = _scored(model, exposure, observed).mask
    rooted = ImageTerm(observed, mask, instrument=model.instrument, exposure=exposure)
    path = "planets[0].physical_model.contrast_grid"
    law = LogUniform(1e-12, 1e-8)
    grid = model.scene.planets[0].physical_model.contrast_grid
    on_scene = Fit(model.scene, {path: law}, (rooted,)).log_density({path: grid})
    on_record = Fit(
        model, {f"scene.{path}": law}, (_scored(model, exposure, observed),)
    ).log_density({f"scene.{path}": grid})
    np.testing.assert_allclose(on_scene, on_record, rtol=1e-12)


@pytest.mark.float64_only(reason="float64 image log-density reduction")
def test_a_count_where_the_mean_is_zero_is_minus_infinity_not_nan():
    """An impossible Poisson count on a covered pixel gives ``-inf``.

    A detector with zero quantum efficiency has zero expected counts everywhere: a
    zero frame scores 0 and a frame with one count scores ``-inf``, with no NaN.
    """
    from image_scene import make_detector as detector_for

    dead = detector_for("poisson")
    dead = eqx.tree_at(lambda d: d._qe, dead, jnp.asarray(0.0))
    instrument = make_instrument(dead)
    scene = make_scene(contrast=1e-10, sky_pa_deg=40.0)
    exposure = make_exposure(scene, instrument)
    model = ImageModel(scene, instrument)
    shape = (1, *dead.shape)
    zero = _scored(model, exposure, jnp.zeros(shape))
    one = _scored(model, exposure, jnp.zeros(shape).at[0, 5, 5].set(1.0))
    assert float(zero(model)) == 0.0
    assert float(one(model)) == -np.inf


def _wide(kind="poisson", n=48):
    """The toy setup on a detector ``n`` pixels wide: wider than the 24 pixel tables."""
    small = make_detector(kind)
    wide = type(small)(
        photon_counting=True,
        pixel_scale_arcsec=small.pixel_scale_arcsec,
        shape=(n, n),
        qe=0.8,
    )
    instrument = make_instrument(wide)
    scene = make_scene(contrast=1e-10, sky_pa_deg=40.0)
    return ImageModel(scene, instrument), make_exposure(scene, instrument)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.medium(reason="renders on two detector sizes, 8 s cold")
def test_a_mask_on_unsupported_pixels_is_refused_at_fit_setup():
    """Fails if a fit scores pixels the coronagraph tables do not cover.

    Basis: exact. A ``SheetSum`` holding the term is refused the same way and reports
    the term's ``footprint_paths``. On a detector twice the tables, an all-True mask
    scores zero-rate pixels: ``check`` and ``Fit`` raise and name the count (the
    pixels outside ``supported_mask``). With ``mask & supported_mask`` the fit builds,
    and its log density equals that of the same data on a detector cropped to the
    supported square (an all-True mask there) to 1e-10 in the sum of Poisson masses.
    """
    model, exposure = _wide()
    rng = np.random.default_rng(0)
    observed = jnp.asarray(rng.poisson(_mean(model, exposure)).astype(float))
    full = ImageTerm(
        observed, jnp.ones(observed.shape, bool), exposure=exposure, **_PATHS
    )
    supported = np.asarray(full.supported_mask(model))
    n_bad = int((~supported).sum())
    assert 0 < n_bad < supported.size
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        full.check(model)
    law = {_CONTRAST: LogUniform(1e-12, 1e-8)}
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        Fit(model, law, (full,))
    # Held in a SheetSum the term is refused exactly as it is bare, and the sum reports
    # the footprint the term does.
    held = SheetSum((full,), orbit_path="orbits")
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        held.check(model)
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        Fit(model, law, (held,))
    assert held.footprint_paths == full.footprint_paths != ()

    scored = ImageTerm(observed, supported, exposure=exposure, **_PATHS)
    scored.check(model)
    fit = Fit(model, law, (scored,))
    x = {_CONTRAST: model.scene.planets[0].physical_model.contrast_grid * 1.2}

    # The supported pixels are a centred square; crop the detector to it.
    rows, _cols = np.nonzero(supported[0])
    lo, hi = int(rows.min()), int(rows.max()) + 1
    assert supported[0, lo:hi, lo:hi].all() and supported.sum() == (hi - lo) ** 2
    assert lo + hi == supported.shape[-1]
    small = make_detector("poisson")
    crop = type(small)(
        photon_counting=True,
        pixel_scale_arcsec=small.pixel_scale_arcsec,
        shape=(hi - lo, hi - lo),
        qe=0.8,
    )
    instrument = make_instrument(crop)
    cropped = ImageTerm(
        observed[:, lo:hi, lo:hi],
        jnp.ones((1, hi - lo, hi - lo), bool),
        # The wide exposure: the spectral field follows the detector extent, so a
        # rebuilt exposure would place different nodes and the two log densities
        # would differ by quadrature error rather than agree exactly.
        exposure=exposure,
        **_PATHS,
    )
    cropped_model = ImageModel(model.scene, instrument)
    cropped.check(cropped_model)
    other = Fit(cropped_model, law, (cropped,))
    np.testing.assert_allclose(
        fit.log_likelihood(x), other.log_likelihood(x), rtol=0, atol=1e-10
    )


def test_a_lit_planet_outside_the_source_domain_is_refused():
    """Fails if a planet the coronagraph cannot render passes the check as zero light.

    Basis: exact. A coronagraph whose source domain is a disk of 2 lambda/D, and a
    planet at 2.65 lambda/D: with the mask restricted to supported pixels the only
    cause left is the planet, and ``check`` names it. A planet inside the domain
    passes.
    """

    class Bounded(type(make_instrument(make_detector()).coronagraph)):
        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            inside = jnp.hypot(jnp.atleast_1d(x_lod), jnp.atleast_1d(y_lod)) <= 2.0
            return type(psfs)(psfs.image, inside)

    detector = make_detector("poisson")
    instrument = eqx.tree_at(
        lambda i: i.coronagraph, make_instrument(detector), Bounded()
    )
    for separation, refused in ((0.05, True), (0.02, False)):
        scene = make_scene(contrast=1e-10, separation_arcsec=separation)
        exposure = make_exposure(scene, instrument)
        model = ImageModel(scene, instrument)
        observed = jnp.zeros((1, *detector.shape))
        term = _scored(model, exposure, observed)
        if refused:
            with pytest.raises(ValueError, match="outside the coronagraph's source"):
                term.check(model)
        else:
            term.check(model)


@pytest.mark.parametrize(
    "nodes", [[[550, 551]], [[550.0, 551.0]]], ids=["int", "float"]
)
def test_source_domain_uses_fractional_center_with_integer_nodes(nodes):
    """Integer endpoint nodes retain the fractional observing-band center.

    At 550.5 nm, truncating the center to 550 nm moves a 9.995 lambda/D
    source beyond 10 lambda/D. A second host center differs by one quarter
    of a float32 spacing, so both centers round to the same float32 value.
    An explicit wavelength edge admits both of those centers and rejects the
    next float32 grid point. Host admission must match both render paths for
    sources inside and outside the spatial edge. The template is flat so flux
    on the detector is positive exactly when its source is admitted.
    """

    class Bounded(GaussianCoronagraph):
        band_averaged: bool = eqx.field(static=True, default=True)

        def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
            psfs = super().source_psfs(wavelength_nm, x_lod, y_lod, grid=grid)
            covered = Band(550.0, 550.5).contains_wavelength(wavelength_nm)
            return type(psfs)(psfs.image, psfs.valid & covered)

        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            flat = eqx.tree_at(
                lambda image: image.data, psfs.image, jnp.ones_like(psfs.image.data)
            )
            return type(psfs)(flat, jnp.hypot(x_lod, y_lod) <= 10)

    coronagraph = Bounded()
    instrument = eqx.tree_at(
        lambda i: i.coronagraph, make_instrument(make_detector()), coronagraph
    )
    scale = float(coronagraph.grid.frame.lod_to_arcsec(550.5))
    scene = make_scene(
        separation_arcsec=9.995 * scale, model=LambertianPhysicalModel(jnp.ones(1))
    )
    outside = make_scene(
        separation_arcsec=10.005 * scale, model=scene.planets[0].physical_model
    )
    template = make_exposure(scene, instrument)
    observed = jnp.zeros((1, *instrument.detector.shape))

    def rendered(system, exposure):
        return jnp.any(cp.expected(system, instrument, exposure).planet > 0)

    compiled = eqx.filter_jit(rendered)
    for high, admitted in (
        (551.0, True),
        (551.0 + 2**-15, True),
        (551.0 + 2**-13, False),
    ):
        band = Band(550.0, high)
        exposure = eqx.tree_at(
            lambda e: e.quadrature,
            template,
            Quadrature((band,), nodes, [[0.5, 0.5]]),
        )
        model = ImageModel(scene, instrument)
        term = _scored(model, exposure, observed)
        if admitted:
            term.check(model)
        else:
            with pytest.raises(
                ValueError, match="outside the coronagraph's source domain"
            ):
                term.check(model)
        with pytest.raises(ValueError, match="outside the coronagraph's source domain"):
            term.check(ImageModel(outside, instrument))
        for evaluate in (rendered, compiled):
            assert bool(evaluate(scene, exposure)) == admitted
            assert not bool(evaluate(outside, exposure))


@pytest.mark.float64_only(reason="float64 image log-density reduction")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_score_keeps_the_reference_footprint_as_wavelength_nodes_move():
    """Wavelength changes must keep the 400 reference pixels in the density.

    Truth source: each zero observation at zero signal and unit Gaussian variance
    contributes -log(2 pi) / 2. The reference footprint is 20 by 20 pixels.
    Tolerance basis: exact-algebra, absolute tolerance 1e-9 on the float64 sum.
    Fails if: shrinking the footprint drops observations or growing it adds them.
    """
    detector = eqx.tree_at(lambda d: d._qe, make_detector("constant"), jnp.asarray(0.0))
    instrument = make_instrument(detector)
    scene = make_scene()
    exposure = make_exposure(scene, instrument)
    model = ImageModel(scene, instrument, (exposure,))
    term = ImageTerm.on_support(
        model, jnp.zeros((1, *detector.shape)), exposure_path="exposures[0]", **_PATHS
    )
    np.testing.assert_array_equal(term.mask, term.supported_mask(model))
    assert int(term.mask.sum()) == 400
    nodes = exposure.quadrature.nodes_nm

    def scaled(factor):
        return eqx.tree_at(
            lambda m: m.exposures[0].quadrature.nodes_nm, model, factor * nodes
        )

    assert np.isnan(term(scaled(0.9)))
    larger = scaled(1.1)
    assert int(term.supported_mask(larger).sum()) > 400
    np.testing.assert_allclose(
        term(larger), -200 * np.log(2 * np.pi), rtol=0, atol=1e-9
    )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_image_mask_requires_a_boolean_array():
    """The scored pixels must be chosen at construction.

    Truth source: the explicit boolean mask contract.
    Tolerance basis: exact exception checks.
    Fails if: None selects a changing footprint or numeric weights become booleans.
    """
    model, exposure = _setup()
    frame = jnp.zeros((1, *model.instrument.detector.shape))
    with pytest.raises(ValueError, match="on_support"):
        ImageTerm(frame, mask=None, exposure=exposure, **_PATHS)
    with pytest.raises(TypeError, match="bool"):
        ImageTerm(frame, frame, exposure=exposure, **_PATHS)


_TWO_BANDS = (Band(500.0, 540.0), Band(560.0, 600.0))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_image_check_requires_the_resolved_band_and_detector_shape():
    """Only leading observation axes can index additional frames.

    Truth source: the exposure has two bands and the detector has 24 by 24 pixels.
    Tolerance basis: exact shapes for the observation and shared mask.
    Fails if: one observed band broadcasts to two, or a batch-shaped mask passes,
    in the check or in a direct call that skips it.
    """
    model, single = _setup()
    exposure = cp.build_exposure(
        model.scene, model.instrument, _TWO_BANDS, time_d=0.0, plan=single.plan
    )
    model = ImageModel(model.scene, model.instrument, (exposure,))
    shape = (2, 24, 24)
    for observed, mask in [
        ((1, 24, 24), shape),
        ((2, 1, 24), (2, 1, 24)),
        (shape, (1, 24, 24)),
        ((3, *shape), (3, *shape)),
    ]:
        term = ImageTerm(
            jnp.zeros(observed),
            jnp.zeros(mask, bool),
            exposure_path="exposures[0]",
            **_PATHS,
        )
        with pytest.raises(ValueError) as exc:
            term.check(model)
        assert str(shape) in str(exc.value)
        assert str(observed if mask == shape else mask) in str(exc.value)
        with pytest.raises(ValueError, match="must match the render"):
            term(model)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_multiband_check_counts_lit_planets_outside_the_second_component():
    """The second band sees a smaller source domain than the first.

    Truth source: the source radii are four lambda/D at 520 nm and two at 580 nm.
    A separation of 0.02 arcsec is inside both, 0.05 is outside the second only
    and 0.10 is outside both.
    Tolerance basis: exact admission and count for one planet.
    Fails if: the check reads a single coronagraph grid or counts an unlit planet.
    """

    class Component(GaussianCoronagraph):
        bandpass: Band = eqx.field(static=True)
        radius_lod: float = eqx.field(static=True)

        def __init__(self, band, radius_lod):
            super().__init__()
            self.bandpass = band
            self.radius_lod = radius_lod

        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            return type(psfs)(psfs.image, jnp.hypot(x_lod, y_lod) <= self.radius_lod)

    components = (Component(_TWO_BANDS[0], 4.0), Component(_TWO_BANDS[1], 2.0))
    model, single = _setup()
    instrument = eqx.tree_at(
        lambda i: i.coronagraph, model.instrument, MultiBandCoronagraph(components)
    )
    scene = make_scene(separation_arcsec=0.02)
    exposure = cp.build_exposure(
        scene, instrument, _TWO_BANDS, time_d=0.0, plan=single.plan
    )
    frame = jnp.zeros((2, *instrument.detector.shape))
    term = ImageTerm.on_support(scene, frame, instrument=instrument, exposure=exposure)
    term.check(scene)
    term.check(make_scene(separation_arcsec=0.05, contrast=0.0))
    for separation in (0.05, 0.10):
        with pytest.raises(ValueError, match=r"^1 lit planet"):
            term.check(make_scene(separation_arcsec=separation))


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_a_free_leaf_that_moves_the_field_of_view_is_checked_at_both_ends():
    """Fails if a fit frees the band nodes and scores pixels uncovered at an end.

    Basis: exact. The scored pixels are supported at the start nodes (500-600 nm),
    but the tables are fixed in lambda/D, so their footprint on the detector
    shrinks at the blue end: a prior reaching 450 nm makes ``Fit`` raise, a prior
    that stays inside the start nodes builds, and an unbounded prior on a leaf
    that moves the footprint is refused as unverifiable.
    """
    from priorities import Normal, Uniform

    model, exposure = _setup("poisson")
    model = ImageModel(model.scene, model.instrument, (exposure,))
    observed = jnp.zeros((1, *model.instrument.detector.shape))
    probe = ImageTerm(
        observed,
        jnp.ones(observed.shape, bool),
        scene_path="scene",
        instrument_path="instrument",
        exposure_path="exposures[0]",
    )
    term = eqx.tree_at(
        lambda t: t.mask, probe, jnp.asarray(probe.supported_mask(model))
    )
    path = "exposures[0].quadrature.nodes_nm"
    nodes = exposure.quadrature.nodes_nm
    Fit(model, {path: Uniform(nodes, nodes * 1.001)}, (term,))
    with pytest.raises(ValueError, match=r"scores .* pixel"):
        Fit(model, {path: Uniform(nodes * 0.9, nodes)}, (term,))
    with pytest.raises(ValueError, match="unbounded"):
        Fit(model, {path: Normal(nodes, jnp.ones_like(nodes))}, (term,))


class _Patch(AbstractSpeckleField):
    """A static speckle excess on a six-cell grid of 0.3 lambda/D, read linearly.

    The excess density is ten times the toy coronagraph's static floor (1e-9 per
    square lambda/D), modulated so that no two cells are equal.
    """

    grid: PixelGrid = eqx.field(static=True)
    data: jax.Array
    epoch_d: float = eqx.field(static=True, converter=float, default=0.0)

    def __init__(self, n=6, step=0.3):
        """Build the field on an ``n`` cell grid of ``step`` lambda/D."""
        self.grid = PixelGrid.focal(n, step, GaussianCoronagraph().grid.frame)
        x, y = self.grid.coordinates()
        modulation = 1 + 0.5 * jnp.cos(0.8 * x + 0.2) * jnp.cos(0.6 * y - 0.1)
        self.data = 1e-8 * step**2 * modulation

    def spectral_breaks_nm(self):
        """The excess is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        """The same cell integrals at every node and time."""
        return Image(
            self.data, self.grid, (), Measure.integral(), "stellar_flux_fraction"
        )


def _speckle_setup(table_n, detector_n, field=True):
    """The toy setup on a ``table_n`` cell table and a ``detector_n`` pixel detector.

    The detector pixel is one table cell at 550 nm (0.3 lambda/D); the speckle field
    is :class:`_Patch`, six cells of 0.3 lambda/D, read linearly.
    """
    small = make_detector("poisson")
    detector = type(small)(
        photon_counting=True,
        pixel_scale_arcsec=small.pixel_scale_arcsec,
        shape=(detector_n, detector_n),
        qe=0.8,
    )
    instrument = eqx.tree_at(
        lambda i: i.coronagraph,
        make_instrument(detector),
        GaussianCoronagraph(n=table_n),
    )
    if field:
        instrument = eqx.tree_at(
            lambda i: i.speckle, instrument, _Patch(), is_leaf=lambda v: v is None
        )
    scene = make_scene(contrast=1e-10, sky_pa_deg=40.0, separation_arcsec=0.025)
    return ImageModel(scene, instrument), make_exposure(scene, instrument)


def _beyond_field(model, exposure):
    """Detector pixels wholly outside the field footprint grown by its read margin.

    Geometry only, at the node where the footprint and the margin are largest: the
    linear sampling margin of 0.5 cells, or ``(k - 1/2) g - 1/2`` cells (``k = 1``)
    when a detector pixel of ``g >= 1`` cells deposits.
    """
    field, detector = model.instrument.speckle, model.instrument.detector
    frame = field.grid.frame
    pixel = float(detector.pixel_scale_arcsec)
    reach = 0.0
    for w in np.asarray(exposure.quadrature.nodes_nm).ravel():
        cell = float(field.grid.step) * float(frame.lod_to_arcsec(w))
        g = pixel / cell
        margin = max(0.5, 0.5 * g - 0.5) if g >= 1 else 0.5
        reach = max(reach, (field.grid.shape[0] / 2 + margin) * cell)
    x, y = (np.asarray(c) for c in detector.grid.coordinates())
    return (np.abs(x) - pixel / 2 >= reach) | (np.abs(y) - pixel / 2 >= reach)


def _support_counts(model, exposure):
    """``(no static product, only the speckle field)`` pixel counts of the render."""
    rates = cp.expected(model.scene, model.instrument, exposure)
    strict = np.asarray(rates.supported)
    static = np.asarray(rates.static_supported)
    return int((~static).sum()), int((static & ~strict).sum())


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(
    reason="compiles three speckle and static image renders and a Newton estimate, "
    "26 s projected on CI"
)
def test_static_light_beyond_the_speckle_field_is_scored_only_under_the_option():
    """Fails if the option is ignored, admits unsupported pixels or is not recorded.

    Fails for a term that ignores the option, an option that admits pixels without
    static data, and a result that loses the record of the option.

    Geometry: the toy coronagraph with a 24-cell table, which covers a 12-pixel
    detector at every node of the 500-600 nm band, and a speckle field half the
    detector's width (:class:`_Patch`); a second instrument puts the same field on
    a 24-pixel detector with the 12-cell table, so the detector is also wider than
    the static table. Every mask is all True unless stated.

    Truth sources: the refusal counts are read from coronagraphoto's own two flags
    on ``coronagraphoto.expected`` (``static_supported`` for pixels no static
    product covers, ``static_supported & ~supported`` for pixels that lack only the
    speckle field), so this test verifies photomancy's routing of the flags; the
    flags themselves are verified from first principles (the field's coverage and
    the render with no field) by coronagraphoto's test of the two flags. The static
    model beyond the field is ``predict`` of the same
    term on the instrument without a speckle field, on pixels wholly outside the
    field footprint grown by the linear read margin; the default model there is
    zero, the strict rule.

    Tolerance basis: exact-algebra throughout (equality). Beyond the grown
    footprint the field adds an exact zero to the same static arrays through the
    same program; on a mask inside the field the two settings score the same
    rates; the counts and the names are integers and strings.

    Assertions: by default ``check`` and ``Fit`` raise and name both counts (0 and
    the speckle-only count); with the option the first instrument passes and the
    second still raises, naming its static count. ``predict`` is exactly 0.0
    where the default render is unsupported, and the static model beyond the
    field with the option. Guard: on the default term's own mask the log
    likelihood is identical for the two settings. ``Fit.assumptions`` and the
    ``LikelihoodEstimate`` of ``measure`` name the option when it is set and are
    empty otherwise. With ``on_support`` the option scores exactly the
    ``static_supported`` set, the speckle-only pixels included: the value equals
    the all-True mask's and the residual is finite on every speckle-only pixel,
    where the default ``on_support`` term leaves it NaN. The option accepts a NumPy
    or 0-d JAX boolean and stores a Python bool; an int or a boolean array is
    refused.
    """
    model, exposure = _speckle_setup(24, 12)
    shape = (1, *model.instrument.detector.shape)
    every = jnp.ones(shape, bool)
    no_static, only = _support_counts(model, exposure)
    assert no_static == 0 and 0 < only < every.size

    def term(observed, mask, option):
        return ImageTerm(
            observed, mask, exposure=exposure, static_beyond_speckle=option, **_PATHS
        )

    for flag in (np.bool_(True), jnp.asarray(True)):
        stored = term(jnp.zeros(shape), every, flag).static_beyond_speckle
        assert type(stored) is bool and stored
    for flag in (1, jnp.ones(2, bool)):
        with pytest.raises(TypeError, match="must be a boolean"):
            term(jnp.zeros(shape), every, flag)

    kept = term(jnp.zeros(shape), every, True)
    mean = np.asarray(kept.predict(model))
    observed = jnp.asarray(np.random.default_rng(0).poisson(mean).astype(float))
    strict, kept = term(observed, every, False), term(observed, every, True)
    law = {_CONTRAST: LogUniform(1e-12, 1e-8)}
    refused = rf"scores {only} pixel\(s\).*: 0 that no static .* and {only} that lack"
    with pytest.raises(ValueError, match=refused):
        strict.check(model)
    with pytest.raises(ValueError, match=refused):
        Fit(model, law, (strict,))
    kept.check(model)
    assert np.asarray(kept.supported_mask(model)).all()

    supported = np.asarray(strict.supported_mask(model))
    np.testing.assert_array_equal(np.asarray(strict.predict(model))[~supported], 0.0)
    beyond = _beyond_field(model, exposure)
    assert beyond.any()
    bare, _ = _speckle_setup(24, 12, field=False)
    static = np.asarray(term(observed, every, False).predict(bare))
    np.testing.assert_array_equal(mean[..., beyond], static[..., beyond])
    assert np.all(mean[~supported] > 0)

    speckle_only = ~supported
    loose = ImageTerm.on_support(
        model, observed, exposure=exposure, static_beyond_speckle=True, **_PATHS
    )
    tight = _scored(model, exposure, observed)
    assert float(loose(model)) == float(kept(model))
    assert np.isfinite(np.asarray(loose.residual(model))[speckle_only]).all()
    assert np.isnan(np.asarray(tight.residual(model))[speckle_only]).all()

    inside = jnp.asarray(supported)
    a, b = term(observed, inside, False), term(observed, inside, True)
    assert float(a(model)) == float(b(model))

    fit = Fit(model, law, (kept,))
    assert fit.assumptions == ("static_beyond_speckle",)
    assert Fit(model, law, (a,)).assumptions == ()
    start = {_CONTRAST: model.scene.planets[0].physical_model.contrast_grid}
    estimate = detection.measure(fit, start, n_newton=1)
    assert estimate.assumptions == ("static_beyond_speckle",)

    wide, wide_exposure = _speckle_setup(12, 24)
    no_static, only = _support_counts(wide, wide_exposure)
    assert no_static > 0 and only > 0
    frame = jnp.zeros((1, *wide.instrument.detector.shape))
    for option, count in ((True, no_static), (False, no_static + only)):
        loose = ImageTerm(
            frame,
            jnp.ones(frame.shape, bool),
            exposure=wide_exposure,
            static_beyond_speckle=option,
            **_PATHS,
        )
        with pytest.raises(
            ValueError, match=rf"scores {count} pixel\(s\).*: {no_static} that no"
        ):
            loose.check(wide)


@pytest.mark.float64_only(reason="MAP uses float64 coordinates with a narrow detector")
@pytest.mark.slow(
    reason="compiles MAP through wide and narrow image detectors, 21 s projected on CI"
)
@vv.case("scene-fits", "code-verification", basis="spike")
def test_map_fits_flux_scale_with_a_narrow_instrument(monkeypatch):
    """A noise-free image fixes the linear flux scale at one in both instruments.

    The measured MAP displacement after eight steps is 1.786e-7 in flux scale.
    The spike bound is two float32 spacings at the unit injected scale.
    """
    model, exposure = _setup("constant")
    term = _scored(model, exposure, _mean(model, exposure))
    path = "instrument.flux_scale"
    free = {path: Normal(1.0, 1.0)}
    narrow = eqx.tree_at(
        lambda m: m.instrument, model, canonical(model.instrument, precision="float32")
    )
    start = {path: jnp.asarray(0.9)}
    backend = MAP(n_steps=8, init=start)
    key = jax.random.key(0)
    wide = run(Fit(model, free, (term,)), backend, key)
    term = eqx.tree_at(
        lambda t: t.exposure, term, canonical(exposure, precision="float32")
    )
    fit = Fit(narrow, free, (term,))
    detector_type = type(narrow.instrument.detector)
    moments = detector_type.mean_and_variance
    mean_dtypes = set()

    def record_moments(detector, rates, plan, **kwargs):
        mean, variance = moments(detector, rates, plan, **kwargs)
        mean_dtypes.add(mean.dtype)
        return mean, variance

    monkeypatch.setattr(detector_type, "mean_and_variance", record_moments)
    leaf = r"model\.instrument\.primary\.circumscribed_diameter_m"
    with pytest.warns(UserWarning, match=rf"unscaled.*{leaf}"):
        post = run(fit, backend, key)
    assert (
        "model.instrument.primary.circumscribed_diameter_m"
        in post.info.scale_fallback_reason
    )
    assert mean_dtypes == {np.dtype("float32")}
    density = fit.log_density(post.samples)
    assert density.dtype == post.samples[path].dtype == jnp.float64
    assert np.all(np.isfinite(density))
    assert fit.model_at(start).instrument.flux_scale.dtype == jnp.float32
    assert fit.draw_schema()[path].dtype == jnp.float64
    with pytest.raises(PrecisionError, match=rf"Fit\.information.*{leaf}"):
        fit.information(start, kind="fisher")
    with pytest.raises(PrecisionError, match=rf"measure.*{leaf}"):
        detection.measure(fit, start)
    bound = float(np.spacing(np.float32(1)))
    np.testing.assert_allclose(wide.samples[path], 1.0, rtol=0, atol=bound)
    np.testing.assert_allclose(
        post.samples[path], wide.samples[path], rtol=0, atol=2 * bound
    )


# ---- ExposureData: the term, the forward and the laws built from data ----


@pytest.fixture(scope="module")
def measured(poisson_image):
    """The shared scene observed once: ``(model, exposure, data, released counts)``."""
    model, exposure, _ = poisson_image
    key = jax.random.key(0)
    data = cp.observe_data(key, model.scene, model.instrument, exposure)
    counts = cp.observe(key, model.scene, model.instrument, exposure).data
    return model, exposure, data, counts


def _displaced(model):
    """The model with its planet contrast scaled by 1.2."""
    grid = model.scene.planets[0].physical_model.contrast_grid
    return eqx.tree_at(
        lambda m: m.scene.planets[0].physical_model.contrast_grid, model, grid * 1.2
    )


def _values(term, model):
    """The residual everywhere, and the score where float64 is available."""
    out = [np.asarray(term.residual(model))]
    if jax.config.jax_enable_x64:
        out.append(np.asarray(term(model)))
    return out


def _finite_where_scored(term, values):
    """The residual is finite on the scored pixels and the score is finite."""
    residual, *score = values
    return bool(
        np.isfinite(residual[np.asarray(term.mask)]).all()
        and all(np.isfinite(x) for x in score)
    )


def _nan_equal(a, b):
    return np.array_equal(a, b, equal_nan=True)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_a_term_built_from_exposure_data_is_the_released_term(measured):
    """Fails if the data spelling stores or scores anything the array spelling does not.

    Truth source: the released term on the same electrons (``observe``'s array, an
    all-True mask, the instrument and the exposure). Tolerance basis: exact-algebra,
    bitwise: both terms hold the same leaves and run the same program. The
    ``__call__`` and ``fisher_root`` comparisons run where float64 is available;
    ``split`` rates and ``residual`` run in both lanes, at the truth model and at a
    displaced one.
    """
    model, exposure, data, counts = measured
    np.testing.assert_array_equal(data.image.data, counts)
    spec = dict(instrument=model.instrument, scene_path="scene")
    supported = np.asarray(ImageTerm(data, **spec).supported_mask(model))
    assert not supported.all()
    for mask in (None, supported):
        ones = jnp.ones(counts.shape, bool) if mask is None else mask
        new = ImageTerm(data, mask, **spec)
        old = ImageTerm(counts, ones, model.instrument, exposure, "scene")
        assert eqx.tree_equal(new, old)
        for at in (model, _displaced(model)):
            assert eqx.tree_equal(new.split(at)[0], old.split(at)[0])
            if jax.config.jax_enable_x64:
                np.testing.assert_array_equal(new.fisher_root(at), old.fisher_root(at))
            got, want = _values(new, at), _values(old, at)
            assert all(_nan_equal(a, b) for a, b in zip(got, want, strict=True))
            if mask is not None:
                assert _finite_where_scored(new, got)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_the_forward_is_the_term_prediction_in_either_quantity(measured):
    """Fails if the forward renders its own way or divides by the wrong time.

    Truth source: exact algebra, one renderer: ``ImageForward`` in electrons is
    ``ImageTerm.predict`` (bitwise), and in electrons per second times the live time
    it is the same array to two roundings (``ulps(2, expected)``). The data
    spelling holds the data's quantity, grid and exposure. ``image`` names the
    prediction: quantity, integral measure, band axis and the detector grid with no
    orientation. The plan has four frames, so the live time is not one frame.
    """
    model, exposure, data, _ = measured
    four = eqx.tree_at(lambda e: e.plan, exposure, FramePlan(2500.0, 4))
    held = dict(instrument=model.instrument, scene_path="scene")
    from_data = ImageForward.from_exposure_data(data, **held)
    assert from_data.exposure is data.exposure and from_data.quantity == "electrons"
    for plan_exposure in (exposure, four):
        electrons = ImageForward(exposure=plan_exposure, **held)
        rate = ImageForward(exposure=plan_exposure, quantity="electron/s", **held)
        want = ImageTerm(data, instrument=model.instrument, scene_path="scene").predict(
            model
        )
        if plan_exposure is exposure:
            np.testing.assert_array_equal(electrons(model), want)
        else:
            want = electrons(model)
        live = float(plan_exposure.plan.live_time_s)
        assert live == 1.0e4
        back = np.asarray(rate(model)) * live
        assert np.all(np.abs(back - np.asarray(want)) <= np.asarray(ulps(2, want)))
    image = rate.image(model)
    assert (image.quantity, image.measure, image.leading_axes) == (
        "electron/s",
        Measure.integral(),
        ("band",),
    )
    assert image.grid == replace(model.instrument.detector.grid, orientation=None)
    np.testing.assert_array_equal(image.data, rate(model))
    assert electrons.image(model).quantity == "electrons"


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_the_forward_passes_static_beyond_speckle_to_the_renderer():
    """Fails if the forward drops the option or keeps a second renderer without it.

    Truth source: ``ImageTerm.predict`` with the same option on a detector wider
    than the speckle field, where the default render is exactly zero outside the
    field and the optioned render is the static scene (``predict`` of the term).
    Tolerance basis: exact-algebra, bitwise. The forward names the option in its
    ``assumptions`` and is the held instrument or the one at a model path.
    """
    model, exposure = _speckle_setup(24, 12)
    model = ImageModel(model.scene, model.instrument, (exposure,))
    shape = (1, *model.instrument.detector.shape)
    term = ImageTerm(
        jnp.zeros(shape),
        jnp.ones(shape, bool),
        exposure=exposure,
        static_beyond_speckle=True,
        **_PATHS,
    )
    plain = ImageForward(exposure=exposure, **_PATHS)
    kept = ImageForward(exposure=exposure, static_beyond_speckle=True, **_PATHS)
    by_path = ImageForward(
        exposure_path="exposures[0]", static_beyond_speckle=True, **_PATHS
    )
    assert not np.array_equal(plain(model), kept(model))
    np.testing.assert_array_equal(kept(model), term.predict(model))
    np.testing.assert_array_equal(by_path(model), term.predict(model))
    assert (kept.assumptions, plain.assumptions) == (("static_beyond_speckle",), ())


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_the_electrons_rate_of_a_dark_only_scene_is_the_dark_rate():
    """Fails if the rate omits the dark term or divides by the frame time.

    Truth source: constants typed here. A star of zero flux and a planet of zero
    contrast leave only dark current, 0.01 electron per pixel per second, so the
    prediction is 0.01 at every pixel, over a plan of four frames of 2500 s: 100
    electrons and 0.01 electron per second. Dividing by the 2500 s frame gives 0.04;
    dropping the dark term gives zero.
    Tolerance basis: ``rtol=1e-5``, float32 accumulation over the quadrature.
    """
    grid = GaussianCoronagraph().grid
    pitch = float(grid.frame.lod_to_arcsec(550.0)) * 0.3
    detector = Detector(
        pixel_scale_arcsec=pitch,
        shape=(24, 24),
        qe=0.8,
        dark_current_e_per_pixel_s=0.01,
    )
    instrument = make_instrument(detector)
    scene = make_scene(contrast=1e-10)
    scene = eqx.tree_at(lambda s: s.star, scene, FlatStar(flux_phot_per_nm_m2=0.0))
    exposure = make_exposure(scene, instrument, time_s=2500.0)
    exposure = eqx.tree_at(lambda e: e.plan, exposure, FramePlan(2500.0, 4))
    model = ImageModel(scene, instrument)
    for quantity, want in (("electrons", 100.0), ("electron/s", 0.01)):
        forward = ImageForward(
            instrument=instrument,
            exposure=exposure,
            scene_path="scene",
            quantity=quantity,
        )
        np.testing.assert_allclose(forward(model), want, rtol=1e-5)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="image log densities require float64")
def test_the_exposure_path_takes_precedence_over_the_data(measured):
    """Fails if the data's exposure wins over the model's, or both exposures are held.

    Truth source: the released term holding the model's exposure. The data's roll is
    changed to 30 degrees: the score at the model is the released one and does not
    move with the data's roll; moving the model's roll changes it. An explicit
    ``exposure=`` beside data raises. Tolerance basis: exact (equality).
    """
    model, exposure, data, counts = measured
    model = ImageModel(model.scene, model.instrument, (exposure,))
    turned = eqx.tree_at(
        lambda d: d.exposure.pointing.roll_deg, data, jnp.asarray(30.0)
    )
    spec = dict(instrument=model.instrument, scene_path="scene")
    path = ImageTerm.on_support(model, data, exposure_path="exposures[0]", **spec)
    other = ImageTerm.on_support(model, turned, exposure_path="exposures[0]", **spec)
    released = ImageTerm.on_support(
        model,
        counts,
        instrument=model.instrument,
        exposure=exposure,
        scene_path="scene",
    )
    assert path.exposure is None and other.exposure is None
    assert float(path(model)) == float(other(model)) == float(released(model))
    rolled = eqx.tree_at(
        lambda m: m.exposures[0].pointing.roll_deg, model, jnp.asarray(20.0)
    )
    assert float(path(rolled)) != float(path(model))
    assert float(path(rolled)) == float(other(rolled))
    with pytest.raises(ValueError, match="supplies the exposure"):
        ImageTerm(data, exposure=exposure, **spec)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_the_data_mask_is_scored_and_combined_with_an_explicit_mask(measured):
    """Fails if ``valid`` is ignored, alone or beside ``mask``, or draws are mixed.

    Truth source: the array spelling with the mask typed here. One invalid pixel in
    the data equals the array term with that pixel masked; an explicit mask is
    combined by logical and; two draws with the same ``valid`` give one mask and two
    frames, and unequal ``valid`` raise. Tolerance basis: exact (equal leaves and
    values).
    """
    model, exposure, data, counts = measured
    spec = dict(instrument=model.instrument, scene_path="scene")
    supported = np.asarray(ImageTerm(data, **spec).supported_mask(model))
    valid = supported.copy()
    valid[0, 5, 5] = False
    extra = np.ones(counts.shape, bool)
    extra[0, 7, 9] = False
    masked = ExposureData(data.image, exposure, valid=valid, built_at=data.built_at)

    def array_term(mask):
        return ImageTerm(counts, mask, model.instrument, exposure, "scene")

    plain = ImageTerm(masked, **spec)
    assert eqx.tree_equal(plain, array_term(valid))
    both = ImageTerm(masked, extra, **spec)
    assert eqx.tree_equal(both, array_term(valid & extra))
    assert not bool(both.mask[0, 5, 5]) and not bool(both.mask[0, 7, 9])
    for term, ref in ((plain, array_term(valid)), (both, array_term(valid & extra))):
        got, want = _values(term, model), _values(ref, model)
        assert all(_nan_equal(a, b) for a, b in zip(got, want, strict=True))
        assert _finite_where_scored(term, got)
    twice = stack_exposure_data([masked, masked], axis="draw")
    drawn = ImageTerm(twice, **spec)
    assert drawn.observed_e.shape == (2, *counts.shape)
    np.testing.assert_array_equal(drawn.mask, valid)
    unequal = stack_exposure_data(
        [masked, ExposureData(data.image, exposure, built_at=data.built_at)],
        axis="draw",
    )
    with pytest.raises(ValueError, match="unequal valid"):
        ImageTerm(unequal, **spec)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_data_the_detector_law_cannot_score_is_refused(measured):
    """Fails if a variance is ignored, an event axis accepted or a grid unchecked.

    Truth source: the contract. ``ImageTerm`` refuses data with its own variance
    (naming the term and law that take it), data with an event axis, data on a
    grid that is not the held detector's, and a quantity that is not electrons;
    ``ImageForward`` refuses an unknown quantity, an event axis and a foreign grid.
    Tolerance basis: exact (the refusals and their messages).
    """
    model, exposure, data, counts = measured
    spec = dict(instrument=model.instrument, scene_path="scene")
    with_variance = ExposureData(data.image, exposure, variance=np.ones(counts.shape))
    with pytest.raises(ValueError, match=r"ForwardTerm.*MaskedLogLikelihood"):
        ImageTerm(with_variance, **spec)
    events = stack_exposure_data([data, data])
    with pytest.raises(ValueError, match="event"):
        ImageTerm(events, **spec)
    with pytest.raises(ValueError, match="event"):
        ImageForward.from_exposure_data(events, **spec)
    other = replace(data.image.grid, center_x=data.image.grid.center_x + 1.0)
    foreign = ExposureData(replace(data.image, grid=other), exposure)
    with pytest.raises(ValueError, match="detector grid"):
        ImageTerm(foreign, **spec)
    with pytest.raises(ValueError, match="detector grid"):
        ImageForward.from_exposure_data(foreign, **spec)
    flux = ExposureData(replace(data.image, quantity="photons"), exposure)
    with pytest.raises(ValueError, match="needs electrons"):
        ImageTerm(flux, **spec)
    with pytest.raises(ValueError, match="predicts one of"):
        ImageForward.from_exposure_data(flux, **spec)
    with pytest.raises(ValueError, match="quantity must be one of"):
        ImageForward(exposure=exposure, quantity="photons", **spec)


def _calibrated(model, exposure, scale=1.05):
    """A rate image with a stated variance and one masked pixel, near the model."""
    rate = ImageForward(
        instrument=model.instrument,
        exposure=exposure,
        scene_path="scene",
        quantity="electron/s",
    )(model)
    values = rate * scale + 1e-3
    valid = np.ones(values.shape, bool)
    valid[0, 5, 5] = False
    return make_data(
        model.instrument,
        exposure,
        values,
        quantity="electron/s",
        variance=(0.1 * values + 0.02) ** 2,
        valid=valid,
    )


def _calibrated_term(model, exposure):
    cal = _calibrated(model, exposure)
    forward = ImageForward.from_exposure_data(
        cal, instrument=model.instrument, scene_path="scene"
    )
    return cal, ForwardTerm(forward, MaskedLogLikelihood.from_exposure_data(cal))


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.float64_only(reason="ForwardTerm log densities require float64")
def test_a_calibrated_rate_image_scores_as_the_gaussian_density(poisson_image):
    """Fails if the forward and the law disagree on quantity, mask or scale.

    Truth source: the Gaussian log density typed here in NumPy float64, summed over
    the valid pixels, at the mean rate that ``detector.mean_and_variance`` gives
    for the forward's rates (divided by the live time), the data and the variance:
    ``-(d - p)^2 / (2 v) - log(v) / 2 - log(2 pi) / 2``. The law equals
    ``gaussian_loglike`` of the same arrays exactly. Tolerance basis:
    floating-point, ``ulps(64, expected)`` (a sum over 575 pixels of terms of
    order one to ten).
    """
    model, exposure, _ = poisson_image
    cal, term = _calibrated_term(model, exposure)
    detector, rates, plan = term.forward.rates(model)
    mean = np.asarray(detector.mean_and_variance(rates, plan)[0], np.float64)
    predicted = mean / float(plan.live_time_s)
    data = np.asarray(cal.image.data, np.float64)
    variance = np.asarray(cal.variance, np.float64)
    used = np.asarray(cal.valid)
    assert not used.all()
    density = (
        -((data - predicted) ** 2) / (2 * variance)
        - 0.5 * np.log(variance)
        - 0.5 * np.log(2 * np.pi)
    )
    want = density[used].sum()
    got = float(term(model))
    assert abs(got - want) <= float(ulps(64, want))
    ref = gaussian_loglike(cal.image.data, jnp.sqrt(cal.variance), cal.valid)
    assert float(ref(term.forward(model))) == got


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.float64_only(reason="float64 central differences")
def test_the_calibrated_term_has_the_flux_gradient(poisson_image):
    """Fails if the forward or law breaks differentiation with respect to flux.

    Truth source: a central difference of the term along a uniform scaling of the
    planet contrast grid, step 1e-4 relative; the autodiff gradient is contracted
    with the same direction. Tolerance basis: ``rtol=1e-3``.
    """
    model, exposure, _ = poisson_image
    _, term = _calibrated_term(model, exposure)
    where = lambda m: m.scene.planets[0].physical_model.contrast_grid  # noqa: E731
    grid = where(model)
    grad = where(eqx.filter_grad(term)(model))
    exact = float(jnp.sum(grad * grid))
    h = 1e-4

    def at(factor):
        return float(term(eqx.tree_at(where, model, grid * factor)))

    numeric = (at(1 + h) - at(1 - h)) / (2 * h)
    assert exact != 0.0
    np.testing.assert_allclose(exact, numeric, rtol=1e-3)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_a_forward_term_reports_the_forwards_footprint_and_check(poisson_image):
    """Fails if ``ForwardTerm`` drops the forward's footprint, check or assumptions.

    Truth source: the ``ImageTerm`` behavior. A ``ForwardTerm`` over an
    ``ImageForward`` reports the same ``footprint_paths`` and refuses an unbounded
    prior on the band nodes as ``ImageTerm`` does; its ``check`` raises for a lit
    planet outside the coronagraph's source domain and passes for one inside; the
    option appears in the fit's assumptions. Tolerance basis: exact.
    """
    from priorities import Normal

    model, exposure, _ = poisson_image
    model = ImageModel(model.scene, model.instrument, (exposure,))
    paths = dict(
        scene_path="scene",
        instrument_path="instrument",
        exposure_path="exposures[0]",
    )
    shape = (1, *model.instrument.detector.shape)
    forward = ImageForward(**paths)
    law = gaussian_loglike(jnp.zeros(shape), jnp.ones(shape))
    term = ForwardTerm(forward, law)
    released = ImageTerm.on_support(model, jnp.zeros(shape), **paths)
    assert term.footprint_paths == released.footprint_paths != ()
    nodes = exposure.quadrature.nodes_nm
    prior = {"exposures[0].quadrature.nodes_nm": Normal(nodes, jnp.ones_like(nodes))}
    for scored in (term, released):
        with pytest.raises(ValueError, match="unbounded"):
            Fit(model, prior, (scored,))
    assert term.assumptions == ()
    kept = ForwardTerm(ImageForward(static_beyond_speckle=True, **paths), law)
    assert Fit(model, {_CONTRAST: LogUniform(1e-12, 1e-8)}, (kept,)).assumptions == (
        "static_beyond_speckle",
    )

    class Bounded(type(make_instrument(make_detector()).coronagraph)):
        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            inside = jnp.hypot(jnp.atleast_1d(x_lod), jnp.atleast_1d(y_lod)) <= 2.0
            return type(psfs)(psfs.image, inside)

    instrument = eqx.tree_at(
        lambda i: i.coronagraph, make_instrument(make_detector()), Bounded()
    )
    for separation, refused in ((0.05, True), (0.02, False)):
        scene = make_scene(contrast=1e-10, separation_arcsec=separation)
        bounded = ImageModel(scene, instrument, (make_exposure(scene, instrument),))
        if refused:
            with pytest.raises(ValueError, match="outside the coronagraph's source"):
                term.check(bounded)
        else:
            term.check(bounded)
