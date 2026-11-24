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
from image_scene import (
    GaussianCoronagraph,
    make_detector,
    make_exposure,
    make_instrument,
    make_scene,
)
from optixstuff import AbstractSpeckleField
from optixstuff.grids import Image, Measure, PixelGrid, SkyOrientation
from priorities import LogUniform
from scipy import stats

from photomancy import detection
from photomancy.image import ImageModel, ImageTerm
from photomancy.infer.fit import Fit
from photomancy.orbit.sheets import SheetSum

_PATHS = {"scene_path": "scene", "instrument_path": "instrument"}


def _setup(kind="poisson", contrast=1e-10):
    instrument = make_instrument(make_detector(kind))
    scene = make_scene(contrast=contrast, sky_pa_deg=40.0)
    exposure = make_exposure(scene, instrument)
    return ImageModel(scene, instrument), exposure


def _term(model, exposure, observed, mask=None):
    observed = jnp.asarray(observed)
    mask = jnp.ones(observed.shape, bool) if mask is None else mask
    return ImageTerm(observed, mask, exposure=exposure, **_PATHS)


def _scored(model, exposure, observed):
    """A term whose mask is the pixels the toy coronagraph's tables cover.

    The detector is as wide as the tables, so the edge pixels (more at the blue
    node of the band) are unsupported; ``Fit`` refuses a mask that scores them.
    """
    blank = _term(model, exposure, observed)
    return _term(model, exposure, observed, blank.supported_mask(model))


def _mean(model, exposure):
    shape = (1, *model.instrument.detector.shape)
    return _term(model, exposure, jnp.zeros(shape)).predict(model)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_poisson_value_is_the_summed_pmf_of_the_expected_counts():
    """A photon-counting detector scores the Poisson pmf of ``predict`` per pixel.

    Truth source: ``scipy.stats.poisson.logpmf`` at the expected counts, summed over the
    scored pixels only (a masked pixel holds NaN and must not reach the value).
    Tolerance basis: exact-algebra (rtol 1e-12; the same sum in float64).
    Fails if: the mask is applied after the sum, a NaN at a masked pixel leaks, or the
    density is not normalized.
    """
    model, exposure = _setup("poisson")
    mean = np.asarray(_mean(model, exposure))
    counts = np.random.default_rng(3).poisson(mean).astype(float)
    mask = np.ones(counts.shape, bool)
    mask[..., :4, :] = False
    observed = np.where(mask, counts, np.nan)
    term = _term(model, exposure, observed, mask)
    want = stats.poisson.logpmf(counts[mask], mean[mask]).sum()
    np.testing.assert_allclose(term(model), want, rtol=1e-12)
    grad = eqx.filter_grad(term)(model)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(grad))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_gaussian_value_is_the_summed_density_of_the_detector_moments():
    """A read-noise detector scores the Gaussian of its own mean and variance.

    Truth source: ``scipy.stats.norm.logpdf`` with the detector's
    ``mean_and_variance`` at the model's expected rates.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: the variance is dropped or the log-determinant constant is missing.
    """
    model, exposure = _setup("read")
    detector = model.instrument.detector
    probe = _term(model, exposure, jnp.zeros((1, *detector.shape)))
    _, rates, plan = probe._rates(model)
    mean, variance = (np.asarray(x) for x in detector.mean_and_variance(rates, plan))
    observed = mean + np.sqrt(variance) * np.random.default_rng(4).normal(
        size=mean.shape
    )
    term = _term(model, exposure, observed)
    want = stats.norm.logpdf(observed, mean, np.sqrt(variance)).sum()
    np.testing.assert_allclose(term(model), want, rtol=1e-12)


_READ_NOISE_E = {"poisson": 0.0, "read": 3.0}


def _contrast(model):
    return model.scene.planets[0].physical_model.contrast_grid


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("kind", ["poisson", "read"])
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
    fit = Fit(canonical(model), {_CONTRAST: LogUniform(1e-12, 1e-8)}, (term,))
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
    Jacobian against a central difference in the planet's contrast scale.
    Tolerance basis: floating-point (rtol 1e-12 for ``2 / 3``; for the central
    difference at a step of 1e-4, 1e-6 of the largest derivative, above both the
    truncation error and the rounding of order-one residuals divided by the step).
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
    np.testing.assert_allclose(out[peak], 2.0 / 3.0, rtol=1e-12)

    def residual_at(s):
        return term.residual(eqx.tree_at(_contrast, model, s * grid))

    jac = np.asarray(jax.jacfwd(residual_at)(1.0))
    assert np.all(np.isfinite(jac))
    step = 1e-4
    central = np.asarray(residual_at(1.0 + step)) - np.asarray(residual_at(1.0 - step))
    used = np.asarray(term.mask)
    assert np.all(np.isnan(np.asarray(residual_at(1.0))[~used]))
    np.testing.assert_allclose(
        jac[used],
        central[used] / (2 * step),
        rtol=0,
        atol=1e-6 * np.max(np.abs(jac[used])),
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


def test_the_value_peaks_at_the_true_model():
    """Noise-free counts are likelier at the truth than at 5 percent off."""
    model, exposure = _setup("poisson")
    term = _term(model, exposure, jnp.round(_mean(model, exposure)))
    off = _setup("poisson", contrast=1.05e-10)[0]
    assert term(model) > term(off)


def test_draw_gives_counts_consistent_with_the_expectation_and_keeps_masked_pixels():
    """Drawn counts are integers about the mean; masked pixels keep their stored value.

    Tolerance: the standardized residual of 576 Poisson pixels has a standard error of
    0.03 on its standard deviation and of 0.04 on its mean; the bounds are 0.2 and 0.25.
    """
    model, exposure = _setup("poisson")
    mean = _mean(model, exposure)
    supported = _term(model, exposure, mean).supported_mask(model)
    mask = supported & jnp.ones(mean.shape, bool).at[0, :3, :].set(False)
    term = _term(model, exposure, jnp.full(mean.shape, -7.0), mask)
    drawn = term.draw(jax.random.PRNGKey(5), model)
    out = np.asarray(drawn.observed_e)
    assert np.all(out[0, :3] == -7.0)
    kept = np.asarray(mask)
    assert np.all(out[kept] == np.round(out[kept]))
    z = (out[kept] - np.asarray(mean)[kept]) / np.sqrt(np.asarray(mean)[kept])
    assert abs(z.std() - 1.0) < 0.2 and abs(z.mean()) < 0.25


def test_batched_frames_are_scored_jointly():
    """A batch of two frames scores as the sum of the two single frames."""
    model, exposure = _setup("poisson")
    mean = np.asarray(_mean(model, exposure))
    rng = np.random.default_rng(6)
    frames = rng.poisson(np.broadcast_to(mean, (2, *mean.shape))).astype(float)
    both = _term(model, exposure, frames)
    singles = [_term(model, exposure, f) for f in frames]
    np.testing.assert_allclose(
        both(model), sum(float(t(model)) for t in singles), rtol=1e-12
    )
    drawn = both.draw(jax.random.PRNGKey(0), model).observed_e
    assert drawn.shape == frames.shape
    assert not np.array_equal(drawn[0], drawn[1])


def test_exposure_comes_from_the_model_by_path_or_the_term_not_both():
    """A path exposure equals a held one; giving both or neither fails."""
    model, exposure = _setup("poisson")
    observed = jnp.round(_mean(model, exposure))
    held = _term(model, exposure, observed)
    by_path = ImageTerm(
        observed, jnp.ones(observed.shape, bool), exposure_path="exposures[0]", **_PATHS
    )
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


def test_the_term_runs_on_a_scene_rooted_model_with_a_held_instrument():
    """The model may be the scene itself, with the instrument held by the term.

    Truth source: the same frame scored through an ``ImageModel`` with both paths.
    Tolerance basis: exact (rtol 1e-12; the same renderer on the same leaves).
    Fails if: the empty scene path does not name the model, or the held instrument is
    ignored (``Fit(scene, ...)`` is the model of every orbit posterior).
    """
    model, exposure = _setup("poisson")
    observed = jnp.round(_mean(model, exposure))
    mask = _scored(model, exposure, observed).mask
    rooted = ImageTerm(observed, mask, instrument=model.instrument, exposure=exposure)
    held = _term(model, exposure, observed)
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
    assert _term(model, exposure, observed).reads == ()


def test_an_observed_image_is_checked_against_the_held_detector_and_roll():
    """An ``Image`` from ``observe`` scores as its data; foreign data are refused.

    Truth source: the same term built from the bare ``Image.data`` array, and, for
    ``mask=None``, the explicit mask ``supported_mask(model)`` (the toy detector is
    wider than its tables, 69 percent supported).
    Tolerance basis: exact (the same pixels and the same renderer, compared bit for
    bit); angles equal modulo 360 to the float32 resolution of the angle.
    Fails if: the term drops the data of an ``Image``; accepts data on another
    pitch, at another roll (also with the instrument read from the model), with no
    orientation where the roll is known, or in another quantity or measure; refuses
    a roll of 360 against 0 or a float32 copy of the roll; or ``mask=None`` scores
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
    supported = ImageTerm(image, instrument=instrument, exposure=exposure)
    assert supported.mask is None
    law = {"planets[0].physical_model.contrast_grid": LogUniform(1e-12, 1e-8)}
    fit = Fit(canonical(model.scene), law, (supported,))
    x = {k: resolve(model.scene, k) for k in law}
    assert float(fit.log_likelihood(x)) == float(from_array(model.scene))
    grid = image.grid
    coarse = PixelGrid.regular(
        grid.shape, 2.0 * grid.step, grid.unit, orientation=grid.orientation
    )
    with pytest.raises(ValueError, match="detector grid"):
        ImageTerm(replace(image, grid=coarse), instrument=instrument, exposure=exposure)
    turned = replace(grid, orientation=SkyOrientation(grid.orientation.pa_y_deg + 30))
    with pytest.raises(ValueError, match="roll"):
        ImageTerm(replace(image, grid=turned), instrument=instrument, exposure=exposure)
    with pytest.raises(ValueError, match="roll"):
        ImageTerm(replace(image, grid=turned), exposure=exposure, **_PATHS)
    bare = replace(grid, orientation=None)
    with pytest.raises(ValueError, match="no sky orientation"):
        ImageTerm(replace(image, grid=bare), instrument=instrument, exposure=exposure)
    for quantity, measure in (("electrons/s", image.measure), ("electrons", None)):
        measure = Measure.intensive() if measure is None else measure
        foreign = replace(image, quantity=quantity, measure=measure)
        with pytest.raises(ValueError, match="electrons per pixel"):
            ImageTerm(foreign, instrument=instrument, exposure=exposure)
    full_turn = replace(grid, orientation=SkyOrientation(360.0))
    ImageTerm(replace(image, grid=full_turn), instrument=instrument, exposure=exposure)
    rolled = eqx.tree_at(lambda e: e.roll_deg, exposure, jnp.asarray(30.1))
    single = replace(grid, orientation=SkyOrientation(float(np.float32(30.1))))
    ImageTerm(replace(image, grid=single), instrument=instrument, exposure=rolled)
    unchecked = ImageTerm(replace(image, grid=coarse), exposure=exposure, **_PATHS)
    assert float(unchecked(model)) == float(
        ImageTerm(image.data, exposure=exposure, **_PATHS)(model)
    )


_CONTRAST = "scene.planets[0].physical_model.contrast_grid"


def test_an_image_fit_frees_a_scene_leaf_and_gradients_flow():
    """``Fit`` accepts the term over a scene leaf: finite density and gradient."""
    model, exposure = _setup("poisson")
    term = _scored(model, exposure, jnp.round(_mean(model, exposure)))
    fit = Fit(canonical(model), {_CONTRAST: LogUniform(1e-12, 1e-8)}, (term,))
    x = {_CONTRAST: model.scene.planets[0].physical_model.contrast_grid}
    assert np.isfinite(fit.log_density(x))
    grad = jax.grad(fit.real_log_prob)(fit.to_real(x))
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad))
    worse = {_CONTRAST: 1.3 * x[_CONTRAST]}
    assert fit.log_likelihood(x) > fit.log_likelihood(worse)


def test_a_scene_rooted_fit_frees_a_leaf_of_the_scene():
    """A held-instrument term on ``Fit(scene, ...)`` scores like the record model."""
    model, exposure = _setup("poisson")
    observed = jnp.round(_mean(model, exposure))
    mask = _scored(model, exposure, observed).mask
    rooted = ImageTerm(observed, mask, instrument=model.instrument, exposure=exposure)
    path = "planets[0].physical_model.contrast_grid"
    law = LogUniform(1e-12, 1e-8)
    grid = model.scene.planets[0].physical_model.contrast_grid
    on_scene = Fit(canonical(model.scene), {path: law}, (rooted,)).log_density(
        {path: grid}
    )
    on_record = Fit(
        canonical(model), {f"scene.{path}": law}, (_scored(model, exposure, observed),)
    ).log_density({f"scene.{path}": grid})
    np.testing.assert_allclose(on_scene, on_record, rtol=1e-12)


def test_a_count_where_the_mean_is_zero_is_minus_infinity_not_nan():
    """Outside the support the term is ``-inf``; NaN is reserved for a real failure.

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
    zero = _term(model, exposure, jnp.zeros(shape))
    one = _term(model, exposure, jnp.zeros(shape).at[0, 5, 5].set(1.0))
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
    full = _term(model, exposure, observed)
    supported = np.asarray(full.supported_mask(model))
    n_bad = int((~supported).sum())
    assert 0 < n_bad < supported.size
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        full.check(model)
    law = {_CONTRAST: LogUniform(1e-12, 1e-8)}
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        Fit(canonical(model), law, (full,))
    # Held in a SheetSum the term is refused exactly as it is bare, and the sum reports
    # the footprint the term does.
    held = SheetSum((full,), orbit_path="orbits")
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        held.check(model)
    with pytest.raises(ValueError, match=rf"scores {n_bad} pixel"):
        Fit(canonical(model), law, (held,))
    assert held.footprint_paths == full.footprint_paths != ()

    scored = _term(model, exposure, observed, jnp.asarray(supported))
    scored.check(model)
    fit = Fit(canonical(model), law, (scored,))
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
    other = Fit(canonical(cropped_model), law, (cropped,))
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
        probe = _term(model, exposure, observed)
        term = _term(model, exposure, observed, probe.supported_mask(model))
        if refused:
            with pytest.raises(ValueError, match="outside the coronagraph's source"):
                term.check(model)
        else:
            term.check(model)


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
    Fit(canonical(model), {path: Uniform(nodes, nodes * 1.001)}, (term,))
    with pytest.raises(ValueError, match=r"scores .* pixel"):
        Fit(canonical(model), {path: Uniform(nodes * 0.9, nodes)}, (term,))
    with pytest.raises(ValueError, match="unbounded"):
        Fit(canonical(model), {path: Normal(nodes, jnp.ones_like(nodes))}, (term,))


class _Patch(AbstractSpeckleField):
    """A static speckle excess on a 12 cell grid of 0.3 lambda/D, read linearly.

    The excess density is ten times the toy coronagraph's static floor (1e-9 per
    square lambda/D), modulated so that no two cells are equal.
    """

    grid: PixelGrid = eqx.field(static=True)
    data: jax.Array
    epoch_d: float = 0.0

    def __init__(self, n=12, step=0.3):
        """Build the field on an ``n`` cell grid of ``step`` lambda/D."""
        self.grid = PixelGrid.focal(n, step, GaussianCoronagraph().grid.frame)
        x, y = self.grid.coordinates()
        modulation = 1 + 0.5 * jnp.cos(0.8 * x + 0.2) * jnp.cos(0.6 * y - 0.1)
        self.data = 1e-8 * step**2 * modulation

    def spectral_breaks_nm(self):
        """The excess is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, time_d=0.0):
        """The same cell integrals at every node and time."""
        return Image(
            self.data, self.grid, (), Measure.integral(), "stellar_flux_fraction"
        )


def _speckle_setup(table_n, detector_n, field=True):
    """The toy setup on a ``table_n`` cell table and a ``detector_n`` pixel detector.

    The detector pixel is one table cell at 550 nm (0.3 lambda/D); the speckle field
    is :class:`_Patch`, 12 cells of 0.3 lambda/D, read linearly.
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
    scene = make_scene(contrast=1e-10, sky_pa_deg=40.0)
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


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.medium(reason="renders three instruments and one Newton step, 7 s cold")
def test_static_light_beyond_the_speckle_field_is_scored_only_under_the_option():
    """Fails if the option is ignored, admits unsupported pixels or is not recorded.

    Fails for a term that ignores the option, an option that admits pixels without
    static data, and a result that loses the record of the option.

    Geometry: the toy coronagraph with a 48 cell table, which covers a 24 pixel
    detector at every node of the 500-600 nm band, and a speckle field half the
    detector's width (:class:`_Patch`); a second instrument puts the same field on
    a 48 pixel detector with the 24 cell table, so the detector is also wider than
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
    empty otherwise. With ``mask=None`` the option scores exactly the
    ``static_supported`` set, the speckle-only pixels included: the value equals
    the all-True mask's and the residual is finite on every speckle-only pixel,
    where the default ``mask=None`` term leaves it NaN. The option accepts a NumPy
    or 0-d JAX boolean and stores a Python bool; an int or a boolean array is
    refused.
    """
    model, exposure = _speckle_setup(48, 24)
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
        Fit(canonical(model), law, (strict,))
    kept.check(model)
    assert np.asarray(kept.supported_mask(model)).all()

    supported = np.asarray(strict.supported_mask(model))
    np.testing.assert_array_equal(np.asarray(strict.predict(model))[~supported], 0.0)
    beyond = _beyond_field(model, exposure)
    assert beyond.any()
    bare, _ = _speckle_setup(48, 24, field=False)
    static = np.asarray(term(observed, every, False).predict(bare))
    np.testing.assert_array_equal(mean[..., beyond], static[..., beyond])
    assert np.all(mean[~supported] > 0)

    speckle_only = ~supported
    loose, tight = term(observed, None, True), term(observed, None, False)
    assert float(loose(model)) == float(kept(model))
    assert np.isfinite(np.asarray(loose.residual(model))[speckle_only]).all()
    assert np.isnan(np.asarray(tight.residual(model))[speckle_only]).all()

    inside = jnp.asarray(supported)
    a, b = term(observed, inside, False), term(observed, inside, True)
    assert float(a(model)) == float(b(model))

    fit = Fit(canonical(model), law, (kept,))
    assert fit.assumptions == ("static_beyond_speckle",)
    assert Fit(canonical(model), law, (a,)).assumptions == ()
    start = {_CONTRAST: model.scene.planets[0].physical_model.contrast_grid}
    estimate = detection.measure(fit, start, n_newton=1)
    assert estimate.assumptions == ("static_beyond_speckle",)

    wide, wide_exposure = _speckle_setup(24, 48)
    no_static, only = _support_counts(wide, wide_exposure)
    assert no_static > 0 and only > 0
    frame = jnp.zeros((1, 48, 48))
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
