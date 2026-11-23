"""The pixel likelihood ``ImageTerm`` over ``coronagraphoto.expected``."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from image_scene import make_detector, make_exposure, make_instrument, make_scene
from priorities import LogUniform
from scipy import stats

from photomancy.image import ImageModel, ImageTerm
from photomancy.infer.fit import Fit

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
    mask = jnp.ones(mean.shape, bool).at[0, :3, :].set(False)
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
    mask = jnp.ones(observed.shape, bool)
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


def test_reads_are_the_held_products_reads_under_the_scene_path():
    """A held exposure that declares reads has them prefixed with the scene path."""
    model, exposure = _setup("poisson")
    observed = jnp.zeros((1, *model.instrument.detector.shape))
    assert _term(model, exposure, observed).reads == ()

    class Built(type(exposure)):
        reads: tuple[str, ...] = eqx.field(static=True, default=())

    built = Built(
        exposure.time_d,
        exposure.roll_deg,
        exposure.quadrature,
        exposure.plan,
        reads=("star.flux_phot_per_nm_m2", "dynamics.orbits"),
    )
    term = ImageTerm(observed, jnp.ones(observed.shape, bool), exposure=built, **_PATHS)
    assert term.reads == (
        "scene.star.flux_phot_per_nm_m2",
        "scene.dynamics.orbits",
    )
    rooted = ImageTerm(
        observed,
        jnp.ones(observed.shape, bool),
        instrument=model.instrument,
        exposure=built,
    )
    assert rooted.reads == ("star.flux_phot_per_nm_m2", "dynamics.orbits")


_CONTRAST = "scene.planets[0].physical_model.contrast_grid"


def test_an_image_fit_frees_a_scene_leaf_and_gradients_flow():
    """``Fit`` accepts the term over a scene leaf: finite density and gradient."""
    model, exposure = _setup("poisson")
    term = _term(model, exposure, jnp.round(_mean(model, exposure)))
    fit = Fit(model, {_CONTRAST: LogUniform(1e-12, 1e-8)}, (term,))
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
    mask = jnp.ones(observed.shape, bool)
    rooted = ImageTerm(observed, mask, instrument=model.instrument, exposure=exposure)
    path = "planets[0].physical_model.contrast_grid"
    law = LogUniform(1e-12, 1e-8)
    grid = model.scene.planets[0].physical_model.contrast_grid
    on_scene = Fit(model.scene, {path: law}, (rooted,)).log_density({path: grid})
    on_record = Fit(
        model, {f"scene.{path}": law}, (_term(model, exposure, observed),)
    ).log_density({f"scene.{path}": grid})
    np.testing.assert_allclose(on_scene, on_record, rtol=1e-12)


def test_a_free_leaf_that_a_held_build_read_is_refused():
    """A held exposure whose build read the contrast grid refuses to have it freed."""
    model, exposure = _setup("poisson")

    class Built(type(exposure)):
        reads: tuple[str, ...] = eqx.field(static=True, default=())

    built = Built(
        exposure.time_d,
        exposure.roll_deg,
        exposure.quadrature,
        exposure.plan,
        reads=("planets[0].physical_model.contrast_grid",),
    )
    observed = jnp.zeros((1, *model.instrument.detector.shape))
    term = ImageTerm(observed, jnp.ones(observed.shape, bool), exposure=built, **_PATHS)
    with pytest.raises(ValueError, match="reads"):
        Fit(model, {_CONTRAST: LogUniform(1e-12, 1e-8)}, (term,))


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

    Basis: exact. On a detector twice the tables, an all-True mask scores zero-rate
    pixels: ``check`` and ``Fit`` raise and name the count (the pixels outside
    ``supported_mask``). With ``mask & supported_mask`` the fit builds, and its
    log density equals that of the same data on a detector cropped to the supported
    square (an all-True mask there) to 1e-10 in the sum of Poisson masses.
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
        Fit(model, law, (full,))

    scored = _term(model, exposure, observed, jnp.asarray(supported))
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
        exposure=make_exposure(model.scene, instrument),
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
    Fit(model, {path: Uniform(nodes, nodes * 1.001)}, (term,))
    with pytest.raises(ValueError, match=r"scores .* pixel"):
        Fit(model, {path: Uniform(nodes * 0.9, nodes)}, (term,))
    with pytest.raises(ValueError, match="unbounded"):
        Fit(model, {path: Normal(nodes, jnp.ones_like(nodes))}, (term,))
