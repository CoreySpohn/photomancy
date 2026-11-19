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

from photomancy.exposure import ExposureModel
from photomancy.image import ImageTerm
from photomancy.infer.fit import Fit


def _setup(kind="poisson", contrast=1e-10):
    instrument = make_instrument(make_detector(kind))
    scene = make_scene(contrast=contrast, sky_pa_deg=40.0)
    exposure = make_exposure(scene, instrument)
    return ExposureModel(instrument, (), scene), exposure


def _term(model, exposure, observed, mask=None):
    observed = jnp.asarray(observed)
    mask = jnp.ones(observed.shape, bool) if mask is None else mask
    return ImageTerm(observed, mask, exposure=exposure)


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
        observed, jnp.ones(observed.shape, bool), exposure_path="nuisance[0]"
    )
    with_exposure = ExposureModel(model.instrument, (exposure,), model.scene)
    np.testing.assert_allclose(by_path(with_exposure), held(model), rtol=1e-12)
    with pytest.raises(ValueError, match="exactly one"):
        ImageTerm(observed, jnp.ones(observed.shape, bool))
    with pytest.raises(ValueError, match="exactly one"):
        ImageTerm(
            observed,
            jnp.ones(observed.shape, bool),
            exposure=exposure,
            exposure_path="nuisance[0]",
        )
    with pytest.raises(ValueError, match="FramePlan"):
        ImageTerm(
            observed,
            jnp.ones(observed.shape, bool),
            exposure=eqx.tree_at(
                lambda e: e.plan, exposure, None, is_leaf=lambda x: x is None
            ),
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
    term = ImageTerm(observed, jnp.ones(observed.shape, bool), exposure=built)
    assert term.reads == (
        "scene.star.flux_phot_per_nm_m2",
        "scene.dynamics.orbits",
    )


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
    term = ImageTerm(observed, jnp.ones(observed.shape, bool), exposure=built)
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
    model = ExposureModel(instrument, (), scene)
    shape = (1, *dead.shape)
    zero = _term(model, exposure, jnp.zeros(shape))
    one = _term(model, exposure, jnp.zeros(shape).at[0, 5, 5].set(1.0))
    assert float(zero(model)) == 0.0
    assert float(one(model)) == -np.inf
