"""Exposure log-density: term routing, restriction and a staged self-fit."""

import equinox as eqx
import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import toy_coronagraph as toy

from photomancy.backends.staged import (
    FitStage,
    group_labels,
    map_fit,
    rescaled_transform,
)
from photomancy.exposure import ExposureLogDensity, ExposureModel, ExposureTerm
from photomancy.likelihoods import huber_loglike, laplace_loglike


def _affine_render(instrument, exposure, scene):
    return instrument * exposure + scene


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_terms_read_their_exposure_block_and_restrict_gives_a_view():
    """Value is the sum over terms, aux the images in order, ``restrict`` a sub-view.

    Truth source: hand-summed Laplace log-likelihoods of the affine images
    ``instrument * exposure[k] + scene`` (two terms on block 0, one on block 1).
    Tolerance basis: exact-algebra (rtol 1e-12; the same expression).
    Fails if: a term reads the wrong exposure block, the prior is dropped or counted
    per term, ``restrict`` reorders predictions or loses the prior, or an empty view
    is accepted.
    """
    model = ExposureModel(
        instrument=jnp.asarray([1.0, 2.0]),
        exposures=(jnp.asarray(3.0), jnp.asarray(-1.0)),
        scene=jnp.asarray([0.5, 0.5]),
    )
    data = [jnp.asarray([3.0, 6.0]), jnp.asarray([0.0, 0.0]), jnp.asarray([1.0, 2.0])]
    terms = tuple(
        ExposureTerm(_affine_render, laplace_loglike(d, 1.0), idx)
        for d, idx in zip(data, (0, 1, 0), strict=True)
    )
    ld = ExposureLogDensity(terms, prior=lambda m: -2.0)
    images = [
        np.asarray([3.5, 6.5]),
        np.asarray([-0.5, -1.5]),
        np.asarray([3.5, 6.5]),
    ]
    expected = [
        float(-np.sum(np.abs(img - np.asarray(d))) - 2.0 * np.log(2.0))
        for img, d in zip(images, data, strict=True)
    ]
    value, aux = ld.value_and_aux(model)
    np.testing.assert_allclose(value, sum(expected) - 2.0, rtol=1e-12)
    for got, want in zip(aux, images, strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(ld(model), value, rtol=1e-12)
    view = ld.restrict((2, 1))
    view_value, view_aux = view.value_and_aux(model)
    np.testing.assert_allclose(view_value, expected[2] + expected[1] - 2.0, rtol=1e-12)
    np.testing.assert_array_equal(view_aux[0], images[2])
    np.testing.assert_array_equal(view_aux[1], images[1])
    with pytest.raises(IndexError):
        ld.restrict((3,))
    with pytest.raises(ValueError, match="at least one term"):
        ld.restrict(())


_N_STEPS = 1000  # optimizer steps of the long reference and science stages
_TILT_STEPS = 150
_SEAM_NM, _TILT_PX = 1e-3, 2e-2  # Adam rates: 1 nm (in micrometers) and 0.02 pixel


class _CountingRender(toy.StarlightRender):
    """The toy render, recording one entry each time it is traced."""

    log: list = eqx.field(static=True)

    def __call__(self, instrument, star, scene):
        self.log.append(1)
        return super().__call__(instrument, star, scene)


def _cosine(rate, n_steps):
    return optax.cosine_decay_schedule(rate, n_steps, alpha=0.01)


def _window_flux(index, data, window):
    """Closed-form flux: the window ratio of data to the pre-step prediction."""

    def update(params, aux):
        star = params.exposures[0]
        flux = star.flux * jnp.sum(window * data) / jnp.sum(window * aux[index])
        return eqx.tree_at(lambda m: m.exposures[0].flux, params, flux)

    return update


def _self_fit(companion_peak):
    """Fit a two-exposure toy scene; return truth, the fit and the render log.

    Reference and science frames are noise free and share one instrument with seams
    ``ote`` and ``si`` (six mode coefficients in nm each). The science frame holds a
    companion whose peak is ``companion_peak`` (in the same units as the frame); the
    model keeps the companion flux at zero in a group no stage frees. The stages
    follow a reference-then-science fit: star position and flux, both seams and the
    star, then (fresh optimizers) the science star position and the exit-pupil seam.
    """
    instrument = toy.build_instrument()
    unocculted = eqx.tree_at(
        lambda i: i.occulter, instrument, jnp.ones_like(instrument.occulter)
    )
    peak = float(jnp.max(toy.point_image(unocculted, jnp.zeros(2))))
    rng = np.random.default_rng(0)
    truth_instrument = eqx.tree_at(
        lambda i: (i.ote, i.si),
        instrument,
        (jnp.asarray(rng.normal(size=6) * 4.0), jnp.asarray(rng.normal(size=6) * 4.0)),
    )
    ref_star = toy.Star(jnp.asarray([0.3, -0.2]), jnp.asarray(1.0 / peak))
    sci_star = toy.Star(jnp.asarray([-0.15, 0.25]), jnp.asarray(1.3 / peak))
    offset = jnp.asarray([6.0, -5.0])
    off = toy.Companion(offset, jnp.asarray(0.0))
    planet = toy.Companion(offset, jnp.asarray(companion_peak / peak))
    log = []
    render = _CountingRender(log=log)
    ref_data = render(truth_instrument, ref_star, off)
    sci_data = render(truth_instrument, sci_star, planet)

    c = (toy.N_PAD - 1) / 2.0
    yy, xx = np.mgrid[: toy.N_PAD, : toy.N_PAD] - c
    mask = jnp.asarray(np.hypot(xx, yy) >= 3.0)
    window = jnp.asarray(np.hypot(xx, yy) < 8.0) & mask
    n_used = int(mask.sum())
    sigma = 0.1 * float(jnp.median(ref_data))
    density = ExposureLogDensity(
        (
            ExposureTerm(render, huber_loglike(ref_data, sigma, mask), 0),
            ExposureTerm(render, huber_loglike(sci_data, sigma, mask), 0),
        )
    )
    init = ExposureModel(
        instrument,
        (toy.Star(jnp.zeros(2), 0.8 * ref_star.flux),),
        toy.Companion(offset, jnp.asarray(0.0)),
    )
    labels = group_labels(
        init,
        {
            "tilt": lambda m: [e.tilt for e in m.exposures],
            "ote": lambda m: m.instrument.ote,
            "si": lambda m: m.instrument.si,
            "companion": lambda m: m.scene.flux,
        },
    )
    n_long = _N_STEPS

    def seam(n):
        return rescaled_transform(optax.adam(_cosine(_SEAM_NM, n)), 1e3)

    first = {
        "tilt": optax.adam(_cosine(_TILT_PX, _TILT_STEPS + n_long)),
        "ote": seam(n_long),
        "si": seam(2 * n_long),
    }
    second = {
        "tilt": optax.adam(_cosine(_TILT_PX, _TILT_STEPS)),
        "si": seam(n_long),
    }
    ref, sci = density.restrict((0,)), density.restrict((1,))
    stages = [
        FitStage(
            _TILT_STEPS,
            ("tilt",),
            logdensity=ref,
            closed_form=_window_flux(0, ref_data, window),
            objective_scale=1.0 / n_used,
            name="reference star",
        ),
        FitStage(
            n_long,
            ("ote", "si", "tilt"),
            logdensity=ref,
            closed_form=_window_flux(0, ref_data, window),
            objective_scale=1.0 / n_used,
            name="reference seams",
        ),
        FitStage(
            _TILT_STEPS,
            ("tilt",),
            logdensity=sci,
            closed_form=_window_flux(0, sci_data, window),
            transforms=second,
            objective_scale=1.0 / n_used,
            name="science star",
        ),
        FitStage(
            n_long,
            ("si",),
            logdensity=sci,
            objective_scale=0.5 / n_used,
            name="science exit seam",
        ),
    ]
    fit = map_fit(density, init, labels, first, stages)
    return truth_instrument, sci_star, render, fit, log


def _relative_error(fit, truth):
    return float(jnp.linalg.norm(fit - truth) / jnp.linalg.norm(truth))


@vv.case("scene-fits", "code-verification", basis="spike")
@pytest.mark.medium
def test_self_fit_recovers_the_seams_and_leaves_the_companion_out():
    """A noise-free two-exposure toy fit recovers both seams; the companion stays at 0.

    Truth source: the injected seam coefficients (4 nm rms, ``default_rng(0)``) and
    star states of the toy coronagraph in ``toy_coronagraph``. The data are rendered
    from them with no noise, so recovery is limited by the optimizer and by the faint
    companion in the science frame (peak 3e-6, the per-pixel scale of the likelihood,
    0.03 percent of the brightest reference speckle), which the model cannot
    represent because its flux sits in a group no stage frees.
    Tolerance basis: spike, measured before the tolerances were fixed. Relative seam
    error (coefficient norm): with no companion ``ote`` 6.5e-3 and ``si`` 1.2e-2;
    with the companion ``ote`` 6.5e-3 and ``si`` 6.4e-3. Science star position error
    1.8e-4 pixel and flux error 1.6e-4 (relative). The tolerances are about 3 times
    the worst of each: seams 2e-2 (``ote``) and 4e-2 (``si``), position 5e-4 pixel,
    flux 5e-4.
    Fails if: a stage frees the wrong group, donated parameters or optimizer states
    are mixed across a stage boundary, the closed-form flux lags a step, a fresh
    transform table does not restart its clock, the step recompiles (the render is
    traced once per stage configuration, four times), or the companion flux is touched
    (it must stay exactly zero and the science model exactly the star-only model).
    """
    n_data_renders = 2  # the two eager renders that make the frames
    for peak in (0.0, 3e-6):
        truth, sci_star, render, fit, log = _self_fit(peak)
        instrument = fit.params.instrument
        assert _relative_error(instrument.ote, truth.ote) < 2e-2
        assert _relative_error(instrument.si, truth.si) < 4e-2
        star = fit.params.exposures[0]
        assert float(jnp.linalg.norm(star.tilt - sci_star.tilt)) < 5e-4
        assert float(abs(star.flux / sci_star.flux - 1.0)) < 5e-4
        assert len(log) - n_data_renders == 4
        assert float(fit.params.scene.flux) == 0.0
        np.testing.assert_array_equal(
            render(instrument, star, fit.params.scene),
            toy.star_only_image(instrument, star),
        )
