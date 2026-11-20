"""Exposure terms: routing in a ``Fit`` and a staged self-fit through ``MAP``."""

import json
from pathlib import Path

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import toy_coronagraph as toy
from hwoutils.model_paths import resolve
from priorities import Normal

from photomancy.exposure import ExposureModel, ExposureTerm
from photomancy.infer import Fit, run
from photomancy.infer.backends.staged import (
    MAP,
    MAPStage,
    group_labels,
    rescaled_transform,
)
from photomancy.infer.likelihoods import huber_loglike, laplace_loglike

_GOLDEN = Path(__file__).parent / "data" / "exposure_staged_golden.json"


def _affine_render(instrument, exposure, scene):
    return instrument * exposure + scene


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_terms_read_their_exposure_block_and_a_fit_sums_them():
    """The fit sums the terms; each term reads its own exposure block.

    Truth source: hand-summed Laplace log-likelihoods of the affine images
    ``instrument * exposure[k] + scene`` (two terms on block 0, one on block 1).
    Tolerance basis: exact-algebra (rtol 1e-12; the same expression).
    Fails if: a term reads the wrong exposure block, a term is dropped or counted
    twice, a subset of the terms (a stage view) scores more than its own terms, or
    ``predict`` returns an image other than the one the term scores.
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
    images = [
        np.asarray([3.5, 6.5]),
        np.asarray([-0.5, -1.5]),
        np.asarray([3.5, 6.5]),
    ]
    expected = [
        float(-np.sum(np.abs(img - np.asarray(d))) - 2.0 * np.log(2.0))
        for img, d in zip(images, data, strict=True)
    ]
    free = {"scene": Normal(jnp.zeros(2), jnp.ones(2))}
    x = {"scene": model.scene}
    fit = Fit(model, free, terms)
    np.testing.assert_allclose(fit.log_likelihood(x), sum(expected), rtol=1e-12)
    for term, want in zip(terms, images, strict=True):
        np.testing.assert_array_equal(term.predict(model), want)
    view = Fit(model, free, (terms[2], terms[1]))
    np.testing.assert_allclose(
        view.log_likelihood(x), expected[2] + expected[1], rtol=1e-12
    )


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


_STAR_FLUX = "exposures[0].flux"


def _window_flux(index, data, window):
    """Closed-form flux: the window ratio of data to the pre-step prediction.

    It reads the unconstrained entry dict (the identity under a ``Normal`` law) and
    ``aux[index]``, the prediction of term ``index`` before the optimizer step.
    """

    def update(u, aux):
        flux = u[_STAR_FLUX] * jnp.sum(window * data) / jnp.sum(window * aux[index])
        return {**u, _STAR_FLUX: flux}

    return update


# Wide enough that its gradient is below round-off next to the likelihood's. Adam
# rescales the step by the gradient size, so near the optimum a width of 1e3 (prior
# gradient 1e-6 per unit) moves the seam coefficients by about 1e-2 relative.
_PRIOR_WIDTH = 1e8


def _wide_laws(model, paths):
    """A ``Normal`` of width ``_PRIOR_WIDTH`` per path, the law of a MAP-only fit."""
    return {
        path: Normal(
            jnp.zeros(jnp.shape(resolve(model, path))),
            _PRIOR_WIDTH * jnp.ones(jnp.shape(resolve(model, path))),
        )
        for path in paths
    }


def _entries(model, paths):
    """The entry dict of ``model``: the value at each free path."""
    return {path: jnp.asarray(resolve(model, path)) for path in paths}


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
    terms = (
        ExposureTerm(render, huber_loglike(ref_data, sigma, mask), 0),
        ExposureTerm(render, huber_loglike(sci_data, sigma, mask), 0),
    )
    init = ExposureModel(
        instrument,
        (toy.Star(jnp.zeros(2), 0.8 * ref_star.flux),),
        toy.Companion(offset, jnp.asarray(0.0)),
    )
    # the companion flux stays at its model value (zero): it is not a free entry
    paths = ("exposures[0].tilt", _STAR_FLUX, "instrument.ote", "instrument.si")
    free = _wide_laws(init, paths)
    fit = Fit(init, free, terms)
    x0 = _entries(init, paths)
    labels = group_labels(
        fit.to_real(x0),
        {
            "tilt": lambda u: u["exposures[0].tilt"],
            "ote": lambda u: u["instrument.ote"],
            "si": lambda u: u["instrument.si"],
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
    ref, sci = Fit(init, free, terms[:1]), Fit(init, free, terms[1:])
    stages = [
        MAPStage(
            _TILT_STEPS,
            ("tilt",),
            fit=ref,
            closed_form=_window_flux(0, ref_data, window),
            objective_scale=1.0 / n_used,
            name="reference star",
        ),
        MAPStage(
            n_long,
            ("ote", "si", "tilt"),
            fit=ref,
            closed_form=_window_flux(0, ref_data, window),
            objective_scale=1.0 / n_used,
            name="reference seams",
        ),
        MAPStage(
            _TILT_STEPS,
            ("tilt",),
            fit=sci,
            closed_form=_window_flux(0, sci_data, window),
            transforms=second,
            objective_scale=1.0 / n_used,
            name="science star",
        ),
        MAPStage(
            n_long,
            ("si",),
            fit=sci,
            objective_scale=0.5 / n_used,
            name="science exit seam",
        ),
    ]
    backend = MAP(labels=labels, transforms=first, stages=tuple(stages), init=x0)
    n_before = len(log)
    posterior = run(fit, backend, jax.random.key(0))
    model = fit.model_at(jax.tree.map(lambda a: a[0], posterior.samples))
    return truth_instrument, sci_star, render, model, log, n_before


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
    represent because its flux is not a free entry.
    Tolerance basis: spike, measured before the tolerances were fixed. Relative seam
    error (coefficient norm): with no companion ``ote`` 6.5e-3 and ``si`` 1.2e-2;
    with the companion ``ote`` 6.5e-3 and ``si`` 6.4e-3. Science star position error
    1.8e-4 pixel and flux error 1.6e-4 (relative). The tolerances are about 3 times
    the worst of each: seams 2e-2 (``ote``) and 4e-2 (``si``), position 5e-4 pixel,
    flux 5e-4.
    Golden: ``exposure_staged_golden.json`` holds the final parameters of the same fit
    run through the legacy staged path (a density object with per-stage restrictions,
    no prior) before this port. The port agrees to 8e-14 relative at a prior width of
    1e8 (and to 1e-2 at 1e3, where Adam amplifies the prior gradient); the golden
    tolerance, rtol 1e-6, leaves room for a platform's round-off along 3000 Adam steps
    and still fails for a closed form that reads the post-update image.
    Fails if: a stage frees the wrong group, donated parameters or optimizer states
    are mixed across a stage boundary, the closed-form flux lags a step, a fresh
    transform table does not restart its clock, the step recompiles (the render is
    traced twice per stage configuration, once in the value and once in the
    prediction, four configurations, plus the two terms that ``run`` scores at the
    optimum), or the companion flux is touched (it must stay exactly zero and the
    science model exactly the star-only model).
    """
    golden = json.loads(_GOLDEN.read_text())["self_fit"]
    stage_configurations, traces_per_configuration = 4, 2
    n_scored_at_the_optimum = 2  # run evaluates both terms for log_target
    for peak in (0.0, 3e-6):
        truth, sci_star, render, model, log, n_before = _self_fit(peak)
        instrument = model.instrument
        assert _relative_error(instrument.ote, truth.ote) < 2e-2
        assert _relative_error(instrument.si, truth.si) < 4e-2
        star = model.exposures[0]
        assert float(jnp.linalg.norm(star.tilt - sci_star.tilt)) < 5e-4
        assert float(abs(star.flux / sci_star.flux - 1.0)) < 5e-4
        assert len(log) - n_before == (
            stage_configurations * traces_per_configuration + n_scored_at_the_optimum
        )
        assert float(model.scene.flux) == 0.0
        np.testing.assert_array_equal(
            render(instrument, star, model.scene),
            toy.star_only_image(instrument, star),
        )
        want = golden[repr(peak)]
        for name, got in (
            ("ote", instrument.ote),
            ("si", instrument.si),
            ("tilt", star.tilt),
            ("flux", star.flux),
        ):
            np.testing.assert_allclose(got, want[name], rtol=1e-6)
