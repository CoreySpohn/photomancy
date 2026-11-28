"""A joint star and planet fit on the toy coronagraph, started from the map peak."""

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
from hwoutils.numerics import roundoff
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from priorities import Normal
from term_utils import ExposureRecord, ExposureRender

from photomancy.detection import likelihood_ratio_map
from photomancy.infer import Fit, ForwardTerm, run
from photomancy.infer.backends.staged import (
    MAP,
    MAPStage,
    group_labels,
    rescaled_transform,
)
from photomancy.infer.likelihoods import gaussian_loglike, huber_loglike

pytestmark = pytest.mark.float64_only(reason="float64 Fit coordinates")

_N_LONG, _N_TILT, _N_PLANET, _N_FINAL = 600, 150, 300, 2500
_SEAM_NM, _TILT_PX = 1e-3, 2e-2
_TRUE_OFFSET = (6.3, -4.8)
_GOLDEN = Path(__file__).parent / "data" / "exposure_staged_golden.json"
# Wide enough that the prior gradient is below round-off next to the likelihood's
# (Adam rescales steps by the gradient size, so a narrower law moves the optimum).
_PRIOR_WIDTH = 1e8


class _PointTemplate(eqx.Module):
    instrument: toy.Instrument

    def __call__(self, position):
        return toy.point_image(self.instrument, position)


def _cosine(rate, n_steps):
    return optax.cosine_decay_schedule(rate, n_steps, alpha=0.01)


_STAR_FLUX = "exposures[0].flux"
_PATHS = (
    "exposures[0].tilt",
    _STAR_FLUX,
    "instrument.ote",
    "instrument.si",
    "scene.offset",
    "scene.flux",
)


def _window_flux(data, window):
    """Closed-form star flux from the pre-step prediction, on the entry dict."""

    def update(u, aux):
        flux = u[_STAR_FLUX] * jnp.sum(window * data) / jnp.sum(window * aux[0])
        return {**u, _STAR_FLUX: flux}

    return update


def _entries(model):
    """The entry dict of ``model``: the value at each free path."""
    return {path: jnp.asarray(resolve(model, path)) for path in _PATHS}


def _run_map(fit, backend):
    """Run ``backend`` on ``fit`` and return the model at the optimum."""
    posterior = run(fit, backend, jax.random.key(0))
    return fit.model_at(jax.tree.map(lambda a: a[0], posterior.samples))


def _joint_fit(peak_fraction):
    """Fit the toy scene without a planet (H0), map it, then free the planet (H1)."""
    instrument = toy.build_instrument()
    unocculted = eqx.tree_at(
        lambda i: i.occulter, instrument, jnp.ones_like(instrument.occulter)
    )
    peak = float(jnp.max(toy.point_image(unocculted, jnp.zeros(2))))
    rng = np.random.default_rng(0)
    truth = eqx.tree_at(
        lambda i: (i.ote, i.si),
        instrument,
        (jnp.asarray(rng.normal(size=6) * 4.0), jnp.asarray(rng.normal(size=6) * 4.0)),
    )
    ref_star = toy.Star(jnp.asarray([0.3, -0.2]), jnp.asarray(1.0 / peak))
    sci_star = toy.Star(jnp.asarray([-0.15, 0.25]), jnp.asarray(1.3 / peak))
    true_offset = jnp.asarray(_TRUE_OFFSET)
    true_flux = peak_fraction / peak
    render = toy.StarlightRender()
    absent = toy.Companion(true_offset, jnp.asarray(0.0))
    ref_data = render(truth, ref_star, absent)
    sci_data = render(
        truth, sci_star, toy.Companion(true_offset, jnp.asarray(true_flux))
    )

    c = (toy.N_PAD - 1) / 2.0
    yy, xx = np.mgrid[: toy.N_PAD, : toy.N_PAD] - c
    mask = jnp.asarray(np.hypot(xx, yy) >= 3.0)
    window = jnp.asarray(np.hypot(xx, yy) < 8.0) & mask
    n_used = int(mask.sum())
    sigma = 0.1 * float(jnp.median(ref_data))
    terms = (
        ForwardTerm(ExposureRender(render), huber_loglike(ref_data, sigma, mask)),
        ForwardTerm(ExposureRender(render), huber_loglike(sci_data, sigma, mask)),
    )
    init = ExposureRecord(
        instrument,
        (toy.Star(jnp.zeros(2), 0.8 * ref_star.flux),),
        absent,
    )
    # a wide Normal per entry: the MAP-only stand-in for a flat prior
    free = {
        path: Normal(
            jnp.zeros(jnp.shape(resolve(init, path))),
            _PRIOR_WIDTH * jnp.ones(jnp.shape(resolve(init, path))),
        )
        for path in _PATHS
    }
    fit = Fit(init, free, terms)
    ref, sci = Fit(init, free, terms[:1]), Fit(init, free, terms[1:])
    labels = group_labels(
        fit.to_real(_entries(init)),
        {
            "tilt": lambda u: u["exposures[0].tilt"],
            "ote": lambda u: u["instrument.ote"],
            "si": lambda u: u["instrument.si"],
            "offset": lambda u: u["scene.offset"],
            "flux": lambda u: u["scene.flux"],
            "star_flux": lambda u: u[_STAR_FLUX],
        },
    )

    def seam(n):
        return rescaled_transform(optax.adam(_cosine(_SEAM_NM, n)), 1e3)

    first = {
        "tilt": optax.adam(_cosine(_TILT_PX, _N_TILT + _N_LONG)),
        "ote": seam(_N_LONG),
        "si": seam(2 * _N_LONG),
    }
    second = {
        "tilt": optax.adam(_cosine(_TILT_PX, _N_TILT)),
        "si": seam(_N_LONG),
    }
    scale = 1.0 / n_used
    h0_stages = [
        MAPStage(
            _N_TILT,
            ("tilt",),
            fit=ref,
            objective_scale=scale,
            closed_form=_window_flux(ref_data, window),
            name="reference star",
        ),
        MAPStage(
            _N_LONG,
            ("ote", "si", "tilt"),
            fit=ref,
            objective_scale=scale,
            closed_form=_window_flux(ref_data, window),
            name="reference seams",
        ),
        MAPStage(
            _N_TILT,
            ("tilt",),
            fit=sci,
            objective_scale=scale,
            closed_form=_window_flux(sci_data, window),
            transforms=second,
            name="science star",
        ),
        MAPStage(
            _N_LONG,
            ("si",),
            fit=sci,
            objective_scale=0.5 * scale,
            name="science exit seam",
        ),
    ]
    h0 = _run_map(
        fit,
        MAP(
            labels=labels,
            transforms=first,
            stages=tuple(h0_stages),
            init=_entries(init),
        ),
    )

    grid = np.array(
        [(x, y) for x in range(-8, 9) for y in range(-8, 9) if np.hypot(x, y) >= 4.0],
        dtype=float,
    )
    base = render(h0.instrument, h0.exposures[0], h0.scene)
    detection = likelihood_ratio_map(
        (
            (
                gaussian_loglike(sci_data, sigma, mask),
                base,
                _PointTemplate(h0.instrument),
            ),
        ),
        jnp.asarray(grid),
    )
    best = int(jnp.argmax(detection.z))
    start = eqx.tree_at(
        lambda m: (m.scene.offset, m.scene.flux),
        h0,
        (jnp.asarray(grid[best]), detection.a_hat[best]),
    )

    planet_lr = {
        "offset": optax.adam(_cosine(2e-2, _N_PLANET + _N_FINAL)),
        "flux": rescaled_transform(
            optax.adam(_cosine(2e-2, _N_PLANET + _N_FINAL)), true_flux
        ),
        "si": seam(_N_FINAL),
        "tilt": optax.adam(_cosine(2e-3, _N_FINAL)),
        "star_flux": rescaled_transform(
            optax.adam(_cosine(1e-3, _N_FINAL)), float(h0.exposures[0].flux)
        ),
    }
    h1_stages = [
        MAPStage(
            _N_PLANET,
            ("offset", "flux"),
            fit=sci,
            objective_scale=scale,
            transforms=planet_lr,
            name="planet",
        ),
        MAPStage(
            _N_FINAL,
            ("si", "tilt", "star_flux", "offset", "flux"),
            fit=sci,
            objective_scale=scale,
            name="planet and exit seam",
        ),
    ]
    h1 = _run_map(
        fit,
        MAP(
            labels=labels,
            transforms=planet_lr,
            stages=tuple(h1_stages),
            init=_entries(start),
        ),
    )
    return {
        "truth": truth,
        "true_flux": true_flux,
        "true_offset": true_offset,
        "h0": h0,
        "h1": h1,
        "grid_best": grid[best],
        "a_hat": float(detection.a_hat[best]),
        "z": float(detection.z[best]),
        "sci_data": sci_data,
        "sci_fit": sci,
        "render": render,
        "mask": mask,
        "sigma": sigma,
    }


def _residual_rms(out, model):
    image = out["render"](model.instrument, model.exposures[0], model.scene)
    resid = (out["sci_data"] - image) / out["sigma"]
    return float(jnp.sqrt(jnp.mean(jnp.where(out["mask"], resid, 0.0) ** 2)))


def _seam_error(out, model):
    truth = out["truth"].si
    return float(jnp.linalg.norm(model.instrument.si - truth) / jnp.linalg.norm(truth))


@vv.case("scene-fits", "code-verification", basis="spike")
@pytest.mark.slow(
    reason="runs the map and staged companion fit and builds its image covariance, "
    "20 s projected on CI"
)
def test_planet_freed_after_the_map_recovers_flux_and_offset():
    """A noise-free toy fit starts the planet from the map peak and recovers it.

    Truth source: the injected companion (offset (6.3, -4.8) pixels, peak flux 3e-5 of
    the unocculted star peak) and seams of the toy coronagraph, rendered with no noise.
    The H0 fit never frees the planet (the frozen-companion control); the map is taken
    at the H0 fit; the H1 fit starts from the map's best pixel and amplitude and frees
    the planet, the exit seam, the star position and the star flux.
    Tolerance basis: spike, measured before the tolerances were fixed. Map: peak at the
    pixel (6, -5) with z 17.1 and an amplitude 0.945 of the truth (the H0 star
    absorbed part of the planet). H1: offset error 7e-4 pixel, flux error 5.6e-4
    (relative), residual rms 0.0076 sigma and exit-seam error 8.3e-3 (relative);
    H0 control: residual rms 0.59 sigma and exit-seam error 5.0e-2. The offset error
    is platform dependent because the fit stops before round-off (7e-4 pixel on macOS,
    2.9e-3 on the Linux runner); its tolerance, 1e-2 pixel, is still 36 times below
    the 0.36 pixel error of the map-peak start. The flux error also varies between
    runs (5.6e-4 on macOS, 2.1e-3 on the Linux runner); its tolerance, 1e-2, sits
    below the 2.5e-2 at which the fit stalls when the stage that frees the planet
    keeps the star flux fixed. The residual rms tolerance is about 4 times the H1 value
    (0.03) and the control must stay above 0.3 and 3e-2. The H1 exit-seam error is
    not bounded: the seam is weakly identified by the image, so where the fit stops on
    it depends on the platform (8.3e-3 on macOS, 3.3e-2 on the Linux runner).
    Golden: ``exposure_staged_golden.json`` holds the H0 and H1 parameters of the same
    fits through the legacy staged path (no prior) before the port to ``MAP``. H0 and
    H1 agree to 5e-14 relative on the capture platform; H0's frozen offset and flux
    are exact and H1 is held to rtol 1e-2 on the offset and flux because that fit
    stops before round-off and its optimum moves with the platform (see above).
    The H0 golden check compares the point images at the origin. Finite Adam
    schedules leave the fitted seams sensitive to platform arithmetic. The masked
    Huber science likelihood gives a formal covariance from the inverse Fisher
    information at the golden H0 fit, with the companion frozen and both seams,
    star tilt, and star flux allowed to vary. Propagating it through the point-image
    Jacobian gives each pixel's formal standard error. The quarter-sigma budget
    keeps the squared maximum image difference below one sixteenth of the largest
    formal pixel variance.
    Fails if: the map's amplitude or position is not a usable start, a stage that frees
    the planet leaves the star position or flux fixed so the planet cannot explain its
    own light, or the frozen-companion control absorbs the planet.
    """
    out = _joint_fit(3e-5)
    np.testing.assert_array_equal(out["grid_best"], [6.0, -5.0])
    assert out["z"] > 10.0
    assert 0.9 < out["a_hat"] / out["true_flux"] < 1.0
    h0, h1 = out["h0"], out["h1"]
    assert float(h0.scene.flux) == 0.0
    assert _residual_rms(out, h0) > 0.3
    assert _seam_error(out, h0) > 3e-2
    assert float(jnp.linalg.norm(h1.scene.offset - out["true_offset"])) < 1e-2
    assert abs(float(h1.scene.flux) / out["true_flux"] - 1.0) < 1e-2
    assert _residual_rms(out, h1) < 3e-2
    golden = json.loads(_GOLDEN.read_text())["joint_fit"]
    for name, model in (("h0", h0), ("h1", h1)):
        want = golden[name]
        rtol = 1e-6 if name == "h0" else 1e-2
        np.testing.assert_allclose(model.scene.offset, want["offset"], rtol=rtol)
        np.testing.assert_allclose(model.scene.flux, want["flux"], rtol=rtol, atol=0)
    golden_entries = {
        "exposures[0].tilt": jnp.asarray(golden["h0"]["tilt"]),
        _STAR_FLUX: jnp.asarray(golden["h0"]["star_flux"]),
        "instrument.ote": jnp.asarray(golden["h0"]["ote"]),
        "instrument.si": jnp.asarray(golden["h0"]["si"]),
    }
    sci = out["sci_fit"]
    h0_science_fit = Fit(
        h0, {path: sci.free[path] for path in golden_entries}, sci.terms
    )
    theta, unravel = ravel_pytree(golden_entries)
    origin = jnp.zeros(2)

    def point_image(flat):
        model = h0_science_fit.model_at(unravel(flat))
        return toy.point_image(model.instrument, origin)

    expected = point_image(theta)
    actual = toy.point_image(h0.instrument, origin)
    jac = jax.jacfwd(point_image)(theta).reshape(-1, theta.size)
    info = h0_science_fit.information(
        golden_entries, kind="fisher", include_prior=False
    )
    # If I = L L^T, the pixel variances are the squared column norms of L^-1 J^T.
    image_sigma = jnp.linalg.norm(
        solve_triangular(jnp.linalg.cholesky(info), jac.T, lower=True), axis=0
    )
    peak_sigma = float(jnp.max(image_sigma))
    assert np.isfinite(peak_sigma) and peak_sigma > 0
    atol = 0.25 * peak_sigma  # tol: statistical, quarter of peak formal error
    # One comparison subtraction at the summed magnitude of the two image operands.
    atol += float(roundoff(jnp.max(jnp.abs(expected) + jnp.abs(actual)), n_ops=1))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
