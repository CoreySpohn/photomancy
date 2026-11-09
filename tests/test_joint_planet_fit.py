"""A joint star and planet fit on the toy coronagraph, started from the map peak."""

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
from photomancy.detection import likelihood_ratio_map
from photomancy.exposure import ExposureLogDensity, ExposureModel, ExposureTerm
from photomancy.likelihoods import gaussian_loglike, huber_loglike

_N_LONG, _N_TILT, _N_PLANET, _N_FINAL = 600, 150, 300, 2500
_SEAM_NM, _TILT_PX = 1e-3, 2e-2
_TRUE_OFFSET = (6.3, -4.8)


class _PointTemplate(eqx.Module):
    instrument: toy.Instrument

    def __call__(self, position):
        return toy.point_image(self.instrument, position)


def _cosine(rate, n_steps):
    return optax.cosine_decay_schedule(rate, n_steps, alpha=0.01)


def _window_flux(data, window):
    def update(params, aux):
        star = params.exposures[0]
        flux = star.flux * jnp.sum(window * data) / jnp.sum(window * aux[0])
        return eqx.tree_at(lambda m: m.exposures[0].flux, params, flux)

    return update


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
    density = ExposureLogDensity(
        (
            ExposureTerm(render, huber_loglike(ref_data, sigma, mask), 0),
            ExposureTerm(render, huber_loglike(sci_data, sigma, mask), 0),
        )
    )
    ref, sci = density.restrict((0,)), density.restrict((1,))
    init = ExposureModel(
        instrument,
        (toy.Star(jnp.zeros(2), 0.8 * ref_star.flux),),
        absent,
    )
    labels = group_labels(
        init,
        {
            "tilt": lambda m: [e.tilt for e in m.exposures],
            "ote": lambda m: m.instrument.ote,
            "si": lambda m: m.instrument.si,
            "offset": lambda m: m.scene.offset,
            "flux": lambda m: m.scene.flux,
            "star_flux": lambda m: m.exposures[0].flux,
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
        FitStage(
            _N_TILT,
            ("tilt",),
            logdensity=ref,
            objective_scale=scale,
            closed_form=_window_flux(ref_data, window),
            name="reference star",
        ),
        FitStage(
            _N_LONG,
            ("ote", "si", "tilt"),
            logdensity=ref,
            objective_scale=scale,
            closed_form=_window_flux(ref_data, window),
            name="reference seams",
        ),
        FitStage(
            _N_TILT,
            ("tilt",),
            logdensity=sci,
            objective_scale=scale,
            closed_form=_window_flux(sci_data, window),
            transforms=second,
            name="science star",
        ),
        FitStage(
            _N_LONG,
            ("si",),
            logdensity=sci,
            objective_scale=0.5 * scale,
            name="science exit seam",
        ),
    ]
    h0 = map_fit(density, init, labels, first, h0_stages).params

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
        FitStage(
            _N_PLANET,
            ("offset", "flux"),
            logdensity=sci,
            objective_scale=scale,
            transforms=planet_lr,
            name="planet",
        ),
        FitStage(
            _N_FINAL,
            ("si", "tilt", "star_flux", "offset", "flux"),
            logdensity=sci,
            objective_scale=scale,
            name="planet and exit seam",
        ),
    ]
    h1 = map_fit(density, start, labels, planet_lr, h1_stages).params
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
@pytest.mark.medium
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
