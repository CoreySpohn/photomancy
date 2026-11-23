"""Photometry integrates a rough tabulated star over a tabulated filter.

The truth is the band integral of the three tables, computed apart from the
operator that reduces the star: Gauss points on every cell between the knots of
the star, the planet's contrast and the filter, each read through its own
linear interpolant. The star is a synthetic line-rich table (80 absorption lines
0.15 to 0.50 nm wide, up to 90 percent deep, on 0.1 nm knots) standing in for a
PHOENIX spectrum, so four nodes per band cannot see the lines and a point sample
is wrong by percent. The filter is tabulated with zero tails that run past the
star table.
"""

import time

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from jax.test_util import check_grads
from numpy.polynomial.legendre import leggauss
from optixstuff.spectral import Band, Quadrature, Resolution, Transmission
from skyscapes.physical_model import TimeSeriesPhysicalModel
from skyscapes.scene import Planet, Scene, Star

from photomancy.measurements import PhotometryData, TargetData
from photomancy.measurements.targets import _nonzero_band

STAR_NM = np.arange(400.0, 900.0001, 0.1)
PLANET_NM = np.arange(300.0, 950.0001, 1.0)
# Zero response beyond 430 and 870 nm: the nonzero extent lies inside the star table.
FILTER_NM = np.array([300.0, 430.0, 460.0, 600.0, 700.0, 840.0, 870.0, 950.0])
FILTER_T = np.array([0.0, 0.0, 0.7, 0.9, 0.8, 0.6, 0.0, 0.0])


def _star(flux_scale=1.0):
    rng = np.random.default_rng(11)
    flux = np.full(STAR_NM.size, 3000.0)
    for centre, width, depth in zip(
        rng.uniform(400.0, 900.0, 80),
        rng.uniform(0.15, 0.50, 80),
        rng.uniform(0.3, 0.9, 80),
        strict=True,
    ):
        flux *= 1.0 - depth * np.exp(-0.5 * ((STAR_NM - centre) / width) ** 2)
    return Star(
        wavelengths_nm=jnp.asarray(STAR_NM),
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.asarray(flux)[:, None],
        interpolation_method="linear",
        flux_scale=flux_scale,
    )


def _contrast(nm):
    bump = lambda c: np.clip(1 - np.abs(nm - c) / 25.0, 0, None)  # noqa: E731
    return 3e-10 * (1 - 0.7 * bump(527.0) - 0.7 * bump(760.0))


def _scene():
    model = TimeSeriesPhysicalModel(
        PLANET_NM, np.zeros(1), _contrast(PLANET_NM)[None, :, None]
    )
    return Scene.from_classical(
        _star(),
        (Planet(jnp.ones(1), model),),
        star_mass_Msun=1.0,
        planet_mass_Msun=317.8 * Mearth2kg / Msun2kg,
        dist_pc=10.0,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.array([1.0]),
        e=0.0,
        W_rad=0.0,
        i_rad=0.5,
        w_rad=0.0,
        M0_rad=0.3,
    )


def _filter():
    return Transmission(FILTER_NM, FILTER_T)


def _target(n_rows=1):
    data = PhotometryData(
        time_d=jnp.zeros(n_rows),
        contrast=jnp.full(n_rows, 1e-10),
        contrast_err=jnp.full(n_rows, 1e-11),
        band_id=jnp.zeros(n_rows, dtype=int),
        planet=jnp.zeros(n_rows, dtype=int),
        is_upper_limit=jnp.zeros(n_rows, dtype=bool),
        valid=jnp.ones(n_rows, dtype=bool),
    )
    return TargetData(
        target={},
        relative_astrometry=None,
        photometry=data,
        band_names=("F",),
        comparisons={},
        epoch=Epoch(2451545.0),
    )


def _term(scene):
    (term,) = _target().terms(system=scene, transmissions={"F": _filter()})
    return term


def _truth(scene):
    """Exact ratio of the planet and star photons through the filter."""
    band = _nonzero_band(_filter())
    grids = [g[(g > band.low_nm) & (g < band.high_nm)] for g in (STAR_NM, FILTER_NM)]
    edges = np.union1d(np.concatenate([[band.low_nm, band.high_nm], *grids]), [])
    x, w = leggauss(8)
    length = np.diff(edges)[:, None]
    nm = (edges[:-1, None] + length * (x + 1) / 2).ravel()
    weight = (length * w / 2).ravel()
    star = np.asarray(
        scene.star.photon_flux_density(jnp.asarray(nm), jnp.zeros(()), dist_pc=10.0)
    )
    response = np.interp(nm, FILTER_NM, FILTER_T)
    contrast = np.interp(nm, PLANET_NM, _contrast(PLANET_NM))
    return np.sum(weight * star * response * contrast) / np.sum(
        weight * star * response
    )


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_photometry_integrates_the_star_over_a_tabulated_filter():
    """Fails if the contrast samples a rough star at nodes or ignores its table.

    Truth: eight-point Gauss quadrature on every cell between the star and
    filter knots, the star and filter through their linear interpolants and the
    contrast through ``numpy.interp``. Tolerance basis: frozen 1e-4 relative; the
    two integrals agree to rounding (about 1e-15). A point sample at the four
    nodes is wrong by about 1e-2 on this star.
    """
    scene = _scene()
    term = _term(scene)
    prediction = float(term.predict(scene)[0])
    np.testing.assert_allclose(prediction, _truth(scene), rtol=1e-4)

    sampled = scene.band_contrasts(term.quadrature, jnp.zeros(1), transmission=None)
    assert abs(float(sampled[0, 0, 0]) / _truth(scene) - 1.0) > 1e-3


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_photometry_gradients_through_the_resolution():
    """Fails if a gradient through the star table or the filter is wrong.

    Truth: central finite differences. Tolerance basis: the existing
    ``check_grads`` setting, eps 1e-4 with atol and rtol 1e-4, float64, forward
    and reverse to first order.
    """
    scene = _scene()
    term = _term(scene)

    def by_star(flux):
        model = eqx.tree_at(lambda s: s.star._flux_density_phot, scene, flux)
        return term.predict(model)[0] * 1e10

    def by_filter(response):
        held = eqx.tree_at(lambda t: t.transmissions[0].transmission, term, response)
        return held.predict(scene)[0] * 1e10

    with jax.enable_x64():
        for fun, arg in (
            (by_star, scene.star._flux_density_phot),
            (by_filter, term.transmissions[0].transmission),
        ):
            check_grads(
                fun,
                (arg,),
                order=1,
                modes=("fwd", "rev"),
                eps=1e-4,
                atol=1e-4,
                rtol=1e-4,
            )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_a_filter_band_is_trimmed_to_its_nonzero_extent():
    """Fails if a band keeps a zero tail, which leaves the star table and is refused.

    Truth: the response is linear between knots, so it is nonzero strictly
    between the knot before the first positive value and the knot after the
    last. The edges are exact table knots (tolerance: equality). The untrimmed
    band 300 to 950 nm leaves the 400 to 900 nm star table and is refused.
    """
    band = _nonzero_band(_filter())
    assert (band.low_nm, band.high_nm) == (430.0, 870.0)
    flat = Transmission([500.0, 600.0], [1.0, 1.0])
    assert _nonzero_band(flat) == Band(500.0, 600.0)
    assert _nonzero_band(Transmission([500.0, 600.0], [0.0, 0.0])) == Band(500.0, 600.0)
    scene = _scene()
    with pytest.raises(ValueError):
        wide = Quadrature.build((Band(300.0, 950.0),), _filter())
        Resolution.build((wide,), *scene.spectral_tables())
    assert _term(scene).quadrature.bands[0] == Band(430.0, 870.0)


def test_warm_cost_of_the_integrated_contrast():
    """Reports the warm call time of the integrated contrast beside the point sample.

    There is no pass criterion on speed (correctness gates, speed is reported);
    both calls must run and be finite.
    """
    scene = _scene()
    term = _term(scene)
    times = jnp.zeros(1)
    integrated = jax.jit(lambda model: term(model))
    sampled = jax.jit(lambda model: model.band_contrasts(term.quadrature, times))

    def warm(fun):
        jax.block_until_ready(fun(scene))
        start = time.perf_counter()
        for _ in range(20):
            jax.block_until_ready(fun(scene))
        return (time.perf_counter() - start) / 20

    cost = warm(integrated), warm(sampled)
    ms = [c * 1e3 for c in cost]
    print(f"warm call ms: integrated {ms[0]:.3f}, point sample {ms[1]:.3f}")
    assert np.isfinite(float(integrated(scene)))
    assert np.all(np.isfinite(np.asarray(sampled(scene))))
