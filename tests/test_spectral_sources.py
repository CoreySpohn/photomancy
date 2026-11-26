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
from hwoutils.testing import ulps
from jax.test_util import check_grads
from numpy.polynomial.legendre import leggauss
from optixstuff.spectral import Band, Quadrature, Resolution, Transmission
from priorities import Normal
from skyscapes.physical_model import TimeSeriesPhysicalModel
from skyscapes.scene import Planet, Scene, Star

from photomancy.infer import Fit
from photomancy.measurements import (
    CountRateTable,
    PhotometryData,
    PhotometryTerm,
    PrecomputedReportPredictor,
    ReportData,
    ReportRule,
    ReportTerm,
    TargetData,
)
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


def _point_sampled(scene, quadrature, time_d):
    """Band contrast from point samples of the star and planet at the nodes."""
    nodes = quadrature.nodes_nm
    planet = jax.vmap(jax.vmap(lambda w: scene.planet_flux_densities(w, time_d)))(nodes)
    star = scene.star.photon_flux_density(
        nodes[..., None], time_d, dist_pc=scene.dist_pc
    )
    return quadrature.integrate(planet) / quadrature.integrate(star)[:, None, :]


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

    sampled = _point_sampled(scene, term.quadrature, jnp.zeros(1))
    assert abs(float(sampled[0, 0, 0]) / _truth(scene) - 1.0) > 1e-3


@pytest.mark.float64_only(reason="finite differences at eps=1e-4")
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
    integrated = jax.jit(lambda model: term.predict(model))
    sampled = jax.jit(lambda model: _point_sampled(model, term.quadrature, times))

    def warm(fun):
        jax.block_until_ready(fun(scene))
        start = time.perf_counter()
        for _ in range(20):
            jax.block_until_ready(fun(scene))
        return (time.perf_counter() - start) / 20

    cost = warm(integrated), warm(sampled)
    ms = [c * 1e3 for c in cost]
    print(f"warm call ms: integrated {ms[0]:.3f}, point sample {ms[1]:.3f}")
    assert np.all(np.isfinite(np.asarray(integrated(scene))))
    assert np.all(np.isfinite(np.asarray(sampled(scene))))


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_the_report_predictor_integrates_a_tabulated_star():
    """Fails if a precomputed report predictor point-samples a rough star.

    Truth: the same Gauss-cell band ratio as the photometric term (``_truth``)
    times the flat tabulated rate and the exposure time. Tolerance basis:
    both are exact quadratures of piecewise polynomials, so they agree to
    rounding of cell interpolation, weighted normalization and scaling.
    Thirty-two eps cover two interpolations, products, a pairwise sum of
    a few hundred terms, the quotient and the two count-scale multipliers.
    The point sample at the nodes is off by 4.5e-2 on this star, and a table
    without a resolution is refused rather than sampled.
    """
    scene = _scene()
    curve = _filter()
    rule = Quadrature.build((_nonzero_band(curve),), curve)
    resolution = Resolution.build((rule,), *scene.spectral_tables())
    table = CountRateTable(
        jnp.array([[0.0, 10.0]]),
        jnp.array([[2.0, 2.0]]),
        jnp.array([0.03]),
        rule,
        (curve,),
        scene.epoch,
        resolution=resolution,
    )
    rows = ReportData.planned(
        jnp.zeros(1), jnp.array([5.0]), jnp.array([1.0]), jnp.array([0])
    )
    predictor = PrecomputedReportPredictor(table, 0)
    counts = predictor(scene, rows).counts[0]
    np.testing.assert_allclose(
        counts,
        _truth(scene) * 2.0 * 5.0,
        rtol=float(ulps(32, jnp.ones((), counts.dtype))),
        atol=0,
    )
    term = ReportTerm(ReportRule(threshold=4.5, kappa=1.3, corr=0.4), rows, predictor)
    assert term.reads == () and table.resolution.rules
    bare = eqx.tree_at(lambda t: t.resolution, table, Resolution())
    with pytest.raises(LookupError):
        PrecomputedReportPredictor(bare, 0)(scene, rows)


class _Holder(eqx.Module):
    system: Scene


def test_terms_holding_a_resolution_declare_no_reads():
    """Fails if a term with a held resolution needs, or accepts, a knot declaration.

    A photometric term and a count-rate table are built directly on a
    resolution that holds rules, with no ``reads``: both construct and declare
    ``()``, and the term the target data builds declares the same. Passing
    ``reads=`` is a ``TypeError`` for both (the argument is gone). Truth: the
    construction contract itself, so there is no tolerance.
    """
    scene = _scene()
    (term,) = _target().terms("system", system=scene, transmissions={"F": _filter()})
    assert term.resolution.rules
    assert term.reads == ()
    parts = (term.data, term.quadrature, term.transmissions, "system")
    held = PhotometryTerm(*parts, resolution=term.resolution)
    assert held.reads == () and held.resolution.rules
    grid = (jnp.array([[0.0, 10.0]]), jnp.array([[2.0, 2.0]]), jnp.array([0.03]))
    table = (*grid, term.quadrature, term.transmissions, scene.epoch)
    counts = CountRateTable(*table, resolution=term.resolution)
    assert counts.resolution.rules and counts.reads == ()
    with pytest.raises(TypeError):
        PhotometryTerm(*parts, resolution=term.resolution, reads=())
    with pytest.raises(TypeError):
        CountRateTable(*table, resolution=term.resolution, reads=())


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_a_fit_frees_the_star_flux_table_under_a_held_resolution():
    """Fails if a fit refuses the star's flux table, or its prediction goes stale.

    ``Fit`` frees the flux table of the star (a leaf, since the knots are
    static) with a photometric term whose resolution was built before the move.
    The fit is accepted, and the prediction of the model with the moved table
    equals that of a term rebuilt on the moved scene. Truth: the rebuilt term.
    Tolerance basis: the operator is linear in the values, so the two differ by
    float64 rounding of a sum of a few hundred cells; rtol 1e-12 (the move
    changes the prediction by far more, which the test also requires). The
    held prediction is also compared with the independent ``_truth`` of the
    moved scene (eight-point Gauss on every cell between the knots), at rtol 1e-10:
    both are float64 sums of the same piecewise-linear integrand over a few
    hundred cells, so they differ by accumulated rounding, about 1e-14.
    """
    scene = _scene()
    (term,) = _target().terms("system", system=scene, transmissions={"F": _filter()})
    flux = scene.star._flux_density_phot
    path = "system.star._flux_density_phot"
    Fit(_Holder(scene), {path: Normal(flux, 1.0)}, (term,))
    ripple = 1.0 + 0.3 * jnp.sin(jnp.arange(flux.shape[0]) / 37.0)[:, None]
    moved = eqx.tree_at(lambda s: s.star._flux_density_phot, scene, flux * ripple)
    held = float(term.predict(_Holder(moved))[0])
    rebuilt = float(_term(moved).predict(moved)[0])
    assert abs(held / float(term.predict(_Holder(scene))[0]) - 1.0) > 1e-4
    np.testing.assert_allclose(held, rebuilt, rtol=1e-12)
    np.testing.assert_allclose(held, _truth(moved), rtol=1e-10)
