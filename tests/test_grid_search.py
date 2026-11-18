"""Tests for orbix grid-search utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G
from hwoutils.epoch import Epoch
from orbix.system import EccentricityVectorChart

from photomancy.orbit._frame import ORBIT_ROWS_CHART, sky_offsets_arcsec
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.grid_search import (
    AdaptiveImportanceSampler,
    EccVectorShape,
    ParamBounds,
    _box_log_jacobian,
    _box_to_chart,
    _ecc_square_disk_area,
    batched_loglike,
    build_evaluator,
    chart_box_bounds,
    chart_grid_search,
    grid_search,
)
from photomancy.orbit.inference import orbit_chart_log_volume
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.quasi_random import roberts_sequence
from photomancy.posterior import SamplePosterior

TWO_PI = 2.0 * jnp.pi
MSUN = 1.98892e30


def test_roberts_shape_and_range():
    """Roberts sequence has correct shape and values in [0, 1)."""
    pts = roberts_sequence(1000, 3)
    assert pts.shape == (1000, 3)
    assert jnp.all((pts >= 0.0) & (pts < 1.0))


def test_roberts_low_discrepancy_1d_margin():
    """Max gap in sorted 1D projection is within 10x mean gap."""
    pts = roberts_sequence(2000, 2)
    xs = jnp.sort(pts[:, 0])
    gaps = jnp.diff(xs)
    assert float(jnp.max(gaps)) < 10.0 * float(jnp.mean(gaps))


def test_parambounds_scale_unit_cube():
    """ParamBounds.scale maps unit-cube corners and midpoint correctly."""
    b = ParamBounds(
        names=("logT", "cos_i"),
        low=jnp.array([0.0, -1.0]),
        high=jnp.array([4.0, 1.0]),
    )
    u = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    phys = b.scale(u)
    assert phys.shape == (3, 2)
    assert jnp.allclose(phys[0], jnp.array([0.0, -1.0]))
    assert jnp.allclose(phys[1], jnp.array([4.0, 1.0]))
    assert jnp.allclose(phys[2], jnp.array([2.0, 0.0]))


def test_eccvector_shape_maps_to_physical():
    """EccVectorShape converts unit-cube midpoint to expected physical values."""
    from photomancy.orbit.grid_search import EccVectorShape

    shape = EccVectorShape()
    bounds = shape.default_bounds(log_T_range=(2.0, 3.0), e_max=0.5)
    u = jnp.full((1, len(bounds.names)), 0.5)
    phys = shape.to_physical(u, bounds, Ms=MSUN)
    for kkey in ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp"):
        assert kkey in phys
        assert phys[kkey].shape == (1,)
    # ex=ey=0 at midpoint of symmetric (-e_max, e_max) -> e=0, cos_w=1, sin_w=0
    assert float(phys["e"][0]) == 0.0
    assert jnp.allclose(phys["cos_w"][0], 1.0)
    assert jnp.allclose(phys["sin_w"][0], 0.0)
    # e <= e_max always
    u2 = jnp.array([[0.5, 1.0, 1.0, 0.5, 0.5, 0.5]])
    e2 = float(shape.to_physical(u2, bounds, Ms=MSUN)["e"][0])
    assert 0.0 <= e2 < 1.0


def test_ais_stage1_fills_unit_cube():
    """AdaptiveImportanceSampler.stage1 produces valid unit-cube points."""
    from photomancy.orbit.grid_search import AdaptiveImportanceSampler

    s = AdaptiveImportanceSampler()
    u = s.stage1(jax.random.PRNGKey(0), ndim=6, n=4096)
    assert u.shape == (4096, 6)
    assert jnp.all((u >= 0.0) & (u < 1.0))


def _toy_astrom():
    """Build a small RelativeAstromData fixture from a known orbit."""
    t = jnp.array([0.0, 120.0, 240.0])
    a, e, cos_i, W = 1.0, 0.1, 0.7, 0.5
    cos_w, sin_w, tp = 1.0, 0.0, 30.0
    ra, dec = predict_relative_astrometry(
        t, a, e, cos_i, W, cos_w, sin_w, tp, MSUN, 10.0
    )
    err = jnp.full(3, 1e-3)
    return RelativeAstromData(
        times=t,
        ra=ra,
        dec=dec,
        ra_err=err,
        dec_err=err,
        corr=jnp.zeros(3),
        planet_id=jnp.zeros(3, int),
        is_valid=jnp.ones(3, bool),
    )


def test_evaluator_matches_direct_loglike():
    """build_evaluator returns a function matching loglike_relative_astrom."""
    from photomancy.orbit.grid_search import EccVectorShape, build_evaluator

    data = _toy_astrom()
    shape = EccVectorShape()
    ev = build_evaluator((data,), Ms=MSUN, dist_pc=10.0, shape=shape)
    phys = {
        "a": jnp.array(1.0),
        "e": jnp.array(0.1),
        "cos_i": jnp.array(0.7),
        "W": jnp.array(0.5),
        "cos_w": jnp.array(1.0),
        "sin_w": jnp.array(0.0),
        "tp": jnp.array(30.0),
    }
    ra, dec = predict_relative_astrometry(
        data.times,
        phys["a"],
        phys["e"],
        phys["cos_i"],
        phys["W"],
        phys["cos_w"],
        phys["sin_w"],
        phys["tp"],
        MSUN,
        10.0,
    )
    assert jnp.allclose(ev(phys), loglike_relative_astrom(ra, dec, data))


# Medium: comparing two independently compiled orbit evaluators exceeds the unit budget.
@pytest.mark.medium
def test_batched_loglike_matches_unfused():
    """batched_loglike via scan+vmap matches direct vmap over all particles."""
    from photomancy.orbit.grid_search import (
        AdaptiveImportanceSampler,
        EccVectorShape,
        batched_loglike,
        build_evaluator,
    )

    data = _toy_astrom()
    shape = EccVectorShape()
    ev = build_evaluator((data,), Ms=MSUN, dist_pc=10.0, shape=shape)
    bounds = shape.default_bounds(log_T_range=(2.0, 3.0), e_max=0.5)
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(0), 6, 24)
    phys = shape.to_physical(u, bounds, Ms=MSUN)
    fused = batched_loglike(ev, phys, n_particles=24, chunk_size=6)
    ref = jax.vmap(ev)({k: v for k, v in phys.items()})
    assert fused.shape == (24,)
    assert jnp.allclose(fused, ref, atol=1e-5)


def test_stage2_returns_samples_and_logq():
    """stage2 returns unit-cube samples and finite log-densities."""
    from photomancy.orbit.grid_search import AdaptiveImportanceSampler

    s = AdaptiveImportanceSampler(n_modes=3)
    survivors = jax.random.uniform(jax.random.PRNGKey(2), (50, 6))
    z, log_q = s.stage2(jax.random.PRNGKey(3), survivors, n=100)
    assert z.shape == (100, 6)
    assert log_q.shape == (100,)
    assert jnp.all(jnp.isfinite(log_q))


def test_sample_posterior_sir_sampling():
    """SamplePosterior.sample draws by weight; near-zero weight never drawn."""
    parts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    logw = jnp.array([-1e9, 0.0])
    pp = SamplePosterior(
        samples=parts,
        log_weights=logw,
        evidence=jnp.asarray(jnp.nan),
        param_names=("x", "y"),
    )
    out = pp.sample_dict(jax.random.PRNGKey(0), n=64)
    assert set(out) == {"x", "y"}
    assert out["x"].shape == (64,)
    assert jnp.allclose(out["x"], 1.0)


def test_grid_search_recovers_period():
    """grid_search weighted posterior median of 'a' brackets truth (a=1.0 AU)."""
    from photomancy.orbit.grid_search import (
        AdaptiveImportanceSampler,
        EccVectorShape,
        grid_search,
    )

    data = _toy_astrom()
    pp = grid_search(
        (data,),
        Ms=MSUN,
        dist_pc=10.0,
        shape=EccVectorShape(),
        sampler=AdaptiveImportanceSampler(n_modes=5),
        log_T_range=(2.0, 3.0),
        e_max=0.5,
        n_particles=20000,
        chunk_size=5000,
        n_survivors=500,
        key=jax.random.PRNGKey(0),
    )
    draws = pp.sample_dict(jax.random.PRNGKey(1), n=2000)
    med_a = float(jnp.median(draws["a"]))
    assert 0.7 < med_a < 1.4
    assert jnp.isfinite(pp.evidence)  # importance-sampling log Z


def test_chunking_is_shape_stable():
    """grid_search log_weights are identical regardless of chunk_size."""
    from photomancy.orbit.grid_search import (
        AdaptiveImportanceSampler,
        EccVectorShape,
        grid_search,
    )

    data = _toy_astrom()
    shape = EccVectorShape()
    common = dict(
        Ms=MSUN,
        dist_pc=10.0,
        shape=shape,
        sampler=AdaptiveImportanceSampler(n_modes=5),
        log_T_range=(2.0, 3.0),
        e_max=0.5,
        n_particles=12000,
        n_survivors=400,
        key=jax.random.PRNGKey(7),
    )
    a = grid_search((data,), chunk_size=2000, **common).log_weights
    b = grid_search((data,), chunk_size=4000, **common).log_weights
    assert jnp.allclose(a, b, atol=1e-5)


def test_public_api_exports():
    """All six grid-search names are in photomancy.orbit.__all__ and accessible."""
    import photomancy.orbit as f

    for name in (
        "grid_search",
        "EccVectorShape",
        "AdaptiveImportanceSampler",
        "AbstractGridStrategy",
        "AbstractShapeParam",
        "ParamBounds",
    ):
        assert name in f.__all__ and hasattr(f, name)


def test_batched_loglike_rejects_indivisible_chunk():
    """batched_loglike raises ValueError when n_particles is not divisible by chunk_size."""  # noqa: E501
    data = _toy_astrom()
    shape = EccVectorShape()
    ev = build_evaluator((data,), Ms=MSUN, dist_pc=10.0, shape=shape)
    bounds = shape.default_bounds(log_T_range=(2.0, 3.0), e_max=0.5)
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(0), 6, 100)
    phys = shape.to_physical(u, bounds, Ms=MSUN)
    with pytest.raises(ValueError):
        batched_loglike(ev, phys, n_particles=100, chunk_size=30)


def test_stage2_rejects_too_few_survivors():
    """stage2 raises ValueError when survivors count is less than n_modes."""
    s = AdaptiveImportanceSampler(n_modes=5)
    survivors = jax.random.uniform(jax.random.PRNGKey(0), (3, 6))
    with pytest.raises(ValueError):
        s.stage2(jax.random.PRNGKey(1), survivors, n=10)


def test_sample_respects_weights():
    """Inverse-CDF SIR draws particles in proportion to their weights."""
    logw = jnp.log(jnp.array([0.1, 0.3, 0.6]))
    parts = jnp.array([[0.0], [1.0], [2.0]])
    pp = SamplePosterior(
        samples=parts,
        log_weights=logw,
        evidence=jnp.asarray(jnp.nan),
        param_names=("x",),
    )
    x = pp.sample_dict(jax.random.PRNGKey(0), n=40000)["x"]
    freq = jnp.array([jnp.mean(x == v) for v in (0.0, 1.0, 2.0)])
    assert jnp.allclose(freq, jnp.array([0.1, 0.3, 0.6]), atol=0.02)


def test_sample_scales_to_large_particle_count():
    """sample() stays O(n_particles + n): a categorical draw here would be ~4 GB."""
    n_particles = 200000
    parts = jnp.arange(n_particles, dtype=float)[:, None]
    logw = jnp.zeros(n_particles)
    pp = SamplePosterior(
        samples=parts,
        log_weights=logw,
        evidence=jnp.asarray(jnp.nan),
        param_names=("x",),
    )
    draws = pp.sample_dict(jax.random.PRNGKey(0), n=5000)
    assert draws["x"].shape == (5000,)


def test_roberts_sequence_in_unit_cube_and_low_discrepancy():
    """roberts_sequence points lie in [0,1)^d with each axis mean near 0.5."""
    from photomancy.orbit.quasi_random import roberts_sequence

    pts = roberts_sequence(4096, 3)
    assert pts.shape == (4096, 3)
    assert (pts >= 0).all() and (pts < 1).all()
    assert jnp.abs(pts.mean(axis=0) - 0.5).max() < 0.01


def _floor(*terms):
    """Float64 accumulation floor: max(1e-10, 64 eps sum(1 + abs(term)))."""
    return max(
        1e-10, 64 * np.finfo(float).eps * sum(np.sum(1 + np.abs(t)) for t in terms)
    )


@pytest.mark.parametrize("e_max", [0.5, 0.9])
def test_chart_grid_target_in_common_coordinates(e_max):
    """Fails if the chart density, its Jacobian or the orbit map drifts from the box.

    At 96 shared box points (e_max=0.9 has e >= 1 in the square's corners): sky
    vectors and likelihood match the legacy shape at the 64 eps floor (rtol 1e-12
    on positions), the analytic cube-to-chart Jacobian equals the autodiff
    determinant to 1e-9, and chart log volume plus that Jacobian is the same
    constant at every point, i.e. the flat box is a flat density against the chart
    measure. The square is intersected with the unit disk exactly as the area
    formula says; legacy to_physical admits e >= 1 there.
    """
    data = _toy_astrom()
    star = MSUN
    bounds = chart_box_bounds((2.0, 3.0), e_max)
    chart = EccentricityVectorChart()
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(0), 6, 96)
    p = bounds.scale(u)
    x, bound = _box_to_chart(p, star)
    old = EccVectorShape().to_physical(u, bounds, Ms=star)
    assert bool(jnp.all(bound == (old["e"] < 1.0)))
    assert bool(jnp.all(old["e"][~bound] >= 1.0))
    assert bool(jnp.any(~bound)) == (e_max > 0.8)
    keep = np.asarray(bound)
    assert keep.sum() >= 60

    old_ll = jax.vmap(build_evaluator((data,), Ms=star, dist_pc=10.0, shape=None))(old)
    orbit = chart.to_orbit(x, mu_AU3_per_d2=G * star)
    ra_new, dec_new = jax.vmap(lambda o: sky_offsets_arcsec(o, data.times, 10.0))(orbit)
    ra_old, dec_old = jax.vmap(
        lambda *args: predict_relative_astrometry(data.times, *args, MSUN, 10.0)
    )(old["a"], old["e"], old["cos_i"], old["W"], old["cos_w"], old["sin_w"], old["tp"])
    np.testing.assert_allclose(ra_new[keep], ra_old[keep], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dec_new[keep], dec_old[keep], rtol=1e-12, atol=1e-14)
    new_ll = jax.vmap(lambda a, b: loglike_relative_astrom(a, b, data))(ra_new, dec_new)
    ll_error = np.abs(np.asarray(new_ll - old_ll))[keep]
    assert ll_error.max() <= _floor(old_ll[keep], new_ll[keep])

    jac = np.asarray(_box_log_jacobian(p, bounds))
    autodiff = jax.vmap(
        lambda point: (
            jnp.linalg.slogdet(jax.jacfwd(lambda q: _box_to_chart(q, star)[0])(point))[
                1
            ]
            + jnp.sum(jnp.log(bounds.high - bounds.low))
        )
    )(p)
    np.testing.assert_allclose(jac[keep], np.asarray(autodiff)[keep], rtol=0, atol=1e-9)
    total = np.asarray(orbit_chart_log_volume(chart, x) + jac)[keep]
    assert np.ptp(total) <= 64 * np.finfo(float).eps * np.sum(1 + np.abs(total))
    square = (2 * e_max) ** 2
    assert _ecc_square_disk_area(e_max) <= square
    assert (_ecc_square_disk_area(e_max) == square) == (e_max <= 2**-0.5)
    print(f"e_max={e_max}: ll_error={ll_error.max():.3g}, spread={np.ptp(total):.3g}")


def test_ecc_square_disk_area_matches_quadrature():
    """Fails if the support area (the flat prior's normalization) is wrong.

    Trapezoid quadrature of the square intersected with the unit disk, 4e5 nodes
    per half-width: agreement to 1e-6, spanning the three geometric regimes.
    """
    for half_width in (0.5, 0.8, 0.95, 1.3):
        x = np.linspace(0.0, min(half_width, 1.0), 400001)
        height = np.minimum(half_width, np.sqrt(np.maximum(1.0 - x**2, 0.0)))
        numeric = 4.0 * np.trapezoid(height, x)
        assert _ecc_square_disk_area(half_width) == pytest.approx(numeric, abs=1e-6)


def test_chart_grid_search_matches_grid_search():
    """Fails if chart_grid_search weights, rows or evidence differ from grid_search.

    With e_max=0.5 the square lies inside the unit disk, so the flat chart prior
    has the box's normalization and, with identical stage-1 and stage-2 draws, the
    two weights and log evidences agree (log weights to the 64 eps floor); the
    sampled orbits predict the same sky positions to rtol 1e-9. Draws clipped onto
    a face (cos i = +-1, under 2 percent) are weightless on the chart route.
    """
    data = _toy_astrom()
    common = dict(
        dist_pc=10.0,
        sampler=AdaptiveImportanceSampler(n_modes=3),
        log_T_range=(2.0, 3.0),
        e_max=0.5,
        n_particles=4000,
        chunk_size=2000,
        n_survivors=200,
        key=jax.random.PRNGKey(7),
    )
    old = grid_search((data,), Ms=MSUN, shape=EccVectorShape(), **common)
    new = chart_grid_search(
        (data,), star_mass_kg=MSUN, epoch=Epoch(2460000.0), **common
    )
    assert (new.chart, new.epoch) == (ORBIT_ROWS_CHART, Epoch(2460000.0))
    assert new.param_names == old.param_names
    assert np.isfinite(old.log_weights).all()
    ok = np.isfinite(new.log_weights)
    # Clipped draws on a cube face (cos i = +-1) sit on the chart's singular set:
    # the chart route gives them no weight, the legacy route an atom.
    assert 0.0 < (~ok).mean() < 0.02
    diff = np.asarray(new.log_weights - old.log_weights)[ok]
    floor = _floor(old.log_weights[:50], new.log_weights[:50])
    assert diff.max() - diff.min() <= floor
    assert float(new.evidence) == pytest.approx(float(old.evidence), abs=0.05)
    for post in (old, new):
        assert jnp.all(post.samples[:, 6] >= 0.0)
    sky = lambda post: jax.vmap(  # noqa: E731
        lambda *row: predict_relative_astrometry(data.times, *row, MSUN, 10.0)
    )(*(post.samples[:, i] for i in range(7)))
    for got, want in zip(sky(new), sky(old), strict=True):
        np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(new.samples[ok, :3], old.samples[ok, :3], rtol=1e-9)


def test_clipped_stage2_puts_atoms_on_the_cube_faces():
    """Fails if boundary atoms of the clipped proposal go undetected.

    Survivors hug a corner of the unit cube, so a quarter or more of the Gaussian
    draws leave it. Clipping piles those onto the faces (more than 20 percent of
    draws sit exactly on a face) while log_q stays the unclipped density; the
    sampling law is then not log_q. Unclipped draws carry no atoms and leave the
    cube with the probability log_q assigns outside (more than 20 percent here).
    """
    survivors = 0.97 + 0.01 * jax.random.uniform(jax.random.PRNGKey(0), (60, 3))
    n = 40000
    key = jax.random.PRNGKey(1)
    z_clip, _ = AdaptiveImportanceSampler(n_modes=3, jitter=0.02).stage2(
        key, survivors, n
    )
    z_free, _ = AdaptiveImportanceSampler(n_modes=3, jitter=0.02, clip=False).stage2(
        key, survivors, n
    )
    on_face = jnp.any((z_clip == 0.0) | (z_clip == 1.0 - 1e-7), axis=-1)
    assert float(on_face.mean()) > 0.2
    assert not bool(jnp.any((z_free == 0.0) | (z_free == 1.0 - 1e-7)))
    inside = jnp.all((z_free >= 0.0) & (z_free < 1.0), axis=-1)
    assert float(inside.mean()) < 0.8  # mass log_q puts outside the cube

    # The clipped sampler puts all its mass in the cube while its log_q says the
    # cube holds only `inside.mean()` of it: the density is not the sampling law.
    assert bool(jnp.all((z_clip >= 0.0) & (z_clip < 1.0)))
