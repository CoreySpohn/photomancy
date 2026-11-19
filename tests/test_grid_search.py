"""Tests for orbix grid-search utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G
from hwoutils.epoch import Epoch
from kinematics import predict_relative_astrometry
from orbix.equations import period_to_sma
from orbix.system import EccentricityVectorChart

from photomancy.infer.legacy_posterior import SamplePosterior
from photomancy.measurements import RelativeAstromData, loglike_relative_astrom
from photomancy.orbit.grid_search import (
    AdaptiveImportanceSampler,
    ParamBounds,
    _box_log_jacobian,
    _box_to_chart,
    batched_loglike,
    chart_box_bounds,
    grid_search,
)
from photomancy.orbit.inference import ORBIT_ROWS_CHART, orbit_chart_log_volume
from photomancy.orbit.quasi_random import roberts_sequence

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


def test_ais_stage1_fills_unit_cube():
    """AdaptiveImportanceSampler.stage1 produces valid unit-cube points."""
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


def _chart_loglike(data, star):
    """Score one chart point: orbit, sky offsets, relative-astrometry loglike."""
    chart = EccentricityVectorChart()

    def single(phys):
        orbit = chart.to_orbit(phys["x"], mu_AU3_per_d2=G * star, epoch_d=0.0)
        ra, dec = orbit.offsets_arcsec(data.times, 10.0)
        return loglike_relative_astrom(ra, dec, data)

    return single


def test_batched_loglike_matches_unfused():
    """batched_loglike via scan+vmap matches direct vmap over all particles.

    Fails if the chunk reshape drops, reorders or repeats particles.
    """
    data = _toy_astrom()
    bounds = chart_box_bounds((2.0, 3.0), 0.5)
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(0), 6, 24)
    x, _ = _box_to_chart(bounds.scale(u), MSUN)
    single = _chart_loglike(data, MSUN)
    fused = batched_loglike(single, {"x": x}, n_particles=24, chunk_size=6)
    ref = jax.vmap(single)({"x": x})
    assert fused.shape == (24,)
    np.testing.assert_allclose(fused, ref, rtol=0, atol=1e-10)


def test_stage2_returns_samples_and_logq():
    """stage2 returns unit-cube samples and finite log-densities."""
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
    data = _toy_astrom()
    pp = grid_search(
        (data,),
        star_mass_kg=MSUN,
        dist_pc=10.0,
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
    data = _toy_astrom()
    common = dict(
        star_mass_kg=MSUN,
        dist_pc=10.0,
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
    """The grid-search names are in photomancy.orbit.__all__ and accessible."""
    import photomancy.orbit as f

    for name in (
        "grid_search",
        "AdaptiveImportanceSampler",
        "AbstractGridStrategy",
        "ParamBounds",
    ):
        assert name in f.__all__ and hasattr(f, name)


def test_batched_loglike_rejects_indivisible_chunk():
    """batched_loglike raises ValueError when n_particles is not divisible by chunk_size."""  # noqa: E501
    data = _toy_astrom()
    bounds = chart_box_bounds((2.0, 3.0), 0.5)
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(0), 6, 100)
    x, _ = _box_to_chart(bounds.scale(u), MSUN)
    with pytest.raises(ValueError):
        batched_loglike(
            _chart_loglike(data, MSUN), {"x": x}, n_particles=100, chunk_size=30
        )


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


# Legacy relative-astrometry log likelihoods at the first ten box points of
# stage1(key 0, 96 points) that have e < 1, frozen from the removed
# perifocal-formula route (toy data above); keyed by e_max. The e_max = 0.9
# square has e >= 1 corners, so its indices skip point 8.
_FROZEN_BOX_LL = {
    0.5: (
        list(range(10)),
        [
            -22761.800492548977,
            -29065.021728825628,
            -35479.042354214405,
            -37538.08375817465,
            -15854.986114954183,
            -18379.51542889183,
            -7829.6162074490885,
            -10119.361238806334,
            -20873.504913462068,
            -25784.315865950415,
        ],
    ),
    0.9: (
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        [
            -24869.64970300576,
            -25407.200129378412,
            -43821.42649766534,
            -44804.577303561025,
            -17201.959414324225,
            -21055.011165399676,
            -8280.003216430086,
            -9048.995536951403,
            -28590.382016309795,
            -45948.950857360986,
        ],
    ),
}


@pytest.mark.parametrize("e_max", [0.5, 0.9])
def test_chart_grid_target_in_common_coordinates(e_max):
    """Fails if the chart density, its Jacobian or the orbit map drifts from the box.

    At 96 shared box points (e_max=0.9 has e >= 1 in the square's corners): the
    sky vectors match the perifocal kinematics (tests/kinematics.py) at rtol
    1e-12; the likelihood matches frozen values from the removed legacy route at
    the 64 eps floor; the analytic cube-to-chart Jacobian equals the autodiff
    determinant to 1e-9, and chart log volume plus that Jacobian is the same
    constant at every point, i.e. the flat box is a flat density against the chart
    measure.
    """
    data = _toy_astrom()
    star = MSUN
    bounds = chart_box_bounds((2.0, 3.0), e_max)
    chart = EccentricityVectorChart()
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(0), 6, 96)
    p = bounds.scale(u)
    x, bound = _box_to_chart(p, star)
    assert bool(jnp.all(bound == (jnp.hypot(p[:, 1], p[:, 2]) < 1.0)))
    assert bool(jnp.any(~bound)) == (e_max > 0.8)
    keep = np.asarray(bound)
    assert keep.sum() >= 60

    orbit = chart.to_orbit(x, mu_AU3_per_d2=G * star)
    ra_new, dec_new = jax.vmap(lambda o: o.offsets_arcsec(data.times, 10.0))(orbit)
    elements = orbit.to_periapsis_elements()
    ra_old, dec_old = jax.vmap(
        lambda *r: predict_relative_astrometry(data.times, *r, MSUN, 10.0)
    )(*(elements[k] for k in ("a_AU", "e", "cos_i", "W_rad", "cos_w", "sin_w", "tp_d")))
    np.testing.assert_allclose(ra_new[keep], ra_old[keep], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dec_new[keep], dec_old[keep], rtol=1e-12, atol=1e-14)
    new_ll = jax.vmap(lambda a, b: loglike_relative_astrom(a, b, data))(ra_new, dec_new)
    idx, frozen = _FROZEN_BOX_LL[e_max]
    frozen = np.asarray(frozen)
    ll_error = np.abs(np.asarray(new_ll)[idx] - frozen)
    assert ll_error.max() <= _floor(frozen, np.asarray(new_ll)[idx])

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
    print(f"e_max={e_max}: ll_error={ll_error.max():.3g}, spread={np.ptp(total):.3g}")


def test_grid_search_weights_and_rows_in_common_coordinates():
    """Fails if grid_search weights, rows or evidence drift from the box target.

    The support is the disk e < e_max, so rows keep e < e_max with a inside the
    period box, only draws inside the cube and disk carry weight (the unclipped
    proposal puts about 78 percent on the cube and the toy peak is narrow, so
    more than 15 percent here), the evidence is finite, and the posterior
    declares its chart and epoch. The removed legacy route's evidence is not kept:
    its clipped proposal piled atoms on the cube faces, so it was not the
    sampling law.
    """
    data = _toy_astrom()
    post = grid_search(
        (data,),
        star_mass_kg=MSUN,
        epoch=Epoch(2460000.0),
        dist_pc=10.0,
        sampler=AdaptiveImportanceSampler(n_modes=3),
        log_T_range=(2.0, 3.0),
        e_max=0.5,
        n_particles=4000,
        chunk_size=2000,
        n_survivors=200,
        key=jax.random.PRNGKey(7),
    )
    assert (post.chart, post.epoch) == (ORBIT_ROWS_CHART, Epoch(2460000.0))
    assert post.param_names == ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")
    ok = np.isfinite(np.asarray(post.log_weights))
    # The unclipped proposal leaves the cube and the disk: 22 percent keep weight.
    assert ok.mean() > 0.15
    assert np.isfinite(float(post.evidence))
    assert bool(jnp.all(post.samples[:, 6] >= 0.0))
    rows = np.asarray(post.samples)[ok]
    assert rows[:, 1].max() < 0.5
    a_lo, a_hi = (float(period_to_sma(10.0**lt, MSUN)) for lt in (2.0, 3.0))
    assert a_lo <= rows[:, 0].min() and rows[:, 0].max() <= a_hi


def test_stage2_draws_follow_the_proposal_density():
    """Fails if stage2 clips draws onto the cube faces (atoms that log_q omits).

    Survivors hug a corner of the unit cube, so more than 20 percent of the
    Gaussian draws leave it (40000 draws, binomial error under 0.3 percent). No
    draw sits exactly on a face, so the sampling law is the density log_q.
    """
    survivors = 0.97 + 0.01 * jax.random.uniform(jax.random.PRNGKey(0), (60, 3))
    z, _ = AdaptiveImportanceSampler(n_modes=3, jitter=0.02).stage2(
        jax.random.PRNGKey(1), survivors, 40000
    )
    assert not bool(jnp.any((z == 0.0) | (z == 1.0 - 1e-7)))
    inside = jnp.all((z >= 0.0) & (z < 1.0), axis=-1)
    assert float(inside.mean()) < 0.8
