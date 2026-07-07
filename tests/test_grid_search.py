"""Tests for orbix grid-search utilities."""

import jax
import jax.numpy as jnp
import pytest

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.grid_search import (
    AdaptiveImportanceSampler,
    EccVectorShape,
    ParamBounds,
    batched_loglike,
    build_evaluator,
)
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
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(1), 6, 200)
    phys = shape.to_physical(u, bounds, Ms=MSUN)
    fused = batched_loglike(ev, phys, n_particles=200, chunk_size=50)
    ref = jax.vmap(ev)({k: v for k, v in phys.items()})
    assert fused.shape == (200,)
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
