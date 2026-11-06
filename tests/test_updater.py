"""Fixed-capacity orbit updater: no-op contract, statuses and targets.

All tests share one updater (256 particles, 4 rungs, 1 move, 4 record slots)
and one compiled vmapped initializer and update, so the file compiles each
program once.
"""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hwoutils.constants import G, Msun2kg  # noqa: E402
from scipy import stats  # noqa: E402

from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.few_epoch import TIBasisModel  # noqa: E402
from photomancy.orbit.forward import predict_relative_astrometry  # noqa: E402
from photomancy.orbit.likelihoods import loglike_relative_astrom  # noqa: E402
from photomancy.orbit.updater import (  # noqa: E402
    ConjugateGaussianTarget,
    FixedMassTarget,
    OrbitUpdater,
    ThieleInnesTarget,
    UpdaterRecord,
    UpdaterStatus,
)

UPDATER = OrbitUpdater(n_particles=256, n_rungs=4, n_moves=1, record_capacity=4)
TARGET = ConjugateGaussianTarget()
SIGMA = 0.5
LANES = 32


@eqx.filter_jit
def vinit(target, records, keys):
    """Initialize a lane batch from one shared target."""
    return jax.vmap(UPDATER.init, in_axes=(None, 0, 0))(target, records, keys)


@eqx.filter_jit
def vupdate(beliefs, records, keys):
    """Update a lane batch."""
    return jax.vmap(UPDATER.update)(beliefs, records, keys)


def _keys(seed, n=LANES):
    return jax.random.split(jax.random.key(seed, impl="threefry2x32"), n)


def _records(ra, dec, record_id, is_valid=True, n=LANES):
    """A lane batch of records (scalars broadcast over lanes)."""
    full = lambda x, dtype=float: jnp.broadcast_to(jnp.asarray(x, dtype), (n,))  # noqa: E731
    return UpdaterRecord(
        time=full(0.0),
        ra=full(ra),
        dec=full(dec),
        ra_err=full(SIGMA),
        dec_err=full(SIGMA),
        corr=full(0.0),
        record_id=full(record_id, int),
        is_valid=full(is_valid, bool),
    )


def _lane(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


def _leaves(tree):
    """Leaf arrays as NumPy, keys as their raw data, for bitwise comparison."""
    out = []
    for leaf in jax.tree.leaves(tree):
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        out.append(np.asarray(leaf))
    return out


def _assert_bitwise_equal(a, b):
    la, lb = _leaves(a), _leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(
            np.atleast_1d(x).view(np.uint8), np.atleast_1d(y).view(np.uint8)
        )


@pytest.fixture(scope="module")
def fitted():
    """Lane batch after the initializer and one more record (ids 1 and 2)."""
    first = _records(0.5, -0.3, 1)
    belief, diag = vinit(TARGET, first, _keys(0))
    second = _records(0.2, 0.4, 2)
    belief, diag2 = vupdate(belief, second, _keys(1))
    return belief, diag, diag2


@pytest.fixture(scope="module")
def empty():
    """Lane batch of prior beliefs (no records)."""
    keys = _keys(2)
    return jax.vmap(UPDATER.prior_belief, in_axes=(None, 0))(TARGET, keys)


def test_conjugate_gaussian_posterior(fitted):
    """Two sequential records reproduce the closed-form posterior.

    Fails if the reweighting uses the wrong likelihood scale, double counts a
    record, or the initializer ignores its record. Each lane is an independent
    run, so the tolerance is 4 standard errors of the lane average plus the
    self-normalized importance bias (of order variance / ESS, about 0.01 here).
    """
    belief, diag, diag2 = fitted
    assert bool(jnp.all(diag.applied & diag2.applied))
    w = jnp.exp(belief.log_weights)[..., None]
    lane_mean = jnp.sum(w * belief.particles, axis=1)
    lane_var = jnp.sum(w * (belief.particles - lane_mean[:, None]) ** 2, axis=1)

    # Prior N(0, 1), two records at sigma 0.5: precision 1 + 2 * 4 = 9.
    exact_mean = 4.0 * (np.array([0.5, -0.3]) + np.array([0.2, 0.4])) / 9.0
    exact_var = np.full(2, 1.0 / 9.0)
    mean_c, var_c = TARGET.conjugate_posterior(_lane(belief.records, 0))
    np.testing.assert_allclose(mean_c, exact_mean, rtol=1e-12)
    np.testing.assert_allclose(var_c, exact_var, rtol=1e-12)

    for est, exact in ((lane_mean, exact_mean), (lane_var, exact_var)):
        se = np.std(est, axis=0, ddof=1) / np.sqrt(LANES)
        assert np.all(se < 0.05)
        assert np.all(np.abs(np.mean(est, axis=0) - exact) < 4.0 * se + 0.01)


@pytest.mark.parametrize("carry", ["fitted", "empty"])
def test_padded_record_is_bitwise_noop(carry, fitted, empty):
    """A padded record returns every leaf unchanged, key and count included.

    Fails if padding advances the key, renormalizes or resamples the weights,
    writes the buffer or overwrites the stored diagnostics.
    """
    belief = fitted[0] if carry == "fitted" else empty
    out, diag = vupdate(belief, _records(9.0, 9.0, 99, is_valid=False), _keys(3))
    _assert_bitwise_equal(out, belief)
    assert not bool(jnp.any(diag.applied))
    assert bool(jnp.all(diag.status == UpdaterStatus.OK))


def test_mixed_lanes_are_independent(fitted):
    """Under vmap, padded lanes stay unchanged and real lanes ignore them.

    Fails if a lane's result depends on another lane's validity (a reduction
    across the batch axis) or if a cond on a batched predicate leaks the
    active branch into padded lanes.
    """
    belief = fitted[0]
    valid = jnp.arange(LANES) % 2 == 0
    mixed = _records(0.1, 0.1, 3)
    mixed = eqx.tree_at(lambda r: r.is_valid, mixed, valid)
    out_mixed, diag_mixed = vupdate(belief, mixed, _keys(4))
    out_all, _ = vupdate(belief, _records(0.1, 0.1, 3), _keys(4))
    np.testing.assert_array_equal(diag_mixed.applied, valid)
    for i in range(4):
        expect = out_all if bool(valid[i]) else belief
        _assert_bitwise_equal(_lane(out_mixed, i), _lane(expect, i))


def test_same_keys_same_belief(fitted):
    """Same inputs and keys give the same belief; a new key changes it.

    Fails if the updater draws from ambient state (a global key, NumPy) or
    ignores its key argument.
    """
    belief, _, _ = fitted
    rec = _records(-0.2, 0.3, 4)
    a, _ = vupdate(belief, rec, _keys(5))
    b, _ = vupdate(belief, rec, _keys(5))
    c, _ = vupdate(belief, rec, _keys(6))
    _assert_bitwise_equal(a, b)
    assert not np.array_equal(np.asarray(a.particles), np.asarray(c.particles))


def test_duplicate_record_adds_no_likelihood(fitted):
    """Re-sending an accepted record id is a no-op; a new id is accepted.

    Fails if a replayed record is assimilated twice (a posterior narrowed by a
    likelihood counted twice) or if identity is taken from values, not ids.
    """
    belief, _, _ = fitted
    replay = _records(0.2, 0.4, 2)
    out, diag = vupdate(belief, replay, _keys(7))
    _assert_bitwise_equal(out, belief)
    assert bool(jnp.all(diag.duplicate & ~diag.applied))
    fresh, diag_fresh = vupdate(belief, _records(0.2, 0.4, 5), _keys(7))
    assert bool(jnp.all(diag_fresh.applied))
    np.testing.assert_array_equal(fresh.n_records, belief.n_records + 1)


def test_no_recompile_across_values_and_counts(fitted, empty):
    """New record values, record counts and target values reuse one trace.

    Fails if any of these enters the compiled program as a static value (a
    Python float on a target, a count used for a shape or a Python branch).
    """
    traces = []

    @eqx.filter_jit
    def step(belief, record, key):
        traces.append(None)
        return jax.vmap(UPDATER.update)(belief, record, key)

    shifted = eqx.tree_at(
        lambda b: b.target,
        empty,
        jax.vmap(lambda m: ConjugateGaussianTarget(prior_mean=m, prior_scale=2.0))(
            jnp.ones((LANES, 2))
        ),
    )
    step(fitted[0], _records(0.1, 0.2, 6), _keys(8))
    step(empty, _records(-1.0, 0.7, 7), _keys(9))
    step(shifted, _records(0.3, 0.3, 8, is_valid=False), _keys(10))
    assert len(traces) == 1


@pytest.mark.parametrize(
    ("ra", "expected"),
    [
        (0.3, UpdaterStatus.OK),
        (5.0, UpdaterStatus.INCOMPLETE_LADDER),
        (14.0, UpdaterStatus.LOW_ESS),
        (np.inf, UpdaterStatus.NO_SUPPORT),
    ],
)
def test_status(ra, expected, empty):
    """Statuses follow the ladder, the reweighted ESS and support; nothing raises.

    With 4 rungs, a record at 10 prior sigma leaves the ladder near
    temperature 0.5 and its final correction keeps a conditional ESS of 84 to
    128 of 256 over the lanes; at 28 prior sigma that correction collapses to
    at most 7 particles. Fails if an unsupported record raises, changes the belief or
    reports OK, if an unfinished ladder reports OK, or if a degenerate final
    correction is not flagged LOW_ESS.
    """
    out, diag = vupdate(empty, _records(ra, 0.0, 1), _keys(11))
    assert bool(jnp.all(diag.status == expected))
    if expected == UpdaterStatus.NO_SUPPORT:
        _assert_bitwise_equal(out, empty)
    else:
        assert bool(jnp.all(diag.applied))
        finished = expected == UpdaterStatus.OK
        assert bool(jnp.all((diag.temperature == 1.0) == finished))


def test_full_buffer_is_budget_exhausted(fitted):
    """A valid record beyond the buffer capacity is refused, belief unchanged.

    Fails if a full buffer silently overwrites or drops a record (the index
    clamps at the last slot under jit) instead of reporting the exhaustion.
    """
    belief = fitted[0]
    for rid in (10, 11):
        belief, _ = vupdate(belief, _records(0.0, 0.0, rid), _keys(rid))
    np.testing.assert_array_equal(belief.n_records, 4)
    out, diag = vupdate(belief, _records(0.0, 0.0, 12), _keys(12))
    assert bool(jnp.all(diag.status == UpdaterStatus.BUDGET_EXHAUSTED))
    _assert_bitwise_equal(out, belief)


def _buffer(times, ra, dec, sigma):
    n = len(times)
    return RelativeAstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(ra),
        dec=jnp.asarray(dec),
        ra_err=jnp.full(n, sigma),
        dec_err=jnp.full(n, sigma),
        corr=jnp.zeros(n),
        planet_id=jnp.zeros(n, dtype=int),
        is_valid=jnp.ones(n, dtype=bool),
    )


def test_fixed_mass_chart_matches_element_forward_model():
    """The physical chart predicts like the element forward model.

    Fails if the chart conversion is wrong (an M_ref sign, a node or periastron
    swap) or the distance or mass is dropped.
    """
    target = FixedMassTarget(t_ref_day=20.0)
    times = jnp.array([0.0, 60.0, 400.0])
    data = _buffer(times, [0.05, -0.02, 0.1], [0.08, 0.01, -0.04], 0.0035)
    for el in ELEMENTS:
        a, e, cos_i, W, omega, m_ref = el
        tp = 20.0 - m_ref / jnp.sqrt(G * target.Ms_kg / a**3)
        ra, dec = predict_relative_astrometry(
            times,
            a,
            e,
            cos_i,
            W,
            jnp.cos(omega),
            jnp.sin(omega),
            tp,
            target.Ms_kg,
            10.0,
        )
        theta = target.from_physical(el)
        np.testing.assert_allclose(target.to_physical(theta), el, rtol=1e-12)
        np.testing.assert_allclose(
            target.log_likelihood(theta, data),
            loglike_relative_astrom(ra, dec, data),
            rtol=1e-10,
        )


ELEMENTS = jnp.array(
    [
        [1.3, 0.21, 0.4, 2.0, 0.7, 4.1],
        [0.6, 0.25, -0.8, 5.5, 3.9, 0.2],
        [4.0, 0.02, 0.1, 0.3, 6.0, 2.5],
    ]
)


@pytest.mark.parametrize(
    ("e_max", "mass_msun", "period_day"),
    [(0.9, 1.0, (50.0, 5000.0)), (0.3, 0.6, (100.0, 800.0)), (0.5, 2.0, (20.0, 200.0))],
)
def test_fixed_mass_prior_is_normalized_density(e_max, mass_msun, period_day):
    """The chart prior is OFTI's normalized element density times the Jacobian.

    The expectation is built independently: SciPy's Beta density over its CDF
    at e_max, uniform ln a between Kepler semimajor axes, 1/2 for cos_i and
    1/(2 pi) for each of W, omega and M_ref, times the autodiff Jacobian of the
    chart-to-elements map. Fails if the chart Jacobian constant (log 2), an
    angle or inclination constant or the truncation normalizer is dropped, if
    the mass or period bounds are ignored, or if a non-finite angle is
    accepted as on support.
    """
    ms_kg = mass_msun * Msun2kg
    target = FixedMassTarget(
        Ms_kg=ms_kg,
        period_lo_day=period_day[0],
        period_hi_day=period_day[1],
        e_max=e_max,
    )
    a_lo, a_hi = (
        (G * ms_kg * (p / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0) for p in period_day
    )

    def chart_to_elements(theta):
        _, h, k, cos_i, W, lam = theta
        omega = jnp.arctan2(k, h)
        return jnp.array([jnp.exp(theta[0]), h**2 + k**2, cos_i, W, omega, lam - omega])

    for row, frac in zip(ELEMENTS, (0.2, 0.5, 0.9), strict=True):
        a = a_lo * (a_hi / a_lo) ** frac
        el = row.at[0].set(a).at[1].set(row[1] * e_max / 0.9)
        e = float(el[1])
        log_elements = (
            -np.log(a * np.log(a_hi / a_lo))
            + np.log(
                stats.beta.pdf(e, 0.867, 3.03) / stats.beta.cdf(e_max, 0.867, 3.03)
            )
            - np.log(2.0)
            - 3.0 * np.log(2.0 * np.pi)
        )
        theta = target.from_physical(el)
        _, logdet = jnp.linalg.slogdet(jax.jacfwd(chart_to_elements)(theta))
        np.testing.assert_allclose(
            target.log_prior(theta), log_elements + logdet, rtol=0, atol=1e-10
        )
        for i in (4, 5):  # W and lambda_ref
            assert target.log_prior(theta.at[i].set(jnp.nan)) == -jnp.inf


def test_wrap_maps_periodic_coordinates_only():
    """Periodic coordinates wrap into [0, period); predictions are invariant.

    Fails if a period is wrong (the physical angles at pi instead of 2 pi, TI
    tau at anything but 1), if a non-periodic coordinate is wrapped, or if
    negative or multi-period inputs are mishandled.
    """
    two_pi = 2.0 * np.pi
    times = jnp.array([0.0, 75.0, 300.0])

    ti = ThieleInnesTarget()
    theta = jnp.array([0.3, 6.0, 0.25, -3.2, 1.0, -0.4, 0.8])
    for shift in (-1.0, 3.0):
        out = ti.wrap(theta.at[2].add(shift))
        np.testing.assert_allclose(out, theta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            ti.predict(theta.at[2].add(shift), times),
            ti.predict(theta, times),
            rtol=0,
            atol=1e-12,
        )

    for tau, expect in ((-0.25, 0.75), (2.6, 0.6)):
        np.testing.assert_allclose(
            ti.wrap(theta.at[2].set(tau)), theta.at[2].set(expect), rtol=0, atol=1e-12
        )

    phys = FixedMassTarget()
    theta = phys.from_physical(jnp.array([1.3, 0.21, -0.4, 2.0, 0.7, 4.1]))
    theta = theta.at[1].set(-0.3)  # negative non-periodic coordinates stay put
    for shift in (-two_pi, 3.0 * two_pi):
        moved = theta.at[4].add(shift).at[5].add(-shift)
        np.testing.assert_allclose(phys.wrap(moved), theta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            phys.predict(phys.wrap(moved), times),
            phys.predict(theta, times),
            rtol=0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        phys.wrap(theta.at[4].set(-1.0).at[5].set(-7.0)),
        theta.at[4].set(two_pi - 1.0).at[5].set(2.0 * two_pi - 7.0),
        rtol=0,
        atol=1e-12,
    )


def test_large_common_likelihood_offset_keeps_ess():
    """Identical particles under a huge finite log likelihood keep full ESS.

    Fails if weights are normalized by subtracting a log-sum-exp at the scale
    of the offset (-5e17 has a float spacing of 64, above log 16), which
    reports ESS 1/16 and LOW_ESS for an exactly uniform posterior.
    """
    updater = OrbitUpdater(n_particles=16, n_rungs=1, n_moves=1, record_capacity=2)
    key = jax.random.key(0, impl="threefry2x32")
    belief = updater.prior_belief(TARGET, key)
    belief = eqx.tree_at(lambda b: b.particles, belief, jnp.zeros((16, 2)))
    record = UpdaterRecord(
        time=0.0, ra=1e8, dec=0.0, ra_err=0.1, dec_err=0.1, record_id=1
    )
    out, diag = updater.update(belief, record, key)
    assert int(diag.status) == UpdaterStatus.OK
    np.testing.assert_allclose(diag.ess, 16.0, rtol=1e-12)
    np.testing.assert_allclose(jnp.exp(out.log_weights), 1.0 / 16.0, rtol=1e-12)


def test_unsupported_prior_reports_zero_ess():
    """A prior with no supported draw reports ESS 0, never infinity.

    Fails if the Kish ESS divides by a zero total weight, at initialization,
    in a later no-support update or in a padded no-op.
    """
    updater = OrbitUpdater(n_particles=16, n_rungs=1, n_moves=1, record_capacity=2)
    key = jax.random.key(0, impl="threefry2x32")
    target = ThieleInnesTarget(TIBasisModel(mu_lo=1.0, mu_hi=0.5))
    belief = updater.prior_belief(target, key)
    assert int(belief.diagnostics.status) == UpdaterStatus.NO_SUPPORT
    assert float(belief.diagnostics.ess) == 0.0
    record = UpdaterRecord(
        time=0.0, ra=0.1, dec=0.0, ra_err=0.0035, dec_err=0.0035, record_id=1
    )
    for valid, status in ((True, UpdaterStatus.NO_SUPPORT), (False, UpdaterStatus.OK)):
        out, diag = updater.update(
            belief, eqx.tree_at(lambda r: r.is_valid, record, jnp.asarray(valid)), key
        )
        assert int(diag.status) == status
        assert float(diag.ess) == 0.0
        _assert_bitwise_equal(out, belief)


@pytest.mark.parametrize("target", ["ti", "physical"])
def test_orbit_targets_initialize_with_status(target):
    """The TI and physical initializers run under jit and return a status.

    One case per target, so each compiles only its own initializer. Fails if
    an orbit target breaks the fixed-shape contract (a shape that depends on
    values) or raises on a real one-epoch record, and if the TI target's
    likelihood drops the AU-to-arcsec distance division.
    """
    record = UpdaterRecord(
        time=0.0, ra=0.12, dec=-0.05, ra_err=0.0035, dec_err=0.0035, record_id=1
    )
    key = jax.random.key(0, impl="threefry2x32")
    data = record.as_data()
    target = ThieleInnesTarget() if target == "ti" else FixedMassTarget()
    belief, diag = UPDATER.init(target, record, key)
    assert int(diag.status) in {s.value for s in UpdaterStatus}
    assert bool(diag.applied)
    assert np.isclose(float(jnp.sum(jnp.exp(belief.log_weights))), 1.0)
    assert bool(jnp.all(jnp.isfinite(jax.vmap(target.log_prior)(belief.particles))))
    if not isinstance(target, ThieleInnesTarget):
        return

    ti = ThieleInnesTarget(dist_pc=20.0)
    theta = jnp.array([0.3, 6.0, 0.2, 1.0, -0.5, 0.4, 1.2])
    east_au, north_au = ti.model.forward_au(theta, data.times)
    np.testing.assert_allclose(
        ti.log_likelihood(theta, data),
        loglike_relative_astrom(east_au / 20.0, north_au / 20.0, data),
        rtol=1e-12,
    )
