"""Tempered SMC kernel of the orbit updater: moves, ladder, posteriors, targets.

The move kernel is checked on its own against densities with exact draws, the
full kernel against the closed-form conjugate Gaussian posterior and evidence,
and the orbit targets through their one-epoch posterior predictive and a
two-epoch comparison with the Lambert importance reference.
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hwoutils.constants import mas2arcsec  # noqa: E402
from scipy import special  # noqa: E402

from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.few_epoch import TIBasisModel, lambert_depth_fit  # noqa: E402
from photomancy.orbit.updater import (  # noqa: E402
    AbstractUpdateTarget,
    ConjugateGaussianTarget,
    FixedMassTarget,
    OrbitUpdater,
    ThieleInnesTarget,
    UpdaterRecord,
    UpdaterStatus,
    _collapsed_step,
    _linear_gaussian,
    _metropolis_step,
    _rest_log_prior,
)

LANES = 32
SIGMA = 0.5


def _key(seed):
    return jax.random.key(seed, impl="threefry2x32")


def _records(ra, dec, record_id, n=LANES):
    """A lane batch of toy records (scalars broadcast over lanes)."""
    full = lambda x, dtype=float: jnp.broadcast_to(jnp.asarray(x, dtype), (n,))  # noqa: E731
    return UpdaterRecord(
        time=full(0.0),
        ra=full(ra),
        dec=full(dec),
        ra_err=full(SIGMA),
        dec_err=full(SIGMA),
        corr=full(0.0),
        record_id=full(record_id, int),
        is_valid=full(True, bool),
    )


def _sweep(log_density, wrap, x, step_chol, steps, seed):
    """Apply ``steps`` Metropolis moves to every row of ``x``."""

    @jax.jit
    def run(x):
        def body(i, carry):
            x, lp = carry
            x, lp, _ = _metropolis_step(
                jax.random.fold_in(_key(seed), i),
                x,
                lp,
                step_chol,
                wrap,
                log_density,
                lambda v: v,
            )
            return x, lp

        return jax.lax.fori_loop(0, steps, body, (x, jax.vmap(log_density)(x)))[0]

    return np.asarray(run(jnp.asarray(x)))


def test_metropolis_step_leaves_known_densities_invariant():
    """Moves keep exact draws exact, across a wrap point and in a log chart.

    Each row starts as an exact independent draw and moves independently, so
    after any number of invariant moves the rows are still exact independent
    draws and every moment has its i.i.d. standard error. N = 4000 resolves a
    mean shift of 4 sigma / sqrt(N) = 0.063 sigma. Targets:

    - a correlated Gaussian pair with a von Mises angle (kappa 4, mean 0.1)
      that straddles the wrap at 0 and 2 pi;
    - x ~ Exponential(1) moved in the chart u = ln x, whose density
      exp(u - e^u) carries the chart Jacobian e^u.

    Fails if the acceptance ratio is inverted, drops the min(1, .), adds a
    proposal correction a symmetric wrapped step does not have, mishandles the
    wrap, or if a chart density without its Jacobian (the mutant below, the
    improper exp(-e^u)) is not visibly biased.
    """
    rng = np.random.default_rng(0)
    n, steps = 4000, 20

    mean = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    kappa, mu = 4.0, 0.1
    x = np.column_stack(
        [
            rng.multivariate_normal(mean, cov, n),
            np.mod(rng.vonmises(mu, kappa, n), 2.0 * np.pi),
        ]
    )
    prec = np.linalg.inv(cov)

    def log_gvm(v):
        d = v[:2] - mean
        return -0.5 * d @ prec @ d + kappa * jnp.cos(v[2] - mu)

    def wrap(v):
        return v.at[2].set(jnp.mod(v[2], 2.0 * jnp.pi))

    scales = np.zeros((3, 3))
    scales[:2, :2] = cov
    scales[2, 2] = 1.0 / kappa
    step = 1.5 * np.linalg.cholesky(scales)
    out = _sweep(log_gvm, wrap, x, step, steps, seed=1)
    assert np.all((out[:, 2] >= 0.0) & (out[:, 2] < 2.0 * np.pi))
    assert np.mean(np.any(out != x, axis=1)) > 0.9  # the rows did move
    se = np.sqrt(np.diag(cov) / n)
    np.testing.assert_array_less(np.abs(out[:, :2].mean(0) - mean), 4.0 * se)
    var_se = np.diag(cov) * np.sqrt(2.0 / n)
    np.testing.assert_array_less(np.abs(out[:, :2].var(0) - np.diag(cov)), 4.0 * var_se)
    resultant = special.i1(kappa) / special.i0(kappa)
    c, s = np.cos(out[:, 2] - mu), np.sin(out[:, 2] - mu)
    assert abs(c.mean() - resultant) < 4.0 * c.std() / math.sqrt(n)
    assert abs(s.mean()) < 4.0 * s.std() / math.sqrt(n)

    u = np.log(rng.exponential(1.0, (n, 1)))
    step_u = np.array([[1.3]])
    identity = lambda v: v  # noqa: E731
    with_jacobian = _sweep(
        lambda v: v[0] - jnp.exp(v[0]), identity, u, step_u, steps, 2
    )
    without = _sweep(lambda v: -jnp.exp(v[0]), identity, u, step_u, steps, 2)
    se_x = 1.0 / math.sqrt(n)  # Exponential(1) has unit standard deviation
    assert abs(np.exp(with_jacobian).mean() - 1.0) < 4.0 * se_x
    assert abs(np.exp(without).mean() - 1.0) > 8.0 * se_x


@pytest.mark.medium
def test_collapsed_move_leaves_tempered_ti_posterior_invariant():
    """Collapsed moves keep exact TI posterior draws exact; window and weight matter.

    Target: the default Thiele-Innes prior (with its implied-mass window), a
    buffered record at full weight and a new record (correlation 0.3) at
    temperature 0.3, errors 0.05 arcsec. Exact draws come from 2^22 prior
    draws importance-weighted by the tempered likelihood (Kish ESS about
    42,000) and resampled to N = 4000, so a moment's standard error is close
    to the i.i.d. one with the reference's own error added: sd sqrt(1/N +
    1/ESS) for a mean, sqrt((m4 - s^4)(1/N + 1/ESS)) for a variance. After
    eight moves every coordinate's mean and variance must lie within 4 such
    errors (the correct move reaches 1.5) and every draw inside the support.
    Negative controls: a move that drops the window indicator from its
    acceptance ratio leaves the window; a move that gives the new record full
    weight (an error in the tempering weights of the conditional Gaussian and
    of log Z) shifts some moment by more than 4 errors (it reaches 7.3).
    Fails if the conditional Gaussian or log Z uses the wrong precision (a
    tempering weight, the correlation or the prior scale), if the random walk
    is not symmetric, if proposals outside the window are accepted, or if the
    TI target's one-solve design matrix drifts from its forward model.
    """
    target = ThieleInnesTarget()
    sigma, beta, n = 0.05, 0.3, 4000
    # The TI design override must equal the generic one built from predict.
    for theta in (ORBIT_CASES["ti"][1], jnp.array([0.7, 7.5, 0.9, 0, 0, 0, 0.0])):
        times = jnp.array([0.0, 200.0, 1234.5])
        for fast, generic in zip(
            target.linear_design(theta, times),
            AbstractUpdateTarget.linear_design(target, theta, times),
            strict=True,
        ):
            np.testing.assert_allclose(fast, generic, rtol=0, atol=1e-14)
    records = RelativeAstromData(
        times=jnp.array([0.0, 200.0]),
        ra=jnp.array([0.08, -0.05]),
        dec=jnp.array([0.06, 0.1]),
        ra_err=jnp.full(2, sigma),
        dec_err=jnp.full(2, sigma),
        corr=jnp.array([0.0, 0.3]),
        planet_id=jnp.zeros(2, dtype=int),
        is_valid=jnp.ones(2, dtype=bool),
    )
    select = jnp.array([False, True])

    def log_weight(theta):
        ll_rest, ll_new = target.log_likelihood_split(theta, records, select)
        inside = jnp.isfinite(target.log_prior(theta))
        return jnp.where(inside, ll_rest + beta * ll_new, -jnp.inf)

    @jax.jit
    def reference(key):
        x = target.sample_prior(key, 2**22)
        chunks = x.reshape(64, -1, x.shape[1])
        log_w = jax.lax.map(jax.vmap(log_weight), chunks).reshape(-1)
        return x, jax.nn.softmax(log_w)

    pool, w = (np.asarray(a) for a in reference(jax.random.key(0)))
    ess = 1.0 / np.sum(w**2)
    assert ess > 30_000
    ref_sd = np.sqrt(w @ (pool - w @ pool) ** 2)
    rng = np.random.default_rng(0)
    start = jnp.asarray(pool[rng.choice(len(w), n, p=w / w.sum())])

    def stats(x):
        tau = 2.0 * np.pi * x[:, 2]
        return np.column_stack([x[:, [0, 1, 3, 4, 5, 6]], np.cos(tau), np.sin(tau)])

    ref_stats = stats(pool)
    ref_m = w @ ref_stats
    ref_var = w @ (ref_stats - ref_m) ** 2
    m4 = w @ (ref_stats - ref_m) ** 4
    factor = 1.0 / n + 1.0 / ess

    def max_z(x):
        out = stats(x)
        z_mean = (out.mean(0) - ref_m) / np.sqrt(ref_var * factor)
        z_var = (out.var(0) - ref_var) / np.sqrt((m4 - ref_var**2) * factor)
        return max(np.max(np.abs(z_mean)), np.max(np.abs(z_var)))

    e_max, lnp_lo, lnp_hi = 0.9, math.log(50.0), math.log(5000.0)

    def no_window(theta):
        inside = (theta[0] > 0.0) & (theta[0] < e_max)
        inside &= (theta[1] > lnp_lo) & (theta[1] < lnp_hi)
        return jnp.where(inside, 0.0, -jnp.inf)

    def run(rest, new_weight):
        weights = jnp.array([1.0, new_weight])

        @jax.jit
        def go(x):
            def evaluate(theta):
                lr, ln = target.log_likelihood_split(theta, records, select)
                return target.log_prior(theta), lr, ln

            aux = jax.vmap(evaluate)(x)
            log_z = jax.vmap(
                lambda t: _linear_gaussian(target, t, records, weights)[2]
            )(x)
            step = 0.8 * jnp.diag(jnp.asarray(ref_sd[:3])) / math.sqrt(3.0)

            def body(i, carry):
                x, aux, log_z = carry
                x, aux, log_z, _ = _collapsed_step(
                    jax.random.fold_in(_key(4), i),
                    x,
                    aux,
                    log_z,
                    step,
                    target,
                    records,
                    weights,
                    select,
                    rest,
                )
                return x, aux, log_z

            return jax.lax.fori_loop(0, 8, body, (x, aux, log_z))[0]

        return np.asarray(go(start))

    def exact(theta):
        return _rest_log_prior(target, theta)

    out = run(exact, beta)
    assert np.mean(np.any(out != np.asarray(start), axis=1)) > 0.5  # the draws moved
    assert np.all(np.isfinite(np.asarray(jax.vmap(target.log_prior)(jnp.asarray(out)))))
    assert max_z(out) < 4.0

    mutant = run(no_window, beta)
    in_window = jax.vmap(target.log_prior)(jnp.asarray(mutant))
    assert float(jnp.mean(jnp.isfinite(in_window))) < 0.9
    assert max_z(run(exact, 1.0)) > 4.0


class _PriorMovesUpdater(OrbitUpdater):
    """Mutant whose moves target the prior instead of the tempered posterior."""

    def _move_log_density(self, aux, beta):
        return aux[0]


def _toy_run(updater):
    """Two sequential toy records on every lane: moments, evidence, diagnostics."""

    @eqx.filter_jit
    def run(keys):
        target = ConjugateGaussianTarget()
        belief, d1 = jax.vmap(updater.init, in_axes=(None, 0, 0))(
            target, _records(0.5, -0.3, 1), keys
        )
        belief, d2 = jax.vmap(updater.update)(
            belief,
            _records(0.2, 0.4, 2),
            jax.vmap(jax.random.fold_in, (0, None))(keys, 9),
        )
        return belief, d1, d2

    belief, d1, d2 = run(jax.random.split(_key(0), LANES))
    w = np.exp(np.asarray(belief.log_weights))[..., None]
    x = np.asarray(belief.particles)
    lane_mean = np.sum(w * x, axis=1)
    lane_var = np.sum(w * (x - lane_mean[:, None]) ** 2, axis=1)
    return lane_mean, lane_var, d1, d2


def _gauss_log_marginal(y, mean, var):
    """Log density of a 2-vector y under N(mean, diag(var))."""
    return float(np.sum(-0.5 * (y - mean) ** 2 / var - 0.5 * np.log(2.0 * np.pi * var)))


def test_conjugate_posterior_evidence_and_wrong_target_mutant():
    """Two tempered updates reproduce the conjugate posterior and evidence.

    Prior N(0, I), records (0.5, -0.3) and (0.2, 0.4) at sigma 0.5: posterior
    precision 1 + 2 x 4 = 9 per axis. Each of 32 lanes is an independent run
    at 512 particles, so lane averages are compared within 4 standard errors
    plus the self-normalized bias (of order variance / ESS, about 0.002 for the
    moments; the log evidence of a ratio estimator is biased low by about
    half its variance). The mutant moves particles under the prior; resampling
    and reweighting still run, so only the move target is wrong. Fails if the
    moves do not leave the tempered posterior invariant (the mutant's variance
    excess must stand out by more than 8 standard errors), if a rung applies
    the wrong power of the likelihood, or if the evidence increments are not
    accumulated over rungs.
    """
    updater = OrbitUpdater(
        n_particles=512, n_rungs=16, n_moves=4, record_capacity=4, proposal_scale=2.38
    )
    lane_mean, lane_var, d1, d2 = _toy_run(updater)
    for d in (d1, d2):
        assert bool(jnp.all(d.status == UpdaterStatus.OK))
        assert bool(jnp.all(d.temperature == 1.0))
        assert bool(jnp.all((d.rungs_used >= 1) & (d.rungs_used < 16)))
        assert float(jnp.min(d.acceptance_rate)) > 0.1

    y1, y2 = np.array([0.5, -0.3]), np.array([0.2, 0.4])
    exact_mean = 4.0 * (y1 + y2) / 9.0
    exact_var = np.full(2, 1.0 / 9.0)
    for est, exact in ((lane_mean, exact_mean), (lane_var, exact_var)):
        se = np.std(est, axis=0, ddof=1) / math.sqrt(LANES)
        np.testing.assert_array_less(np.abs(est.mean(0) - exact), 4.0 * se + 0.002)

    # log p(y1) under the prior predictive, then log p(y2 | y1).
    log_z1 = _gauss_log_marginal(y1, 0.0, 1.0 + SIGMA**2)
    log_z2 = _gauss_log_marginal(y2, 0.8 * y1, 0.2 + SIGMA**2)
    for d, exact in ((d1, log_z1), (d2, log_z2)):
        z = np.asarray(d.log_evidence_increment)
        se = np.std(z, ddof=1) / math.sqrt(LANES)
        assert abs(z.mean() - exact) < 4.0 * se + 0.5 * np.var(z) + 0.005

    mutant = _PriorMovesUpdater(
        n_particles=512, n_rungs=16, n_moves=4, record_capacity=4, proposal_scale=2.38
    )
    _, mutant_var, _, _ = _toy_run(mutant)
    se = np.std(mutant_var, axis=0, ddof=1) / math.sqrt(LANES)
    assert np.all(mutant_var.mean(0) - exact_var > 8.0 * se)


def test_rungs_after_temperature_one_are_noops():
    """A ladder that finishes early gives the same belief under a larger cap.

    Rung keys are folded from the rung index, so once the temperature reaches
    1 any further rung that resampled or moved would change the particles.
    Fails if masked rungs still resample, move, count acceptances or add to
    the evidence, or if the rung count is not reported.
    """
    target = ConjugateGaussianTarget()
    record = UpdaterRecord(
        time=0.0, ra=0.3, dec=-0.2, ra_err=SIGMA, dec_err=SIGMA, record_id=1
    )
    out = []
    for n_rungs in (8, 24):
        updater = OrbitUpdater(
            n_particles=256, n_rungs=n_rungs, n_moves=2, record_capacity=2
        )
        out.append(updater.init(target, record, _key(3)))
    (a, da), (b, db) = out
    assert 1 <= int(da.rungs_used) < 8
    assert float(da.temperature) == 1.0
    for x, y in zip(jax.tree.leaves((a, da)), jax.tree.leaves((b, db)), strict=True):
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


ORBIT_UPDATER = OrbitUpdater(n_particles=1024, n_rungs=32, n_moves=4, record_capacity=3)
ORBIT_SIGMA = 3.5 * mas2arcsec
ORBIT_TIMES = jnp.array([0.0, 180.0, 360.0])


def _orbit_records(target, theta, seed):
    """Noisy records of one orbit at ``ORBIT_TIMES``."""
    ra, dec = target.predict(theta, ORBIT_TIMES)
    noise = ORBIT_SIGMA * jax.random.normal(jax.random.key(seed), (2, 3))
    return [
        UpdaterRecord(
            time=ORBIT_TIMES[k],
            ra=ra[k] + noise[0, k],
            dec=dec[k] + noise[1, k],
            ra_err=ORBIT_SIGMA,
            dec_err=ORBIT_SIGMA,
            record_id=k + 1,
        )
        for k in range(3)
    ]


_PHYSICAL = FixedMassTarget()
ORBIT_CASES = {
    "ti": (
        ThieleInnesTarget(),
        jnp.array([0.25, math.log(365.25), 0.3, 0.7, -0.5, 0.4, 0.8]),
    ),
    "physical": (
        _PHYSICAL,
        _PHYSICAL.from_physical(jnp.array([1.3, 0.2, 0.4, 2.0, 0.7, 4.1])),
    ),
}


PREDICTIVE_LANES = 4


def _lanes(tree):
    """Broadcast a single record or target over the predictive lanes."""
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (PREDICTIVE_LANES, *jnp.shape(x))), tree
    )


@pytest.mark.medium
@pytest.mark.parametrize("name", ["ti", "physical"])
def test_orbit_target_one_epoch_predictive_and_no_recompile(name):
    """One epoch pins the predicted position to the record; one trace serves all.

    With a prior on the sky position hundreds of times wider than the 3.5 mas
    error, the one-epoch posterior of the predicted position at the record's
    epoch is close to N(record, sigma^2) per axis (an importance reference
    with 2^21 prior draws gives 0.96 and 1.02 sigma for the TI case). Over 4
    independent lanes the weighted mean must lie within sigma / 2 of the
    record and the lane-averaged weighted standard deviation within 15 percent
    of sigma. A single lane at 1024 particles occasionally collapses to half
    its distinct particles and reads 0.75 sigma, hence the lane average. The
    same jitted update then assimilates a second record (a different buffer
    count) and a record on a target with other array values. Fails if the
    tempering or the collapsed move applies the record twice (standard
    deviation sigma / sqrt 2) or stops short of temperature 1 (too wide), if
    the distance or chart conversion is dropped, or if a record count or
    target value enters the program as a static value.
    """
    target, theta = ORBIT_CASES[name]
    records = [_lanes(r) for r in _orbit_records(target, theta, seed=0)]
    traces = []

    @eqx.filter_jit
    def step(belief, record, keys):
        traces.append(None)
        return jax.vmap(ORBIT_UPDATER.update)(belief, record, keys)

    keys = jax.random.split(_key(10), 3 * PREDICTIVE_LANES).reshape(3, -1)
    prior = jax.vmap(ORBIT_UPDATER.prior_belief, in_axes=(None, 0))(
        target, jax.random.split(_key(9), PREDICTIVE_LANES)
    )
    belief, diag = step(prior, records[0], keys[0])
    assert bool(jnp.all(diag.status == UpdaterStatus.OK))
    assert bool(jnp.all(diag.temperature == 1.0))
    ra, dec = jax.vmap(jax.vmap(target.predict, in_axes=(0, None)), in_axes=(0, None))(
        belief.particles, ORBIT_TIMES[:1]
    )
    w = np.exp(np.asarray(belief.log_weights))
    for pred, obs in ((ra[..., 0], records[0].ra[0]), (dec[..., 0], records[0].dec[0])):
        pred = np.asarray(pred)
        mean = np.sum(w * pred, axis=1)
        sd = np.sqrt(np.sum(w * (pred - mean[:, None]) ** 2, axis=1))
        assert np.all(np.abs(mean - float(obs)) < 0.5 * ORBIT_SIGMA)
        assert 0.85 < np.mean(sd) / ORBIT_SIGMA < 1.15

    two, diag2 = step(belief, records[1], keys[1])
    assert bool(jnp.all(diag2.applied))
    assert bool(jnp.all(two.n_records == 2))
    other = eqx.tree_at(lambda t: t.dist_pc, prior.target, 1.2 * prior.target.dist_pc)
    step(eqx.tree_at(lambda b: b.target, prior, other), records[2], keys[2])
    assert len(traces) == 1


TI_MODEL = TIBasisModel(lnP_lo=math.log(80.0), lnP_hi=math.log(3000.0))
TI_DIST_PC = 10.0


def _two_epoch_ti_data():
    """Two epochs, 400 days apart, of a 1 AU orbit with 2 mas errors."""
    target = ThieleInnesTarget(TI_MODEL, dist_pc=TI_DIST_PC)
    theta = jnp.array([0.25, math.log(365.25), 0.3, 0.33, -0.59, 0.8, 0.27])
    times = jnp.array([0.0, 400.0])
    ra, dec = target.predict(theta, times)
    sigma = 0.002
    noise = sigma * jax.random.normal(jax.random.key(11), (2, 2))
    return RelativeAstromData(
        times=times,
        ra=ra + noise[0],
        dec=dec + noise[1],
        ra_err=jnp.full(2, sigma),
        dec_err=jnp.full(2, sigma),
        corr=jnp.zeros(2),
        planet_id=jnp.zeros(2, dtype=int),
        is_valid=jnp.ones(2, dtype=bool),
    )


def _period_masses(ln_period, log_weights):
    edges = np.linspace(TI_MODEL.lnP_lo, TI_MODEL.lnP_hi, 9)
    lw = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    w = np.exp(lw - lw.max())
    return np.histogram(ln_period, edges, weights=w / w.sum())[0]


@pytest.mark.slow
def test_ti_two_epoch_period_masses_match_lambert_reference():
    """Two-epoch ln-period masses agree roughly with the Lambert reference.

    The reference is the exact Lambert depth importance fit on the same model,
    with a depth window covering the support ((1 + e_max) a_max) and every
    revolution family the baseline allows, at 20,000 proposals per family
    (ESS of several hundred; two such runs differ by a total variation of
    about 0.05 over these 8 bins). The updater runs 8 independent lanes at
    8,192 particles, 32 rungs and 4 moves. Smoke tolerance: the total
    variation of the lane-averaged masses from the reference is at most 0.15.
    Fails if the kernel drops period aliases or misweights them; random-walk
    moves over all seven coordinates fail it (total variation about 0.25),
    because they cannot follow the thin two-epoch posterior and resampling
    then loses period lineages.
    """
    data = _two_epoch_ti_data()
    a_max = (TI_MODEL.mu_hi * math.exp(TI_MODEL.lnP_hi) ** 2 / (4.0 * math.pi**2)) ** (
        1.0 / 3.0
    )
    post = lambert_depth_fit(
        data,
        dist_pc=TI_DIST_PC,
        model=TI_MODEL,
        key=jax.random.PRNGKey(0),
        n_per_family=20_000,
        n_max_rev=math.ceil(400.0 / math.exp(TI_MODEL.lnP_lo)),
        z_max_au=(1.0 + TI_MODEL.e_max) * a_max,
    )
    reference = _period_masses(
        np.asarray(post.samples)[:, 1], np.asarray(post.log_weights)
    )

    updater = OrbitUpdater(n_particles=8192, n_rungs=32, n_moves=4, record_capacity=2)
    target = ThieleInnesTarget(TI_MODEL, dist_pc=TI_DIST_PC)
    records = [
        UpdaterRecord(
            time=data.times[k],
            ra=data.ra[k],
            dec=data.dec[k],
            ra_err=data.ra_err[k],
            dec_err=data.dec_err[k],
            record_id=k + 1,
        )
        for k in range(2)
    ]

    @eqx.filter_jit
    def run(key):
        belief, _ = updater.init(target, records[0], key)
        return updater.update(belief, records[1], jax.random.fold_in(key, 1))

    belief, diag = jax.vmap(run)(jax.random.split(_key(0), 8))
    assert bool(jnp.all(diag.applied))
    masses = np.mean(
        [
            _period_masses(
                np.asarray(belief.particles[i, :, 1]),
                np.asarray(belief.log_weights[i]),
            )
            for i in range(8)
        ],
        axis=0,
    )
    assert 0.5 * np.abs(masses - reference).sum() <= 0.15
