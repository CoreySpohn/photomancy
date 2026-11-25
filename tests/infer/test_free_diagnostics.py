"""Free diagnostics: khat against a reference fit, mc_se, batching, quantiles."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.infer import (
    diagnostics,
    khat_threshold,
    kish_ess,
    mc_se,
    normalize_log_weights,
    pareto_khat,
    ratio_mc_se,
    weighted_cov,
    weighted_mean,
    weighted_quantile,
)


def _zhang_stephens(x):
    """Reference GPD fit (Zhang and Stephens 2009) with the PSIS shrinkage."""
    x = np.sort(x)
    n = len(x)
    m = 30 + int(np.floor(np.sqrt(n)))
    xq = x[int(np.floor(n / 4 + 0.5)) - 1]
    theta = np.array(
        [1 / x[-1] + (1 - np.sqrt(m / (j - 0.5))) / (3 * xq) for j in range(1, m + 1)]
    )
    k = np.array([np.mean(np.log1p(-t * x)) for t in theta])
    prof = n * (np.log(-theta / k) - k - 1)
    w = np.array([1 / np.sum(np.exp(prof - p)) for p in prof])
    theta_hat = np.sum(theta * w)
    k_hat = np.mean(np.log1p(-theta_hat * x))
    return (n * k_hat + 10 * 0.5) / (n + 10)


def _reference_khat(log_w):
    log_w = log_w[np.isfinite(log_w)]
    s = len(log_w)
    m = int(np.ceil(min(s / 5, 3 * np.sqrt(s))))
    r = np.exp(np.sort(log_w[np.isfinite(log_w)])[::-1] - np.max(log_w))
    return _zhang_stephens(r[:m] - r[m])


def _pareto_log_ratios(rng, size, xi):
    """Ratios with a generalized-Pareto tail of shape ``xi``."""
    return -xi * np.log(rng.uniform(size=size))


def _max_bound(log_w, level=1e-3):
    """Pareto max bound ``(lw_max - lw_cut) / -log(1 - level**(1/n))`` in NumPy."""
    lw = np.sort(log_w[np.isfinite(log_w)])[::-1]
    s = len(lw)
    m = int(np.ceil(min(s / 5, 3 * np.sqrt(s))))
    n = int(np.sum(lw[:m] > lw[m]))
    return (lw[0] - lw[m]) / -math.log1p(-(level ** (1.0 / n)))


def _near_plateau(seed, n=1000, jitter=1e-4):
    """Bounded weights: a near-plateau at 0.67 of the maximum under the cutoff."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    w = rng.uniform(0.05, 0.67, size=n)
    plateau = u < 0.25
    w[plateau] = 0.67 * (1.0 + jitter * rng.standard_normal(plateau.sum()))
    above = u > 0.97
    w[above] = rng.uniform(0.67, 1.0, size=above.sum())
    return np.log(w)


@pytest.mark.parametrize("xi", [0.2, 0.6, 1.1])
def test_pareto_khat_matches_reference_fit(xi):
    """Khat equals an independent Zhang-Stephens fit of the same tail.

    The max bound is slack in each case, so the comparison is of the fit itself.
    """
    rng = np.random.default_rng(3)
    log_w = _pareto_log_ratios(rng, 4000, xi)
    log_w[::7] = -np.inf
    assert _max_bound(log_w) > _reference_khat(log_w)
    got = float(pareto_khat(jnp.asarray(log_w)))
    np.testing.assert_allclose(got, _reference_khat(log_w), rtol=0, atol=1e-9)


def test_the_max_bound_clears_a_near_plateau_of_bounded_weights(monkeypatch):
    """Bounded weights with the cutoff inside a plateau pass the gate.

    Truth source: the fit alone (the cap switched off) reads a heavy tail on
    these weights, which have a largest ratio of 1.5 times the cutoff.
    Fails if: the cap is removed from ``_khat``.
    """
    log_w = jnp.asarray(_near_plateau(0))
    assert float(pareto_khat(log_w)) <= 0.7
    monkeypatch.setattr(diagnostics, "_MAX_BOUND_LEVEL", 0.0)
    assert float(diagnostics._khat(log_w)) > 1.0


def test_khat_equals_the_pareto_max_bound_where_the_bound_binds():
    """The reported value is ``(lw_max - lw_cut) / -log(1 - 1e-3**(1/n))``.

    Fails if: the level or the exponent ``1/n`` is wrong.
    """
    log_w = _near_plateau(0)
    np.testing.assert_allclose(
        float(pareto_khat(jnp.asarray(log_w))), _max_bound(log_w), rtol=1e-12, atol=0
    )


def test_a_heavy_tail_behind_a_plateau_is_still_refused():
    """Pareto k = 1.2 ratios floored at their 0.95 quantile still fail the gate.

    Fails if: the cap is replaced by a mean of log exceedances.
    """
    rng = np.random.default_rng(4)
    log_w = _pareto_log_ratios(rng, 1000, 1.2)
    floor = -1.2 * math.log(0.05)
    low = log_w < floor
    log_w[low] = floor + 1e-4 * rng.uniform(size=low.sum())
    assert float(pareto_khat(jnp.asarray(log_w))) > khat_threshold(1000)


def test_the_max_bound_misses_a_true_pareto_tail_rarely():
    """For exact Pareto k = 0.8 ratios the bound is below 0.8 in at most 2 of 200 seeds.

    Fails if: the level or the exponent ``1/n`` is wrong.
    """
    rng = np.random.default_rng(5)
    log_w = _pareto_log_ratios(rng, (200, 1000), 0.8)
    bound = np.array([_max_bound(row) for row in log_w])
    assert np.sum(bound < 0.8) <= 2
    khat = np.asarray(pareto_khat(jnp.asarray(log_w)))
    assert np.all(khat <= bound + 1e-12)


def test_khat_edge_cases_with_the_max_bound():
    """Flat weights give -inf, a short tail NaN, an exact plateau a finite value."""
    assert pareto_khat(jnp.zeros(200)) == -jnp.inf
    short = np.zeros(500)
    short[:4] = [1.0, 2.0, 3.0, 4.0]
    assert jnp.isnan(pareto_khat(jnp.asarray(short)))
    plateau = np.zeros(500)
    plateau[:6] = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
    got = float(pareto_khat(jnp.asarray(plateau)))
    assert np.isfinite(got) and got <= _max_bound(plateau) + 1e-12
    pair = jnp.stack([jnp.asarray(plateau), jnp.asarray(_near_plateau(1, n=500))])
    np.testing.assert_allclose(
        pareto_khat(pair), [pareto_khat(row) for row in pair], rtol=1e-12
    )


def test_khat_jits_and_batches_on_the_near_plateau():
    """``jit`` and a vmapped batch equal the eager value."""
    log_w = jnp.asarray(_near_plateau(0))
    eager = float(pareto_khat(log_w))
    np.testing.assert_allclose(float(jax.jit(pareto_khat)(log_w)), eager, rtol=1e-12)
    batch = jax.vmap(pareto_khat)(jnp.stack([log_w, log_w]))
    np.testing.assert_allclose(batch, [eager, eager], rtol=1e-12)


_EXAMPLES = Path(__file__).parents[1] / "data" / "khat_examples"


@pytest.mark.parametrize(("star", "want"), [(29, 0.147), (125, 0.152), (0, -0.204)])
def test_khat_on_the_consumer_example_weights(star, want):
    """Three combined weight vectors from a 150-star survey (1,000 draws each).

    Stars 29 and 125 were refused (k-hat 1.9 and 1.7) at ratios of at most 2.3
    to the mean; star 0 is the control that always passed.
    Fails if: the cap is removed (stars 29 and 125 read above 1.6 again).
    """
    log_w = jnp.asarray(np.load(_EXAMPLES / f"log_weights_star{star}.npy"))
    assert log_w.shape == (1000,)
    got = float(pareto_khat(log_w))
    assert got == pytest.approx(want, abs=1e-3)
    assert got <= 0.7


def test_pareto_khat_recovers_known_shape():
    """A large Pareto sample returns its true tail shape."""
    rng = np.random.default_rng(5)
    log_w = _pareto_log_ratios(rng, 40000, 0.5)
    assert abs(float(pareto_khat(jnp.asarray(log_w))) - 0.5) < 0.08


def test_heavy_tail_trips_threshold_and_light_does_not():
    """A shape above the limit is flagged and a light tail is not."""
    rng = np.random.default_rng(11)
    limit = khat_threshold(4000)
    assert limit == pytest.approx(0.7)
    heavy = _pareto_log_ratios(rng, 4000, 1.2)
    light = rng.normal(size=4000) * 0.3
    assert float(pareto_khat(jnp.asarray(heavy))) > limit
    assert float(pareto_khat(jnp.asarray(light))) <= limit
    assert khat_threshold(100) == pytest.approx(0.5)


def test_pareto_khat_edge_cases():
    """Too few entries give NaN and a bounded tail gives -inf."""
    assert jnp.isnan(pareto_khat(jnp.array([0.0, -jnp.inf, -jnp.inf, 1.0])))
    assert pareto_khat(jnp.zeros(200)) == -jnp.inf


def test_mc_se_known_variance_importance_case():
    """Target N(0,1) from a N(0,s^2) proposal: the delta-method error is known."""
    s, n, reps = 1.5, 4000, 150
    c = (1 - 1 / s**2) / 2
    expected = np.sqrt(s * (1 + 2 * c) ** (-1.5) / n)
    x = s * jax.random.normal(jax.random.key(0), (reps, n))
    log_w = -(x**2) / 2 + x**2 / (2 * s**2)
    se = mc_se(log_w, x)
    est = weighted_mean(log_w, x)
    assert se.shape == (reps,)
    np.testing.assert_allclose(jnp.sqrt(jnp.mean(se**2)), expected, rtol=0.05)
    np.testing.assert_allclose(jnp.std(est), expected, rtol=0.2)
    np.testing.assert_allclose(mc_se(log_w, x, r_eff=0.25), 2 * se, rtol=1e-12)


def _cloud(key, batch=(2,), n=64, d=3):
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (*batch, n)), jax.random.normal(k2, (*batch, n, d))


def test_batch_prefix_equals_per_slice_loop():
    """Batched, vmapped and per-slice results agree."""
    log_w, vals = _cloud(jax.random.key(1))
    q = jnp.array([0.1, 0.5, 0.9])
    r_eff = jnp.array([0.3, 0.9])
    cases = {
        "ess": (kish_ess, (log_w,)),
        "khat": (pareto_khat, (log_w,)),
        "mean": (weighted_mean, (log_w, vals)),
        "cov": (weighted_cov, (log_w, vals)),
        "se": (mc_se, (log_w, vals)),
        "quantile": (lambda a, b: weighted_quantile(a, b, q), (log_w, vals)),
    }
    for name, (fn, args) in cases.items():
        batched = fn(*args)
        for i in range(2):
            one = fn(*(a[i] for a in args))
            np.testing.assert_allclose(batched[i], one, rtol=1e-12, err_msg=name)
        viewed = jax.vmap(fn)(*args)
        np.testing.assert_allclose(viewed, batched, rtol=1e-12, err_msg=name)
    out = mc_se(log_w, vals, r_eff=r_eff)
    np.testing.assert_allclose(
        out[1], mc_se(log_w[1], vals[1], r_eff=r_eff[1]), rtol=1e-12
    )
    norm, total, none = normalize_log_weights(log_w)
    assert norm.shape == log_w.shape and total.shape == none.shape == (2,)


def test_weighted_quantile_matches_numpy_with_repetition(precision):
    """Integer weights reproduce the repeated-sample quantile exactly."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(12, 3)).astype(precision)
    counts = rng.integers(0, 6, size=12)
    counts[0] = 3
    q = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    expanded = np.repeat(values, counts, axis=0)
    want = np.quantile(expanded, q, axis=0, method="inverted_cdf")
    with np.errstate(divide="ignore"):
        log_w = np.log(counts.astype(float))
    got = weighted_quantile(jnp.asarray(log_w), jnp.asarray(values), jnp.asarray(q))
    assert got.shape == (7, 3)
    np.testing.assert_array_equal(got, want)


def test_weighted_moments_match_numpy_and_offsets():
    """Weighted mean and covariance match NumPy at a large weight offset."""
    rng = np.random.default_rng(4)
    values = rng.normal(size=(50, 3))
    w = rng.uniform(size=50)
    # The offset rounds each log weight by about 2e-12, which sets the tolerances.
    log_w = jnp.log(w) + 1e4
    mean = np.average(values, axis=0, weights=w)
    cov = np.cov(values.T, aweights=w, ddof=0)
    np.testing.assert_allclose(weighted_mean(log_w, values), mean, rtol=1e-9)
    np.testing.assert_allclose(weighted_cov(log_w, values), cov, rtol=1e-10)
    np.testing.assert_allclose(weighted_cov(log_w, values[:, 0]), cov[0, 0], rtol=1e-10)


def test_no_support_gives_nan_and_zero_weight_values_are_ignored():
    """No support gives NaN; zero-weight draws cannot poison a result."""
    log_w = jnp.full(5, -jnp.inf)
    vals = jnp.arange(5.0)
    assert jnp.isnan(weighted_mean(log_w, vals))
    assert jnp.isnan(mc_se(log_w, vals))
    assert jnp.isnan(weighted_quantile(log_w, vals, 0.5))
    assert kish_ess(log_w) == 0
    poisoned = jnp.array([1.0, jnp.nan, 3.0])
    log_w = jnp.array([0.0, -jnp.inf, 0.0])
    assert weighted_mean(log_w, poisoned) == 2.0


def test_ratio_mc_se_matches_stand_in_form():
    """The error is the ratio-estimator form and inf without condition mass."""
    rng = np.random.default_rng(8)
    n = 400
    log_w = rng.normal(size=n)
    g = rng.uniform(size=n)
    h = g * rng.normal(size=n)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    den = w @ g
    q = (w @ h) / den
    want = np.sqrt(np.sum(w**2 * (h - q * g) ** 2)) / den
    got = ratio_mc_se(jnp.asarray(log_w), jnp.asarray(h), jnp.asarray(g))
    np.testing.assert_allclose(got, want, rtol=1e-12)
    np.testing.assert_allclose(
        ratio_mc_se(jnp.asarray(log_w), jnp.asarray(h), jnp.asarray(g), r_eff=0.5),
        want * np.sqrt(2),
        rtol=1e-12,
    )
    assert jnp.isinf(ratio_mc_se(jnp.asarray(log_w), jnp.asarray(h), jnp.zeros(n)))
    lw2 = jnp.stack([jnp.asarray(log_w)] * 2)
    both = ratio_mc_se(
        lw2, jnp.stack([jnp.asarray(h)] * 2), jnp.stack([jnp.asarray(g)] * 2)
    )
    np.testing.assert_allclose(both, [want, want], rtol=1e-12)


def test_khat_rounding_flat_weights_are_flat():
    """Weights that differ only by logsumexp rounding return -inf."""
    rng = np.random.default_rng(6)
    raw = rng.normal(size=500)
    log_w = raw - np.log(np.sum(np.exp(raw))) + np.log(np.sum(np.exp(raw)))
    jitter = 4 * np.finfo(float).eps * rng.uniform(-1, 1, size=500)
    assert pareto_khat(jnp.asarray(np.full(500, -6.2) + jitter)) == -jnp.inf
    assert pareto_khat(jnp.asarray(log_w)) > -jnp.inf


@pytest.mark.parametrize("levels", [2, 3, 4])
def test_khat_discrete_weight_levels_fail_closed_or_pass(levels):
    """A few discrete levels leave fewer than five distinct exceedances: NaN."""
    rng = np.random.default_rng(levels)
    log_w = np.log(np.arange(1, levels + 1))[rng.integers(0, levels, size=1000)]
    log_w = log_w + 1e-15 * rng.uniform(size=1000)
    out = pareto_khat(jnp.asarray(log_w))
    assert jnp.isnan(out) or out == -jnp.inf
    assert not bool(out <= 0.7) or out == -jnp.inf


def test_khat_bin_probability_reweight_levels():
    """Reweighting by a per-bin probability ratio (few levels, big plateau)."""
    rng = np.random.default_rng(9)
    ratio = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 9.0])
    log_w = np.log(ratio[rng.integers(0, 6, size=2000)])
    # The top plateau holds the whole tail: no draw exceeds the cutoff.
    assert pareto_khat(jnp.asarray(log_w)) == -jnp.inf
    # A continuous perturbation on top restores a fittable, light tail.
    noisy = log_w + 0.05 * rng.normal(size=2000)
    assert float(pareto_khat(jnp.asarray(noisy))) < 0.7
