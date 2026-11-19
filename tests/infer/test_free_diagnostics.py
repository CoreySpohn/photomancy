"""Free diagnostics: khat against a reference fit, mc_se, batching, quantiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.infer import (
    khat_threshold,
    kish_ess,
    mc_se,
    normalize_log_weights,
    pareto_khat,
    weighted_cov,
    weighted_mean,
    weighted_quantile,
)

jax.config.update("jax_enable_x64", True)


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


@pytest.mark.parametrize("xi", [0.2, 0.6, 1.1])
def test_pareto_khat_matches_reference_fit(xi):
    """Khat equals an independent Zhang-Stephens fit of the same tail."""
    rng = np.random.default_rng(3)
    log_w = _pareto_log_ratios(rng, 4000, xi)
    log_w[::7] = -np.inf
    got = float(pareto_khat(jnp.asarray(log_w)))
    np.testing.assert_allclose(got, _reference_khat(log_w), rtol=0, atol=1e-9)


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


def test_weighted_quantile_matches_numpy_with_repetition():
    """Integer weights reproduce the repeated-sample quantile exactly."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(12, 3))
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
