"""Posterior operations: draws, prior swap, append, forecast and information gain.

The one-leaf fit of ``test_run`` (a uniform radius prior times an unnormalized Gaussian
term) has a closed-form evidence, so a prior swap and an appended term are checked
against the exact answer and against a direct run. The importance-weight cases of the
gate use a bare Gaussian prior so the weights are known in closed form.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priorities import Normal, Uniform
from test_fit import _scene
from test_run import HIGH, KEY, WIDTH, _Gauss, _leaf_fit

from photomancy.infer import (
    SMC,
    Fit,
    Grid,
    Laplace,
    Posterior,
    eig_contrastive,
    eig_nested,
    kish_ess,
    pareto_khat,
    run,
)
from photomancy.infer.backends.grid import midpoint

jax.config.update("jax_enable_x64", True)

norm = jax.scipy.stats.norm


def _check_target(post):
    """``log_target`` equals the density of the retained fit at the stored draws."""
    np.testing.assert_allclose(
        np.asarray(post.log_target),
        np.asarray(post.fit.log_density(post.samples)),
        rtol=1e-11,
        atol=1e-11,
    )


def _moments(post):
    w = np.exp(np.asarray(post.log_weights))
    r = np.asarray(post.samples[KEY])[..., 0]
    mean = np.sum(w * r)
    return mean, math.sqrt(np.sum(w * (r - mean) ** 2))


def _log_z_window(low, high, center=1.5):
    """Exact log evidence of the radius fit with a ``Uniform(low, high)`` prior."""
    mass = norm.cdf((high - center) / WIDTH) - norm.cdf((low - center) / WIDTH)
    return float(jnp.log(mass * WIDTH * math.sqrt(2 * math.pi) / (high - low)))


def _gaussian_cloud(proposal_sd, prior_sd=1.0, n=4000, seed=0):
    """Draws of a ``N(0, proposal_sd^2)`` proposal weighted to a ``N(0, prior_sd^2)``.

    The fit is the bare prior on the radius leaf (no terms), so ``log_evidence`` is 0.
    """
    fit = Fit(_scene(), {KEY: Normal(0.0, prior_sd)})
    x = proposal_sd * jax.random.normal(jax.random.key(seed), (n, 1))
    log_q = jnp.sum(norm.logpdf(x, 0.0, proposal_sd), axis=-1)
    samples = {KEY: x}
    log_target = fit.log_density(samples)
    return Posterior(
        samples=samples,
        log_weights=jax.nn.log_softmax(log_target - log_q),
        log_target=log_target,
        log_evidence=jnp.asarray(0.0, jnp.float64),
        log_evidence_se=jnp.asarray(0.0, jnp.float64),
        fit=fit,
        info=None,
        kind="monte_carlo",
    )


@pytest.fixture(scope="module")
def cloud():
    """The one-leaf posterior from an importance-weighted Laplace run."""
    return run(_leaf_fit(), Laplace(n_draws=8000), jax.random.key(0))


# ----- sample and resample -------------------------------------------------------


def test_sample_is_categorical_with_the_requested_shape(cloud):
    """Draws follow the weights for any ``sample_shape``; keys give independent draws.

    Truth source: the weighted mean and standard deviation of the stored cloud.
    Tolerance basis: 20000 iid categorical draws give a mean error of about
    ``sd / sqrt(20000)`` = 2e-3; the bound is 5 times that.
    Fails if: ``sample`` ignores the weights, mixes up the axes or returns the same
    draws for different keys.
    """
    draws = cloud.sample(jax.random.key(1), (4, 5000))[KEY]
    assert draws.shape == (4, 5000, 1)
    mean, sd = _moments(cloud)
    assert abs(float(jnp.mean(draws)) - mean) < 5 * sd / math.sqrt(20000)
    assert abs(float(jnp.std(draws)) - sd) < 0.02
    keys = jax.random.split(jax.random.key(2), 3)
    each = jax.vmap(lambda k: cloud.sample(k, (6,))[KEY])(keys)
    assert each.shape == (3, 6, 1)
    assert len({tuple(np.asarray(r).ravel()) for r in each}) == 3
    scalar = cloud.sample(jax.random.key(3))[KEY]
    assert scalar.shape == (1,)


def test_resample_is_systematic_with_uniform_weights(cloud):
    """Counts differ from ``n w`` by less than one; weights are uniform; target indexed.

    Truth source: systematic resampling gives each draw ``floor(n w)`` or
    ``ceil(n w)`` copies.
    Fails if: the offsets are redrawn per output, ``log_target`` is not carried or
    chain information is kept.
    """
    small = cloud.resample(jax.random.key(4), 200)
    assert small.log_weights.shape == (200,) and small.info is None
    np.testing.assert_allclose(np.asarray(small.log_weights), -math.log(200))
    np.testing.assert_allclose(small.log_evidence, cloud.log_evidence)
    _check_target(small)
    n = len(cloud.log_weights)
    ids = np.arange(n)
    # one-hot marker: resample the index through the draw axis
    marked = eqx.tree_at(
        lambda p: p.samples, cloud, {KEY: jnp.asarray(ids, jnp.float64)[:, None]}
    )
    out = np.asarray(marked.resample(jax.random.key(5), n).samples[KEY])[:, 0]
    counts = np.bincount(out.astype(int), minlength=n)
    expected = n * np.exp(np.asarray(cloud.log_weights))
    assert np.all(np.abs(counts - expected) < 1.0 + 1e-9)


def test_sample_and_resample_mark_a_slice_with_no_support():
    """A NaN-weight slice returns NaN draws and keeps NaN weights, per slice."""
    good = _gaussian_cloud(1.0, n=64)
    bad = eqx.tree_at(lambda p: p.log_weights, good, jnp.full(64, jnp.nan, jnp.float64))
    stack = jax.tree.map(lambda a, b: jnp.stack([a, b]), good, bad)
    draws = stack.sample(jax.random.key(0), (5,))[KEY]
    assert draws.shape == (2, 5, 1)
    assert np.all(np.isfinite(draws[0])) and np.all(np.isnan(draws[1]))
    out = stack.resample(jax.random.key(0), 8)
    assert np.all(np.isfinite(out.log_weights[0]))
    assert np.all(np.isnan(out.log_weights[1]))


# ----- prior swap ----------------------------------------------------------------


def test_prior_swap_agrees_with_direct_sampling_and_the_exact_evidence(cloud):
    """Swapping the radius prior reproduces a direct run of the new prior.

    Truth source: the closed-form evidence of the Gaussian term under
    ``Uniform(1, 3)`` and a quadrature run of the same fit.
    Tolerance basis: the reported standard error, 4 sigma (the reported error is an
    upper bound by construction), plus 1e-3 on the mean and 1e-2 on the sd, which is
    the Monte Carlo error of 20000 weighted draws.
    Fails if: the ratio uses the wrong sign or prior, the evidence is not updated or
    ``log_target`` goes stale.
    """
    swapped = cloud.with_priors({KEY: Uniform(1.0, 3.0)}, min_n_eff=1000, max_khat=0.7)
    exact = _log_z_window(1.0, 3.0)
    se = float(swapped.log_evidence_se)
    assert se > float(cloud.log_evidence_se) and se < 0.05
    assert abs(float(swapped.log_evidence) - exact) < 4 * se
    _check_target(swapped)
    direct = run(swapped.fit, Grid({KEY: midpoint(1.0, 3.0, 2000)}), jax.random.key(0))
    assert abs(float(swapped.log_evidence) - float(direct.log_evidence)) < 4 * se
    (m1, s1), (m2, s2) = _moments(swapped), _moments(direct)
    assert abs(m1 - m2) < 1e-2 and abs(s1 - s2) < 1e-2
    assert swapped.fit.free[KEY].low == 1.0
    assert cloud.fit.free[KEY].low == 0.0


def test_a_swap_to_a_narrow_prior_is_refused_by_the_size_gate(cloud):
    """A prior that keeps a sliver of the posterior comes back as NaN weights."""
    out = cloud.with_priors(
        {KEY: Uniform(1.49, 1.51)}, min_n_eff=2000, max_khat=jnp.inf
    )
    assert np.all(np.isnan(out.log_weights)) and np.isnan(out.log_evidence)
    assert np.isnan(out.log_evidence_se)
    _check_target(out)
    raw = cloud.with_priors({KEY: Uniform(1.49, 1.51)}, min_n_eff=0, max_khat=jnp.inf)
    assert not np.any(np.isnan(raw.log_weights))
    assert abs(float(raw.log_evidence) - _log_z_window(1.49, 1.51)) < 4 * float(
        raw.log_evidence_se
    )


def test_the_gate_reads_the_combined_weights_not_the_ratio():
    """Two consumer cases on 4000 weighted draws, refit prior N(0, 1).

    Proposal N(0, 0.7^2) with role prior N(0, 1.3^2): the ratio alone has a small
    k-hat but the combined weights have a heavy tail, so the swap must refuse at
    ``max_khat=0.5``. Proposal N(0, 1.5^2) with role N(0, 1.4^2): the combined
    weights are mild (negative k-hat) and the swap passes.
    Truth source: k-hat and Kish of the combined weights computed here directly.
    Fails if: the gate sees the ratio only, or compares ``khat <= limit`` without
    refusing NaN.
    """
    heavy = _gaussian_cloud(0.7)
    role = {KEY: Normal(0.0, 1.3)}
    ratio = role[KEY].log_prob(heavy.samples[KEY]) - heavy.fit.free[KEY].log_prob(
        heavy.samples[KEY]
    )
    ratio = ratio[:, 0]
    combined = jax.nn.log_softmax(heavy.log_weights + ratio)
    assert float(pareto_khat(ratio)) < 0.5
    khat = float(pareto_khat(combined))
    assert 0.5 < khat < 0.8
    assert 400 < float(kish_ess(combined)) < 1500
    refused = heavy.with_priors(role, min_n_eff=100, max_khat=0.5)
    assert np.all(np.isnan(refused.log_weights))
    assert np.all(
        np.isfinite(heavy.with_priors(role, min_n_eff=100, max_khat=1.0).log_weights)
    )
    mild = _gaussian_cloud(1.5)
    role = {KEY: Normal(0.0, 1.4)}
    ok = mild.with_priors(role, min_n_eff=100, max_khat=0.5)
    assert np.all(np.isfinite(ok.log_weights))
    assert float(pareto_khat(ok.log_weights)) < 0.0
    _check_target(ok)


def test_an_identity_swap_on_a_heavy_tailed_cloud_is_refused():
    """The gate also re-gates the input cloud's own tail."""
    heavy = _gaussian_cloud(0.5, n=4000)
    out = heavy.with_priors({KEY: Normal(0.0, 1.0)}, min_n_eff=0, max_khat=0.5)
    assert np.all(np.isnan(out.log_weights))


def test_gate_arguments_follow_the_kind_and_a_nan_term_refuses(cloud):
    """Missing thresholds raise on the host; a NaN term value refuses its slice."""
    with pytest.raises(ValueError, match="required"):
        cloud.with_priors({KEY: Uniform(1.0, 3.0)}, min_n_eff=None, max_khat=None)
    with pytest.raises(ValueError, match="cannot change the free set"):
        cloud.with_priors({"other": Uniform(1.0, 3.0)}, min_n_eff=0, max_khat=1.0)
    grid = run(_leaf_fit(), Grid({KEY: midpoint(0.0, HIGH, 400)}), jax.random.key(0))
    with pytest.raises(ValueError, match="no Monte Carlo gate"):
        grid.with_priors({KEY: Uniform(1.0, 3.0)}, min_n_eff=10, max_khat=0.7)
    swapped = grid.with_priors({KEY: Uniform(1.0, 3.0)}, min_n_eff=None, max_khat=None)
    assert swapped.kind == "quadrature" and np.isnan(swapped.log_evidence_se)
    assert abs(float(swapped.log_evidence) - _log_z_window(1.0, 3.0)) < 1e-4
    _check_target(swapped)

    class _Nan(_Gauss):
        def __call__(self, model):
            return jnp.where(model.planets[0].Rp_Rearth[0] > 1.5, jnp.nan, 0.0)

    out = cloud.append(_Nan(jnp.asarray(0.0)), min_n_eff=0, max_khat=jnp.inf)
    assert np.all(np.isnan(out.log_weights)) and np.isnan(out.log_evidence)


# ----- append --------------------------------------------------------------------


@pytest.mark.medium
def test_append_agrees_with_a_refit_and_the_exact_evidence():
    """One more Gaussian term by reweighting equals a quadrature refit of both terms.

    Truth source: a 4000-node midpoint quadrature of the two-term fit (evidence error
    below 1e-6) and the closed-form product of the two Gaussians.
    Tolerance basis: 4 reported standard errors on the evidence; 1e-2 on the moments.
    Fails if: the term is evaluated on the wrong model, the old terms are counted
    twice, or the evidence increment is dropped.
    """
    first = run(_leaf_fit(1.5), Laplace(n_draws=8000), jax.random.key(0))
    extra = _Gauss(jnp.asarray(1.8))
    out = first.append(extra, min_n_eff=1000, max_khat=0.7)
    assert len(out.fit.terms) == 2 and len(first.fit.terms) == 1
    _check_target(out)
    refit = run(out.fit, Grid({KEY: midpoint(0.0, HIGH, 4000)}), jax.random.key(0))
    se = float(out.log_evidence_se)
    assert 0.0 < se < 0.05
    assert abs(float(out.log_evidence) - float(refit.log_evidence)) < 4 * se
    (m1, s1), (m2, s2) = _moments(out), _moments(refit)
    assert abs(m1 - m2) < 1e-2 and abs(s1 - s2) < 1e-2
    # Product of N(r; 1.5, w) N(r; 1.8, w) = N(1.65, w / sqrt 2) times a constant.
    assert abs(m2 - 1.65) < 1e-3 and abs(s2 - WIDTH / math.sqrt(2)) < 1e-3


def test_append_refuses_a_term_that_reads_a_free_path(cloud):
    """The static check of ``Fit`` runs first and raises on the host."""

    class _Reads(_Gauss):
        reads = (KEY,)

    with pytest.raises(ValueError, match="Rp_Rearth"):
        cloud.append(_Reads(jnp.asarray(1.0)), min_n_eff=0, max_khat=1.0)


# ----- per-slice refusal under vmap ----------------------------------------------


def test_the_refusal_is_per_slice_under_filter_vmap():
    """In a stack of two worlds the swap passes for one and refuses the other.

    World a has its posterior at 1.5 and world b at 3.5; the role prior
    ``Uniform(1.2, 1.8)`` keeps all of a and almost nothing of b.
    Fails if: the gate raises on a traced value, applies one decision to the whole
    stack, or differs from the per-world result.
    """
    a = run(_leaf_fit(1.5), Laplace(n_draws=4000), jax.random.key(0))
    b = run(_leaf_fit(3.5), Laplace(n_draws=4000), jax.random.key(1))
    stack = jax.tree.map(lambda x, y: jnp.stack([x, y]), a, b)
    role = {KEY: Uniform(1.2, 1.8)}

    def swap(p):
        return p.with_priors(role, min_n_eff=500, max_khat=0.7)

    out = eqx.filter_vmap(swap)(stack)
    assert out.log_weights.shape == (2, 4000)
    assert not np.any(np.isnan(out.log_weights[0]))
    assert np.all(np.isnan(out.log_weights[1]))
    assert np.isfinite(out.log_evidence[0]) and np.isnan(out.log_evidence[1])
    alone = swap(a)
    np.testing.assert_allclose(out.log_weights[0], alone.log_weights, rtol=1e-12)
    np.testing.assert_allclose(out.log_evidence[0], alone.log_evidence, rtol=1e-12)
    np.testing.assert_allclose(out.log_target[0], alone.log_target, rtol=1e-12)


# ----- model and forecast --------------------------------------------------------


def test_model_reads_the_draw_through_the_fit(cloud):
    """``model(i)`` has the draw at the free path and the held leaves elsewhere."""
    i = 123
    model = cloud.model(i)
    np.testing.assert_array_equal(model.planets[0].Rp_Rearth, cloud.samples[KEY][i])
    assert float(model.planets[0].physical_model.Ag[0]) == 0.3
    assert float(cloud.model(jnp.asarray(i)).planets[0].Rp_Rearth[0]) == float(
        cloud.samples[KEY][i, 0]
    )


def test_forecast_obeys_the_law_of_total_variance(cloud):
    """``noise + spread`` is the variance of the mixture of per-draw predictives.

    Truth source: the second moment ``sum w (v + m^2) - mean^2`` computed from the
    arrays, and a Monte Carlo variance of 200000 draws of the mixture (5 standard
    errors of the variance, about 3 percent).
    Fails if: the spread is centered on the wrong mean, the weights are dropped from
    either term, or the two parts are swapped.
    """

    def predict(model):
        r = model.planets[0].Rp_Rearth[0]
        return 2.0 * r, 0.05 + 0.1 * r

    out = cloud.forecast(predict)
    w = np.exp(np.asarray(cloud.log_weights))
    r = np.asarray(cloud.samples[KEY])[:, 0]
    m, v = 2.0 * r, 0.05 + 0.1 * r
    mean = np.sum(w * m)
    total = np.sum(w * (v + m**2)) - mean**2
    np.testing.assert_allclose(out.mean, mean, rtol=1e-12)
    np.testing.assert_allclose(out.noise_variance, np.sum(w * v), rtol=1e-12)
    np.testing.assert_allclose(
        out.spread_variance, np.sum(w * (m - mean) ** 2), rtol=1e-10
    )
    np.testing.assert_allclose(out.variance, total, rtol=1e-10)
    chunked = cloud.forecast(predict, batch_size=500)
    np.testing.assert_allclose(chunked.variance, out.variance, rtol=1e-12)
    idx = np.random.default_rng(0).choice(len(w), size=200000, p=w / w.sum())
    y = m[idx] + np.sqrt(v[idx]) * np.asarray(
        jax.random.normal(jax.random.key(1), (200000,))
    )
    assert abs(np.var(y) / total - 1.0) < 0.03
    assert out.noise_variance > 0 and out.spread_variance > 0


# ----- information gain ----------------------------------------------------------


class _Toy(eqx.Module):
    theta: jax.Array


def _gaussian_channel(n=2048, sigma=0.7, seed=0):
    """A prior cloud of a scalar read through ``y = theta + sigma * noise``.

    The information gain of one observation is ``0.5 log(1 + 1 / sigma^2)``.
    """
    fit = Fit(_Toy(jnp.zeros(())), {"theta": Normal(0.0, 1.0)})
    x = fit.sample_prior(jax.random.key(seed), (n,))
    log_target = fit.log_density(x)
    post = Posterior(
        samples=x,
        log_weights=jnp.full(n, -math.log(n), jnp.float64),
        log_target=log_target,
        log_evidence=jnp.asarray(0.0, jnp.float64),
        log_evidence_se=jnp.asarray(0.0, jnp.float64),
        fit=fit,
        info=None,
        kind="monte_carlo",
    )

    def log_obs(y, model):
        return norm.logpdf(y, model.theta, sigma)

    def draw(key, model):
        return model.theta + sigma * jax.random.normal(key)

    return post, log_obs, draw, 0.5 * math.log1p(1.0 / sigma**2)


@pytest.mark.medium
def test_nested_eig_bias_falls_with_the_inner_size_and_matches_its_scale():
    """The nested estimate is biased upward by about ``bias_scale``, ``O(1/n_inner)``.

    Truth source: the closed-form gain of the Gaussian channel,
    ``0.5 log(1 + 1 / sigma^2)``, 0.5561 nats at sigma 0.7 (a posterior that is the
    prior cloud and a normalized Gaussian observation density).
    Tolerance basis: spike over four seeds of the cloud and outer draws (4000 outer
    draws, Monte Carlo error about 0.01 nat): bias 0.60 to 0.68 at ``n_inner`` 2,
    0.11 to 0.13 at 8, 0.02 to 0.04 at 32, 0.00 to 0.03 at 128, against scales 0.12,
    0.07, 0.022 and 0.006. The scale is leading order, so it is compared to the bias
    within a factor of 2.5 from ``n_inner`` 8 up and its ratio between inner sizes
    to the expected 4 within 2.5 to 5.
    Fails if: the inner mean includes the diagonal, the inner draws share the outer
    draw, or the bias scale is not the explicit variance sum.
    """
    post, log_obs, draw, truth = _gaussian_channel()
    biases, scales = [], []
    for n_inner in (2, 8, 32, 128):
        est = eig_nested(
            post, log_obs, draw, jax.random.key(7), 4000, n_inner, False, 100.0
        )
        assert est.estimator == "nested" and est.bias == "upward"
        biases.append(float(est.eig) - truth)
        scales.append(float(est.bias_scale))
    assert all(b > -0.03 for b in biases)
    assert biases[0] > 0.4 and biases[0] > biases[1] > biases[2] > biases[3] - 0.03
    assert abs(biases[3]) < 0.05
    assert scales[0] > scales[1] > scales[2] > scales[3] > 0
    for bias, scale in zip(biases[1:3], scales[1:3], strict=True):
        assert scale / 2.5 < bias < 2.5 * scale
    assert 2.5 < scales[1] / scales[2] < 5.0 and 2.5 < scales[2] / scales[3] < 5.0


def test_contrastive_eig_is_a_lower_bound_capped_at_log_n_eff():
    """The contrastive estimate sits at or below the truth and under ``log n_eff``.

    Truth source: the closed-form Gaussian-channel gain (0.5561 nats at sigma 0.7)
    and, for the ceiling, a 16-draw cloud whose bound cannot pass ``log 16``.
    Tolerance basis: spike over four seeds: the estimate minus the truth is between
    -0.022 and +0.014 at 4000 outer draws (Monte Carlo error about 0.015), so the bound
    is checked as ``< truth + 0.04`` and ``> truth - 0.1``.
    Fails if: the inner sum drops the diagonal (the estimate then exceeds the truth) or
    the weights are not applied.
    """
    post, log_obs, draw, truth = _gaussian_channel()
    est = eig_contrastive(post, log_obs, draw, jax.random.key(8), n_outer=4000)
    assert est.estimator == "contrastive" and est.bias == "downward"
    assert truth - 0.1 < float(est.eig) < truth + 0.04
    assert 0 < float(est.bias_scale) < 0.01
    tiny, log_obs, draw, _ = _gaussian_channel(n=16, sigma=0.05)
    capped = eig_contrastive(tiny, log_obs, draw, jax.random.key(9), n_outer=2000)
    assert 1.5 < float(capped.eig) <= math.log(16.0) + 1e-9
    assert float(capped.bias_scale) > 0.1 > float(est.bias_scale)


def test_nested_eig_gates_on_the_cloud_size_and_with_the_diagonal_is_a_bound():
    """A small Kish size returns NaN; the diagonal form lands below the open form."""
    post, log_obs, draw, truth = _gaussian_channel(n=512)
    refused = eig_nested(
        post, log_obs, draw, jax.random.key(1), 100, 8, False, 10_000.0
    )
    assert np.isnan(float(refused.eig)) and np.isnan(float(refused.bias_scale))
    up = eig_nested(post, log_obs, draw, jax.random.key(2), 3000, 4, False, 1.0)
    down = eig_nested(post, log_obs, draw, jax.random.key(2), 3000, 4, True, 1.0)
    assert down.bias == "downward"
    assert float(down.eig) < truth + 0.03 < float(up.eig) + 0.03
    assert float(down.eig) < float(up.eig)


def test_a_prior_swap_on_an_orbit_block_keeps_log_target_equal_to_the_density():
    """The block's normalized density enters the ratio and ``log_target`` follows.

    Truth source: ``fit.log_density`` of the swapped fit at the stored draws.
    Fails if: the block prior is left out of the ratio or counted per coordinate.
    """
    from orbix.system import OrbitLaw
    from priorities import LogUniform
    from skyscapes.population import kipping_eccentricity
    from test_fit import PERIOD_RANGE, _block, _relative

    fit = Fit(_scene(), {"dynamics.orbits": _block()}, (_relative(),))
    post = run(fit, SMC(n_particles=200, n_mcmc_steps=3), jax.random.key(1))
    law = OrbitLaw(
        period_d=LogUniform(PERIOD_RANGE[0] * 0.8, PERIOD_RANGE[1] * 1.2),
        eccentricity=kipping_eccentricity(),
    )
    out = post.with_priors(
        {"dynamics.orbits": _block(law)}, min_n_eff=0, max_khat=jnp.inf
    )
    _check_target(out)
    assert not np.any(np.isnan(out.log_weights))
    assert out.fit.digest() != post.fit.digest()
