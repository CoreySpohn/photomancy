"""Posterior, run and the generic backends.

Two kinds of test. The fast ones use a one-leaf fit with a closed-form evidence: a
uniform prior on a planet radius times an unnormalized Gaussian likelihood, so every
backend has a known answer. The ``slow`` ones run each backend on the golden orbit (the
fixed-mass relative-astrometry fit whose density is pinned against the old system
density in ``test_fit.py``) and compare with the old backend on the same problem, or,
where the old chains are trapped by its walls, NUTS against MCLMC.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from priorities import Uniform
from test_fit import (
    GOLDEN,
    _block,
    _chart_coords,
    _coords,
    _relative,
    _scene,
)

from photomancy.infer import (
    MAP,
    MCLMC,
    NUTS,
    AbstractTerm,
    AdaptiveImportance,
    Fit,
    Grid,
    Importance,
    Laplace,
    MixtureProposal,
    NestedSampling,
    Pathfinder,
    Posterior,
    PriorProposal,
    run,
)
from photomancy.infer.backends._flat import draw_template, start_points
from photomancy.infer.backends.grid import gauss_legendre, midpoint
from photomancy.infer.backends.laplace import _merge_modes
from photomancy.infer.diagnostics import khat_threshold, pareto_khat
from photomancy.infer.evidence import importance_evidence
from photomancy.infer.posterior import cluster_to_mixture, ravel_samples

jax.config.update("jax_enable_x64", True)

KEY = "planets[0].Rp_Rearth"
HIGH = 4.0
WIDTH = 0.3


class _Gauss(AbstractTerm):
    """Unnormalized Gaussian likelihood of the planet radius, centered at ``center``."""

    center: jax.Array
    reads = ()

    def __call__(self, model):
        return -0.5 * jnp.sum(((model.planets[0].Rp_Rearth - self.center) / WIDTH) ** 2)

    def draw(self, key, model):
        return self


def _leaf_fit(center=1.5):
    return Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_Gauss(jnp.asarray(center)),))


def _exact_log_z(center=1.5):
    """Log of the mean over ``[0, HIGH]`` of ``exp(-0.5 ((r - c) / w)^2)``."""
    cdf = jax.scipy.stats.norm.cdf
    mass = cdf((HIGH - center) / WIDTH) - cdf((0.0 - center) / WIDTH)
    return float(jnp.log(mass * WIDTH * math.sqrt(2 * math.pi) / HIGH))


def _moments(post):
    """Weighted mean and standard deviation of the radius."""
    w = np.exp(np.asarray(post.log_weights))
    r = np.asarray(post.samples[KEY])[..., 0]
    mean = np.sum(w * r)
    return mean, math.sqrt(np.sum(w * (r - mean) ** 2))


def _check_contract(post, fit, n=None):
    """Fields every posterior has, whichever backend made it."""
    assert isinstance(post, Posterior)
    assert set(post.samples) == set(fit.free)
    for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
        assert getattr(post, name).dtype == jnp.float64
    if n is not None:
        assert post.log_weights.shape == (n,)
    np.testing.assert_allclose(
        float(jax.scipy.special.logsumexp(post.log_weights)), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(post.log_target),
        np.asarray(fit.log_density(post.samples)),
        rtol=1e-12,
    )


# ----- the Posterior type --------------------------------------------------------


def test_posterior_checks_its_fields():
    """Wrong kind, keys, dtype and shapes are refused; batch axes are draw axes."""
    fit = _leaf_fit()
    post = run(fit, Laplace(n_draws=64), jax.random.key(0))
    fields = {
        name: getattr(post, name)
        for name in (
            "samples",
            "log_weights",
            "log_target",
            "log_evidence",
            "log_evidence_se",
            "fit",
            "info",
            "kind",
        )
    }
    Posterior(**fields)
    with pytest.raises(ValueError, match="kind"):
        Posterior(**{**fields, "kind": "gibbs"})
    with pytest.raises(ValueError, match=r"keys of fit\.free"):
        Posterior(**{**fields, "samples": {"other": post.samples[KEY]}})
    with pytest.raises(ValueError, match="float64"):
        Posterior(**{**fields, "log_target": post.log_target.astype(jnp.float32)})
    with pytest.raises(ValueError, match="log_target has shape"):
        Posterior(**{**fields, "log_target": post.log_target[:-1]})
    with pytest.raises(ValueError, match="expected the batch shape"):
        Posterior(**{**fields, "log_evidence": jnp.zeros(3)})
    with pytest.raises(ValueError, match="leading axes"):
        Posterior(**{**fields, "samples": {KEY: post.samples[KEY][:-1]}})


def test_ravel_samples_round_trips_a_block_fit():
    """``ravel_samples`` flattens leaf and block entries and ``unravel`` inverts it."""
    fit = Fit(
        _scene(),
        {KEY: Uniform(0.5, 4.0), "dynamics.orbits": _block()},
        (_relative(),),
    )
    post = run(fit, Laplace(n_draws=8, n_steps=5), jax.random.key(1))
    flat, unravel = ravel_samples(post)
    assert flat.shape[:-1] == (8,)
    back = unravel(flat)
    for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(post.samples), strict=True):
        np.testing.assert_array_equal(a, b)
    only, _ = ravel_samples(post, keys=(KEY,))
    assert only.shape == (8, 1)


def test_cluster_to_mixture_follows_the_modes():
    """Two well-separated clusters give two modes with their masses."""
    fit = _leaf_fit()
    post = run(fit, Laplace(n_draws=400), jax.random.key(2))
    left = jnp.where(jnp.arange(400) % 4 == 0, 0.5, 3.0)
    split = eqx.tree_at(
        lambda p: (p.samples, p.log_weights),
        post,
        ({KEY: left[:, None]}, jnp.full(400, -math.log(400))),
    )
    mixture = cluster_to_mixture(split, 2, jax.random.key(3))
    means = np.sort(np.asarray(mixture.component_distribution.loc).ravel())
    np.testing.assert_allclose(means, [0.5, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        np.sort(np.exp(np.asarray(jax.nn.log_softmax(mixture.mixing_logits)))),
        [0.25, 0.75],
        atol=1e-6,
    )


# ----- backends on the one-leaf fit with a closed-form evidence ------------------


def test_grid_is_quadrature_with_the_exact_evidence():
    """A fine midpoint rule recovers the evidence and mean; refining converges."""
    fit = _leaf_fit()
    errors = []
    for n in (200, 800):
        post = run(fit, Grid({KEY: midpoint(0.0, HIGH, n)}), jax.random.key(0))
        assert post.kind == "quadrature"
        _check_contract(post, fit, n)
        errors.append(abs(float(post.log_evidence) - _exact_log_z()))
        assert np.isnan(float(post.log_evidence_se))
    assert errors[1] < errors[0] and errors[1] < 1e-5
    mean, sd = _moments(post)
    assert abs(mean - 1.5) < 1e-6 and abs(sd - WIDTH) < 1e-3
    gl = run(fit, Grid({KEY: gauss_legendre(0.0, HIGH, 200)}), jax.random.key(0))
    assert abs(float(gl.log_evidence) - _exact_log_z()) < 1e-6
    with pytest.raises(ValueError, match="non-positive weight"):
        run(fit, Grid({KEY: (jnp.arange(3.0), jnp.array([1.0, -1.0, 1.0]))}), None)
    with pytest.raises(ValueError, match="exactly the degrees of freedom"):
        run(fit, Grid({"x": midpoint(0.0, 1.0, 3)}), None)


def test_laplace_corrects_the_gaussian_with_importance_weights():
    """Weighted draws and evidence are right though the real-line target is skewed."""
    fit = _leaf_fit()
    post = run(fit, Laplace(n_draws=20000), jax.random.key(0))
    _check_contract(post, fit, 20000)
    assert abs(float(post.log_evidence) - _exact_log_z()) < 4 * float(
        post.log_evidence_se
    )
    mean, _ = _moments(post)
    assert abs(mean - 1.5) < 0.01
    assert post.info.flat_means.shape == (1, 1)


class _TwoBumps(AbstractTerm):
    """Likelihood with a bump at each of two radii, a bimodal target."""

    centers: jax.Array
    reads = ()

    def __call__(self, model):
        z = (model.planets[0].Rp_Rearth - self.centers) / WIDTH
        return jax.scipy.special.logsumexp(-0.5 * z**2)

    def draw(self, key, model):
        return self


def _bimodal_fit():
    centers = jnp.array([1.0, 3.0])
    return Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_TwoBumps(centers),))


def _mode_radii(fit, post):
    """Radii of the modes of a Laplace or Pathfinder run, in the stored coordinates."""
    x, _ = fit.from_real({KEY: post.info.flat_means})
    return np.asarray(x[KEY])[:, 0]


@pytest.mark.parametrize("backend", [Laplace, Pathfinder])
def test_explicit_starts_land_on_distinct_modes(backend):
    """One start per bump gives one mode per bump; a repeated start gives one mode."""
    fit = _bimodal_fit()
    starts = {KEY: jnp.array([[1.1], [2.9]])}
    post = run(fit, backend(n_draws=400, n_modes=2, init=starts), jax.random.key(0))
    np.testing.assert_allclose(np.sort(_mode_radii(fit, post)), [1.0, 3.0], atol=0.15)

    one = run(
        fit,
        backend(n_draws=400, n_modes=2, init={KEY: jnp.array([1.1])}),
        jax.random.key(0),
    )
    assert np.all(np.abs(_mode_radii(fit, one) - 1.0) < 0.15)


def test_laplace_merges_starts_that_reach_the_same_mode():
    """Three starts on two bumps keep two modes and map the duplicate to its mode.

    Truth source: the two bumps of ``_TwoBumps`` (radii 1 and 3). Fails if: a
    duplicate start enters the proposal as a second component of its mode, which
    doubles that mode's weight and biases the evidence.
    """
    fit = _bimodal_fit()
    starts = {KEY: jnp.array([[1.1], [0.9], [2.9]])}
    backend = Laplace(n_draws=400, n_modes=3, init=starts)
    post = run(fit, backend, jax.random.key(0))
    info = post.info
    assert int(info.n_modes_kept) == 2
    index = np.asarray(info.mode_index)
    assert index[2] == 2 and index[0] == index[1] and index[0] in (0, 1)
    assert np.asarray(info.kept).tolist().count(True) == 2
    assert not bool(info.kept[index[0] ^ 1]) and bool(info.kept[index[0]])
    radii = np.asarray(info.modes[KEY])[:, 0]
    np.testing.assert_allclose(np.sort(radii[[index[0], 2]]), [1.0, 3.0], atol=0.15)
    again = run(
        fit, Laplace(n_draws=400, n_modes=3, init=info.modes), jax.random.key(0)
    )
    assert np.asarray(again.info.kept).tolist() == np.asarray(info.kept).tolist()
    np.testing.assert_array_equal(again.info.mode_index, info.mode_index)


def test_merge_distance_wraps_on_periodic_coordinates():
    """Two modes on either side of the period boundary are one mode."""
    means = jnp.array([[0.02, 0.0], [5.98, 0.0], [3.0, 0.0]])
    covs = jnp.broadcast_to(jnp.eye(2) * 0.01, (3, 2, 2))
    args = (means, covs, jnp.array([2.0, 1.0, 0.0]), jnp.ones(3, bool))
    kept, index = _merge_modes(*args, (6.0, None), 1.0)
    assert np.asarray(kept).tolist() == [True, False, True]
    assert np.asarray(index).tolist() == [0, 0, 2]
    kept, _ = _merge_modes(*args, (None, None), 1.0)
    assert np.asarray(kept).all()
    kept, index = _merge_modes(
        *args[:3], jnp.array([True, False, True]), (6.0, None), 1.0
    )
    assert np.asarray(index).tolist() == [0, -1, 2]


def test_explicit_starts_refuse_a_wrong_leading_length():
    """The leading axis must match the number of starts."""
    with pytest.raises(ValueError, match="one start each"):
        run(
            _bimodal_fit(),
            Laplace(n_draws=100, n_modes=2, init={KEY: jnp.ones((3, 1))}),
            jax.random.key(0),
        )


@pytest.mark.parametrize("with_leaf", [False, True], ids=["block_only", "block_leaf"])
def test_stacked_starts_pass_through_for_a_block_fit(with_leaf):
    """Stacked starts of a block fit become ``(n, d)`` flat starts, one per start.

    Fails if: a block entry's rank is compared as a whole, so the starts raise a
    misleading error (block plus leaf) or are silently broadcast to ``(n, n, d)``.
    """
    priors = {"dynamics.orbits": _block()}
    if with_leaf:
        priors[KEY] = Uniform(0.0, HIGH)
    fit = Fit(_scene(), priors, (_relative(),))
    n = 3
    starts = fit.sample_prior(jax.random.key(1), (n,))
    flat = start_points(fit, starts, jax.random.key(0), n)
    assert flat.shape == (n, ravel_pytree(fit.to_real(draw_template(fit)))[0].shape[0])
    for i in range(n):
        one = jax.tree.map(lambda a, i=i: a[i], starts)
        np.testing.assert_allclose(flat[i], fit.to_flat(one), rtol=1e-12)
    single = start_points(fit, one, jax.random.key(0), n)
    assert single.shape == flat.shape
    np.testing.assert_allclose(single, jnp.broadcast_to(fit.to_flat(one), flat.shape))


def test_mixed_rank_starts_name_the_entry():
    """One entry stacked and another not raises, and the message names an entry."""
    fit = Fit(
        _scene(),
        {"dynamics.orbits": _block(), KEY: Uniform(0.0, HIGH)},
        (_relative(),),
    )
    starts = fit.sample_prior(jax.random.key(1), (3,))
    bad = {**starts, KEY: starts[KEY][0]}
    with pytest.raises(ValueError, match="extra leading axis"):
        start_points(fit, bad, jax.random.key(0), 3)


def test_nuts_has_no_evidence_and_the_right_moments():
    """NUTS draws match the known mean; the evidence is NaN; chains are in info."""
    fit = _leaf_fit()
    post = run(fit, NUTS(n_warmup=200, n_draws=500, n_chains=4), jax.random.key(0))
    _check_contract(post, fit, 2000)
    assert np.isnan(float(post.log_evidence)) and np.isnan(float(post.log_evidence_se))
    assert post.kind == "monte_carlo"
    np.testing.assert_array_equal(np.asarray(post.info.chain), np.repeat(range(4), 500))
    r = np.asarray(post.samples[KEY])[:, 0].reshape(4, 500)
    se = r.mean(axis=1).std(ddof=1) / 2.0
    assert abs(r.mean() - 1.5) < 5 * max(se, 0.01)
    assert not np.any(np.asarray(post.info.divergent))


def test_mclmc_recovers_the_mean():
    """MCLMC (two or more dimensions) matches the known mean and has no evidence."""
    fit = Fit(
        _scene(),
        {KEY: Uniform(0.0, HIGH), "planets[0].physical_model.Ag": Uniform(0.1, 1.0)},
        (_Gauss(jnp.asarray(1.5)),),
    )
    post = run(fit, MCLMC(n_tune=500, n_draws=2000, n_chains=2), jax.random.key(0))
    _check_contract(post, fit, 4000)
    assert np.isnan(float(post.log_evidence))
    assert abs(_moments(post)[0] - 1.5) < 0.05
    ag = np.asarray(post.samples["planets[0].physical_model.Ag"]).ravel()
    assert abs(ag.mean() - 0.55) < 0.05


def test_pathfinder_importance_weights_target_the_fit():
    """Pathfinder's mixture proposal is corrected to the fit's density."""
    fit = _leaf_fit()
    post = run(fit, Pathfinder(n_draws=5000), jax.random.key(0))
    _check_contract(post, fit, 5000)
    assert abs(_moments(post)[0] - 1.5) < 0.02


def test_map_finds_the_mode():
    """MAP returns one draw at the optimum."""
    fit = _leaf_fit()
    post = run(fit, MAP(n_steps=100), jax.random.key(0))
    assert post.log_weights.shape == (1,)
    assert np.isnan(float(post.log_evidence))
    # the optimum of the real-line density, which carries the logit volume
    x = float(post.samples[KEY][0, 0])
    assert abs(x - 1.5) < 0.1
    np.testing.assert_allclose(
        float(post.log_target[0]), float(fit.log_density(post.samples)[0]), rtol=1e-12
    )


def test_nested_sampling_recovers_the_evidence():
    """Jaxns gives the exact evidence within its standard deviation.

    A run cut by ``max_samples`` (jaxns termination reason 1) keeps its samples and
    weights but has no evidence: fails if a finite value comes back from a run that
    stopped before convergence.
    """
    pytest.importorskip("jaxns")
    fit = _leaf_fit()
    cut = run(fit, NestedSampling(max_samples=120), jax.random.key(0))
    assert int(cut.info.termination_reason) == 1
    assert np.isnan(float(cut.log_evidence)) and np.isnan(float(cut.log_evidence_se))
    assert cut.log_weights.shape == (120,) and np.all(np.isfinite(cut.log_weights))
    post = run(fit, NestedSampling(max_samples=20000), jax.random.key(0))
    assert int(post.info.termination_reason) == 0
    _check_contract(post, fit)
    se = float(post.log_evidence_se)
    assert abs(float(post.log_evidence) - _exact_log_z()) < 5 * max(se, 0.02)
    assert abs(_moments(post)[0] - 1.5) < 0.05
    block_fit = Fit(_scene(), {"dynamics.orbits": _block()}, (_relative(),))
    unit = block_fit.unit_template()["dynamics.orbits"]
    assert unit.shape == (_block().unit_size,)
    stored = block_fit.from_unit(
        jax.tree.map(lambda u: u + 0.5, block_fit.unit_template())
    )
    assert bool(jnp.isfinite(block_fit.log_density(stored)))


def test_a_stack_of_worlds_equals_the_per_world_loop():
    """``eqx.filter_vmap(run)`` over two stacked fits equals running each in turn."""
    centers = (1.2, 2.0)
    fits = [_leaf_fit(c) for c in centers]
    static = eqx.partition(fits[0], eqx.is_array)[1]
    stack = eqx.combine(
        jax.tree.map(
            lambda *a: jnp.stack(a), *[eqx.partition(f, eqx.is_array)[0] for f in fits]
        ),
        static,
    )
    backend = Laplace(n_draws=300)
    keys = jax.random.split(jax.random.key(7), 2)
    batched = eqx.filter_vmap(lambda f, k: run(f, backend, k))(stack, keys)
    assert batched.log_weights.shape == (2, 300)
    assert batched.log_evidence.shape == (2,)
    assert batched.samples[KEY].shape == (2, 300, 1)
    for i, fit in enumerate(fits):
        one = run(fit, backend, keys[i])
        np.testing.assert_allclose(
            np.asarray(batched.log_evidence[i]), float(one.log_evidence), rtol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(batched.samples[KEY][i]), np.asarray(one.samples[KEY]), rtol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(batched.log_target[i]), np.asarray(one.log_target), rtol=1e-9
        )


def _stack(fits):
    """Stack fits of one structure along a new leading world axis."""
    static = eqx.partition(fits[0], eqx.is_array)[1]
    arrays = [eqx.partition(f, eqx.is_array)[0] for f in fits]
    return eqx.combine(jax.tree.map(lambda *a: jnp.stack(a), *arrays), static)


_WORLD_BACKENDS = {
    "nuts": NUTS(n_warmup=60, n_draws=40, n_chains=2),
    "pathfinder": Pathfinder(n_draws=200),
    "laplace": Laplace(n_draws=200),
    "importance": Importance(PriorProposal(), n_draws=200),
    "adaptive": AdaptiveImportance(
        PriorProposal(),
        n_draws=200,
        n_adapt_draws=400,
        n_rounds=2,
        n_centers=16,
        target_cess=0.05,
    ),
    "grid": Grid({KEY: midpoint(0.0, HIGH, 120)}),
}


@pytest.mark.parametrize("name", list(_WORLD_BACKENDS))
def test_every_batching_backend_maps_over_two_worlds(name):
    """``filter_vmap(run)`` over two stacked fits gives two posteriors of one type.

    Truth source: each world's own fit: ``log_target`` is its density at the stored
    draws and the posterior mean follows that world's data center (1.2 against 2.0).
    Tolerance basis: shapes and the density identity are exact (rtol 1e-9); the means
    differ by 0.8 between worlds, far above the sampler scatter, so each is compared to
    its center within 0.25 (200 particles, 80 draws, quadrature exact).
    Fails if: a backend closes over the fit instead of taking it as a traced argument,
    mixes the worlds or breaks under a batch prefix.
    """
    centers = (1.2, 2.0)
    fits = [_leaf_fit(c) for c in centers]
    backend = _WORLD_BACKENDS[name]
    keys = jax.random.split(jax.random.key(3), 2)
    out = eqx.filter_vmap(lambda f, k: run(f, backend, k))(_stack(fits), keys)
    n = out.log_weights.shape[-1]
    assert out.log_weights.shape == (2, n) and out.log_evidence.shape == (2,)
    assert out.samples[KEY].shape == (2, n, 1)
    for i, (fit, center) in enumerate(zip(fits, centers, strict=True)):
        np.testing.assert_allclose(
            np.asarray(out.log_target[i]),
            np.asarray(fit.log_density({KEY: out.samples[KEY][i]})),
            rtol=1e-9,
        )
        w = np.exp(np.asarray(out.log_weights[i]))
        mean = float(np.sum(w * np.asarray(out.samples[KEY][i])[:, 0]))
        assert abs(mean - center) < 0.25


class _NanAbove(_Gauss):
    """The Gaussian term, NaN for radii above 2.2 (a failed evaluation)."""

    def __call__(self, model):
        r = model.planets[0].Rp_Rearth[0]
        return jnp.where(r > 2.2, jnp.nan, super().__call__(model))


def test_a_nan_term_refuses_the_slice_instead_of_zero_weight():
    """NaN values of a term make the whole slice NaN, in weights and evidence.

    Fails if: a backend or ``run`` maps NaN to a zero weight, which would return a
    clean-looking posterior and a finite evidence from the draws that did not fail.
    """
    fit = Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_NanAbove(jnp.asarray(1.5)),))
    start = {KEY: jnp.asarray([1.5])}
    laplace = run(fit, Laplace(n_draws=4000, init=start), jax.random.key(0))
    assert np.any(np.isnan(np.asarray(fit.log_density(laplace.samples))))
    grid = run(fit, Grid({KEY: midpoint(0.0, HIGH, 200)}), jax.random.key(0))
    for post in (laplace, grid):
        assert np.all(np.isnan(np.asarray(post.log_weights)))
        assert np.isnan(float(post.log_evidence))


def _prior_only_orbit_fit():
    return Fit(_scene(), {"dynamics.orbits": _block()}, ())


@pytest.mark.medium(reason="eight 4000-draw runs of one compiled Laplace, 25 s cold")
def test_laplace_prior_only_evidence_on_an_orbit_block_is_zero():
    """The prior-only evidence through ``Laplace`` averages to 0 over eight keys.

    Truth source: the block prior is normalized, so ``log Z = 0``. A single key passes
    on only about 79 percent of keys at the old floor of 0.1, which carried a bias of
    about -0.02 nat (24 keys: -0.027 +- 0.005; at 0.03: +0.001 +- 0.005), so the test
    averages eight keys.
    Tolerance basis: the mean within 3 standard errors of the mean, the delta-method
    errors of the eight runs added in quadrature over 8, with every k-hat below 1 (a
    finite mean of the weights). The estimator is taken on the raw weights of every
    key, because a key whose k-hat is above ``khat_threshold`` publishes a NaN
    evidence; the published evidence is that estimate where the k-hat passes and NaN
    where it does not.
    Fails if: the wrapped proposal density does not integrate to one over a cell, or
    the default floor narrows the proposal enough to bias the evidence.
    """
    fit = _prior_only_orbit_fit()
    laplace = Laplace(n_draws=4000)
    draws = [laplace._run(fit, jax.random.key(k)) for k in range(8)]
    z, se = np.array([importance_evidence(d.log_weights) for d in draws]).T
    assert abs(z.mean()) < 3 * np.sqrt(np.sum(se**2)) / 8, (z, se)
    khat = np.array([float(d.info.importance.khat) for d in draws])
    assert khat.max() < 1.0
    published = np.array([float(d.log_evidence) for d in draws])
    passes = khat <= khat_threshold(4000)
    np.testing.assert_allclose(published[passes], z[passes], rtol=1e-12)
    assert np.all(np.isnan(published[~passes]))


def test_adaptive_importance_seeded_from_the_laplace_cloud_is_consistent():
    """The documented remedy: refit a mixture on the Laplace cloud and adapt from it.

    Truth source: the prior-only evidence is exactly 0. Measured: log Z 0.002 +- 0.010,
    k-hat 0.28. Fails if: ``MixtureProposal.from_cloud`` or the adaptive rounds lose
    the periodic wrap or the tail cover.
    """
    fit = _prior_only_orbit_fit()
    laplace = run(fit, Laplace(n_draws=4000), jax.random.key(0))
    flat = fit.to_flat(laplace.samples)
    seed = MixtureProposal.from_cloud(
        jax.random.key(1),
        flat,
        laplace.log_weights,
        n_centers=64,
        periods=fit.periods(),
    )
    post = run(
        fit,
        AdaptiveImportance(seed, n_draws=4000, n_adapt_draws=2**14),
        jax.random.key(2),
    )
    assert abs(float(post.log_evidence)) < 3 * float(post.log_evidence_se)
    assert float(pareto_khat(post.log_weights)) < 0.7


@pytest.mark.parametrize("sequence", ["random", "roberts"])
def test_prior_importance_prior_only_evidence_is_exactly_zero(sequence):
    """Drawing from the prior itself gives unit weights, so ``log Z = 0`` to rounding.

    Fails if: ``log_q`` of ``PriorProposal`` differs from the fit's log density by
    anything but rounding (a volume term or a periodic cell mismatch).
    """
    post = run(
        _prior_only_orbit_fit(),
        Importance(PriorProposal(sequence), n_draws=500),
        jax.random.key(0),
    )
    assert abs(float(post.log_evidence)) < 1e-8
    assert float(post.log_evidence_se) < 1e-8


@pytest.mark.parametrize("sequence", ["random", "roberts"])
def test_prior_importance_recovers_the_conjugate_evidence(sequence):
    """Prior-sampling importance on the one-leaf fit recovers the exact evidence.

    Truth source: ``_exact_log_z`` (closed form). Tolerance basis: 3 standard errors of
    the importance mean (a quasi-random sequence is held to the same bound).
    """
    post = run(
        _leaf_fit(),
        Importance(PriorProposal(sequence), n_draws=20000),
        jax.random.key(0),
    )
    se = float(post.log_evidence_se)
    assert abs(float(post.log_evidence) - _exact_log_z()) < 3 * se


def test_a_ladder_that_stops_short_of_one_refuses_the_slice():
    """An adaptive run whose last proposal was fitted below temperature one is NaN.

    Truth source: ``_exact_log_z`` for the one-leaf fit. With 2000 draws per round and
    16 centers, a rung needs a Kish size of 1000 and the full step to temperature one
    leaves about 380: two rounds stop at a temperature below one (NaN evidence, standard
    error and weights, ``reached_target`` false, the rung on record); eight rounds reach
    one and the evidence is within 3 standard errors. Fails if a short ladder returns
    weights from a proposal fitted at a lower temperature, whose Kish size and k-hat
    look healthy.
    """
    fit = _leaf_fit()
    settings = {"n_draws": 2000, "n_adapt_draws": 2000, "n_centers": 16}
    short = run(
        fit,
        AdaptiveImportance(PriorProposal(), n_rounds=2, **settings),
        jax.random.key(0),
    )
    assert not bool(short.info.reached_target)
    assert 0.0 < float(short.info.temperatures[0]) < 1.0
    assert float(short.info.temperatures[-1]) == 1.0
    assert np.isnan(float(short.log_evidence))
    assert np.isnan(float(short.log_evidence_se))
    assert np.isnan(np.asarray(short.log_weights)).all()
    full = run(
        fit,
        AdaptiveImportance(PriorProposal(), n_rounds=8, **settings),
        jax.random.key(0),
    )
    assert bool(full.info.reached_target)
    assert bool(np.all(np.diff(np.asarray(full.info.temperatures)) >= 0))
    se = float(full.log_evidence_se)
    assert abs(float(full.log_evidence) - _exact_log_z()) < 3 * se


def test_a_seeded_adaptive_run_starts_at_temperature_one():
    """A proposal other than the prior is a seed: every round is at temperature one.

    Fails if a seeded run tempers from zero (it then discards the seed and, with a
    short ladder, is refused for a posterior its seed already covers).
    """
    fit = _leaf_fit()
    laplace = run(fit, Laplace(n_draws=2000), jax.random.key(0))
    seed = MixtureProposal.from_cloud(
        jax.random.key(1),
        fit.to_flat(laplace.samples),
        laplace.log_weights,
        n_centers=16,
        periods=fit.periods(),
    )
    post = run(
        fit,
        AdaptiveImportance(
            seed, n_draws=2000, n_adapt_draws=2000, n_rounds=2, n_centers=16
        ),
        jax.random.key(0),
    )
    assert bool(post.info.reached_target)
    assert float(post.info.temperatures[0]) == 1.0


def test_importance_evidence_is_the_importance_mean_of_its_weights():
    """The backend evidence equals ``importance_evidence`` on the same weights.

    The weights are rebuilt from the same key through the proposal's own ``sample``;
    equality is to rounding (rtol 1e-10). Fails also if the evidence is published
    above the k-hat limit: a Gaussian proposal a little wider than the target keeps a
    finite evidence within 4 SE of the exact value, one four times narrower (weights of
    infinite variance) returns NaN evidence and error with finite weights and its
    ``info.khat`` above ``khat_threshold``.
    """
    fit = _leaf_fit()
    proposal = PriorProposal()
    key = jax.random.key(4)
    post = run(fit, Importance(proposal, n_draws=500), key)
    z, log_q = proposal.sample(key, fit, 500)
    x, volume = fit.from_real(jax.vmap(_unravel_leaf(fit))(z))
    log_w = fit.log_density(x) + volume - log_q
    log_z, se = importance_evidence(log_w)
    np.testing.assert_allclose(float(post.log_evidence), float(log_z), rtol=1e-10)
    np.testing.assert_allclose(float(post.log_evidence_se), float(se), rtol=1e-10)
    # the target is about N(-0.51, 0.32^2) in the flat coordinate of the leaf
    for sd, refused in ((0.4, False), (0.08, True)):
        gauss = MixtureProposal(jnp.array([[-0.51]]), jnp.array([[[sd**2]]]))
        post = run(fit, Importance(gauss, n_draws=2000), key)
        assert np.all(np.isfinite(np.asarray(post.log_weights)))
        if refused:
            assert float(post.info.khat) > khat_threshold(2000)
            assert np.isnan(float(post.log_evidence))
            assert np.isnan(float(post.log_evidence_se))
        else:
            gap = abs(float(post.log_evidence) - _exact_log_z())
            assert gap < 4 * float(post.log_evidence_se)


def _unravel_leaf(fit):
    return ravel_pytree(fit.to_real(fit.sample_prior(jax.random.key(0))))[1]


@pytest.mark.parametrize(
    "backend",
    [NUTS(n_warmup=30, n_draws=20, n_chains=2)],
)
def test_stored_angles_lie_in_one_cell(backend):
    """Sampled orbit angles are stored reduced to ``[0, 2 pi)`` for every sampler.

    Fails if: ``Fit.from_real`` leaves a periodic coordinate unwrapped, so a chain that
    drifts past the cell edge stores an angle outside it.
    """
    fit = _prior_only_orbit_fit()
    post = run(fit, backend, jax.random.key(0))
    block = post.samples["dynamics.orbits"]
    for name, period in zip(_block().coordinate_names, _block().periods, strict=True):
        values = np.asarray(block[name])
        if period is None:
            continue
        assert values.min() >= 0.0 and values.max() < period, name


class _AlwaysNan(_Gauss):
    """The Gaussian term with every evaluation NaN."""

    def __call__(self, model):
        return super().__call__(model) * jnp.nan


def test_laplace_with_no_surviving_mode_returns_a_nan_slice():
    """A fit whose only mode fails the guard gives NaN weights and evidence.

    The term is NaN everywhere, so no mode has a finite position and none is kept.
    Fails if: the guard returns a clean-looking posterior.
    """
    fit = Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_AlwaysNan(jnp.asarray(1.5)),))
    post = run(fit, Laplace(n_draws=200), jax.random.key(0))
    assert int(post.info.n_modes_kept) == 0
    assert np.all(np.isnan(np.asarray(post.log_weights)))
    assert np.isnan(float(post.log_evidence))


def test_run_refuses_a_fit_built_without_x64():
    """Fit refuses float32 mode at construction, so ``run`` never sees one."""
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            _leaf_fit()
    finally:
        jax.config.update("jax_enable_x64", True)


# ----- the golden orbit -----------------------------------------------------------


def _golden_fit():
    return Fit(_scene(), {"dynamics.orbits": _block()}, (_relative(),))


def _golden_start():
    point = GOLDEN["g1"]["points"][0]
    return {"dynamics.orbits": _coords(point)}


def _new_coords(post):
    """Chart coordinates of each draw as ``(N, 6)`` in the old coordinate order."""
    return np.asarray(_chart_coords(post.samples["dynamics.orbits"])[0])


def _wrap_angles(z):
    """Mean of the three non-angle coordinates, then unit vectors of the angles."""
    angles = z[..., 3:]
    return np.concatenate(
        [z[..., :3], np.cos(angles), np.sin(angles)],
        axis=-1,
    )


def _chain_features(post, n_chains, n_draws):
    """Wrap-invariant features of each chain mean, ``(n_chains, 9)``."""
    f = _wrap_angles(_new_coords(post).reshape(n_chains, n_draws, 6))
    return f.mean(axis=1), f.reshape(-1, f.shape[-1]).std(axis=0)


GOLDEN_LOG_Z = -19.04


@pytest.mark.slow
def test_adaptive_importance_on_the_golden_orbit_converges_or_refuses():
    """Adaptive importance from the prior returns the converged evidence or NaN.

    Truth source: the converged evidence -19.04 of a 40-round run (two keys gave -19.046
    and -19.039, standard error 0.015). With the default three rounds the ladder cannot
    reach temperature one on this concentrated posterior, so the slice is NaN with
    ``reached_target`` false; the old rule returned -175, -82 and -106 with a standard
    error of 1.0 and a k-hat of 0.02. With 40 rounds of 2**13 the evidence is within 3
    standard errors of -19.04 and the k-hat is below 0.7. Fails if a short ladder
    returns a finite number, or a long one converges to a different value.
    """
    fit = _golden_fit()
    short = run(fit, AdaptiveImportance(PriorProposal()), jax.random.key(0))
    assert not bool(short.info.reached_target)
    assert np.isnan(float(short.log_evidence))
    assert np.isnan(np.asarray(short.log_weights)).all()
    full = run(
        fit,
        AdaptiveImportance(
            PriorProposal(),
            n_draws=2**13,
            n_adapt_draws=2**13,
            n_rounds=40,
            n_centers=64,
        ),
        jax.random.key(0),
    )
    assert bool(full.info.reached_target)
    se = float(full.log_evidence_se)
    assert abs(float(full.log_evidence) - GOLDEN_LOG_Z) < 3 * max(se, 0.015)
    assert float(pareto_khat(full.log_weights)) < 0.7


@pytest.mark.slow
def test_nuts_and_mclmc_agree_on_the_golden_orbit():
    """Two independent gradient samplers give the same posterior feature means.

    Truth source: each other. The old backend is no longer the reference, because its
    hard walls (the period range and each angle cell) trap its chains at the edge of
    the cell, while the mapped block has no wall inside the support; an importance
    reference degenerates on this multimodal posterior (Kish size of a few draws).
    Tolerance basis: five combined standard errors (scatter of the chain means of each
    sampler: four NUTS chains, three MCLMC chains), floored at 2 percent of the
    posterior standard deviation. Features are the three non-angle chart coordinates
    and cos and sin of each angle.
    Fails if: the mapped block or the chart coordinates misplace the density for one
    sampler's geometry (the two samplers share nothing but the density).
    """
    nuts = run(
        _golden_fit(),
        NUTS(n_warmup=400, n_draws=600, n_chains=4, init=_golden_start()),
        jax.random.key(11),
    )
    mclmc = run(
        _golden_fit(),
        MCLMC(n_tune=1500, n_draws=3000, n_chains=3, init=_golden_start()),
        jax.random.key(5),
    )
    m_n, sd = _chain_features(nuts, 4, 600)
    m_c, _ = _chain_features(mclmc, 3, 3000)
    se = np.sqrt(m_n.var(axis=0, ddof=1) / 4 + m_c.var(axis=0, ddof=1) / 3)
    diff = np.abs(m_n.mean(axis=0) - m_c.mean(axis=0))
    assert np.all(diff < 5 * np.maximum(se, 0.02 * sd)), (diff, se, sd)
