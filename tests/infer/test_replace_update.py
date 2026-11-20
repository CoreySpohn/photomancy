"""``replace_term``, ``update`` and ``batch_size`` on the Posterior operations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priorities import Uniform
from test_fit import _scene
from test_operations import _check_target, _gaussian_channel
from test_run import HIGH, KEY, WIDTH, _Gauss

from photomancy.infer import (
    AbstractTerm,
    Fit,
    Grid,
    Laplace,
    ReweightInfo,
    eig_contrastive,
    eig_nested,
    kish_ess,
    pareto_khat,
    run,
    update,
)
from photomancy.infer.backends.grid import midpoint

jax.config.update("jax_enable_x64", True)

GATE = {"min_n_eff": 500, "max_khat": 0.7}


class _Padded(AbstractTerm):
    """Gaussian rows of the radius in a fixed-size buffer; ``valid`` marks live rows."""

    centers: jax.Array
    valid: jax.Array
    reads = ()

    def __call__(self, model):
        z = (model.planets[0].Rp_Rearth - self.centers) / WIDTH
        return -0.5 * jnp.sum(jnp.where(self.valid, z**2, 0.0))

    def draw(self, key, model):
        return self


def _empty():
    return _Padded(jnp.zeros(3), jnp.zeros(3, bool))


def _with_row(center):
    return _Padded(jnp.array([center, 0.0, 0.0]), jnp.array([True, False, False]))


def _padded_fit(center=1.5):
    return Fit(
        _scene(),
        {KEY: Uniform(0.0, HIGH)},
        (_Gauss(jnp.asarray(center)), _empty()),
    )


def _stack(a, b):
    return jax.tree.map(lambda x, y: jnp.stack([x, y]), a, b)


@pytest.fixture(scope="module")
def cloud():
    """A 4000-draw importance cloud of the padded fit (the padding is invalid)."""
    return run(_padded_fit(), Laplace(n_draws=4000), jax.random.key(0))


def test_fit_replace_term_edits_one_slot_and_refuses_a_bad_index():
    """Only the named term changes; an index outside the tuple raises."""
    fit = _padded_fit()
    new = fit.replace_term(1, _with_row(1.8))
    assert new.terms[0] is fit.terms[0] and new.terms[1] is not fit.terms[1]
    assert len(new.terms) == 2 and new.free is fit.free
    for bad in (2, -1):
        with pytest.raises(IndexError, match="outside range"):
            fit.replace_term(bad, _with_row(1.8))


def test_replace_term_of_a_padded_row_equals_append_of_the_row(cloud):
    """Writing a row into the padding equals appending that row to an unpadded fit.

    Truth source: ``append_term`` of ``_Gauss(1.8)`` on the same cloud viewed as a
    fit with the one real term (the invalid padding contributes exactly zero).
    Tolerance basis: rounding only; the two ratios are the same numbers up to the
    subtraction of an exact zero, so 1e-12 relative.
    Fails if: the old term is dropped from the ratio, the gate reads the ratio alone,
    or the evidence increment differs from ``append_term``'s.
    """
    swapped = cloud.replace_term(1, _with_row(1.8), **GATE)
    plain = eqx.tree_at(
        lambda p: p.fit,
        cloud,
        Fit(cloud.fit.model, cloud.fit.free, cloud.fit.terms[:1]),
    )
    appended = plain.append_term(_Gauss(jnp.asarray(1.8)), **GATE)
    assert np.all(np.isfinite(swapped.log_weights))
    for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
        np.testing.assert_allclose(
            getattr(swapped, name), getattr(appended, name), rtol=1e-12, atol=1e-12
        )
    _check_target(swapped)
    assert len(swapped.fit.terms) == 2 and swapped.kind == "monte_carlo"


def test_replace_term_removes_the_old_row_and_gates_on_the_combined_weights(cloud):
    """Replacing a live row by another equals the exact ratio of the two densities.

    Truth source: weights ``w * exp(new - old)`` computed directly from the draws.
    Fails if: ``replace_term`` adds the new row without removing the old one.
    """
    first = cloud.replace_term(1, _with_row(1.8), **GATE)
    second = first.replace_term(1, _with_row(1.6), **GATE)
    r = np.asarray(cloud.samples[KEY])[:, 0]
    log_ratio = -0.5 * ((r - 1.6) / WIDTH) ** 2
    want = np.asarray(cloud.log_weights) + log_ratio
    want = want - np.log(np.sum(np.exp(want)))
    np.testing.assert_allclose(second.log_weights, want, rtol=0, atol=1e-10)
    _check_target(second)


@pytest.mark.medium
def test_the_refusal_is_per_slice_and_update_refits_only_where_refused():
    """Two worlds, one row written: world a keeps the cloud, world b is refitted.

    World a has its posterior at 1.5 and b at 3.5; the new row at 1.8 keeps all of a
    and almost nothing of b, so ``replace_term`` refuses b alone and ``update`` runs
    the backend on b's new fit while a keeps its reweighted cloud.
    Medium tier: it compiles the Laplace refit under a world vmap and a ``lax.map``.
    ``update`` returns the gate reading of ``replace_term`` in both outcomes, under the
    vmap and under ``lax.map`` over three worlds (one ``(3,)`` leaf per field).
    Fails if: the gate raises on a traced value, one decision covers the stack, the
    refit is not the posterior of the new fit, or the refit drops the gate reading.
    """
    backend = Laplace(n_draws=4000)
    a = run(_padded_fit(1.5), backend, jax.random.key(0))
    b = run(_padded_fit(3.5), backend, jax.random.key(1))
    stack = _stack(a, b)
    row = _with_row(1.8)

    def swap(p):
        return p.replace_term(1, row, **GATE)

    out = eqx.filter_vmap(swap)(stack)
    assert not np.any(np.isnan(out.log_weights[0]))
    assert np.all(np.isnan(out.log_weights[1]))
    assert np.isfinite(out.log_evidence[0]) and np.isnan(out.log_evidence[1])
    alone = swap(a)
    np.testing.assert_allclose(out.log_weights[0], alone.log_weights, rtol=1e-12)
    np.testing.assert_allclose(out.log_evidence[0], alone.log_evidence, rtol=1e-12)

    def refresh(p, key):
        return update(p, row, backend, key, index=1, **GATE)

    keys = jax.random.split(jax.random.key(5), 2)
    fresh = eqx.filter_vmap(refresh)(stack, keys)
    assert fresh.log_weights.shape == (2, 4000)
    np.testing.assert_array_equal(np.asarray(out.info.refused), [False, True])
    for field in ("n_eff", "khat", "refused"):
        np.testing.assert_array_equal(
            getattr(fresh.info, field), getattr(out.info, field)
        )
    three = jax.tree.map(lambda x, y: jnp.stack([x[0], x[1], y]), stack, a)
    mapped = jax.lax.map(
        lambda pk: refresh(*pk), (three, jax.random.split(jax.random.key(5), 3))
    )
    for field in ("n_eff", "khat", "refused"):
        leaf = getattr(mapped.info, field)
        assert leaf.shape == (3,)
        np.testing.assert_allclose(leaf[:2], getattr(out.info, field), rtol=1e-12)
    np.testing.assert_allclose(fresh.log_weights[0], alone.log_weights, rtol=1e-12)
    np.testing.assert_allclose(fresh.log_evidence[0], alone.log_evidence, rtol=1e-12)
    assert np.all(np.isfinite(fresh.log_weights[1]))
    own = refresh(b, keys[1])
    np.testing.assert_allclose(fresh.log_weights[1], own.log_weights, rtol=1e-10)
    np.testing.assert_allclose(fresh.log_evidence[1], own.log_evidence, rtol=1e-10)
    # The refit is the posterior of both narrow terms: N(2.65, WIDTH / sqrt 2).
    w = np.exp(np.asarray(fresh.log_weights[1]))
    r = np.asarray(fresh.samples[KEY][1])[:, 0]
    assert abs(np.sum(w * r) - 2.65) < 0.03
    _check_target(own)


def test_update_appends_without_an_index_and_keeps_a_passing_cloud(cloud):
    """With no index ``update`` is ``append_term`` when the gate passes.

    Both return the gate reading: on the passing slice it is the Kish size and k-hat of
    the returned weights. A refused ``append_term`` (a row far from the cloud) has NaN
    weights and the reading of the same call with no gate.
    Fails if: the backend replaces a cloud that passed, ``append_term`` is not used,
    or the reading is not the gate's (taken after the refusal, or dropped).
    """
    kept = update(
        cloud,
        _Gauss(jnp.asarray(1.6)),
        Laplace(n_draws=4000),
        jax.random.key(1),
        **GATE,
    )
    direct = cloud.append_term(_Gauss(jnp.asarray(1.6)), **GATE)
    np.testing.assert_array_equal(kept.log_weights, direct.log_weights)
    np.testing.assert_array_equal(kept.samples[KEY], cloud.samples[KEY])
    assert len(kept.fit.terms) == 3 and isinstance(kept.info, ReweightInfo)
    assert not bool(kept.info.refused)
    np.testing.assert_array_equal(kept.info.n_eff, direct.info.n_eff)
    np.testing.assert_array_equal(kept.info.khat, direct.info.khat)
    np.testing.assert_allclose(kept.info.n_eff, kish_ess(kept.log_weights), rtol=1e-12)
    np.testing.assert_allclose(
        kept.info.khat, pareto_khat(kept.log_weights), rtol=1e-12
    )
    far = _Gauss(jnp.asarray(3.0))
    refused = cloud.append_term(far, **GATE)
    ungated = cloud.append_term(far, min_n_eff=0.0, max_khat=jnp.inf)
    assert np.all(np.isnan(refused.log_weights)) and bool(refused.info.refused)
    assert not bool(ungated.info.refused)
    np.testing.assert_array_equal(refused.info.n_eff, ungated.info.n_eff)
    np.testing.assert_array_equal(refused.info.khat, ungated.info.khat)


class _NanAbove(_Gauss):
    """Gaussian that is NaN for radii above 1.5."""

    def __call__(self, model):
        return jnp.where(
            model.planets[0].Rp_Rearth[0] > 1.5, jnp.nan, super().__call__(model)
        )


def test_update_on_a_quadrature_posterior_reads_the_real_refusal():
    """``info.refused`` is true exactly where a quadrature slice has a NaN weight.

    Fails if: the quadrature branch hardcodes ``refused`` to false, so a refused
    slice reads as passed although ``update`` refits it.
    """
    grid = run(
        Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_Gauss(jnp.asarray(1.5)),)),
        Grid({KEY: midpoint(0.0, HIGH, 200)}),
        jax.random.key(0),
    )
    for term, expected in ((_Gauss(jnp.asarray(1.6)), False), (_NanAbove(1.5), True)):
        out = update(
            grid,
            term,
            Grid({KEY: midpoint(0.0, HIGH, 200)}),
            jax.random.key(1),
            min_n_eff=None,
            max_khat=None,
        )
        assert bool(out.info.refused) == expected
        assert bool(out.info.refused) == bool(np.any(np.isnan(out.log_weights)))


def test_update_refuses_a_refit_with_another_layout(cloud):
    """A backend whose draw count differs from the cloud's raises on the host."""
    with pytest.raises(ValueError, match="layout"):
        update(
            cloud,
            _with_row(1.8),
            Laplace(n_draws=1000),
            jax.random.key(1),
            index=1,
            **GATE,
        )


def test_a_nan_gate_limit_is_refused_on_the_host(cloud):
    """NaN would switch the k-hat test off (or refuse every slice): raise instead."""
    row = _with_row(1.8)
    nan = float("nan")
    with pytest.raises(ValueError, match="max_khat is NaN"):
        cloud.append_term(row, min_n_eff=0, max_khat=nan)
    with pytest.raises(ValueError, match="max_khat is NaN"):
        cloud.replace_term(1, row, min_n_eff=0, max_khat=jnp.asarray(nan))
    with pytest.raises(ValueError, match="min_n_eff is NaN"):
        cloud.with_priors({KEY: Uniform(0.0, HIGH)}, min_n_eff=nan, max_khat=1.0)
    with pytest.raises(ValueError, match="max_khat is NaN"):
        update(
            cloud,
            row,
            Laplace(n_draws=4000),
            jax.random.key(0),
            max_khat=nan,
            min_n_eff=0,
            index=1,
        )
    ok = cloud.append_term(row, min_n_eff=0, max_khat=jnp.inf)
    assert np.all(np.isfinite(ok.log_weights))
    # Inside a jit the limits are traced or constant: no host check, no failure.
    jitted = eqx.filter_jit(lambda t: cloud.replace_term(1, t, **GATE).log_weights)
    np.testing.assert_allclose(
        jitted(row), cloud.replace_term(1, row, **GATE).log_weights, rtol=1e-12
    )


def test_batch_size_gives_the_same_results_as_all_at_once(cloud):
    """Chunked ``lax.map`` over draws equals the vmapped result to rounding.

    4000 draws in chunks of 1000 (divides) and 777 (leaves a remainder).
    Tolerance basis: 1e-12 relative; only the reduction order of independent per-draw
    values can differ, and each draw is evaluated by the same code.
    Fails if: the chunking drops or repeats draws, or reorders them.
    """
    row = _with_row(1.8)
    prior = {KEY: Uniform(0.0, HIGH - 0.5)}
    cases = {
        "with_priors": lambda **kw: cloud.with_priors(prior, **GATE, **kw),
        "append_term": lambda **kw: cloud.append_term(
            _Gauss(jnp.asarray(1.7)), **GATE, **kw
        ),
        "replace_term": lambda **kw: cloud.replace_term(1, row, **GATE, **kw),
        "update": lambda **kw: update(
            cloud, row, Laplace(n_draws=4000), jax.random.key(2), index=1, **GATE, **kw
        ),
    }
    for name, make in cases.items():
        want = make()
        for size in (1000, 777):
            got = make(batch_size=size)
            for field in (
                "log_weights",
                "log_target",
                "log_evidence",
                "log_evidence_se",
            ):
                np.testing.assert_allclose(
                    getattr(got, field),
                    getattr(want, field),
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"{name} {field} batch_size={size}",
                )
    fit, x = cloud.fit, cloud.samples
    for method in (fit.log_prior, fit.log_likelihood, fit.log_density):
        np.testing.assert_allclose(
            method(x, batch_size=777), method(x), rtol=1e-12, atol=1e-12
        )


@pytest.mark.medium
def test_eig_batch_size_gives_the_same_estimate():
    """Both EIG estimators agree with and without ``batch_size`` (same key).

    Medium tier: four estimator variants compile (about 5 s).
    Tolerance basis: 1e-12 relative on the estimate and the bias scale; the draws are
    the same, so only the reduction order differs.
    Fails if: the chunked evaluation scores a different set of models.
    """
    post, term, _ = _gaussian_channel(n=300)
    key = jax.random.key(3)
    for size in (64,):
        a = eig_contrastive(post, term, key, n_outer=40)
        b = eig_contrastive(post, term, key, n_outer=40, batch_size=size)
        c = eig_nested(post, term, key, n_outer=40, n_inner=16, min_n_eff=10.0)
        d = eig_nested(
            post, term, key, n_outer=40, n_inner=16, min_n_eff=10.0, batch_size=size
        )
        for x, y in ((a, b), (c, d)):
            np.testing.assert_allclose(x.eig, y.eig, rtol=1e-12)
            np.testing.assert_allclose(x.bias_scale, y.bias_scale, rtol=1e-12)
            np.testing.assert_allclose(x.n_eff, y.n_eff, rtol=1e-12)
