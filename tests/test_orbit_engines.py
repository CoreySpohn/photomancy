"""Orbit importance engines: completion, scale-and-rotate and the two backends.

One tiny scene (one planet, a Thiele-Innes block, four relative astrometry records) is
shared by every test; the structures compiled are the completed sample, the vmapped
completed run and the two backends.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.model_paths import resolve  # noqa: F401
from orbix.system import OrbitLaw, System, ThieleInnesBlock
from orbix.system.core import Body, SkyFrame
from priorities import LogUniform, Uniform
from skyscapes.population import kipping_eccentricity

from photomancy.infer import AbstractTerm, AdaptiveImportance, Fit, Importance, run
from photomancy.measurements import AstrometryData, RelativeAstrometryTerm
from photomancy.orbit.backends import any_epoch, few_epoch
from photomancy.orbit.completion import (
    CompletedProposal,
    Completion,
    _log_reference,
)
from photomancy.orbit.proposals import ScaleRotateProposal, ShapePriorProposal

jax.config.update("jax_enable_x64", True)

TIMES = np.array([31.0, 120.0, 290.0, 470.0])
KEY = jax.random.key(0)


def _system():
    return System.from_classical(
        Body(jnp.asarray(1.0)),
        Body(jnp.array([0.0])),
        SkyFrame(jnp.asarray(10.0)),
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.1]),
        i_rad=jnp.array([1.0]),
        W_rad=jnp.array([1.0]),
        w_rad=jnp.array([2.0]),
        M0_rad=jnp.array([1.0]),
    )


def _block(system=""):
    # a uniform mass law: for a log-uniform one, log p(M) + log M is constant and hides
    # a wrong enclosed mass in the proposal density
    law = OrbitLaw(
        period_d=LogUniform(50.0, 5000.0), eccentricity=kipping_eccentricity(0.9)
    )
    return ThieleInnesBlock(law, Uniform(0.9, 1.1), rows=(0,), system=system)


def _term(n_rows=4, err=0.004, shift=0.0, system_path=""):
    data = AstrometryData.from_sep_pa(
        time_d=TIMES[:n_rows],
        sep=np.array([0.13, 0.11, 0.09, 0.15])[:n_rows] + shift,
        sep_err=[err] * n_rows,
        pa_deg=[45.0, 120.0, 210.0, 310.0][:n_rows],
        pa_err_deg=[1.0] * n_rows,
        max_n=4,
    )
    return RelativeAstrometryTerm(data, system_path=system_path)


def _fit(term=None):
    return Fit(_system(), {"orbits": _block()}, (term or _term(),))


@pytest.fixture(scope="module")
def fit():
    """The shared four-row fit."""
    return _fit()


@pytest.mark.medium(reason="one conditional and one fit density per draw, 12 s cold")
def test_completion_identity_pins_the_linear_form(fit):
    """Fails if the form or the conditional disagrees with the fit's own density.

    The fit density at random prior draws equals the density without the term minus the
    reference Gaussian plus ``log Z`` plus the conditional Gaussian of the constants, to
    rtol 1e-9; the form is in arcsec so no scale constant enters.
    """
    comp = Completion.from_fit(fit)
    x = fit.sample_prior(jax.random.key(0), (3,))
    shape, c = comp.split(fit.to_flat(x))
    for i in range(3):
        mean, chol, log_z = comp.conditional(shape[i])
        diff = chol.T @ (c[i] - mean)
        log_cond = (
            -0.5 * jnp.sum(diff**2)
            + jnp.sum(jnp.log(jnp.diag(chol)))
            - 2.0 * jnp.log(2 * jnp.pi)
        )
        xi = jax.tree.map(lambda a, i=i: a[i], x)
        rest = comp.fit_rest.log_density(xi) - _log_reference(c[i], comp.scale_AU)
        np.testing.assert_allclose(
            float(rest + log_z + log_cond), float(fit.log_density(xi)), rtol=1e-9
        )
    assert float(comp.scale_AU) == pytest.approx(10.0 * 0.15, rel=1e-6)


def _unscreened(comp, shape, key, n_pool):
    """The estimator with no screen, every constant scored: ``(z_kept, lw)``.

    The completion of a fit with no term left to score must equal it draw for draw.
    """
    mean, chol, log_z = comp.conditional(shape)
    k_draw, k_keep = jax.random.split(key)
    eps = jax.random.normal(k_draw, (n_pool, 4))
    c = mean + jax.scipy.linalg.solve_triangular(chol.T, eps.T, lower=False).T
    z = comp.join(shape, c)
    x, volume = comp.fit_rest.from_flat(z)
    rest = comp.fit_rest.log_density(x) + volume - _log_reference(c, comp.scale_AU)
    keep = jnp.argmax(rest + jax.random.gumbel(k_keep, (n_pool,)))
    return z[keep], log_z + jax.scipy.special.logsumexp(rest) - jnp.log(n_pool)


@pytest.mark.medium(reason="compiles the completed sample once, 15 s cold")
def test_completed_weight_is_the_generic_weight(fit):
    """Fails if a completed weight or its reported density is not the generic one.

    The log weight of a kept draw is ``log Z + log mean rest - log q(shape)``, and the
    density ``weigh`` reports from the form equals the fit's own density at the draw,
    so the generic importance weight ``fit.log_density + volume - log_q`` and the
    completed estimate agree. With no term left to score the default pool is 8 and the
    completion is the unscreened estimator draw for draw (same key): weights and kept
    draws to 1e-12 (compiled against eager rounding), so the screen cannot change the
    astrometry-only path.
    """
    comp = Completion.from_fit(fit)
    proposal = CompletedProposal(ShapePriorProposal())
    z, log_w, log_target, _ = eqx.filter_jit(proposal.weigh)(jax.random.key(0), fit, 32)
    x, _ = fit.from_flat(z)
    np.testing.assert_allclose(log_target, fit.log_density(x), rtol=1e-9)
    k_base, k_c = jax.random.split(jax.random.key(0))
    shape, log_q_shape = ShapePriorProposal().sample(k_base, comp, 32)
    z_c, lw, _, _ = comp.complete(k_c, shape, 8, 4)
    np.testing.assert_allclose(log_w, lw - log_q_shape, rtol=1e-9)
    keys = jax.random.split(k_c, 32)
    z_ref, lw_ref = jax.vmap(lambda t, k: _unscreened(comp, t, k, 8))(shape, keys)
    np.testing.assert_allclose(lw, lw_ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(z_c, z_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z, z_ref, rtol=1e-12, atol=1e-12)


class _MassTerm(AbstractTerm):
    """A Gaussian on the star mass: no linear form, so the completion screens for it."""

    reads: ClassVar[tuple[str, ...]] = ()

    def __call__(self, model):
        return -0.5 * ((model.star.mass_Msun - 1.02) / 0.03) ** 2

    def draw(self, key, model):
        raise NotImplementedError


@pytest.mark.medium(reason="2000 keys at four shapes on one compiled stage, 12 s cold")
def test_the_screened_weight_is_unbiased(fit):
    """Fails if the screen biases the marginal weight of ``shape``.

    A term with no linear form (a Gaussian on the star mass, which the constants and
    the period fix) is left to score. At the four prior shapes with the most live
    constants, the mean of ``exp(lw)`` over 2000 keys at a pool of 8 with 2 scored
    equals the unscreened estimate (16 keys of 4096 constants) within 4 combined
    standard errors (N = 2000 for a 4 SE near 0.05 of the mean at the weight spread
    here). A pool with no live member weighs zero, never NaN.
    """
    fit = Fit(fit.model, fit.free, (*fit.terms, _MassTerm()))
    comp = Completion.from_fit(fit)
    # the four prior shapes whose pools are most alive, and one with a dead pool
    shape, _ = ShapePriorProposal().sample(jax.random.key(1), comp, 400)
    live = jax.vmap(lambda t: jnp.mean(jnp.isfinite(comp._constants(t, KEY, 64)[2])))(
        shape
    )
    order = np.argsort(-np.asarray(live))
    shape = shape[np.concatenate([order[:4], order[-1:]])]
    keys = jax.random.split(jax.random.key(2), 2000)
    lw = eqx.filter_jit(jax.vmap(lambda k: comp.complete(k, shape, 8, 2)[1]))(keys)
    one = jax.vmap(lambda t, k: _unscreened(comp, t, k, 4096)[1], (0, None))
    ref = eqx.filter_jit(jax.vmap(one, (None, 0)))(shape, keys[:16])
    lw, ref = np.asarray(lw), np.asarray(ref)
    assert not np.any(np.isnan(lw))
    live = np.isfinite(ref[0])
    assert live.sum() == 4
    assert np.all(np.isneginf(lw[:, ~live]))
    top = ref[:, live].max(0)
    w_ref = np.exp(ref[:, live] - top)
    ratio = np.exp(lw[:, live] - top) / w_ref.mean(0)
    se = np.hypot(
        ratio.std(0, ddof=1) / np.sqrt(len(keys)),
        w_ref.std(0, ddof=1) / np.sqrt(len(w_ref)) / w_ref.mean(0),
    )
    assert np.all(np.abs(ratio.mean(0) - 1.0) < 4.0 * se)


class _NanTerm(AbstractTerm):
    """Zero, but NaN on a fine comb of star masses (one draw in 500)."""

    reads: ClassVar[tuple[str, ...]] = ()

    def __call__(self, model):
        comb = jnp.mod(model.star.mass_Msun * 1e6, 1.0)
        return jnp.where(comb < 0.002, jnp.nan, 0.0)

    def draw(self, key, model):
        raise NotImplementedError


@pytest.mark.medium(reason="one small any_epoch run on the scoring path, 10 s cold")
def test_a_nan_score_refuses_the_slice(fit):
    """Fails if a NaN term at a live draw is read as a zero weight.

    The term is NaN at scattered draws: at this key some scored draws are NaN while
    every kept draw is finite, so only the score can carry the NaN to the weights.
    The run of ``any_epoch`` must be a refused slice (every weight and the evidence
    NaN), never a finite cloud that dropped the broken draws as zero weights.
    """
    broken = Fit(fit.model, fit.free, (*fit.terms, _NanTerm()))
    post = run(
        broken,
        any_epoch(n_draws=2**12, n_adapt_draws=2**12, n_rounds=2, n_centers=64),
        KEY,
    )
    assert np.all(np.isnan(np.asarray(post.log_weights)))
    assert np.isnan(float(post.log_evidence))


def test_completion_refuses_a_term_on_another_system():
    """Fails if a term that reads another system than the block frees is completed.

    The form would be built at the wrong orbit; the refusal names the term's path.
    """

    class Model(eqx.Module):
        a: System
        b: System

    model = Model(_system(), _system())
    block = _block("a")
    term = _term(system_path="b")
    fit = Fit(model, {"a.orbits": block}, (term,))
    with pytest.raises(ValueError, match="'b'"):
        Completion.from_fit(fit)


@pytest.mark.medium(reason="4000 draws on two programs, 14 s cold")
def test_scale_rotate_density_is_normalized_and_matches_the_draws():
    """Fails if the Jacobian or the sheet factor of ``log_q`` is wrong.

    A broad anchor kernel (0.3 arcsec) makes the proposal cover much of the prior. The
    probability that a draw is inside the period bounds, from the proposal's own draws,
    equals the integral of its density estimated with prior draws, within 4 SE; and
    ``log_q`` of a draw equals ``log_prob`` of it. N = 4000 for a 4 SE of 0.04. A draw
    outside the period bounds is NaN with ``log_q = -inf``, and ``weigh`` gives it
    weight zero, so the evidence stays finite.
    """
    fit = _fit(_term(n_rows=1, err=0.3))
    proposal = ScaleRotateProposal()
    n = 4000
    z, log_q = eqx.filter_jit(proposal.sample)(jax.random.key(0), fit, n)
    inside = np.isfinite(np.asarray(log_q))
    ok = np.where(inside)[0][:200]
    np.testing.assert_allclose(
        proposal.log_prob(z[ok], fit), log_q[ok], rtol=1e-8, atol=1e-8
    )
    assert 0 < inside.sum() < n
    assert np.all(np.isnan(np.asarray(z)[~inside][:, :3]))
    draws = few_epoch(n_draws=n)._run(fit, jax.random.key(0))
    np.testing.assert_array_equal(np.isneginf(draws.log_weights), ~inside)
    assert np.isfinite(float(draws.log_evidence))
    p_draws = inside.mean()
    se_draws = np.sqrt(p_draws * (1 - p_draws) / n)
    xp = fit.sample_prior(jax.random.key(1), (n,))
    zp = fit.to_flat(xp)
    log_w = proposal.log_prob(zp, fit) - fit.log_prior(xp)
    w = np.exp(np.asarray(log_w))
    se_prior = w.std() / np.sqrt(n)
    assert abs(w.mean() - p_draws) < 4 * np.hypot(se_draws, se_prior)


@pytest.mark.medium(reason="both backends and one vmapped run, 21 s cold")
def test_any_epoch_and_few_epoch_agree_and_vmap_equals_the_loop():
    """Fails if the engines disagree on the evidence of one row, or vmap differs.

    With one anchor row the scale-and-rotate weights are constant, so ``few_epoch`` is
    exact; ``any_epoch`` must agree within 4 combined SE. The log weights of the
    completed run over three worlds under ``filter_vmap`` equal the per-world loop at
    rtol 1e-9 (shared key). The third world has no valid row, so no data scale: its
    slice is refused (NaN weights) in both, and the other two keep live weights.
    ``any_epoch(n_draws)`` sizes the returned cloud only: the adaptation rounds keep
    their own size, so the two can be passed together. The pool of one anchor row is
    starved (under half effective), so the default ``any_epoch`` publishes no evidence
    there; the agreement is read with the pool condition switched off.
    """
    sizes = any_epoch(n_draws=4000, n_adapt_draws=2**16)
    assert (sizes.n_draws, sizes.n_adapt_draws) == (4000, 2**16)
    assert any_epoch().n_draws == any_epoch().n_adapt_draws == 2**16
    fit = _fit(_term(n_rows=1))
    exact = few_epoch(n_draws=2**12)._run(fit, jax.random.key(0))
    settings = {"n_draws": 2**12, "n_adapt_draws": 2**12, "n_rounds": 2}
    gated = any_epoch(**settings, n_centers=64)._run(fit, jax.random.key(0))
    assert not bool(gated.info.proposal_valid)
    assert np.isnan(float(gated.log_evidence))
    assert not np.any(np.isnan(gated.log_weights))
    assert np.any(np.isfinite(gated.log_weights))
    adaptive = AdaptiveImportance(
        CompletedProposal(ShapePriorProposal(), min_pool_fraction=0.0),
        **settings,
        n_centers=64,
    )._run(fit, jax.random.key(0))
    gap = abs(float(exact.log_evidence) - float(adaptive.log_evidence))
    assert gap < 4 * float(adaptive.log_evidence_se) + 1e-6

    imp = Importance(CompletedProposal(ShapePriorProposal()), 256)
    empty = eqx.tree_at(lambda t: t.data.is_valid, _term(), jnp.zeros(4, bool))
    fits = [_fit(_term(shift=0.0)), _fit(_term(shift=0.01)), _fit(empty)]
    dyn, static = eqx.partition(fits, eqx.is_array)
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *dyn)
    key = jax.random.key(0)
    batched = eqx.filter_vmap(
        lambda d: imp._run(eqx.combine(d, static[0]), key).log_weights
    )(stacked)
    loop = [imp._run(f, key).log_weights for f in fits]
    np.testing.assert_allclose(batched, jnp.stack(loop), rtol=1e-9)
    assert np.all(np.isfinite(batched[:2]).any(-1)) and np.isnan(batched[2]).all()


def test_any_epoch_refits_with_over_dispersed_kernels_in_every_round():
    """The bandwidth of ``ShapePriorProposal`` reaches ``from_cloud`` in every round.

    Truth source: the kernel covariance scales with the squared bandwidth, so a
    bandwidth of 2 gives 4 times the covariance of 1 on the same cloud and key (rtol
    1e-6, the diagonal jitter). The second refit goes through the adapted
    ``DefensiveProposal``, not the shape proposal. Fails if: the factor is not passed to
    ``from_cloud`` or is dropped after round one (the refit falls back to Silverman).
    """

    class _Fit:
        def periods(self):
            return (None, None, None)

    z = jax.random.normal(KEY, (512, 3))
    lw = jnp.zeros(512)
    kw = {"n_centers": 32, "defensive_fraction": 0.1}
    key = jax.random.key(1)
    assert any_epoch().proposal.base.bandwidth == 2.0

    def covariances(bandwidth):
        first = ShapePriorProposal(bandwidth).adapt(key, z, lw, _Fit(), **kw)
        second = first.adapt(key, z, lw, _Fit(), **kw)
        return first.base.chols, second.base.chols

    narrow, wide = covariances(1.0), covariances(2.0)
    for n, w in zip(narrow, wide, strict=True):
        np.testing.assert_allclose(w @ w.mT, 4.0 * (n @ n.mT), rtol=1e-6)


@pytest.mark.medium(reason="one completed sample on two thresholds, 15 s cold")
def test_the_pool_reading_gates_the_evidence_and_matches_a_direct_one(fit):
    """Fails if the pool fraction is not the Kish size of the screen over the pool.

    The direct reading takes ``(sum s)**2 / sum s**2`` from the screen of each shape
    (``Completion._constants`` at the draw key of ``complete``) and averages it over
    the draws with a finite weight, unweighted; ``CompletedProposal.weigh`` must agree
    on the verdict at a threshold just below and just above that reading. When the
    verdict is false the ``Importance`` backend returns finite weights and a NaN
    evidence.
    """
    comp = Completion.from_fit(fit)
    n = 256
    k_base, k_c = jax.random.split(jax.random.key(3))
    shape, log_q = ShapePriorProposal().sample(k_base, comp, n)
    _, lw, _, kish = comp.complete(k_c, shape, 8, 4)
    keys = jax.random.split(k_c, n)
    screens = jax.vmap(lambda t, k: comp._constants(t, jax.random.split(k)[0], 8)[2])(
        shape, keys
    )
    direct = np.exp(
        2 * jax.scipy.special.logsumexp(screens, -1)
        - jax.scipy.special.logsumexp(2 * screens, -1)
    )
    np.testing.assert_allclose(
        np.where(np.isfinite(lw), kish, 0.0),
        np.where(np.isfinite(lw), direct, 0.0),
        rtol=1e-9,
    )
    live = np.isfinite(np.asarray(lw - log_q))
    reading = float(np.mean(np.asarray(kish)[live]) / 8)
    for threshold, expected in ((reading * 0.99, True), (reading * 1.01, False)):
        proposal = CompletedProposal(ShapePriorProposal(), min_pool_fraction=threshold)
        *_, valid = eqx.filter_jit(proposal.weigh)(jax.random.key(3), fit, n)
        assert bool(valid) == expected
    draws = Importance(proposal, n)._run(fit, jax.random.key(3))
    assert not bool(draws.info.proposal_valid)
    assert np.isnan(float(draws.log_evidence))
    assert np.all(np.isfinite(draws.log_weights) | np.isneginf(draws.log_weights))
