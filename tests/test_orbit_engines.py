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
from jax.flatten_util import ravel_pytree
from orbix.system import OrbitLaw, System, ThieleInnesBlock
from orbix.system.core import Body, SkyFrame
from priorities import LogUniform, Uniform
from skyscapes.population import kipping_eccentricity

from photomancy.infer import AbstractTerm, Fit, Importance, run
from photomancy.infer.backends._flat import draw_template
from photomancy.measurements import RelativeAstromData, RelativeAstrometryTerm
from photomancy.orbit.backends import collapsed, few_epoch
from photomancy.orbit.completion import (
    CompletedProposal,
    Completion,
    ThetaPrior,
    _log_reference,
)
from photomancy.orbit.proposals import ScaleRotateProposal

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
    data = RelativeAstromData.from_sep_pa(
        times=TIMES[:n_rows],
        sep=np.array([0.13, 0.11, 0.09, 0.15])[:n_rows] + shift,
        sep_err=[err] * n_rows,
        pa_deg=[45.0, 120.0, 210.0, 310.0][:n_rows],
        pa_err_deg=[1.0] * n_rows,
        max_n=4,
    )
    return RelativeAstrometryTerm(data, system_path=system_path)


def _fit(term=None):
    return Fit(_system(), {"orbits": _block()}, (term or _term(),))


def _flat(fit, x):
    return jax.vmap(lambda one: ravel_pytree(one)[0])(fit.to_real(x))


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
    theta, c = comp.split(_flat(fit, x))
    for i in range(3):
        mean, chol, log_z = comp.conditional(theta[i])
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


def _unscreened(comp, theta, key, n_constants):
    """The estimator with no screen, every constant scored: ``(z_kept, lw)``.

    The completion of a fit with no term left to score must equal it draw for draw.
    """
    mean, chol, log_z = comp.conditional(theta)
    k_draw, k_keep = jax.random.split(key)
    eps = jax.random.normal(k_draw, (n_constants, 4))
    c = mean + jax.scipy.linalg.solve_triangular(chol.T, eps.T, lower=False).T
    z = comp.join(theta, c)
    _, unravel = ravel_pytree(comp.fit_rest.to_real(draw_template(comp.fit_rest)))
    x, volume = comp.fit_rest.from_real(jax.vmap(unravel)(z))
    rest = comp.fit_rest.log_density(x) + volume - _log_reference(c, comp.scale_AU)
    keep = jnp.argmax(rest + jax.random.gumbel(k_keep, (n_constants,)))
    return z[keep], log_z + jax.scipy.special.logsumexp(rest) - jnp.log(n_constants)


@pytest.mark.medium(reason="compiles the completed sample once, 15 s cold")
def test_completed_weight_is_the_generic_weight(fit):
    """Fails if ``log_q`` of a completed draw is not the one that gives its weight.

    ``fit.log_density - log_q`` of the kept draw equals ``log Z + log mean rest -
    log q(theta)``, so the generic importance path and the completed estimate agree.
    With no term left to score the default pool is 8 and the completion is the
    unscreened estimator draw for draw (same key): weights and kept draws to 1e-12
    (compiled against eager rounding), so the screen cannot change the astrometry-only
    path.
    """
    comp = Completion.from_fit(fit)
    proposal = CompletedProposal(ThetaPrior())
    z, log_q = eqx.filter_jit(proposal.sample)(jax.random.key(0), fit, 32)
    _, unravel = ravel_pytree(fit.to_real(draw_template(fit)))
    x, _ = fit.from_real(jax.vmap(unravel)(z))
    generic = fit.log_density(x) - log_q
    k_base, k_c = jax.random.split(jax.random.key(0))
    theta, log_q_theta = ThetaPrior().sample(k_base, comp, 32)
    z_c, lw, _ = comp.complete(k_c, theta, 8, 4)
    np.testing.assert_allclose(generic, lw - log_q_theta, rtol=1e-9)
    keys = jax.random.split(k_c, 32)
    z_ref, lw_ref = jax.vmap(lambda t, k: _unscreened(comp, t, k, 8))(theta, keys)
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


@pytest.mark.medium(reason="2000 keys at four thetas on one compiled stage, 12 s cold")
def test_the_screened_weight_is_unbiased(fit):
    """Fails if the screen biases the marginal weight of ``theta``.

    A term with no linear form (a Gaussian on the star mass, which the constants and
    the period fix) is left to score. At the four prior thetas with the most live
    constants, the mean of ``exp(lw)`` over 2000 keys at a pool of 8 with 2 scored
    equals the unscreened estimate (16 keys of 4096 constants) within 4 combined
    standard errors (N = 2000 for a 4 SE near 0.05 of the mean at the weight spread
    here). A pool with no live member weighs zero, never NaN.
    """
    fit = Fit(fit.model, fit.free, (*fit.terms, _MassTerm()))
    comp = Completion.from_fit(fit)
    # the four prior thetas whose pools are most alive, and one with a dead pool
    theta, _ = ThetaPrior().sample(jax.random.key(1), comp, 400)
    live = jax.vmap(lambda t: jnp.mean(jnp.isfinite(comp._constants(t, KEY, 64)[2])))(
        theta
    )
    order = np.argsort(-np.asarray(live))
    theta = theta[np.concatenate([order[:4], order[-1:]])]
    keys = jax.random.split(jax.random.key(2), 2000)
    lw = eqx.filter_jit(jax.vmap(lambda k: comp.complete(k, theta, 8, 2)[1]))(keys)
    one = jax.vmap(lambda t, k: _unscreened(comp, t, k, 4096)[1], (0, None))
    ref = eqx.filter_jit(jax.vmap(one, (None, 0)))(theta, keys[:16])
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


@pytest.mark.medium(reason="one small collapsed run on the scoring path, 10 s cold")
def test_a_nan_score_refuses_the_slice(fit):
    """Fails if a NaN term at a live draw is read as a zero weight.

    The term is NaN at scattered draws: at this key some scored draws are NaN while
    every kept draw is finite, so only the score can carry the NaN to the weights.
    The run of ``collapsed`` must be a refused slice (every weight and the evidence
    NaN), never a finite cloud that dropped the broken draws as zero weights.
    """
    broken = Fit(fit.model, fit.free, (*fit.terms, _NanTerm()))
    post = run(broken, collapsed(n_round=2**12, n_rounds=2, n_centers=64), KEY)
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
    outside the period bounds is NaN with ``log_q = +inf``, and the importance backend
    weighs it zero, so the evidence stays finite.
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
    draws = few_epoch(n_proposals=n)._run(fit, jax.random.key(0))
    np.testing.assert_array_equal(np.isneginf(draws.log_weights), ~inside)
    assert np.isfinite(float(draws.log_evidence))
    p_draws = inside.mean()
    se_draws = np.sqrt(p_draws * (1 - p_draws) / n)
    xp = fit.sample_prior(jax.random.key(1), (n,))
    zp = _flat(fit, xp)
    log_w = proposal.log_prob(zp, fit) - fit.log_prior(xp)
    w = np.exp(np.asarray(log_w))
    se_prior = w.std() / np.sqrt(n)
    assert abs(w.mean() - p_draws) < 4 * np.hypot(se_draws, se_prior)


@pytest.mark.medium(reason="both backends and one vmapped run, 21 s cold")
def test_collapsed_and_few_epoch_agree_and_vmap_equals_the_loop():
    """Fails if the engines disagree on the evidence of one row, or vmap differs.

    With one anchor row the scale-and-rotate weights are constant, so ``few_epoch`` is
    exact; ``collapsed`` must agree within 4 combined SE. The completed run over two
    worlds under ``filter_vmap`` equals the per-world loop at rtol 1e-9 (shared key).
    """
    fit = _fit(_term(n_rows=1))
    exact = few_epoch(n_proposals=2**12)._run(fit, jax.random.key(0))
    adaptive = collapsed(n_round=2**12, n_rounds=2, n_centers=64)._run(
        fit, jax.random.key(0)
    )
    gap = abs(float(exact.log_evidence) - float(adaptive.log_evidence))
    assert gap < 4 * float(adaptive.log_evidence_se) + 1e-6

    imp = Importance(CompletedProposal(ThetaPrior()), 256)
    fits = [_fit(_term(shift=s)) for s in (0.0, 0.01)]
    dyn, static = eqx.partition(fits, eqx.is_array)
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *dyn)
    key = jax.random.key(0)
    batched = eqx.filter_vmap(
        lambda d: imp._run(eqx.combine(d, static[0]), key).log_evidence
    )(stacked)
    loop = [imp._run(f, key).log_evidence for f in fits]
    np.testing.assert_allclose(batched, jnp.stack(loop), rtol=1e-9)
