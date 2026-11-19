"""Orbit importance engines: completion, scale-and-rotate and the two backends.

One tiny scene (one planet, a Thiele-Innes block, four relative astrometry records) is
shared by every test; the structures compiled are the completed sample, the vmapped
completed run and the two backends.
"""

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

from photomancy.infer import Fit, Importance
from photomancy.infer.backends._flat import draw_template
from photomancy.measurements import RelativeAstromData, RelativeAstrometryTerm
from photomancy.orbit.backends import collapsed, few_epoch
from photomancy.orbit.completion import (
    CdfMixture,
    CompletedProposal,
    Completion,
    ThetaPrior,
    _log_reference,
)
from photomancy.orbit.proposals import ScaleRotateProposal

jax.config.update("jax_enable_x64", True)

TIMES = np.array([31.0, 120.0, 290.0, 470.0])


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


def test_cdf_mixture_picks_follow_the_weights_and_skip_zero_weight():
    """Fails if the inverse-CDF pick is biased or draws a zero-weight component.

    Shares 0.2, 0.5, 0.3 with zero-weight components between and after them; N = 8192
    (4 sigma of 0.02 on the largest variance).
    """
    pi = np.array([0.2, 0.0, 0.5, 0.3, 0.0])
    with np.errstate(divide="ignore"):
        log_pi = jnp.log(jnp.asarray(pi))
    mixture = CdfMixture(jnp.zeros((5, 2)), jnp.eye(2), log_pi)
    n = 8192
    share = (
        np.bincount(np.asarray(mixture._pick(jax.random.key(0), n)), minlength=5) / n
    )
    assert share[1] == 0.0 and share[4] == 0.0
    assert np.all(np.abs(share - pi) <= 4.0 * np.sqrt(pi * (1.0 - pi) / n) + 1e-12)


@pytest.mark.medium(reason="compiles the completed sample once, 15 s cold")
def test_completed_weight_is_the_generic_weight(fit):
    """Fails if ``log_q`` of a completed draw is not the one that gives its weight.

    ``fit.log_density - log_q`` of the kept draw equals ``log Z + log mean rest -
    log q(theta)``, so the generic importance path and the completed estimate agree.
    """
    comp = Completion.from_fit(fit)
    proposal = CompletedProposal(ThetaPrior())
    z, log_q = eqx.filter_jit(proposal.sample)(jax.random.key(0), fit, 32)
    _, unravel = ravel_pytree(fit.to_real(draw_template(fit)))
    x, _ = fit.from_real(jax.vmap(unravel)(z))
    generic = fit.log_density(x) - log_q
    k_base, k_c = jax.random.split(jax.random.key(0))
    theta, log_q_theta = ThetaPrior().sample(k_base, comp.view, 32)
    _, lw, _ = comp.complete(k_c, theta, 8)
    np.testing.assert_allclose(generic, lw - log_q_theta, rtol=1e-9)


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
    ``log_q`` of a draw equals ``log_prob`` of it. N = 4000 for a 4 SE of 0.04.
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
    adaptive = collapsed(n_draws=2000, n_round=2**12, n_rounds=2, n_centers=64)._run(
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
