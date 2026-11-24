"""Posterior.with_priors across a change of support (an orbit law with other bounds).

An orbit block is always mapped, so a law with other size bounds is another map and
the stored coordinates change meaning. The swap re-expresses each draw through the new
block's inverse and weights it by the law ratio on the orbit.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbix.system import EccentricityVectorChart, OrbitLaw
from priorities import Beta, LogUniform, Uniform
from test_fit import _block, _scene

from photomancy.infer import AbstractTerm, Fit, Posterior
from photomancy.infer.evidence import importance_evidence

jax.config.update("jax_enable_x64", True)

KEY = "dynamics.orbits"
N = 20000
GATE = dict(min_n_eff=0.0, max_khat=jnp.inf)


def _law(low, high):
    return OrbitLaw(Uniform(0.0, 0.8), a_AU=LogUniform(low, high))


class _SizeTerm(AbstractTerm):
    """Gaussian log likelihood of the semi-major axis around 1.5 AU."""

    reads = ()

    def __call__(self, model):
        a = model.dynamics.orbits.a_AU[0]
        return -0.5 * ((a - 1.5) / 0.4) ** 2

    def draw(self, key, model):
        return self


def _fit(low, high, terms=()):
    return Fit(_scene(), {KEY: _block(_law(low, high))}, terms)


def _cloud(fit, n=N, seed=0):
    """Importance cloud of ``fit`` proposed from its prior."""
    x = fit.sample_prior(jax.random.key(seed), (n,))
    loglike = fit.log_likelihood(x)
    log_z, se = importance_evidence(loglike)
    return Posterior(
        samples=x,
        log_weights=loglike - jax.scipy.special.logsumexp(loglike),
        log_target=fit.log_density(x),
        log_evidence=log_z,
        log_evidence_se=se,
        fit=fit,
        info=None,
        kind="monte_carlo",
    )


def _check_target(post):
    np.testing.assert_allclose(
        np.asarray(post.log_target),
        np.asarray(post.fit.log_density(post.samples)),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.medium(reason="compiles the re-expression and its Jacobian, 8 s cold")
def test_widening_is_refused_and_the_walled_mass_is_read_from_the_info():
    """LogUniform(0.8, 3) to LogUniform(0.5, 6) is refused; the mass is 0.5319.

    Truth source: with no term the cloud is the old prior, so the new law's mass on
    the old support is ln(3 / 0.8) / ln(6 / 0.5) = 0.5319 (the walled block's mean
    ratio), read from ``info.log_mass_ratio``; the narrowing swap back is accepted and
    a draw outside the narrower support has weight zero.
    Tolerance basis: exact-algebra, 1e-9 in the log (the ratio is the same constant at
    every draw for log-uniform laws, so the Monte Carlo error is zero).
    Fails if: the support test is missing (the evidence of the swap is reported as the
    new fit's), ``log_mass_ratio`` is NaN on refusal, the Jacobian of the
    re-expression is dropped or doubled, or a draw is left in the old coordinates.
    """
    cloud = _cloud(_fit(0.8, 3.0))
    out = cloud.with_priors({KEY: _block(_law(0.5, 6.0))}, **GATE)
    assert bool(out.info.refused) and not bool(out.info.support_covered)
    assert np.all(np.isnan(out.log_weights)) and np.isnan(out.log_evidence)
    truth = np.log(np.log(3.0 / 0.8) / np.log(6.0 / 0.5))
    np.testing.assert_allclose(out.info.log_mass_ratio, truth, atol=1e-9)
    wide = _cloud(_fit(0.5, 6.0))
    back = wide.with_priors({KEY: _block(_law(0.8, 3.0))}, **GATE)
    assert bool(back.info.support_covered) and not bool(back.info.refused)
    a_old = jax.vmap(lambda x: wide.fit.model_at(x).dynamics.orbits.a_AU[0])(
        wide.samples
    )
    outside = np.asarray((a_old < 0.8) | (a_old > 3.0))
    assert outside.any() and not outside.all()
    assert np.all(np.isneginf(np.asarray(back.log_weights)[outside]))
    assert np.all(np.isfinite(np.asarray(back.log_weights)[~outside]))


@pytest.mark.medium(reason="compiles the re-expression twice, 8 s cold")
def test_narrowing_gives_minus_inf_outside_and_matches_a_refit():
    """Narrowing the law zeroes the draws it excludes and keeps the evidence exact.

    Truth source: a direct prior-sampling estimate of the evidence under the narrow
    law (the refit), with the size term.
    Tolerance basis: 3 combined standard errors of the swap and the refit.
    Dead draws carry NaN coordinates (visibly dead), the live ones are finite, and an
    ``append_term`` on the swapped cloud is not refused by the dead draws.
    Fails if: out-of-support draws keep weight or finite wall coordinates, the
    evidence ignores the cut, or a dead draw's NaN refuses a later operation.
    """
    term = (_SizeTerm(),)
    wide = _fit(0.5, 6.0, term)
    narrow_block = _block(_law(1.0, 2.0))
    out = _cloud(wide).with_priors({KEY: narrow_block}, **GATE)
    a_old = jax.vmap(lambda x: wide.model_at(x).dynamics.orbits.a_AU[0])(
        _cloud(wide).samples
    )
    outside = np.asarray((a_old < 1.0) | (a_old > 2.0))
    assert outside.any() and not outside.all()
    assert np.all(np.isneginf(np.asarray(out.log_weights)[outside]))
    assert np.all(np.isfinite(np.asarray(out.log_weights)[~outside]))
    assert np.all(np.isneginf(np.asarray(out.log_target)[outside]))
    for v in out.samples[KEY].values():
        assert np.isnan(np.asarray(v)[outside]).all()
        assert np.isfinite(np.asarray(v)[~outside]).all()
    appended = out.append_term(_SizeTerm(), min_n_eff=0.0, max_khat=jnp.inf)
    assert np.isfinite(float(appended.log_evidence))
    assert np.all(np.isneginf(np.asarray(appended.log_weights)[outside]))
    assert np.all(np.isfinite(np.asarray(appended.log_weights)[~outside]))
    ref = _cloud(Fit(_scene(), {KEY: narrow_block}, term), seed=5)
    se = np.hypot(float(out.log_evidence_se), float(ref.log_evidence_se))
    assert abs(float(out.log_evidence) - float(ref.log_evidence)) < 3 * se
    kept = ~outside
    np.testing.assert_allclose(
        np.asarray(out.log_target)[kept],
        np.asarray(out.fit.log_density(out.samples))[kept],
        rtol=1e-10,
    )


def test_a_chart_change_is_refused_by_name():
    """Another chart is not a prior swap: the layout differs.

    Fails if: a chart change is treated as a support change (the stored coordinates
    would then be read as another chart's).
    """
    cloud = _cloud(_fit(0.8, 3.0))
    other = _block(_law(0.8, 3.0), chart=EccentricityVectorChart())
    with pytest.raises(ValueError, match="same layout"):
        cloud.with_priors({KEY: other}, **GATE)


def test_an_identity_swap_changes_nothing():
    """The same block gives the same weights, evidence, error and draws.

    Tolerance basis: rtol 1e-12 (the identity path runs no re-expression).
    """
    cloud = _cloud(_fit(0.5, 6.0, (_SizeTerm(),)))
    out = cloud.with_priors({KEY: cloud.fit.free[KEY]}, **GATE)
    for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
        np.testing.assert_allclose(
            getattr(out, name), getattr(cloud, name), rtol=1e-12, atol=1e-12
        )
    for name, v in cloud.samples[KEY].items():
        np.testing.assert_array_equal(out.samples[KEY][name], v)


def test_a_shape_change_inside_the_same_bounds_is_the_plain_path():
    """A law that changes shape inside the same bounds keeps the stored coordinates.

    Fails if: a shape change were routed through the re-expression.
    """
    flat = OrbitLaw(Uniform(0.0, 1.0), a_AU=LogUniform(0.8, 3.0))
    shaped = _block(OrbitLaw(Beta(1.0, 3.0), a_AU=LogUniform(0.8, 3.0)))
    cloud = _cloud(Fit(_scene(), {KEY: _block(flat)}))
    out = cloud.with_priors({KEY: shaped}, **GATE)
    for name, v in cloud.samples[KEY].items():
        np.testing.assert_array_equal(out.samples[KEY][name], v)
    _check_target(out)


def test_the_gate_still_refuses_a_swap_that_leaves_few_draws():
    """The gate runs on the combined weights across a support change."""
    cloud = _cloud(_fit(0.5, 6.0))
    out = cloud.with_priors(
        {KEY: _block(_law(1.4, 1.45))}, min_n_eff=2000.0, max_khat=jnp.inf
    )
    assert np.all(np.isnan(out.log_weights)) and np.isnan(out.log_evidence)


@pytest.mark.medium(reason="compiles the swap batched and per world, 14 s cold")
def test_a_swap_over_two_worlds_equals_the_loop():
    """``filter_vmap`` over stacked worlds equals the per-world loop.

    World 0 is widened by the swap and refused, world 1 is narrowed and kept.
    Tolerance basis: rtol 1e-9 (the same arithmetic, batched).
    Fails if: one uncovered world raises on the host or refuses its neighbor.
    """
    clouds = [
        _cloud(_fit(low, high), n=2000, seed=seed)
        for seed, (low, high) in ((1, (0.8, 3.0)), (2, (0.3, 8.0)))
    ]
    new = _block(_law(0.5, 6.0))

    def swap(c):
        return c.with_priors({KEY: new}, **GATE)

    stack = jax.tree.map(lambda *a: jnp.stack(a), *clouds)
    batched = eqx.filter_vmap(swap)(stack)
    np.testing.assert_array_equal(batched.info.refused, [True, False])
    np.testing.assert_array_equal(batched.info.support_covered, [False, True])
    for i, cloud in enumerate(clouds):
        one = swap(cloud)
        np.testing.assert_allclose(
            batched.log_weights[i], one.log_weights, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(batched.log_evidence[i], one.log_evidence, rtol=1e-9)
        for name, v in one.samples[KEY].items():
            np.testing.assert_allclose(
                batched.samples[KEY][name][i], v, rtol=1e-9, atol=1e-9
            )
