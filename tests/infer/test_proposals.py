"""Proposals, periods, tempering and the Laplace and Pathfinder importance route.

Truth sources are closed forms (a periodic density is invariant under a shift by one
period, a conjugate toy has a known evidence, the conditional effective sample size of a
tempering step is computed directly), the stored modes of ``Laplace`` and
``Pathfinder`` (``tests/data/laplace_pathfinder_modes.json``) and the k-hat contract of
the importance evidence.
"""

import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from orbix.system import FrameChart
from priorities import Uniform
from test_fit import _block, _relative, _scene

from photomancy.infer import (
    DefensiveProposal,
    Fit,
    Importance,
    Laplace,
    MixtureProposal,
    Pathfinder,
    PriorProposal,
    importance_evidence,
    khat_threshold,
    run,
)
from photomancy.infer.diagnostics import kish_temperature

jax.config.update("jax_enable_x64", True)

TWO_PI = 2.0 * math.pi


def _orbit_fit():
    return Fit(_scene(), {"dynamics.orbits": _block()}, ())


def _frame_fit():
    return Fit(
        _scene(),
        {"dynamics.orbits": _block(chart=FrameChart())},
        (_relative(),),
    )


def test_periods_follow_the_flat_vector_by_coordinate_name():
    """``Fit.periods`` has one entry per flat element: the period, or ``None``.

    Truth source: the block's own ``periods`` and ``coordinate_names``, sorted by name
    as the flat vector is; a fit with only a leaf law has no periodic coordinate.
    """
    fit = _orbit_fit()
    block = fit.free["dynamics.orbits"]
    by_name = dict(zip(block.coordinate_names, block.periods, strict=True))
    expected = tuple(by_name[n] for n in sorted(by_name))
    assert fit.periods() == expected
    assert any(p is not None for p in expected) and any(p is None for p in expected)
    leaf = Fit(_scene(), {"planets[0].Rp_Rearth": Uniform(0.5, 4.0)}, ())
    assert leaf.periods() == (None,)


def test_to_flat_inverts_from_flat_on_a_leaf_law_and_an_orbit_block():
    """``fit.from_flat(fit.to_flat(x))[0]`` is ``x``, with the batch axes kept.

    Truth source: the round trip itself, to 1e-12, on a leaf-law fit and on an
    orbit-block fit with periodic coordinates; an angle shifted by whole periods gives
    the flat vector of the stored angle. Fails if: the order of ``to_flat`` differs from
    that of ``unravel``, a periodic coordinate is not reduced to its cell, or a batch
    axis is lost.
    """
    leaf = Fit(_scene(), {"planets[0].Rp_Rearth": Uniform(0.5, 4.0)}, ())
    orbit = _orbit_fit()
    for fit in (leaf, orbit):
        x = fit.sample_prior(jax.random.key(0), (2, 3))
        z = fit.to_flat(x)
        assert z.shape == (2, 3, len(fit.periods()))
        back, _ = fit.from_flat(z)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12),
            back,
            x,
        )
    # x and z are now the orbit fit's, the last of the loop.
    block = orbit.free["dynamics.orbits"]
    shifted = dict(x["dynamics.orbits"])
    for name, period in zip(block.coordinate_names, block.periods, strict=True):
        if period is not None:
            shifted[name] = shifted[name] + 3 * period
    np.testing.assert_allclose(
        orbit.to_flat({"dynamics.orbits": shifted}), z, atol=1e-12
    )


def test_from_real_reduces_periodic_coordinates_to_one_cell():
    """A shifted angle is stored in ``[0, period)`` and has the same density.

    Fails if: ``Fit.from_real`` stores an angle outside its cell, or the density of the
    reduced point differs from the density of the original by more than rounding.
    """
    fit = _orbit_fit()
    key = jax.random.key(0)
    draw = fit.sample_prior(key)
    z, unravel = ravel_pytree(fit.to_real(draw))
    shift = jnp.asarray([3 * p if p is not None else 0.0 for p in fit.periods()])
    shifted, _ = fit.from_real(unravel(z + shift))
    block = fit.free["dynamics.orbits"]
    for name, period in zip(block.coordinate_names, block.periods, strict=True):
        if period is not None:
            value = np.asarray(shifted["dynamics.orbits"][name])
            assert value.min() >= 0.0 and value.max() < period
    plain, _ = fit.from_real(unravel(z))
    np.testing.assert_allclose(
        float(fit.log_density(shifted)), float(fit.log_density(plain)), rtol=1e-9
    )


def test_a_mixture_density_is_periodic_and_its_draws_match_log_q():
    """The wrapped density is unchanged by a shift of one period and matches ``sample``.

    Truth source: periodicity of the wrapped density (exact in the image sum up to the
    window); tolerance rtol 1e-9. Fails if: the density is not reduced to a cell first.
    """
    fit = _orbit_fit()
    d = len(fit.periods())
    mixture = MixtureProposal(
        jnp.zeros((2, d)).at[1].set(0.3),
        0.05 * jnp.eye(d),
        jnp.log(jnp.asarray([0.3, 0.7])),
    )
    z, log_q = mixture.sample(jax.random.key(1), fit, 64)
    np.testing.assert_allclose(
        np.asarray(mixture.log_prob(z, fit)), np.asarray(log_q), rtol=1e-9
    )
    shift = jnp.asarray([p if p is not None else 0.0 for p in fit.periods()])
    np.testing.assert_allclose(
        np.asarray(mixture.log_prob(z + shift, fit)), np.asarray(log_q), rtol=1e-9
    )


def test_mixture_picks_follow_the_weights_and_skip_zero_weight():
    """The inverse-CDF pick draws components in proportion to their weights.

    Shares 0.2, 0.5, 0.3 with zero-weight components between and after them, tagged by
    well separated means along one unwrapped coordinate; N = 8192 (4 sigma of 0.02 on
    the largest variance). Fails if the pick is biased or draws a zero-weight component.
    """
    fit = _orbit_fit()
    periods = fit.periods()
    d = len(periods)
    j = next(i for i, p in enumerate(periods) if p is None)
    pi = np.array([0.2, 0.0, 0.5, 0.3, 0.0])
    with np.errstate(divide="ignore"):
        log_pi = jnp.log(jnp.asarray(pi))
    means = jnp.zeros((5, d)).at[:, j].set(10.0 * jnp.arange(5.0))
    mixture = MixtureProposal(means, 1e-4 * jnp.eye(d), log_pi)
    n = 8192
    z, _ = mixture.sample(jax.random.key(0), fit, n)
    pick = np.rint(np.asarray(z[:, j]) / 10.0).astype(int)
    share = np.bincount(pick, minlength=5) / n
    assert share[1] == 0.0 and share[4] == 0.0
    assert np.all(np.abs(share - pi) <= 4.0 * np.sqrt(pi * (1.0 - pi) / n) + 1e-12)


def test_the_shared_covariance_path_equals_the_general_path():
    """One shared covariance gives the density of the same one tiled per component.

    Tolerance: rtol 1e-9 (the same sum by two evaluation orders).
    """
    fit = _orbit_fit()
    d = len(fit.periods())
    means = jax.random.normal(jax.random.key(2), (5, d)) * 0.5
    cov = 0.04 * jnp.eye(d) + 0.01
    shared = MixtureProposal(means, cov)
    tiled = MixtureProposal(means, jnp.tile(cov[None], (5, 1, 1)))
    z, _ = shared.sample(jax.random.key(3), fit, 40)
    np.testing.assert_allclose(
        np.asarray(shared.log_prob(z, fit)),
        np.asarray(tiled.log_prob(z, fit)),
        rtol=1e-9,
    )


def test_from_cloud_unwraps_the_cell_edge_and_skips_dead_draws():
    """A cloud straddling the cell edge gets its true spread, not a cell-wide one.

    Truth source: the cloud is a normal of sd 0.1 around the edge of the angle cell, so
    the kernel sd is well below one radian. The same cloud with NaN rows at weight
    ``-inf`` inserted (a proposal's dead draws) gives the same mixture to 1e-12. Fails
    if: the moments are taken without the circular unwrap, which inflates the
    covariance to the order of the period, or a dead draw reaches the moments (NaN
    Cholesky factors).
    """
    fit = _orbit_fit()
    periods = fit.periods()
    i = next(k for k, p in enumerate(periods) if p is not None)
    n, d = 4000, len(periods)
    x = jnp.zeros((n, d))
    x = x.at[:, i].set(
        jnp.mod(0.1 * jax.random.normal(jax.random.key(0), (n,)), TWO_PI)
    )
    mixture = MixtureProposal.from_cloud(
        jax.random.key(1), x, jnp.zeros(n), n_centers=32, periods=periods
    )
    sd = float(mixture.chols[0, i, i])
    assert 0.0 < sd < 0.2
    assert mixture.means.shape == (32, d)
    rows = jnp.asarray([0, 1000, n])
    dead_x = jnp.insert(x, rows, jnp.nan, axis=0)
    dead_lw = jnp.insert(jnp.zeros(n), rows, -jnp.inf)
    dead = MixtureProposal.from_cloud(
        jax.random.key(1), dead_x, dead_lw, n_centers=32, periods=periods
    )
    for got, want in ((dead.means, mixture.means), (dead.chols, mixture.chols)):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_a_defensive_proposal_is_the_prior_mixture_of_its_base():
    """The log density is ``log(f prior + (1 - f) base)`` and matches ``sample``.

    Tolerance: rtol 1e-9 (a closed-form logaddexp of the two densities).
    """
    fit = _orbit_fit()
    d = len(fit.periods())
    base = MixtureProposal(jnp.zeros((1, d)), 0.05 * jnp.eye(d))
    proposal = DefensiveProposal(base, 0.2)
    z, log_q = proposal.sample(jax.random.key(0), fit, 50)
    expected = jnp.logaddexp(
        math.log(0.2) + PriorProposal().log_prob(z, fit),
        math.log(0.8) + base.log_prob(z, fit),
    )
    np.testing.assert_allclose(np.asarray(log_q), np.asarray(expected), rtol=1e-9)


def test_kish_temperature_hits_the_target_kish_size_of_the_cloud():
    """The rung lands where the Kish size of the tempered cloud is the target.

    Truth source: the Kish size ``(sum w) ** 2 / sum w ** 2`` of ``exp(b * loglik)``
    (a base of zero, equal prior and proposal, so 2000 at ``b = 0``) evaluated at the
    returned ``b``; tolerance 1e-3 relative (bisection to machine precision, the size
    is smooth). A full step returns 1, a floor above the current size does not move,
    and NaN propagates. Fails if the criterion measures the move from the previous
    temperature rather than the cloud the next proposal is fitted from, or ignores
    the floor.
    """
    loglik = -jnp.linspace(0.0, 200.0, 2000)
    base = jnp.zeros(2000)
    beta = float(kish_temperature(base, loglik, 0.0, 0.5, 100.0))
    a = jnp.exp(beta * loglik)
    assert 0.0 < beta < 1.0
    np.testing.assert_allclose(
        float(jnp.sum(a) ** 2 / jnp.sum(a**2)), 1000.0, rtol=1e-3
    )
    floored = float(kish_temperature(base, loglik, 0.0, 0.5, 1500.0))
    b = jnp.exp(floored * loglik)
    np.testing.assert_allclose(
        float(jnp.sum(b) ** 2 / jnp.sum(b**2)), 1500.0, rtol=1e-3
    )
    assert float(kish_temperature(base, 0.001 * loglik, 0.0, 0.5)) == 1.0
    assert float(kish_temperature(base, loglik, 1.0, 0.5)) == 1.0
    assert float(kish_temperature(base, loglik, 0.0, 0.5, 2001.0)) < 1e-6
    nan = loglik.at[3].set(jnp.nan)
    assert math.isnan(float(kish_temperature(base, nan, 0.0, 0.5)))
    dead = base.at[:1500].set(-jnp.inf)
    assert float(kish_temperature(dead, 0.001 * loglik, 0.0, 0.5)) == 1.0


def test_a_batch_size_gives_the_same_importance_run():
    """Evaluating the density in blocks of draws equals evaluating it at once.

    Tolerance: rtol 1e-9 (the same terms summed per draw in a different batching).
    """
    fit = _frame_fit()
    d = len(fit.periods())
    proposal = MixtureProposal(jnp.zeros((1, d)), 0.5 * jnp.eye(d))
    key = jax.random.key(0)
    whole = run(fit, Importance(proposal, n_draws=100), key)
    blocks = run(fit, Importance(proposal, n_draws=100), key, batch_size=16)
    np.testing.assert_allclose(
        np.asarray(blocks.log_weights), np.asarray(whole.log_weights), rtol=1e-9
    )


def test_the_unit_cube_of_a_block_has_the_block_unit_size():
    """``unit_template`` and ``from_unit`` walk the block chain.

    Truth source: the block's own ``unit_size``; the stored draw at the cube center has
    a finite density. Fails if: ``from_unit`` leaves a block entry unwritten.
    """
    fit = _orbit_fit()
    block = fit.free["dynamics.orbits"]
    template = fit.unit_template()
    assert template["dynamics.orbits"].shape == (block.unit_size,)
    stored = fit.from_unit(jax.tree.map(lambda u: u + 0.5, template))
    assert bool(jnp.isfinite(fit.log_density(stored)))


MODES = json.loads(
    (
        Path(__file__).parent.parent / "data" / "laplace_pathfinder_modes.json"
    ).read_text()
)


def _modes_and_evidence(label, backend):
    """Stored modes on four seeds, and the k-hat contract of the published evidence."""
    fit = _frame_fit()
    threshold = khat_threshold(backend.n_draws)
    for seed, modes in enumerate(MODES[label][:4]):
        draws = backend._run(fit, jax.random.key(seed))
        info = draws.info
        khat = float(info.importance.khat)
        if khat <= threshold:
            log_z, se = importance_evidence(draws.log_weights)
            np.testing.assert_allclose(draws.log_evidence, log_z, rtol=1e-12)
            np.testing.assert_allclose(draws.log_evidence_se, se, rtol=1e-12)
        else:
            assert np.isnan(float(draws.log_evidence)), (seed, label, khat)
            assert np.isnan(float(draws.log_evidence_se)), (seed, label, khat)
        score = np.asarray(info.mode_log_evidence if label == "laplace" else info.elbo)
        kept = np.asarray(info.kept)
        order = [i for i in np.argsort(-score) if kept[i]]
        assert len(order) == len(modes), (seed, label)
        for i, mode in zip(order, modes, strict=True):
            np.testing.assert_allclose(score[i], mode["score"], atol=1e-2)
            np.testing.assert_allclose(info.means[i], mode["mean"], atol=1e-3)
            np.testing.assert_allclose(
                np.diagonal(np.asarray(info.covs[i])), mode["var"], rtol=1e-2, atol=1e-8
            )


@pytest.mark.medium(reason="four seeds of four L-BFGS modes on the frame-chart orbit")
def test_laplace_keeps_its_modes_and_refuses_the_evidence_above_the_khat_limit():
    """``Laplace`` on the frame-chart orbit keeps its stored modes, and F2 holds.

    Truth source: the stored modes of ``Laplace`` on the golden scene with a
    ``FrameChart`` block, which has no angle coordinate. The kept modes, sorted by
    evidence, match the stored ones: score within 0.01, means within 1e-3, variances
    within 1 percent (floor 1e-8, the stored rounding); they do not read the draws, so
    a small ``n_draws`` keeps the test cheap. The published evidence and its error are
    NaN where ``info.importance.khat`` is above ``khat_threshold(n_draws)`` and equal
    ``importance_evidence`` of the returned weights (rtol 1e-12) where it is not.
    Fails if: a mode moves, a mode is kept or dropped differently, or an evidence is
    published past the k-hat limit or differs from the importance mean.
    """
    _modes_and_evidence("laplace", Laplace(n_draws=2000, n_modes=4, min_eigenvalue=1.0))


@pytest.mark.medium(
    reason="four seeds of four Pathfinder paths on the frame-chart orbit"
)
def test_pathfinder_keeps_its_paths_and_refuses_the_evidence_above_the_khat_limit():
    """``Pathfinder`` keeps its stored paths (sorted by ELBO), and F2 holds.

    Tolerances and the evidence contract as for ``Laplace``. The weights of this fit
    are heavy tailed (k-hat 0.95 to 1.08 at 2,000 draws), so the evidence is refused
    on all four seeds; the finite branch is exercised by the ``Laplace`` test.
    Fails if: a path moves, or an evidence is published past the k-hat limit or
    differs from the importance mean.
    """
    _modes_and_evidence("pathfinder", Pathfinder(n_draws=2000, n_modes=4))


class _Reporting(PriorProposal):
    """A prior proposal that reports its weights valid or invalid as told."""

    reports: bool = eqx.field(static=True, default=True)

    def weigh(self, key, fit, n, *, batch_size=None):
        z, log_w, log_target, _ = super().weigh(key, fit, n, batch_size=batch_size)
        return z, log_w, log_target, jnp.asarray(self.reports)


@pytest.mark.parametrize("reports", [True, False])
def test_the_evidence_needs_the_proposal_to_report_its_weights_valid(reports):
    """Fails if ``valid`` from ``weigh`` does not gate the importance evidence.

    A scene whose likelihood is mild enough that the k-hat passes: with ``valid`` true
    the evidence is the importance mean of the weights; with it false the evidence and
    its error are NaN while the weights are the same finite values, and
    ``info.proposal_valid`` carries the proposal's verdict.
    """
    fit = _orbit_fit()
    draws = Importance(_Reporting(reports=reports), 512)._run(fit, jax.random.key(0))
    assert bool(draws.info.proposal_valid) == reports
    assert np.all(np.isfinite(draws.log_weights))
    if reports:
        assert float(draws.info.khat) <= khat_threshold(512)
        assert np.isfinite(float(draws.log_evidence))
    else:
        assert np.isnan(float(draws.log_evidence))
        assert np.isnan(float(draws.log_evidence_se))
