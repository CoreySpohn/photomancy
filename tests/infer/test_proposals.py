"""Proposals, periods, tempering and the Laplace and Pathfinder importance route.

Truth sources are closed forms (a periodic density is invariant under a shift by one
period, a conjugate toy has a known evidence, the conditional effective sample size of a
tempering step is computed directly) and the card 5 outputs of the previous
``Laplace`` and ``Pathfinder`` (``tests/data/laplace_pathfinder_golden.json``).
"""

import json
import math
from pathlib import Path

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
    run,
)
from photomancy.infer.diagnostics import next_temperature

jax.config.update("jax_enable_x64", True)

GOLDEN = json.loads(
    (
        Path(__file__).parent.parent / "data" / "laplace_pathfinder_golden.json"
    ).read_text()
)
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


def test_from_cloud_unwraps_a_cloud_across_the_cell_edge():
    """A cloud straddling the cell edge gets its true spread, not a cell-wide one.

    Truth source: the cloud is a normal of sd 0.1 around the edge of the angle cell, so
    the kernel sd is well below one radian. Fails if: the moments are taken without the
    circular unwrap, which inflates the covariance to the order of the period.
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


def test_next_temperature_hits_the_target_conditional_ess():
    """The step lands where the conditional ESS fraction equals the target.

    Truth source: the fraction ``(sum a) ** 2 / (N sum a ** 2)`` for equal weights and
    ``a = exp(delta * loglik)`` is evaluated directly at the returned step; tolerance
    1e-6 (the bisection runs to machine precision). A full step is returned as 1
    when the target is met at beta = 1.
    """
    loglik = -jnp.linspace(0.0, 200.0, 2000)
    lw = jnp.zeros(2000)
    beta = float(next_temperature(loglik, lw, 0.0, 0.5))
    a = jnp.exp(beta * loglik)
    assert 0.0 < beta < 1.0
    assert abs(float(jnp.sum(a) ** 2 / (2000 * jnp.sum(a**2))) - 0.5) < 1e-6
    assert float(next_temperature(0.001 * loglik, lw, 0.0, 0.5)) == 1.0
    assert float(next_temperature(loglik, lw, 1.0, 0.5)) == 1.0
    assert math.isnan(float(next_temperature(loglik.at[3].set(jnp.nan), lw, 0.0, 0.5)))


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


def _agreement(label, backend):
    """Per-seed log Z, mean and sd of a backend against the card 5 outputs."""
    fit = _frame_fit()
    names = GOLDEN["names"]
    for seed, old in enumerate(GOLDEN[label]):
        post = run(fit, backend, jax.random.key(seed))
        block = post.samples["dynamics.orbits"]
        z = np.stack([np.asarray(block[n])[:, 0] for n in names], -1)
        w = np.exp(np.asarray(post.log_weights) - float(jnp.max(post.log_weights)))
        w = w / w.sum()
        mean = w @ z
        sd = np.sqrt(w @ (z - mean) ** 2)
        se = math.hypot(float(post.log_evidence_se), old["se"])
        assert abs(float(post.log_evidence) - old["log_z"]) < 4 * se, (seed, label)
        np.testing.assert_allclose(
            mean, old["mean"], atol=0.1 * np.asarray(old["sd"]).max()
        )
        np.testing.assert_allclose(sd, old["sd"], rtol=0.2, atol=1e-3)


@pytest.mark.slow
def test_laplace_matches_the_previous_backend_on_the_frame_chart_orbit():
    """``Laplace`` as importance over a mixture reproduces the old per-seed outputs.

    Truth source: the card 5 outputs of the old ``Laplace`` (six seeds) on the golden
    scene with a ``FrameChart`` block, which has no angle coordinate, so the wrapped
    and plain densities coincide. Tolerance basis: per-seed log Z within 4 combined
    standard errors; means within 0.1 of the largest posterior sd, sd within 20 percent.
    Measured: identical to rounding on all six seeds.
    """
    _agreement("laplace", Laplace(n_draws=20000, n_modes=4, min_eigenvalue=1.0))


@pytest.mark.slow
def test_pathfinder_matches_the_previous_backend_on_the_frame_chart_orbit():
    """``Pathfinder`` through the mixture route reproduces the old per-seed outputs.

    Tolerance basis as for ``Laplace``. Measured: identical to rounding on all six
    seeds. The weights are heavy tailed (k-hat 0.8 to 1.04), which is a property of the
    fit and of the old output too.
    """
    _agreement("pathfinder", Pathfinder(n_draws=20000, n_modes=4))
