"""Staged MAP backend: groups, rescaled transforms and the staged fit loop."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from photomancy.backends.staged import (
    UNLABELED,
    FitStage,
    group_labels,
    map_fit,
    rescaled_transform,
)


class _Tree(eqx.Module):
    a: jax.Array
    b: jax.Array
    c: jax.Array
    n: int = 3


def _tree():
    return _Tree(a=jnp.arange(3.0), b=jnp.ones(2), c=jnp.asarray(2.0))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_group_labels_follow_the_selectors_and_leave_the_rest_unlabeled():
    """Labels are exactly the selected leaves; an unselected float leaf stays unlabeled.

    Truth source: the selectors written in the test.
    Tolerance basis: exact-algebra (labels are strings, compared exactly).
    Fails if: a leaf is labeled by default (``is_inexact_array`` fallback), a group
    misses part of a multi-leaf selector, or the python-int field gets a label.
    """
    labels = group_labels(
        _tree(), {"first": lambda t: t.a, "second": lambda t: (t.b, t.c)}
    )
    assert jax.tree.leaves(labels) == ["first", "second", "second", UNLABELED]
    lonely = group_labels(_tree(), {"first": lambda t: t.a})
    assert jax.tree.leaves(lonely) == ["first", UNLABELED, UNLABELED, UNLABELED]


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({"x": lambda t: t.a, "y": lambda t: (t.b, t.a)}, "claimed by both"),
        ({"x": lambda t: t.n}, "not an inexact array"),
        ({"": lambda t: t.a}, "non-empty"),
    ],
)
def test_group_labels_reject_double_claims_and_non_float_leaves(groups, message):
    """A doubly claimed leaf, a non-float leaf or an empty name raises.

    Truth source: the contract in the ``group_labels`` docstring.
    Tolerance basis: exact-algebra (an exception, or none).
    Fails if: an overlap silently lets the last group win (a leaf then moves under
    the wrong learning rate), or a python int is handed to the optimizer.
    """
    with pytest.raises(ValueError, match=message):
        group_labels(_tree(), groups)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_rescaled_transform_is_adam_on_the_scaled_variable():
    """Three steps equal Adam run directly on ``q = p / s``, mapped back by ``s``.

    Truth source: ``optax.adam`` on ``q`` with the gradient ``s * dL/dp``, the
    definition of a fit in the scaled units.
    Tolerance basis: floating-point (rtol 1e-12, float64; the algebra is exact).
    Fails if: the gradient or the update is not scaled, so epsilon acts in leaf
    units and the steps differ from a fit in meters and radians.
    """
    s = jnp.asarray([1e9, 2e9, 3e9])
    p = jnp.asarray([4.0, -5.0, 6.0])
    grads = [jnp.asarray([1e-9, -2e-9, 3e-9]) * (k + 1) for k in range(3)]
    base = optax.adam(0.01)
    wrapped = rescaled_transform(base, s)
    state, q_state = wrapped.init(p), base.init(p / s)
    q = p / s
    for g in grads:
        update, state = wrapped.update(g, state, p)
        p = p + update
        q_update, q_state = base.update(g * s, q_state, q)
        q = q + q_update
    np.testing.assert_allclose(p, q * s, rtol=1e-12)


class _Toy(eqx.Module):
    a: jax.Array
    b: jax.Array
    fixed: jax.Array


def _toy_init():
    return _Toy(
        a=jnp.asarray([0.0, 0.5, -1.0]), b=jnp.asarray([2.0, -1.0]), fixed=jnp.ones(2)
    )


_TOY_GROUPS = {"a": lambda t: t.a, "b": lambda t: t.b}


def _toy_logdensity(p):
    """Coupled, nonlinear toy: maximum at a = (1, 2, 3), b = (0.5, -0.5)."""
    target_a, target_b = jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([0.5, -0.5])
    coupling = jnp.sum(p.fixed) * jnp.sum(jnp.sin(p.a[:2] * p.b))
    return -jnp.sum((p.a - target_a) ** 2) - jnp.sum((p.b - target_b) ** 2) - coupling


def _toy_transforms():
    return {"a": optax.adam(0.05), "b": optax.adam(0.02)}


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_hold_leaves_frozen_leaves_and_moments_bitwise_unchanged():
    """A group that is not free keeps its values and Adam moments, bit for bit.

    Truth source: the definition of ``freeze="hold"`` (a group outside ``free`` is
    not touched). Frozen state is read from the optimizer state after each stage.
    Tolerance basis: exact-algebra (bitwise equality).
    Fails if: a frozen group is updated on stale momentum or its moments keep
    accumulating gradients while frozen (the failure of zeroed-gradient freezing).
    """
    init = _toy_init()
    labels = group_labels(init, _TOY_GROUPS)
    kwargs = dict(labels=labels, transforms=_toy_transforms(), freeze="hold")
    after_a = map_fit(_toy_logdensity, init, stages=[FitStage(20, ("a",))], **kwargs)
    np.testing.assert_array_equal(after_a.params.b, init.b)
    np.testing.assert_array_equal(after_a.params.fixed, init.fixed)
    fresh_b_state = _toy_transforms()["b"].init(init.b)
    for got, want in zip(
        jax.tree.leaves(after_a.opt_states["b"]),
        jax.tree.leaves(fresh_b_state),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)
    assert not np.array_equal(after_a.params.a, init.a)

    both = map_fit(
        _toy_logdensity,
        init,
        stages=[FitStage(20, ("a",)), FitStage(15, ("b",)), FitStage(10, ("a",))],
        **kwargs,
    )
    mid = map_fit(
        _toy_logdensity,
        init,
        stages=[FitStage(20, ("a",)), FitStage(15, ("b",))],
        **kwargs,
    )
    # Stage three freezes b, so it equals b after stage two; the trace lengths follow.
    np.testing.assert_array_equal(both.params.b, mid.params.b)
    assert [t.n_steps for t in both.stages] == [20, 15, 10]
    assert [t.start for t in both.stages] == [0, 20, 35]
    assert not np.array_equal(both.params.a, mid.params.a)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_map_fit_does_not_consume_init_and_reaches_the_optimum():
    """``init`` stays valid after a fit and a long fit lands on the toy maximum.

    Truth source: the stationary point of the toy density with the coupling term
    switched off (``fixed = 0``), a = (1, 2, 3) and b = (0.5, -0.5).
    Tolerance basis: floating-point (atol 1e-3; Adam at lr 0.05 on a quadratic, run
    to a gradient below 1e-4, not a roundoff floor).
    Fails if: donation invalidates the caller's arrays, or the minimized sign is
    wrong (the fit would walk away from the maximum).
    """
    init = eqx.tree_at(lambda t: t.fixed, _toy_init(), jnp.zeros(2))
    labels = group_labels(init, _TOY_GROUPS)
    result = map_fit(
        _toy_logdensity,
        init,
        labels,
        {"a": optax.adam(0.05), "b": optax.adam(0.05)},
        [FitStage(600, ("a", "b"))],
    )
    np.testing.assert_allclose(result.params.a, [1.0, 2.0, 3.0], atol=1e-3)
    np.testing.assert_allclose(result.params.b, [0.5, -0.5], atol=1e-3)
    assert np.isfinite(np.asarray(init.a)).all()
    assert result.stages[0].loss.shape == (600,)
    assert result.stages[0].loss[-1] < result.stages[0].loss[0]


_STAGES = [(12, ("a",)), (9, ("b",)), (7, ("a", "b"))]
_T_MAX = sum(n for n, _ in _STAGES)


def _cosine(base, floor):
    """Cosine decay from ``base`` to ``floor`` over the whole run, one shared clock."""
    return lambda t: floor + (base - floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * t / _T_MAX))


def _legacy_transforms():
    return {"a": optax.adam(_cosine(0.05, 0.05)), "b": optax.adam(_cosine(0.02, 0.0))}


def _hand_written_loop(init, policy):
    """One ``optax.multi_transform`` loop with dynamic masks, freezing by ``policy``.

    This is the structure of a staged fit written without the backend: one optimizer,
    one state, one schedule clock, and a 0/1 mask per group chosen by the stage.
    """
    optimizer = optax.multi_transform(_legacy_transforms(), {"a": "a", "b": "b"})
    params = {"a": init.a, "b": init.b}
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, mask):
        def loss_fn(q):
            return -_toy_logdensity(_Toy(q["a"], q["b"], init.fixed))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        if policy == "zero_gradient":
            grads = jax.tree.map(lambda g, m: g * m, grads, mask)
            updates, state = optimizer.update(grads, state, params)
        else:
            updates, state = optimizer.update(grads, state, params)
            updates = jax.tree.map(lambda u, m: u * m, updates, mask)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    for n, free in _STAGES:
        mask = {g: jnp.asarray(1.0 if g in free else 0.0) for g in ("a", "b")}
        for _ in range(n):
            params, state, loss = step(params, state, mask)
            losses.append(float(loss))
    return params, np.asarray(losses)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("policy", ["zero_gradient", "drop_update"])
def test_legacy_freeze_policies_match_a_hand_written_optax_loop(policy):
    """Three stages under each legacy policy equal the hand-written loop, and repeat.

    Truth source: a separate implementation, one ``optax.multi_transform`` loop with
    float masks, a cosine schedule on one clock across the stages and a loss traced
    per step (independent code, so this is code verification, not validation).
    Tolerance basis: floating-point (rtol 1e-12, float64; the same operations in the
    same order, so the observed difference is far below it).
    Fails if: a policy freezes with the wrong convention (``zero_gradient`` must keep
    the frozen group drifting on momentum, ``drop_update`` must not), the clock
    restarts at a stage boundary, or the loss trace is offset by a step.
    """
    init = _toy_init()
    labels = group_labels(init, _TOY_GROUPS)
    stages = [FitStage(n, free) for n, free in _STAGES]
    got = map_fit(
        _toy_logdensity, init, labels, _legacy_transforms(), stages, freeze=policy
    )
    want_params, want_loss = _hand_written_loop(init, policy)
    np.testing.assert_allclose(got.params.a, want_params["a"], rtol=1e-12)
    np.testing.assert_allclose(got.params.b, want_params["b"], rtol=1e-12)
    np.testing.assert_allclose(
        np.concatenate([t.loss for t in got.stages]), want_loss, rtol=1e-12
    )
    again = map_fit(
        _toy_logdensity, init, labels, _legacy_transforms(), stages, freeze=policy
    )
    np.testing.assert_array_equal(again.params.a, got.params.a)
    np.testing.assert_array_equal(again.params.b, got.params.b)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_freeze_policies_differ_where_they_should():
    """A later stage that frees ``b`` moves ``a`` only under ``zero_gradient``.

    Truth source: the three definitions in the ``map_fit`` docstring. ``hold`` and
    ``drop_update`` keep ``a`` bitwise; ``drop_update`` still advances ``a``'s Adam
    moments while ``hold`` does not; ``zero_gradient`` moves ``a`` on momentum.
    Tolerance basis: exact-algebra (bitwise comparisons).
    Fails if: the negative control is lost, so the equivalence test above could pass
    with the policies swapped.
    """
    init = _toy_init()
    labels = group_labels(init, _TOY_GROUPS)
    one = [FitStage(10, ("a",))]
    two = [*one, FitStage(5, ("b",))]
    runs = {}
    for policy in ("hold", "zero_gradient", "drop_update"):
        args = (_toy_logdensity, init, labels, _toy_transforms())
        runs[policy] = (
            map_fit(*args, one, freeze=policy),
            map_fit(*args, two, freeze=policy),
        )
    for policy in ("hold", "drop_update"):
        np.testing.assert_array_equal(
            runs[policy][1].params.a, runs[policy][0].params.a
        )
    assert not np.array_equal(
        runs["zero_gradient"][1].params.a, runs["zero_gradient"][0].params.a
    )
    first_moment = lambda r: jax.tree.leaves(r[1].opt_states["a"])[0]  # noqa: E731
    np.testing.assert_array_equal(
        first_moment(runs["hold"]), first_moment((None, runs["hold"][0]))
    )
    assert not np.array_equal(
        first_moment(runs["drop_update"]), first_moment((None, runs["drop_update"][0]))
    )
