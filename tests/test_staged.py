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
