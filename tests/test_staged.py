"""Staged MAP backend: groups, rescaled transforms and the staged fit loop."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from photomancy.backends.staged import UNLABELED, group_labels, rescaled_transform


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
