"""Staged MAP fitting over a PyTree: parameter groups, stages and per-group optimizers.

A fit is a sequence of stages. Each stage frees some named parameter groups and
runs a fixed number of optimizer steps on a log-density; the next stage continues
from the current values, and, unless it brings its own transforms, from the current
optimizer state. This module holds the pieces a staged fit is described with:

- :func:`group_labels` names the groups of a parameter PyTree from explicit selectors,
- :func:`rescaled_transform` runs a transform on ``params / scale`` so a leaf keeps
  its physical units while the steps match a fit in other units.
"""

import equinox as eqx
import jax
import optax

__all__ = ["UNLABELED", "group_labels", "rescaled_transform"]

UNLABELED = ""
"""Label of a leaf that belongs to no group (never optimized)."""


def group_labels(tree, groups):
    """Label every leaf of ``tree`` with the name of the group that claims it.

    Each group is an explicit selector ``where(tree) -> leaf | subtree | sequence of
    them``, the same form ``equinox.tree_at`` takes. The groups never default to
    "all inexact arrays": a leaf that no selector names is labeled
    :data:`UNLABELED` and is not optimized, so a fixed array that happens to be
    floating point is never fitted by accident.

    Args:
        tree: The parameter PyTree.
        groups: ``{name: where}``. Names are non-empty strings.

    Returns:
        A PyTree with the structure of ``tree`` whose leaves are the group names
        (strings, or :data:`UNLABELED`).

    Raises:
        ValueError: If a selector claims a leaf another selector already claimed, a
            leaf that is not an inexact array, or a name is empty.
    """
    leaves, treedef = jax.tree.flatten(tree)
    labels = [UNLABELED] * len(leaves)
    all_false = jax.tree.map(lambda _: False, tree)
    for name, where in groups.items():
        if not isinstance(name, str) or name == UNLABELED:
            raise ValueError(f"group names are non-empty strings, got {name!r}")
        marks = eqx.tree_at(
            where, all_false, replace_fn=lambda sub: jax.tree.map(lambda _: True, sub)
        )
        claimed = jax.tree.leaves(marks)
        for k, (flag, leaf) in enumerate(zip(claimed, leaves, strict=True)):
            if not flag:
                continue
            if labels[k] != UNLABELED:
                raise ValueError(
                    f"leaf {k} is claimed by both group {labels[k]!r} and {name!r}"
                )
            if not eqx.is_inexact_array(leaf):
                raise ValueError(
                    f"group {name!r} claims leaf {k}, which is not an inexact array "
                    f"({type(leaf).__name__})"
                )
            labels[k] = name
    return jax.tree.unflatten(treedef, labels)


def _scaled(tree, scale, op):
    """Apply ``op(x, s)`` leafwise; ``scale`` is a scalar or matches ``tree``."""
    if jax.tree.structure(scale) == jax.tree.structure(tree):
        return jax.tree.map(op, tree, scale)
    return jax.tree.map(lambda x: op(x, scale), tree)


def rescaled_transform(transform, scale):
    """Run ``transform`` on ``params / scale`` while the leaves keep their units.

    The returned transform multiplies the gradient by ``scale``, applies
    ``transform`` in the scaled variable ``q = params / scale``, and multiplies the
    resulting update by ``scale``. For Adam this means the step size and epsilon act
    on ``q``: a fit that keeps an optical path difference in nanometers and a tilt in
    arcseconds takes the same steps as one written in meters and radians, with
    ``scale`` the number of leaf units per scaled unit (``1e9`` for meters to
    nanometers).

    Args:
        transform: An ``optax.GradientTransformation``.
        scale: A positive scalar, or a PyTree of scalars with the structure of the
            group's parameters.

    Returns:
        An ``optax.GradientTransformation``. It passes ``params / scale`` to
        ``transform.update``, so a transform that reads ``params`` (weight decay)
        sees the scaled variable.
    """

    def init(params):
        return transform.init(_scaled(params, scale, lambda x, s: x / s))

    def update(updates, state, params=None, **extra_args):
        scaled_params = (
            None if params is None else _scaled(params, scale, lambda x, s: x / s)
        )
        out, state = transform.update(
            _scaled(updates, scale, lambda x, s: x * s),
            state,
            scaled_params,
            **extra_args,
        )
        return _scaled(out, scale, lambda x, s: x * s), state

    return optax.GradientTransformationExtraArgs(init, update)
