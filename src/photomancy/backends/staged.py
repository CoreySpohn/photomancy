"""Staged MAP fitting over a PyTree: parameter groups, stages and per-group optimizers.

A fit is a sequence of stages. Each stage frees some named parameter groups and
runs a fixed number of optimizer steps on a log-density; the next stage continues
from the current values, and, unless it brings its own transforms, from the current
optimizer state. This module holds the pieces a staged fit is described with:

- :func:`group_labels` names the groups of a parameter PyTree from explicit selectors,
- :func:`rescaled_transform` runs a transform on ``params / scale`` so a leaf keeps
  its physical units while the steps match a fit in other units,
- :class:`FitStage` describes one stage and :func:`map_fit` runs the stages.

Every group has its own optimizer state (the transform applied to that group's
leaves only), so a group's state is exactly what ``optax.multi_transform`` would
hold for it, and a stage that does not free a group can leave its state alone.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

__all__ = [
    "UNLABELED",
    "FitStage",
    "MAPResult",
    "StageTrace",
    "group_labels",
    "map_fit",
    "rescaled_transform",
]

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


class FitStage(eqx.Module):
    """One stage of a staged fit: run ``n_steps`` optimizer steps with some groups free.

    Attributes:
        n_steps: Number of optimizer steps in the stage.
        free: Names of the groups the stage optimizes, a tuple of strings.
        name: Label carried into the trace (default ``""``).
    """

    n_steps: int
    free: tuple[str, ...]
    name: str = ""

    def __check_init__(self):
        """Validate the step count and the free-group names."""
        if not isinstance(self.n_steps, int) or self.n_steps < 0:
            raise ValueError(
                f"n_steps must be a non-negative int, got {self.n_steps!r}"
            )
        if not isinstance(self.free, tuple) or not all(
            isinstance(g, str) for g in self.free
        ):
            raise ValueError(f"free must be a tuple of group names, got {self.free!r}")


class StageTrace(eqx.Module):
    """What one stage recorded.

    Attributes:
        name: The stage name.
        start: Global index of the stage's first step.
        n_steps: Steps actually run.
        loss: Per-step objective (``-objective_scale * log_density`` before the step),
            a host array of length ``n_steps``.
    """

    name: str = eqx.field(static=True)
    start: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    loss: np.ndarray


class MAPResult(eqx.Module):
    """The outcome of :func:`map_fit`.

    Attributes:
        params: The fitted parameter PyTree, same structure as ``init``.
        stages: One :class:`StageTrace` per stage, in order.
        opt_states: ``{group: optimizer state}`` at the end of the run.
    """

    params: object
    stages: tuple[StageTrace, ...]
    opt_states: dict


_POLICIES = ("hold",)


def _value_and_aux(logdensity, params):
    """``(value, aux)`` from ``value_and_aux`` when present, else a plain callable."""
    method = getattr(logdensity, "value_and_aux", None)
    if method is not None:
        return method(params)
    return logdensity(params), None


def map_fit(logdensity, init, labels, transforms, stages, *, freeze="hold"):
    """Run a staged MAP fit of ``logdensity`` over the PyTree ``init``.

    Each stage optimizes the groups it frees (``FitStage.free``) with the group's
    transform for ``n_steps`` steps, minimizing ``-logdensity``. The optimizer state
    of every group persists across stages. The parameters are never mutated: ``init``
    stays valid and the result holds fresh buffers.

    Compilation: one ``filter_jit`` step per distinct stage configuration (the set of
    free groups), with the parameters, optimizer states and the trace buffer donated,
    so stages that repeat a configuration or differ in length do not recompile.

    Args:
        logdensity: A callable ``params -> scalar``, or an object with
            ``value_and_aux(params) -> (scalar, aux)`` (an
            :class:`~photomancy.core.model.AbstractLogDensity`).
        init: The initial parameter PyTree.
        labels: Group label PyTree from :func:`group_labels` (same structure as
            ``init``).
        transforms: ``{group: optax.GradientTransformation}``. A group with no
            transform cannot be freed.
        stages: Sequence of :class:`FitStage`.
        freeze: How a group that is not free in a stage is treated. ``"hold"``
            leaves its leaves and its optimizer state untouched, bitwise.

    Returns:
        A :class:`MAPResult`.
    """
    if freeze not in _POLICIES:
        raise ValueError(f"freeze must be one of {_POLICIES}, got {freeze!r}")
    stages = tuple(stages)
    leaves, treedef = jax.tree.flatten(init)
    lab = jax.tree.leaves(labels)
    if len(lab) != len(leaves):
        raise ValueError("labels must have the same structure as init")
    is_arr = [eqx.is_array(leaf) for leaf in leaves]
    members = {}
    for k, name in enumerate(lab):
        if name != UNLABELED:
            members.setdefault(name, []).append(k)
    for name in transforms:
        if name not in members:
            raise ValueError(f"transform for group {name!r}, which labels do not use")
    for stage in stages:
        for name in stage.free:
            if name not in transforms:
                raise ValueError(f"stage {stage.name!r} frees {name!r}: no transform")

    static_leaves = [
        None if arr else leaf for arr, leaf in zip(is_arr, leaves, strict=True)
    ]
    dyn = tuple(
        jnp.array(leaf, copy=True) if arr else None
        for arr, leaf in zip(is_arr, leaves, strict=True)
    )
    total = sum(stage.n_steps for stage in stages)
    dtype = jnp.zeros(()).dtype
    bufs = {"loss": jnp.full((total,), jnp.nan, dtype)}

    def rebuild(arrays):
        full = [
            a if arr else s
            for a, arr, s in zip(arrays, is_arr, static_leaves, strict=True)
        ]
        return jax.tree.unflatten(treedef, full)

    def group_slices(arrays, groups):
        return {g: tuple(arrays[k] for k in members[g]) for g in groups}

    steps = {}

    def get_step(free):
        if free in steps:
            return steps[free]

        @eqx.filter_jit(donate="all-except-first")
        def step(static_args, arrays, states, bufs, count):
            ld, scale = static_args
            fit = group_slices(arrays, free)

            def objective(fit):
                full = list(arrays)
                for g in free:
                    for k, v in zip(members[g], fit[g], strict=True):
                        full[k] = v
                value, _ = _value_and_aux(ld, rebuild(full))
                return -scale * value

            loss, grads = jax.value_and_grad(objective)(fit)
            new_arrays, new_states = list(arrays), dict(states)
            for g in free:
                update, new_states[g] = transforms[g].update(
                    grads[g], states[g], fit[g]
                )
                for k, v in zip(
                    members[g], optax.apply_updates(fit[g], update), strict=True
                ):
                    new_arrays[k] = v
            bufs = {"loss": bufs["loss"].at[count].set(loss.astype(bufs["loss"].dtype))}
            return tuple(new_arrays), new_states, bufs, count + 1

        steps[free] = step
        return step

    states = {g: transforms[g].init(group_slices(dyn, [g])[g]) for g in transforms}
    offset, records = 0, []
    for stage in stages:
        step = get_step(tuple(stage.free))
        static_args = (logdensity, jnp.asarray(1.0, dtype))
        count = jnp.asarray(offset, jnp.int32)
        for _ in range(stage.n_steps):
            dyn, states, bufs, count = step(static_args, dyn, states, bufs, count)
        records.append((stage.name, offset, stage.n_steps))
        offset += stage.n_steps

    host_loss = np.asarray(bufs["loss"])
    traces = tuple(
        StageTrace(name=name, start=start, n_steps=n, loss=host_loss[start : start + n])
        for name, start, n in records
    )
    return MAPResult(params=rebuild(dyn), stages=traces, opt_states=states)
