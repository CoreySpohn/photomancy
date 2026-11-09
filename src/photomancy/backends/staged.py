"""Staged MAP fitting over a PyTree: parameter groups, stages and per-group optimizers.

A fit is a sequence of stages. Each stage frees some named parameter groups and
runs a fixed number of optimizer steps on a log-density; the next stage continues
from the current values, and, unless it brings its own transforms, from the current
optimizer state. This module holds the pieces a staged fit is described with:

- :func:`group_labels` names the groups of a parameter PyTree from explicit selectors,
- :func:`rescaled_transform` runs a transform on ``params / scale`` so a leaf keeps
  its physical units while the steps match a fit in other units,
- :class:`FitStage` describes one stage and :func:`map_fit` runs the stages,
- :class:`MAPBackend` packages a staged fit behind the backend protocol.

Every group has its own optimizer state (the transform applied to that group's
leaves only), so a group's state is exactly what ``optax.multi_transform`` would
hold for it, and a stage that does not free a group can leave its state alone.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import PointPosterior

__all__ = [
    "UNLABELED",
    "EarlyStopping",
    "FitStage",
    "MAPBackend",
    "MAPResult",
    "StageTrace",
    "StepRecord",
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
        An ``optax.GradientTransformationExtraArgs``. It passes ``params / scale`` to
        ``transform.update``, so a transform that reads ``params`` (weight decay)
        sees the scaled variable. The extra arguments of a line-search solver are
        mapped the same way: ``grad`` is multiplied by ``scale`` and ``value_fn``
        takes the scaled variable (``value`` is unchanged), so a solver stage with
        ``rescaled_transform(optax.lbfgs(), scale)`` runs L-BFGS in ``q``.
    """

    def init(params):
        return transform.init(_scaled(params, scale, lambda x, s: x / s))

    def update(updates, state, params=None, **extra_args):
        scaled_params = (
            None if params is None else _scaled(params, scale, lambda x, s: x / s)
        )
        extra = dict(extra_args)
        if extra.get("grad") is not None:
            extra["grad"] = _scaled(extra["grad"], scale, lambda x, s: x * s)
        if extra.get("value_fn") is not None:
            value_fn = extra["value_fn"]
            extra["value_fn"] = lambda q: value_fn(
                _scaled(q, scale, lambda x, s: x * s)
            )
        out, state = transform.update(
            _scaled(updates, scale, lambda x, s: x * s),
            state,
            scaled_params,
            **extra,
        )
        return _scaled(out, scale, lambda x, s: x * s), state

    return optax.GradientTransformationExtraArgs(init, update)


class EarlyStopping(eqx.Module):
    """End a stage when the objective stops improving.

    The stage stops once ``patience`` consecutive steps fail to lower the best
    objective seen in the stage by more than ``min_delta``. The check reads the loss
    trace on the host every ``check_every`` steps, so the stage can run up to
    ``check_every - 1`` steps past the step where the count was reached; use
    ``check_every=1`` for the exact step at the cost of a host read per step.

    Attributes:
        patience: Consecutive non-improving steps that end the stage.
        min_delta: Smallest decrease that counts as an improvement.
        check_every: Steps between host checks.
    """

    patience: int
    min_delta: float = 0.0
    check_every: int = 10

    def __check_init__(self):
        """Validate the counts."""
        if self.patience < 1 or self.check_every < 1:
            raise ValueError("patience and check_every must be at least 1")


class FitStage(eqx.Module):
    """One stage of a staged fit: run ``n_steps`` optimizer steps with some groups free.

    Attributes:
        n_steps: Number of optimizer steps in the stage (the most, with early stopping).
        free: Names of the groups the stage optimizes, a tuple of strings.
        logdensity: Log-density for this stage, replacing the one given to
            :func:`map_fit` (for example a view over one exposure). ``None`` keeps it.
        objective_scale: The stage minimizes ``-objective_scale * log_density``; the
            loss trace records that scaled objective. It is an array at run time, so
            changing it does not recompile.
        closed_form: ``(params, aux) -> params`` applied after each optimizer update
            with the ``aux`` of the log-density evaluation that produced the update's
            gradient (the pre-step prediction). It updates parameters that are not
            gradient leaves, for example a flux set by a ratio of windows.
        transforms: ``None`` continues the run's optimizers, their states and their
            schedule clock. A dict ``{group: transform}`` replaces the transform
            table and starts fresh states for its groups at the current values.
        early_stopping: An :class:`EarlyStopping`, or ``None`` to run all steps.
        solver: ``None`` steps each free group with its own transform. An
            ``optax.GradientTransformationExtraArgs`` (for example ``optax.lbfgs()``)
            instead optimizes the union of the free groups jointly, as one PyTree
            ``{group: leaves}``, and receives ``value``, ``grad`` and ``value_fn`` (the
            stage objective of that PyTree) for a line search. Its state starts fresh
            at the stage and ends with it; the groups the stage does not free stay
            untouched, whatever ``freeze`` says, and their transform states do not
            advance. It excludes ``closed_form`` (an update outside the solver would
            invalidate its curvature pairs) and ``transforms``.
        name: Label carried into the trace.
    """

    n_steps: int
    free: tuple[str, ...]
    logdensity: Any = None
    objective_scale: float = 1.0
    closed_form: Callable | None = None
    transforms: dict | None = None
    early_stopping: EarlyStopping | None = None
    solver: Any = None
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
        if self.solver is not None and (
            self.closed_form is not None or self.transforms is not None
        ):
            raise ValueError("a solver stage takes no closed_form and no transforms")
        if self.solver is not None and not self.free:
            raise ValueError("a solver stage must free at least one group")


class StageTrace(eqx.Module):
    """What one stage recorded.

    Attributes:
        name: The stage name.
        start: Global index of the stage's first step.
        n_steps: Steps actually run (fewer than planned after an early stop).
        stopped_early: Whether early stopping ended the stage.
        loss: Per-step objective (``-objective_scale * log_density`` before the step),
            a host array of length ``n_steps``.
        metrics: ``{name: host array of length n_steps}`` for each metric.
    """

    name: str = eqx.field(static=True)
    start: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    stopped_early: bool = eqx.field(static=True)
    loss: np.ndarray
    metrics: dict


class StepRecord(eqx.Module):
    """What a callback receives: the state at one step, as host copies.

    Everything is evaluated at the same point, the parameters before the step.

    Attributes:
        step: Global step index across all stages.
        stage: The :class:`FitStage` being run.
        stage_step: Step index within the stage.
        params: The parameter PyTree before the step (NumPy arrays).
        loss: The scaled objective at ``params``.
        aux: The ``aux`` of the log-density at ``params`` (NumPy arrays).
        metrics: ``{name: float}`` at ``params``.
    """

    step: int
    stage: FitStage
    stage_step: int
    params: Any
    loss: float
    aux: Any
    metrics: dict


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


_POLICIES = ("hold", "zero_gradient", "drop_update")


def _value_and_aux(logdensity, params):
    """``(value, aux)`` from ``value_and_aux`` when present, else a plain callable."""
    method = getattr(logdensity, "value_and_aux", None)
    if method is not None:
        return method(params)
    return logdensity(params), None


def _strong_copy(x):
    """A fresh buffer with the dtype of ``x`` and no weak type (so no retrace later)."""
    return jax.lax.convert_element_type(jnp.array(x, copy=True), jnp.asarray(x).dtype)


def _host_copy(tree):
    """Copy every array leaf to host memory (a real copy, safe under donation)."""
    return jax.tree.map(lambda x: np.array(x) if eqx.is_array(x) else x, tree)


def map_fit(
    logdensity,
    init,
    labels,
    transforms,
    stages,
    *,
    freeze="hold",
    metrics=None,
    callback=None,
    callback_every=1,
):
    """Run a staged MAP fit of ``logdensity`` over the PyTree ``init``.

    Each stage optimizes the groups it frees (``FitStage.free``) with the group's
    transform for ``n_steps`` steps, minimizing ``-objective_scale * logdensity``.
    The optimizer state of every group persists across stages unless a stage brings
    its own transforms. The parameters are never mutated: ``init`` stays valid and
    the result holds fresh buffers. There is no randomness, and on CPU a repeated run
    is bitwise identical.

    Compilation: one ``filter_jit`` step per distinct stage configuration (the set of
    free groups, the transform table and the closed form), with the parameters,
    optimizer states and trace buffers donated. Stage length and objective scale are
    runtime values, so stages that repeat a configuration do not recompile. A stage
    whose ``logdensity`` has a different PyTree structure compiles its own program.

    Args:
        logdensity: A callable ``params -> scalar``, or an object with
            ``value_and_aux(params) -> (scalar, aux)`` (an
            :class:`~photomancy.core.model.AbstractLogDensity`).
        init: The initial parameter PyTree.
        labels: Group label PyTree from :func:`group_labels` (same structure as
            ``init``).
        transforms: ``{group: optax.GradientTransformation}``. A group with no
            transform cannot be freed. The transform of a group sees only that
            group's leaves, as a tuple.
        stages: Sequence of :class:`FitStage`.
        freeze: How a group that is not free in a stage is treated.

            - ``"hold"``: its leaves and its optimizer state are untouched, bitwise,
              and its gradient is not computed. Its optimizer clock (the step count a
              learning-rate schedule reads) advances only in stages that free it.
            - ``"zero_gradient"``: every group in ``transforms`` takes an optimizer
              step each iteration; a frozen group's gradient is replaced by zero, so
              it keeps moving on its momentum and its moments decay.
            - ``"drop_update"``: every group's gradient is computed and every
              group's optimizer state advances, but a frozen group's update is
              dropped, so it never moves while its moments keep accumulating its
              gradients.

            The last two reproduce the two freezing conventions of a single
            ``optax.multi_transform`` optimizer with one shared clock; use ``"hold"``
            for a new fit.
        metrics: ``{name: fn(params, aux) -> scalar}`` evaluated inside every step
            at the pre-step parameters and recorded in the trace.
        callback: ``callback(record)`` called with a :class:`StepRecord` of host
            copies on the global steps ``callback_every`` selects; its return value is
            ignored. It costs a host copy of the parameters on those steps.
        callback_every: Steps between callbacks (an int), or a predicate
            ``step -> bool`` on the global step index (for example a set's
            ``__contains__``), so a fit can report at stage boundaries without a
            host copy of the parameters every step.

    Returns:
        A :class:`MAPResult`.
    """
    if freeze not in _POLICIES:
        raise ValueError(f"freeze must be one of {_POLICIES}, got {freeze!r}")
    if callable(callback_every):
        due = callback_every
    elif int(callback_every) >= 1:
        every = int(callback_every)

        def due(step):
            return step % every == 0

    else:
        raise ValueError(f"callback_every must be >= 1, got {callback_every!r}")
    metrics = dict(metrics or {})
    if "loss" in metrics:
        raise ValueError("'loss' is reserved for the objective trace")
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

    def check_table(table, where):
        for name in table:
            if name not in members:
                raise ValueError(f"{where}: transform for unused group {name!r}")

    check_table(transforms, "transforms")
    table = transforms
    for stage in stages:
        if stage.transforms is not None:
            check_table(stage.transforms, f"stage {stage.name!r}")
            table = stage.transforms
        for name in stage.free:
            if stage.solver is not None:
                if name not in members:
                    raise ValueError(f"stage {stage.name!r} frees unknown {name!r}")
            elif name not in table:
                raise ValueError(f"stage {stage.name!r} frees {name!r}: no transform")

    static_leaves = [
        None if arr else leaf for arr, leaf in zip(is_arr, leaves, strict=True)
    ]
    dyn = tuple(
        _strong_copy(leaf) if arr else None
        for arr, leaf in zip(is_arr, leaves, strict=True)
    )
    total = sum(stage.n_steps for stage in stages)
    dtype = jnp.zeros(()).dtype
    bufs = {name: jnp.full((total,), jnp.nan, dtype) for name in ("loss", *metrics)}
    want_aux = callback is not None

    def rebuild(arrays):
        full = [
            a if arr else s
            for a, arr, s in zip(arrays, is_arr, static_leaves, strict=True)
        ]
        return jax.tree.unflatten(treedef, full)

    def group_slices(arrays, groups):
        return {g: tuple(arrays[k] for k in members[g]) for g in groups}

    steps, keep_alive = {}, []

    def get_step(table, free, closed_form):
        key = (id(table), free, id(closed_form))
        if key in steps:
            return steps[key]
        keep_alive.append((table, closed_form))
        active = free if freeze == "hold" else tuple(table)

        @eqx.filter_jit(donate="all-except-first")
        def step(static_args, arrays, states, bufs, count):
            ld, scale = static_args
            fit = group_slices(arrays, active)

            def objective(fit):
                full = list(arrays)
                for g in active:
                    for k, v in zip(members[g], fit[g], strict=True):
                        full[k] = v
                value, aux = _value_and_aux(ld, rebuild(full))
                return -scale * value, aux

            (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(fit)
            new_arrays, new_states = list(arrays), dict(states)
            for g in active:
                grad = grads[g]
                if freeze == "zero_gradient" and g not in free:
                    grad = jax.tree.map(jnp.zeros_like, grad)
                update, new_states[g] = table[g].update(grad, states[g], fit[g])
                if g not in free and freeze == "drop_update":
                    continue
                for k, v in zip(
                    members[g], optax.apply_updates(fit[g], update), strict=True
                ):
                    new_arrays[k] = v
            if closed_form is not None:
                updated = closed_form(rebuild(new_arrays), aux)
                flat = jax.tree.leaves(updated)
                new_arrays = [
                    jnp.asarray(flat[k], old.dtype) if arr else None
                    for k, (arr, old) in enumerate(zip(is_arr, new_arrays, strict=True))
                ]
            record = {"loss": loss}
            if metrics:
                pre = rebuild(arrays)
                record.update({name: fn(pre, aux) for name, fn in metrics.items()})
            bufs = {
                name: bufs[name].at[count].set(jnp.asarray(v, bufs[name].dtype))
                for name, v in record.items()
            }
            return (
                tuple(new_arrays),
                new_states,
                bufs,
                count + 1,
                aux if want_aux else None,
            )

        steps[key] = step
        return step

    def get_solver_step(solver, free):
        key = ("solver", id(solver), free)
        if key in steps:
            return steps[key]
        keep_alive.append(solver)

        @eqx.filter_jit(donate="all-except-first")
        def step(static_args, arrays, state, bufs, count):
            ld, scale = static_args

            def objective(fit):
                full = list(arrays)
                for g in free:
                    for k, v in zip(members[g], fit[g], strict=True):
                        full[k] = v
                value, aux = _value_and_aux(ld, rebuild(full))
                return -scale * value, aux

            fit = group_slices(arrays, free)
            (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(fit)
            update, state = solver.update(
                grads,
                state,
                fit,
                value=loss,
                grad=grads,
                value_fn=lambda f: objective(f)[0],
            )
            moved = optax.apply_updates(fit, update)
            new_arrays = list(arrays)
            for g in free:
                for k, v in zip(members[g], moved[g], strict=True):
                    new_arrays[k] = v
            record = {"loss": loss}
            if metrics:
                pre = rebuild(arrays)
                record.update({name: fn(pre, aux) for name, fn in metrics.items()})
            bufs = {
                name: bufs[name].at[count].set(jnp.asarray(v, bufs[name].dtype))
                for name, v in record.items()
            }
            return (
                tuple(new_arrays),
                state,
                bufs,
                count + 1,
                aux if want_aux else None,
            )

        steps[key] = step
        return step

    def init_states(table, arrays):
        return {g: table[g].init(group_slices(arrays, [g])[g]) for g in table}

    table = transforms
    states = init_states(table, dyn)
    offset, records = 0, []
    for stage in stages:
        if stage.transforms is not None:
            table = stage.transforms
            states = init_states(table, dyn)
        solver_state = None
        if stage.solver is not None:
            free = tuple(stage.free)
            step = get_solver_step(stage.solver, free)
            # fresh buffers: a solver state may hold one array in two places, and
            # the step donates its inputs
            solver_state = jax.tree.map(
                lambda x: jnp.array(x, copy=True),
                stage.solver.init(group_slices(dyn, free)),
            )
        else:
            step = get_step(table, tuple(stage.free), stage.closed_form)
        static_args = (
            logdensity if stage.logdensity is None else stage.logdensity,
            jnp.asarray(stage.objective_scale, dtype),
        )
        count = jnp.asarray(offset, jnp.int32)
        stopping = stage.early_stopping
        best, bad, checked, stopped = np.inf, 0, 0, False
        n_run = 0
        for j in range(stage.n_steps):
            gi = offset + j
            pre = _host_copy(rebuild(dyn)) if want_aux and due(gi) else None
            if solver_state is None:
                dyn, states, bufs, count, aux = step(
                    static_args, dyn, states, bufs, count
                )
            else:
                dyn, solver_state, bufs, count, aux = step(
                    static_args, dyn, solver_state, bufs, count
                )
            n_run = j + 1
            if pre is not None:
                callback(
                    StepRecord(
                        step=gi,
                        stage=stage,
                        stage_step=j,
                        params=pre,
                        loss=float(bufs["loss"][gi]),
                        aux=_host_copy(aux),
                        metrics={name: float(bufs[name][gi]) for name in metrics},
                    )
                )
            if stopping is not None and (
                n_run % stopping.check_every == 0 or n_run == stage.n_steps
            ):
                fresh = np.asarray(bufs["loss"])[offset + checked : offset + n_run]
                checked = n_run
                for value in fresh:
                    if value < best - stopping.min_delta:
                        best, bad = value, 0
                    else:
                        bad += 1
                    if bad >= stopping.patience:
                        stopped = True
                        break
                if stopped:
                    break
        records.append((stage.name, offset, n_run, stopped))
        offset += n_run

    host = {name: np.asarray(buf) for name, buf in bufs.items()}
    traces = tuple(
        StageTrace(
            name=name,
            start=start,
            n_steps=n,
            stopped_early=stopped,
            loss=host["loss"][start : start + n],
            metrics={m: host[m][start : start + n] for m in metrics},
        )
        for name, start, n, stopped in records
    )
    return MAPResult(params=rebuild(dyn), stages=traces, opt_states=states)


@eqx.filter_jit
def _log_density_at(logdensity, params):
    return _value_and_aux(logdensity, params)[0]


class MAPBackend(AbstractBackend):
    """Staged MAP fit over a PyTree, returning a ``PointPosterior``.

    The backend holds the fit's description; ``run(logdensity, init)`` is
    :func:`map_fit` followed by one evaluation of ``logdensity`` at the result. Unlike
    the other backends, ``init`` is the parameter PyTree, not a flat array, and there
    is no randomness (``key`` is unused). Use :meth:`fit` for the per-stage traces.

    Args:
        labels: Group label PyTree from :func:`group_labels`.
        transforms: ``{group: optax.GradientTransformation}``.
        stages: Tuple of :class:`FitStage`.
        freeze: ``"hold"``, ``"zero_gradient"`` or ``"drop_update"``; see
            :func:`map_fit`.
        metrics: ``{name: fn(params, aux) -> scalar}`` recorded every step.
        callback: ``callback(StepRecord)`` called every ``callback_every`` steps.
        callback_every: Steps between callbacks, or a predicate on the global
            step; see :func:`map_fit`.
    """

    labels: Any
    transforms: dict
    stages: tuple[FitStage, ...]
    freeze: str = eqx.field(static=True, default="hold")
    metrics: dict | None = None
    callback: Callable | None = None
    callback_every: int | Callable = eqx.field(static=True, default=1)

    def fit(self, logdensity, init):
        """Run the staged fit and return the :class:`MAPResult` with its traces."""
        return map_fit(
            logdensity,
            init,
            self.labels,
            self.transforms,
            self.stages,
            freeze=self.freeze,
            metrics=self.metrics,
            callback=self.callback,
            callback_every=self.callback_every,
        )

    def run(self, logdensity, init, key=None):
        """Fit from the parameter PyTree ``init``; return a ``PointPosterior``.

        ``PointPosterior.log_density`` is the run-level ``logdensity`` at the result,
        even when the last stage fitted a restricted view of it.
        """
        result = self.fit(logdensity, init)
        return PointPosterior(
            params=result.params,
            log_density=_log_density_at(logdensity, result.params),
        )
