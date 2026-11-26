"""Staged MAP fitting over a PyTree: parameter groups, stages and per-group optimizers.

A fit is a sequence of stages. Each stage frees some named parameter groups and
runs a fixed number of optimizer steps on a log-density; the next stage continues
from the current values, and, unless it brings its own transforms, from the current
optimizer state. This module holds the pieces a staged fit is described with:

- :func:`group_labels` names the groups of a parameter PyTree from explicit selectors,
- :func:`rescaled_transform` runs a transform on ``params / scale`` so a leaf keeps
  its physical units while the steps match a fit in other units,
- :class:`MAPStage` describes one stage and :func:`map_fit` runs the stages (the
  engine of :class:`MAP`, which runs a staged fit of a ``Fit`` through ``run``).

Every group has its own optimizer state (the transform applied to that group's
leaves only), so a group's state is exactly what ``optax.multi_transform`` would
hold for it, and a stage that does not free a group can leave its state alone.
"""

import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hwoutils.leaves import canonical
from hwoutils.numerics import roundoff
from hwoutils.precision import require_float64

from photomancy.infer._scale import _map_scale
from photomancy.infer.backends.base import AbstractBackend, BackendDraws
from photomancy.infer.fit import Fit
from photomancy.infer.posterior import Posterior

__all__ = [
    "MAP",
    "UNLABELED",
    "EarlyStopping",
    "MAPInfo",
    "MAPResult",
    "MAPStage",
    "StageTrace",
    "StepRecord",
    "group_labels",
    "map_fit",
    "rescaled_transform",
]

UNLABELED = ""
"""Label of a leaf that belongs to no group (never optimized)."""

_GRAD_TOL = 1e-3
"""Scaled gradient norm below which a stalled default :class:`MAP` stage has converged.

In the information-normalized coordinates a gradient norm ``g`` bounds the Newton
decrement by ``g**2 / (2 lambda_min)`` nats, with ``lambda_min`` the smallest eigenvalue
of the scaled Hessian: about 5e-7 nat when the scaling is good (``lambda_min`` near
one), and larger where the information misses the curvature.
"""


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

    A loss stall is not convergence: a quasi-Newton run in badly scaled coordinates
    can fail to lower the loss for several steps and then speed up. With ``grad_tol``
    and ``grad_scale`` the stage stops only when the loss has stalled **and** the
    gradient norm in the scaled coordinates, ``|grad * grad_scale|`` at the step read,
    is below ``grad_tol``; a stalled stage whose gradient is still large runs on.
    The gate needs a ``solver`` stage (the gradient is recorded by the solver step).

    Attributes:
        patience: Consecutive non-improving steps that end the stage.
        min_delta: Smallest decrease that counts as an improvement.
        check_every: Steps between host checks.
        grad_tol: Largest scaled gradient norm of a stopped stage, or ``None`` for
            the loss stall alone.
        grad_scale: With ``grad_tol``, the per-leaf scale that makes the stage's
            coordinates unit-scale, with the structure of the stage's free groups
            (the scale a ``rescaled_transform`` solver divides by).
    """

    patience: int
    min_delta: float = 0.0
    check_every: int = 10
    grad_tol: float | None = eqx.field(static=True, default=None)
    grad_scale: Any = None

    def __check_init__(self):
        """Validate the counts and the gradient gate."""
        if self.patience < 1 or self.check_every < 1:
            raise ValueError("patience and check_every must be at least 1")
        if (self.grad_tol is None) != (self.grad_scale is None):
            raise ValueError("grad_tol and grad_scale go together")


class MAPStage(eqx.Module):
    """One stage of a staged fit: run ``n_steps`` optimizer steps with some groups free.

    Attributes:
        n_steps: Number of optimizer steps in the stage (the most, with early stopping).
        groups: Names of the groups the stage optimizes, a tuple of strings.
        fit: The density for this stage, replacing the one given to :func:`map_fit`
            (for example a ``Fit`` over one exposure). :class:`MAP` requires a
            ``Fit`` with the same coordinates as the run fit. The engine
            :func:`map_fit` also takes a callable. Before the first ``prepare``, this
            override applies only to this stage. After it, the replacement persists
            in later stages.
            ``None`` keeps the current density.
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
        prepare: ``params -> density`` called once on the host when the stage
            starts, even with zero steps. It receives independent JAX arrays of
            the current parameters, with the structure of ``init``. The returned
            density is used for this and later stages until another stage supplies
            ``prepare`` or ``fit``. Excludes ``fit`` on the same stage.
    """

    n_steps: int
    groups: tuple[str, ...]
    fit: Any = None
    objective_scale: float = 1.0
    closed_form: Callable | None = None
    transforms: dict | None = None
    early_stopping: EarlyStopping | None = None
    solver: Any = None
    name: str = ""
    prepare: Callable | None = None

    def __check_init__(self):
        """Validate the step count and the group names."""
        if not isinstance(self.n_steps, int) or self.n_steps < 0:
            raise ValueError(
                f"n_steps must be a non-negative int, got {self.n_steps!r}"
            )
        if not isinstance(self.groups, tuple) or not all(
            isinstance(g, str) for g in self.groups
        ):
            raise ValueError(
                f"groups must be a tuple of group names, got {self.groups!r}"
            )
        if self.solver is not None and (
            self.closed_form is not None or self.transforms is not None
        ):
            raise ValueError("a solver stage takes no closed_form and no transforms")
        if self.solver is not None and not self.groups:
            raise ValueError("a solver stage must free at least one group")
        if self.prepare is not None and self.fit is not None:
            raise ValueError("a stage takes either prepare or fit, not both")


class StageTrace(eqx.Module):
    """What one stage recorded.

    Attributes:
        name: The stage name.
        start: Global index of the stage's first step.
        n_steps: Steps actually run (fewer than planned after an early stop).
        stopped_early: Whether early stopping ended the stage. For the default
            :class:`MAP` stage this means converged: the loss stalled and the scaled
            gradient norm was below its tolerance. Never true when ``failed``.
        loss: Per-step objective (``-objective_scale * log_density`` before the step),
            a host array of length ``n_steps``.
        metrics: ``{name: host array of length n_steps}`` for each metric.
        failed: Whether the stage met a non-finite loss. A stage with early stopping
            ends at the check that reads it (at most ``check_every`` steps after it);
            a stage without reads its trace once at the end. The parameters returned
            are the last iterate, which can be non-finite then. It is also set when the
            loss at the returned point (``final_loss``) is non-finite, which the trace
            alone cannot show because it holds the loss before each step.
        regressed: Whether the loss at the returned point is worse than the best loss
            in the trace by more than the stall tolerance (the larger of
            ``STOP_MIN_DELTA`` and 1e-10 of the best). The returned point is the last
            iterate, not the best one: check ``regressed``, or use
            :func:`photomancy.detection.measure`, which reports it as not converged.
        final_loss: The objective evaluated once at the returned point (NaN for a stage
            that ran no step). ``loss`` holds the value before each step, so this is
            the one number that says where the last step landed.
    """

    name: str = eqx.field(static=True)
    start: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    stopped_early: bool = eqx.field(static=True)
    loss: np.ndarray
    metrics: dict
    failed: bool = eqx.field(static=True, default=False)
    regressed: bool = eqx.field(static=True, default=False)
    final_loss: float = float("nan")


class StepRecord(eqx.Module):
    """What a callback receives: the state at one step, as JAX arrays.

    Everything is evaluated at the same point, the parameters before the step.

    Attributes:
        step: Global step index across all stages.
        stage: The :class:`MAPStage` being run.
        stage_step: Step index within the stage.
        params: The parameter PyTree before the step (JAX arrays).
        loss: The scaled objective at ``params``.
        aux: The ``aux`` of the log-density at ``params`` (JAX arrays).
        metrics: ``{name: float}`` at ``params``.
    """

    step: int
    stage: MAPStage
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
    value, aux = method(params) if method is not None else (logdensity(params), None)
    require_float64("map_fit log density", jnp.asarray(value))
    return value, aux


def _strong_copy(x):
    """A fresh buffer with the dtype of ``x`` and no weak type (so no retrace later)."""
    return jax.lax.convert_element_type(jnp.array(x, copy=True), jnp.asarray(x).dtype)


def _device_copy(tree):
    """Fresh device buffers for every array leaf, safe under donation."""
    return jax.tree.map(lambda x: _strong_copy(x) if eqx.is_array(x) else x, tree)


def _line_search_stalled(state):
    """Whether the last zoom line search ended without meeting its conditions.

    Reads the public ``info`` (``optax.ZoomLinesearchInfo``) of the
    ``optax.ScaleByZoomLinesearchState`` inside ``optax.lbfgs``: a positive decrease
    must exceed objective roundoff to resolve a stall. The curvature error is
    the slope excess per unit of the dimensionless line-search parameter.
    Compare that unit-step scale to objective roundoff independently of the
    final learning rate, which may come from a different trial.
    This assumes the zoom search's default tolerance of zero
    (``optax.scale_by_zoom_linesearch(tol=0.0)``), which the state does not record.
    A solver without a zoom line search never stalls.

    Args:
        state: The optax state of a solver stage after a step.

    Returns:
        True when the search failed to meet its conditions.
    """
    found = jax.tree.leaves(
        state, is_leaf=lambda x: isinstance(x, optax.ScaleByZoomLinesearchState)
    )
    for part in found:
        if isinstance(part, optax.ScaleByZoomLinesearchState):
            info = part.info
            decrease = float(info.decrease_error)
            curvature = float(info.curvature_error)
            if int(info.num_linesearch_steps) == 0 or (decrease == curvature == 0.0):
                return False
            if not np.isfinite(decrease) or not np.isfinite(curvature):
                return True
            floor = float(roundoff(jnp.abs(part.value)))
            return max(decrease, curvature) > floor
    return False


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

    Optimized coordinates use float64. Narrower labeled leaves are widened at
    entry. Unlabeled leaves retain their original dtype.

    Each stage optimizes the groups it frees (``MAPStage.groups``) with the group's
    transform for ``n_steps`` steps, minimizing ``-objective_scale * logdensity``.
    The optimizer state of every group persists across stages unless a stage brings
    its own transforms. The parameters are never mutated: ``init`` stays valid and
    the result holds fresh buffers. There is no randomness, and on CPU a repeated run
    is bitwise identical.

    A stage's ``prepare`` hook runs on the host with independent JAX-array copies of the
    current parameters before any step in that stage. Its returned log density
    persists until another stage supplies ``prepare`` or ``fit``. Before the
    first hook, explicit stage densities remain local to their stage, preserving
    fits without hooks. Hooks do not reset optimizer states.

    Compilation: one ``filter_jit`` step per distinct stage configuration (the set of
    free groups, the transform table and the closed form, or the solver object of a
    solver stage, compared by identity: reuse one solver object across stages that
    should share a program), with the parameters,
    optimizer states and trace buffers donated. Stage length and objective scale are
    runtime values, so stages that repeat a configuration do not recompile. A stage
    whose ``fit`` has a different PyTree structure compiles its own program.
    Densities returned by ``prepare`` reuse the same step when their static values,
    PyTree structure, array shapes and dtypes match; changing array values does not
    recompile.

    Args:
        logdensity: A callable ``params -> scalar``, or an object with
            ``value_and_aux(params) -> (scalar, aux)``. ``aux`` is what a stage's
            ``closed_form``, the metrics and the callback receive; the duck-typed
            ``value_and_aux`` is the protocol.
        init: The initial parameter PyTree.
        labels: Group label PyTree from :func:`group_labels` (same structure as
            ``init``).
        transforms: ``{group: optax.GradientTransformation}``. A group with no
            transform cannot be freed. The transform of a group sees only that
            group's leaves, as a tuple.
        stages: Sequence of :class:`MAPStage`.
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
        callback: ``callback(record)`` called with a :class:`StepRecord` of JAX
            arrays on the global steps ``callback_every`` selects; its return value is
            ignored. It costs a copy of the parameters on those steps.
        callback_every: Steps between callbacks (an int), or a predicate
            ``step -> bool`` on the global step index (for example a set's
            ``__contains__``), so a fit can report at stage boundaries without a
            copy of the parameters every step.

    Returns:
        A :class:`MAPResult`. The parameters are the last iterate, not the best one.
        A stage that meets a non-finite loss ends with ``StageTrace.failed`` true and
        ``stopped_early`` false, whatever else its rule said. Each stage's loss is also
        evaluated once at the point returned (``StageTrace.final_loss``): non-finite
        sets ``failed``, and worse than the trace's best by more than the stall
        tolerance sets ``regressed``.
    """
    require_float64("map_fit")
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
    if "loss" in metrics or "grad_norm" in metrics:
        raise ValueError("'loss' and 'grad_norm' are reserved for the stage traces")
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
            leaves[k] = canonical(leaves[k], precision="float64")

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
        for name in stage.groups:
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
    gated = any(
        s.early_stopping is not None and s.early_stopping.grad_scale is not None
        for s in stages
    )
    names = ("loss", *metrics, *(("grad_norm",) if gated else ()))
    bufs = {name: jnp.full((total,), jnp.nan, dtype) for name in names}
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

    @eqx.filter_jit
    def final_objective(static_args, arrays):
        ld, scale, _ = static_args
        return -scale * _value_and_aux(ld, rebuild(arrays))[0]

    def get_step(table, free, closed_form):
        key = (id(table), free, id(closed_form))
        if key in steps:
            return steps[key]
        keep_alive.append((table, closed_form))
        active = free if freeze == "hold" else tuple(table)

        @eqx.filter_jit(donate="all-except-first")
        def step(static_args, arrays, states, bufs, count):
            ld, scale, _ = static_args
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
                name: (
                    bufs[name]
                    .at[count]
                    .set(jnp.asarray(record[name], bufs[name].dtype))
                    if name in record
                    else bufs[name]
                )
                for name in bufs
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
            ld, scale, grad_scale = static_args

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
            if grad_scale is not None:
                scaled = jax.tree.map(lambda g, s: g * s, grads, grad_scale)
                record["grad_norm"] = jnp.sqrt(
                    sum(jnp.sum(x**2) for x in jax.tree.leaves(scaled))
                )
            if metrics:
                pre = rebuild(arrays)
                record.update({name: fn(pre, aux) for name, fn in metrics.items()})
            bufs = {
                name: (
                    bufs[name]
                    .at[count]
                    .set(jnp.asarray(record[name], bufs[name].dtype))
                    if name in record
                    else bufs[name]
                )
                for name in bufs
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
    prepared = False
    density = logdensity
    for stage in stages:
        if stage.prepare is not None:
            density = stage.prepare(_device_copy(rebuild(dyn)))
            prepared = True
        elif stage.fit is not None:
            density = stage.fit
        elif not prepared:
            density = logdensity
        if stage.transforms is not None:
            table = stage.transforms
            states = init_states(table, dyn)
        solver_state = None
        if stage.solver is not None:
            free = tuple(stage.groups)
            step = get_solver_step(stage.solver, free)
            # fresh buffers: a solver state may hold one array in two places, and
            # the step donates its inputs
            solver_state = jax.tree.map(
                lambda x: jnp.array(x, copy=True),
                stage.solver.init(group_slices(dyn, free)),
            )
        else:
            step = get_step(table, tuple(stage.groups), stage.closed_form)
        stopping = stage.early_stopping
        static_args = (
            density,
            jnp.asarray(stage.objective_scale, dtype),
            None if stopping is None else stopping.grad_scale,
        )
        count = jnp.asarray(offset, jnp.int32)
        best, bad, checked, stopped, failed = np.inf, 0, 0, False, False
        n_run = 0
        for j in range(stage.n_steps):
            gi = offset + j
            pre = _device_copy(rebuild(dyn)) if want_aux and due(gi) else None
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
                        aux=aux,
                        metrics={name: float(bufs[name][gi]) for name in metrics},
                    )
                )
            # Only a stage with an early-stopping rule ends here: without one the
            # stage runs its n_steps (a failed search in badly scaled coordinates
            # can be followed by steps that make progress).
            stalled = (
                stopping is not None
                and solver_state is not None
                and _line_search_stalled(solver_state)
            )
            if stopping is not None and (
                n_run % stopping.check_every == 0 or n_run == stage.n_steps or stalled
            ):
                window = slice(offset + checked, offset + n_run)
                fresh = np.asarray(bufs["loss"])[window]
                if stopping.grad_scale is None:
                    gradient = np.zeros_like(fresh)
                else:
                    gradient = np.asarray(bufs["grad_norm"])[window]
                checked = n_run
                for value, norm in zip(fresh, gradient, strict=True):
                    if not np.isfinite(value):
                        failed = True
                        break
                    if value < best - stopping.min_delta:
                        best, bad = value, 0
                    else:
                        bad += 1
                    if bad >= stopping.patience and (
                        stopping.grad_tol is None or norm < stopping.grad_tol
                    ):
                        stopped = True
                        break
                if stopped or failed:
                    break
            if stalled:
                # The line search ended without meeting its conditions: direct
                # evidence that the loss has stalled, so the loss half of the
                # rule is taken as met and only the gradient half is evaluated, at
                # the last recorded iterate. A small scaled gradient means
                # converged; otherwise the stage ends not converged.
                stopped = stopping.grad_tol is None or float(norm) < stopping.grad_tol
                break
        if not failed and stopping is None and n_run:
            failed = not np.all(
                np.isfinite(np.asarray(bufs["loss"])[offset : offset + n_run])
            )
        final, regressed = float("nan"), False
        if n_run:
            final = float(final_objective(static_args, dyn))
            seen = np.asarray(bufs["loss"])[offset : offset + n_run]
            seen = seen[np.isfinite(seen)]
            if not np.isfinite(final):
                failed = True
            elif seen.size:
                best_seen = float(seen.min())
                tolerance = max(STOP_MIN_DELTA, STOP_REL_DELTA * abs(best_seen))
                regressed = final > best_seen + tolerance
            stopped = stopped and not failed
        records.append((stage.name, offset, n_run, stopped, failed, regressed, final))
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
            failed=failed,
            regressed=regressed,
            final_loss=final,
        )
        for name, start, n, stopped, failed, regressed, final in records
    )
    return MAPResult(params=rebuild(dyn), stages=traces, opt_states=states)


STOP_PATIENCE = 5
"""Consecutive steps without a gain of ``STOP_MIN_DELTA`` that stall a default stage."""

STOP_MIN_DELTA = 1e-7  # tol: algorithm, 1e-7 nats of log-density improvement
"""Smallest decrease of the best objective that counts as a gain in a default stage."""

STOP_REL_DELTA = 1e-10  # tol: algorithm, 1e-10 of the objective scale (dimensionless)
"""Relative allowance for comparing the last objective to the best in the trace."""

STOP_CHECK_EVERY = 5
"""Steps between the stopping checks of a default stage."""


def default_stage(n_steps, solver, scale):
    """The default L-BFGS stage over every parameter.

    With a ``scale`` (the ``information_scale`` entry dict of ``photomancy.infer``)
    the solver runs in the scaled coordinates and the stage stops when the loss has
    stalled for ``STOP_PATIENCE`` steps (read every ``STOP_CHECK_EVERY``) and the scaled
    gradient norm is below ``_GRAD_TOL``. Without one the stage runs ``n_steps``.

    Args:
        n_steps: The most iterations.
        solver: The optax solver to run, usually ``optax.lbfgs()``.
        scale: The information scale entry dict, or ``None`` for an unscaled stage.

    Returns:
        The :class:`MAPStage` named ``"lbfgs"`` over the group ``"all"``.
    """
    stopping = None
    if scale is not None:
        table = {"all": tuple(jax.tree.leaves(scale))}
        solver = rescaled_transform(solver, table)
        stopping = EarlyStopping(
            patience=STOP_PATIENCE,
            min_delta=STOP_MIN_DELTA,
            check_every=STOP_CHECK_EVERY,
            grad_tol=_GRAD_TOL,
            grad_scale=table,
        )
    return MAPStage(
        n_steps, ("all",), solver=solver, early_stopping=stopping, name="lbfgs"
    )


class _RealLogProb(eqx.Module):
    """``u -> fit.real_log_prob(u)`` over the unconstrained entry dict of one draw.

    ``value_and_aux`` also returns the prediction of every term at ``u`` (``None`` for
    a term without one), the pre-step images a ``closed_form`` reads.

    Attributes:
        fit: The fit whose ``real_log_prob`` is the objective.
    """

    fit: Fit

    def __call__(self, u):
        return self.fit.real_log_prob(u)

    def value_and_aux(self, u):
        """Return ``(real_log_prob(u), per-term predictions)``.

        Args:
            u: Unconstrained entry dict of one draw.

        Returns:
            ``(real_log_prob(u), predictions)``, the scalar objective and the tuple of
            per-term predictions.
        """
        model = self.fit.model_at(self.fit.from_real(u)[0])
        preds = []
        for term in self.fit.terms:
            try:
                preds.append(term.predict(model))
            except NotImplementedError:
                preds.append(None)
        return self.fit.real_log_prob(u), tuple(preds)


def _real_view(view, fit):
    """Wrap a stage ``Fit`` after checking its coordinate map.

    Raises:
        TypeError: If ``view`` is not a ``Fit``.
        ValueError: If its free keys or coordinate maps differ from those of ``fit``.
    """
    if not isinstance(view, Fit):
        raise TypeError(f"a MAP stage density must be a Fit, got {type(view).__name__}")
    if set(view.free) != set(fit.free):
        raise ValueError(
            "a stage Fit must have the free entries of the run fit; "
            f"got {sorted(view.free)}, expected {sorted(fit.free)}"
        )
    difference = fit._coordinate_difference(view)
    if difference is not None:
        key, reason = difference
        raise ValueError(f"stage Fit coordinates differ at {key!r}: {reason}")
    return _RealLogProb(view)


def _real_stage(stage, fit):
    """Copy of ``stage`` whose ``fit`` and ``prepare`` result are real-coordinate."""
    if stage.fit is not None:
        stage = eqx.tree_at(lambda s: s.fit, stage, _real_view(stage.fit, fit))
    if stage.prepare is not None:
        prepare = stage.prepare

        def wrapped(params):
            with jax.ensure_compile_time_eval():
                view = prepare(params)
            return _real_view(view, fit)

        stage = eqx.tree_at(lambda s: s.prepare, stage, wrapped)
    return stage


class MAPInfo(eqx.Module):
    """The stage traces of a :class:`MAP` run.

    Attributes:
        stages: One :class:`StageTrace` per stage, in order.
        scale_fallback_reason: Precision diagnostic naming the forward leaf that
            prevented Fisher scaling, or ``None`` if no precision fallback occurred.
    """

    stages: tuple[StageTrace, ...]
    scale_fallback_reason: str | None = eqx.field(static=True, default=None)


class MAP(AbstractBackend):
    """Staged MAP fit of a ``Fit`` in unconstrained coordinates, one draw out.

    The optimized parameters are the entry dict of ``fit.to_real`` for one draw and
    the objective is ``fit.real_log_prob``, the density in those coordinates with the
    log volume of the map from the stored coordinates (the density a gradient sampler
    sees), so the optimum is its mode there. The result is a one-draw posterior at the
    optimum with weight one, no evidence and ``info = MAPInfo`` (the stage traces).
    The optimizer is :func:`map_fit` and is host-driven, so ``MAP`` runs one fit at a
    time and is not mapped over a stack of worlds. A fit with walls
    (:attr:`Fit.walls`) is not refused: an optimum inside the support is unaffected
    by a wall elsewhere, and an optimum at a wall is a boundary optimum that ``MAP``
    does not certify.

    With no ``stages`` the fit is one stage of at most ``n_steps`` L-BFGS iterations
    over every parameter. It runs in coordinates scaled by the information at the
    start (:func:`photomancy.infer._scale.information_scale` with the prior
    included): the free parameters of an imaging fit span many orders of magnitude,
    and unscaled L-BFGS crawls along the stiff ones. The scale changes the path and
    the point the run reaches in a given number of steps, not the density it
    maximizes. On the scaled path the stage stops early only when it has converged:
    five consecutive steps (read every five) fail to lower the best objective by more
    than 1e-7 **and** the gradient norm in the scaled coordinates is below
    1e-3 (a Newton decrement near 5e-7 nat); the stage trace reports
    ``stopped_early=True`` for that and only that. A loss stall alone does not stop
    it, because L-BFGS in badly scaled coordinates stalls and then speeds up. A fit
    with a term that lacks a Fisher root runs unscaled, as does one with
    ``scale=False``. A float32 forward owner also runs unscaled, with a warning
    naming the leaf and the reason recorded in ``info.scale_fallback_reason``.
    The unscaled stage runs all ``n_steps`` and never reports
    ``stopped_early``.
    Explicit ``stages`` are never rescaled: their transforms are written in the
    coordinates of ``fit.to_real`` (use :func:`rescaled_transform`
    to change them). Pass ``labels``, ``transforms``
    and ``stages`` as for :func:`map_fit`
    (labels have the structure of ``fit.to_real`` of one draw) for a staged fit.

    The objective of a step is ``value_and_aux(u)``, which returns
    ``(fit.real_log_prob(u), predictions)`` with ``predictions`` the tuple of
    ``term.predict(model)`` at the parameters before the step (``None`` for a term
    that has no prediction). A stage's ``closed_form``,
    the ``metrics`` and the ``callback`` receive that tuple as ``aux``. ``closed_form``
    and the metrics see the unconstrained entry dict ``u`` (identity under a ``Normal``
    law); the prediction is a second call of the term on the same model, so a render is
    traced twice per stage configuration. A stage's ``fit``, or what its
    ``prepare`` returns, may change terms and prior densities. It must have the
    same free keys, trailing shapes and coordinate maps as the run fit. A
    difference raises ``ValueError`` on the host. During tracing a leaf support
    bound that depends on traced values is checked only for shape. A stage
    density of another type raises ``TypeError``.

    Attributes:
        n_steps: The most L-BFGS iterations of the default stage. The scaled stage
            stops earlier once it has converged (see above); the unscaled stage runs
            all of them.
        labels: Group label PyTree from :func:`group_labels`, or ``None`` for one
            group ``"all"`` over every leaf.
        transforms: ``{group: optax.GradientTransformation}``, or ``None``.
        stages: Tuple of :class:`MAPStage`, or ``None`` for the default stage.
        freeze: ``"hold"``, ``"zero_gradient"`` or ``"drop_update"``; see
            :func:`map_fit`.
        metrics: ``{name: fn(params, aux) -> scalar}`` recorded every step.
        callback: ``callback(StepRecord)`` called every ``callback_every`` steps.
        callback_every: Steps between callbacks, or a predicate on the global step.
        init: ``None`` for a prior draw, one constrained draw (a dict of the fit's
            entries), or a ``Posterior`` whose highest-``log_target`` draw starts it.
        scale: Scale the default stage by the information at the start (see above).
    """

    n_steps: int = eqx.field(static=True, default=200)
    labels: Any = None
    transforms: dict | None = None
    stages: tuple[MAPStage, ...] | None = None
    freeze: str = eqx.field(static=True, default="hold")
    metrics: dict | None = None
    callback: Callable | None = None
    callback_every: int | Callable = eqx.field(static=True, default=1)
    init: Posterior | dict | None = None
    scale: bool = eqx.field(static=True, default=True)

    def _start(self, fit, key):
        if self.init is None:
            x = fit.sample_prior(key)
        elif isinstance(self.init, Posterior):
            best = jnp.argmax(self.init.log_target)
            x = jax.tree.map(lambda a: a[best], self.init.samples)
        else:
            x = self.init
        leading = fit.check_draws(x)
        if leading:
            raise ValueError(f"MAP init needs one draw, got leading shape {leading}")
        return fit.to_real(x)

    def _run(self, fit, key, *, batch_size=None):
        """Optimize from the start and return the optimum as a one-draw cloud."""
        start = self._start(fit, key)
        labels = self.labels
        if labels is None:
            labels = jax.tree.map(lambda _: "all", start)
        stages = self.stages
        transforms = self.transforms
        scale_fallback_reason = None
        if stages is None:
            scale = None
            if self.scale:
                scale, scale_fallback_reason = _map_scale(fit, start)
                if scale_fallback_reason is not None:
                    warnings.warn(
                        f"MAP is running unscaled. {scale_fallback_reason}",
                        UserWarning,
                        stacklevel=2,
                    )
            stages = (default_stage(self.n_steps, optax.lbfgs(), scale),)
        stages = tuple(_real_stage(stage, fit) for stage in stages)
        result = map_fit(
            _RealLogProb(fit),
            start,
            labels,
            {} if transforms is None else transforms,
            stages,
            freeze=self.freeze,
            metrics=self.metrics,
            callback=self.callback,
            callback_every=self.callback_every,
        )
        x, _ = fit.from_real(result.params)
        return BackendDraws(
            samples=jax.tree.map(lambda a: jnp.asarray(a)[None], x),
            log_weights=jnp.zeros(1),
            log_evidence=jnp.asarray(jnp.nan),
            log_evidence_se=jnp.asarray(jnp.nan),
            info=MAPInfo(
                stages=result.stages, scale_fallback_reason=scale_fallback_reason
            ),
            kind="monte_carlo",
        )
