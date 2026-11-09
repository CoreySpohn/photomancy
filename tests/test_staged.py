"""Staged MAP backend: groups, rescaled transforms and the staged fit loop."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from photomancy.backends.base import AbstractBackend
from photomancy.backends.staged import (
    UNLABELED,
    EarlyStopping,
    FitStage,
    MAPBackend,
    group_labels,
    map_fit,
    rescaled_transform,
)
from photomancy.posterior import PointPosterior


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


class _Q(eqx.Module):
    x: jax.Array
    y: jax.Array


class _PreStepAux(eqx.Module):
    """Quadratic around ``target``; ``aux`` is the ``x`` the value was scored at."""

    target: jax.Array

    def __call__(self, p):
        return -0.5 * jnp.sum((p.x - self.target) ** 2)

    def value_and_aux(self, p):
        return self(p), p.x


def _q_init():
    return _Q(x=jnp.asarray([8.0, -4.0]), y=jnp.asarray([0.0, 0.0]))


_Q_GROUPS = {"x": lambda t: t.x}


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_closed_form_sees_the_pre_step_aux_and_runs_after_the_update():
    """``closed_form`` receives the aux scored before the step, on updated params.

    Truth source: a run one step shorter. With ``aux = x`` and the closed form
    ``y <- aux``, the final ``y`` equals ``x`` after ``n - 1`` steps, while the final
    ``x`` is the ``n``-step value.
    Tolerance basis: exact-algebra (bitwise; the same compiled step, one run longer).
    Fails if: the closed form is handed the post-step prediction (the flux ratio of
    a fit would then lag or lead by one iteration), or is applied before the update.
    """
    init, ld = _q_init(), _PreStepAux(jnp.asarray([1.0, 2.0]))
    labels = group_labels(init, _Q_GROUPS)
    tx = {"x": optax.sgd(0.25)}

    def closed_form(params, aux):
        return eqx.tree_at(lambda t: t.y, params, aux)

    run = lambda n: map_fit(  # noqa: E731
        ld, init, labels, tx, [FitStage(n, ("x",), closed_form=closed_form)]
    )
    short, full = run(4), run(5)
    np.testing.assert_array_equal(full.params.y, short.params.x)
    assert not np.array_equal(full.params.x, short.params.x)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_early_stopping_ends_a_stage_at_the_expected_step():
    """Patience 3 on a one-step quadratic stops after step 5, or 6 with a 3-step check.

    Truth source: ``sgd(1.0)`` on ``0.5 x^2`` reaches ``x = 0`` in one step, so the
    losses are ``0.5 x0^2, 0, 0, 0, 0 ...``: the best value is at step 1, steps 2, 3
    and 4 do not improve, and the third non-improving step is step 4 (5 steps run).
    Checking every 3 steps reads steps 0-2, then 3-5, and stops at the check (6 run).
    Tolerance basis: exact-algebra (integer step counts).
    Fails if: patience counts from the wrong step, the check overshoots by more than
    ``check_every - 1``, or the next stage does not start at the step the stage ended.
    """
    init, ld = _q_init(), _PreStepAux(jnp.zeros(2))
    labels = group_labels(init, _Q_GROUPS)
    tx = {"x": optax.sgd(1.0)}

    def run(check_every):
        stages = [
            FitStage(
                50,
                ("x",),
                early_stopping=EarlyStopping(3, check_every=check_every),
                name="s1",
            ),
            FitStage(2, ("x",), name="s2"),
        ]
        return map_fit(ld, init, labels, tx, stages)

    exact, coarse = run(1), run(3)
    assert [t.n_steps for t in exact.stages] == [5, 2]
    assert [t.n_steps for t in coarse.stages] == [6, 2]
    assert exact.stages[0].stopped_early and not exact.stages[1].stopped_early
    assert exact.stages[1].start == 5 and coarse.stages[1].start == 6
    assert len(exact.stages[0].loss) == 5
    np.testing.assert_array_equal(exact.stages[0].loss[1:], 0.0)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_callbacks_metrics_scale_and_stage_logdensity():
    """Callbacks get pre-step host copies; metrics, scale and stage densities apply.

    Truth source: ``sgd(lr)`` on ``scale * 0.5 (x - t)^2`` moves ``x`` by
    ``-lr * scale * (x - t)``; with ``lr * scale = 1`` it lands on ``t``. The loss
    traced at a step is ``scale * 0.5 (x - t)^2`` at the pre-step ``x``.
    Tolerance basis: exact-algebra (powers of two, so every operation is exact).
    Fails if: the callback sees post-step or donated (invalidated) buffers, the
    metric is evaluated at a different point than the loss, ``objective_scale`` is
    applied twice or not at all, or a stage keeps the run's density.
    """
    init = _q_init()
    labels = group_labels(init, _Q_GROUPS)
    seen = []
    stages = [
        FitStage(2, ("x",), objective_scale=0.5, name="first"),
        FitStage(
            2,
            ("x",),
            logdensity=_PreStepAux(jnp.asarray([4.0, 4.0])),
            objective_scale=0.5,
            name="second",
        ),
    ]
    result = map_fit(
        _PreStepAux(jnp.asarray([2.0, 2.0])),
        init,
        labels,
        {"x": optax.sgd(2.0)},
        stages,
        metrics={"x0": lambda p, aux: aux[0]},
        callback=seen.append,
        callback_every=2,
    )
    np.testing.assert_array_equal(result.params.x, [4.0, 4.0])
    first, second = result.stages
    # First stage: x = (8, -4) lands on 2 in one step (lr * scale = 1), then stays.
    np.testing.assert_array_equal(first.metrics["x0"], [8.0, 2.0])
    np.testing.assert_array_equal(first.loss, [0.5 * 0.5 * (36 + 36), 0.0])
    np.testing.assert_array_equal(second.metrics["x0"], [2.0, 4.0])
    assert [r.step for r in seen] == [0, 2]
    assert all(isinstance(r.params.x, np.ndarray) for r in seen)
    np.testing.assert_array_equal(seen[0].params.x, [8.0, -4.0])
    np.testing.assert_array_equal(seen[1].params.x, [2.0, 2.0])
    np.testing.assert_array_equal(seen[1].aux, [2.0, 2.0])
    assert seen[0].loss == first.loss[0] and seen[1].stage.name == "second"
    np.testing.assert_array_equal(init.x, [8.0, -4.0])
    predicate = []
    map_fit(
        _PreStepAux(jnp.asarray([2.0, 2.0])),
        init,
        labels,
        {"x": optax.sgd(2.0)},
        stages,
        callback=predicate.append,
        callback_every={1, 3}.__contains__,
    )
    assert [r.step for r in predicate] == [1, 3]


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_first_step_group_rates_match_the_analytic_adam_step():
    """The first Adam step of each group is ``-lr g / (|g| + eps)``, rescaled or not.

    Truth source: the Adam definition at step one (bias-corrected moments equal
    ``g`` and ``g^2``), with optax's default ``eps = 1e-8``. Gradients are exact
    (``dL/dx = x - t`` for ``L = -log density``). A rescaled group runs Adam on
    ``x / s``: ``dx = -s lr (s g) / (|s g| + eps)``.
    Tolerance basis: floating-point (rtol 1e-12, float64).
    Fails if: a group's learning rate is applied to the wrong group, epsilon acts in
    leaf units instead of scaled units, or a schedule is read one step late.
    """
    init = _Q(x=jnp.asarray([8.0, -4.0]), y=jnp.asarray([3.0, 0.5]))
    labels = group_labels(init, {"x": lambda t: t.x, "y": lambda t: t.y})
    ld = lambda p: -0.5 * (jnp.sum((p.x - 1.0) ** 2) + jnp.sum((p.y + 1.0) ** 2))  # noqa: E731
    lr_x, lr_y, scale, eps = 0.1, 0.03, 1e6, 1e-8
    tx = {
        "x": optax.adam(lr_x),
        "y": rescaled_transform(optax.adam(lr_y), scale),
    }
    got = map_fit(ld, init, labels, tx, [FitStage(1, ("x", "y"))])
    gx, gy = np.asarray(init.x) - 1.0, np.asarray(init.y) + 1.0
    want_x = np.asarray(init.x) - lr_x * gx / (np.abs(gx) + eps)
    want_y = np.asarray(init.y) - scale * lr_y * (scale * gy) / (
        np.abs(scale * gy) + eps
    )
    np.testing.assert_allclose(got.params.x, want_x, rtol=1e-12)
    np.testing.assert_allclose(got.params.y, want_y, rtol=1e-12)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_one_compile_per_stage_configuration_and_fresh_transforms():
    """Four stages over three configurations trace thrice; fresh transforms restart.

    Truth source: the number of times the log-density body runs, which is once per
    trace (a Python side effect at trace time). Stages differ in length and scale
    but two share their free set, transform table and closed form.
    Tolerance basis: exact-algebra (integer counts).
    Fails if: a stage length, scale or repeated configuration retraces the step
    (six compiles per run would become one per stage), or a stage that supplies
    transforms continues the old Adam moments instead of starting fresh.
    """
    traces = []

    def ld(p):
        traces.append(1)
        return -0.5 * jnp.sum((p.x - 1.0) ** 2) - 0.5 * jnp.sum((p.y - 2.0) ** 2)

    init = _q_init()
    labels = group_labels(init, {"x": lambda t: t.x, "y": lambda t: t.y})
    tx = {"x": optax.adam(0.1), "y": optax.adam(0.1)}
    stages = [
        FitStage(4, ("x",)),
        FitStage(3, ("y",)),
        FitStage(6, ("x",), objective_scale=0.5),
        FitStage(2, ("x", "y")),
    ]
    map_fit(ld, init, labels, tx, stages)
    assert len(traces) == 4 - 1  # (x), (y), (x, y): the third stage repeats the first

    fresh = {"x": optax.adam(0.1), "y": optax.adam(0.1)}
    cont = map_fit(ld, init, labels, tx, [FitStage(3, ("x",)), FitStage(1, ("x",))])
    restarted = map_fit(
        ld,
        init,
        labels,
        tx,
        [FitStage(3, ("x",)), FitStage(1, ("x",), transforms=fresh)],
    )
    count = lambda r: int(r.opt_states["x"][0].count)  # noqa: E731
    assert count(cont) == 4 and count(restarted) == 1


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_map_backend_returns_a_point_posterior_through_the_backend_protocol():
    """``MAPBackend.run`` equals ``map_fit`` and returns a density-free point estimate.

    Truth source: ``sgd(1.0)`` on ``0.5 (x - t)^2`` lands on ``t`` in one step, where
    the log-density is exactly 0.
    Tolerance basis: exact-algebra (bitwise).
    Fails if: ``run`` fits a different problem than ``fit``, the log density is
    evaluated before instead of after the fit, or the point posterior claims an
    evidence or a sample it cannot have.
    """
    init, ld = _q_init(), _PreStepAux(jnp.asarray([1.0, 2.0]))
    backend = MAPBackend(
        labels=group_labels(init, _Q_GROUPS),
        transforms={"x": optax.sgd(1.0)},
        stages=(FitStage(2, ("x",)),),
    )
    assert isinstance(backend, AbstractBackend)
    post = backend.run(ld, init)
    assert isinstance(post, PointPosterior)
    np.testing.assert_array_equal(post.params.x, [1.0, 2.0])
    assert post.log_density == 0.0
    assert np.isnan(post.evidence)
    with pytest.raises(NotImplementedError):
        post.sample(jax.random.PRNGKey(0), 3)
    with pytest.raises(NotImplementedError):
        post.log_prob(jnp.zeros(2))


_ROT = np.linalg.qr(np.random.default_rng(7).standard_normal((5, 5)))[0]
_CURV = _ROT @ np.diag([1.0, 10.0, 100.0, 1e3, 1e4]) @ _ROT.T
_MODE = np.asarray([1.0, 2.0, 3.0, 0.5, -0.5])


def _coupled_quadratic(p):
    """Log density ``-0.5 d^T H d`` with H rotated (it couples a and b), cond 1e4."""
    d = jnp.concatenate([p.a, p.b]) - _MODE
    return -0.5 * d @ jnp.asarray(_CURV) @ d


def _optax_lbfgs_loop(x, n_steps):
    """The same fit written directly with optax: one vector, the solver's own loop."""

    def f(v):
        return -_coupled_quadratic(_Toy(v[:3], v[3:], jnp.ones(2)))

    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(f)

    @jax.jit
    def step(x, state):
        value, grad = value_and_grad(x, state=state)
        update, state = solver.update(
            grad, state, x, value=value, grad=grad, value_fn=f
        )
        return optax.apply_updates(x, update), state

    state = solver.init(x)
    for _ in range(n_steps):
        x, state = step(x, state)
    return np.asarray(x)


@vv.case("scene-fits", "code-verification", basis="mutation")
def test_solver_stage_reaches_the_mode_of_an_ill_conditioned_coupled_quadratic():
    """An L-BFGS stage over two groups reaches the mode; per-group Adam does not.

    Truth source: the mode of the quadratic, a = (1, 2, 3), b = (0.5, -0.5), with a
    rotated Hessian of condition number 1e4 that couples the two groups, and the same
    fit written as a plain optax L-BFGS loop on one vector.
    Tolerance basis: mutation and floating-point. The stage equals the plain loop after
    30 steps to 1e-10 (the same arithmetic in a different PyTree layout) and reaches
    the mode to 1e-8 after 60 steps, as the plain loop does; per-group Adam at lr 0.05
    in 60 steps stays more than 0.1 away, so the test fails if the solver stage falls
    back to the group transforms.
    Fails if: the solver is not given the stage objective (the line search needs it),
    the joint update is not written back to both groups, or the sign is wrong.
    """
    init = _toy_init()
    labels = group_labels(init, _TOY_GROUPS)
    x0 = jnp.concatenate([init.a, init.b])

    def solve(n):
        return map_fit(
            _coupled_quadratic,
            init,
            labels,
            _toy_transforms(),
            [FitStage(n, ("a", "b"), solver=optax.lbfgs())],
        )

    mid = solve(30)
    np.testing.assert_allclose(
        np.concatenate([mid.params.a, mid.params.b]),
        _optax_lbfgs_loop(x0, 30),
        rtol=0,
        atol=1e-10,
    )
    solved = solve(60)
    got = np.concatenate([solved.params.a, solved.params.b])
    np.testing.assert_allclose(got, _MODE, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(solved.params.fixed, init.fixed)
    adam = map_fit(
        _coupled_quadratic,
        init,
        labels,
        _toy_transforms(),
        [FitStage(60, ("a", "b"))],
    )
    assert np.max(np.abs(np.concatenate([adam.params.a, adam.params.b]) - _MODE)) > 0.1
    assert solved.stages[0].loss.shape == (60,)
    assert solved.stages[0].loss[-1] < solved.stages[0].loss[0]


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_solver_stage_leaves_other_groups_and_their_transform_states_untouched():
    """A solver stage moves only its free groups; the group transforms resume after it.

    Truth source: the definition of a solver stage (fresh solver state; groups it does
    not free, and every group transform state, are not touched).
    Tolerance basis: exact-algebra (bitwise equality).
    Fails if: the solver stage advances a group transform's clock or moments, moves a
    group it does not free, or leaks its state into the next stage.
    """
    init = _toy_init()
    labels = group_labels(init, _TOY_GROUPS)
    kwargs = dict(labels=labels, transforms=_toy_transforms(), freeze="drop_update")
    before = map_fit(_toy_logdensity, init, stages=[FitStage(10, ("a",))], **kwargs)
    after = map_fit(
        _toy_logdensity,
        init,
        stages=[
            FitStage(10, ("a",)),
            FitStage(8, ("b",), solver=optax.lbfgs()),
        ],
        **kwargs,
    )
    np.testing.assert_array_equal(after.params.a, before.params.a)
    assert not np.array_equal(after.params.b, before.params.b)
    for group in ("a", "b"):
        for got, want in zip(
            jax.tree.leaves(after.opt_states[group]),
            jax.tree.leaves(before.opt_states[group]),
            strict=True,
        ):
            np.testing.assert_array_equal(got, want)
    resumed = map_fit(
        _toy_logdensity,
        init,
        stages=[
            FitStage(10, ("a",)),
            FitStage(8, ("b",), solver=optax.lbfgs()),
            FitStage(5, ("a",)),
        ],
        **kwargs,
    )
    assert [t.n_steps for t in resumed.stages] == [10, 8, 5]
    assert int(resumed.opt_states["a"][0].count) == 15


def test_solver_stage_rejects_a_closed_form_transforms_or_an_unknown_group():
    """A solver stage refuses a closed form, its own transforms, and unknown groups."""
    with pytest.raises(ValueError, match="no closed_form"):
        FitStage(1, ("a",), closed_form=lambda p, aux: p, solver=optax.lbfgs())
    with pytest.raises(ValueError, match="no closed_form"):
        FitStage(1, ("a",), transforms=_toy_transforms(), solver=optax.lbfgs())
    with pytest.raises(ValueError, match="at least one group"):
        FitStage(1, (), solver=optax.lbfgs())
    init = _toy_init()
    with pytest.raises(ValueError, match="unknown"):
        map_fit(
            _toy_logdensity,
            init,
            group_labels(init, _TOY_GROUPS),
            _toy_transforms(),
            [FitStage(1, ("z",), solver=optax.lbfgs())],
        )


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_rescaled_solver_is_lbfgs_on_the_scaled_variable():
    """``rescaled_transform(lbfgs, s)`` on ``p`` is L-BFGS on ``q = p / s``.

    Truth source: the same solver run directly on the scaled problem
    ``g(q) = f(q * s)``, a quadratic whose scales differ by 1e8 in ``p`` and are
    equal in ``q``.
    Tolerance basis: floating-point (rtol 1e-9 on ``p / s`` after 8 steps; the two
    runs do the same arithmetic up to the multiplications by ``s``).
    Fails if: the solver's line search is handed the gradient or the objective in
    the unscaled variable, which mixes units inside the zoom line search.
    """
    s = jnp.asarray([1e-7, 1.0, 10.0])
    target = jnp.asarray([0.3, -1.0, 2.0])

    def f(p):
        return jnp.sum((p / s - target) ** 2 * jnp.asarray([1.0, 3.0, 9.0]))

    def g(q):
        return f(q * s)

    def run(solver, objective, x):
        # a fresh gradient each step, as map_fit computes it (a cached one from the
        # solver state would be in the solver's scaled variable)
        value_and_grad = jax.value_and_grad(objective)

        @jax.jit
        def step(x, state):
            value, grad = value_and_grad(x)
            update, state = solver.update(
                grad, state, x, value=value, grad=grad, value_fn=objective
            )
            return optax.apply_updates(x, update), state

        state = solver.init(x)
        for _ in range(8):
            x, state = step(x, state)
        return np.asarray(x)

    q0 = jnp.asarray([1.0, 1.0, 1.0])
    direct = run(optax.lbfgs(), g, q0)
    scaled = run(rescaled_transform(optax.lbfgs(), s), f, q0 * s)
    np.testing.assert_allclose(scaled / np.asarray(s), direct, rtol=1e-9, atol=0)
