"""Coordinate maps and draw shapes at fit boundaries."""

from dataclasses import replace

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from priorities import Beta, MultivariateNormal, Normal, Uniform
from test_staged import _Bowl, _BowlTerm
from test_support_swap import KEY as BLOCK
from test_support_swap import _cloud
from test_support_swap import _fit as orbit_fit

from photomancy.infer import MAP, Fit, Posterior, run
from photomancy.infer.backends._flat import start_points
from photomancy.infer.backends.staged import MAPStage, _real_stage

pytestmark = pytest.mark.float64_only(reason="float64 Fit coordinates")


def _fit(law, target=0.5, shape=()):
    return Fit(_Bowl(jnp.zeros(shape), jnp.zeros(())), {"x": law}, (_BowlTerm(target),))


def _optimum(run_fit, **override):
    stage = MAPStage(12, ("x",), solver=optax.lbfgs(), **override)
    backend = MAP(labels={"x": "x"}, stages=(stage,), init={"x": jnp.array(0.5)})
    return run(run_fit, backend, jax.random.key(0)).samples["x"]


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("prepared", [False, True])
def test_stage_rejects_different_bounds(prepared):
    """The same key on a different interval raises before the first step.

    Truth source: the intervals [0, 1] and [10, 20] map the same u differently.
    Tolerance basis: exact-algebra (a named exception).
    Fails if: a stage optimizes at 16 in its interval and reports 0.6 in the run.
    """
    fit = _fit(Uniform(0.0, 1.0))
    # At 16 the log-volume slope in x is 1/6 - 1/4.
    view = _fit(Uniform(10.0, 20.0), 16.0 + 1 / 12)
    override = {"prepare": lambda p: view} if prepared else {"fit": view}
    with pytest.raises(ValueError, match=r"x.*bounds"):
        _optimum(fit, **override)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_prepared_map_bounds_under_trace():
    """A traced prepare compares concrete endpoints and refuses dynamic endpoints.

    Truth source: a concrete interval differs from the run interval at both ends.
    Tolerance basis: exact-algebra (a host raise during tracing).
    Fails if: support arithmetic hides concrete bounds or reads a tracer on the host.
    """
    fit = _fit(Uniform(0.0, 1.0))

    def prepare(high):
        return fit.with_priors({"x": Uniform(0.0, high)})

    def trace(fn):
        wrapped = _real_stage(MAPStage(0, (), prepare=fn), fit)
        return jax.eval_shape(lambda h: wrapped.prepare(h).fit.model.x, 2.0)

    with pytest.raises(ValueError, match=r"x.*bounds"):
        trace(lambda _: prepare(2.0))
    # A bound that depends on a traced value cannot be shown equal, so it is refused.
    with pytest.raises(ValueError, match=r"x.*traced"):
        trace(prepare)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.medium(reason="two L-BFGS runs compile separate stage objectives")
@pytest.mark.parametrize(
    "law",
    [
        Uniform(0.0, 1.0),
        pytest.param(
            Beta(2.0, 3.0),
            marks=pytest.mark.slow(
                reason="compiles two staged L-BFGS objectives after replacing the fit "
                "term, 6 s projected on CI"
            ),
        ),
    ],
)
def test_stage_accepts_changed_terms_and_density(law):
    """A compatible stage reaches the bitwise optimum of its standalone objective.

    Truth source: the optimizer on the stage fit with no override.
    Tolerance basis: exact-algebra (identical steps).
    Fails if: changing a term or the density inside a fixed interval is refused.
    """
    fit = _fit(Uniform(0.0, 1.0))
    view = _fit(law, 0.75)
    assert fit.same_coordinates(view)
    np.testing.assert_array_equal(_optimum(fit, fit=view), _optimum(view))


class _IdentityUniform(Uniform):
    """Keep the interval endpoints but select the identity real transform."""

    bounds = "real"


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_same_coordinates_compares_transform_and_trailing_shapes():
    """Map equality includes the transform family, model shape and event shape.

    Truth source: priorities selects the map by bounds and sums event axes.
    Tolerance basis: exact-algebra (boolean equality).
    Fails if: bounds alone decide compatibility or density parameters enter it.
    """
    fit = _fit(Uniform(0.0, 1.0))
    assert fit.same_coordinates(_fit(Beta(2.0, 3.0)))
    assert not fit.same_coordinates(_fit(_IdentityUniform(0.0, 1.0)))
    assert not fit.same_coordinates(_fit(Uniform(0.0, 1.0), shape=(2,)))
    assert not fit.same_coordinates(Fit(fit.model, {}))
    # A density change that keeps the map is accepted whatever its event rank,
    # and equal bounds broadcast to the leaf shape are equal.
    scalar_events = _fit(Normal(jnp.zeros(2), jnp.ones(2)), shape=(2,))
    vector_event = _fit(MultivariateNormal(jnp.zeros(2), jnp.eye(2)), shape=(2,))
    assert scalar_events.same_coordinates(vector_event)
    pair = _fit(Uniform(jnp.zeros(2), jnp.ones(2)), shape=(2,))
    assert pair.same_coordinates(_fit(Uniform(0.0, 1.0), shape=(2,)))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_stage_compares_block_map_and_coordinate_schema():
    """Changing a mapped orbit size bound changes the stage coordinates.

    Truth source: the orbit block maps its size interval and owns one row.
    Tolerance basis: exact-algebra (a named exception and exact shapes).
    Fails if: the block map or its coordinate names and row shapes are skipped.
    """
    fit, view = orbit_fit(0.5, 3.0), orbit_fit(1.0, 2.0)
    stage = MAPStage(0, (), fit=view)
    with pytest.raises(ValueError, match=BLOCK):
        _real_stage(stage, fit)
    schema = fit.draw_schema()[BLOCK]
    x = {BLOCK: {name: jnp.zeros((4, 1)) for name in fit.free[BLOCK].coordinate_names}}
    assert fit.check_draws(x) == (4,)
    for name, field in schema.items():
        assert field.shape == (1,) and field.dtype == x[BLOCK][name].dtype
    name = next(iter(schema))
    x[BLOCK][name] = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match=rf"{BLOCK}.*{name}.*\(4, 2\).*\(1,\)"):
        fit.check_draws(x)
    x[BLOCK]["other"] = x[BLOCK].pop(name)
    with pytest.raises(ValueError, match=rf"{BLOCK}.*missing.*{name}.*extra.*other"):
        fit.check_draws(x)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_map_rejects_wrong_leaf_shape():
    """A scalar law on a vector cannot change its model leaf length.

    Truth source: the model fixes x at shape (2,).
    Tolerance basis: exact-algebra (integer dimensions).
    Fails if: dtype-only assignment lets MAP produce a (1, 3) cloud.
    """
    fit = _fit(Normal(0.0, 1.0), shape=(2,))
    backend = MAP(n_steps=1, init={"x": jnp.array([1.0, 2.0, 3.0])})
    with pytest.raises(ValueError, match=r"x.*\(3,\).*\(2,\)"):
        run(fit, backend, jax.random.key(0))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_leaf_assignment_checks_shape_while_tracing():
    """A traced single-draw assignment preserves the model shape.

    Truth source: model_at writes into a two-element leaf.
    Tolerance basis: exact-algebra (static shapes).
    Fails if: assignment accepts a different trailing shape during jit or vmap.
    """
    fit = _fit(Normal(0.0, 1.0), shape=(2,))
    apply = jax.jit(jax.vmap(lambda x: fit.model_at({"x": x}).x))
    assert apply(jnp.ones((4, 2))).shape == (4, 2)
    with pytest.raises(ValueError, match=r"x.*\(3,\).*\(2,\)"):
        apply(jnp.ones((4, 3)))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_posterior_checks_world_draw_and_leaf_shapes():
    """World and draw axes precede the whole model leaf in a posterior.

    Truth source: two worlds each hold four draws of a two-element vector.
    Tolerance basis: exact-algebra (static shapes).
    Fails if: the constructor only checks the leading axes or refuses worlds.
    """
    fit = _fit(Normal(0.0, 1.0), shape=(2,))
    post = Posterior(
        {"x": jnp.zeros((2, 4, 2))},
        jnp.full((2, 4), -jnp.log(4.0)),
        jnp.zeros((2, 4)),
        jnp.zeros(2),
        jnp.zeros(2),
        fit,
        None,
        "monte_carlo",
    )
    with pytest.raises(ValueError, match=r"x.*\(2, 4, 3\).*\(2,\)"):
        replace(post, samples={"x": jnp.zeros((2, 4, 3))})


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("backend", ["MAP", "shared"])
@pytest.mark.parametrize(
    ("entries", "fault"),
    [
        ({"x": jnp.zeros(2)}, "missing.*y"),
        ({"x": jnp.zeros(2), "y": 0.0, "z": 1.0}, "extra.*z"),
        ({"x": jnp.zeros((4, 2)), "y": jnp.zeros(3)}, r"y.*\(3,\).*\(4,\)"),
    ],
)
def test_init_checks_free_keys_and_common_batch(backend, entries, fault):
    """Each init entry describes the same batch of model draws.

    Truth source: the two free keys and their model shapes.
    Tolerance basis: exact-algebra (key and shape equality).
    Fails if: either backend path drops extra entries or assumes a common batch.
    """
    law = Normal(0.0, 1.0)
    fit = Fit(_Bowl(jnp.zeros(2), jnp.zeros(())), {"x": law, "y": law})
    with pytest.raises(ValueError, match=fault):
        if backend == "MAP":
            run(fit, MAP(init=entries), jax.random.key(0))
        else:
            start_points(fit, entries, jax.random.key(0), 4)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("backend", ["MAP", "shared"])
def test_posterior_start_is_checked_against_receiving_fit(backend):
    """Posterior starts must match the model that receives them.

    Truth source: the source holds vectors of length three and the run expects two.
    Tolerance basis: exact-algebra (static shapes).
    Fails if: validating the source posterior bypasses the receiving fit schema.
    """
    law = Normal(0.0, 1.0)
    source = _fit(law, shape=(3,))
    post = _cloud(source, n=4)
    fit = _fit(law, shape=(2,))
    with pytest.raises(ValueError, match=r"x.*3,?\).*\(2,\)"):
        if backend == "MAP":
            run(fit, MAP(init=post), jax.random.key(0))
        else:
            start_points(fit, post, jax.random.key(0), 2)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_schema_follows_an_edited_free_entry():
    """Editing a free entry in place changes the draw schema with it.

    Truth source: the schema of a fit is the shape of its current entries.
    Tolerance basis: exact-algebra (shapes).
    Fails if: the schema is computed once at construction and goes stale.
    """
    fit = _fit(Uniform(0.0, 1.0))
    wide = eqx.tree_at(lambda f: f.model.x, fit, jnp.zeros(3))
    assert wide.draw_schema()["x"].shape == (3,)
    assert wide.check_draws({"x": jnp.zeros((5, 3))}) == (5,)
