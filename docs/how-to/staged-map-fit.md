# Fit a model in stages with the MAP backend

Some fits cannot free every parameter at once. A coronagraph model, for example, is fitted
by first locating the star, then adjusting the optics, then holding the optics while the
science exposure is fitted. {class}`~photomancy.infer.MAP` runs that kind
of fit on a `Fit`: a sequence of stages, each of which frees some named groups of the free
entries. The result is a {class}`~photomancy.infer.Posterior` with one draw, the optimum, at
weight one. It has no evidence, so `log_evidence` is NaN, and its `info` is a
{class}`~photomancy.infer.MAPInfo` that holds the trace of every stage.

## Build the fit

The fit below has a model with two leaves, `shape` (two numbers) and `offset` (one number),
and three measurements. The first two measurements see `shape + offset`, and the third
sees `offset` alone. Each group of measurements is one term, so a stage can score one term
without the other. Every free entry gets a wide Normal prior, so that the prior does not move
the optimum in this toy fit (see "The likelihood, not the posterior").

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from priorities import Normal

from photomancy.infer import (
    MAP,
    Fit,
    ForwardTerm,
    MAPStage,
    group_labels,
    run,
)
from photomancy.infer.likelihoods import gaussian_loglike

jax.config.update("jax_enable_x64", True)


class Model(eqx.Module):
    shape: jax.Array
    offset: jax.Array


class Render(eqx.Module):
    rows: jax.Array

    def __call__(self, m):
        everything = jnp.concatenate([m.shape, jnp.zeros(1)]) + m.offset
        return everything[self.rows]


truth = Model(shape=jnp.array([4.0, 6.0]), offset=jnp.asarray(2.0))
data = Render(jnp.array([0, 1, 2]))(truth)
first = ForwardTerm(Render(jnp.array([0, 1])), gaussian_loglike(data[:2], 0.1))
second = ForwardTerm(Render(jnp.array([2])), gaussian_loglike(data[2:], 0.1))

init = Model(shape=jnp.zeros(2), offset=jnp.asarray(0.0))
free = {
    "shape": Normal(jnp.zeros(2), 1e8 * jnp.ones(2)),
    "offset": Normal(jnp.asarray(0.0), jnp.asarray(1e8)),
}
fit = Fit(init, free, (first, second))
```

A `ForwardTerm` pairs a forward model, here `Render`, with a normalized log likelihood of the
data. The forward model is an Equinox module so that any array it holds stays a leaf of
the term.

## Name the groups

A stage frees groups, so the first job is to say which free entries belong to which group.
The optimizer works on the free entries in unconstrained coordinates, which is the dict
`fit.to_real` returns for one draw: one key per free entry, holding that entry's array. For
a Normal prior the unconstrained value equals the stored value. A group is a selector that
picks leaves out of that dict, in the form `equinox.tree_at` takes. Nothing is a group by
default, so an entry that should stay fixed is never fitted by accident.

```python continuation
start = {"shape": init.shape, "offset": init.offset}
labels = group_labels(
    start,
    {"shape": lambda u: u["shape"], "offset": lambda u: u["offset"]},
)
```

## Write the stages

A stage names the groups it frees, and every group has an `optax` transform. A group that a
stage does not free keeps its values and its optimizer state exactly (`freeze="hold"`).

The first stage below frees `shape` alone and scores only the first term. Its `fit`
is a second `Fit` over a subset of the terms, a view of the exposure. A stage may
change the terms and prior densities while keeping the same free keys, trailing
shapes and coordinate maps as the run fit. Changing a mapped bound raises a
`ValueError`, for example `stage Fit coordinates differ at 'x': support bounds differ`.
A stage density of another type raises a `TypeError`. The second stage frees both
groups and scores every term, the density of the run.

```python continuation
backend = MAP(
    labels=labels,
    transforms={"shape": optax.adam(0.2), "offset": optax.adam(0.2)},
    stages=(
        MAPStage(
            n_steps=400,
            groups=("shape",),
            fit=Fit(init, free, (first,)),
            name="first exposure",
        ),
        MAPStage(n_steps=600, groups=("offset", "shape"), name="together"),
    ),
    init=start,
)
posterior = run(fit, backend, jax.random.key(0))
assert abs(float(posterior.samples["offset"][0]) - 2.0) < 1e-3
assert jnp.allclose(posterior.samples["shape"][0], jnp.array([4.0, 6.0]), atol=1e-3)
```

`run` returns the optimum as the single draw of `posterior.samples`, under the keys of the
free entries. The loss of each stage is in `posterior.info.stages`, one trace per stage.
Without `init`, the fit starts from a draw of the prior, and `init` may also be one
constrained draw or an earlier `Posterior`, whose highest-density draw is used. With no
`stages`, the backend runs one stage of at most `n_steps` L-BFGS iterations over every
free entry. The default stage, which runs in scaled coordinates (see below), stops
earlier only once it has converged: five consecutive steps (checked every five) fail to
lower the best objective by more than 1e-7 and the gradient norm in the scaled
coordinates is below 1e-3, and the stage trace then reports `stopped_early=True`. A loss
stall alone does not stop it. The unscaled default stage (`scale=False`, or a fit whose
terms have no Fisher root) runs all `n_steps` and never reports `stopped_early`. Explicit
`stages` run as written: stopping applies only to a stage that sets `early_stopping`.
A non-finite loss is a failure, not a stall: the stage ends with `failed=True` on its trace and
`stopped_early=False`, and the parameters returned are the last iterate, which can be non-finite.

## Choices a stage can make

- `objective_scale` multiplies the objective the stage minimizes, so a stage can be
  down-weighted without changing its learning rates.
- `closed_form` sets a parameter by a formula rather than by a gradient. It is described
  in the next section.
- `fit` replaces the run's density for the stage with a `Fit` over a subset of the
  terms, as above. `prepare` is the same replacement made late: a function from the current
  parameters to the `Fit`, called once on the host when the stage starts. It receives JAX arrays, and so does a
  callback record.
- `transforms` replaces the table of optimizers and starts them fresh at the current values.
  Without it the stage continues the optimizer states and their schedule clocks.
- `early_stopping` ends the stage once the objective stops improving.
- `solver` swaps the per-group optimizers for one `optax` line-search solver, such as
  `optax.lbfgs()`, that optimizes all the freed groups together.

{func}`~photomancy.infer.rescaled_transform` runs a transform on
`params / scale` so that a leaf kept in nanometers takes the steps of a fit in meters.

## Set a parameter by a formula

Some parameters have a closed-form value once the rest are known, such as a flux set by the
ratio of the data to the prediction. A stage's `closed_form` is a function
`(u, aux) -> u` applied after each optimizer update. Its first argument `u` is the dict of
free entries after the update. Its second argument `aux` is the tuple of predictions made
before the update: one entry per term, each the `predict` output of that term at the
parameters the gradient was taken at, or `None` for a term that has no prediction. The
metrics and the callback of the backend receive the same `aux`.

The order matters. The prediction describes the parameters before the step, while `u` holds
the parameters after it, so a formula that corrects a parameter from the residual lags the
optimizer by one step. It still has the right fixed point, which is what the example
needs. The stage below frees `shape` only and gives `offset` the empty label, so no
gradient moves it. After each step, `closed_form` shifts `offset` by the mean residual
between the data and the pre-step predictions of both terms.

```python continuation
def closed_form(u, aux):
    prediction = jnp.concatenate([aux[0], aux[1]])
    return {**u, "offset": u["offset"] + jnp.mean(data - prediction)}


formula = MAP(
    labels={"shape": "shape", "offset": ""},
    transforms={"shape": optax.adam(0.2)},
    stages=(MAPStage(n_steps=800, groups=("shape",), closed_form=closed_form),),
    init=start,
)
posterior = run(fit, formula, jax.random.key(0))
assert abs(float(posterior.samples["offset"][0]) - 2.0) < 1e-3
```

A fit with a closed form renders each term twice per step, once for the gradient and once
for the prediction, so it compiles a larger program than a fit without one.

## Scaling of the coordinates

`MAP` scales its unconstrained coordinates by the information scale, `1 / sqrt(diag
Fisher)` (with the prior's curvature added, and a cap so a flat direction does not
stretch the step), through {func}`~photomancy.infer.rescaled_transform`. A fit whose
parameters differ by many orders of magnitude, such as a flux and a position, then takes
the same L-BFGS steps in every coordinate. Pass `scale=False` to start from unscaled
coordinates; a fit whose terms have no Fisher root falls back to them. The scaled stage
stops once it has converged (the loss has stalled and the scaled gradient norm is below
1e-3), so `n_steps` is the most steps it takes; the unscaled stage has no step-size-free
convergence test and runs all `n_steps`. `Laplace` finds each mode with the same stage
and the same stopping rule.

`Fit` converts the leaves of the model it is given with `hwoutils.leaves.canonical`: NumPy
arrays, NumPy scalars, Python floats and complex numbers, and weakly typed arrays become JAX
arrays in the active precision. Python `int` and `bool` fields (an `IntEnum` included) are left as
they are: under `eqx.filter_jit` they are compile-time constants, so a model may use one for an
array size or a branch.
`fit.model` is the converted tree, so read the base values from it rather than from the object
you passed in.

## The likelihood, not the posterior

A MAP fit returns the point where the whole objective, the prior included, is largest. To ask
for the peak of the likelihood and its width, do not make the prior negligible: use
{func}`~photomancy.detection.measure` with a law of signed support on each amplitude, which
reads no prior density, and see {doc}`fit-a-scene`. The wide Normal laws on this page keep the
prior out of the way in a toy fit, but a prior of width `1e8` is a numerical hazard in a
fit with many parameters and gives no width, so it is not the route to a likelihood summary.

A coefficient that a stage solves for (by gradient or by `closed_form`) and that the data
constrain only weakly is pulled toward the prior mean by about `(x - m) / w**2` for a
Normal of mean `m` and width `w`, divided by how sharply the data constrain it. When that pull
matters, say so in the prior you choose rather than hiding it.

## Aliases and modes

One `MAP` point is one answer. Two kinds of repeated answer look alike and are not.

- **An alias is the same model written another way.** The disk orientation `(i, PA)` has four
  such points: `(i, PA)`, `(180 - i, PA + 180)`, `(i + 180, PA)`, and the two radial slopes
  exchanged. The images agree to rounding, so the four points have the same log likelihood,
  and the data cannot tell them apart because there is nothing to tell. They are one disk. The
  log prior can differ between them (it does for `default_disk_prior`), and the fit ends at
  whichever is nearer the start or higher in density.
- **A mode is a different model that the data cannot exclude.** An orbit at inclination `i` or
  `180 - i` from sky positions alone is a pair of such modes: the two have different log
  likelihoods, and each is a separate answer with its own weight.

To tell them apart, evaluate `fit.log_likelihood` at the relabeled point. Equal to rounding
(relative `1e-10`) is an alias, and the value is the same disk. Different is a mode.

After a `MAP` run on a fit with an alias-prone orientation, evaluate the log density at the
relabeled points of the optimum, a handful of evaluations, and report the highest. A start
near edge-on can end in an alias other than the one you expect. The example below does this
for a model that depends on its parameter only through `s**2`, so `s` and `-s` are aliases.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from priorities import Normal

from photomancy.infer import MAP, Fit, ForwardTerm, Laplace, run
from photomancy.infer.likelihoods import gaussian_loglike

jax.config.update("jax_enable_x64", True)


class Shape(eqx.Module):
    s: jax.Array


class Curve(eqx.Module):
    tilt: float = eqx.field(static=True)

    def __call__(self, m):
        k = jnp.arange(1.0, 6.0)
        return jnp.concatenate([k * m.s**2, jnp.atleast_1d(self.tilt * m.s)])


def build(tilt):
    render = Curve(tilt)
    data = render(Shape(jnp.asarray(2.0))) + np.random.default_rng(0).normal(
        0.0, 0.5, 6
    )
    term = ForwardTerm(render, gaussian_loglike(data, 0.5))
    return Fit(Shape(jnp.asarray(1.0)), {"s": Normal(0.0, 5.0)}, (term,))


def point(fit, start):
    posterior = run(fit, MAP(init={"s": jnp.asarray(start)}), jax.random.key(0))
    return jax.tree.map(lambda a: a[0], posterior.samples)


fit = build(tilt=0.0)  # symmetric: s and -s are one answer
x = point(fit, -1.0)
mirror = {"s": -x["s"]}
assert abs(float(fit.log_likelihood(x)) - float(fit.log_likelihood(mirror))) < 1e-10
```

The same relabeling in a model that is not symmetric is a different answer. With `tilt=0.8` the
mirror point has a log likelihood 20 nats lower, so it is a mode: a second, much less probable
answer, not a second way of writing the first.

```python continuation
fit = build(tilt=0.8)
x = point(fit, 1.0)
assert float(fit.log_likelihood(x)) - float(fit.log_likelihood({"s": -x["s"]})) > 10.0
```

A mode is reported by running from one start per candidate. {class}`~photomancy.infer.Laplace`
takes `n_modes` and an `init` with that many starts, and puts a Gaussian at each mode, weighted
by its evidence; `posterior.info.n_modes_kept` counts the modes it kept, and
`posterior.info.mode_log_evidence` says how much each is worth. For a sampler, start
{class}`~photomancy.infer.NUTS` chains in both candidates. A single `MAP` point reports one
mode, so a posterior with two modes needs one of these.

```python continuation
two = run(
    fit,
    Laplace(n_draws=200, n_modes=2, init={"s": jnp.array([2.0, -2.0])}),
    jax.random.key(1),
)
assert int(two.info.n_modes_kept) == 2
evidence = np.asarray(two.info.mode_log_evidence)
assert evidence[0] - evidence[1] > 10.0
```

Random restarts are for a fit with no known symmetry and a residual that looks wrong, not a
routine step: where the alias or the mirror is known, start from it.

## Freezing policies

`freeze="hold"` is the one to use for a new fit. The two other policies reproduce what a
single `optax.multi_transform` optimizer does when a group is frozen by zeroing its
gradient (`"zero_gradient"`, the group keeps moving on its momentum) or by discarding its
update (`"drop_update"`, the group stays put while its moments keep accumulating).

The compiled step donates its working copies of the parameters and optimizer states, so the
`init` PyTree you pass in stays valid. There is no randomness, and a repeated run on CPU is
bitwise identical.
