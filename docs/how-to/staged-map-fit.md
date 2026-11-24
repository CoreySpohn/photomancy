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
without the other. Every free entry gets a wide Normal prior, which the last section
explains.

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
is a second `Fit` over a subset of the terms, a view of the exposure. The view must have the
same free entries as the run's fit, and a different set raises a `ValueError`; anything but a `Fit` raises a `TypeError`. The second
stage frees both groups and scores every term, the density of the run.

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

## Choices a stage can make

- `objective_scale` multiplies the objective the stage minimizes, so a stage can be
  down-weighted without changing its learning rates.
- `closed_form` sets a parameter by a formula rather than by a gradient. It is described
  in the next section.
- `fit` replaces the run's density for the stage with a `Fit` over a subset of the
  terms, as above. `prepare` is the same replacement made late: a function from the current
  parameters to the `Fit`, called once on the host when the stage starts.
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
convergence test and runs all `n_steps`.

## Give the solved coefficients a wide prior

A MAP fit returns the point where the whole objective, the prior included, is largest.
The coefficients that a stage solves for, by gradient or by `closed_form`, should
therefore carry a prior wide enough to be negligible next to the data, a Normal of width
`1e8` rather than a few units.

The prior pulls the optimum toward its mean by about its own gradient, `(x - m) / w**2`
for a Normal with mean `m` and width `w`, divided by how sharply the data constrain that
coefficient. For a coefficient the data pin down tightly the pull is invisible. For one the
data constrain only weakly, it is not: in an exposure fit whose detector seam coefficients
are weakly constrained, a Normal of width `1e3` moved them by about `1e-2` relative, while
a width of `1e8` matched the closed-form answer to `1e-13`. A wide prior does not change
what the data say, because it is flat across the range the data allow. The fits on this
page use `1e8` for that reason.

## Freezing policies

`freeze="hold"` is the one to use for a new fit. The two other policies reproduce what a
single `optax.multi_transform` optimizer does when a group is frozen by zeroing its
gradient (`"zero_gradient"`, the group keeps moving on its momentum) or by discarding its
update (`"drop_update"`, the group stays put while its moments keep accumulating).

The compiled step donates its working copies of the parameters and optimizer states, so the
`init` PyTree you pass in stays valid. There is no randomness, and a repeated run on CPU is
bitwise identical.
