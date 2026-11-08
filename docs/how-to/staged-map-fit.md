# Fit a PyTree in stages with a MAP backend

Some fits cannot free every parameter at once. A coronagraph model, for example, is fitted
by first locating the star, then adjusting the optics, then holding the optics while the
science exposure is fitted. {class}`~photomancy.backends.staged.MAPBackend` runs that kind
of fit: a sequence of stages, each of which frees some named groups of parameters, over a
parameter PyTree that keeps its structure. The result is a point estimate, a
{class}`~photomancy.posterior.PointPosterior`, with no covariance and no evidence.

## Name the groups

A group is a selector that picks leaves out of the parameter PyTree, in the form
`equinox.tree_at` takes. Nothing is a group by default, so a floating-point array that
should stay fixed is never fitted by accident.

```python
import equinox as eqx
import jax.numpy as jnp
import optax

from photomancy.backends.staged import FitStage, MAPBackend, group_labels


class Model(eqx.Module):
    shape: jnp.ndarray
    offset: jnp.ndarray
    fixed: jnp.ndarray


init = Model(
    shape=jnp.array([0.0, 0.0]), offset=jnp.array(5.0), fixed=jnp.array([1.0, 2.0])
)
labels = group_labels(
    init,
    {"shape": lambda m: m.shape, "offset": lambda m: m.offset},
)
```

## Write the log-density and the stages

The log-density is any callable from the PyTree to a scalar. A stage names the groups it
frees, and every group has an `optax` transform. A group that a stage does not free keeps
its values and its optimizer state exactly (`freeze="hold"`).

```python
def logdensity(m):
    return (
        -0.5 * jnp.sum((m.shape + m.offset * m.fixed - jnp.array([4.0, 6.0])) ** 2)
        - 0.5 * (m.offset - 2.0) ** 2
    )


backend = MAPBackend(
    labels=labels,
    transforms={"shape": optax.adam(0.1), "offset": optax.adam(0.05)},
    stages=(
        FitStage(n_steps=300, free=("shape",), name="shape first"),
        FitStage(n_steps=300, free=("offset", "shape"), name="together"),
    ),
)
posterior = backend.run(logdensity, init)
assert abs(float(posterior.params.offset) - 2.0) < 0.05
```

`run` returns the fitted PyTree in `posterior.params`. To read the loss of each stage, call
`backend.fit(logdensity, init)`, which returns the same parameters with one trace per stage.

## Choices a stage can make

- `objective_scale` multiplies the objective the stage minimizes, so a stage can be
  down-weighted without changing its learning rates.
- `closed_form` is a function `(params, aux) -> params` applied after each optimizer update,
  for a parameter that is set by a formula rather than by a gradient (a flux set by a ratio
  of windows, for example). It sees the `aux` of the evaluation that produced the update,
  that is, the prediction made before the step.
- `logdensity` replaces the run's log-density for the stage, for instance a view over one
  exposure of an {class}`~photomancy.exposure.ExposureLogDensity`.
- `transforms` replaces the table of optimizers and starts them fresh at the current values.
  Without it the stage continues the optimizer states and their schedule clocks.
- `early_stopping` ends the stage once the objective stops improving.

A log-density that also returns auxiliary output derives from
{class}`~photomancy.core.model.AbstractLogDensity` and overrides `value_and_aux`.
{func}`~photomancy.backends.staged.rescaled_transform` runs a transform on `params / scale`
so that a leaf kept in nanometers takes the steps of a fit in meters.

## Freezing policies

`freeze="hold"` is the one to use for a new fit. The two other policies reproduce what a
single `optax.multi_transform` optimizer does when a group is frozen by zeroing its
gradient (`"zero_gradient"`, the group keeps moving on its momentum) or by discarding its
update (`"drop_update"`, the group stays put while its moments keep accumulating).

The compiled step donates its working copies of the parameters and optimizer states, so the
`init` PyTree you pass in stays valid. There is no randomness, and a repeated run on CPU is
bitwise identical.
