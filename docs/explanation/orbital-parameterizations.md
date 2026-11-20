# Orbital parameterizations

An orbit fit is only as good as the coordinates it searches. This page explains how a fit
holds a planet's orbit, why those coordinates were chosen, and how the two orbit engines
of `photomancy.orbit` use them. It assumes the vocabulary of
[Fits and posteriors](fit-and-posterior): a model, its free entries, its terms, and the
backend that turns them into a `Posterior`.

## An orbit is a block of the model

The model of an orbit fit is an orbix `System`: a star, the planets, their stacked orbits,
and the sky frame. One planet at 10 pc around a solar-mass star looks like this.

```python
import jax
import jax.numpy as jnp
from orbix.system import Body, OrbitLaw, SkyFrame, System, ThieleInnesBlock
from priorities import LogUniform, Uniform
from skyscapes.population import kipping_eccentricity

jax.config.update("jax_enable_x64", True)

system = System.from_classical(
    Body(jnp.asarray(1.0)),  # star mass, solar masses
    Body(jnp.array([0.0])),  # planet masses, solar masses
    SkyFrame(jnp.asarray(10.0)),  # distance, pc
    a_AU=jnp.array([1.5]),
    e=jnp.array([0.3]),
    i_rad=jnp.array([0.8]),
    W_rad=jnp.array([1.2]),
    w_rad=jnp.array([0.5]),
    M0_rad=jnp.array([2.0]),
)
```

The orbit is freed by a block, a free entry that owns a node of the model and writes it
from coordinates of its own. The block holds a law for the orbit (here a log-uniform
period and the Kipping eccentricity distribution, with isotropic orientation and uniform
phase), a law for the star mass, the row of the planet it frees, and the path of the
`System` in the model, which is the root here.

```python continuation
law = OrbitLaw(
    period_d=LogUniform(100.0, 10_000.0), eccentricity=kipping_eccentricity(0.9)
)
block = ThieleInnesBlock(law, Uniform(0.9, 1.1), rows=(0,), system="")
print(block.coordinate_names)
print(block.writes)
```

The block has seven coordinates and writes two paths: the orbits and the star mass. The
rest of this page explains the seven coordinates one group at a time.

## The size and the shape

The first three coordinates fix the size and the shape of the orbit.

The first is the log of the mean motion `n` in radians per day. Because the period law is
bounded, the block maps the log mean motion from its bounded interval onto the whole real
line with a logit, and the coordinate takes the prefix `logit_`. Every real value of every
coordinate is then a valid orbit, so a sampler or an optimizer never meets a wall, and the
density gains the log derivative of the map.

The next two, `ux` and `uy`, are the eccentricity vector mapped from the unit disk onto
the plane, `(ex, ey) = u / sqrt(1 + |u|^2)`. The eccentricity vector
`(ex, ey) = e (cos w, sin w)` is used instead of the pair `(e, w)` because of the circular
limit. As `e` goes to zero the argument of periapsis loses its meaning, a posterior
develops a ridge over it, and the map from `(ex, ey)` back to `(e, w)` passes through a
square root and an `atan2` that are not differentiable at the origin. The Jacobian of that
map has a condition number that grows as `2 / e`. A Hessian evaluated in `(e, w)` at a
circular mode is undefined, and a Gaussian pushed into `(e, w)` near the circular limit
manufactures uncertainty that is not in the data. In the eccentricity vector the circular
orbit is an ordinary interior point, which is the reparameterization long recommended for
orbit sampling (Ford 2006).

The phase of the orbit needs no coordinate of its own. The orbit is stored at the epoch of
the model, and the eccentricity vector together with that epoch frame fixes where the
planet is on its orbit.

## The four sky constants

The last four coordinates, `A_AU`, `B_AU`, `F_AU` and `G_AU`, are the Thiele-Innes
constants: the projection of the orbit's frame onto the sky, in AU. They carry the
semi-major axis, the inclination and the two angles of the orientation, and with the
mean motion they imply the total mass, which is why this block also writes the star mass.

The constants matter because the sky position of the planet is linear in them. At a fixed
size and shape, the predicted east and north offsets at every epoch are a design matrix
times the four constants. With Gaussian astrometric errors the likelihood is therefore
Gaussian in the constants, and their conditional distribution given the shape is known
exactly. orbix computes the design matrix (`sky_design`) and the exact conditional
(`linear_gaussian_conditional`), and a measurement term that has this linear form says so
by its type (`AbstractLinearGaussianTerm` in `photomancy.infer`).

The consequence is that an orbit search never has to explore seven dimensions blindly.
Only the three coordinates of the size and the shape need a proposal, and the four
constants can be drawn from their exact conditional at each proposed shape.

## Two engines, named for the data in hand

`photomancy.orbit` provides two configured backends. Each is an ordinary backend of
`photomancy.infer`, passed to `run`, and the name of each says when to call it.

| Call | When | What it runs |
|---|---|---|
| `few_epoch()` | one or two astrometric epochs, with the orbit block the only free entry | `Importance(ScaleRotateProposal())` |
| `any_epoch()` | anything else: any number of epochs, radial velocity, a detection report, another free leaf | `AdaptiveImportance(CompletedProposal(ShapePriorProposal()))` |

`few_epoch` uses the scale-and-rotate move of Blunt et al. (2017). Each draw takes the
eccentricity, the inclination, the argument of periapsis and the phase from the prior,
draws a sky position near one measured epoch, and then solves for the size and the node
that put the orbit through that position. The draws are exact in the prior, so with one
epoch every weight is equal, and the engine costs about a quarter of `any_epoch`. As more
epochs arrive the draws spread over orbits that the later epochs rule out, the weights
concentrate on a few draws, and the evidence is refused under the k-hat rule of
[Fits and posteriors](fit-and-posterior). A fit with a free entry other than the orbit
block is refused by name.

`any_epoch` is the general engine. Round zero draws the size and shape from the block's
own law (`ShapePriorProposal`), `CompletedProposal` completes each draw with the four sky
constants from their exact conditional (a collapsed, or Rao-Blackwellized, proposal), and
`AdaptiveImportance` refits a Gaussian mixture to the weighted cloud over the following
rounds. The returned draws and the evidence come from the last round alone.

## One orbit, two sheets

Relative astrometry sees the orbit only in projection. An orbit and its reflection
through the sky plane, the same track traced on the near side instead of the far side,
predict identical positions at every epoch. The Thiele-Innes coordinates describe one of
these two sheets, and the block's density is the orbit law summed over both, so the
fit is normalized whichever sheet the truth is on.

A measurement that tells the sheets apart, radial velocity or a phase-dependent
brightness, must then be averaged over both. `SheetSum` holds such terms and scores them
on the orbit and on its reflection, and the fit refuses a second `SheetSum` over the same
planet, because a product of two sheet sums is not the sheet sum of the product. Terms that
cannot see the sheet, such as relative astrometry, stay outside it.

## Reading the orbit back

The coordinates are chosen for the search, not for reading. `classical_elements` draws
from a posterior, builds the model of each draw through the fit, and returns the classical
elements of one planet under the keys `period_d`, `a_AU`, `e`, `cos_i`, `W_rad` and
`M0_rad`. The [orbit fitting example](../examples/orbit_fitting) runs both engines on
synthetic data and reads the result this way.

## Other charts

The Thiele-Innes coordinates are one chart of several in orbix. A chart is a map between
an orbit and a vector of real coordinates, with the exact volume of that map, and a block
combines a chart with an orbit law and a placement in the model. `OrbitBlock` accepts the
classical, eccentricity-vector, frame and Cartesian charts and reads the star mass from
the model instead of implying it. Every backend of `photomancy.infer` sees only the flat
coordinates and the fit's density, so adding a chart touches no backend and no
measurement term. The engines of `photomancy.orbit` are built on the Thiele-Innes chart,
because only its coordinates make the astrometry linear.

## References

- Blunt, S. et al. (2017), "Orbits for the impatient: a Bayesian rejection-sampling method
  for quickly fitting the orbits of long-period exoplanets", *Astronomical Journal* 153, 229.
- Ford, E. B. (2006), "Improving the efficiency of Markov chain Monte Carlo for analyzing
  the orbits of extrasolar planets", *Astrophysical Journal* 642, 505.
- Kipping, D. M. (2013), "Parametrizing the exoplanet eccentricity distribution with the
  beta distribution", *MNRAS* 434, L51.
- Vehtari, A. et al. (2024), "Pareto smoothed importance sampling", *Journal of Machine
  Learning Research* 25, 1.
