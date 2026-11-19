---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Fitting an orbit as the epochs arrive

A directly imaged planet is measured a few times over its orbit, and each measurement
changes what is known about it. This example follows one planet through five epochs of
synthetic astrometry: two epochs fitted from scratch, then three more folded in one at a
time. The coordinates and the two engines it uses are explained in
[Orbital parameterizations](../explanation/orbital-parameterizations), and the posterior
operations in [Fits and posteriors](../explanation/fit-and-posterior).

## The planet

The truth is one planet at 1.5 AU from a solar-mass star 10 pc away, on an orbit with
eccentricity 0.3 and a period of about 670 days. The model of the fit is this same orbix
`System`; the fit will free its orbit and its star mass and hold everything else.

```{code-cell} ipython3
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from orbix.system import Body, OrbitLaw, SkyFrame, System, ThieleInnesBlock
from priorities import LogUniform, Uniform
from skyscapes.population import kipping_eccentricity

from photomancy import viz
from photomancy.infer import Fit, run, update
from photomancy.measurements import AstrometryData, RelativeAstrometryTerm
from photomancy.orbit import any_epoch, classical_elements, few_epoch

hwostyle.use("dark")
jax.config.update("jax_enable_x64", True)

# Docs-builder settings: a font every builder has, and a resolution that holds up
# on a high-DPI screen.
matplotlib.rcParams["font.sans-serif"] = list(
    matplotlib.rcParams["font.sans-serif"]
) + ["DejaVu Sans"]
matplotlib.rcParams["figure.dpi"] = 160

truth = System.from_classical(
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
period_true = float(truth.orbits.period_d[0])
period_true
```

## The measurements

Each epoch measures the separation to 10 milliarcseconds and the position angle to 4
degrees. The helper below lays out the epochs in an `AstrometryData` and lets the term
draw a noisy realization at the truth, which is how a simulation makes data from a known
orbit. The values passed for the separation and the position angle only fill the layout,
and `draw` replaces them.

```{code-cell} ipython3
def astrometry(times, key):
    """A relative astrometry term with one noisy epoch per time, drawn at the truth."""
    n = len(times)
    layout = AstrometryData.from_sep_pa(
        times=times,
        sep=[0.1] * n,
        sep_err=[0.01] * n,
        pa_deg=[0.0] * n,
        pa_err_deg=[4.0] * n,
    )
    return RelativeAstrometryTerm(layout).draw(key, truth)


def track(system, n=300):
    """East and north offsets (arcsec) of planet 1 over one of its periods."""
    t = jnp.linspace(0.0, system.orbits.period_d[0], n)
    return system.solve(t).offset_arcsec(1)


def sky_axes():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter([0.0], [0.0], marker="*", s=200, color="orange", zorder=4)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("east offset [arcsec]")
    ax.set_ylabel("north offset [arcsec]")
    return fig, ax


first = astrometry([0.0, 60.0], jax.random.key(0))

fig, ax = sky_axes()
ax.plot(*track(truth), color="0.6", lw=1, label="true orbit")
ax.scatter(first.data.ra, first.data.dec, color="crimson", zorder=3, label="epochs")
ax.legend()
plt.show()
```

Two epochs 60 days apart sample a short arc of the orbit. Many orbits pass through both
points, and the fit has to say which ones.

## Two epochs: the scale-and-rotate engine

The orbit is freed by a `ThieleInnesBlock`. Its law gives the period a log-uniform prior
from 100 to 10,000 days and the eccentricity the Kipping distribution, with isotropic
orientation and uniform phase, and the star mass gets a uniform prior from 0.9 to 1.1
solar masses. The fit is the model, that one free entry, and the astrometry term.

```{code-cell} ipython3
law = OrbitLaw(
    period_d=LogUniform(100.0, 10_000.0), eccentricity=kipping_eccentricity(0.9)
)
block = ThieleInnesBlock(law, Uniform(0.9, 1.1), rows=(0,), system="")
fit = Fit(truth, {"orbits": block}, (first,))

posterior = run(fit, few_epoch(n_draws=2**16), jax.random.key(1))
print(f"log evidence {float(posterior.log_evidence):.3f} "
      f"+- {float(posterior.log_evidence_se):.3f}")
```

With two epochs and the orbit as the only free entry, `few_epoch` is the engine to call:
each draw is an orbit from the prior rotated and scaled to pass near one measured
position, so the draws already sit where the data allow. The draw count is set to
`2**16`, the size `any_epoch` returns, so that the later epochs can fall back to a refit
with the same layout.

Drawing orbits from the posterior and tracing each over its period shows what two epochs
leave open.

```{code-cell} ipython3
def draw_tracks(posterior, key, n=60):
    """Sky tracks of ``n`` posterior draws, each over its own period."""
    draws = posterior.sample(key, (n,))
    systems = jax.vmap(posterior.fit.model_at)(draws)
    return jax.vmap(track)(systems)


east, north = draw_tracks(posterior, jax.random.key(2))
fig, ax = sky_axes()
ax.plot(np.asarray(east).T, np.asarray(north).T, color="cyan", lw=0.5, alpha=0.35)
ax.plot(*track(truth), color="0.6", lw=1.5)
ax.scatter(first.data.ra, first.data.dec, color="crimson", zorder=3)
plt.show()
```

Every track passes through both epochs, and away from them the tracks fan out into orbits
of very different size and shape. `classical_elements` reads the period of each draw in
days, and its spread says the same thing in numbers.

```{code-cell} ipython3
elements = classical_elements(posterior, jax.random.key(3), 4000, orbit_path="orbits")
np.percentile(elements["period_d"], [5, 50, 95])
```

## Three more epochs, one at a time

Each later epoch is folded in with `update`. It first tries to reweight the existing
cloud by the new term (`Posterior.append_term`), which costs one evaluation of the term
per draw. If the reweighted cloud fails the gate (fewer than `min_n_eff` effective draws,
or a k-hat above `max_khat`), it refits the new fit with the backend it is given. With
three or more epochs the general engine, `any_epoch`, is that backend. The cloud is
resampled to equal weights before each update, so that the gate judges the new epoch and
not the tail of the old weights.

```{code-cell} ipython3
history = [(2, np.percentile(elements["period_d"], [5, 50, 95]))]
for i, t in enumerate([250.0, 450.0, 700.0]):
    term = astrometry([t], jax.random.key(10 + i))
    posterior = update(
        posterior.resample(jax.random.key(20 + i)),
        term,
        any_epoch(),
        jax.random.key(30 + i),
        min_n_eff=1000,
        max_khat=0.7,
    )
    elements = classical_elements(
        posterior, jax.random.key(3), 4000, orbit_path="orbits"
    )
    history.append((3 + i, np.percentile(elements["period_d"], [5, 50, 95])))
    print(f"{t:5.0f} d: log evidence {float(posterior.log_evidence):.3f}")
```

The evidence of each step is finite here. When an estimate fails its own validity check
(a k-hat of the importance weights above the limit for the draw count), the evidence and
its error come back NaN while the draws are still returned. A NaN is a refusal, never a
number to compare.

The 90 percent interval of the period narrows as the epochs arrive, from thousands of
days with two epochs to tens of days with five, and the median settles on the true
period.

```{code-cell} ipython3
n_epochs = [n for n, _ in history]
low, mid, high = np.array([q for _, q in history]).T
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.fill_between(n_epochs, low, high, color="cyan", alpha=0.3, label="90% interval")
ax.plot(n_epochs, mid, color="cyan", marker="o", label="median")
ax.axhline(period_true, color="0.6", ls="--", label="truth")
ax.set_yscale("log")
ax.set_xlabel("epochs")
ax.set_ylabel("period [days]")
ax.legend()
plt.show()
```

## Five epochs

The tracks of the final posterior lie on top of one another along the whole orbit, and a
corner plot of the classical elements shows the remaining uncertainty and its
correlations. `viz.plot_corner` takes the posterior directly and reads the elements
through the fit.

```{code-cell} ipython3
east, north = draw_tracks(posterior, jax.random.key(5))
epochs = [first.data, *(term.data for term in posterior.fit.terms[1:])]
fig, ax = sky_axes()
ax.plot(np.asarray(east).T, np.asarray(north).T, color="cyan", lw=0.5, alpha=0.35)
ax.plot(*track(truth), color="0.6", lw=1.5)
for data in epochs:
    valid = np.asarray(data.is_valid)
    ra, dec = np.asarray(data.ra)[valid], np.asarray(data.dec)[valid]
    ax.scatter(ra, dec, color="crimson", zorder=3)
plt.show()
```

```{code-cell} ipython3
classical = truth.orbits.to_classical()
truths = {
    "period_d": period_true,
    "a_AU": float(classical["a_AU"][0]),
    "e": float(classical["e"][0]),
    "cos_i": float(np.cos(classical["i_rad"][0])),
}
result = viz.plot_corner(
    posterior,
    key=jax.random.key(4),
    orbit_path="orbits",
    params=["period_d", "a_AU", "e", "cos_i"],
    truths=truths,
)
```

The period and the semi-major axis lie on a ridge because the star mass is not known
exactly. Kepler's third law ties the two only through the mass, and its prior allows ten
percent either way.
