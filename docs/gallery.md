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

# The viz reference

Every function `photomancy.viz` exports, with the smallest call that produces a
real picture. This page is a reference rather than a walkthrough: it is
organized by function, not by task, so it answers "what does this one draw?"
For the narrative versions, which build an argument around the figures, see
{doc}`examples/orbit_fitting` and {doc}`examples/eig_scheduling`.

`photomancy.viz` needs the `viz` extra:

```bash
pip install 'photomancy[viz]'
```

which brings [eyepiece](https://eyepiece.readthedocs.io) and, through it,
matplotlib and hwostyle. The base install stays free of all three, and the
plot functions are resolved lazily, so importing photomancy without the extra
still works right up until a plot function is touched.

Every figure below is executed when these docs are built, on seeded synthetic
data and nothing else, so a function that stops rendering fails the build
rather than the documentation host.

## The cast

One warm super-Earth, ten parsecs away, and a posterior over it. The samples
are drawn from a hand-built distribution around a known truth rather than from
a real fit, which keeps this page fast and deterministic; the shapes are
exactly what `OrbitProblem.to_physical` and `sample_physical` produce, so the
calls below are the calls you would make on a real posterior.

```{code-cell} ipython3
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from hwoutils.constants import AU2m, Mearth2kg, Msun2kg, Rearth2AU, d2s
from photomancy import viz
from photomancy.orbit import ImagingData, RVData, predict_photometry, predict_rv

hwostyle.use("dark")
jax.config.update("jax_enable_x64", True)

# Docs-builder concerns, not lines to copy into your own scripts. hwostyle asks
# for Inter/Helvetica/Arial and a CI builder has none of them, so name the face
# matplotlib always ships as a last resort, otherwise every figure emits a
# findfont warning. And render at a resolution that holds up on a high-DPI
# screen; the notebook default of 100 dpi does not.
matplotlib.rcParams["font.sans-serif"] = list(
    matplotlib.rcParams["font.sans-serif"]
) + ["DejaVu Sans"]
matplotlib.rcParams["figure.dpi"] = 160

AU_PER_D_TO_M_PER_S = AU2m / d2s
DIST_PC = 10.0
# The periapsis orientation is held fixed across the page so that the corner
# plot and the predictive panels describe the same planet.
COS_W, SIN_W = np.cos(0.6), np.sin(0.6)

TRUTH = {
    "T": 520.0,
    "a": 1.28,
    "e": 0.24,
    "cos_i": 0.42,
    "W": 2.1,
    "M0": 0.7,
    "Rp": 1.8 * Rearth2AU,
    "Ag": 0.30,
    "Mp": 5.0 * Mearth2kg,
    "Ms": Msun2kg,
}
TRUTH["Lambda"] = TRUTH["Ag"] * TRUTH["Rp"] ** 2
TRUTH["Mp_sini"] = TRUTH["Mp"] * np.sqrt(1.0 - TRUTH["cos_i"] ** 2)
TRUTH["tp"] = -TRUTH["M0"] * TRUTH["T"] / (2.0 * np.pi)


def posterior_fan(n, seed, width=1.0):
    """A physical-sample dict shaped like the one to_physical emits."""
    rng = np.random.default_rng(seed)
    jitter = lambda scale: scale * width * rng.standard_normal(n)  # noqa: E731

    s = {"T": TRUTH["T"] * (1.0 + jitter(0.03))}
    # The stellar mass is measured, not known, so Kepler's third law leaves a
    # correlated ridge in the (T, a) plane rather than pinning a to T exactly.
    Ms = TRUTH["Ms"] * (1.0 + jitter(0.05))
    s["a"] = (Ms / Msun2kg) ** (1.0 / 3.0) * (s["T"] / 365.25) ** (2.0 / 3.0)
    s["e"] = np.clip(TRUTH["e"] + jitter(0.05), 0.0, 0.85)
    s["cos_i"] = np.clip(TRUTH["cos_i"] + jitter(0.07), -1.0, 1.0)
    s["W"] = TRUTH["W"] + jitter(0.15)
    s["M0"] = TRUTH["M0"] + jitter(0.10)
    s["tp"] = -s["M0"] * s["T"] / (2.0 * np.pi)
    s["Rp"] = np.abs(TRUTH["Rp"] * (1.0 + jitter(0.12)))
    s["Ag"] = np.clip(TRUTH["Ag"] + jitter(0.05), 0.05, 0.9)
    s["Lambda"] = s["Ag"] * s["Rp"] ** 2
    s["Mp"] = np.abs(TRUTH["Mp"] * (1.0 + jitter(0.18)))
    s["Mp_sini"] = s["Mp"] * np.sqrt(np.clip(1.0 - s["cos_i"] ** 2, 0.0, 1.0))
    # to_physical also emits the deterministic log-likelihood site. It is a
    # diagnostic, not a parameter, and the corner defaults below drop it.
    s["ll_total"] = rng.standard_normal(n)
    return s


samples = posterior_fan(2500, seed=11)
sorted(samples)
```

## PARAM_LABELS

The axis-label vocabulary: one ASCII label for every key `to_physical` emits.
It is plain data, so it imports without the plotting stack and can be edited or
extended in place before a call.

```{code-cell} ipython3
viz.PARAM_LABELS
```

## default_corner_params

The curated corner-plot parameter list for a sample dict: the physically
meaningful, non-redundant subset, in a stable reading order. Anything absent,
anything not one-dimensional (a multi-planet batch needs an explicit `params`),
and `ll_total` are dropped. Every corner call below goes through this when
`params` is not given.

```{code-cell} ipython3
viz.default_corner_params(samples)
```

## plot_corner

The corner plot over physical orbital parameters. It takes either a
`{name: (n,)}` sample dict, as here, or a fitted posterior together with the
`OrbitProblem` and a PRNG key, in which case it draws the samples itself.
`truths` marks the generating values and `ranges` pins the axis limits so that
two figures stay comparable.

The `(T, a)` panel is the one to read first: the two are tied by Kepler's third
law, and the ridge is diagonal rather than a line only because the stellar mass
carries its own uncertainty.

```{code-cell} ipython3
result = viz.plot_corner(
    samples,
    params=["T", "a", "e", "cos_i"],
    truths=TRUTH,
    title="a five-epoch posterior",
)
```

## plot_corner_overlay

Several sample sets on one grid, for comparing backends against each other or a
belief against itself before and after new data. The second set here is the
same planet after three more epochs, which is the shape any scheduling argument
eventually has to show.

```{code-cell} ipython3
result = viz.plot_corner_overlay(
    [posterior_fan(2500, seed=11), posterior_fan(2500, seed=12, width=0.45)],
    params=["T", "e", "cos_i"],
    names=["1 epoch", "4 epochs"],
)
```

## plot_rv

Radial-velocity curves, converted from the library's native AU/day to the m/s a
reader expects. A `(T,)` array draws one curve and a `(K, T)` array draws a fan
of posterior-predictive draws, which is how a posterior is usually shown
against its data. Passing an `RVData` adds its valid prefix as errorbars,
converted the same way, so the two can never drift apart in units.

```{code-cell} ipython3
t_d = jnp.linspace(0.0, TRUTH["T"], 300)
n_draws = 80
draws = {k: jnp.asarray(v[:n_draws]) for k, v in samples.items()}

rv_fan = jax.vmap(
    lambda T, Mp_sini, e, tp: predict_rv(
        t_d, T, Msun2kg, Mp_sini, e, COS_W, SIN_W, tp
    )
)(draws["T"], draws["Mp_sini"], draws["e"], draws["tp"])

# Five epochs of the truth, with noise, entered the way real data would be.
rng = np.random.default_rng(4)
t_obs = np.array([15.0, 90.0, 180.0, 275.0, 360.0])
rv_err_m_s = 0.08
rv_truth = np.asarray(
    predict_rv(
        jnp.asarray(t_obs), TRUTH["T"], TRUTH["Ms"], TRUTH["Mp_sini"],
        TRUTH["e"], COS_W, SIN_W, TRUTH["tp"],
    )
)
rv_obs = rv_truth + (rv_err_m_s / AU_PER_D_TO_M_PER_S) * rng.standard_normal(
    t_obs.size
)
rv_data = RVData.pad(
    times=jnp.asarray(t_obs),
    rv=jnp.asarray(rv_obs),
    rv_err=jnp.full(t_obs.size, rv_err_m_s / AU_PER_D_TO_M_PER_S),
    inst_ids=jnp.zeros(t_obs.size, dtype=int),
    n_inst=1,
)

result = viz.plot_rv(np.asarray(t_d), np.asarray(rv_fan), data=rv_data)
```

## plot_dmag

Delta-magnitude against time, with the y axis inverted so that fainter is
downward. `limit_dmag` draws the detection limit. An `ImagingData` contributes
both of its epoch kinds: detections as errorbars at their measured dMag, and
nulls as downward triangles sitting at the limit, since a null has no measured
dMag of its own to sit at.

The planet in this posterior spends most of its period above the limit and
drops below it near conjunction, which is the whole reason scheduling a
direct-imaging revisit is a decision rather than a routine.

```{code-cell} ipython3
sep_arcsec, dmag = jax.vmap(
    lambda a, e, cos_i, W, tp, Lambda: predict_photometry(
        t_d, a, e, cos_i, W, COS_W, SIN_W, tp, Msun2kg, Lambda, DIST_PC
    )
)(
    draws["a"], draws["e"], draws["cos_i"], draws["W"], draws["tp"],
    draws["Lambda"],
)

# One instrument contrast curve, reused as the per-epoch limit grid.
cc_sep = np.linspace(0.045, 0.26, 60)
cc_dmag = 26.5 - 9.0 * np.exp(-(cc_sep - 0.045) / 0.035)

imaging = ImagingData.from_detections_and_nulls(
    det_epochs=np.array([120.0, 300.0]),
    det_dmag_obs=np.array([23.4, 22.9]),
    det_dmag_err=np.array([0.15, 0.15]),
    det_sep_grid=np.tile(cc_sep, (2, 1)),
    det_dmag0_grid=np.tile(cc_dmag, (2, 1)),
    null_epochs=np.array([500.0]),
    null_sep_grid=np.tile(cc_sep, (1, 1)),
    null_dmag0_grid=np.tile(cc_dmag, (1, 1)),
)

result = viz.plot_dmag(
    np.asarray(t_d), np.asarray(dmag), data=imaging, limit_dmag=26.5
)
```

## plot_detectability

The separation-dMag plane, which is where a contrast curve becomes a statement
about a specific planet. The predicted points are drawn against the limit
curve, and with the y axis inverted the undetectable ones sit visually below
it. `iwa_arcsec` shades the region inside the inner working angle.

The same posterior that produced the track above appears here as a loop:
brightest and closest in near quadrature, faint and unreachable at both
conjunctions.

```{code-cell} ipython3
result = viz.plot_detectability(
    np.asarray(sep_arcsec),
    np.asarray(dmag),
    contrast_curve=(cc_sep, cc_dmag),
    iwa_arcsec=0.045,
    scatter_kw={"s": 2, "alpha": 0.12, "linewidths": 0},
)
```

## plot_eig

The expected-information-gain decision panel: whichever of `total_eig`,
`alias_eig`, and `geometric_eig` a candidate evaluation returned, drawn against
candidate epoch. The total is the sum of the other two rather than a third peer
measurement, so it is drawn as a thick envelope underneath them instead of a
third equal line. Existing epochs enter as faint reference lines, and the best
candidate is marked in the total's color.

The values below are synthesized to make the two regimes legible in one figure.
{doc}`examples/eig_scheduling` runs the real `evaluate_candidates` and puts the
result in a closed loop.

```{code-cell} ipython3
candidates_d = np.linspace(0.0, 900.0, 400)
geometric = 0.35 + 0.12 * np.sin(2.0 * np.pi * candidates_d / TRUTH["T"])
alias = 1.1 * np.exp(-0.5 * ((candidates_d - 300.0) / 70.0) ** 2)

result = viz.plot_eig(
    candidates_d,
    {
        "total_eig": geometric + alias,
        "geometric_eig": geometric,
        "alias_eig": alias,
    },
    epochs_d=[0.0, 60.0, 130.0],
)
```
