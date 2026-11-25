# Fit a scene from an image

You have a scene (a star, perhaps planets and a disk), an instrument, and an
{class}`~optixstuff.grids.Image` of electrons from an exposure. This page takes that image to a
posterior in stages. Each stage answers one question, and each hands a plain result to the next:

1. **Data into terms.** Score the image under the detector's own noise law.
2. **Stage one.** Free the amplitude of one candidate source with a law that allows any sign.
3. **Detect.** Ask where a source of unknown brightness would help most, over a grid of positions.
4. **Admit.** Put the source in the model only if its detection clears a stated threshold.
5. **Measure.** Maximize the likelihood from the detected start and report a width.
6. **Check.** Look at the residuals, and search for one more source at the best fit.
7. **Posterior.** Run `MAP` or `NUTS` with the priors you believe, from the measured start.

The reason for the order is that a posterior sampler or a MAP run needs a start inside the right
basin, and a position cannot be found by a gradient from a distance. The detection map finds the
basin, `measure` finds the peak in it, and only then do priors enter.

The example on this page is a one-dimensional line with two sources, so that every cell runs in a
few seconds. The calls are the ones an image fit makes; section 1 shows what changes when the
term is an image.

## 1. Data into terms

An {class}`~photomancy.image.ImageTerm` scores one exposure. It takes the `Image` that
{func}`coronagraphoto.observe` returns, the instrument that made it and the exposure, and it checks
the image grid against the detector grid when it is built. Choose the scored pixels once per
exposure. `on_support` stores the pixels covered by the reference model. To exclude bad pixels
or restrict the fit to a smaller field, pass a boolean mask of shape `(W, ny, nx)` shared by
all frames.

This block is not run with the documentation, which holds no rendered image or instrument.

```python notest
from photomancy.image import ImageTerm

terms = tuple(
    ImageTerm.on_support(scene, image, instrument=instrument, exposure=exposure)
    for image, exposure in zip(images, exposures)
)

# Keep only calibrated pixels in the reference footprint of the first exposure.
mask = terms[0].mask & good_pixels
first = ImageTerm(images[0], mask, instrument=instrument, exposure=exposures[0])
```

When a free instrument parameter or wavelength changes the field of view, a NaN image term
can mean that the proposed render lost coverage of a scored pixel. A larger footprint adds
no observations to the score. Choose a mask covered throughout the allowed parameter range.
`Fit` checks coverage at setup and at both prior endpoints for leaves that move the footprint.

One term per exposure: a frame that the model does not describe then shows up in its own sum of
squares (stage 6). The detector law (counting statistics, white) belongs to `ImageTerm`. For any
other forward model or noise law, use a {class}`~photomancy.infer.ForwardTerm`.

The running example uses a `ForwardTerm`: a flat background of 5 counts plus two Gaussian sources of
width 2 pixels, observed with unit Gaussian noise.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from priorities import LogNormal, Normal, Uniform
from scipy import stats

from photomancy.detection import curvature_block, detect, measure
from photomancy.infer import MAP, Fit, ForwardTerm, run
from photomancy.infer.likelihoods import gaussian_loglike

jax.config.update("jax_enable_x64", True)


class Sky(eqx.Module):
    amp0: jax.Array
    pos0: jax.Array
    amp1: jax.Array
    pos1: jax.Array


class Render(eqx.Module):
    x: jax.Array

    def __call__(self, sky):
        def source(amp, pos):
            return amp * jnp.exp(-0.5 * ((self.x - pos) / 2.0) ** 2)

        return 5.0 + source(sky.amp0, sky.pos0) + source(sky.amp1, sky.pos1)


x = jnp.arange(200.0)
render = Render(x)
truth = Sky(*(jnp.asarray(v) for v in (6.0, 70.3, 1.4, 140.0)))
data = render(truth) + np.random.default_rng(3).normal(0.0, 1.0, x.size)
term = ForwardTerm(render, gaussian_loglike(data, 1.0))
```

## 2. Stage one: an amplitude that may be zero or negative

The model for stage one holds every candidate source at zero amplitude, and the fit frees one
source: its amplitude and its position. The amplitude gets a law with signed support, a wide
`Normal`. A source that may be faint then has a maximum that is a point, not the edge of a
positive support, where no width exists (see {doc}`likelihood-ratio-map`). The width of the law
is irrelevant to this stage because neither `detect` nor `measure` reads the density.

```python continuation
absent = Sky(*(jnp.asarray(v) for v in (0.0, 100.0, 0.0, 100.0)))


def stage_one(model, k):
    """The fit that frees source k, with the amplitude allowed below zero."""
    free = {f"amp{k}": Normal(0.0, 100.0), f"pos{k}": Uniform(0.0, 200.0)}
    return Fit(model, free, (term,))


fit = stage_one(absent, 0)
```

For an image fit the amplitude is the source's `flux_scale` and the position is a
`SkyPositionBlock` on the same planet (`free = {"planets[0].flux_scale": Normal(0.0, 10.0),
"dynamics.orbits": SkyPositionBlock(...)}`), and at zero flux the source is absent from the
scene.

## 3. Detect over a grid

{func}`~photomancy.detection.detect` profiles the amplitude at each trial position and returns the
signed root `z` of the likelihood ratio, the amplitude `a_hat` and its formal width, using the
fit's own terms. The trials have one leading axis and the keys of the fit's free entries other
than the amplitude.

```python continuation
grid = jnp.linspace(2.0, 198.0, 40)
found = detect(fit, {"pos0": grid}, amplitude="amp0")
best = int(jnp.argmax(found.z))
print(float(grid[best]), float(found.z[best]))
```

The grid sets how well the peak lies on a trial: a source between trials loses some of its
`z`. In the test scene of this recipe the grid is 0.015 arcsec apart, the trial nearest the
injected planet is 0.002 arcsec from it, and it gives `z` = 26.6 against 27.7 at the injected
position, a loss of 4 percent. For a sky position the grid must also avoid the star itself. A trial exactly on the star has
no defined offset and returns NaN, so shift the grid off it and take the maximum with
`jnp.nanargmax`.

## 4. Admit a source only above the threshold

A source has free position entries only when its detection exceeds a threshold. Without one, a
position is fitted for a source whose amplitude the data cannot tell from zero, and its position is
meaningless. The threshold is a quantile of the largest `z` on the grid when no source is present,
because a search over a grid reports the largest value it finds.

The measured value for an imaging fit is 3.8: for white Poisson noise, a model equal to the truth,
a grid of 81 trial positions and two rolls, the 99th percentile of the per-map maximum of `z` is
3.77, with an interval from 3.59 to 3.84. The grid behind it is nine separations from 4 to 24
resolution elements (steps of 2.5) times nine position angles 40 degrees apart, on a face-on
circular orbit, with 20 frames of 360 s per roll. **It is calibrated under exactly that null and
grid.** It is not valid for a speckle-limited map, where the noise is not white and the model
is not equal to the truth, until that null has been measured; the values are larger there. A
denser or sparser grid has its own value, which has not been measured.

The example below is a one-dimensional line with 40 trials, so it takes a different number: the
Bonferroni bound `isf(0.01 / 40)`, which is the exact 99th percentile for 40 independent trials
and so errs high where neighboring trials overlap. It is a stand-in for the example, not a
calibrated threshold.

```python continuation
threshold = float(stats.norm.isf(0.01 / grid.size))
assert float(found.z[best]) > threshold  # admitted
print(threshold)
```

Below the threshold the source is left out of the model. Limits on an undetected source are not
part of this recipe.

## 5. Measure from the detected start

The start comes from the map, never from a guess or from the truth: the trial at the peak, and the
amplitude there. Three lines assemble it. Then {func}`~photomancy.detection.measure` maximizes the
likelihood in the support coordinates of each law and reports the covariance from the observed
information there.

```python continuation
start = {"amp0": found.a_hat[best], "pos0": grid[best]}
estimate = measure(fit, start)
print({k: float(v) for k, v in estimate.value.items()}, estimate.sigma)
```

### What `regular` certifies

The result says three things, and each is a field: `finite` (the log likelihood, gradient,
information and value are all finite), `converged` and `positive_definite`. `regular` is their
conjunction.

- **Converged** means stationary where the data constrain: `newton_decrement`, the nats the
  estimate still lies below the peak of the Gaussian that the covariance describes, is at most
  1e-3 (about 0.045 formal widths); the gradient has no component above 1e-3 along a direction
  the data do not constrain; and the last iterate is not below the start or the best the run
  reached. Both `grad_norm` and the decrement are unit free (the gradient is scaled by the
  square roots of the diagonal of the information).
- **Positive definite** means the smallest eigenvalue of the information in correlation form
  (scaled by the square roots of its diagonal, so that a coordinate in arcseconds, radians or
  flux units gives the same matrix) is above 1e-8. That is a floor on numerical rank: below it
  the matrix cannot be inverted. It does not say that every combination of parameters is well
  measured, which is the block test of the rule below.
- Three cases differ. A run that a step cap ended on a ridge is positive definite and not
  converged. The same run given more steps converges, and is still positive definite: the ridge
  has a maximum far from the source, where the flux is many times its width. A saddle is
  converged and not positive definite. `regular` is true only for a converged run that is
  positive definite, so it is true at that far maximum; the block test below is what removes it.

### The rule that keeps or removes a source

In order, for each source that the threshold admitted (a source below the threshold has no
position entries at all):

1. **Measure** from the detection start.
2. **Not finite, not converged, or ended by the step cap:** run `measure` once more from the same
   start with more steps. A run the cap ended (`stopped_early` is false) is rerun whatever its
   decrement, because the decrement of a walk along a ridge can sit just under its tolerance. If
   the rerun is still not finite and converged, the source comes out of the model.
3. **Converged:** decide whether the call is marginal. It is when the smallest eigenvalue of the
   source's own block (from {func}`~photomancy.detection.curvature_block`, scaled by its own
   diagonal, so that the same flux and position in other units give the same number) is at or
   below 0.05, or when the amplitude is under the detection threshold in marginal widths (the
   amplitude over the square root of its entry of `estimate.cov`). Otherwise the source stays.
4. **A marginal call goes to the refit:** fit again without the source and keep it only if
   `2 * delta lnL` exceeds the square of the threshold, the statistic the admission threshold is
   stated in.

The 0.05 is a screen and not a verdict. The block eigenvalue measures how collinear a
combination of the source's parameters is, not whether the data constrain the source: a source
detected at z = 26 whose flux and separation correlate at 0.97 has an eigenvalue of 0.03 and
is kept by the refit, and a source the data cannot place can have an eigenvalue of 0.5. On the
reduced imaging scene of the tests, a planet at 0.03 of the bright one's flux, admitted by
construction, converges far from the source with an eigenvalue of 8e-4 and `2 * delta lnL` of
3.4, so it comes out; the same planet from another start converges with an eigenvalue of 0.5 and an
amplitude under 3.8 widths (`2 * delta lnL` of 1.1), and also comes out.

`regular` is not part of the rule. It is true at a converged ridge, because the matrix there
is invertible; the screen and the refit are what find it.

```python continuation
block = curvature_block(
    fit.log_likelihood, estimate.value, lambda p: (p["amp0"], p["pos0"])
)
root = np.sqrt(np.abs(np.diag(block)))
eigenvalues = np.linalg.eigvalsh(np.asarray(block) / np.outer(root, root))
assert (
    bool(estimate.converged) and bool(estimate.finite) and bool(estimate.stopped_early)
)
amplitude_widths = abs(float(estimate.value["amp0"])) / float(estimate.sigma[0])
if eigenvalues.min() <= 0.05 or amplitude_widths < threshold:  # a marginal call
    without = fit.log_likelihood({**estimate.value, "amp0": jnp.zeros(())})
    assert 2.0 * (float(estimate.log_likelihood) - float(without)) > threshold**2
print(eigenvalues.min(), amplitude_widths)  # kept: neither trigger fires
```

`measure` is for tens of free scalars, not for free pixels. On a full-scene fit with positions
free for one source and four frames it took about 1200 s (1900 s with another job running) and 6.6 GB in one process. Run it in
a process of its own, and plan the memory for the compile of its programs, which dominates the
peak.

### More than one source

Sources are found one at a time. Hold the admitted source in the model at its estimate, free the
next candidate, and detect again. The found source is in the model, so the next map does not peak
on it. This loop is exercised here for two sources and is untested beyond that.

```python continuation
held = fit.model_at(estimate.value)
second = stage_one(held, 1)
found_second = detect(second, {"pos1": grid}, amplitude="amp1")
z_second = float(found_second.z.max())
print(z_second)
assert z_second < threshold  # not admitted: left out of the model
```

## 6. Check the model

Two checks use what already exists. The first is the whitened residual of each term,
{meth}`photomancy.infer.Fit.residuals`: one array per term, in the shape of its data, NaN where
a pixel is not scored. Its sum of squares is chi-square with the number of scored pixels minus the
number of fitted scalars as its degrees of freedom, so a frame the model does not describe lands
in the tail of that distribution. For counting statistics the residual is close to normal when
the counts per pixel are in the tens or more.

The second is the matched test for a point source that remains: run `detect` for one more source
at the best fit, over the same grid. No `|z|` above the threshold, positive or negative, should
appear. A strongly negative value means the model over-subtracts there.

The best fit comes from the next stage, so the checks follow it.

## 7. The posterior, with the priors you believe

Now the priors enter, from the measured start. Replace the stage-one laws by the ones you hold
(a positive flux scale, a position prior around the detection) with
{meth}`~photomancy.infer.Fit.with_priors`, or build the fit again, and run
{class}`~photomancy.infer.MAP`, {class}`~photomancy.infer.Laplace` or
{class}`~photomancy.infer.NUTS` with `init=estimate.value`.

```python continuation
final = Fit(
    held,
    {"amp0": LogNormal(0.0, 1.0), "pos0": Normal(70.0, 30.0)},
    (term,),
)
posterior = run(final, MAP(init=estimate.value), jax.random.key(0))
trace = posterior.info.stages[0]
assert trace.stopped_early  # converged, not the step cap
best_fit = jax.tree.map(lambda a: a[0], posterior.samples)

(residual,) = final.residuals(best_fit)
r = np.asarray(residual)
dof = x.size - 2
p = stats.chi2.cdf(float((r**2).sum()), dof)
assert 0.001 < p < 0.999

extra = detect(stage_one(final.model_at(best_fit), 1), {"pos1": grid}, amplitude="amp1")
assert float(jnp.abs(extra.z).max()) < threshold
```

Every optimizer call in the recipe stops on convergence. The default `MAP` stage runs in scaled
coordinates and stops when the loss has stalled and the scaled gradient norm is below 1e-3
(`stopped_early` is then true); `measure` uses the same stage, then polishes with Newton steps and reports `stopped_early`,
`n_steps` and `converged`; `Laplace` finds each mode with the same rule. A run that reaches its step count has not
converged. The unscaled `MAP` stage has no step-size-free test and runs all its steps.

For a disk orientation, or an orbit seen in projection, one point is not always the whole
answer. {doc}`staged-map-fit` says how to tell an alias from a mode and what to run for
each.
