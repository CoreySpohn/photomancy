# Map the evidence for a source with a likelihood-ratio map

A fit that leaves a source out can be asked a direct question: if a source of unknown
brightness were added at this position, how much better would the data be fitted? Asking
it at every trial position gives a detection map. {func}`~photomancy.detection.likelihood_ratio_map`
answers it from a base prediction (the best fit without the source) and a function that
renders the source at one position. It never renders anything itself, so the optics and
detector stay in your own code.

## The statistic

At a trial position `x` the source image at unit amplitude is the template `t(x)`. The
amplitude `a` is profiled out, and the map reports

    Lambda(x) = 2 [max_a log L(base + a t(x)) - log L(base)],   z(x) = sign(a_hat) sqrt(Lambda).

For a Gaussian likelihood with residual `r = data - base` and weights `W = 1 / sigma**2`,
this is the whitened matched filter: `a_hat = t'Wr / t'Wt` and `z = t'Wr / sqrt(t'Wt)`. If
the data are the base prediction plus Gaussian noise, `z` has zero mean and unit variance
at every position.

## Profile one amplitude

{func}`~photomancy.detection.profile_amplitude` maximizes over the amplitude at one
position with a fixed number of Newton steps. It takes any likelihood builder from
{mod}`photomancy.infer.likelihoods` that is curved in the prediction.

```python
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from photomancy.detection import likelihood_ratio_map, profile_amplitude
from photomancy.infer.likelihoods import gaussian_loglike, huber_loglike

rng = np.random.default_rng(0)
x = np.linspace(-1.0, 1.0, 64)


def bump(center):
    return np.exp(-0.5 * ((x - center) / 0.05) ** 2)


base = 1.0 + 0.2 * x  # best fit of the model without the source
data = base + 0.7 * bump(0.3) + 0.05 * rng.normal(size=x.size)
likelihood = gaussian_loglike(data, 0.05)

a_hat, two_delta, curvature = profile_amplitude(likelihood, base, bump(0.3))
print(float(a_hat), float(two_delta), 1.0 / float(jnp.sqrt(curvature)))
assert abs(float(a_hat) - 0.7) < 0.1
```

A Gaussian likelihood is quadratic in the amplitude, so one step is exact; a Huber one is
piecewise quadratic and settles in a few (`n_newton=4` by default). The Laplace builder
has zero curvature and is rejected with a `ValueError`. Pixels the likelihood masks out
may be NaN in `base` and in the template: they add nothing to the value or the
derivatives.

## Map it over positions

Each term is `(likelihood, base_prediction, template_fn)`. The template function is an
`eqx.Module`, so any array it holds is a traced input rather than a compiled constant.
Several terms (several exposures) share one amplitude at each position.

```python continuation
class BumpTemplate(eqx.Module):
    grid: jnp.ndarray
    width: float = eqx.field(static=True)

    def __call__(self, center):
        return jnp.exp(-0.5 * ((self.grid - center) / self.width) ** 2)


centers = jnp.linspace(-0.9, 0.9, 181)
terms = ((likelihood, base, BumpTemplate(jnp.asarray(x), 0.05)),)
detection = likelihood_ratio_map(terms, centers, batch_size=32)

best = int(jnp.argmax(detection.z))
print(float(centers[best]), float(detection.z[best]), float(detection.a_hat[best]))
assert abs(float(centers[best]) - 0.3) < 0.03
assert float(detection.z[best]) > 10.0
```

`detection.a_hat` is the amplitude, `detection.z` the signed root of the statistic and
`detection.sigma` the formal standard error of `a_hat`, `1 / sqrt(curvature)`, which
for a Gaussian likelihood equals `a_hat / z`. Positions are processed with
{func}`jax.lax.map`, `batch_size` at a time (the default of 1 holds one position's
intermediates; `None` maps all at once, the fast choice on a GPU), so `batch_size` sets
the memory use. The renderer runs once per call rather than once per position, which
assumes the prediction is linear in the amplitude. A
batched map agrees with the one-at-a-time map to rounding, not bit for bit.

The significance holds for the likelihood you give it. A Huber likelihood gives a map
that resists outliers, but its `z` is then a likelihood-ratio root rather than a matched
filter, and its null distribution has to be calibrated (against residuals of the same
fit, for example) instead of assumed to be a unit Gaussian.

## Check that a group of parameters can be freed

Before a stage frees a new group, the curvature of the log-density over that group says
whether the data constrain it.
{func}`~photomancy.detection.curvature_block` returns the negative Hessian over the
leaves a selector names, with everything else held fixed. It is the same selector form as
{func}`~photomancy.infer.group_labels`.

```python continuation
from photomancy.detection import curvature_block


class Scene(eqx.Module):
    center: jnp.ndarray
    amplitude: jnp.ndarray
    held: jnp.ndarray


def logdensity(scene):
    model = base + scene.amplitude * jnp.exp(
        -0.5 * ((jnp.asarray(x) - scene.center) / 0.05) ** 2
    )
    return huber_loglike(data, 0.05)(model)


fitted = Scene(jnp.asarray(0.3), jnp.asarray(0.7), jnp.asarray(0.0))
block = curvature_block(logdensity, fitted, lambda s: (s.center, s.amplitude))
print(np.asarray(block))
assert block.shape == (2, 2)
assert np.all(np.linalg.eigvalsh(np.asarray(block)) > 0.0)
```

The eigenvalues of the block are the information in each direction of the group: a value
near zero is a direction the data do not constrain, and at a maximum the inverse of the
block is the formal covariance of the group (the position error of a source, for
example).

## Report a width, and say whether the point is a maximum

{func}`~photomancy.detection.measure` maximizes a fit's likelihood from a start and
reports the formal covariance there. The covariance is the inverse of the observed
information, `Fit.information(x, kind="observed", include_prior=False)`: the exact
negative Hessian of the log likelihood, built one column at a time. It is inverted in
correlation form (scaled by the square roots of its diagonal first): the free
parameters of an imaging fit span many orders of magnitude (condition number 4e12 on
a full-scene fit, 3.7 after the scaling), and the scaled matrix decides without
rounding whether it is positive definite.

The result also says whether the point is a maximum. `regular` is true when the
information is positive definite, and `min_eigenvalue` is its smallest eigenvalue. Where
`regular` is false the point is a saddle, a ridge or an undetected source, and the
widths of the parameters that carry the negative curvature are NaN. That is the correct
statement: a Gaussian width is not defined there, and a number would hide it.

Two more fields say whether the run converged. `grad_norm` is the norm of the gradient of
the log likelihood at the estimate, in the stored coordinates (judge it against the
widths, not against one), and `newton_decrement` is `0.5 g' cov g`, the nats the
estimate still lies below the peak of the Gaussian that `cov` describes. `regular` is
true only when the information is positive definite and the decrement is at most
`max_decrement` (1 nat by default). The run uses the optimizer `MAP` uses, in
coordinates scaled by the Fisher information at the start, so it reaches the same
objective from a start several formal errors away.

A source that may be faint needs a law with signed support. A bounded law, such as a
positive amplitude, puts the maximum of a source consistent with zero on the edge of the
support, where the position drops out of the likelihood; once the data pull past that edge by
more than `max_decrement` nats the nonzero stored-coordinate gradient makes `regular`
false. A wide
`Normal` on the amplitude lets the same source return a finite summary, and its width is
irrelevant because `measure` reads no prior density. A detection threshold for the
amplitude is a separate question: for a high-count, white Poisson null and an 81-trial
grid on two rolls, the 99 percent per-map maximum of `z` is 3.8 sigma (3.77, interval
3.59 to 3.84). That value holds only for that grid and null; a denser grid needs the
threshold `isf(0.01 / N_eff)` for its own effective number of independent trials.

`measure` is for tens of free scalars. The observed information over every free scalar
costs about 20 s and 7 GB for ten parameters on a full-scene fit, and `measure` raises
`MemoryLimit` before it optimizes when the closed-form estimate is above `max_bytes`.
Free pixels do not belong in it.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from priorities import Uniform

from photomancy.detection import measure
from photomancy.infer import Fit, ForwardTerm
from photomancy.infer.likelihoods import gaussian_loglike

jax.config.update("jax_enable_x64", True)


class Bump(eqx.Module):
    amplitude: jnp.ndarray


class Render(eqx.Module):
    grid: jnp.ndarray

    def __call__(self, model):
        return model.amplitude * jnp.exp(-0.5 * (self.grid / 0.2) ** 2)


grid = jnp.linspace(-1.0, 1.0, 64)
data = 0.7 * jnp.exp(-0.5 * (grid / 0.2) ** 2) + 0.05 * jax.random.normal(
    jax.random.key(0), grid.shape
)
term = ForwardTerm(Render(grid), gaussian_loglike(data, 0.05))
fit = Fit(Bump(jnp.asarray(0.5)), {"amplitude": Uniform(0.0, 5.0)}, (term,))

estimate = measure(fit, {"amplitude": jnp.asarray(0.5)})
print(
    float(estimate.value["amplitude"]), float(estimate.sigma[0]), bool(estimate.regular)
)
assert bool(estimate.regular)
```

## Two curvature matrices, behind one call

`Fit.information` returns either matrix at a point, in the stored coordinates of the
fit's entries, as a `(P, P)` float64 array (it needs `jax_enable_x64`).

- `kind="observed"` is the exact Hessian of the negative log density. It is the width at
  a converged maximum, the matrix of the Laplace evidence, and the test of whether a
  point is a regular maximum. It is not positive definite away from one.
- `kind="fisher"` is `J^T J` of the terms' `fisher_root`, the expected curvature of the
  likelihood at the point. It does not depend on the data, is always positive
  semidefinite, and costs about half the time of the observed matrix, so it scales an
  optimizer or forecasts a width before data exist. It cannot flag a bad point, so it is
  not the reported width: for an undetected source it gives a finite, smaller width at a
  larger fitted amplitude, the opposite of the honest statement. Every term of the fit
  needs a root (`ImageTerm`, and `ForwardTerm` over a Gaussian or robust builder
  have one); if one does not, the call raises and names it.

```python continuation
x = {"amplitude": estimate.value["amplitude"]}
fisher = fit.information(x, kind="fisher", include_prior=False)
observed = fit.information(x, kind="observed", include_prior=False)
# a Gaussian likelihood linear in the amplitude: the two matrices coincide
np.testing.assert_allclose(observed, fisher, rtol=1e-9)
```

`include_prior=True` (the default) adds the Hessian of the negative log prior, in which
case the sum is not positive semidefinite in general. `batch_size` is the number of
Hessian columns (or Jacobian tangents) built at once: `None` builds all at once, which
is the fast choice on a GPU with the memory for it, and `1` builds one at a time, which
holds one gradient's intermediates and is the choice on a CPU with a large image.
`max_bytes` refuses, before anything is built, a matrix whose closed-form size (the
`P` by `P` result and the tangents) is above the limit, and names the batch size that
fits.
