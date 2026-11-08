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
{mod}`photomancy.likelihoods` that is curved in the prediction.

```python
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from photomancy.detection import likelihood_ratio_map, profile_amplitude
from photomancy.likelihoods import gaussian_loglike, huber_loglike

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
{func}`jax.lax.map`, `batch_size` at a time, so `batch_size` sets the memory use. A
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
{func}`~photomancy.backends.staged.group_labels`.

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
