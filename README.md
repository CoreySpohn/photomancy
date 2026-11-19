# photomancy

<p align="center">
  <a href="https://pypi.org/project/photomancy/"><img src="https://img.shields.io/pypi/v/photomancy.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://photomancy.readthedocs.io"><img src="https://readthedocs.org/projects/photomancy/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/photomancy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/photomancy/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/photomancy/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/CoreySpohn/photomancy/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
</p>

> Divination from light.

photomancy is a JAX-native engine for Bayesian inference and value of information over
astrophysical scenes observed by direct imaging. It is part of the Habitable Worlds
Observatory simulation suite: [orbix](https://github.com/CoreySpohn/orbix) builds the
geometry, [skyscapes](https://github.com/CoreySpohn/skyscapes) builds the scene, and
photomancy divines the scene back from the data, producing posteriors over the scene
parameters, the Bayesian evidence for model comparison, and the expected information gain
of a candidate next observation.

The engine is forward-model agnostic. A fit is a normalized log density over the free
entries of a model: the model holds every quantity, a dictionary names the free entries and
their priors, and data terms score the measurements. `run(fit, backend, key)` turns a fit
into a `Posterior`, a cloud of weighted draws with the evidence and its standard error,
whichever backend produced it. The same machinery serves orbit fitting from astrometry,
radial velocity and photometry, and fitting of coronagraph images pixel by pixel
(`photomancy.image.ImageTerm`), so an improvement to a backend or to the information-gain
calculation reaches every domain at once.

## Installation

```bash
pip install photomancy
```

The importance, Laplace, quadrature and staged MAP backends work from the base install.
The sampler backends are optional extras: `photomancy[blackjax]` for NUTS, MCLMC and
Pathfinder, `photomancy[jaxns]` for nested sampling, or `photomancy[samplers]` for both.

```bash
pip install "photomancy[samplers]"
```

photomancy requires Python 3.12 or newer and is in early development, so the API may still
change. For a source checkout with the test and documentation extras, clone the repository
and run `pip install -e ".[test,docs]"`.

## Quick start

Fit the orbit of one planet from four epochs of relative astrometry. The model is an
[orbix](https://github.com/CoreySpohn/orbix) `System`, the orbit is freed by a
`ThieleInnesBlock` with its prior, and the data are a `RelativeAstrometryTerm` drawn at the
true orbit.

```python
import jax
import jax.numpy as jnp
import numpy as np
from orbix.system import Body, OrbitLaw, SkyFrame, System, ThieleInnesBlock
from priorities import LogUniform, Uniform
from skyscapes.population import kipping_eccentricity

from photomancy.infer import Fit, run
from photomancy.measurements import AstrometryData, RelativeAstrometryTerm
from photomancy.orbit import any_epoch, classical_elements

jax.config.update("jax_enable_x64", True)

truth = System.from_classical(
    Body(jnp.asarray(1.0)),  # star mass, solar masses
    Body(jnp.array([0.0])),  # planet masses, solar masses
    SkyFrame(jnp.asarray(10.0)),  # distance, pc
    a_AU=jnp.array([1.0]),
    e=jnp.array([0.1]),
    i_rad=jnp.array([1.0]),
    W_rad=jnp.array([1.0]),
    w_rad=jnp.array([2.0]),
    M0_rad=jnp.array([1.0]),
)
layout = AstrometryData.from_sep_pa(
    time_d=[0.0, 90.0, 180.0, 270.0],
    sep=[0.1] * 4,
    sep_err=[0.004] * 4,
    pa_deg=[0.0] * 4,
    pa_err_deg=[1.0] * 4,
)
term = RelativeAstrometryTerm(layout).draw(jax.random.key(0), truth)

law = OrbitLaw(
    period_d=LogUniform(50.0, 5000.0), eccentricity=kipping_eccentricity(0.9)
)
block = ThieleInnesBlock(law, Uniform(0.9, 1.1), rows=(0,), system="")
fit = Fit(truth, {"orbits": block}, (term,))

posterior = run(fit, any_epoch(), jax.random.key(1))
elements = classical_elements(posterior, jax.random.key(2), 1000, orbit_path="orbits")
print(posterior.log_evidence, np.percentile(elements["period_d"], [16, 50, 84]))
```

When a fifth epoch arrives, `update(posterior, new_term, any_epoch(), key, min_n_eff=...,
max_khat=...)` reweights the existing draws by the new term and refits only when the
reweighting fails its gate.

## What it does

- Posteriors over the free entries of a model behind one type, `Posterior`, whichever
  backend produced them: the draws, their normalized weights, the fit's density at each
  draw, and the fit itself, so any quantity of the model can be read at every draw.
- Bayesian evidence with a standard error for model comparison and detection. A backend
  that cannot vouch for its estimate (an importance estimate whose Pareto k-hat is above
  the limit for its draw count, say) returns NaN rather than a wrong number.
- Multimodal posteriors carried as weighted draws, so period aliases and mirror
  ambiguities stay in the cloud rather than being collapsed onto one answer.
- Sequential updating: `Posterior.append_term` and `Posterior.replace_term` reweight a
  posterior by new data without refitting, under a gate on the effective sample size and
  the k-hat of the new weights, and `update` refits only where that gate refuses.
- Expected information gain of a candidate next observation, turning a fit into a
  recommendation for where to look next.

## Backends

Every backend is passed to `run(fit, backend, key)` and returns a `Posterior`.

| Backend | Method | Evidence |
|---|---|---|
| `Importance` | importance sampling from one proposal | yes |
| `AdaptiveImportance` | rounds of Gaussian mixtures refitted to the weighted draws | yes |
| `Laplace` | a Gaussian at each mode, importance weighted to the fit | yes |
| `Pathfinder` | Pathfinder Gaussians, importance weighted to the fit | yes |
| `NUTS` | the No-U-Turn sampler with window adaptation | no |
| `MCLMC` | microcanonical Langevin Monte Carlo | no |
| `NestedSampling` | jaxns nested sampling | yes |
| `MAP` | staged gradient optimization, one draw at the optimum | no |
| `Grid` | quadrature on a product grid | yes, without a standard error |

The orbit package adds two configured backends named for the data in hand: `few_epoch()`
for one or two astrometric epochs, and `any_epoch()` for everything else.

## Documentation

The full documentation lives in
[`docs/`](https://github.com/CoreySpohn/photomancy/tree/main/docs) and builds with Sphinx
once the `docs` extra is installed:

```bash
sphinx-build -b html docs docs/.build/html
```

It explains
[fits and posteriors](https://github.com/CoreySpohn/photomancy/blob/main/docs/explanation/fit-and-posterior.md)
step by step on a problem with a known answer, describes the
[orbit coordinates and engines](https://github.com/CoreySpohn/photomancy/blob/main/docs/explanation/orbital-parameterizations.md),
and works through
[an orbit fit as the epochs arrive](https://github.com/CoreySpohn/photomancy/blob/main/docs/examples/orbit_fitting.md).

## Status

Orbit fitting is implemented across radial-velocity, astrometry, photometry and
detection-report data, and the pixel likelihood fits coronagraph images through the
coronagraphoto forward model. Disk fitting and atmospheric retrieval are in progress. On
the backend side, a stochastic variational backend with a normalizing-flow guide is
planned.

## License

photomancy is released under the MIT license, see
[LICENSE](https://github.com/CoreySpohn/photomancy/blob/main/LICENSE).
