# photomancy

> Divination from light.

photomancy is a JAX-native engine for Bayesian inference and value of information over
astrophysical scenes observed by direct imaging. It is part of the Habitable Worlds
Observatory simulation suite: [orbix](https://github.com/CoreySpohn/orbix) builds the
geometry, [skyscapes](https://github.com/CoreySpohn/skyscapes) builds the scene, and
photomancy divines the scene back from the data, producing posteriors over the scene
parameters, the Bayesian evidence for model comparison, and the expected information gain
of a candidate next observation.

The engine is forward-model agnostic. A fit is a `logdensity` over a partitioned scene
PyTree, assembled from three plug-ins, a forward model, a likelihood, and a prior, and run
through a uniform backend that returns one posterior. The same inference machinery serves
orbit fitting today and disk, atmosphere, and image-domain fitting as those forward models
come online, so an improvement to a sampler or to the information-gain calculation reaches
every domain at once.

## Installation

```bash
pip install photomancy
```

photomancy requires Python 3.11 or newer and is in early development, so the API may still
change. For a source checkout with the test and documentation extras, clone the repository
and run `pip install -e ".[test,docs]"`.

## Quick start

A domain supplies three plug-ins over a scene, and the engine turns them into a fit:

```python
from photomancy import LaplaceBackend, build_scene_logdensity

# forward:    scene -> predicted data
# likelihood: predicted -> scalar log-likelihood
# prior:      an AbstractPrior over the fitted leaves
logdensity, z0, unravel = build_scene_logdensity(
    scene, forward_model=forward, likelihood=likelihood, prior=prior
)

posterior = LaplaceBackend().run(logdensity, z0)
samples = posterior.sample(key, 1000)  # (1000, d), flat parameter space
log_evidence = posterior.evidence
```

The orbit, disk, and atmosphere domains package these plug-ins for their own data, so a
typical fit calls a domain helper rather than wiring the three pieces by hand.

## What it does

- Posteriors over scene parameters behind one interface (`sample`, `log_prob`, `evidence`),
  whichever backend produced them.
- Bayesian evidence and Bayes factors for model comparison and detection, answering
  questions such as whether an orbit needs eccentricity or whether a spectral feature is
  present.
- Multimodal posteriors as evidence-weighted mixtures, so period aliases and mirror
  ambiguities are carried honestly rather than collapsed onto one answer.
- Sequential updating, where a posterior becomes the prior for the next observation through
  `to_prior`, so information accumulates across a campaign.
- Expected information gain over candidate observations, turning a fit into a recommendation
  for where to look next.

## Backends

Every backend exposes one method, `run(logdensity, init, key) -> Posterior`, and sees only
the flat logdensity, never the scene or the forward model, so a new sampler becomes useful
to every domain at once.

| Backend | Method | Returns |
|---|---|---|
| `LaplaceBackend` | MAP optimization plus the eigenvalue-clamped inverse Hessian | a Gaussian posterior and the Laplace evidence |
| `LaplaceMixtureBackend` | multi-start Laplace, weighted by evidence | a mixture of Gaussians over the modes |
| `PathfinderBackend` | quasi-Newton variational inference | a Gaussian posterior and the ELBO |
| `PathfinderMixtureBackend` | multi-start Pathfinder, weighted by ELBO | a mixture of Gaussians over the modes |
| `NUTSBackend` | the No-U-Turn sampler with window adaptation | equally weighted samples |
| `MCLMCBackend` | microcanonical Langevin Monte Carlo | equally weighted samples |
| `SMCBackend` | adaptive-tempered sequential Monte Carlo | samples and the evidence |
| `JaxnsBackend` | nested sampling | samples and the evidence for model comparison |

## Documentation

The full documentation lives in
[`docs/`](https://github.com/CoreySpohn/photomancy/tree/main/docs) and builds with Sphinx
once the `docs` extra is installed:

```bash
sphinx-build -b html docs docs/_build/html
```

It covers the
[architecture and design principles](https://github.com/CoreySpohn/photomancy/blob/main/docs/explanation/architecture.md),
a [mathematical treatment](https://github.com/CoreySpohn/photomancy/blob/main/docs/explanation/mathematical-foundations.md)
of the fit, the evidence, and the information gain, and worked examples for a
[visual walkthrough](https://github.com/CoreySpohn/photomancy/blob/main/docs/examples/walkthrough.ipynb)
on an abstract problem and for
[orbit fitting](https://github.com/CoreySpohn/photomancy/blob/main/docs/examples/orbit_fitting.ipynb).
If the JAX and Bayesian vocabulary is unfamiliar, the
[glossary](https://github.com/CoreySpohn/photomancy/blob/main/docs/explanation/glossary.md)
defines the terms the rest of the documentation relies on.

## Status

Orbit fitting is implemented across radial-velocity, astrometry, and imaging data, and disk
fitting rides the same engine. Atmospheric retrieval is in progress, and image-domain
fitting against a coronagraph forward is the next major target. On the backend side, a
stochastic variational backend with a normalizing-flow guide is planned.

## License

photomancy is released under the MIT license, see
[LICENSE](https://github.com/CoreySpohn/photomancy/blob/main/LICENSE).
