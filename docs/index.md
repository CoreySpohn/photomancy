# photomancy

> Divination from light.

photomancy is a JAX-native engine for Bayesian inference and value of information over
astrophysical scenes observed by direct imaging. orbix builds the geometry and skyscapes
builds the scene, and photomancy divines the scene back from the data: posteriors over
the scene parameters, the Bayesian evidence for model comparison, and the expected
information gain of a candidate next observation.

The engine is forward-model agnostic. A fit is a `logdensity` over a partitioned scene
PyTree, assembled from three plug-ins (a forward model, a likelihood, and a prior) and
run through a uniform `Backend` that returns one `Posterior` exposing `.sample`,
`.log_prob`, and `.evidence`. The [architecture overview](explanation/architecture)
describes how the pieces fit together and where the library is headed.

## Ecosystem position

```text
  orbix (geometry)  ---\
                        >--->  photomancy  --->  posteriors + evidence + next-best obs
  skyscapes (scene) ---/      logdensity -> backend -> posterior -> EIG
```

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
samples = posterior.sample(key, 1000)  # (1000, d), flat z-space
log_evidence = posterior.evidence
```

The orbit, disk, and atmosphere domains package these plug-ins for their own data, so a
typical fit calls a domain helper rather than wiring the plug-ins by hand.

## Installation

```bash
pip install photomancy
```

photomancy is in early development, and the API may still change.

## Status

Orbit fitting is implemented across radial-velocity, astrometry, and imaging data, and
disk fitting rides the same engine. Atmospheric retrieval is in progress, and
image-domain fitting against a coronagraph forward is the next major target. See the
[architecture overview](explanation/architecture) for the design and the roadmap.

```{toctree}
:maxdepth: 1
:caption: Explanation
:hidden:

explanation/architecture
```

<!-- TODO: add how-to guides and runnable tutorials as the domains stabilize. -->

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/photomancy/index
```
