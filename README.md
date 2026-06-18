# photomancy

> Divination from light.

A JAX-native Bayesian inference and value-of-information engine for the Habitable
Worlds Observatory direct-imaging simulation suite. `orbix` builds the geometry,
`skyscapes` builds the scene, and **photomancy** divines the scene back from the
data: posteriors, evidence, and the next-best observation, over orbits, disks, and
(later) atmospheres and images.

The engine is forward-model agnostic. A fit is a `logdensity` over a partitioned
scene PyTree, assembled from three plug-ins -- a forward model, a likelihood, and a
prior -- and run through a uniform `Backend` (Laplace mixture, NUTS, adaptive
tempered SMC, MCLMC, Pathfinder) that returns one `Posterior` exposing `.sample`,
`.log_prob`, and `.evidence`.

Status: early development. Orbit fitting is implemented; disk, atmosphere, and
image-domain fitting are planned.
