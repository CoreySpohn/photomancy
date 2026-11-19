"""photomancy inference backends.

Each backend is an ``AbstractBackend`` (Laplace, NUTS, nested
sampling, MCLMC, Pathfinder, importance sampling and its adaptive form, a staged MAP
fit and a quadrature Grid) that
``photomancy.infer.run(fit, backend, key)`` runs on a ``Fit`` and returns a
``Posterior``.
The earlier ``*Backend`` classes (``run(logdensity, init, key)``) remain until their
consumers move.

The Laplace, Grid and staged MAP backends need only optax. The BlackJAX backends (NUTS,
MCLMC, Pathfinder) and the jaxns backend import their sampler when they run:
install ``photomancy[blackjax]``, ``photomancy[jaxns]`` or ``photomancy[samplers]``.
"""

from photomancy.infer.backends.base import AbstractBackend, BackendDraws
from photomancy.infer.backends.grid import Grid, gauss_legendre, midpoint
from photomancy.infer.backends.importance import (
    AdaptiveImportance,
    Importance,
    ImportanceInfo,
)
from photomancy.infer.backends.laplace import (
    Laplace,
    LaplaceBackend,
    LaplaceInfo,
    LaplaceMixtureBackend,
)
from photomancy.infer.backends.mclmc import MCLMC, MCLMCBackend, MCLMCInfo
from photomancy.infer.backends.nested import (
    JaxnsBackend,
    NestedInfo,
    NestedSampling,
    build_scene_nested_model,
)
from photomancy.infer.backends.nuts import NUTS, NUTSBackend, NUTSInfo
from photomancy.infer.backends.pathfinder import (
    Pathfinder,
    PathfinderBackend,
    PathfinderInfo,
    PathfinderMixtureBackend,
)
from photomancy.infer.backends.staged import MAP, MAPInfo, map_fit

__all__ = [
    "MAP",
    "MCLMC",
    "NUTS",
    "AbstractBackend",
    "AdaptiveImportance",
    "BackendDraws",
    "Grid",
    "Importance",
    "ImportanceInfo",
    "JaxnsBackend",
    "Laplace",
    "LaplaceBackend",
    "LaplaceInfo",
    "LaplaceMixtureBackend",
    "MAPInfo",
    "MCLMCBackend",
    "MCLMCInfo",
    "NUTSBackend",
    "NUTSInfo",
    "NestedInfo",
    "NestedSampling",
    "Pathfinder",
    "PathfinderBackend",
    "PathfinderInfo",
    "PathfinderMixtureBackend",
    "build_scene_nested_model",
    "gauss_legendre",
    "map_fit",
    "midpoint",
]
