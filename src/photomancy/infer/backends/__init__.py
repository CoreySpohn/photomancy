"""photomancy inference backends.

Each backend is an ``AbstractBackend`` (Laplace, NUTS, nested
sampling, MCLMC, Pathfinder, importance sampling and its adaptive form, a staged MAP
fit and a quadrature Grid) that
``photomancy.infer.run(fit, backend, key)`` runs on a ``Fit`` and returns a
``Posterior``.
The Laplace, Grid and staged MAP backends need only optax. The BlackJAX backends (NUTS,
MCLMC, Pathfinder) and the jaxns backend import their sampler when they run:
install ``photomancy[blackjax]``, ``photomancy[jaxns]`` or ``photomancy[samplers]``.
"""

from photomancy.infer.backends.base import AbstractBackend, BackendDraws
from photomancy.infer.backends.grid import Grid, gauss_legendre_rule, midpoint_rule
from photomancy.infer.backends.importance import (
    AdaptiveImportance,
    Importance,
    ImportanceInfo,
)
from photomancy.infer.backends.laplace import (
    Laplace,
    LaplaceInfo,
)
from photomancy.infer.backends.mclmc import MCLMC, MCLMCInfo
from photomancy.infer.backends.nested import (
    NestedInfo,
    NestedSampling,
)
from photomancy.infer.backends.nuts import NUTS, NUTSInfo
from photomancy.infer.backends.pathfinder import (
    Pathfinder,
    PathfinderInfo,
)
from photomancy.infer.backends.staged import (
    MAP,
    EarlyStopping,
    MAPInfo,
    MAPStage,
    group_labels,
    rescaled_transform,
)

__all__ = [
    "MAP",
    "MCLMC",
    "NUTS",
    "AbstractBackend",
    "AdaptiveImportance",
    "BackendDraws",
    "EarlyStopping",
    "Grid",
    "Importance",
    "ImportanceInfo",
    "Laplace",
    "LaplaceInfo",
    "MAPInfo",
    "MAPStage",
    "MCLMCInfo",
    "NUTSInfo",
    "NestedInfo",
    "NestedSampling",
    "Pathfinder",
    "PathfinderInfo",
    "gauss_legendre_rule",
    "group_labels",
    "midpoint_rule",
    "rescaled_transform",
]
