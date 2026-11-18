"""photomancy inference backends.

Each backend wraps one inference method (Laplace, NUTS, adaptive-tempered SMC,
nested sampling, MCLMC, Pathfinder, and a staged PyTree MAP fit today; a
normalizing-flow VI backend to come) behind a uniform
``run(logdensity, init, key)`` that returns a Posterior.
"""

from photomancy.infer.backends.base import AbstractBackend
from photomancy.infer.backends.laplace import LaplaceBackend, LaplaceMixtureBackend
from photomancy.infer.backends.mclmc import MCLMCBackend
from photomancy.infer.backends.nested import (
    JaxnsBackend,
    build_scene_nested_model,
)
from photomancy.infer.backends.nuts import NUTSBackend
from photomancy.infer.backends.pathfinder import (
    PathfinderBackend,
    PathfinderMixtureBackend,
)
from photomancy.infer.backends.smc import SMCBackend
from photomancy.infer.backends.staged import MAPBackend, map_fit

__all__ = [
    "AbstractBackend",
    "JaxnsBackend",
    "LaplaceBackend",
    "LaplaceMixtureBackend",
    "MAPBackend",
    "MCLMCBackend",
    "NUTSBackend",
    "PathfinderBackend",
    "PathfinderMixtureBackend",
    "SMCBackend",
    "build_scene_nested_model",
    "map_fit",
]
