"""photomancy inference backends.

Each backend wraps one inference method (Laplace, NUTS, adaptive-tempered SMC,
nested sampling, MCLMC, and Pathfinder today; a normalizing-flow VI backend to
come) behind a uniform ``run(logdensity, init, key)`` that returns a Posterior.
"""

from photomancy.backends.base import AbstractBackend
from photomancy.backends.laplace import LaplaceBackend, LaplaceMixtureBackend
from photomancy.backends.mclmc import MCLMCBackend
from photomancy.backends.nested import JaxnsBackend, build_scene_nested_model
from photomancy.backends.nuts import NUTSBackend
from photomancy.backends.pathfinder import PathfinderBackend, PathfinderMixtureBackend
from photomancy.backends.smc import SMCBackend

__all__ = [
    "AbstractBackend",
    "JaxnsBackend",
    "LaplaceBackend",
    "LaplaceMixtureBackend",
    "MCLMCBackend",
    "NUTSBackend",
    "PathfinderBackend",
    "PathfinderMixtureBackend",
    "SMCBackend",
    "build_scene_nested_model",
]
