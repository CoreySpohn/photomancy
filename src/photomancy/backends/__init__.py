"""photomancy inference backends.

Each backend wraps one inference method (Laplace today; NUTS, adaptive-tempered
SMC, MCLMC, Pathfinder to come) behind a uniform ``run(logdensity, init, key)``
that returns a Posterior.
"""

from photomancy.backends.base import AbstractBackend
from photomancy.backends.laplace import LaplaceBackend

__all__ = ["AbstractBackend", "LaplaceBackend"]
