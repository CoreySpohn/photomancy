"""photomancy.core: the domain-agnostic inference engine.

Assembles a logdensity over a partitioned scene PyTree from plug-in forward
models, likelihoods, and priors. The partition / ravel boundary helpers and the
Backend protocol + unified Posterior build on top of this.
"""

from photomancy.core.model import (
    build_gaussian_fit,
    build_logdensity,
    build_scene_logdensity,
)

__all__ = ["build_gaussian_fit", "build_logdensity", "build_scene_logdensity"]
