"""photomancy.atmosphere: atmospheric-retrieval inference over skyscapes forwards.

The atmosphere parallel of ``photomancy.orbit`` -- photomancy owns the retrieval glue,
skyscapes owns the physics. Retrieval is organized as *targets* (model paths + a
default prior) crossed with *forward strategies* (which skyscapes model: the fixed-T
precomputed ``for_retrieval`` vs the live-recompute ``from_default_setup``). The first
target is fixed-T abundance retrieval; temperature / TP-structure targets and an
evidence ("is O2 there?") entry point slot in alongside.
"""

from photomancy.atmosphere.forward import contrast_forward
from photomancy.atmosphere.priors import abundance_paths, default_abundance_prior

__all__ = [
    "abundance_paths",
    "contrast_forward",
    "default_abundance_prior",
]
