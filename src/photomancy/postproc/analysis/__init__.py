"""Analysis subpackage for photomancy post-processing.

Contains high-level analysis workflows including yield estimation tools.
"""

from photomancy.postproc.analysis.yields import (
    get_perfect_residuals,
)

__all__ = [
    "get_perfect_residuals",
]
