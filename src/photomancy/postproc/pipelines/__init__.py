"""Yield simulation pipelines for coronagraphoto integration.

High-level workflows:
    - calculate_yield_snr: End-to-end subtraction + SNR calculation
    - klip_subtract: PCA/KLIP PSF subtraction

Subtraction primitives are in photomancy.postproc.core.modeling:
    - subtract_star, subtract_disk
"""

from photomancy.postproc.pipelines.yield_pipeline import (
    calculate_yield_snr,
    klip_subtract,
)

__all__ = [
    "calculate_yield_snr",
    "klip_subtract",
]
