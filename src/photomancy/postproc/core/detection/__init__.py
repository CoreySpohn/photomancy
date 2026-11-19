"""Composable detection: filters, samplers, significance tests, estimator."""

from photomancy.postproc.core.detection.base import (
    AbstractFilter,
    AbstractSampler,
    AbstractTest,
)
from photomancy.postproc.core.detection.estimator import DetectionEstimator
from photomancy.postproc.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    MapFilter,
    PSFTemplateFilter,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)
from photomancy.postproc.core.detection.patches import extract_patch
from photomancy.postproc.core.detection.samplers import AnnulusSampler, ApertureSampler
from photomancy.postproc.core.detection.significance import (
    AnnulusSigmaTest,
    GrubbsTest,
    TwoSampleTTest,
)

__all__ = [
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "AnnulusSampler",
    "AnnulusSigmaTest",
    "ApertureFilter",
    "ApertureSampler",
    "DetectionEstimator",
    "GaussianFilter",
    "GrubbsTest",
    "MapFilter",
    "PSFTemplateFilter",
    "TwoSampleTTest",
    "extract_patch",
    "gaussian_filter_2d",
    "gaussian_kernel_1d",
]
