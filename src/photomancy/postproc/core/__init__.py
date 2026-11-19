# ruff: noqa: RUF022
"""Core JAX-based analysis primitives for photomancy post-processing.

This module contains pure JAX mathematical functions built on jax, equinox,
and hwoutils only. All functions are JIT-compilable and differentiable.
"""

# Composable detection
from photomancy.postproc.core.detection import (
    AbstractFilter,
    AbstractSampler,
    AbstractTest,
    AnnulusSampler,
    AnnulusSigmaTest,
    ApertureFilter,
    ApertureSampler,
    DetectionEstimator,
    GaussianFilter,
    GrubbsTest,
    MapFilter,
    PSFTemplateFilter,
    TwoSampleTTest,
    extract_patch,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)
from photomancy.postproc.core.geometry import (
    calculate_n_apertures,
    generate_aperture_coords,
    n_reference_apertures,
)
from photomancy.postproc.core.modeling import (
    inject_planet,
    make_simple_disk,
    subtract_disk,
    subtract_star,
)
from photomancy.postproc.core.pca import get_pca_basis, pca_subtract
from photomancy.postproc.core.photometry import (
    aperture_photometry,
    aperture_solid_angle,
    circular_aperture_mask,
    flux_map,
    make_aperture_kernel,
    soft_aperture_mask,
)
from photomancy.postproc.core.statistics import (
    grubbs_fpf,
    masked_mean,
    masked_std,
    nanmasked_mean,
    nanmasked_population_std,
    normal_sf,
    small_sample_penalty,
    student_t_sf,
)

__all__ = [
    # Composable detection
    "DetectionEstimator",
    "MapFilter",
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "ApertureFilter",
    "GaussianFilter",
    "PSFTemplateFilter",
    "ApertureSampler",
    "AnnulusSampler",
    "TwoSampleTTest",
    "AnnulusSigmaTest",
    "GrubbsTest",
    "gaussian_kernel_1d",
    "gaussian_filter_2d",
    "extract_patch",
    # Geometry
    "calculate_n_apertures",
    "generate_aperture_coords",
    "n_reference_apertures",
    # PCA
    "get_pca_basis",
    "pca_subtract",
    # Modeling
    "inject_planet",
    "make_simple_disk",
    "subtract_star",
    "subtract_disk",
    # Photometry
    "aperture_photometry",
    "aperture_solid_angle",
    "circular_aperture_mask",
    "flux_map",
    "make_aperture_kernel",
    "soft_aperture_mask",
    # Statistics
    "grubbs_fpf",
    "masked_mean",
    "masked_std",
    "nanmasked_mean",
    "nanmasked_population_std",
    "normal_sf",
    "small_sample_penalty",
    "student_t_sf",
]
