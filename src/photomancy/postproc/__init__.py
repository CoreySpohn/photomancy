# ruff: noqa: RUF022
"""Post-processing for coronagraphic imaging: detection statistics and subtraction.

Everything here is domain code over images, kept outside the domain-free
``photomancy.infer`` subpackage. Importing ``photomancy`` does not import this
package; import ``photomancy.postproc`` when you need it.

DetectionEstimator.mawet and DetectionEstimator.matched_filter build
JIT-ready estimators for aperture photometry and Gaussian matched filtering.
Call an estimator at candidate positions or use its map method for a 2D
statistic image.

Seam contracts: FrameSet -> AbstractPostProcessing.detect -> DetectionStats.

The PSF-template matched filter (PSFTemplateFilter) is exported at the top
level. YippyTemplateProvider lives in photomancy.postproc.templates.yippy and
is not imported here. The experimental calibration tools (candidate finding,
null counting, injection throughput) live in photomancy.postproc.calibration.
"""

# Analysis workflows
from photomancy.postproc.analysis import (
    get_perfect_residuals,
)

# Post-processing seam (interface + arms)
from photomancy.postproc.arms import (
    AbstractPostProcessing,
    MatchedFilterPostProcessing,
    MawetPostProcessing,
)

# Seam contract types and the post-processing interface
from photomancy.postproc.contracts import DetectionStats, FrameSet

# Core primitives
from photomancy.postproc.core import (
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
    aperture_photometry,
    aperture_solid_angle,
    calculate_n_apertures,
    circular_aperture_mask,
    extract_patch,
    flux_map,
    gaussian_filter_2d,
    gaussian_kernel_1d,
    generate_aperture_coords,
    get_pca_basis,
    grubbs_fpf,
    inject_planet,
    make_aperture_kernel,
    make_simple_disk,
    masked_mean,
    masked_std,
    n_reference_apertures,
    normal_sf,
    pca_subtract,
    small_sample_penalty,
    soft_aperture_mask,
    student_t_sf,
)

# Modeling primitives (including subtraction)
from photomancy.postproc.core.modeling import (
    subtract_disk,
    subtract_star,
)

# Yield pipelines (high-level workflows)
from photomancy.postproc.pipelines import (
    calculate_yield_snr,
    klip_subtract,
)

# Template providers (PSF-template matched filtering)
from photomancy.postproc.templates import (
    AbstractTemplateProvider,
    ArrayTemplateProvider,
)

__all__ = [
    # Composable detection (filter / sampler / significance)
    "DetectionEstimator",
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "ApertureFilter",
    "GaussianFilter",
    "MapFilter",
    "PSFTemplateFilter",
    "ApertureSampler",
    "AnnulusSampler",
    "TwoSampleTTest",
    "AnnulusSigmaTest",
    "GrubbsTest",
    "gaussian_kernel_1d",
    "gaussian_filter_2d",
    "extract_patch",
    "normal_sf",
    "grubbs_fpf",
    "n_reference_apertures",
    # Subtraction Primitives
    "subtract_star",
    "subtract_disk",
    "klip_subtract",
    "calculate_yield_snr",
    # PCA/KLIP
    "get_pca_basis",
    "pca_subtract",
    # Forward Modeling
    "inject_planet",
    "make_simple_disk",
    # Yield Analysis
    "get_perfect_residuals",
    # Seam contracts (post-processing interface)
    "FrameSet",
    "DetectionStats",
    "AbstractPostProcessing",
    "MawetPostProcessing",
    "MatchedFilterPostProcessing",
    "student_t_sf",
    # Template providers (PSF-template matched filtering)
    "AbstractTemplateProvider",
    "ArrayTemplateProvider",
    # Core utilities
    "calculate_n_apertures",
    "flux_map",
    "generate_aperture_coords",
    "make_aperture_kernel",
    "masked_mean",
    "masked_std",
    "small_sample_penalty",
    # Photometry
    "aperture_photometry",
    "aperture_solid_angle",
    "circular_aperture_mask",
    "soft_aperture_mask",
]
