"""Orbit fitting: differentiable forward models, likelihoods, and inference.

The orbit-domain plug-ins for the photomancy inference engine: data containers,
differentiable RV / astrometry / photometry forward models, pure-JAX likelihoods,
eccentricity priors, blind initializers (Thiele-Innes grid search, OFTI, adaptive
importance sampling), and the MAP + Laplace / EIG machinery. Built on orbix
geometry (``orbix.equations``, ``orbix.kepler``).
"""

from orbix.equations import period_to_sma
from orbix.kepler.core import diff_solve_trig

from photomancy.orbit.data import (
    ImagingData,
    NullData,
    OrbitData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
)
from photomancy.orbit.diagnostics import mode_summary, sample_physical
from photomancy.orbit.eig import (
    alias_breaking_eig,
    evaluate_orbit_candidates,
    geometric_eig,
)
from photomancy.orbit.forward import (
    predict_photometry,
    predict_pm_anomaly,
    predict_relative_astrometry,
    predict_rv,
    predict_stellar_astrometry,
)
from photomancy.orbit.grid_search import (
    AbstractGridStrategy,
    AbstractShapeParam,
    AdaptiveImportanceSampler,
    EccVectorShape,
    ParamBounds,
    grid_search,
)
from photomancy.orbit.inference import OrbitProblem, build_orbit_logdensity
from photomancy.orbit.init import find_init, find_init_top_k, ti_to_init
from photomancy.orbit.laplace import map_laplace_fit, map_laplace_mixture_fit
from photomancy.orbit.likelihoods import (
    loglike_imaging,
    loglike_null,
    loglike_pm_anomaly,
    loglike_relative_astrom,
    loglike_rv_marginalized,
    loglike_stellar_astrom,
)
from photomancy.orbit.nested import orbit_nested_sampling
from photomancy.orbit.ofti import AbstractConditioner, ScaleAndRotate, ofti
from photomancy.orbit.priors import (
    ECC_PRIOR_NAMES,
    ecc_distribution,
    eccentricity_disk_transform,
    sample_ecc_prior,
)
from photomancy.orbit.thiele_innes import (
    TIFitResult,
    thiele_innes_fit,
    thiele_innes_grid_search,
)

__all__ = [
    "ECC_PRIOR_NAMES",
    # OFTI (Orbits For The Impatient) rejection sampler
    "AbstractConditioner",
    # Grid-search (adaptive importance sampling)
    "AbstractGridStrategy",
    "AbstractShapeParam",
    "AdaptiveImportanceSampler",
    "EccVectorShape",
    "ImagingData",
    "NullData",
    # Orbit -> generic-backend bridge
    "OrbitData",
    "OrbitProblem",
    "PMAnomalyData",
    "ParamBounds",
    # Data containers
    "RVData",
    "RelativeAstromData",
    "ScaleAndRotate",
    "StellarAstromData",
    # Thiele-Innes fitter
    "TIFitResult",
    "alias_breaking_eig",
    "build_orbit_logdensity",
    # Orbital-mechanics primitives (re-exported from orbix)
    "diff_solve_trig",
    "ecc_distribution",
    # Priors
    "eccentricity_disk_transform",
    "evaluate_orbit_candidates",
    "find_init",
    "find_init_top_k",
    # Bayesian experimental design
    "geometric_eig",
    "grid_search",
    "loglike_imaging",
    "loglike_null",
    "loglike_pm_anomaly",
    "loglike_relative_astrom",
    # Likelihoods
    "loglike_rv_marginalized",
    "loglike_stellar_astrom",
    "map_laplace_fit",
    "map_laplace_mixture_fit",
    "mode_summary",
    "ofti",
    # Nested sampling (NumPyro/jaxns) -> evidence / model comparison
    "orbit_nested_sampling",
    "period_to_sma",
    "predict_photometry",
    "predict_pm_anomaly",
    "predict_relative_astrometry",
    # Forward models
    "predict_rv",
    "predict_stellar_astrometry",
    "sample_ecc_prior",
    "sample_physical",
    "thiele_innes_fit",
    "thiele_innes_grid_search",
    # Initialization
    "ti_to_init",
]
