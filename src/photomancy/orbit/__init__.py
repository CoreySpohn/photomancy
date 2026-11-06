"""Orbit fitting: differentiable forward models, likelihoods, and inference.

The orbit-domain plug-ins for the photomancy inference engine: data containers,
differentiable RV / astrometry / photometry forward models, pure-JAX likelihoods,
eccentricity priors, blind initializers (Thiele-Innes grid search, OFTI, adaptive
importance sampling), the MAP + Laplace / EIG machinery, and a fixed-capacity
sequential updater (``OrbitUpdater``) that assimilates one record at a time.
Built on orbix geometry (``orbix.equations``, ``orbix.kepler``).
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
from photomancy.orbit.diagnostics import (
    mode_scalars,
    mode_summary,
    orbits_from_samples,
    sample_physical,
)
from photomancy.orbit.eig import (
    alias_breaking_eig,
    alias_breaking_eig_from_covariances,
    evaluate_orbit_candidates,
    geometric_eig,
)
from photomancy.orbit.few_epoch import (
    TIBasisModel,
    admissible_region_fit,
    lambert_depth_fit,
    state_to_theta,
    ti_semimajor_axis,
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
from photomancy.orbit.migration import migrate_sky_convention
from photomancy.orbit.nested import orbit_nested_sampling
from photomancy.orbit.ofti import AbstractConditioner, ScaleAndRotate, ofti
from photomancy.orbit.priors import (
    ECC_PRIOR_NAMES,
    ecc_distribution,
    eccentricity_disk_transform,
    sample_ecc_prior,
)
from photomancy.orbit.projected import astrometric_loss, make_astrometric_evaluator
from photomancy.orbit.thiele_innes import (
    TIFitResult,
    thiele_innes_fit,
    thiele_innes_grid_search,
)
from photomancy.orbit.updater import (
    AbstractUpdateTarget,
    ConjugateGaussianTarget,
    FixedMassTarget,
    OrbitUpdater,
    ThieleInnesTarget,
    UpdaterBelief,
    UpdaterDiagnostics,
    UpdaterRecord,
    UpdaterStatus,
)

__all__ = [
    "ECC_PRIOR_NAMES",
    # OFTI (Orbits For The Impatient) rejection sampler
    "AbstractConditioner",
    # Grid-search (adaptive importance sampling)
    "AbstractGridStrategy",
    "AbstractShapeParam",
    # Fixed-capacity sequential updater
    "AbstractUpdateTarget",
    "AdaptiveImportanceSampler",
    "ConjugateGaussianTarget",
    "EccVectorShape",
    "FixedMassTarget",
    "ImagingData",
    "NullData",
    # Orbit -> generic-backend bridge
    "OrbitData",
    "OrbitProblem",
    "OrbitUpdater",
    "PMAnomalyData",
    "ParamBounds",
    # Data containers
    "RVData",
    "RelativeAstromData",
    "ScaleAndRotate",
    "StellarAstromData",
    "TIBasisModel",
    # Thiele-Innes fitter
    "TIFitResult",
    "ThieleInnesTarget",
    "UpdaterBelief",
    "UpdaterDiagnostics",
    "UpdaterRecord",
    "UpdaterStatus",
    # Bayesian experimental design
    # Exact few-epoch fitters
    "admissible_region_fit",
    "alias_breaking_eig",
    "alias_breaking_eig_from_covariances",
    "astrometric_loss",
    "build_orbit_logdensity",
    # Orbital-mechanics primitives (re-exported from orbix)
    "diff_solve_trig",
    "ecc_distribution",
    # Priors
    "eccentricity_disk_transform",
    "evaluate_orbit_candidates",
    "find_init",
    "find_init_top_k",
    "geometric_eig",
    "grid_search",
    "lambert_depth_fit",
    "loglike_imaging",
    "loglike_null",
    "loglike_pm_anomaly",
    "loglike_relative_astrom",
    # Likelihoods
    "loglike_rv_marginalized",
    "loglike_stellar_astrom",
    "make_astrometric_evaluator",
    "map_laplace_fit",
    "map_laplace_mixture_fit",
    "migrate_sky_convention",
    "mode_scalars",
    "mode_summary",
    "ofti",
    # Nested sampling (NumPyro/jaxns) -> evidence / model comparison
    "orbit_nested_sampling",
    "orbits_from_samples",
    "period_to_sma",
    "predict_photometry",
    "predict_pm_anomaly",
    "predict_relative_astrometry",
    # Forward models
    "predict_rv",
    "predict_stellar_astrometry",
    "sample_ecc_prior",
    "sample_physical",
    "state_to_theta",
    "thiele_innes_fit",
    "thiele_innes_grid_search",
    # Initialization
    "ti_semimajor_axis",
    "ti_to_init",
]
