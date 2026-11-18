"""Orbit fitting: data containers, likelihoods, chart problems and initializers.

The orbit-domain plug-ins for the photomancy inference engine: data containers,
pure-JAX likelihoods, one-planet chart problems every backend consumes, blind
initializers (Thiele-Innes grid search, OFTI, adaptive importance sampling), the
MAP + Laplace / EIG machinery, and a fixed-capacity sequential updater
(``OrbitUpdater``) that assimilates one record at a time. Predictions come from
the explicit system terms and orbix observables. Built on orbix geometry
(``orbix.equations``, ``orbix.kepler``, ``orbix.system``).
"""

from photomancy.orbit.data import (
    ImagingData,
    NullData,
    PhotometryData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
)
from photomancy.orbit.diagnostics import (
    mode_observables,
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
)
from photomancy.orbit.grid_search import (
    AbstractGridStrategy,
    AdaptiveImportanceSampler,
    ParamBounds,
    grid_search,
)
from photomancy.orbit.inference import (
    ChartOrbitProblem,
    build_chart_logdensity,
)
from photomancy.orbit.init import find_init
from photomancy.orbit.laplace import (
    map_laplace_fit,
    map_laplace_mixture_fit,
)
from photomancy.orbit.likelihoods import (
    loglike_imaging,
    loglike_null,
    loglike_pm_anomaly,
    loglike_relative_astrom,
    loglike_rv_marginalized,
    loglike_stellar_astrom,
)
from photomancy.orbit.migration import migrate_sky_convention
from photomancy.orbit.nested import (
    OrbitPriorTransform,
    orbit_nested_sampling,
)
from photomancy.orbit.ofti import AbstractConditioner, ScaleAndRotate, ofti
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
    # OFTI (Orbits For The Impatient) rejection sampler
    "AbstractConditioner",
    # Grid-search (adaptive importance sampling)
    "AbstractGridStrategy",
    # Fixed-capacity sequential updater
    "AbstractUpdateTarget",
    "AdaptiveImportanceSampler",
    "ChartOrbitProblem",
    "ConjugateGaussianTarget",
    "FixedMassTarget",
    "ImagingData",
    "NullData",
    "OrbitPriorTransform",
    "OrbitUpdater",
    "PMAnomalyData",
    "ParamBounds",
    "PhotometryData",
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
    "build_chart_logdensity",
    "evaluate_orbit_candidates",
    "find_init",
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
    "mode_observables",
    "mode_scalars",
    "mode_summary",
    "ofti",
    # Nested sampling (jaxns) -> evidence / model comparison
    "orbit_nested_sampling",
    "orbits_from_samples",
    "sample_physical",
    "thiele_innes_fit",
    "thiele_innes_grid_search",
]
