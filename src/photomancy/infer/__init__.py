"""photomancy.infer: inference machinery with no astronomy in it.

A fit is a log density over a partitioned model PyTree (``model``), optionally a
sum of data terms (``terms``) scored with residual kernels (``likelihoods``) and
priors over the fitted leaves (``priors``). A backend (``backends``) turns the
density into a ``Posterior`` (``posterior``), from which come the evidence
(``evidence``), weighted-particle statistics (``particles``) and expected
information gain (``eig``).

Nothing here imports an astronomy library or any other part of photomancy; a
test of the import graph holds the boundary.
"""

from photomancy.infer.backends import (
    MAP,
    MCLMC,
    NUTS,
    SMC,
    AbstractBackend,
    Grid,
    JaxnsBackend,
    Laplace,
    LaplaceBackend,
    LaplaceMixtureBackend,
    MAPBackend,
    MCLMCBackend,
    NestedSampling,
    NUTSBackend,
    Pathfinder,
    PathfinderBackend,
    PathfinderMixtureBackend,
    SMCBackend,
    gauss_legendre,
    midpoint,
)
from photomancy.infer.diagnostics import (
    khat_threshold,
    kish_ess,
    mc_se,
    normalize_log_weights,
    pareto_khat,
    ratio_mc_se,
    weighted_cov,
    weighted_mean,
    weighted_quantile,
)
from photomancy.infer.eig import EIGEstimate, eig_contrastive, eig_nested
from photomancy.infer.evidence import importance_evidence, sigma_from_log_bayes_factor
from photomancy.infer.fit import AbstractLinearTerm, AbstractTerm, Fit, LinearForm
from photomancy.infer.legacy_posterior import (
    AbstractPosterior,
    GaussianPosterior,
    MixturePosterior,
    PointPosterior,
    SamplePosterior,
)
from photomancy.infer.model import (
    AbstractLogDensity,
    build_gaussian_fit,
    build_scene_logdensity,
)
from photomancy.infer.posterior import (
    Forecast,
    Posterior,
    cluster_to_mixture,
    ravel_samples,
)
from photomancy.infer.posterior_utils import project_samples, sample_capped
from photomancy.infer.priors import IndependentPrior
from photomancy.infer.run import run
from photomancy.infer.terms import TermLogDensity
from photomancy.infer.update import update

__all__ = [
    "MAP",
    "MCLMC",
    "NUTS",
    "SMC",
    "AbstractBackend",
    "AbstractLinearTerm",
    "AbstractLogDensity",
    "AbstractPosterior",
    "AbstractTerm",
    "EIGEstimate",
    "Fit",
    "Forecast",
    "GaussianPosterior",
    "Grid",
    "IndependentPrior",
    "JaxnsBackend",
    "Laplace",
    "LaplaceBackend",
    "LaplaceMixtureBackend",
    "LinearForm",
    "MAPBackend",
    "MCLMCBackend",
    "MixturePosterior",
    "NUTSBackend",
    "NestedSampling",
    "Pathfinder",
    "PathfinderBackend",
    "PathfinderMixtureBackend",
    "PointPosterior",
    "Posterior",
    "SMCBackend",
    "SamplePosterior",
    "TermLogDensity",
    "build_gaussian_fit",
    "build_scene_logdensity",
    "cluster_to_mixture",
    "eig_contrastive",
    "eig_nested",
    "gauss_legendre",
    "importance_evidence",
    "khat_threshold",
    "kish_ess",
    "mc_se",
    "midpoint",
    "normalize_log_weights",
    "pareto_khat",
    "project_samples",
    "ratio_mc_se",
    "ravel_samples",
    "run",
    "sample_capped",
    "sigma_from_log_bayes_factor",
    "update",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
]
