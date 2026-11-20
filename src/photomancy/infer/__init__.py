"""photomancy.infer: inference machinery with no astronomy in it.

A ``Fit`` (``fit``) is a model PyTree with leaf laws and a sum of data terms
(``terms``) scored with residual kernels (``likelihoods``). ``run(fit, backend, key)``
(``backends``) turns it into a ``Posterior`` (``posterior``), from which come the
evidence (``evidence``), weighted-particle statistics (``diagnostics``) and expected
information gain (``eig``).

Nothing here imports an astronomy library or any other part of photomancy; a
test of the import graph holds the boundary.
"""

from photomancy.infer.backends import (
    MAP,
    MCLMC,
    NUTS,
    AbstractBackend,
    AdaptiveImportance,
    EarlyStopping,
    Grid,
    Importance,
    ImportanceInfo,
    Laplace,
    LaplaceInfo,
    MAPInfo,
    MAPStage,
    MCLMCInfo,
    NestedInfo,
    NestedSampling,
    NUTSInfo,
    Pathfinder,
    PathfinderInfo,
    gauss_legendre_rule,
    group_labels,
    midpoint_rule,
    rescaled_transform,
)
from photomancy.infer.diagnostics import (
    khat_threshold,
    kish_ess,
    kish_temperature,
    mc_se,
    normalize_log_weights,
    pareto_khat,
    ratio_mc_se,
    weighted_cov,
    weighted_mean,
    weighted_quantile,
)
from photomancy.infer.eig import (
    EIGEstimate,
    LinearizedEIG,
    eig_contrastive,
    eig_linearized,
    eig_nested,
)
from photomancy.infer.evidence import importance_evidence, sigma_from_log_bayes_factor
from photomancy.infer.fit import (
    AbstractLinearGaussianTerm,
    AbstractProposingTerm,
    AbstractTerm,
    Fit,
    LeafProposal,
    LinearGaussian,
)
from photomancy.infer.posterior import (
    Forecast,
    Posterior,
    ReweightInfo,
    UpdateInfo,
    cluster_to_mixture,
    ravel_samples,
)
from photomancy.infer.proposals import (
    AbstractDensityProposal,
    AbstractProposal,
    DefensiveProposal,
    MixtureProposal,
    PriorProposal,
)
from photomancy.infer.run import run
from photomancy.infer.terms import ForwardTerm
from photomancy.infer.update import update

__all__ = [
    "MAP",
    "MCLMC",
    "NUTS",
    "AbstractBackend",
    "AbstractDensityProposal",
    "AbstractLinearGaussianTerm",
    "AbstractProposal",
    "AbstractProposingTerm",
    "AbstractTerm",
    "AdaptiveImportance",
    "DefensiveProposal",
    "EIGEstimate",
    "EarlyStopping",
    "Fit",
    "Forecast",
    "ForwardTerm",
    "Grid",
    "Importance",
    "ImportanceInfo",
    "Laplace",
    "LaplaceInfo",
    "LeafProposal",
    "LinearGaussian",
    "LinearizedEIG",
    "MAPInfo",
    "MAPStage",
    "MCLMCInfo",
    "MixtureProposal",
    "NUTSInfo",
    "NestedInfo",
    "NestedSampling",
    "Pathfinder",
    "PathfinderInfo",
    "Posterior",
    "PriorProposal",
    "ReweightInfo",
    "UpdateInfo",
    "cluster_to_mixture",
    "eig_contrastive",
    "eig_linearized",
    "eig_nested",
    "gauss_legendre_rule",
    "group_labels",
    "importance_evidence",
    "khat_threshold",
    "kish_ess",
    "kish_temperature",
    "mc_se",
    "midpoint_rule",
    "normalize_log_weights",
    "pareto_khat",
    "ratio_mc_se",
    "ravel_samples",
    "rescaled_transform",
    "run",
    "sigma_from_log_bayes_factor",
    "update",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
]
