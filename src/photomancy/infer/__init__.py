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
    Grid,
    Importance,
    ImportanceInfo,
    Laplace,
    LaplaceInfo,
    MAPInfo,
    MCLMCInfo,
    NestedInfo,
    NestedSampling,
    NUTSInfo,
    Pathfinder,
    PathfinderInfo,
    gauss_legendre,
    midpoint,
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
from photomancy.infer.eig import EIGEstimate, eig_contrastive, eig_nested
from photomancy.infer.evidence import importance_evidence, sigma_from_log_bayes_factor
from photomancy.infer.fit import (
    AbstractConditionalTerm,
    AbstractLinearTerm,
    AbstractTerm,
    Fit,
    LeafConditional,
    LinearForm,
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
    "AbstractConditionalTerm",
    "AbstractDensityProposal",
    "AbstractLinearTerm",
    "AbstractProposal",
    "AbstractTerm",
    "AdaptiveImportance",
    "DefensiveProposal",
    "EIGEstimate",
    "Fit",
    "Forecast",
    "ForwardTerm",
    "Grid",
    "Importance",
    "ImportanceInfo",
    "Laplace",
    "LaplaceInfo",
    "LeafConditional",
    "LinearForm",
    "MAPInfo",
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
    "eig_nested",
    "gauss_legendre",
    "importance_evidence",
    "khat_threshold",
    "kish_ess",
    "kish_temperature",
    "mc_se",
    "midpoint",
    "normalize_log_weights",
    "pareto_khat",
    "ratio_mc_se",
    "ravel_samples",
    "run",
    "sigma_from_log_bayes_factor",
    "update",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
]
