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
    AbstractBackend,
    JaxnsBackend,
    LaplaceBackend,
    LaplaceMixtureBackend,
    MAPBackend,
    MCLMCBackend,
    NUTSBackend,
    PathfinderBackend,
    PathfinderMixtureBackend,
    SMCBackend,
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
from photomancy.infer.evidence import sigma_from_log_bayes_factor
from photomancy.infer.model import (
    AbstractLogDensity,
    build_gaussian_fit,
    build_scene_logdensity,
)
from photomancy.infer.posterior import (
    AbstractPosterior,
    GaussianPosterior,
    MixturePosterior,
    PointPosterior,
    SamplePosterior,
)
from photomancy.infer.posterior_utils import project_samples, sample_capped
from photomancy.infer.priors import IndependentPrior
from photomancy.infer.terms import AbstractTerm, TermLogDensity

__all__ = [
    "AbstractBackend",
    "AbstractLogDensity",
    "AbstractPosterior",
    "AbstractTerm",
    "GaussianPosterior",
    "IndependentPrior",
    "JaxnsBackend",
    "LaplaceBackend",
    "LaplaceMixtureBackend",
    "MAPBackend",
    "MCLMCBackend",
    "MixturePosterior",
    "NUTSBackend",
    "PathfinderBackend",
    "PathfinderMixtureBackend",
    "PointPosterior",
    "SMCBackend",
    "SamplePosterior",
    "TermLogDensity",
    "build_gaussian_fit",
    "build_scene_logdensity",
    "khat_threshold",
    "kish_ess",
    "mc_se",
    "normalize_log_weights",
    "pareto_khat",
    "project_samples",
    "ratio_mc_se",
    "sample_capped",
    "sigma_from_log_bayes_factor",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
]
