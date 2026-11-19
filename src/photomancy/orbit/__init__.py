"""Orbit fitting: chart problems and initializers.

The orbit-domain plug-ins for the photomancy inference engine: one-planet chart
problems every backend consumes, blind initializers (Thiele-Innes grid search, OFTI,
adaptive importance sampling) and the MAP + Laplace / EIG machinery. Data, likelihoods
and terms live in ``photomancy.measurements``. Built on orbix geometry
(``orbix.equations``, ``orbix.kepler``, ``orbix.system``).
"""

from photomancy.orbit.completion import CompletedProposal, Completion, ThetaPrior
from photomancy.orbit.diagnostics import posterior_physical
from photomancy.orbit.eig import (
    alias_breaking_eig,
    alias_breaking_eig_from_covariances,
    evaluate_orbit_candidates,
    geometric_eig,
)
from photomancy.orbit.proposals import ScaleRotateProposal

__all__ = [
    "CompletedProposal",
    "Completion",
    "ScaleRotateProposal",
    "ThetaPrior",
    # Bayesian experimental design
    "alias_breaking_eig",
    "alias_breaking_eig_from_covariances",
    "evaluate_orbit_candidates",
    "geometric_eig",
    "posterior_physical",
]
