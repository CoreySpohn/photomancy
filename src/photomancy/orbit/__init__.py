"""Orbit fitting on a ``Fit`` with a Thiele-Innes orbit block.

The importance engines are configured backends of ``photomancy.infer``: ``few_epoch``
weights scale-and-rotate draws (one or two astrometric epochs) and ``collapsed`` runs
adaptive importance on a proposal that completes the four sky constants of the orbit
from the linear astrometry terms. ``SheetSum`` sums terms that tell the two
line-of-sight sheets of an orbit row apart. ``posterior_physical`` reads the orbit of a
``Posterior`` in physical units and ``evaluate_orbit_candidates`` scores the next
astrometric epoch. Data and terms live in ``photomancy.measurements``; the geometry is
orbix (``orbix.system``).
"""

from photomancy.orbit.backends import collapsed, few_epoch
from photomancy.orbit.completion import CompletedProposal, Completion
from photomancy.orbit.diagnostics import posterior_physical
from photomancy.orbit.eig import evaluate_orbit_candidates
from photomancy.orbit.proposals import ScaleRotateProposal, ThetaPrior
from photomancy.orbit.sheets import SheetSum

__all__ = [
    "CompletedProposal",
    "Completion",
    "ScaleRotateProposal",
    "SheetSum",
    "ThetaPrior",
    "collapsed",
    "evaluate_orbit_candidates",
    "few_epoch",
    "posterior_physical",
]
