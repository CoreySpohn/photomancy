"""Orbit fitting on a ``Fit`` with a Thiele-Innes orbit block.

The engines are configured backends of ``photomancy.infer``, named for the epochs the
caller holds: ``few_epoch`` weights scale-and-rotate draws (one or two astrometric
epochs, the orbit block the only free entry) and ``any_epoch`` is the general engine,
adaptive importance on a proposal that completes the four sky constants of the orbit
from the linear astrometry terms. ``SheetSum`` sums terms that tell the two
line-of-sight sheets of an orbit row apart. ``classical_elements`` reads the orbit of a
``Posterior`` in classical elements and ``evaluate_orbit_candidates`` scores the next
astrometric epoch. Data and terms live in ``photomancy.measurements``; the geometry is
orbix (``orbix.system``).
"""

from photomancy.orbit.backends import any_epoch, few_epoch
from photomancy.orbit.completion import CompletedProposal, Completion
from photomancy.orbit.diagnostics import classical_elements
from photomancy.orbit.eig import evaluate_orbit_candidates
from photomancy.orbit.proposals import ScaleRotateProposal, ShapePriorProposal
from photomancy.orbit.sheets import SheetSum

__all__ = [
    "CompletedProposal",
    "Completion",
    "ScaleRotateProposal",
    "ShapePriorProposal",
    "SheetSum",
    "any_epoch",
    "classical_elements",
    "evaluate_orbit_candidates",
    "few_epoch",
]
