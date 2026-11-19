"""Importance engines for a Thiele-Innes orbit fit, as configured backends.

A factory is named for the epochs the caller holds. ``few_epoch`` weights
scale-and-rotate draws, for one or two astrometric epochs with the orbit block the only
free entry; ``any_epoch`` is the general engine, adaptive importance on a proposal that
completes the four sky constants analytically, so the search is over three coordinates.
"""

from photomancy.infer import AdaptiveImportance, Importance
from photomancy.orbit.completion import CompletedProposal
from photomancy.orbit.proposals import ScaleRotateProposal, ShapePriorProposal

__all__ = ["any_epoch", "few_epoch"]


def few_epoch(n_draws=2**17):
    """Importance sampling from scale-and-rotate draws.

    Exact in the block's prior, and efficient while one or two astrometric rows
    constrain the orbit. It costs about a quarter of :func:`any_epoch` and refuses, by
    name, a fit it cannot draw (a free entry other than the orbit block). The returned
    cloud holds ``n_draws`` weighted draws and the evidence is the importance mean over
    all of them; ``Posterior.resample`` gives equal-weight draws.

    Args:
        n_draws: Draws of the proposal, all returned.
    """
    return Importance(ScaleRotateProposal(), n_draws)


def any_epoch(n_draws=2**16, n_rounds=3, **adaptive):
    """Adaptive importance on a proposal that completes the sky constants.

    The general engine: any number of epochs, radial velocity, a report, a free leaf a
    term proposes. Round zero draws the nonlinear coordinates from the block's own law
    (:class:`ShapePriorProposal`) and every round refits a kernel mixture on them, with
    a defensive share of that law. Every round, the last included, has ``n_draws``
    draws, and the returned cloud and the evidence come from the last round alone.

    Args:
        n_draws: Draws of every round, the returned cloud included.
        n_rounds: Number of rounds (the last is the returned cloud).
        **adaptive: Further settings of
            :class:`~photomancy.infer.AdaptiveImportance` (``n_centers``,
            ``defensive``).
    """
    return AdaptiveImportance(
        CompletedProposal(ShapePriorProposal()),
        n_draws=n_draws,
        n_round=n_draws,
        n_rounds=n_rounds,
        **adaptive,
    )
