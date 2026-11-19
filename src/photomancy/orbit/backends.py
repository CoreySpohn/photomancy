"""Importance engines for a Thiele-Innes orbit fit, as configured backends.

``few_epoch`` weights scale-and-rotate draws (one or two astrometric epochs);
``collapsed`` runs adaptive importance on a proposal that completes the four sky
constants analytically, so the search is over three coordinates.
"""

from photomancy.infer import AdaptiveImportance, Importance
from photomancy.orbit.completion import CompletedProposal, ThetaPrior
from photomancy.orbit.proposals import ScaleRotateProposal

__all__ = ["collapsed", "few_epoch"]


def few_epoch(n_draws=4000, n_proposals=2**17):
    """Importance sampling from scale-and-rotate draws.

    Exact in the block's prior, and efficient while one or two astrometric rows
    constrain the orbit. The returned cloud holds ``max(n_draws, n_proposals)`` weighted
    draws; the evidence is the importance mean over all of them.

    Args:
        n_draws: Smallest number of draws returned.
        n_proposals: Draws of the proposal.
    """
    return Importance(ScaleRotateProposal(), max(n_draws, n_proposals))


def collapsed(n_draws=4000, **rounds):
    """Adaptive importance on a proposal that completes the sky constants.

    Round zero draws the nonlinear coordinates from the block's own law
    (:class:`ThetaPrior`) and every round refits a kernel mixture on them, with a
    defensive share of that law. ``rounds`` are the settings of
    :class:`~photomancy.infer.AdaptiveImportance` (``n_round``, ``n_rounds``,
    ``n_centers``, ``defensive``).

    Args:
        n_draws: Draws of the last round, the returned cloud.
        **rounds: Round settings passed to ``AdaptiveImportance``.
    """
    return AdaptiveImportance(
        CompletedProposal(ThetaPrior()), n_draws=n_draws, **rounds
    )
