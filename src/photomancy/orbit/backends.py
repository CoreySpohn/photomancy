"""Importance engines for a Thiele-Innes orbit fit, as configured backends.

``few_epoch`` weights scale-and-rotate draws (one or two astrometric epochs);
``collapsed`` runs adaptive importance on a proposal that completes the four sky
constants analytically, so the search is over three coordinates.
"""

from photomancy.infer import AdaptiveImportance, Importance
from photomancy.orbit.completion import CompletedProposal, ThetaPrior
from photomancy.orbit.proposals import ScaleRotateProposal

__all__ = ["collapsed", "few_epoch"]


def few_epoch(n_proposals=2**17):
    """Importance sampling from scale-and-rotate draws.

    Exact in the block's prior, and efficient while one or two astrometric rows
    constrain the orbit. The returned cloud holds ``n_proposals`` weighted draws and
    the evidence is the importance mean over all of them; ``Posterior.resample``
    gives equal-weight draws.

    Args:
        n_proposals: Draws of the proposal, all returned.
    """
    return Importance(ScaleRotateProposal(), n_proposals)


def collapsed(n_round=2**16, n_rounds=3, **adaptive):
    """Adaptive importance on a proposal that completes the sky constants.

    Round zero draws the nonlinear coordinates from the block's own law
    (:class:`ThetaPrior`) and every round refits a kernel mixture on them, with a
    defensive share of that law. Every round, the last included, has ``n_round`` draws,
    and the returned cloud and the evidence come from the last round alone.

    Args:
        n_round: Draws of every round.
        n_rounds: Number of rounds (the last is the returned cloud).
        **adaptive: Further settings of
            :class:`~photomancy.infer.AdaptiveImportance` (``n_centers``,
            ``defensive``).
    """
    return AdaptiveImportance(
        CompletedProposal(ThetaPrior()),
        n_draws=n_round,
        n_round=n_round,
        n_rounds=n_rounds,
        **adaptive,
    )
