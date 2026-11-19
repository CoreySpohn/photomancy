"""Importance sampling backends: a proposal, or a proposal adapted over rounds.

``Importance`` draws from one :class:`~photomancy.infer.proposals.AbstractProposal`
and weights each draw by ``fit.real_log_prob(z) - log_q``, the target density in the
flat unconstrained vector over the proposal's own (wrapped) density. The evidence is
the importance mean (:func:`photomancy.infer.evidence.importance_evidence`) and is
consistent whenever the proposal has positive density wherever the target does and the
weights have a finite mean. ``AdaptiveImportance`` is population Monte Carlo: it
tempers the target, refits a Gaussian mixture on each round's weighted cloud and
draws the returned cloud from the last proposal.

A draw in a zero-density region of the fit has weight zero, and so has a draw the
proposal marks dead with ``log_q = +inf`` (its coordinates may be NaN and are not
read). Any other NaN target or proposal density is an error: the weights of the slice
become NaN and ``run`` returns a NaN slice, which the operations of a ``Posterior``
refuse. So does an ``AdaptiveImportance`` ladder that did not reach temperature one
(see its docstring).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.infer.backends._flat import draw_template
from photomancy.infer.backends.base import AbstractBackend, BackendDraws
from photomancy.infer.diagnostics import kish_ess, kish_temperature
from photomancy.infer.evidence import importance_evidence
from photomancy.infer.proposals import (
    AbstractProposal,
    MixtureProposal,
    PriorProposal,
)

__all__ = ["AdaptiveImportance", "Importance", "ImportanceInfo"]


class ImportanceInfo(eqx.Module):
    """Diagnostics of an :class:`Importance` or :class:`AdaptiveImportance` run.

    Per round ``r`` (one for ``Importance``): the temperature of the tempered target
    whose weights were taken on that round's draws and their Kish effective sample
    size. A round before the last measures the cloud that the next proposal was fitted
    to (temperature below or at one); the last round is always at temperature one, the
    fit's own density, and its Kish size is that of the returned weights.

    Attributes:
        n_rounds: Number of rounds, a static count.
        temperatures: Temperature of each round's weights ``(n_rounds,)``: the ladder
            of the adaptation rounds, then one for the last round.
        kish_per_round: Kish effective sample size of each round's weights.
        reached: True when the proposal of the last round was fitted at temperature
            one (always for ``Importance``). When false, the evidence, its standard
            error and the log weights of the slice are NaN.
    """

    n_rounds: int = eqx.field(static=True)
    temperatures: jax.Array
    kish_per_round: jax.Array
    reached: jax.Array


def _unravel(fit):
    return ravel_pytree(fit.to_real(draw_template(fit)))[1]


def _stored(fit, z):
    """Stored entries and log volume of the map of flat vectors ``z (n, d)``."""
    return fit.from_real(jax.vmap(_unravel(fit))(z))


def _weigh(fit, proposal, key, n, batch_size):
    """Draw ``n`` points; return ``(x, log_target, log_weights)`` of the fit's density.

    ``log_weights = log_target + volume - log_q``, the density in the flat vector over
    the proposal density. A zero-density point (``-inf`` target) keeps weight zero, and
    a dead draw (``log_q = +inf``) gets weight zero and a ``-inf`` target.
    """
    z, log_q = proposal.sample(key, fit, n)
    x, volume = _stored(fit, z)
    log_target = fit.log_density(x, batch_size=batch_size)
    dead = jnp.isposinf(log_q)
    log_target = jnp.where(dead, -jnp.inf, log_target)
    return x, log_target, jnp.where(dead, -jnp.inf, log_target + volume - log_q)


class Importance(AbstractBackend):
    """Importance sampling from one proposal.

    The draws are weighted ``fit.real_log_prob(z) - log_q`` and the evidence is the
    importance mean, so it is consistent for any proposal that covers the target. How
    well the proposal matches shows in the weights (``khat``, ``n_eff``); the backend
    does not gate on them, the operations of the ``Posterior`` do. ``PriorProposal``
    gives the prior-sampling estimate, whose weights are the likelihood.

    Attributes:
        proposal: The :class:`~photomancy.infer.proposals.AbstractProposal` to draw
            from. Only its draws and their ``log_q`` are used, so a properly weighted
            proposal that is not a density (a completed orbit proposal) is valid.
        n_draws: Number of draws.
    """

    proposal: AbstractProposal
    n_draws: int = eqx.field(static=True, default=2000)

    @eqx.filter_jit
    def _run(self, fit, key, *, batch_size=None):
        """Draw from the proposal and weight the draws to the fit."""
        x, log_target, log_w = _weigh(fit, self.proposal, key, self.n_draws, batch_size)
        log_z, se = importance_evidence(log_w)
        info = ImportanceInfo(1, jnp.ones(1), kish_ess(log_w)[None], jnp.asarray(True))
        return BackendDraws(
            samples=x,
            log_weights=log_w,
            log_evidence=log_z,
            log_evidence_se=se,
            info=info,
            kind="monte_carlo",
            log_target=log_target,
        )


def modes_importance(fit, key, means, covs, log_pi, kept, n_draws, batch_size):
    """Importance sampling from the mixture of the ``kept`` modes.

    Modes that are not kept leave the mixture (zero weight; their means and covariances
    are replaced by harmless placeholders so a NaN cannot reach the density). With no
    mode kept the slice is NaN: weights, evidence and standard error. Returns
    :class:`BackendDraws` whose ``info`` is the importance diagnostics.
    """
    any_kept = jnp.any(kept)
    d = means.shape[-1]
    mixture = MixtureProposal(
        jnp.where(kept[:, None], means, 0.0),
        jnp.where(kept[:, None, None], covs, jnp.eye(d)),
        jnp.where(any_kept, jnp.where(kept, log_pi, -jnp.inf), 0.0),
    )
    draws = Importance(mixture, n_draws)._run(fit, key, batch_size=batch_size)
    nan = jnp.asarray(jnp.nan)
    return BackendDraws(
        samples=draws.samples,
        log_weights=jnp.where(any_kept, draws.log_weights, nan),
        log_evidence=jnp.where(any_kept, draws.log_evidence, nan),
        log_evidence_se=jnp.where(any_kept, draws.log_evidence_se, nan),
        info=draws.info,
        kind=draws.kind,
        log_target=draws.log_target,
    )


def _temper(log_likelihood, beta):
    """``beta * log_likelihood`` with ``0 * -inf`` taken as zero."""
    return jnp.where(beta == 0.0, 0.0, beta * log_likelihood)


class AdaptiveImportance(AbstractBackend):
    """Population Monte Carlo: tempered rounds of refitted Gaussian mixtures.

    Round zero draws ``n_round`` points from ``proposal``. When ``proposal`` is a
    :class:`~photomancy.infer.proposals.PriorProposal` the target is tempered,
    ``prior * likelihood ** beta`` from ``beta = 0``: each earlier round weights its
    draws to the next temperature, the largest at which the Kish size of the cloud
    that the next proposal is fitted from stays at or above ``target_cess`` times its
    size at the current temperature and at or above ``2 * n_centers``
    (:func:`~photomancy.infer.diagnostics.kish_temperature`), and refits the proposal to
    that weighted cloud (``AbstractProposal.adapt``: by default
    ``MixtureProposal.from_cloud`` mixed with the prior, share ``defensive``). Any other
    proposal is a seed that already targets the posterior: the rounds start at
    temperature one and each refits on the weights of the fit's own density. The last
    round draws ``n_draws`` points from the final proposal and weights them to the
    fit's own density, so the returned weights and evidence come from that round alone
    and are importance estimates from independent draws. The round count is fixed and
    there is no early exit.

    The ladder needs one rung per factor of the likelihood that the cloud can absorb,
    and a concentrated posterior needs about a dozen. If the last proposal was fitted
    below temperature one (``info.reached`` false), the slice is refused: the
    evidence, its standard error and the log weights are NaN, as for ``SMC`` past
    ``max_rungs``, because weights from a proposal that missed the posterior can carry
    one live draw, a small k-hat and a wrong evidence. The default three rounds suit a
    proposal that already covers the posterior (a seed from ``Laplace`` or a previous
    fit). A start from the prior on a concentrated posterior needs tens of rounds
    (about 40 of 2**13 draws for an eight-parameter orbit) or the ``Laplace`` route.

    Attributes:
        proposal: Proposal of round zero (for example ``PriorProposal()``, or a mixture
            fitted to a Laplace cloud, which starts at temperature one).
        n_draws: Draws of the last round, the returned cloud.
        n_round: Draws of every earlier round.
        n_rounds: Total rounds (at least two for any adaptation).
        n_centers: Kernels of each refitted mixture.
        defensive: Share of the prior in each refitted proposal.
        target_cess: Fraction of the Kish size that a rung may keep.
    """

    proposal: AbstractProposal
    n_draws: int = eqx.field(static=True, default=2000)
    n_round: int = eqx.field(static=True, default=2**16)
    n_rounds: int = eqx.field(static=True, default=3)
    n_centers: int = eqx.field(static=True, default=256)
    defensive: float = eqx.field(static=True, default=0.1)
    target_cess: float = eqx.field(static=True, default=0.5)

    def __check_init__(self):
        """Refuse a round count below one."""
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be at least 1, got {self.n_rounds}")

    @eqx.filter_jit
    def _run(self, fit, key, *, batch_size=None):
        """Run the rounds; return the last round's weighted cloud and evidence."""
        proposal = self.proposal
        tempering = isinstance(proposal, PriorProposal)
        min_size = 2.0 * self.n_centers
        beta = jnp.zeros(()) if tempering else jnp.ones(())
        temperatures, kish = [], []
        for _ in range(self.n_rounds - 1):
            k_draw, k_fit, key = jax.random.split(key, 3)
            z, log_q = proposal.sample(k_draw, fit, self.n_round)
            x, volume = _stored(fit, z)
            if tempering:
                log_prior = fit.log_prior(x, batch_size=batch_size) + volume
                log_like = fit.log_likelihood(x, batch_size=batch_size)
                log_base = log_prior - log_q
                beta = kish_temperature(
                    log_base, log_like, beta, self.target_cess, min_size
                )
                log_w = jnp.where(
                    jnp.isneginf(log_prior),
                    -jnp.inf,
                    log_base + _temper(log_like, beta),
                )
            else:
                log_w = fit.log_density(x, batch_size=batch_size) + volume - log_q
            temperatures.append(beta)
            kish.append(kish_ess(log_w))
            proposal = proposal.adapt(
                k_fit,
                z,
                log_w,
                fit,
                n_centers=self.n_centers,
                defensive=self.defensive,
            )
        x, log_target, log_w = _weigh(fit, proposal, key, self.n_draws, batch_size)
        log_z, se = importance_evidence(log_w)
        reached = jnp.asarray(True) if self.n_rounds == 1 else beta >= 1.0
        nan = jnp.asarray(jnp.nan)
        temperatures.append(jnp.ones(()))
        kish.append(kish_ess(log_w))
        info = ImportanceInfo(
            self.n_rounds, jnp.stack(temperatures), jnp.stack(kish), reached
        )
        return BackendDraws(
            samples=x,
            log_weights=jnp.where(reached, log_w, nan),
            log_evidence=jnp.where(reached, log_z, nan),
            log_evidence_se=jnp.where(reached, se, nan),
            info=info,
            kind="monte_carlo",
            log_target=log_target,
        )
