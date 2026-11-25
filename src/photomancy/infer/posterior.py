"""The one posterior type, built only by :func:`photomancy.infer.run`.

A :class:`Posterior` is a weighted cloud of draws of a :class:`Fit`
with the fit it targets: ``samples`` keyed by the fit's free paths, normalized log
weights, the fit's own log density at each draw, and the evidence and its standard error
where the backend has them. Every backend fills the same fields, so a consumer reads one
layout whichever method ran.

Operations. ``sample`` and ``resample`` draw from the cloud, ``with_priors``,
``append_term`` and ``replace_term`` reweight it to a changed ``Fit`` (``replace_term``
evaluates the one old term it removes and no other),
``model_at`` and ``forecast`` read the stored draws through the fit. Each acts on one
slice (one world) and raises on a batched posterior; map worlds with
``eqx.filter_vmap``. A refusal is a NaN slice, never a raise on a traced value.

Stored block coordinates are reduced to their cell (``Fit.from_real``): the angle
coordinates of an orbit chart (``W_rad``, ``w_rad``, ``M0_rad``) lie in ``[0, 2 pi)``
and their density repeats every ``2 pi``, so a cloud near the cell edge is split across
it. Take no plain moment, quantile or covariance of such a coordinate (read the orbit
through ``model_at`` instead).

Layout. The draw axis is flat and in stored order (chains are concatenated, never an
axis). Leading axes are batch axes: a stack of worlds is the same type with a longer
prefix, produced by mapping ``run`` over a stacked fit with ``eqx.filter_vmap``, and
every operation is mapped over it the same way. Shape checks read only the trailing
axes of ``log_weights`` and the batch prefix they define.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.blocks import AbstractBlock
from hwoutils.leaves import canonical
from hwoutils.model_paths import assign
from jax.flatten_util import ravel_pytree
from priorities import AbstractDistribution, Mixture, MultivariateNormal

from photomancy.infer.diagnostics import kish_ess, mc_se, pareto_khat, weighted_mean
from photomancy.infer.evidence import importance_evidence
from photomancy.infer.fit import Fit, _check_draws
from photomancy.infer.particles import (
    categorical_indices,
    systematic_indices,
    take_draws,
)

__all__ = [
    "Forecast",
    "Posterior",
    "ReweightInfo",
    "cluster_to_mixture",
    "ravel_samples",
]

_KINDS = ("monte_carlo", "quadrature")


class Forecast(eqx.Module):
    """Weighted forecast of a predictive distribution over the stored draws.

    ``noise_variance`` is the weighted mean of the per-draw predictive variance and
    ``spread_variance`` the weighted variance of the per-draw predictive mean, so
    ``noise_variance + spread_variance`` is the total variance of the mixture over
    the posterior (the law of total variance).

    Attributes:
        mean: Weighted mean of the per-draw predictive means.
        noise_variance: Weighted mean of the per-draw predictive variances.
        spread_variance: Weighted variance of the per-draw predictive means.
    """

    mean: object
    noise_variance: object
    spread_variance: object

    @property
    def variance(self):
        """Total predictive variance, ``noise_variance + spread_variance``."""
        return jax.tree.map(jnp.add, self.noise_variance, self.spread_variance)


class ReweightInfo(eqx.Module):
    """The gate reading of a reweight, one value per slice.

    ``Posterior.with_priors``, ``append_term`` and ``replace_term`` return it in
    ``info``; :func:`~photomancy.infer.update` returns it as the ``reweight`` field of
    :class:`UpdateInfo`. The reading is taken on the combined weights before the gate
    decides, so a refused slice still says why. A ``quadrature`` posterior has no
    Monte Carlo gate: ``n_eff`` and ``khat`` are NaN and ``refused`` is true when any
    weight of the slice is NaN.

    Attributes:
        n_eff: Kish effective sample size of the combined weights.
        khat: Pareto k-hat of the combined weights.
        refused: The decision: true when the slice was refused (its weights and
            evidence are NaN), by the gate or because the new support is not
            covered. After ``update``, true means the refit ran.
        support_covered: False when a prior swap could not be shown to keep the new
            support inside the old one (the rule is in
            :meth:`Posterior.with_priors`); the slice is then refused. True for every
            term change.
        log_mass_ratio: Log of the mean, under the old weights, of the per-draw
            density ratio (new over old), taken before the gate decides, so it is
            read whether or not the slice is refused. It is the log evidence
            increment the reweighting would apply. For a prior swap on a cloud that
            is the old prior (a fit with no terms) it is also the log mass of the new
            prior on the old support; on a posterior cloud it is not a prior mass.
    """

    n_eff: jax.Array
    khat: jax.Array
    refused: jax.Array
    support_covered: jax.Array
    log_mass_ratio: jax.Array


class UpdateInfo(eqx.Module):
    """What :func:`~photomancy.infer.update` returns in ``info``, one value per slice.

    Attributes:
        reweight: The gate's :class:`ReweightInfo`, read before any refit;
            ``reweight.refused`` true means the refit ran.
        refit: The refit backend's own info, equal to
            ``run(new_fit, backend, key).info`` where ``reweight.refused`` is true
            and filled elsewhere (floats NaN, integers zero, booleans false), with
            the same structure either way.
    """

    reweight: ReweightInfo
    refit: eqx.Module


def _reweight_slice(log_weights, log_ratio, min_n_eff, max_khat):
    """Reweight one slice by ``log_ratio`` and gate on the combined weights.

    Returns the new normalized log weights (all NaN on refusal), the log evidence
    increment and the standard error of that increment (both NaN on refusal), and the
    :class:`ReweightInfo` of the gate, whose ``log_mass_ratio`` is the increment
    before the decision. The increment is the self-normalized mean of the ratio under
    the old weights, so its error is the delta-method error of that mean; it is zero for
    a ratio of one. A draw with zero old weight (dead, with NaN coordinates after a
    support swap) cannot refuse the slice: the gate counts NaN and ``+inf`` at live
    draws only.
    """
    lw = jnp.asarray(log_weights, jnp.float64)
    ratio = jnp.asarray(log_ratio, jnp.float64)
    live = ~jnp.isneginf(lw)
    bad = jnp.any(jnp.isnan(lw)) | jnp.any(jnp.isnan(ratio) & live)
    bad = bad | jnp.any(jnp.isposinf(ratio) & live)
    combined = jnp.where(live, lw + ratio, -jnp.inf)
    log_z_new, _ = importance_evidence(combined)
    log_z_old, _ = importance_evidence(lw)
    log_total = log_z_new + jnp.log(lw.shape[-1])
    no_support = ~jnp.isfinite(log_total)
    new = jnp.where(bad | no_support, -jnp.inf, combined - log_total)
    top = jnp.max(jnp.where(live, ratio, -jnp.inf))
    scaled = jnp.exp(ratio - jnp.where(jnp.isfinite(top), top, 0.0))
    increment_se = mc_se(lw, scaled) / weighted_mean(lw, scaled)
    n_eff = kish_ess(new)
    khat = pareto_khat(new)
    khat_fails = jnp.isfinite(max_khat) & ~(khat <= max_khat)
    refuse = bad | no_support | ~(n_eff >= min_n_eff) | khat_fails
    nan = jnp.asarray(jnp.nan, jnp.float64)
    return (
        jnp.where(refuse, nan, new),
        jnp.where(refuse, nan, log_z_new - log_z_old),
        jnp.where(refuse, nan, increment_se),
        ReweightInfo(n_eff, khat, refuse, jnp.asarray(True), log_z_new - log_z_old),
    )


def _live_draws(samples, log_weights):
    """``samples`` with each dead draw replaced by a live one, shapes unchanged.

    A dead draw (zero weight) may hold NaN coordinates after a support swap; a term
    or prior evaluated there returns garbage that is masked afterwards. Evaluating at a
    live draw instead keeps every evaluation on valid values.
    """
    live = ~jnp.isneginf(log_weights)
    first = jnp.argmax(live)

    def pick(a):
        mask = live.reshape(live.shape + (1,) * (a.ndim - 1))
        return jnp.where(mask, a, a[first])

    return jax.tree.map(pick, samples)


def _is_law(x):
    return isinstance(x, AbstractDistribution)


def _pieces(law):
    """Support of a law as a union of closed intervals, ``(low, high)``.

    Each array has the law's support shape plus a trailing axis of pieces. A
    ``Mixture`` contributes the pieces of every component with nonzero weight (a weight
    is zero only where its normalized log weight is ``-inf``, decided per slice when
    traced); a dropped piece is the empty interval ``(+inf, -inf)``. Any other law is
    one piece, its ``support()``. The edges are NaN where the law's own ``support()``
    is NaN (parameters outside their domain).
    """
    if not isinstance(law, Mixture):
        low, high = law.support()
        return low[..., None], high[..., None]
    low, high = _pieces(law.component_distribution)
    log_w = jax.nn.log_softmax(law.mixing_logits, -1)
    log_w = log_w[..., None] if law.event_dim else log_w
    active = ~jnp.isneginf(log_w[..., None])
    low = jnp.where(active, low, jnp.inf)
    high = jnp.where(active, high, -jnp.inf)
    axis = low.ndim - 2 - law.event_dim
    low, high = (jnp.moveaxis(a, axis, -2) for a in (low, high))
    low, high = (a.reshape(*a.shape[:-2], -1) for a in (low, high))
    hull_low, hull_high = law.support()
    invalid = (jnp.isnan(hull_low) | jnp.isnan(hull_high))[..., None]
    return jnp.where(invalid, jnp.nan, low), jnp.where(invalid, jnp.nan, high)


def _union_contains(old, new):
    """Whether the union of the ``new`` pieces lies inside that of the ``old`` ones.

    ``old`` and ``new`` are ``(low, high)`` pairs from :func:`_pieces`. The old pieces
    are merged into connected components (sorted by lower edge; a piece that starts at
    or below the running upper edge joins the component), and every non-empty new
    piece must lie inside one component. An unbounded old side constrains nothing; an
    unbounded new side where the old one is bounded is not inside; a NaN edge is never
    inside. Returns a traced boolean, one per slice.
    """
    old_low, old_high = old
    new_low, new_high = new
    order = jnp.argsort(old_low, axis=-1)
    low = jnp.take_along_axis(old_low, order, -1)
    high = jnp.take_along_axis(old_high, order, -1)
    reach = jax.lax.cummax(high, axis=high.ndim - 1)
    before = jnp.concatenate(
        [jnp.full_like(reach[..., :1], -jnp.inf), reach[..., :-1]], -1
    )
    component = jnp.cumsum(low > before, -1)
    same = component[..., :, None] == component[..., None, :]
    comp_low = jnp.min(jnp.where(same, low[..., None, :], jnp.inf), -1)
    comp_high = jnp.max(jnp.where(same, high[..., None, :], -jnp.inf), -1)
    inside = (comp_low[..., None, :] <= new_low[..., :, None]) & (
        new_high[..., :, None] <= comp_high[..., None, :]
    )
    empty = new_low > new_high
    held = jnp.all(jnp.any(inside, -1) | empty, -1)
    nan = jnp.any(jnp.isnan(old_low) | jnp.isnan(old_high), -1)
    nan = nan | jnp.any(jnp.isnan(new_low) | jnp.isnan(new_high), -1)
    return jnp.all(held & ~nan)


_LAW = object()


def _block_covered(old, new):
    """Whether the support of block ``new`` lies inside that of block ``old``.

    A block that defines ``support_contains(new)`` decides for itself. Otherwise every
    law the block holds (each ``priorities`` distribution in its fields, at any depth,
    an orbit law's scalar laws included) is compared by support, paired by position,
    and everything else (the tree structure, every other leaf and every static field)
    must be equal (``eqx.tree_equal``): a block that builds a law from numeric fields,
    or whose map reads another field such as a time, cannot show containment, so any
    change there refuses.
    """
    contains = getattr(old, "support_contains", None)
    if contains is not None:
        return jnp.all(jnp.asarray(contains(new)))

    def rest(block):
        return jax.tree.map(lambda x: _LAW if _is_law(x) else x, block, is_leaf=_is_law)

    same = eqx.tree_equal(rest(old), rest(new))
    if same is False:
        return jnp.asarray(False)
    covered = jnp.asarray(same)
    pairs = zip(
        (x for x in jax.tree.leaves(old, is_leaf=_is_law) if _is_law(x)),
        (x for x in jax.tree.leaves(new, is_leaf=_is_law) if _is_law(x)),
        strict=True,
    )
    for old_law, new_law in pairs:
        covered = covered & _union_contains(_pieces(old_law), _pieces(new_law))
    return covered


def _support_covered(old, new):
    """Whether the support of entry ``new`` lies inside that of ``old`` (traced).

    A leaf law is compared as a union of intervals per element (:func:`_pieces`), so a
    mixture's gap between components is not support; a block by
    :func:`_block_covered`. Containment that cannot be shown is false.
    """
    if isinstance(old, AbstractBlock):
        return _block_covered(old, new)
    return _union_contains(_pieces(old), _pieces(new))


def _support_swap(old_fit, new_fit, moved):
    """Per-draw function re-expressing one draw through the ``moved`` new blocks.

    Returns ``one(x) -> (x_new, log_ratio, log_target_ratio)``. ``log_target_ratio`` is
    the change of the normalized prior at the draw (the target in the new
    coordinates); ``log_ratio`` adds the log Jacobian of the old-to-new coordinate map,
    the weight factor of a draw whose cloud density lives in the old coordinates. Both
    are ``-inf`` where the new support does not hold the draw; such a draw keeps the
    NaN coordinates of ``new_block.inverse`` so that it is visibly dead in every later
    use.
    """

    def one(x):
        model = old_fit.model_at(x)
        x_new = dict(x)
        logdet = jnp.zeros((), jnp.float64)
        finite = jnp.asarray(True)
        for key in moved:
            old_block, new_block = old_fit.free[key], new_fit.free[key]
            x_new[key] = new_block.inverse(model)
            flat, unravel = ravel_pytree(x[key])

            def to_new(v, old_block=old_block, new_block=new_block, unravel=unravel):
                mapped = model
                for path, value in old_block.forward(unravel(v), model).items():
                    mapped = assign(mapped, path, canonical(value))
                return ravel_pytree(new_block.inverse(mapped))[0]

            logdet = logdet + jnp.linalg.slogdet(jax.jacfwd(to_new)(flat))[1]
            finite = finite & jnp.all(jnp.isfinite(ravel_pytree(x_new[key])[0]))
        old = old_fit._log_prior_one(x)
        new = new_fit._log_prior_one(x_new)
        outside = ~finite | jnp.isneginf(old) | ~jnp.isfinite(logdet)
        outside = outside | jnp.isneginf(new)
        target = jnp.where(outside, -jnp.inf, new - old)
        ratio = jnp.where(outside, -jnp.inf, target + logdet)
        return x_new, ratio, target

    return one


class Posterior(eqx.Module):
    """A weighted cloud of draws of a ``Fit`` and the density it targets.

    Attributes:
        samples: Entry dict with exactly the keys of ``fit.free``. A leaf-law entry is
            ``(*batch, N, *leaf_shape)``; a block entry is a dict of coordinate arrays
            ``(*batch, N, R)``. The draw axis is flat, in the backend's stored order.
        log_weights: Normalized log weights ``(*batch, N)``, float64
            (``logsumexp`` over ``N`` is zero). A slice that has no support is NaN.
        log_target: ``fit.log_density`` at the draws ``(*batch, N)``, float64: the
            normalized prior plus terms against Lebesgue measure on the stored
            coordinates, exactly ``-inf`` off the prior's support. It is never a
            proposal density.
        log_evidence: Log marginal likelihood ``(*batch,)``, float64. This is the one
            statement of the evidence contract; every other docstring refers here. The
            value is NaN when the backend has none (the MCMC backends, ``MAP``: no
            estimate is invented). A finite value means all three of the following
            held:

            1. Tail: the Pareto k-hat of the weights is at or below
               ``khat_threshold(n_draws)``, so their variance is finite (importance
               backends only).
            2. Proposal validity: the proposal reports its weights valid,
               ``info.proposal_valid``. A proposal defines its own condition; one with
               none always reports valid.
            3. No refusal: the run reached its target (a tempering ladder that
               reached temperature one, a mode kept, a nested-sampling run that
               converged and was not cut by a hard stop, no NaN weight) and every
               later gate passed (``with_priors``, ``append_term``, ``replace_term``).

            A refused run keeps its draws and weights; only the evidence and its
            standard error are NaN. The contract has one limit that no check can
            detect: a fit that misses a mode entirely gives a finite evidence that is
            wrong. ``log_evidence_se`` follows the same rule.
        log_evidence_se: Standard error of ``log_evidence`` ``(*batch,)``, float64, NaN
            when the backend has none.
        fit: The ``Fit`` whose density the weights target.
        info: The backend's diagnostics module (chain index, divergence flags, thin
            factor, ladder, stage traces), or ``None``. Chain structure lives here and
            never in ``samples``.
        kind: Describes the draws: ``"monte_carlo"`` (random draws, so a random
            estimator) or ``"quadrature"`` (a positive-weight rule whose error is
            judged by refinement). A resampled grid is ``"monte_carlo"``.
    """

    samples: dict
    log_weights: jax.Array
    log_target: jax.Array
    log_evidence: jax.Array
    log_evidence_se: jax.Array
    fit: Fit
    info: eqx.Module | None
    kind: str = eqx.field(static=True)

    def __check_init__(self):
        """Refuse a malformed posterior on the host.

        Raises:
            ValueError: For a wrong ``kind``, key set, dtype or shape.
        """
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        leading = self.fit.check_draws(self.samples)
        for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
            dtype = jnp.asarray(getattr(self, name)).dtype
            if dtype != jnp.float64:
                raise ValueError(f"{name} must be float64, got {dtype}")
        shape = jnp.shape(self.log_weights)
        if not shape:
            raise ValueError("log_weights needs a draw axis")
        batch = shape[:-1]
        if jnp.shape(self.log_target) != shape:
            raise ValueError(
                f"log_target has shape {jnp.shape(self.log_target)}, "
                f"log_weights {shape}"
            )
        for name in ("log_evidence", "log_evidence_se"):
            if jnp.shape(getattr(self, name)) != batch:
                raise ValueError(
                    f"{name} has shape {jnp.shape(getattr(self, name))}, "
                    f"expected the batch shape {batch}"
                )
        if self.samples and leading != shape:
            key = next(iter(self.samples))
            raise ValueError(
                f"samples[{key!r}] has leading axes {leading}, expected {shape}"
            )

    # ----- draws -----------------------------------------------------------------

    def _batch_ndim(self):
        return jnp.ndim(self.log_weights) - 1

    def _require_slice(self, name):
        if self._batch_ndim():
            raise ValueError(
                f"Posterior.{name} takes one slice; map it over the batch with "
                "eqx.filter_vmap"
            )

    def sample(self, key, sample_shape=()):
        """Draw independent samples from the weighted cloud, with replacement.

        Categorical draws on the normalized weights (inverse CDF, so memory is linear in
        the draw count): duplicates occur, and for a
        correlated cloud (MCMC chains) the draws bootstrap the chain. Use
        :meth:`resample` to build a new fixed-size cloud. A slice with no support
        returns NaN draws. Takes one slice; map worlds with ``eqx.filter_vmap``.

        Args:
            key: PRNG key.
            sample_shape: Static shape of the draws.

        Returns:
            Entry dict, leaves ``(*sample_shape, *leaf_shape)``.
        """
        self._require_slice("sample")
        sample_shape = tuple(sample_shape)
        lw = jnp.asarray(self.log_weights)
        none = jnp.any(jnp.isnan(lw))
        idx = categorical_indices(key, lw, sample_shape)
        out = take_draws(self.samples, idx)
        return jax.tree.map(lambda leaf: jnp.where(none, jnp.nan, leaf), out)

    def resample(self, key, n=None):
        """Return a cloud of ``n`` systematically resampled draws with uniform weights.

        Fixed size (``n`` defaults to the current size), one random offset per slice.
        The draws are mutually dependent and chain structure is dropped, so ``info``
        becomes ``None``; ``log_target`` is indexed, the evidence, its standard error
        and the fit are kept. The result is ``kind="monte_carlo"`` whatever the input
        kind: a random subsample of a grid is a random cloud, so a later reweighting
        needs the Monte Carlo gate arguments. A slice with no support keeps NaN
        weights.

        Args:
            key: PRNG key.
            n: Static number of draws, default the stored count.

        Returns:
            The resampled posterior, with uniform log weights ``-log(n)``.
        """
        self._require_slice("resample")
        lw = jnp.asarray(self.log_weights)
        size = lw.shape[-1] if n is None else int(n)
        none = jnp.any(jnp.isnan(lw))
        safe = jnp.where(none, -math.log(lw.shape[-1]), lw)
        idx = systematic_indices(key, safe, size)
        weights = jnp.full(size, jnp.where(none, jnp.nan, -math.log(size)))
        return Posterior(
            samples=take_draws(self.samples, idx),
            log_weights=weights.astype(jnp.float64),
            log_target=jnp.take(self.log_target, idx, axis=0),
            log_evidence=self.log_evidence,
            log_evidence_se=self.log_evidence_se,
            fit=self.fit,
            info=None,
            kind="monte_carlo",
        )

    # ----- reweighting -----------------------------------------------------------

    def _check_gate(self, min_n_eff, max_khat):
        for name, value in (("min_n_eff", min_n_eff), ("max_khat", max_khat)):
            if (
                value is not None
                and not isinstance(value, jax.core.Tracer)
                and bool(np.isnan(np.asarray(value)))
            ):
                raise ValueError(
                    f"{name} is NaN, which would turn its test off or refuse every "
                    "slice; use jnp.inf for max_khat (no limit) or 0 for min_n_eff"
                )
        given = (min_n_eff is not None, max_khat is not None)
        if self.kind == "quadrature":
            if any(given):
                raise ValueError(
                    "a quadrature posterior has no Monte Carlo gate: pass "
                    "min_n_eff=None and max_khat=None (the error is judged by "
                    "refinement)"
                )
        elif not all(given):
            raise ValueError("min_n_eff and max_khat are required for monte_carlo")

    def _reweighted(
        self,
        new_fit,
        log_ratio,
        min_n_eff,
        max_khat,
        *,
        samples=None,
        log_target_ratio=None,
        support_covered=None,
    ):
        """Reweight by ``log_ratio``, gated.

        ``samples`` and ``log_target_ratio`` default to the stored draws and
        ``log_ratio``; they differ after a change of support. A false
        ``support_covered`` (default true) refuses the slice.
        """
        self._check_gate(min_n_eff, max_khat)
        quadrature = self.kind == "quadrature"
        lo = 0.0 if quadrature else min_n_eff
        hi = jnp.inf if quadrature else max_khat
        new_lw, increment, increment_se, info = _reweight_slice(
            self.log_weights, log_ratio, lo, hi
        )
        if support_covered is not None:
            covered = jnp.asarray(support_covered)
            nan = jnp.asarray(jnp.nan, jnp.float64)
            new_lw = jnp.where(covered, new_lw, nan)
            increment = jnp.where(covered, increment, nan)
            increment_se = jnp.where(covered, increment_se, nan)
            info = ReweightInfo(
                info.n_eff,
                info.khat,
                info.refused | ~covered,
                covered,
                info.log_mass_ratio,
            )
        if quadrature:
            refused = jnp.any(jnp.isnan(new_lw))
            increment_se = jnp.where(refused, jnp.nan, 0.0)
            nan = jnp.asarray(jnp.nan, jnp.float64)
            info = ReweightInfo(
                nan, nan, refused, info.support_covered, info.log_mass_ratio
            )
        target_ratio = log_ratio if log_target_ratio is None else log_target_ratio
        return Posterior(
            samples=self.samples if samples is None else samples,
            log_weights=new_lw,
            log_target=jnp.asarray(self.log_target, jnp.float64) + target_ratio,
            log_evidence=self.log_evidence + increment,
            log_evidence_se=jnp.sqrt(self.log_evidence_se**2 + increment_se**2),
            fit=new_fit,
            info=info,
            kind=self.kind,
        )

    def with_priors(self, priors, *, min_n_eff, max_khat, batch_size=None):
        """Swap priors by importance reweighting; no term is evaluated.

        The new ``Fit`` comes from ``fit.with_priors(priors)`` (which refuses a
        changed free set and a block with another map). Weights become
        ``log_weights + log pi_new - log pi_old`` and ``log_target`` follows; the
        evidence gains ``log sum w pi_new / pi_old``. Its standard error is the old
        error and the delta-method error of that self-normalized mean under the old
        weights, added in quadrature, so an identity swap leaves both unchanged. Exact
        in expectation only for normalized priors.

        The gate runs on the combined reweighted weights (``log_weights`` plus the
        log ratio), never the ratio alone, so it also re-gates the input cloud's own
        tail: an identity swap on a heavy-tailed cloud refuses. A slice that fails
        (Kish size below ``min_n_eff``, k-hat above ``max_khat`` by ``~(khat <=
        max_khat)`` so NaN refuses, zero total weight, a NaN ratio) comes back with
        all-NaN ``log_weights``, evidence and standard error; the caller decides per
        world with ``jnp.isnan``. ``max_khat=jnp.inf`` skips the k-hat test and
        ``min_n_eff=0`` the size test. Clouds under about 25 draws always fail a
        k-hat gate (fewer than five tail draws). The size is the plain Kish size:
        multiply by a chain efficiency yourself for correlated draws. The reading of
        the gate (that Kish size, that k-hat and the decision) comes back in ``info``
        as a :class:`ReweightInfo`, for a refused slice too.

        ``min_n_eff`` is an absolute Kish count that the caller states, with no
        default: it is an accuracy requirement, like a solver tolerance, and it is
        not a fraction of ``n_draws``. The error of the evidence increment, of a
        forecast and of the next proposal all scale with the Kish size alone: one
        evidence increment has a standard error of about ``1 / sqrt(n_eff)`` nat and a
        forecast probability ``p`` has ``sqrt(p (1 - p) / n_eff)``. A value of 400
        gives 0.05 nat and at most 0.025 on a probability; 100 gives 0.1 nat and
        0.05. The k-hat gate does not replace it: a cloud that has collapsed onto one
        draw can read a small k-hat, and only the count refuses it.

        A block whose new law moves the bounds its map carries (an orbit law with other
        size or eccentricity bounds) is another map, so the stored coordinates change
        meaning. Each stored draw is then read through the old block into the model and
        re-expressed with ``new_block.inverse(model)``; ``samples`` come back in the new
        coordinates, so ``log_target`` still equals ``fit.log_density`` of the result.
        The weight gains the ratio of the two laws on the orbit, computed as the new
        block density plus the log Jacobian of the old-to-new coordinate map (by
        forward-mode autodiff) minus the old block density. A draw outside the new
        support has weight ``-inf`` (and ``-inf`` ``log_target``; its stored
        coordinates are not finite). The evidence gains ``log sum w ratio``, which is
        the new prior's mass on the cloud's support when the old cloud is the prior.
        The swap can narrow a prior but not widen one: a cloud has no draws where the
        old prior had no mass, so for each replaced entry the new support must lie
        inside the old one, and a slice where that cannot be shown is refused. A leaf
        law is compared by its support as a union of intervals: a ``Mixture`` covers
        the union of its components with nonzero weight, not their hull, so a gap
        between components is not support. A block is compared field by field: each
        law it holds (each ``priorities`` distribution at any depth, the scalar laws
        of an orbit law included) by support as above, and everything else (its
        structure, every other leaf and every static field) must be equal
        (``eqx.tree_equal``). So a block that builds its law from numeric fields
        refuses any change of them, another epoch of a sky-position block refuses,
        and so does a change of parameterization (a size law on ``a_AU`` replaced by
        one on ``period_d``, even an equivalent one): express both priors in the same
        parameterization. A block that can decide containment itself may define
        ``support_contains(self, new)``, returning a boolean or a boolean array (true
        where the support of ``new`` lies inside its own); it replaces the field
        comparison (the orbix blocks define none). A refused slice is refused
        like a gate failure (NaN weights, evidence and standard error;
        ``info.refused`` true, ``info.support_covered`` false), also for a
        ``quadrature`` posterior, and a widening needs a refit.
        ``info.log_mass_ratio`` is read whether or not the slice is refused; for a
        cloud that is the old prior, its exponential is the mass of the new prior on
        the old support. A change of chart, rows, system or block type is refused by
        name. The Jacobian is taken with the other entries of the draw held, which is
        exact because the orbit itself does not move.

        Where the gate does not apply: k-hat is not valid on nested-sampling clouds
        (pass ``max_khat=jnp.inf`` and gate on ``min_n_eff``) and a ``quadrature``
        cloud has no gate (see below). A ``MAP`` posterior has one draw, so any
        ``min_n_eff`` above 1 refuses it. NaN is propagated, never repaired. Stored
        block angle coordinates lie in one cell, so a cloud near the cell edge is
        split across it: take no plain moment, quantile or covariance of one; the swap
        itself is unaffected because the priors and terms are periodic.

        Args:
            priors: Replacement entries, a subset of ``fit.free`` with the same kinds.
            min_n_eff: Smallest acceptable Kish size of the new weights, or ``None``
                for a ``quadrature`` posterior (which has no gate).
            max_khat: Largest acceptable Pareto k-hat, or ``None`` for ``quadrature``.
            batch_size: Draws whose priors are evaluated at once (a ``jax.lax.map``
                chunk, the same values to rounding), default all at once.

        Returns:
            The reweighted posterior on the new fit, with ``info`` a
            :class:`ReweightInfo`.

        Raises:
            ValueError: On the host, for a static fault in ``priors``, a NaN
                ``min_n_eff`` or ``max_khat``, or gate arguments that do not match
                ``kind``.
        """
        self._require_slice("with_priors")
        new_fit = self.fit.with_priors(priors)
        moved = tuple(
            key
            for key, entry in priors.items()
            if isinstance(entry, AbstractBlock)
            and not self.fit.free[key].same_map(entry)
        )
        covered = jnp.asarray(True)
        for key in priors:
            covered = covered & _support_covered(self.fit.free[key], new_fit.free[key])
        live = _live_draws(self.samples, self.log_weights)
        dead = jnp.isneginf(self.log_weights)
        if not moved:
            old = self.fit.log_prior(live, batch_size=batch_size)
            new = new_fit.log_prior(live, batch_size=batch_size)
            ratio = jnp.where(jnp.isneginf(old) | dead, -jnp.inf, new - old)
            return self._reweighted(
                new_fit, ratio, min_n_eff, max_khat, support_covered=covered
            )
        one = _support_swap(self.fit, new_fit, moved)
        x_new, ratio, target_ratio = self.fit._over_batch(one, live, batch_size)
        ratio = jnp.where(dead, -jnp.inf, ratio)
        target_ratio = jnp.where(dead, -jnp.inf, target_ratio)
        x_new = jax.tree.map(
            lambda new, old: jnp.where(
                dead.reshape(dead.shape + (1,) * (new.ndim - 1)), old, new
            ),
            x_new,
            self.samples,
        )
        return self._reweighted(
            new_fit,
            ratio,
            min_n_eff,
            max_khat,
            samples=x_new,
            log_target_ratio=target_ratio,
            support_covered=covered,
        )

    def _term_ratio(self, term, old, batch_size):
        """Per-draw ``term(model) - old(model)`` (just ``term`` when ``old`` is None).

        The one code path of ``append_term`` and ``replace_term``: each stored draw is
        built into its model once and both terms read it.
        """
        fit = self.fit

        def one(x):
            model = fit.model_at(x)
            ratio = jnp.asarray(term(model), jnp.float64)
            return (
                ratio if old is None else ratio - jnp.asarray(old(model), jnp.float64)
            )

        dead = jnp.isneginf(self.log_weights)
        ratio = fit._over_batch(
            one, _live_draws(self.samples, self.log_weights), batch_size
        )
        return jnp.where(dead, -jnp.inf, ratio)

    def append_term(self, term, *, min_n_eff, max_khat, batch_size=None):
        """Add one observation by importance reweighting; no old term is evaluated.

        The new ``Fit`` comes from ``fit.append_term(term)``. The term is evaluated at
        the model of every stored draw, weights gain ``+ term``, ``log_target`` follows
        and the evidence gains ``log sum w exp(term)``, with the standard error updated
        as in :meth:`with_priors`. The gate is the same, on the combined weights, and
        its reading is returned in ``info``.

        The result is consistent with bias of order ``1 / N_eff``, not exact, and it
        holds only when (a) the new datum is conditionally independent of the old given
        the model, (b) the term reads only parameters already in the cloud (``free`` is
        never changed), (c) the held values and builds are the cloud's (the term is
        evaluated on ``fit.model_at``), and (d) the gate passes.

        The gate arguments are required and keyword-only. A ``monte_carlo`` posterior
        needs both; a ``quadrature`` posterior has no gate and needs both written out as
        ``None``.

        Args:
            term: An :class:`~photomancy.infer.fit.AbstractTerm`, normalized.
            min_n_eff: As in :meth:`with_priors`.
            max_khat: As in :meth:`with_priors`.
            batch_size: As in :meth:`with_priors`, for the term evaluations.

        Returns:
            The reweighted posterior on the new fit, with ``info`` a
            :class:`ReweightInfo`.

        Raises:
            ValueError: On the host, for a term whose reads meet a free path, a NaN
                gate limit, or gate arguments that do not match ``kind``.
        """
        self._require_slice("append_term")
        new_fit = self.fit.append_term(term)
        ratio = self._term_ratio(term, None, batch_size)
        return self._reweighted(new_fit, ratio, min_n_eff, max_khat)

    def replace_term(self, index, term, *, min_n_eff, max_khat, batch_size=None):
        """Swap one term by importance reweighting; every other term is untouched.

        The new ``Fit`` comes from ``fit.replace_term(index, term)``. The weights gain
        ``new(model) - old(model)`` at every stored draw, with ``old`` the term
        ``fit.terms[index]``. The evidence, its standard error, ``log_target``, the
        gate and ``info`` are exactly those of :meth:`append_term`: the combined
        weights are gated by ``min_n_eff`` and ``max_khat``, a NaN ratio refuses the
        slice, and the same conditions (a) to (d) apply, with the datum replaced rather
        than added.

        The use is a padded row: the old term holds the row as invalid padding and the
        new term the same data with the row valid. The structure is unchanged, so a
        stack of worlds does not recompile, and the result equals ``append_term`` of
        that row on an unpadded fit. A draw where both terms are ``-inf`` gives a NaN
        ratio and refuses the slice. ``index`` is a static Python integer.

        Args:
            index: Position of the term to replace in ``fit.terms``.
            term: The replacement, an :class:`~photomancy.infer.fit.AbstractTerm`.
            min_n_eff: As in :meth:`with_priors`.
            max_khat: As in :meth:`with_priors`.
            batch_size: As in :meth:`with_priors`, for the term evaluations.

        Returns:
            The reweighted posterior on the new fit, with ``info`` a
            :class:`ReweightInfo`.

        Raises:
            IndexError: For an ``index`` outside ``fit.terms``.
            ValueError: As :meth:`append_term`.
        """
        self._require_slice("replace_term")
        new_fit = self.fit.replace_term(index, term)
        ratio = self._term_ratio(term, self.fit.terms[index], batch_size)
        return self._reweighted(new_fit, ratio, min_n_eff, max_khat)

    # ----- reading the draws through the fit -------------------------------------

    def model_at(self, i):
        """Return the model of stored draw ``i``, every held leaf included.

        Args:
            i: Index of the stored draw along the draw axis.

        Returns:
            The base model with the free entries of draw ``i`` written.
        """
        self._require_slice("model_at")
        x = jax.tree.map(lambda a: jnp.take(a, i, axis=0), self.samples)
        return self.fit.model_at(x)

    def forecast(self, fn, *, batch_size=None):
        """Weighted forecast and its total-variance split over the stored draws.

        ``fn(model) -> (mean, variance)`` is a predictive mean and variance of a future
        datum given the model (pytrees of arrays, equal structure); the datum is taken
        independent of the old data given the model. The result is the weighted mean
        of the means, the weighted mean of the variances (``noise_variance``) and the
        weighted variance of the means (``spread_variance``); their sum is the
        total variance. ``fn`` is traced over ``fit.model_at(x)``, so it must be
        jittable and may use only builds made from ``fit.model``; a path the fit varies
        must not be read through a host build.

        Subsample with ``p.resample(key, m).forecast(fn)``: the variance of a mean over
        ``m`` resampled draws is inflated by about ``1 + N_eff / m`` relative to the
        weighted cloud.

        A dead draw (zero weight, possibly NaN coordinates after a support swap) is
        evaluated at a live draw instead and keeps zero weight.

        Args:
            fn: ``model -> (mean, variance)``.
            batch_size: Draws evaluated at once (``jax.lax.map`` chunk), default all
                at once with ``jax.vmap``.

        Returns:
            A :class:`Forecast` of the weighted mean, noise variance and spread
            variance.
        """
        self._require_slice("forecast")
        fit = self.fit
        samples = _live_draws(self.samples, self.log_weights)

        def one(x):
            return fn(fit.model_at(x))

        if batch_size is None:
            means, variances = jax.vmap(one)(samples)
        else:
            means, variances = jax.lax.map(one, samples, batch_size=batch_size)
        w = jnp.exp(jnp.asarray(self.log_weights, jnp.float64))

        def average(a):
            return jnp.tensordot(w, jnp.asarray(a, jnp.float64), axes=(0, 0))

        mean = jax.tree.map(average, means)
        noise = jax.tree.map(average, variances)
        spread = jax.tree.map(
            lambda m, mu: average((jnp.asarray(m) - mu) ** 2), means, mean
        )
        return Forecast(mean, noise, spread)


def ravel_samples(posterior, keys=None):
    """Flatten each draw to a vector: ``(*batch, N, D)`` and an unravel function.

    The columns are the entries' leaves in the deterministic order of
    ``jax.tree.leaves`` (sorted keys, then each leaf row-major). ``unravel`` maps an
    array ``(*lead, D)`` back to the entry dict of those keys, leaves shaped
    ``(*lead, *leaf_shape)``.

    Args:
        posterior: A :class:`Posterior`.
        keys: Entries to flatten, default all of ``posterior.samples``.

    Returns:
        ``(flat, unravel)``: the draws as an array ``(*batch, N, D)`` and the
        function that maps an array ``(..., D)`` back to the entry dict.
    """
    samples = (
        posterior.samples
        if keys is None
        else {key: posterior.samples[key] for key in keys}
    )
    batch = jnp.ndim(posterior.log_weights) - 1
    leaves, treedef = jax.tree.flatten(samples)
    trailing = [jnp.shape(leaf)[batch + 1 :] for leaf in leaves]
    sizes = [math.prod(t) for t in trailing]
    lead = jnp.shape(posterior.log_weights)
    flat = jnp.concatenate(
        [
            jnp.reshape(leaf, (*lead, size))
            for leaf, size in zip(leaves, sizes, strict=True)
        ],
        axis=-1,
    )
    offsets = [sum(sizes[:i]) for i in range(len(sizes) + 1)]

    def unravel(vector):
        parts = [
            jnp.reshape(vector[..., a:b], (*jnp.shape(vector)[:-1], *t))
            for a, b, t in zip(offsets[:-1], offsets[1:], trailing, strict=True)
        ]
        return jax.tree.unflatten(treedef, parts)

    return flat, unravel


def _flat_periods(posterior, keys):
    """Periods of the raveled draw (``None`` for none) for the entries ``keys``."""
    periods = posterior.fit.periods()
    names = sorted(posterior.samples)
    sizes = {
        name: sum(
            math.prod(jnp.shape(leaf)[1:])
            for leaf in jax.tree.leaves(posterior.samples[name])
        )
        for name in names
    }
    if len(periods) != sum(sizes.values()):
        return (None,) * sum(sizes[name] for name in (keys or names))
    out, start = [], 0
    for name in names:
        if keys is None or name in keys:
            out += periods[start : start + sizes[name]]
        start += sizes[name]
    return tuple(out)


def cluster_to_mixture(
    posterior, n_modes, key, *, keys=None, init=None, n_steps=25, cov_floor=1e-6
):
    """Cluster a posterior into an ``n_modes``-mode Gaussian ``priorities.Mixture``.

    The draws, flattened by :func:`ravel_samples`, are assigned to ``n_modes`` clusters
    by weighted hard-assignment clustering: three rounds of k-means (each draw to the
    nearest center), then each draw goes to the cluster that maximizes its log mass plus
    the log density of the cluster's own Gaussian, for ``n_steps`` rounds in all. Each
    mode is the cluster's weighted mean and covariance and its weight is the cluster's
    mass, so the mixture follows the posterior's modes; draws of zero weight, such as
    the NaN dead draws of a proposal, take no part. The result is a distribution over
    the flat vector: use it as the law of one free leaf of shape ``(D,)`` in the next
    ``Fit``. It has no inverse CDF, so it serves the gradient and sample backends and
    not nested sampling.

    Distances are taken in standardized coordinates (each coordinate divided by the
    weighted standard deviation of the cloud) and wrapped on the periodic coordinates
    of ``posterior.fit``, so a coordinate with a large range does not outweigh one
    that separates the modes. From a random start the clustering can still put two
    centers on one mode and none on a minor one when the modes are of very unequal
    mass, so ``init`` is required for separated modes: pass the modes as ``init``
    (``Laplace``'s ``info.modes``, rows selected by ``info.kept``, is one such set).

    Args:
        posterior: An unbatched :class:`Posterior` (one slice; map over worlds).
        n_modes: Number of modes.
        key: PRNG key for the random start (``n_modes`` draws picked by weight); unused
            when ``init`` is given.
        keys: Entries to cluster, default all.
        init: Optional entry dict of the ``n_modes`` starting centers in stored
            coordinates (every leaf has a leading axis of length ``n_modes``, entries
            as ``keys``).
        n_steps: Clustering iterations.
        cov_floor: Diagonal added to every covariance.

    Returns:
        A :class:`priorities.Mixture` of :class:`priorities.MultivariateNormal`.
    """
    if jnp.ndim(posterior.log_weights) != 1:
        raise ValueError("cluster_to_mixture takes one slice; map it over a batch")
    x, _ = ravel_samples(posterior, keys)
    n, d = x.shape
    w = jax.nn.softmax(posterior.log_weights)
    x = jnp.where(w[:, None] > 0, x, 0.0)
    period = jnp.asarray(
        [0.0 if p is None else p for p in _flat_periods(posterior, keys)], x.dtype
    )
    safe = jnp.where(period > 0, period, 1.0)

    def offset(a, b):
        diff = a - b
        return jnp.where(period > 0, diff - safe * jnp.round(diff / safe), diff)

    mean = w @ x
    sd = jnp.sqrt(w @ offset(x, mean) ** 2)
    scale = jnp.where(sd > 0, sd, 1.0)
    if init is None:
        centers = x[jax.random.choice(key, n, (n_modes,), p=w)]
    else:
        wanted = sorted(posterior.samples if keys is None else keys)
        if sorted(init) != wanted:
            raise ValueError(f"init needs the entries {wanted}, got {sorted(init)}")
        schema = posterior.fit.draw_schema()
        leading = _check_draws(init, {name: schema[name] for name in wanted})
        if leading != (n_modes,):
            raise ValueError(
                f"init has leading shape {leading}, expected n_modes={n_modes}"
            )
        leaves = jax.tree.leaves({name: init[name] for name in wanted})
        centers = jnp.concatenate(
            [jnp.reshape(leaf, (n_modes, -1)) for leaf in leaves], axis=-1
        )

    floor = (1e-2 * scale) ** 2

    def moments(centers, mass_w):
        mass = jnp.sum(mass_w, axis=0)
        shift = jnp.einsum(
            "nk,nkd->kd", mass_w, offset(x[:, None, :], centers[None, :, :])
        )
        # An empty cluster keeps its center, a live draw, rather than the origin.
        means = jnp.where(
            mass[:, None] > 0,
            centers + shift / jnp.maximum(mass[:, None], 1e-30),
            centers,
        )

        def cov(mu, wcol):
            diff = offset(x, mu)
            return (diff * wcol[:, None]).T @ diff / jnp.maximum(jnp.sum(wcol), 1e-30)

        covs = jax.vmap(cov)(means, mass_w.T) + jnp.diag(floor)
        return means, covs, mass

    def assign(centers, covs, mass):
        chol = jnp.linalg.cholesky(covs)
        diff = offset(x[:, None, :], centers[None, :, :])
        z = jax.vmap(
            lambda c, v: jax.scipy.linalg.solve_triangular(c, v.T, lower=True),
            in_axes=(0, 1),
        )(chol, diff)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=-1)
        score = (jnp.log(jnp.maximum(mass, 1e-30)) - 0.5 * log_det)[
            :, None
        ] - 0.5 * jnp.sum(z**2, axis=1)
        return jax.nn.one_hot(jnp.argmax(score, axis=0), n_modes) * w[:, None]

    covs0 = jnp.broadcast_to(jnp.diag(scale**2), (n_modes, d, d))
    mass0 = jnp.full(n_modes, 1.0 / n_modes)

    def step(carry, i):
        centers, covs, mass = carry
        # The first rounds are plain k-means in the standardized coordinates, which
        # moves a center off a draw that shares its mode with another center; the
        # Gaussian reassignment after them keeps a cluster from drifting between modes.
        place = i < 3
        mass_w = assign(
            centers, jnp.where(place, covs0, covs), jnp.where(place, mass0, mass)
        )
        return moments(centers, mass_w), None

    state = (centers, covs0, mass0)
    (centers, covs, mass), _ = jax.lax.scan(step, state, jnp.arange(n_steps))
    means, covs, mass = moments(centers, assign(centers, covs, mass))
    covs = covs + cov_floor * jnp.eye(d)[None]
    log_mass = jnp.log(jnp.maximum(mass, 1e-30))
    return Mixture(log_mass, MultivariateNormal(means, jnp.linalg.cholesky(covs)))
