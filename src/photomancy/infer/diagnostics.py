"""Free diagnostics of a weighted draw set: pure functions of arrays.

Conventions (shared by every function here):

* ``log_weights`` has shape ``(*batch, S)``: the draw axis is the LAST axis and
  every leading axis is a batch axis.
* ``values`` has shape ``(*batch, S, *event)``: the same batch prefix, then the
  draw axis, then any per-draw trailing axes.
* Every function acts on one slice and is mapped over the batch prefix, so the
  result has shape ``(*batch, *out)`` and a caller may also wrap a function in
  ``jax.vmap`` over a batch prefix.
* Log weights are unnormalized. NaN counts as zero weight. A slice with no
  finite log weight has no support and returns NaN statistics.
* Nothing forces a dtype: float64 inputs stay float64.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = [
    "khat_threshold",
    "kish_ess",
    "mc_se",
    "normalize_log_weights",
    "pareto_khat",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
]

_KHAT_PRIOR_COUNT = 10.0


def _over_batch(fn, log_weights, *values, **kwargs):
    """Map ``fn(log_weights_slice, *values_slice, **kwargs)`` over the batch prefix."""
    log_weights = jnp.asarray(log_weights)
    if log_weights.ndim < 1:
        raise ValueError("log_weights needs a draw axis")
    mapped = functools.partial(fn, **kwargs)
    for _ in range(log_weights.ndim - 1):
        mapped = jax.vmap(mapped)
    return mapped(log_weights, *(jnp.asarray(v) for v in values))


def _shift_log_weights(log_weights):
    log_weights = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
    peak = jnp.max(log_weights)
    no_support = ~jnp.isfinite(peak)
    shifted = jnp.where(no_support, -jnp.inf, log_weights - peak)
    return shifted, peak, no_support


def _normalize(log_weights):
    shifted, peak, no_support = _shift_log_weights(log_weights)
    log_sum = logsumexp(jnp.where(no_support, 0.0, shifted))
    return (
        shifted - log_sum,
        jnp.where(no_support, -jnp.inf, peak + log_sum),
        no_support,
    )


def normalize_log_weights(log_weights):
    """Return ``(normalized, log_total, no_support)`` for log weights.

    Normalize relative to the largest log weight to retain precision at large
    common offsets. NaNs count as zero weight. A nonfinite maximum gives
    ``no_support=True``, all ``-inf`` normalized weights and a total of ``-inf``.
    Leading axes are batch axes; the draw axis is last.
    """
    return _over_batch(_normalize, log_weights)


def _kish(log_weights):
    shifted, _, no_support = _shift_log_weights(log_weights)
    weights = jnp.exp(shifted)
    sum_squares = jnp.sum(weights**2)
    return jnp.sum(weights) ** 2 / jnp.where(no_support, 1.0, sum_squares)


def kish_ess(log_weights):
    """Return the Kish effective sample size, or zero without support.

    Leading axes are batch axes; the draw axis is last.
    """
    return _over_batch(_kish, log_weights)


def khat_threshold(n_draws):
    """Return the usable-tail limit ``min(1 - 1/log10(S), 0.7)`` for ``S`` draws.

    A ``pareto_khat`` at or below this value is reliable for the importance
    ratios of ``S`` iid proposal draws; above it the variance of the weighted
    mean is not trustworthy. ``S`` must exceed 10 (the formula is negative or
    undefined below that).
    """
    return min(1.0 - 1.0 / math.log10(n_draws), 0.7)


def _tail_size(n):
    """Tail size ``ceil(min(n/5, 3 sqrt(n)))`` for a float or integer count."""
    return jnp.ceil(jnp.minimum(n / 5.0, 3.0 * jnp.sqrt(n)))


def _khat(log_weights):
    """PSIS tail shape of one slice (Zhang and Stephens 2009, Vehtari et al. 2024)."""
    lw = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
    lw = jnp.where(lw == jnp.inf, -jnp.inf, lw)
    n_draws = lw.shape[0]
    n_finite = jnp.sum(jnp.isfinite(lw))
    tail_max = math.ceil(min(n_draws / 5.0, 3.0 * math.sqrt(n_draws)))
    tail_max = max(1, min(tail_max, n_draws - 1))
    tail = jnp.clip(_tail_size(n_finite.astype(lw.dtype)), 1, tail_max).astype(int)

    descending = jnp.sort(lw)[::-1]
    ratios = jnp.exp(descending - descending[0])
    cutoff = ratios[tail]
    exceed = ratios[:tail_max] - cutoff
    slot = jnp.arange(tail_max)
    valid = slot < tail
    ascending = jnp.where(valid, exceed[jnp.clip(tail - 1 - slot, 0, None)], 0.0)
    count = tail.astype(ascending.dtype)

    x_max = ascending[tail - 1]
    quartile = ascending[
        jnp.clip(jnp.floor(count / 4.0 + 0.5).astype(int) - 1, 0, None)
    ]
    quartile = jnp.where(quartile > 0, quartile, x_max)
    safe_max = jnp.where(x_max > 0, x_max, 1.0)
    safe_quartile = jnp.where(x_max > 0, quartile, 1.0)

    n_grid = 30 + jnp.floor(jnp.sqrt(count))
    grid_max = 30 + math.floor(math.sqrt(tail_max))
    j = jnp.arange(1, grid_max + 1, dtype=ascending.dtype)
    grid_valid = j <= n_grid
    theta = 1.0 / safe_max + (1.0 - jnp.sqrt(n_grid / (j - 0.5))) / (
        3.0 * safe_quartile
    )
    shape_j = (
        jnp.sum(
            jnp.where(valid, jnp.log1p(-theta[:, None] * ascending[None, :]), 0.0),
            axis=1,
        )
        / count
    )
    profile = count * (jnp.log(-theta / shape_j) - shape_j - 1.0)
    profile = jnp.where(grid_valid & jnp.isfinite(profile), profile, -jnp.inf)
    post = jax.nn.softmax(profile)
    theta_hat = jnp.sum(jnp.where(post > 0, post * theta, 0.0))
    shape = jnp.sum(jnp.where(valid, jnp.log1p(-theta_hat * ascending), 0.0)) / count
    shape = (count * shape + _KHAT_PRIOR_COUNT * 0.5) / (count + _KHAT_PRIOR_COUNT)

    # Bounded ratios (no spread in the tail) have no heavy tail at all.
    shape = jnp.where(x_max > 0, shape, -jnp.inf)
    return jnp.where(n_finite >= 5, shape, jnp.nan)


def pareto_khat(log_weights):
    """Return the Pareto tail shape of the largest raw importance ratios.

    Fit a generalized Pareto distribution (Zhang and Stephens 2009, with the
    weakly informative shrinkage of Vehtari et al. 2024) to the exceedances of
    the top ``M = ceil(min(S / 5, 3 sqrt(S)))`` ratios over the largest ratio
    outside the tail. ``S`` counts the entries kept after ``-inf`` (and NaN)
    log weights are dropped. A value below 0.5 means finite variance, at or
    below ``khat_threshold(S)`` usable, and at or above 1 no finite mean.

    Valid for the raw ratios of iid proposal draws (an importance reweight or a
    Laplace-IS cloud), not for nested-sampling dead points or a grid. A slice
    with fewer than five finite entries returns NaN; a tail with no spread
    returns ``-inf``. Both compare as "not heavy" only for ``-inf``: gate on
    ``~(khat <= limit)`` to refuse NaN. Leading axes are batch axes.
    """
    return _over_batch(_khat, log_weights)


def _prepare(log_weights, values):
    normalized, _, no_support = _normalize(log_weights)
    weights = jnp.exp(normalized)
    return weights, no_support, values


def _event_shape_mask(weights, values):
    return weights.reshape(weights.shape + (1,) * (values.ndim - 1))


def _mean(log_weights, values):
    weights, no_support, values = _prepare(log_weights, values)
    expanded = _event_shape_mask(weights, values)
    mean = jnp.sum(jnp.where(expanded > 0, expanded * values, 0.0), axis=0)
    return jnp.where(no_support, jnp.nan, mean)


def weighted_mean(log_weights, values):
    """Return the weighted mean over the draw axis, shape ``(*batch, *event)``.

    Zero-weight draws are skipped, so a non-finite value attached to a zero
    weight does not poison the result.
    """
    return _over_batch(_mean, log_weights, values)


def _cov(log_weights, values):
    weights, no_support, values = _prepare(log_weights, values)
    if values.ndim == 1:
        values = values[:, None]
        scalar = True
    else:
        scalar = False
    values = values.reshape(values.shape[0], -1)
    mean = jnp.sum(jnp.where(weights[:, None] > 0, weights[:, None] * values, 0.0), 0)
    centered = jnp.where(weights[:, None] > 0, values - mean, 0.0)
    cov = (centered * weights[:, None]).T @ centered
    cov = jnp.where(no_support, jnp.nan, cov)
    return cov[0, 0] if scalar else cov


def weighted_cov(log_weights, values):
    """Return the weighted covariance, ``sum w (x - mean)(x - mean)^T``.

    ``values`` has one event axis ``d`` (shape ``(*batch, S, d)``) and the
    result is ``(*batch, d, d)``; scalar draws ``(*batch, S)`` give the
    variance ``(*batch,)``. Weights are normalized to sum to one and no
    small-sample correction is applied (the plain plug-in estimator).
    """
    return _over_batch(_cov, log_weights, values)


def _quantile(log_weights, values, *, q):
    weights, no_support, values = _prepare(log_weights, values)
    q = jnp.asarray(q, dtype=values.dtype)
    event = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)

    def one_column(column):
        order = jnp.argsort(column)
        sorted_w = weights[order]
        cum = jnp.cumsum(sorted_w)
        cum = cum / cum[-1]
        hit = (cum[None, :] >= q.reshape(-1, 1)) & (sorted_w[None, :] > 0)
        return column[order][jnp.argmax(hit, axis=1)]

    out = jax.vmap(one_column, in_axes=1, out_axes=1)(flat)
    out = out.reshape(q.shape + event)
    return jnp.where(no_support, jnp.nan, out)


def weighted_quantile(log_weights, values, q):
    """Return weighted quantiles over the draw axis.

    The quantile is the smallest value whose cumulative normalized weight
    reaches ``q`` (the inverted-CDF definition), which equals
    ``numpy.quantile(..., method="inverted_cdf")`` of the sample expanded by
    repetition for integer weights. ``q`` is a scalar or array in ``[0, 1]``
    shared by every slice; the result has shape ``(*batch, *q.shape, *event)``.
    """
    return _over_batch(_quantile, log_weights, values, q=q)


def _mc_se(log_weights, values, *, r_eff):
    weights, no_support, values = _prepare(log_weights, values)
    mean = _mean(log_weights, values)
    expanded = _event_shape_mask(weights, values)
    centered = jnp.where(expanded > 0, values - mean, 0.0)
    var = jnp.sum((expanded * centered) ** 2, axis=0)
    if r_eff is not None:
        var = var / r_eff
    return jnp.where(no_support, jnp.nan, jnp.sqrt(var))


def mc_se(log_weights, values, *, r_eff=None):
    """Return the Monte Carlo standard error of the weighted mean of ``values``.

    The delta-method error of a self-normalized importance estimate for iid
    draws, ``sqrt(sum w^2 (f - mean)^2)`` with normalized weights ``w``. With
    ``r_eff=None`` this is the iid error and it understates the error of
    correlated (MCMC) draws; a given ``r_eff`` (the relative efficiency,
    effective draws over total draws, scalar or broadcastable over the batch
    prefix) divides the variance by ``r_eff``. The result has shape
    ``(*batch, *event)``.
    """
    if r_eff is not None:
        r_eff = jnp.asarray(r_eff)
        if r_eff.ndim:
            return _over_batch(
                lambda lw, v, r: _mc_se(lw, v, r_eff=r), log_weights, values, r_eff
            )
    return _over_batch(_mc_se, log_weights, values, r_eff=r_eff)
