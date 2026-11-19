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
_MIN_EXCEEDANCES = 5
# Two log weights closer than this many float epsilons (relative, floored at
# one) are tied: values that went through logsumexp or a subtraction differ by
# rounding even when they are mathematically equal.
_TIE_ULPS = 64


def _tied(diff, value):
    """Whether a nonnegative log-weight difference is zero within rounding."""
    eps = jnp.finfo(diff.dtype).eps
    return diff <= _TIE_ULPS * eps * jnp.maximum(1.0, jnp.abs(value))


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
    top = descending[0]
    # Draws tied with the cutoff within rounding carry no exceedance.
    cutoff_lw = descending[tail]
    tail_lw = descending[:tail_max]
    slot = jnp.arange(tail_max)
    exceeds = (slot < tail) & ~_tied(tail_lw - cutoff_lw, tail_lw)
    n_fit = jnp.sum(exceeds)
    exceed = jnp.where(exceeds, jnp.exp(tail_lw - top) - jnp.exp(cutoff_lw - top), 0.0)
    valid = slot < n_fit
    ascending = jnp.where(valid, exceed[jnp.clip(n_fit - 1 - slot, 0, None)], 0.0)
    count = jnp.maximum(n_fit, 1).astype(ascending.dtype)
    tail = jnp.maximum(n_fit, 1)

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

    # Rounding-flat weights have no tail at all; a tail with fewer than five
    # distinct exceedances cannot be fitted and fails closed.
    low = jnp.min(jnp.where(jnp.isfinite(lw), lw, jnp.inf))
    flat = _tied(top - low, jnp.maximum(jnp.abs(top), jnp.abs(low)))
    return jnp.select(
        [n_finite < 1, flat | (n_fit == 0), n_fit < _MIN_EXCEEDANCES],
        [jnp.nan, -jnp.inf, jnp.nan],
        shape,
    )


def pareto_khat(log_weights):
    """Return the Pareto tail shape of the largest raw importance ratios.

    Fit a generalized Pareto distribution (Zhang and Stephens 2009, with the
    weakly informative shrinkage of Vehtari et al. 2024) to the exceedances of
    the top ``M = ceil(min(S / 5, 3 sqrt(S)))`` ratios over the largest ratio
    outside the tail. ``S`` counts the entries kept after ``-inf`` (and NaN)
    log weights are dropped. A value below 0.5 means finite variance, at or
    below ``khat_threshold(S)`` usable, and at or above 1 no finite mean.

    Rounding ties count as ties: log weights within 64 float epsilons of one
    another over the whole slice are flat and return ``-inf``, and draws within
    that tolerance of the cutoff are left out of the fit (discrete weight
    levels such as quadrature or bin-probability weights).

    Valid for the raw ratios of iid proposal draws (an importance reweight or a
    Laplace-IS cloud), not for nested-sampling dead points or a grid. A slice
    with no finite entry, or fewer than five distinct exceedances, returns NaN
    (the fit fails closed); flat weights return ``-inf``. Gate on
    ``~(khat <= limit)`` so that NaN refuses and ``-inf`` passes. Leading axes
    are batch axes.
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


def _mc_se(log_weights, values, g, *, r_eff):
    weights, no_support, values = _prepare(log_weights, values)
    expanded = _event_shape_mask(weights, values)
    if g is None:
        mean = _mean(log_weights, values)
        centered = jnp.where(expanded > 0, values - mean, 0.0)
        var = jnp.sum((expanded * centered) ** 2, axis=0)
        bad = no_support
    else:
        g = jnp.broadcast_to(jnp.asarray(g, dtype=weights.dtype), weights.shape)
        den = weights @ g
        bad = no_support | ~(den > 0)
        safe = jnp.where(bad, 1.0, den)
        q = jnp.sum(jnp.where(expanded > 0, expanded * values, 0.0), axis=0) / safe
        gx = g.reshape(g.shape + (1,) * (values.ndim - 1))
        resid = jnp.where(expanded > 0, values - q * gx, 0.0)
        var = jnp.sum((expanded * resid) ** 2, axis=0) / safe**2
    if r_eff is not None:
        var = var / r_eff
    return jnp.where(bad, jnp.where(no_support, jnp.nan, jnp.inf), jnp.sqrt(var))


def mc_se(log_weights, values, *, g=None, r_eff=None):
    """Return the Monte Carlo standard error of a weighted mean of ``values``.

    Without ``g``: the delta-method error of a self-normalized importance
    estimate for iid draws, ``sqrt(sum w^2 (f - mean)^2)`` with normalized
    weights ``w``. With ``g`` (per-draw probability of a condition, shape
    ``(*batch, S)``, ``values`` then the joint ``h`` of condition and value):
    the ratio estimator ``q = sum w h / sum w g`` has error
    ``sqrt(sum w^2 (h - q g)^2) / sum w g``, and the result is ``inf`` when the
    condition has zero weight (``sum w g = 0``). With ``r_eff=None`` the result
    is the iid error and it understates the error of correlated (MCMC) draws;
    a given ``r_eff`` (effective draws over total draws, scalar or with the
    batch shape) divides the variance by ``r_eff``. The result has shape
    ``(*batch, *event)``; a slice without support gives NaN.
    """
    batch_r = r_eff is not None and jnp.asarray(r_eff).ndim > 0

    def one(lw, v, gg, rr):
        return _mc_se(lw, v, gg, r_eff=rr if r_eff is not None else None)

    r = jnp.asarray(r_eff) if r_eff is not None else jnp.zeros(())
    g_arr = None if g is None else jnp.asarray(g)
    log_weights = jnp.asarray(log_weights)
    fn = one
    nb = log_weights.ndim - 1
    in_axes = (0, 0, None if g is None else 0, 0 if batch_r else None)
    for _ in range(nb):
        fn = jax.vmap(fn, in_axes=in_axes)
    return fn(log_weights, jnp.asarray(values), g_arr, r)
