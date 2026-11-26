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
* Posterior weights and their summaries require float64.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf
from hwoutils.precision import float_dtype, require_float64
from jax.scipy.special import logsumexp

__all__ = [
    "khat_threshold",
    "kish_ess",
    "kish_temperature",
    "mc_se",
    "normalize_log_weights",
    "pareto_khat",
    "ratio_mc_se",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
]

_KHAT_PRIOR_COUNT = 10.0
_MIN_EXCEEDANCES = 5
# Level of the upper confidence bound that the largest exceedance puts on a
# Pareto tail index: the bound is below the true index with this probability.
_MAX_BOUND_LEVEL = 1e-3
# Two log weights closer than this many float epsilons (relative, floored at
# one) are tied: values that went through logsumexp or a subtraction differ by
# rounding even when they are mathematically equal.
_TIE_ULPS = 64


def _tied(diff, value):
    """Whether a nonnegative log-weight difference is zero within rounding."""
    eps = jnp.finfo(diff.dtype).eps
    return diff <= _TIE_ULPS * eps * jnp.maximum(1.0, jnp.abs(value))


def _over_batch(fn, log_weights, *values, center_host=False, **kwargs):
    """Map ``fn(log_weights_slice, *values_slice, **kwargs)`` over the batch prefix."""
    require_float64("posterior weight diagnostics")
    if center_host and isinstance(log_weights, np.ndarray) and log_weights.ndim:
        log_weights, _, _ = _shift_log_weights(log_weights)
    log_weights = jnp.asarray(log_weights, dtype=float_dtype())
    if log_weights.ndim < 1:
        raise ValueError("log_weights needs a draw axis")
    mapped = functools.partial(fn, **kwargs)
    for _ in range(log_weights.ndim - 1):
        mapped = jax.vmap(mapped)
    return mapped(log_weights, *(jnp.asarray(v) for v in values))


def _shift_log_weights(log_weights):
    # Host deliveries can resolve contrasts smaller than a device ULP at the
    # absolute weight. Difference on the host before transferring the offsets.
    host = isinstance(log_weights, np.ndarray)
    xp = np if host else jnp
    if host:
        log_weights = np.asarray(log_weights, dtype=np.float64)
    lw = xp.where(xp.isnan(log_weights), -xp.inf, log_weights)
    peak = xp.max(lw, axis=-1)
    no_support = ~xp.isfinite(peak)
    reference = xp.where(no_support, 0.0, peak)
    if host:
        # A negative overflow is log zero: the weight is far below exp's range.
        with np.errstate(over="ignore"):
            contrast = lw - reference[..., None]
        shifted = as_float_leaf(np.where(no_support[..., None], -np.inf, contrast))
    else:
        shifted = jnp.where(no_support[..., None], -jnp.inf, lw - reference[..., None])
    return shifted, peak, no_support


def _normalize(log_weights):
    shifted, peak, no_support = _shift_log_weights(log_weights)
    log_sum = logsumexp(jnp.where(no_support[..., None], 0.0, shifted), axis=-1)
    return (
        shifted - log_sum[..., None],
        jnp.where(no_support, -jnp.inf, peak + log_sum),
        jnp.asarray(no_support),
    )


def normalize_log_weights(log_weights):
    """Return ``(normalized, log_total, no_support)`` for log weights.

    Normalize relative to the largest log weight to retain precision at large
    common offsets. NaNs count as zero weight. A nonfinite maximum gives
    ``no_support=True``, all ``-inf`` normalized weights and a total of ``-inf``.
    Leading axes are batch axes; the draw axis is last.

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.

    Returns:
        ``normalized`` are the log weights with the log total removed, shape
        ``(*batch, S)``; ``log_total`` is the log of the summed weights and
        ``no_support`` is the boolean flag, both with shape ``(*batch,)``.
    """
    require_float64("posterior weight diagnostics")
    if not isinstance(log_weights, np.ndarray):
        log_weights = jnp.asarray(log_weights, dtype=float_dtype())
    if log_weights.ndim < 1:
        raise ValueError("log_weights needs a draw axis")
    return _normalize(log_weights)


def _kish(log_weights):
    shifted, _, no_support = _shift_log_weights(log_weights)
    weights = jnp.exp(shifted)
    sum_squares = jnp.sum(weights**2)
    return jnp.sum(weights) ** 2 / jnp.where(no_support, 1.0, sum_squares)


def kish_ess(log_weights):
    """Return the Kish effective sample size, or zero without support.

    Leading axes are batch axes; the draw axis is last.

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.

    Returns:
        Kish effective sample size in draws, shape ``(*batch,)``.
    """
    return _over_batch(_kish, log_weights)


def khat_threshold(n_draws):
    """Return the usable-tail limit ``min(1 - 1/log10(S), 0.7)`` for ``S`` draws.

    A ``pareto_khat`` at or below this value is reliable for the importance
    ratios of ``S`` iid proposal draws; above it the variance of the weighted
    mean is not trustworthy. ``S`` must exceed 10 (the formula is negative or
    undefined below that).

    Args:
        n_draws: Number of proposal draws ``S``.

    Returns:
        The largest usable k-hat for ``n_draws`` draws, as a float.
    """
    return min(1.0 - 1.0 / math.log10(n_draws), 0.7)


def _tail_size(n):
    """Tail size ``ceil(min(n/5, 3 sqrt(n)))`` for a float or integer count."""
    return jnp.ceil(jnp.minimum(n / 5.0, 3.0 * jnp.sqrt(n)))


def _khat(log_weights):
    """PSIS tail shape of one slice (Zhang and Stephens 2009, Vehtari et al. 2024).

    Capped by the Pareto bound of the largest exceedance.
    """
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

    # A Pareto tail of index k puts the largest of count exceedances at or
    # below r times the cutoff with probability (1 - r**(-1/k))**count, so the
    # observed maximum bounds the index from above.
    level = _MAX_BOUND_LEVEL ** (1.0 / count)
    bound = (top - cutoff_lw) / -jnp.log1p(-level)
    shape = jnp.minimum(shape, bound)

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

    The fitted shape is capped by the bound that the largest ratio puts on a
    Pareto tail: with ``n`` distinct exceedances, cutoff ``u`` and largest ratio
    ``w_max``, a tail of index ``k`` gives ``w_max / u <= r`` with probability
    ``(1 - r**(-1/k))**n``, so ``log(w_max / u) / -log(1 - 0.001**(1/n))`` is a
    99.9 percent upper bound on the index. The cap matters when the cutoff
    falls inside a cluster of nearly equal ratios: the exceedances then span
    many decades and the fit alone reads a heavy tail on bounded weights. The
    cap tests a Pareto tail in the ratios themselves; a heavy component whose
    scale is far below the cutoff can pass, and carries a correspondingly small
    share of the total weight.

    Valid for the raw ratios of iid proposal draws (an importance reweight or a
    Laplace-IS cloud), not for nested-sampling dead points or a grid. A slice
    with no finite entry, or fewer than five distinct exceedances, returns NaN
    (the fit fails closed); flat weights return ``-inf``. Gate on
    ``~(khat <= limit)`` so that NaN refuses and ``-inf`` passes. Leading axes
    are batch axes.

    Args:
        log_weights: Unnormalized log importance ratios, shape ``(*batch, S)``.

    Returns:
        Pareto tail shape k-hat, shape ``(*batch,)``.
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

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.
        values: Draws, shape ``(*batch, S, *event)``.

    Returns:
        Weighted mean, shape ``(*batch, *event)``; NaN for a slice without support.
    """
    return _over_batch(_mean, log_weights, values, center_host=True)


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

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.
        values: Draws, shape ``(*batch, S, d)`` or ``(*batch, S)`` for scalars.

    Returns:
        Weighted covariance, shape ``(*batch, d, d)``, or the variance
        ``(*batch,)`` for scalar draws; NaN for a slice without support.
    """
    return _over_batch(_cov, log_weights, values, center_host=True)


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

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.
        values: Draws, shape ``(*batch, S, *event)``.
        q: Quantile level or levels in ``[0, 1]``, shared by every slice.

    Returns:
        Weighted quantiles, shape ``(*batch, *q.shape, *event)``; NaN for a slice
        without support.
    """
    return _over_batch(_quantile, log_weights, values, q=q)


def _scale_by_r_eff(var, r_eff):
    return var if r_eff is None else var / r_eff


def _mc_se(log_weights, values, r_eff=None):
    weights, no_support, values = _prepare(log_weights, values)
    expanded = _event_shape_mask(weights, values)
    mean = _mean(log_weights, values)
    centered = jnp.where(expanded > 0, values - mean, 0.0)
    var = _scale_by_r_eff(jnp.sum((expanded * centered) ** 2, axis=0), r_eff)
    return jnp.where(no_support, jnp.nan, jnp.sqrt(var))


def _ratio_mc_se(log_weights, numerator, denominator, r_eff=None):
    weights, no_support, numerator = _prepare(log_weights, numerator)
    expanded = _event_shape_mask(weights, numerator)
    g = jnp.broadcast_to(jnp.asarray(denominator, dtype=weights.dtype), weights.shape)
    den = weights @ g
    bad = no_support | ~(den > 0)
    safe = jnp.where(bad, 1.0, den)
    q = jnp.sum(jnp.where(expanded > 0, expanded * numerator, 0.0), axis=0) / safe
    gx = g.reshape(g.shape + (1,) * (numerator.ndim - 1))
    resid = jnp.where(expanded > 0, numerator - q * gx, 0.0)
    var = jnp.sum((expanded * resid) ** 2, axis=0) / safe**2
    var = _scale_by_r_eff(var, r_eff)
    return jnp.where(bad, jnp.where(no_support, jnp.nan, jnp.inf), jnp.sqrt(var))


def _over_batch_r_eff(fn, log_weights, *values, r_eff):
    """``_over_batch`` with an optional ``r_eff`` that is scalar or batch shaped."""
    if r_eff is None:
        return _over_batch(fn, log_weights, *values)
    r_eff = jnp.broadcast_to(jnp.asarray(r_eff), np.shape(log_weights)[:-1])
    return _over_batch(fn, log_weights, *values, r_eff)


def mc_se(log_weights, values, *, r_eff=None):
    """Return the Monte Carlo standard error of a weighted mean of ``values``.

    The delta-method error of a self-normalized importance estimate for iid
    draws, ``sqrt(sum w^2 (f - mean)^2)`` with normalized weights ``w``. With
    ``r_eff=None`` the result is the iid error and it understates the error of
    correlated (MCMC) draws; a given ``r_eff`` (effective draws over total
    draws, scalar or with the batch shape) divides the variance by ``r_eff``.
    The result has shape ``(*batch, *event)``; a slice without support gives NaN.

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.
        values: Draws, shape ``(*batch, S, *event)``.
        r_eff: Effective draws over total draws, scalar or with the batch shape, or
            ``None`` for iid draws.

    Returns:
        Standard error of the weighted mean, shape ``(*batch, *event)``.
    """
    return _over_batch_r_eff(_mc_se, log_weights, values, r_eff=r_eff)


def ratio_mc_se(log_weights, numerator, denominator, *, r_eff=None):
    """Return the Monte Carlo standard error of a weighted ratio estimate.

    For a conditional mean ``q = sum w h / sum w g``, with ``g`` the per-draw
    probability of a condition (shape ``(*batch, S)``) and ``h`` the joint of
    condition and value (``numerator``, shape ``(*batch, S, *event)``), the
    error is ``sqrt(sum w^2 (h - q g)^2) / sum w g``. The result is ``inf`` when
    the condition has zero weight (``sum w g = 0``) and NaN for a slice without
    support. ``r_eff`` is as in :func:`mc_se`.

    Args:
        log_weights: Unnormalized log weights, shape ``(*batch, S)``.
        numerator: Joint of condition and value ``h``, shape ``(*batch, S, *event)``.
        denominator: Per-draw probability of the condition ``g``, shape
            ``(*batch, S)``.
        r_eff: Effective draws over total draws, scalar or with the batch shape, or
            ``None`` for iid draws.

    Returns:
        Standard error of the ratio estimate, shape ``(*batch, *event)``.
    """
    denominator = jnp.broadcast_to(jnp.asarray(denominator), np.shape(log_weights))
    return _over_batch_r_eff(
        _ratio_mc_se, log_weights, numerator, denominator, r_eff=r_eff
    )


def kish_temperature(
    log_base, log_likelihood, beta, target_fraction, min_size=0.0, n_bisect=60
):
    """Return the next temperature, by the Kish size of the cloud the next fit sees.

    The cloud was drawn from a proposal and the tempered target is ``prior *
    likelihood ** b``, so a draw has the log weight ``log_base + b * log_likelihood``
    with ``log_base = log prior - log q``. The Kish size of that cloud at ``beta`` is
    ``n_now``; the result is the largest ``b`` in ``[beta, 1]`` at which the size is
    still at least ``clip(target_fraction * n_now, min_size, n_now)``, found by
    bisection (the size falls as ``b`` grows). It is ``1`` when the full step keeps
    that size, and ``beta`` when no step does. The criterion is on the cloud the next
    proposal is fitted from (the weights of the draws at the new temperature), not on
    the move from the previous temperature, and the floor keeps enough draws for the
    fit. One slice (draw axis last); a ``-inf`` base or likelihood is a draw with zero
    weight, ``0 * -inf`` at ``b = 0`` is zero, and NaN propagates to a NaN temperature.

    Args:
        log_base: Log prior density minus log proposal density of each draw ``(N,)``.
        log_likelihood: Log likelihood of each draw ``(N,)``.
        beta: Current temperature in ``[0, 1]``.
        target_fraction: Fraction of the current Kish size that the step may keep, in
            ``(0, 1]``.
        min_size: Smallest Kish size, in draws, that a step may leave.
        n_bisect: Bisection steps (a static count).

    Returns:
        The new temperature in ``[beta, 1]``.
    """
    require_float64("Kish temperature")
    log_base = jnp.asarray(log_base, dtype=float_dtype())
    log_likelihood = jnp.asarray(log_likelihood, dtype=float_dtype())

    def size(b):
        scaled = jnp.where(b == 0.0, 0.0, b * log_likelihood)
        return _kish(jnp.where(jnp.isneginf(log_base), -jnp.inf, log_base + scaled))

    n_now = size(beta)
    target = jnp.clip(target_fraction * n_now, min_size, n_now)

    def body(_, bounds):
        low, high = bounds
        mid = 0.5 * (low + high)
        ok = size(mid) >= target
        return jnp.where(ok, mid, low), jnp.where(ok, high, mid)

    low, _ = jax.lax.fori_loop(0, n_bisect, body, (beta, jnp.ones(())))
    new = jnp.where(size(jnp.ones(())) >= target, 1.0, low)
    nan = jnp.any(jnp.isnan(log_base)) | jnp.any(jnp.isnan(log_likelihood))
    return jnp.where(nan, jnp.nan, jnp.where(beta >= 1.0, beta, new))
