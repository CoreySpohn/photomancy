"""Statistical functions for masked arrays and small-sample corrections.

Implements JAX-native masked statistics and the Mawet et al. (2014)
small-sample penalty for high-contrast imaging SNR calculations.
"""

import jax.numpy as jnp
from jax.scipy.special import betainc, erfc


def masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Compute the mean of masked values.

    Args:
        values: 1D array of values.
        mask: Boolean mask (True for valid values).

    Returns:
        Mean of valid values, or 0 if no valid values.
    """
    count = jnp.sum(mask)
    masked_sum = jnp.sum(values * mask)
    return masked_sum / jnp.maximum(count, 1.0)


def masked_std(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    mean: float | None = None,
) -> float:
    """Compute the standard deviation of masked values.

    Uses Bessel's correction (N-1 denominator) for unbiased estimation.

    Args:
        values: 1D array of values.
        mask: Boolean mask (True for valid values).
        mean: Pre-computed mean. If None, computed from masked values.

    Returns:
        Standard deviation of valid values.
    """
    if mean is None:
        mean = masked_mean(values, mask)

    count = jnp.sum(mask)
    residuals = (values - mean) * mask
    variance = jnp.sum(residuals**2) / jnp.maximum(count - 1, 1.0)
    return jnp.sqrt(variance)


def small_sample_penalty(n: int | jnp.ndarray) -> float:
    """Compute the Mawet et al. (2014) small-sample statistics correction.

    At small angular separations, fewer reference apertures are available,
    which inflates the noise estimate. This penalty factor accounts for
    the additional uncertainty.

    Reference: Mawet et al. (2014) ApJ
               Equation 9: sigma_corrected = sigma * sqrt(1 + 1/n)

    Args:
        n: Number of reference apertures.

    Returns:
        Correction factor sqrt(1 + 1/n).
    """
    return jnp.sqrt(1 + 1 / jnp.maximum(n, 1.0))


def student_t_sf(t: jnp.ndarray, df: jnp.ndarray) -> jnp.ndarray:
    """Upper-tail probability P(T > t) for the Student-t distribution.

    Computed via the regularized incomplete beta function::

        P(T > t) = 0.5 * I_{df / (df + t^2)}(df / 2, 1 / 2)   for t >= 0

    For t < 0, use 1 minus that. This is the false positive fraction of a
    t-distributed detection statistic (Mawet et al. 2014, two-sample t-test
    with n - 1 degrees of freedom).

    Args:
        t: Test statistic value(s).
        df: Degrees of freedom (positive). Non-positive values yield NaN.

    Returns:
        Elementwise survival probability, NaN where df <= 0 or t is NaN.
    """
    t = jnp.asarray(t, dtype=jnp.result_type(float))
    df = jnp.asarray(df, dtype=jnp.result_type(float))
    safe_df = jnp.maximum(df, 1e-6)
    x = safe_df / (safe_df + t**2)
    tail = 0.5 * betainc(safe_df / 2.0, 0.5, x)
    sf = jnp.where(t >= 0, tail, 1.0 - tail)
    return jnp.where(df > 0, sf, jnp.nan)


def normal_sf(z: jnp.ndarray) -> jnp.ndarray:
    """Upper-tail probability P(Z > z) for the standard normal distribution.

    This is the false positive fraction of a Gaussian-null detection
    statistic (e.g. the annulus sigma test at large sample counts).

    Args:
        z: Statistic value(s).

    Returns:
        Elementwise survival probability, NaN where z is NaN.
    """
    z = jnp.asarray(z, dtype=jnp.result_type(float))
    return 0.5 * erfc(z / jnp.sqrt(2.0))


def nanmasked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean of masked values via NaN exclusion (NaN when mask is empty).

    This is the matched-filter annulus idiom: excluded entries become NaN and
    jnp.nanmean skips them, so an empty mask yields NaN rather than zero
    (callers choose their own fill policy).

    Args:
        values: 1D array of values.
        mask: Boolean mask (True for valid values).

    Returns:
        Mean of valid values, NaN if none are valid.
    """
    return jnp.nanmean(jnp.where(mask, values, jnp.nan))


def nanmasked_population_std(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    mean: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Population (N-denominator) std of masked values via NaN exclusion.

    Matches the matched-filter annulus idiom exactly: the variance is the
    nanmean of squared deviations from ``mean``, with NO Bessel correction,
    and an empty mask yields NaN.

    Args:
        values: 1D array of values.
        mask: Boolean mask (True for valid values).
        mean: Pre-computed mean. If None, computed from the masked values.

    Returns:
        Population standard deviation of valid values, NaN if none are valid.
    """
    if mean is None:
        mean = nanmasked_mean(values, mask)
    variance = jnp.nanmean(jnp.where(mask, (values - mean) ** 2, jnp.nan))
    return jnp.sqrt(variance)


def grubbs_fpf(g: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """One-sided Grubbs (extreme studentized deviate) p-value.

    For a pooled sample of size n with statistic
    G = (max(x) - mean(x)) / std(x), the one-sided p-value is the Bonferroni
    bound P <= n * P(T > t) with T Student-t distributed on n - 2 degrees of
    freedom and t**2 = n (n-2) G**2 / ((n-1)**2 - n G**2) (Grubbs 1950).
    G values at or beyond the attainable maximum (n-1)/sqrt(n) give 0;
    negative G (maximum below the mean) gives 1; n < 3 gives NaN.

    Args:
        g: Grubbs statistic value(s).
        n: Pooled sample size(s), including the candidate.

    Returns:
        Elementwise false positive fraction in [0, 1], NaN where invalid.
    """
    g = jnp.asarray(g, dtype=jnp.result_type(float))
    n = jnp.asarray(n, dtype=jnp.result_type(float))
    denom = (n - 1.0) ** 2 - n * g**2
    t_sq = n * (n - 2.0) * g**2 / jnp.maximum(denom, 1e-10)
    t = jnp.sqrt(jnp.maximum(t_sq, 0.0))
    p = jnp.minimum(n * student_t_sf(t, n - 2.0), 1.0)
    p = jnp.where(g < (n - 1.0) / jnp.sqrt(n), p, 0.0)
    p = jnp.where(g >= 0, p, 1.0)
    p = jnp.where(n >= 3, p, jnp.nan)
    return jnp.where(jnp.isnan(g), jnp.nan, p)
