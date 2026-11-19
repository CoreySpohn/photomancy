"""Null-set bookkeeping: independent trials and false-alarm counts."""

import math

import numpy as np


def n_resolution_elements(r_min: float, r_max: float, fwhm: float) -> int:
    """Count independent resolution elements in an annulus.

    The annulus from r_min to r_max is cut into rings one fwhm wide, starting
    at r_min; ring k is centered at r_min + (k + 0.5) * fwhm and holds
    floor(2 pi r / fwhm) elements at its center radius. Only whole rings
    count: a remainder narrower than one fwhm at the outer edge is dropped.
    The total is the sum over rings. Residuals with correlated structure
    have fewer independent elements than this count.

    Args:
        r_min: Inner radius (any length unit, the same as fwhm).
        r_max: Outer radius.
        fwhm: Resolution element width (full width at half maximum).

    Returns:
        Total number of resolution elements (0 when no whole ring fits).
    """
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    if r_max < r_min:
        raise ValueError("r_max must not be smaller than r_min")
    n_rings = math.floor((r_max - r_min) / fwhm * (1.0 + 1e-12))
    return sum(
        math.floor(2.0 * math.pi * (r_min + (k + 0.5) * fwhm) / fwhm)
        for k in range(n_rings)
    )


def _binomial_cdf(n: int, p: float) -> np.ndarray:
    """Exact binomial CDF at 0..n, accumulated in log space."""
    k = np.arange(n + 1)
    lgamma = np.vectorize(math.lgamma, otypes=[float])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = (
            math.lgamma(n + 1)
            - lgamma(k + 1.0)
            - lgamma(n - k + 1.0)
            + k * math.log(p)
            + (n - k) * math.log1p(-p)
        )
    return np.minimum(np.cumsum(np.exp(log_pmf)), 1.0)


def false_alarm_band(n: int, fpf: float, level: float = 0.99) -> tuple[int, int]:
    """Central binomial band for a false-alarm count.

    If a statistic is calibrated, each of n independent trials exceeds the
    threshold with probability fpf, so the count of exceedances is
    Binomial(n, fpf). The band is [lo, hi] with lo the smallest count whose
    CDF reaches (1 - level) / 2 and hi the smallest count whose CDF reaches
    1 - (1 - level) / 2: each tail outside the band holds at most
    (1 - level) / 2 of the probability. Quantiles are exact (no normal
    approximation).

    Args:
        n: Number of independent trials.
        fpf: Per-trial false positive fraction, in [0, 1].
        level: Central coverage of the band, in (0, 1).

    Returns:
        Tuple (lo, hi) of counts.
    """
    if not 0.0 <= fpf <= 1.0:
        raise ValueError("fpf must lie in [0, 1]")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    if n < 0:
        raise ValueError("n must be non-negative")
    if fpf == 0.0:
        return 0, 0
    if fpf == 1.0:
        return n, n
    cdf = _binomial_cdf(int(n), float(fpf))
    tail = (1.0 - level) / 2.0
    lo = int(np.searchsorted(cdf, tail, side="left"))
    hi = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
    return lo, min(hi, int(n))
