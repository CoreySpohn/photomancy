"""Candidate finding: local maxima of a statistic map."""

from typing import NamedTuple

import numpy as np


class Candidates(NamedTuple):
    """Candidate peaks ordered by statistic, largest first.

    Attributes:
        positions_yx: (k, 2) integer pixel positions (y, x), as floats.
        statistic: (k,) map values at the peaks.
        radius_px: (k,) distance of each peak from the star center in pixels.
    """

    positions_yx: np.ndarray
    statistic: np.ndarray
    radius_px: np.ndarray


def find_candidates(
    stat_map,
    center_yx,
    r_min_px: float,
    r_max_px: float,
    threshold: float,
    min_sep_px: float,
    validity=None,
    *,
    neighborhood: int = 2,
) -> Candidates:
    """Local maxima of a statistic map above a threshold inside an annulus.

    A pixel is a candidate when it is the maximum of its (2 * neighborhood + 1)
    square window, exceeds the threshold, lies strictly between r_min_px and
    r_max_px of the center, and is valid and finite. Candidates are then
    thinned greedily in order of decreasing statistic: a peak is dropped when
    it lies within min_sep_px of a stronger accepted peak. Ties in the
    statistic break by row-major pixel order, so the result is deterministic.

    The number of candidates is data dependent, so this function runs on the
    host and is not jit-compatible.

    Args:
        stat_map: (ny, nx) statistic map; NaN marks pixels with no value.
        center_yx: (2,) star center (y, x) in pixels.
        r_min_px: Inner radius of the search annulus in pixels (exclusive).
        r_max_px: Outer radius of the search annulus in pixels (exclusive).
        threshold: Minimum statistic (exclusive).
        min_sep_px: Minimum separation between accepted peaks in pixels; two
            accepted peaks are at least this far apart.
        validity: Optional (ny, nx) mask, nonzero = usable pixel. Invalid
            pixels can neither be candidates nor suppress a neighbor's
            local-maximum status.
        neighborhood: Half width of the local-maximum window in pixels.

    Returns:
        Candidates ordered by statistic, largest first; empty arrays when
        nothing passes.
    """
    stat = np.asarray(stat_map, dtype=float)
    ny, nx = stat.shape
    usable = np.isfinite(stat)
    if validity is not None:
        usable &= np.asarray(validity) > 0
    s0 = np.where(usable, stat, -np.inf)

    k = int(neighborhood)
    padded = np.pad(s0, k, constant_values=-np.inf)
    window_max = np.full_like(s0, -np.inf)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            window_max = np.maximum(window_max, padded[dy : dy + ny, dx : dx + nx])

    yy, xx = np.mgrid[:ny, :nx]
    radius = np.hypot(yy - center_yx[0], xx - center_yx[1])
    peaks = (s0 == window_max) & (s0 > threshold)
    peaks &= (radius > r_min_px) & (radius < r_max_px)

    iy, ix = np.nonzero(peaks)
    values = s0[iy, ix]
    order = np.lexsort((ix, iy, -values))  # statistic descending, then row-major
    iy, ix, values = iy[order], ix[order], values[order]

    keep = []
    for i in range(len(values)):
        if all(np.hypot(iy[i] - iy[j], ix[i] - ix[j]) >= min_sep_px for j in keep):
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    iy, ix, values = iy[keep], ix[keep], values[keep]
    return Candidates(
        positions_yx=np.stack([iy, ix], axis=1).astype(float),
        statistic=values,
        radius_px=radius[iy, ix],
    )
