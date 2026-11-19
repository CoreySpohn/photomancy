"""Injection-recovery statistics and the contrast curve they imply."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class ThroughputProfile(NamedTuple):
    """Throughput statistics per separation group.

    Attributes:
        separation: (m,) group values: the distinct separations, or the bin
            centers when bin edges are given.
        mean: (m,) mean throughput; NaN for an empty group.
        std: (m,) sample standard deviation (ddof = 1); NaN below 2 records.
        count: (m,) number of records in the group.
    """

    separation: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


def throughput_profile(
    separation, injected, recovered, edges=None
) -> ThroughputProfile:
    """Throughput mean, standard deviation, and count per separation.

    Each record is one injected source: its separation, the amplitude
    injected, and the amplitude recovered by the fit or filter. The record's
    throughput is recovered / injected. The caller supplies the injected
    source (a template placed at the separation); nothing here shifts or
    resamples a stamp.

    Args:
        separation: (n,) separation of each record.
        injected: (n,) injected amplitude, nonzero.
        recovered: (n,) recovered amplitude.
        edges: Optional (m + 1,) increasing bin edges. Bins are half open,
            [edges[i], edges[i + 1]), the last one closed on the right;
            records outside the edges are ignored. When None, records are
            grouped by exactly equal separation.

    Returns:
        ThroughputProfile ordered by increasing separation.
    """
    sep = np.asarray(separation, dtype=float)
    ratio = np.asarray(recovered, dtype=float) / np.asarray(injected, dtype=float)
    if edges is None:
        values, group = np.unique(sep, return_inverse=True)
        n_groups = len(values)
    else:
        edges = np.asarray(edges, dtype=float)
        n_groups = len(edges) - 1
        values = 0.5 * (edges[:-1] + edges[1:])
        group = np.searchsorted(edges, sep, side="right") - 1
        group = np.where(sep == edges[-1], n_groups - 1, group)
        group = np.where((group < 0) | (group >= n_groups), -1, group)
    mean = np.full(n_groups, np.nan)
    std = np.full(n_groups, np.nan)
    count = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        r = ratio[group == g]
        count[g] = len(r)
        if len(r) >= 1:
            mean[g] = r.mean()
        if len(r) >= 2:
            std[g] = r.std(ddof=1)
    return ThroughputProfile(values, mean, std, count)


def contrast_curve(threshold, noise, throughput) -> jnp.ndarray:
    """Detection-limit contrast curve: threshold * noise / throughput.

    Args:
        threshold: Detection threshold in units of the noise (for example 5).
        noise: Noise profile versus separation, in contrast units.
        throughput: Throughput profile on the same separations, fractions of
            the injected signal recovered.

    Returns:
        The contrast at which a source is recovered at the threshold; inf
        where the throughput is not positive.
    """
    noise = jnp.asarray(noise)
    throughput = jnp.asarray(throughput)
    positive = throughput > 0
    safe = jnp.where(positive, throughput, 1.0)
    return jnp.where(positive, threshold * noise / safe, jnp.inf)
