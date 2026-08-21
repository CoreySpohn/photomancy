"""Corner plots over physical orbital parameters: extract, delegate, decorate.

photomancy supplies what eyepiece cannot know: how a fitted posterior turns
into a physical-sample dict (``sample_physical``), which parameters are
worth showing by default (and that ``ll_total`` never is), the axis-label
vocabulary, and per-parameter axis ranges -- the one feature whose absence
kept a third hand-rolled corner implementation alive. The triangle itself
is ``eyepiece.corner`` / ``eyepiece.corner_overlay``.
"""

from collections.abc import Mapping

import numpy as np

from photomancy.orbit.diagnostics import sample_physical
from photomancy.viz._require import eyepiece
from photomancy.viz.labels import PARAM_LABELS, default_corner_params


def _as_sample_dict(samples):
    """Convert a sample dict's values to NumPy arrays."""
    return {name: np.asarray(value) for name, value in samples.items()}


def _resolve_samples(samples_or_posterior, problem, key, n):
    """Normalize the input to a NumPy physical-sample dict.

    A mapping passes through (the ``sample_physical`` shape, also the door
    for samples loaded from a file); anything else is treated as a fitted
    posterior and needs ``problem`` and ``key`` to draw from.
    """
    if isinstance(samples_or_posterior, Mapping):
        return _as_sample_dict(samples_or_posterior)
    missing = [
        name for name, val in (("problem", problem), ("key", key)) if val is None
    ]
    if missing:
        raise TypeError(
            "plotting a posterior requires "
            + ", ".join(missing)
            + " to draw physical samples"
        )
    return _as_sample_dict(sample_physical(samples_or_posterior, problem, key, n))


def _apply_ranges(axes, params, ranges):
    """Force per-parameter axis limits across a corner grid.

    A range on parameter ``k`` pins the x limits of its column (diagonal
    included) and the y limits of its off-diagonal row, so a grid of corner
    panels across figures stays comparable.
    """
    if not ranges:
        return
    n = len(params)
    for k, name in enumerate(params):
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        for i in range(k, n):
            axes[i, k].set_xlim(lo, hi)
        for j in range(k):
            axes[k, j].set_ylim(lo, hi)


def plot_corner(
    samples_or_posterior,
    problem=None,
    *,
    key=None,
    n=2000,
    params=None,
    truths=None,
    ranges=None,
    bins=40,
    color=None,
    title=None,
    axes=None,
):
    """Corner plot of physical orbital parameters.

    Args:
        samples_or_posterior: A ``{name: (n,)}`` physical-sample dict (the
            ``sample_physical`` shape; also the door for samples loaded
            from a file), or a fitted posterior, which requires ``problem``
            and ``key``.
        problem: The ``OrbitProblem`` supplying ``to_physical``. Posterior
            door only.
        key: PRNG key for drawing posterior samples. Posterior door only.
        n: Number of posterior samples to draw on the posterior door.
        params: Ordered parameter names to show. None uses the curated
            default order over the 1D parameters present, never showing
            ``ll_total``.
        truths: Optional ``{name: value}`` dict drawn as dashed guides.
        ranges: Optional ``{name: (lo, hi)}`` axis limits, applied across
            the grid so corner panels stay comparable between figures.
        bins: Bin count for the histograms.
        color: Histogram/density color override.
        title: Optional figure suptitle (owned-figure only, per eyepiece).
        axes: Optional ``(n, n)`` axes grid to draw into.

    Returns:
        The ``eyepiece.MosaicResult`` from ``corner``.
    """
    ep = eyepiece()
    samples = _resolve_samples(samples_or_posterior, problem, key, n)
    resolved_params = params or default_corner_params(samples)
    result = ep.corner(
        samples,
        resolved_params,
        truths=truths,
        labels=PARAM_LABELS,
        color=color,
        bins=bins,
        title=title,
        axes=axes,
    )
    _apply_ranges(result.axes, resolved_params, ranges)
    return result


def plot_corner_overlay(
    datasets,
    *,
    params=None,
    names=None,
    colors=None,
    ranges=None,
    bins=30,
    axes=None,
):
    """Overlay several physical-sample sets on one corner plot.

    Args:
        datasets: List of ``{name: (n,)}`` physical-sample dicts, one per
            overlay (backend comparisons, before/after an epoch, ...).
        params: Ordered parameter names. None uses the curated default
            order over the first dataset.
        names: Per-dataset legend labels.
        colors: Per-dataset color overrides.
        ranges: Optional ``{name: (lo, hi)}`` axis limits across the grid.
        bins: Bin count for the diagonal histograms.
        axes: Optional ``(n, n)`` axes grid (a ``plot_corner`` result's
            ``axes``) to draw over.

    Returns:
        The ``eyepiece.MosaicResult`` from ``corner_overlay``.
    """
    ep = eyepiece()
    converted = [_as_sample_dict(samples) for samples in datasets]
    resolved_params = params or default_corner_params(converted[0])
    result = ep.corner_overlay(
        converted,
        resolved_params,
        axes=axes,
        colors=colors,
        labels=PARAM_LABELS,
        names=names,
        bins=bins,
    )
    _apply_ranges(result.axes, resolved_params, ranges)
    return result
