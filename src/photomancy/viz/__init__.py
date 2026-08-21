"""Plotting for photomancy types, built on eyepiece primitives.

Requires the ``viz`` extra (``pip install 'photomancy[viz]'``), which
brings eyepiece and, through it, matplotlib and hwostyle. The base install
stays free of all three: plot functions are re-exported lazily (PEP 562)
and the eyepiece requirement is checked when one is first touched.
``PARAM_LABELS`` and ``default_corner_params`` are plain data/logic and
import without the plotting stack.
"""

import importlib

_LAZY = {
    "plot_corner": "photomancy.viz.corner",
    "plot_corner_overlay": "photomancy.viz.corner",
    "plot_detectability": "photomancy.viz.predictive",
    "plot_dmag": "photomancy.viz.predictive",
    "plot_eig": "photomancy.viz.eig",
    "plot_rv": "photomancy.viz.predictive",
}

_DATA = {
    "PARAM_LABELS": "photomancy.viz.labels",
    "default_corner_params": "photomancy.viz.labels",
}

__all__ = sorted([*_LAZY, *_DATA])


def __getattr__(name):
    """Resolve a lazy re-export; plot functions check eyepiece first.

    Args:
        name: Attribute being looked up on ``photomancy.viz``.

    Returns:
        The requested function or data object.

    Raises:
        AttributeError: If ``name`` is not one of the lazy re-exports.
    """
    if name in _DATA:
        return getattr(importlib.import_module(_DATA[name]), name)
    if name in _LAZY:
        from photomancy.viz import _require

        _require.eyepiece()
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'photomancy.viz' has no attribute {name!r}")


def __dir__():
    """List the lazy re-exports alongside the module's real attributes.

    Returns:
        Sorted attribute names, including the lazily provided ones.
    """
    return sorted(set(globals()) | set(__all__))
