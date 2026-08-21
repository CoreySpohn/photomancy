"""Axis-label vocabulary for the physical orbital parameters.

Plain data, importable without the plotting stack: the ASCII label for
every key ``OrbitProblem.to_physical`` emits, plus the curated default
ordering corner plots use when the caller does not choose one.
"""

import numpy as np

PARAM_LABELS = {
    "T": "T (days)",
    "log_P": "log10 P (days)",
    "e": "e",
    "a": "a (AU)",
    "cos_i": "cos i",
    "W": "Omega (rad)",
    "M0": "M0 (rad)",
    "tp": "tp (days)",
    "cos_w": "cos omega",
    "sin_w": "sin omega",
    "Lambda": "Ag Rp^2 (AU^2)",
    "Rp": "Rp (AU)",
    "log_Rp": "log10 Rp (AU)",
    "Ag": "Ag",
    "log_Ag": "log10 Ag",
    "Mp": "Mp (kg)",
    "log_Mp": "log10 Mp (Mearth)",
    "Mp_sini": "Mp sin i (kg)",
    "jitter": "RV jitter",
}

# Corner-plot default: the physically meaningful, non-redundant subset in a
# stable reading order. The log duplicates and the (cos, sin) periapsis pair
# are reachable explicitly via params=; ll_total (a deterministic site
# to_physical leaks) is never shown.
_DEFAULT_CORNER_ORDER = (
    "T",
    "a",
    "e",
    "cos_i",
    "W",
    "M0",
    "tp",
    "Lambda",
    "Rp",
    "Ag",
    "Mp",
    "Mp_sini",
)


def default_corner_params(samples):
    """The default corner-plot parameter list for a physical-sample dict.

    Keeps the curated order, drops anything absent, non-1D (multi-planet
    batches need an explicit ``params``), or excluded (``ll_total``).

    Args:
        samples: Dict ``{name: array}`` of physical samples.

    Returns:
        List of parameter names.
    """
    out = []
    for name in _DEFAULT_CORNER_ORDER:
        if name in samples and np.asarray(samples[name]).ndim == 1:
            out.append(name)
    return out
