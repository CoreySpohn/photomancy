"""Convert saved orbit samples between the earlier and current sky conventions.

orbix 2.x and photomancy before 0.4.0 evaluated the sky position with the east and
north offsets swapped (the first output was the north offset). An astrometry-only
fit made then carries mirrored elements. With the node measured north through east
and the classical Thiele-Innes relations (Green 1985, ch. 17), the old model
returned ``(N, E)`` where the current one returns ``(E, N)``, with

    N = r [cos(W) cos(u) - sin(W) cos(i) sin(u)],
    E = r [sin(W) cos(u) + cos(W) cos(i) sin(u)],    u = omega + f.

Setting ``W' = pi/2 - W`` and ``cos_i' = -cos_i`` gives ``E(W', i') = N(W, i)`` and
``N(W', i') = E(W, i)``, so the current model at the migrated elements reproduces
the old model's sky track. The semimajor axis, eccentricity, argument of
periapsis, period and epoch terms are unchanged.
"""

from collections.abc import Mapping

import numpy as np

# Names that only occur in transformed sampling coordinates: the unconstrained
# NumPyro sites (``log_P``, ``M0``, ``e_raw``, ``w_raw``), unit-cube draws
# (``*_base``) and the Thiele-Innes coordinates of the few-epoch fitters.
_SAMPLING_NAMES = frozenset(
    {
        "log_P",
        "M0",
        "e_raw",
        "w_raw",
        "lnP",
        "tau",
        "A",
        "F",
        "B",
        "G",
        "G_ti",
        "east_X",
        "east_Y",
        "north_X",
        "north_Y",
    }
)

_TWO_PI = 2.0 * np.pi


def migrate_sky_convention(samples):
    """Convert physical-element samples fitted under the earlier sky convention.

    Applies ``cos_i -> -cos_i`` and ``W -> (pi/2 - W) mod 2 pi`` to samples of an
    astrometry-only fit made with orbix 2.x or photomancy before 0.4.0. Every other
    entry (``a``, ``e``, ``T``, ``cos_w``, ``sin_w``, ``tp``, ...) is returned
    unchanged. Applying it twice returns the input.

    Only physical elements can be converted. Posteriors stored in transformed
    sampling coordinates (the unconstrained NumPyro sites ``log_P``, ``M0``,
    ``e_raw``, ``w_raw``, unit-cube ``*_base`` draws, or the Thiele-Innes
    coordinates of the few-epoch fitters) raise ``ValueError``: they must be
    refit, or first mapped to physical elements with the ``to_physical`` map of the
    problem that produced them (which needs the old release). Fits that used
    photometry also need a refit, because the earlier phase angle was inverted.

    Args:
        samples: Mapping with at least ``cos_i`` and ``W`` (radians), each scalar
            or array, keyed as ``sample_physical`` emits them.

    Returns:
        A new dict with the same keys; ``cos_i`` and ``W`` are converted.

    Raises:
        TypeError: If ``samples`` is not a mapping.
        ValueError: If ``cos_i`` or ``W`` is missing, a key marks transformed
            sampling coordinates, or ``cos_i`` or ``W`` lies outside its physical
            range (a sign of unconstrained coordinates).
    """
    if not isinstance(samples, Mapping):
        raise TypeError(
            "samples must be a mapping of physical orbital elements, got "
            f"{type(samples).__name__}"
        )
    sampling = sorted(
        k for k in samples if k in _SAMPLING_NAMES or str(k).endswith("_base")
    )
    if sampling:
        raise ValueError(
            f"samples include transformed sampling coordinates {sampling}; only "
            "physical elements (cos_i, W, ...) can be converted. Refit the "
            "posterior with the current release."
        )
    missing = [k for k in ("cos_i", "W") if k not in samples]
    if missing:
        raise ValueError(f"samples are missing the physical elements {missing}")

    cos_i, W = (
        v if hasattr(v, "dtype") else np.asarray(v, dtype=float)
        for v in (samples["cos_i"], samples["W"])
    )
    if np.any(np.abs(np.asarray(cos_i)) > 1.0) or np.any(
        (np.asarray(W) < -1e-9) | (np.asarray(W) > _TWO_PI + 1e-9)
    ):
        raise ValueError(
            "cos_i must lie in [-1, 1] and W in [0, 2 pi]; these samples look like "
            "unconstrained sampling coordinates and need a refit"
        )
    out = dict(samples)
    out["cos_i"] = -cos_i
    out["W"] = (np.pi / 2.0 - W) % _TWO_PI
    return out
