"""Load target metadata and non-image measurements from TOML."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from hwoutils.epoch import Epoch
from hwoutils.model_paths import join
from optixstuff.spectral import Band, Quadrature, Resolution

from photomancy.measurements.astrometry import (
    AstrometryData,
    RelativeAstrometryTerm,
)
from photomancy.measurements.photometry import PhotometryData, PhotometryTerm

__all__ = ["TargetData", "load_target"]


@dataclass(frozen=True)
class TargetData:
    """Target metadata, converted observations, and unused raw rows.

    ``band_names`` lists filters in first-used order. ``comparisons`` maps
    table names to raw rows: comparison rows, skipped pending rows, and pixels.
    A transmission mapping supplied at load time is the default for ``terms``.
    Observation containers retain ``time_d``, in days from ``epoch``, the same
    origin required by the system used to evaluate their terms.

    Attributes:
        target: Target metadata table from the file.
        relative_astrometry: Relative astrometry rows, or None when there are none.
        photometry: Photometry rows, or None when there are none.
        band_names: Filter names in first-used order.
        comparisons: Raw comparison, pending and pixel rows by table name.
        epoch: Origin of the observation times.
    """

    target: dict
    relative_astrometry: AstrometryData | None
    photometry: PhotometryData | None
    band_names: tuple[str, ...]
    comparisons: dict
    epoch: Epoch
    _transmissions: Mapping | None = field(default=None, repr=False, compare=False)

    def terms(self, system_path="", transmissions=None, *, system=None):
        """Build terms for present data, using filter-name to transmission mappings.

        ``system_path`` is the model path of the system the terms read; the empty
        default is the model itself.
        An explicit mapping overrides the mapping supplied to ``load_target``.
        Photometric terms require the concrete reference system. Each band spans
        the nonzero extent of its filter curve and is split at the curve's knots;
        the system's tabulated star and planet spectra are integrated exactly
        over each piece, so they need no knots of their own as breaks and must
        cover every band. The photometric term declares the system's knot
        leaves (``Scene.knot_leaves``, under ``system_path``) as its reads, so a
        fit refuses to free a knot grid its operators were built on.
        Missing photometric filters raise ``ValueError`` naming those filters.

        Args:
            system_path: Model path of the system the terms read.
            transmissions: Mapping from filter name to transmission. Defaults to the
                mapping given to ``load_target``.
            system: Concrete reference system, required when photometry is present.

        Returns:
            A tuple of the terms for the data that are present.
        """
        result = []
        if self.relative_astrometry is not None:
            result.append(RelativeAstrometryTerm(self.relative_astrometry, system_path))
        if self.photometry is not None:
            mapping = self._transmissions if transmissions is None else transmissions
            missing = [
                name
                for name in self.band_names
                if mapping is None or name not in mapping
            ]
            if missing:
                raise ValueError(
                    f"Missing transmissions for filters: {', '.join(missing)}"
                )
            if system is None:
                raise ValueError(
                    "Photometric terms need a system to build the quadrature"
                )
            curves = tuple(mapping[name] for name in self.band_names)
            bands = tuple(_nonzero_band(curve) for curve in curves)
            rule = Quadrature.build(bands, *curves)
            resolution = _resolve(system, rule)
            reads = tuple(join(system_path, path) for path in system.knot_leaves())
            result.append(
                PhotometryTerm(
                    self.photometry,
                    rule,
                    curves,
                    system_path,
                    resolution=resolution,
                    reads=reads,
                )
            )
        return tuple(result)


def _nonzero_band(curve):
    """Return the band a filter curve responds over, trimmed to its nonzero extent.

    The response is linear between knots, so it is nonzero strictly between the
    knot before its first positive value and the knot after its last. Zero tails
    beyond that carry no signal and would only leave a tabulated source's range.
    An all-zero curve keeps its full range.
    """
    wave = np.asarray(curve.wavelength_nm, dtype=float)
    positive = np.flatnonzero(np.asarray(curve.transmission) > 0.0)
    if positive.size == 0:
        return Band(float(wave[0]), float(wave[-1]))
    first, last = positive[0], positive[-1]
    low = wave[first - 1] if first > 0 else wave[first]
    high = wave[last + 1] if last < wave.size - 1 else wave[last]
    return Band(float(low), float(high))


def _resolve(system, quadrature):
    """Build the Resolution of the system's tabulated spectra on ``quadrature``.

    A disk calibrated under the bolometric luminosity rule averages a tabulated
    star over its own band, so each such band gets a quadrature resolved for the
    star's table only.
    """
    tables = system.spectral_tables()
    resolution = Resolution.build((quadrature,), *tables)
    star = system.star
    if not any(table is star for table in tables):
        return resolution
    quadratures = list(resolution.quadratures)
    keys, rules = list(resolution.keys), list(resolution.rules)
    for band in system.calibration_bands():
        extra = Quadrature.build((band,))
        added = Resolution.build((extra,), star)
        quadratures.append(extra)
        for key, rule in zip(added.keys, added.rules, strict=True):
            if key not in keys:
                keys.append(key)
                rules.append(rule)
    return Resolution(tuple(quadratures), tuple(keys), tuple(rules))


def _mean_error(value):
    """Reduce a scalar or a [minus, plus] error to one standard deviation."""
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("asymmetric errors must be [minus, plus]")
        return (value[0] + value[1]) / 2.0
    return value


def load_target(
    path, *, epoch: Epoch, pixel_data_ids=(), include_pending=False, transmissions=None
):
    """Read target metadata and astrometry/photometry rows from a TOML file.

    Asymmetric errors ``[minus, plus]`` become their arithmetic mean.
    Separations and errors in mas are converted to arcsec. Julian dates are
    converted to days from the required ``epoch`` in NumPy float64 on the host,
    before building data containers. Photometry becomes linear contrast.
    Comparison rows and pending rows excluded by ``include_pending=False`` are
    retained in ``comparisons`` by table name. Pixel rows are also retained;
    unknown top-level tables are ignored. Only the explicit ``pixel_data_ids``
    identifies image data used in a fit: listing pixels alone does not use them.

    A term row sharing a ``data_id`` with ``pixel_data_ids`` raises ``ValueError``
    to prevent counting the same measurement twice. Pending rows become term
    rows when requested; any unknown ``use`` value raises. ``transmissions`` may
    supply the default filter mapping for the returned object's ``terms`` method.

    Args:
        path: Path of the TOML file.
        epoch: Origin of the observation times.
        pixel_data_ids: Identifiers of the data used by image terms.
        include_pending: Whether pending rows become term rows.
        transmissions: Default mapping from filter name to transmission for ``terms``.

    Returns:
        The target metadata and converted observations.
    """
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    pixel_ids = set(pixel_data_ids)
    selected = {"astrometry": [], "photometry": []}
    comparisons = {
        "pixels": document.get("pixels", []),
        "astrometry": [],
        "photometry": [],
    }
    for table in selected:
        for index, row in enumerate(document.get(table, [])):
            use = row.get("use")
            label = f"{table}[{index}] (data_id={row.get('data_id')!r})"
            if use not in ("term", "comparison", "pending"):
                raise ValueError(f"{label}: unknown use {use!r}")
            if use == "comparison" or (use == "pending" and not include_pending):
                comparisons[table].append(row)
                continue
            if row["data_id"] in pixel_ids:
                raise ValueError(f"{label}: data already used by a pixel term")
            selected[table].append(row)

    rows = selected["astrometry"]
    astrometry = None
    if rows:
        astrometry = AstrometryData.from_separation_pa(
            time_d=epoch.days(jd=np.asarray([r["jd"] for r in rows], dtype=np.float64)),
            separation_arcsec=[r["sep_mas"] / 1000.0 for r in rows],
            separation_err_arcsec=[
                _mean_error(r["sep_err_mas"]) / 1000.0 for r in rows
            ],
            pa_deg=[r["pa_deg"] for r in rows],
            pa_err_deg=[_mean_error(r["pa_err_deg"]) for r in rows],
            planet=[r.get("planet", 0) for r in rows],
        )
    rows = selected["photometry"]
    band_names = tuple(dict.fromkeys(r["filter"] for r in rows))
    photometry = None
    if rows:
        photometry = PhotometryData.from_dmag(
            time_d=epoch.days(jd=np.asarray([r["jd"] for r in rows], dtype=np.float64)),
            dmag=[r["dmag"] for r in rows],
            dmag_err=[_mean_error(r["dmag_err"]) for r in rows],
            band_id=[band_names.index(r["filter"]) for r in rows],
            planet=[r.get("planet", 0) for r in rows],
            is_upper_limit=[r.get("upper_limit", False) for r in rows],
        )
    return TargetData(
        document.get("target", {}),
        astrometry,
        photometry,
        band_names,
        comparisons,
        epoch,
        transmissions,
    )
