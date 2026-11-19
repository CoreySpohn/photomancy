"""Load target metadata and non-image measurements from TOML."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from hwoutils.epoch import Epoch
from optixstuff.spectral import Band, Quadrature

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
    A bandpass mapping supplied at load time is the default for ``terms``.
    Observation containers retain ``time_d``, in days from ``epoch``, the same
    origin required by the system used to evaluate their terms.
    """

    target: dict
    relative_astrometry: AstrometryData | None
    photometry: PhotometryData | None
    band_names: tuple[str, ...]
    comparisons: dict
    epoch: Epoch
    _bandpasses: Mapping | None = field(default=None, repr=False, compare=False)

    def terms(self, system_path="", bandpasses=None, *, system=None):
        """Build terms for present data, using filter-name to bandpass mappings.

        ``system_path`` is the model path of the system the terms read; the empty
        default is the model itself.
        An explicit mapping overrides the mapping supplied to ``load_target``.
        Photometric terms require the concrete reference system to split the
        integration at its spectral breaks before tracing the likelihood.
        Missing photometric filters raise ``ValueError`` naming those filters.
        """
        result = []
        if self.relative_astrometry is not None:
            result.append(RelativeAstrometryTerm(self.relative_astrometry, system_path))
        if self.photometry is not None:
            mapping = self._bandpasses if bandpasses is None else bandpasses
            missing = [
                name
                for name in self.band_names
                if mapping is None or name not in mapping
            ]
            if missing:
                raise ValueError(
                    f"Missing bandpasses for filters: {', '.join(missing)}"
                )
            if system is None:
                raise ValueError(
                    "Photometric terms need a system to build the quadrature"
                )
            curves = tuple(mapping[name] for name in self.band_names)
            bands = tuple(Band(curve.low_nm, curve.high_nm) for curve in curves)
            rule = Quadrature.build(bands, system, *curves)
            result.append(PhotometryTerm(self.photometry, rule, curves, system_path))
        return tuple(result)


def _mean_error(value):
    """Reduce a scalar or a [minus, plus] error to one standard deviation."""
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("asymmetric errors must be [minus, plus]")
        return (value[0] + value[1]) / 2.0
    return value


def load_target(
    path, *, epoch: Epoch, pixel_data_ids=(), include_pending=False, bandpasses=None
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
    rows when requested; any unknown ``use`` value raises. ``bandpasses`` may
    supply the default filter mapping for the returned object's ``terms`` method.
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
        astrometry = AstrometryData.from_sep_pa(
            time_d=epoch.days(np.asarray([r["jd"] for r in rows], dtype=np.float64)),
            sep=[r["sep_mas"] / 1000.0 for r in rows],
            sep_err=[_mean_error(r["sep_err_mas"]) / 1000.0 for r in rows],
            pa_deg=[r["pa_deg"] for r in rows],
            pa_err_deg=[_mean_error(r["pa_err_deg"]) for r in rows],
            planet_id=[r.get("planet", 0) for r in rows],
        )
    rows = selected["photometry"]
    band_names = tuple(dict.fromkeys(r["filter"] for r in rows))
    photometry = None
    if rows:
        photometry = PhotometryData.from_dmag(
            time_d=epoch.days(np.asarray([r["jd"] for r in rows], dtype=np.float64)),
            dmag=[r["dmag"] for r in rows],
            dmag_err=[_mean_error(r["dmag_err"]) for r in rows],
            band_id=[band_names.index(r["filter"]) for r in rows],
            planet_id=[r.get("planet", 0) for r in rows],
            is_upper_limit=[r.get("upper_limit", False) for r in rows],
        )
    return TargetData(
        document.get("target", {}),
        astrometry,
        photometry,
        band_names,
        comparisons,
        epoch,
        bandpasses,
    )
