"""Joint counts and astrometry reports, with an interchangeable count predictor.

Times are days from the selected system's epoch. Positions, errors and diffraction
scales are in arcsec. A null is a censored counts measurement; a detection carries
counts and a position with covariance determined by its own reported observables.
"""

import math
from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm

from photomancy.system_terms import _band_contrasts, _relative_positions
from photomancy.terms import AbstractTerm

__all__ = [
    "AbstractReportPredictor",
    "ReportLaw",
    "ReportPrediction",
    "ReportRows",
    "ReportTable",
    "ReportTerm",
    "TabulatedReportPredictor",
]


def _float64(value):
    return jnp.asarray(value, dtype=jnp.float64)


class ReportLaw(eqx.Module):
    """Static detection threshold and observable-based centroid error law."""

    threshold: float = eqx.field(static=True, default=5.0)
    kappa: float = eqx.field(static=True, default=1.0)
    corr: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        """Validate positive finite scales and a proper correlation."""
        for name in ("threshold", "kappa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.corr) or not -1 < self.corr < 1:
            raise ValueError("corr must be finite and lie in (-1, 1)")

    def log_null_probability(self, limit_counts, prediction):
        """Log probability of counts below the detection threshold, shape (R,)."""
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        return log_ndtr((self.threshold * limit_counts - prediction.counts) / noise)

    def log_counts_density(self, counts, limit_counts, prediction):
        """Unconditional counts log density on the detection region, shape (R,)."""
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        value = norm.logpdf(counts, prediction.counts, noise)
        return jnp.where(
            jnp.isnan(counts),
            jnp.nan,
            jnp.where(counts >= self.threshold * limit_counts, value, -jnp.inf),
        )


class ReportRows(eqx.Module):
    """One planet slot per exposure; all fields have shape (R,).

    ``planned`` stores valid exposure metadata with zero placeholders for outcomes.
    Call ``sample`` on a term to realize these rows before treating them as data.
    Null outcome floats are zero. Padding may contain arbitrary stored values.
    """

    time_d: jax.Array = eqx.field(converter=_float64)
    science_s: jax.Array = eqx.field(converter=_float64)
    limit_counts: jax.Array = eqx.field(converter=_float64)
    mode: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, jnp.int32))
    is_valid: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, bool))
    is_detected: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, bool))
    counts: jax.Array = eqx.field(converter=_float64)
    ra: jax.Array = eqx.field(converter=_float64)
    dec: jax.Array = eqx.field(converter=_float64)
    ra_err: jax.Array = eqx.field(converter=_float64)
    dec_err: jax.Array = eqx.field(converter=_float64)
    corr: jax.Array = eqx.field(converter=_float64)

    def __check_init__(self):
        """Require one common row axis."""
        if self.time_d.ndim != 1 or any(
            x.shape != self.time_d.shape for x in jax.tree.leaves(self)
        ):
            raise ValueError("report row fields must all have shape (R,)")

    @classmethod
    def zeros(cls, n_rows) -> Self:
        """All-invalid rows with zero payloads."""
        z = jnp.zeros(n_rows, jnp.float64)
        return cls(
            z,
            z,
            z,
            jnp.zeros(n_rows, jnp.int32),
            jnp.zeros(n_rows, bool),
            jnp.zeros(n_rows, bool),
            z,
            z,
            z,
            z,
            z,
            z,
        )

    @classmethod
    def planned(cls, time_d, science_s, limit_counts, mode) -> Self:
        """Valid exposure metadata, with no outcome yet; each input has shape (R,)."""
        rows = cls.zeros(jnp.asarray(time_d).size)
        return cls(
            time_d,
            science_s,
            limit_counts,
            mode,
            jnp.ones(rows.time_d.shape, bool),
            rows.is_detected,
            rows.counts,
            rows.ra,
            rows.dec,
            rows.ra_err,
            rows.dec_err,
            rows.corr,
        )


class ReportPrediction(eqx.Module):
    """Expected counts, total count variance and east/north position, each (R,)."""

    counts: jax.Array = eqx.field(converter=_float64)
    variance: jax.Array = eqx.field(converter=_float64)
    ra: jax.Array = eqx.field(converter=_float64)
    dec: jax.Array = eqx.field(converter=_float64)


class AbstractReportPredictor(eqx.Module):
    """Predict counts, their variance and the centroid for a row bank."""

    @abstractmethod
    def __call__(self, model, rows: ReportRows) -> ReportPrediction:
        """Return one prediction per row, without reading the outcomes."""
        raise NotImplementedError


class ReportTable(eqx.Module):
    """A mode bank of count rates, diffraction scales and bandpasses.

    Separations have shape (n_modes, P). Finite knots increase strictly;
    padding is a trailing +inf separation with zero rate. Rates are expected
    planet counts per unit linear contrast per second. ``bandpasses`` contains
    one bandpass per mode, used by the same model reader as PhotometryTerm.
    Diffraction scales have shape (n_modes,).
    """

    sep_arcsec: jax.Array = eqx.field(converter=_float64)
    counts_per_contrast_s: jax.Array = eqx.field(converter=_float64)
    lod_arcsec: jax.Array = eqx.field(converter=_float64)
    bandpasses: tuple

    def __check_init__(self):
        """Require compatible mode-bank dimensions."""
        if (
            self.sep_arcsec.ndim != 2
            or self.counts_per_contrast_s.shape != self.sep_arcsec.shape
            or self.lod_arcsec.shape != (self.sep_arcsec.shape[0],)
            or len(self.bandpasses) != self.sep_arcsec.shape[0]
            or self.sep_arcsec.shape[1] < 1
        ):
            raise ValueError(
                "table needs matching (n_modes, P) grids, P >= 1, "
                "and one diffraction scale and bandpass per mode"
            )


class TabulatedReportPredictor(AbstractReportPredictor):
    """Read the bare system model using the astrometry/photometry term helpers."""

    table: ReportTable
    planet: int = eqx.field(static=True)

    def __check_init__(self):
        """Require a static nonnegative planet index."""
        if not isinstance(self.planet, int) or self.planet < 0:
            raise ValueError("planet must be a nonnegative static integer")

    def __call__(self, model, rows: ReportRows) -> ReportPrediction:
        """Predict linear-interpolated counts and sky positions for all rows."""
        clean = _prediction_rows(rows)
        planet_id = jnp.full(clean.time_d.shape, self.planet, jnp.int32)
        ra, dec = _relative_positions(model, clean.time_d, planet_id)
        contrast = _band_contrasts(
            model, self.table.bandpasses, clean.time_d, clean.mode, planet_id
        )
        radius2 = ra**2 + dec**2
        sep = jnp.sqrt(jnp.where(radius2 > 0, radius2, 1.0))
        sep = jnp.where(radius2 > 0, sep, 0.0)
        xp = self.table.sep_arcsec[clean.mode]
        fp = self.table.counts_per_contrast_s[clean.mode]
        finite = jnp.isfinite(xp)
        last = jnp.max(jnp.where(finite, xp, -jnp.inf), axis=1)
        first = jnp.min(jnp.where(finite, xp, jnp.inf), axis=1)
        safe_last = jnp.where(jnp.isfinite(last), last, 0.0)
        xp = jnp.where(finite, xp, safe_last[:, None] + jnp.arange(xp.shape[1]) + 1.0)
        rate = jax.vmap(
            lambda x, knots, rates: jnp.interp(x, knots, rates, left=0.0, right=0.0)
        )(sep, xp, fp)
        rate = jnp.where((sep >= first) & (sep <= last), rate, 0.0)
        counts = contrast * rate * clean.science_s
        return ReportPrediction(counts, clean.limit_counts**2 + counts, ra, dec)


def _prediction_rows(rows):
    """Sanitize padded metadata before handing it to a predictor."""
    return eqx.tree_at(
        lambda r: (r.time_d, r.science_s, r.limit_counts, r.mode),
        rows,
        (
            jnp.where(rows.is_valid, rows.time_d, 0.0),
            jnp.where(rows.is_valid, rows.science_s, 0.0),
            jnp.where(rows.is_valid, rows.limit_counts, 1.0),
            jnp.where(rows.is_valid, rows.mode, 0),
        ),
    )


class ReportTerm(AbstractTerm):
    """An exact or tabulated joint report law on the same observed rows.

    For a tabulated predictor, diffraction scales always come from
    ``predictor.table.lod_arcsec[rows.mode]``. For any other predictor,
    ``lod_arcsec`` is required and has shape (R,). It is used only for sampling;
    scoring always uses the errors and correlation stored in each observed row.
    """

    law: ReportLaw
    rows: ReportRows
    predictor: AbstractReportPredictor
    lod_arcsec: jax.Array | None = eqx.field(
        default=None, converter=lambda x: None if x is None else _float64(x)
    )

    def __check_init__(self):
        """Require row diffraction scales for an exact predictor."""
        if not isinstance(self.predictor, TabulatedReportPredictor):
            if (
                self.lod_arcsec is None
                or jnp.shape(self.lod_arcsec) != self.rows.time_d.shape
            ):
                raise ValueError(
                    "a non-tabulated predictor needs lod_arcsec with shape (R,)"
                )

    def _row_density(self, model, rows):
        clean = _prediction_rows(rows)
        prediction = self.predictor(model, clean)
        detected = rows.is_valid & rows.is_detected
        # Safe operands keep the unused position branch differentiable on nulls.
        sx = jnp.where(detected, rows.ra_err, 1.0)
        sy = jnp.where(detected, rows.dec_err, 1.0)
        rho = jnp.where(detected, rows.corr, 0.0)
        dx = jnp.where(detected, rows.ra, 0.0) - jnp.where(detected, prediction.ra, 0.0)
        dy = jnp.where(detected, rows.dec, 0.0) - jnp.where(
            detected, prediction.dec, 0.0
        )
        has_errors = (sx > 0) & (sy > 0) & (jnp.abs(rho) < 1)
        sx = jnp.where(has_errors, sx, 1.0)
        sy = jnp.where(has_errors, sy, 1.0)
        rho = jnp.where(has_errors, rho, 0.0)
        one_minus = 1 - rho**2
        position = (
            -0.5
            * ((dx / sx) ** 2 + (dy / sy) ** 2 - 2 * rho * dx * dy / (sx * sy))
            / one_minus
        )
        position -= jnp.log(2 * jnp.pi * sx * sy * jnp.sqrt(one_minus))
        position = jnp.where(has_errors, position, -jnp.inf)
        counts = jnp.where(detected, rows.counts, 0.0)
        detection = (
            self.law.log_counts_density(counts, clean.limit_counts, prediction)
            + position
        )
        null = self.law.log_null_probability(clean.limit_counts, prediction)
        value = jnp.where(rows.is_detected, detection, null)
        outcome_nan = jnp.any(
            jnp.stack(
                [
                    jnp.isnan(x)
                    for x in (
                        rows.counts,
                        rows.ra,
                        rows.dec,
                        rows.ra_err,
                        rows.dec_err,
                        rows.corr,
                    )
                ]
            ),
            axis=0,
        )
        value = jnp.where(outcome_nan, jnp.nan, value)
        return jnp.where(rows.is_valid, value, 0.0), prediction

    def __call__(self, model) -> tuple[jax.Array, ReportPrediction]:
        """Sum valid null log probabilities and detection joint log densities."""
        values, prediction = self._row_density(model, self.rows)
        return jnp.sum(values), prediction

    def sample(self, key, model) -> Self:
        """Draw counts then two position normals per row, preserving padding."""
        rows = self.rows
        valid = rows.is_valid & (rows.science_s > 0) & (rows.limit_counts > 0)
        clean = _prediction_rows(eqx.tree_at(lambda r: r.is_valid, rows, valid))
        prediction = self.predictor(model, clean)
        eps = jax.random.normal(key, (rows.time_d.size, 3), dtype=jnp.float64)
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        counts = prediction.counts + noise * eps[:, 0]
        detected = valid & (counts >= self.law.threshold * clean.limit_counts)
        reported_counts = jnp.where(detected, counts, 0.0)
        snr = reported_counts / jnp.sqrt(
            clean.limit_counts**2 + jnp.maximum(reported_counts, 0.0)
        )
        lod_scale = (
            self.predictor.table.lod_arcsec[clean.mode]
            if isinstance(self.predictor, TabulatedReportPredictor)
            else self.lod_arcsec
        )
        sigma = self.law.kappa * lod_scale / jnp.where(detected, snr, 1.0)
        sigma = jnp.where(detected, sigma, 0.0)
        ra = prediction.ra + sigma * eps[:, 1]
        dec = prediction.dec + sigma * (
            self.law.corr * eps[:, 1] + math.sqrt(1 - self.law.corr**2) * eps[:, 2]
        )
        finite = (
            jnp.isfinite(prediction.counts)
            & jnp.isfinite(prediction.variance)
            & jnp.isfinite(rows.limit_counts)
        )
        values = (
            reported_counts,
            jnp.where(detected, ra, 0.0),
            jnp.where(detected, dec, 0.0),
            sigma,
            sigma,
            jnp.where(detected, self.law.corr, 0.0),
        )
        # The boolean outcome flag cannot carry NaN; counts carries failed draws.
        values = tuple(
            jnp.where(rows.is_valid, jnp.where(finite, x, jnp.nan), old)
            for x, old in zip(
                values,
                (rows.counts, rows.ra, rows.dec, rows.ra_err, rows.dec_err, rows.corr),
                strict=True,
            )
        )
        sampled = eqx.tree_at(
            lambda r: (
                r.is_valid,
                r.is_detected,
                r.counts,
                r.ra,
                r.dec,
                r.ra_err,
                r.dec_err,
                r.corr,
            ),
            rows,
            (valid, jnp.where(rows.is_valid, detected, rows.is_detected), *values),
        )
        return eqx.tree_at(lambda t: t.rows, self, sampled)

    def surrogate(self, table, planet) -> Self:
        """Copy with a tabulated predictor and its diffraction scales."""
        return ReportTerm(self.law, self.rows, TabulatedReportPredictor(table, planet))

    def residual(self, model, other) -> jax.Array:
        """Per-row log p(this) - log p(other), both scored on this term's rows."""
        return (
            self._row_density(model, self.rows)[0]
            - other._row_density(model, self.rows)[0]
        )
