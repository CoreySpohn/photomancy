"""Counts reports of planned exposures, with an interchangeable count predictor.

Times are days from the selected system's epoch; a count-rate table carries that epoch
and a predictor refuses a model whose epoch differs. Positions, errors and diffraction
scales are in arcsec. A null is a censored counts measurement; a detection carries
counts and a position whose covariance is a function of its reported counts. Those
counts are data, so the joint density of a detection factorizes into the counts, which
``ReportTerm`` scores, and a Gaussian position with a data-fixed covariance, which a
``RelativeAstrometryTerm`` on ``ReportData.astrometry()`` scores.
"""

import math
from abc import abstractmethod
from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.epoch import Epoch
from hwoutils.model_paths import resolve
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm
from optixstuff.spectral import Quadrature, Transmission

from photomancy.infer.fit import AbstractProposingTerm, LeafProposal
from photomancy.measurements._observables import (
    _band_contrasts,
    _relative_positions,
)
from photomancy.measurements.astrometry import AstrometryData

__all__ = [
    "AbstractReportPredictor",
    "CountRateTable",
    "PrecomputedReportPredictor",
    "ReportData",
    "ReportPrediction",
    "ReportRule",
    "ReportTerm",
]


def _float64(value):
    return jnp.asarray(
        value, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32
    )


class ReportRule(eqx.Module):
    """Static detection threshold and observable-based centroid error rule."""

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

    def detection_limit(self, limit_counts):
        """Minimum reported counts for a detection at the given noise level."""
        return self.threshold * limit_counts

    def log_null_probability(self, limit_counts, prediction):
        """Log probability of counts below the detection threshold, shape (R,)."""
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        return log_ndtr(
            (self.detection_limit(limit_counts) - prediction.counts) / noise
        )

    def log_counts_density(self, counts, limit_counts, prediction):
        """Unconditional counts log density on the detection region, shape (R,)."""
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        value = norm.logpdf(counts, prediction.counts, noise)
        return jnp.where(
            jnp.isnan(counts),
            jnp.nan,
            jnp.where(counts >= self.detection_limit(limit_counts), value, -jnp.inf),
        )


class ReportData(eqx.Module):
    """One planet slot per exposure; all fields have shape (R,).

    ``planned`` stores valid exposure metadata with zero placeholders for outcomes.
    Call ``ReportTerm.draw`` to realize these rows before treating them as data.
    Null outcome floats are zero. Padding may contain arbitrary stored values.
    """

    time_d: jax.Array = eqx.field(converter=_float64)
    science_s: jax.Array = eqx.field(converter=_float64)
    limit_counts: jax.Array = eqx.field(converter=_float64)
    mode: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, jnp.int32))
    valid: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, bool))
    is_detected: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, bool))
    counts: jax.Array = eqx.field(converter=_float64)
    east_arcsec: jax.Array = eqx.field(converter=_float64)
    north_arcsec: jax.Array = eqx.field(converter=_float64)
    east_err_arcsec: jax.Array = eqx.field(converter=_float64)
    north_err_arcsec: jax.Array = eqx.field(converter=_float64)
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
            rows.east_arcsec,
            rows.north_arcsec,
            rows.east_err_arcsec,
            rows.north_err_arcsec,
            rows.corr,
        )

    def astrometry(self, planet: int = 0) -> AstrometryData:
        """The positions of the detected rows, as relative astrometry data.

        ``ReportTerm`` scores the counts; a ``RelativeAstrometryTerm`` on these data
        scores the positions. Rows that are not valid detections are padding.

        Args:
            planet: Index of the planet the rows report (the predictor's ``planet``).
        """
        return AstrometryData(
            time_d=self.time_d,
            east_arcsec=self.east_arcsec,
            north_arcsec=self.north_arcsec,
            east_err_arcsec=self.east_err_arcsec,
            north_err_arcsec=self.north_err_arcsec,
            corr=self.corr,
            planet=jnp.full(self.time_d.shape, planet, int),
            valid=self.valid & self.is_detected,
        )


class ReportPrediction(eqx.Module):
    """Expected counts, total count variance and east/north position, each (R,)."""

    counts: jax.Array = eqx.field(converter=_float64)
    variance: jax.Array = eqx.field(converter=_float64)
    east_arcsec: jax.Array = eqx.field(converter=_float64)
    north_arcsec: jax.Array = eqx.field(converter=_float64)

    @classmethod
    def from_limit(
        cls, counts, limit_counts, east_arcsec=None, north_arcsec=None
    ) -> Self:
        """Combine planet-free variance and shot noise, with optional positions."""
        return cls(
            counts,
            limit_counts**2 + counts,
            jnp.zeros_like(counts) if east_arcsec is None else east_arcsec,
            jnp.zeros_like(counts) if north_arcsec is None else north_arcsec,
        )


class AbstractReportPredictor(eqx.Module):
    """Predict counts, their variance and the centroid for a row bank."""

    @abstractmethod
    def __call__(self, model, rows: ReportData) -> ReportPrediction:
        """Return one prediction per row, without reading the outcomes."""
        raise NotImplementedError


class CountRateTable(eqx.Module):
    """A mode bank of count rates, diffraction scales and transmissions.

    Separations have shape (n_modes, P). Finite knots increase strictly;
    padding is a trailing +inf separation with zero rate. Rates are expected
    planet counts per unit linear contrast per second. ``transmissions`` contains
    one response per mode. Build the matching quadrature on the host, with
    the system and these responses, as for PhotometryTerm.
    Diffraction scales have shape (n_modes,). ``epoch`` is the static origin of
    the report times; a predictor raises if the model's epoch differs.
    """

    separation_arcsec: jax.Array = eqx.field(converter=_float64)
    counts_per_contrast_s: jax.Array = eqx.field(converter=_float64)
    lod_arcsec: jax.Array = eqx.field(converter=_float64)
    quadrature: Quadrature
    transmissions: tuple[Transmission, ...]
    epoch: Epoch = eqx.field(static=True)

    def __check_init__(self):
        """Require compatible mode-bank dimensions."""
        if (
            self.separation_arcsec.ndim != 2
            or self.counts_per_contrast_s.shape != self.separation_arcsec.shape
            or self.lod_arcsec.shape != (self.separation_arcsec.shape[0],)
            or len(self.quadrature.bands) != self.separation_arcsec.shape[0]
            or len(self.transmissions) != self.separation_arcsec.shape[0]
            or self.separation_arcsec.shape[1] < 1
        ):
            raise ValueError(
                "table needs matching (n_modes, P) grids, P >= 1, "
                "and one diffraction scale and transmission per mode"
            )


class PrecomputedReportPredictor(AbstractReportPredictor):
    """Read the bare system model using the astrometry/photometry term helpers."""

    table: CountRateTable
    planet: int = eqx.field(static=True)

    def __check_init__(self):
        """Require a static nonnegative planet index."""
        if not isinstance(self.planet, int) or self.planet < 0:
            raise ValueError("planet must be a nonnegative static integer")

    def __call__(self, model, rows: ReportData) -> ReportPrediction:
        """Predict linear-interpolated counts and sky positions for all rows."""
        if model.epoch != self.table.epoch:
            raise ValueError(
                f"count-rate table epoch {self.table.epoch} differs from "
                f"model epoch {model.epoch}"
            )
        clean = _prediction_rows(rows)
        planet = jnp.full(clean.time_d.shape, self.planet, jnp.int32)
        ra, dec = _relative_positions(model, clean.time_d, planet)
        contrast = _band_contrasts(
            model,
            self.table.quadrature,
            self.table.transmissions,
            clean.time_d,
            clean.mode,
            planet,
        )
        radius2 = ra**2 + dec**2
        sep = jnp.sqrt(jnp.where(radius2 > 0, radius2, 1.0))
        sep = jnp.where(radius2 > 0, sep, 0.0)
        xp = self.table.separation_arcsec[clean.mode]
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
        return ReportPrediction.from_limit(counts, clean.limit_counts, ra, dec)


def _prediction_rows(rows):
    """Sanitize padded metadata before handing it to a predictor."""
    return eqx.tree_at(
        lambda r: (r.time_d, r.science_s, r.limit_counts, r.mode),
        rows,
        (
            jnp.where(rows.valid, rows.time_d, 0.0),
            jnp.where(rows.valid, rows.science_s, 0.0),
            jnp.where(rows.valid, rows.limit_counts, 1.0),
            jnp.where(rows.valid, rows.mode, 0),
        ),
    )


def _scale_fit(counts, limit, rows, rule):
    """Detections as a Gaussian in the log of a common counts scale.

    ``counts`` are the expected planet counts of every row at the model as it is. With
    the detection variance fixed at ``limit^2`` plus the measured counts, the detected
    rows are linear in a multiplier ``exp(t)`` of ``counts``, so their log likelihood
    is Gaussian in ``t`` about ``t_hat`` with standard deviation ``sd``, up to
    ``-chi2 / 2`` and ``log_norm``. ``has`` is False when no valid row is a detection
    (no peak), and ``on_region`` is False when a detection lies below the threshold.

    Returns:
        ``(has, t_hat, sd, chi2, log_norm, on_region)``.
    """
    det = rows.valid & rows.is_detected
    measured = jnp.where(det, rows.counts, 0.0)
    var = limit**2 + measured
    g = jnp.where(det, counts, 0.0)
    s_gg = jnp.sum(jnp.where(det, g**2 / var, 0.0))
    s_cg = jnp.sum(jnp.where(det, measured * g / var, 0.0))
    s_cc = jnp.sum(jnp.where(det, measured**2 / var, 0.0))
    has = (s_cg > 0.0) & (s_gg > 0.0)
    safe_gg = jnp.where(has, s_gg, 1.0)
    safe_cg = jnp.where(has, s_cg, 1.0)
    t_hat = jnp.log(safe_cg / safe_gg)
    sd = jnp.sqrt(safe_gg) / safe_cg
    chi2 = s_cc - jnp.where(has, s_cg**2 / safe_gg, 0.0)
    log_norm = jnp.sum(jnp.where(det, -0.5 * jnp.log(2.0 * jnp.pi * var), 0.0))
    on_region = jnp.all(~det | (rows.counts >= rule.detection_limit(limit)))
    return has, t_hat, sd, chi2, log_norm, on_region


class ReportTerm(AbstractProposingTerm):
    """The counts likelihood of a report, from an exact or a precomputed predictor.

    The term scores the counts of every valid row: a null is the probability of counts
    below the threshold, a detection the density of its counts on the detection region.
    The position of a detection is not scored here; ``data.astrometry()`` gives the data
    of the ``RelativeAstrometryTerm`` that scores it. ``draw`` generates the whole
    report, positions and their errors included.

    For a precomputed predictor, diffraction scales always come from
    ``predictor.table.lod_arcsec[data.mode]``. For any other predictor,
    ``lod_arcsec`` is required and has shape (R,). It is used only for sampling the
    position errors.

    The term is the exact likelihood at the model: it reads the planet's photometric
    leaf from the model and holds no integral over it. The counts depend on the phase,
    so on a Thiele-Innes block, whose model holds the orbit on one line-of-sight sheet,
    the term belongs inside a :class:`~photomancy.orbit.SheetSum` of that orbit row.

    ``leaf_proposal`` offers a proposal density for the photometric leaf from the
    detections (see :class:`~photomancy.infer.LeafProposal`); it never enters the
    likelihood.

    Attributes:
        leaf_path: Path of the positive photometric leaf (a radius, say), or ``None``
            when the term offers no leaf proposal.
        leaf_exponent: Power of the leaf in the expected counts (2 for a radius). It
            shapes the proposal only; the likelihood reads the model.
    """

    rule: ReportRule
    data: ReportData
    predictor: AbstractReportPredictor
    lod_arcsec: jax.Array | None = eqx.field(
        default=None, converter=lambda x: None if x is None else _float64(x)
    )
    leaf_path: str | None = eqx.field(static=True, default=None)
    leaf_exponent: float = eqx.field(static=True, default=2.0)
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require row diffraction scales for an exact predictor."""
        if not isinstance(self.predictor, PrecomputedReportPredictor):
            if (
                self.lod_arcsec is None
                or jnp.shape(self.lod_arcsec) != self.data.time_d.shape
            ):
                raise ValueError(
                    "a predictor that is not precomputed needs lod_arcsec with "
                    "shape (R,)"
                )
        if not self.leaf_exponent > 0:
            raise ValueError("leaf_exponent must be positive")

    def _row_density(self, model, rows):
        clean = _prediction_rows(rows)
        prediction = self.predictor(model, clean)
        detected = rows.valid & rows.is_detected
        counts = jnp.where(detected, rows.counts, 0.0)
        detection = self.rule.log_counts_density(counts, clean.limit_counts, prediction)
        null = self.rule.log_null_probability(clean.limit_counts, prediction)
        value = jnp.where(rows.is_detected, detection, null)
        outcome_nan = jnp.any(
            jnp.stack(
                [
                    jnp.isnan(x)
                    for x in (
                        rows.counts,
                        rows.east_arcsec,
                        rows.north_arcsec,
                        rows.east_err_arcsec,
                        rows.north_err_arcsec,
                        rows.corr,
                    )
                ]
            ),
            axis=0,
        )
        value = jnp.where(outcome_nan, jnp.nan, value)
        return jnp.where(rows.valid, value, 0.0), prediction

    def __call__(self, model) -> jax.Array:
        """Sum valid null log probabilities and detection counts log densities."""
        return jnp.sum(self._row_density(model, self.data)[0])

    def leaf_proposal(self, model) -> LeafProposal:
        """A Gaussian proposal for the photometric leaf, as a one-component mixture.

        The detections are Gaussian in the log of the counts scale
        (:func:`_scale_fit`); the component is that Gaussian in ``ln(leaf)``, with its
        own mass (the Gaussian integral of the likelihood) as the log weight. A report
        with no detection has no peak: the log mass is ``-inf``.

        Raises:
            ValueError: With no ``leaf_path``, or a leaf that is not a single value.
        """
        if self.leaf_path is None:
            raise ValueError("this ReportTerm has no leaf_path to propose")
        leaf = jnp.asarray(resolve(model, self.leaf_path))
        if leaf.size != 1:
            raise ValueError(f"leaf {self.leaf_path!r} must hold a single value")
        clean = _prediction_rows(self.data)
        prediction = self.predictor(model, clean)
        has, t_hat, sd, chi2, log_norm, on_region = _scale_fit(
            prediction.counts, clean.limit_counts, self.data, self.rule
        )
        log_pi = jnp.where(
            has & on_region,
            log_norm - 0.5 * chi2 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(sd),
            -jnp.inf,
        )
        return LeafProposal(
            self.leaf_path,
            (jnp.log(jnp.reshape(leaf, ())) + t_hat / self.leaf_exponent)[None],
            (sd / self.leaf_exponent)[None],
            log_pi[None],
            jnp.asarray(-jnp.inf),
            jnp.asarray(jnp.inf),
        )

    def predict(self, model) -> ReportPrediction:
        """Return the predictor's report for every row."""
        return self._row_density(model, self.data)[1]

    def draw(self, key, model) -> Self:
        """Draw counts then two position normals per row, preserving padding."""
        rows = self.data
        valid = rows.valid & (rows.science_s > 0) & (rows.limit_counts > 0)
        clean = _prediction_rows(eqx.tree_at(lambda r: r.valid, rows, valid))
        prediction = self.predictor(model, clean)
        eps = jax.random.normal(key, (rows.time_d.size, 3), dtype=jnp.float64)
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        counts = prediction.counts + noise * eps[:, 0]
        detected = valid & (counts >= self.rule.detection_limit(clean.limit_counts))
        reported_counts = jnp.where(detected, counts, 0.0)
        snr = reported_counts / jnp.sqrt(
            clean.limit_counts**2 + jnp.maximum(reported_counts, 0.0)
        )
        lod_scale = (
            self.predictor.table.lod_arcsec[clean.mode]
            if isinstance(self.predictor, PrecomputedReportPredictor)
            else self.lod_arcsec
        )
        sigma = self.rule.kappa * lod_scale / jnp.where(detected, snr, 1.0)
        sigma = jnp.where(detected, sigma, 0.0)
        ra = prediction.east_arcsec + sigma * eps[:, 1]
        dec = prediction.north_arcsec + sigma * (
            self.rule.corr * eps[:, 1] + math.sqrt(1 - self.rule.corr**2) * eps[:, 2]
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
            jnp.where(detected, self.rule.corr, 0.0),
        )
        # The boolean outcome flag cannot carry NaN; counts carries failed draws.
        values = tuple(
            jnp.where(rows.valid, jnp.where(finite, x, jnp.nan), old)
            for x, old in zip(
                values,
                (
                    rows.counts,
                    rows.east_arcsec,
                    rows.north_arcsec,
                    rows.east_err_arcsec,
                    rows.north_err_arcsec,
                    rows.corr,
                ),
                strict=True,
            )
        )
        sampled = eqx.tree_at(
            lambda r: (
                r.valid,
                r.is_detected,
                r.counts,
                r.east_arcsec,
                r.north_arcsec,
                r.east_err_arcsec,
                r.north_err_arcsec,
                r.corr,
            ),
            rows,
            (valid, jnp.where(rows.valid, detected, rows.is_detected), *values),
        )
        return eqx.tree_at(lambda t: t.data, self, sampled)

    def surrogate(self, table, planet) -> Self:
        """Copy with a precomputed predictor and its diffraction scales."""
        return ReportTerm(
            self.rule,
            self.data,
            PrecomputedReportPredictor(table, planet),
            leaf_path=self.leaf_path,
            leaf_exponent=self.leaf_exponent,
        )

    def log_ratio(self, model, other) -> jax.Array:
        """Per-row log p(this) - log p(other), both scored on this term's rows."""
        return (
            self._row_density(model, self.data)[0]
            - other._row_density(model, self.data)[0]
        )
