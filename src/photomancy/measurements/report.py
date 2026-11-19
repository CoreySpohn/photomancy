"""Counts reports of planned exposures, with an interchangeable count predictor.

Times are days from the selected system's epoch; a report table carries that epoch
and a predictor refuses a model whose epoch differs. Positions, errors and diffraction
scales are in arcsec. A null is a censored counts measurement; a detection carries
counts and a position whose covariance is a function of its reported counts. Those
counts are data, so the joint density of a detection factorizes into the counts, which
``ReportTerm`` scores, and a Gaussian position with a data-fixed covariance, which a
``RelativeAstrometryTerm`` on ``ReportRows.astrometry()`` scores.
"""

import math
from abc import abstractmethod
from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.epoch import Epoch
from hwoutils.model_paths import assign, resolve
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm
from optixstuff.spectral import Quadrature, Transmission
from orbix.system.rotation import from_basis, rotate

from photomancy.infer.fit import AbstractConditionalTerm, LeafConditional
from photomancy.measurements._observables import (
    _band_contrasts,
    _relative_positions,
)
from photomancy.measurements.astrometry import RelativeAstromData

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
    return jnp.asarray(
        value, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32
    )


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

    def astrometry(self, planet: int = 0) -> RelativeAstromData:
        """The positions of the detected rows, as relative astrometry data.

        ``ReportTerm`` scores the counts; a ``RelativeAstrometryTerm`` on these data
        scores the positions. Rows that are not valid detections are padding.

        Args:
            planet: Index of the planet the rows report (the predictor's ``planet``).
        """
        return RelativeAstromData(
            times=self.time_d,
            ra=self.ra,
            dec=self.dec,
            ra_err=self.ra_err,
            dec_err=self.dec_err,
            corr=self.corr,
            planet_id=jnp.full(self.time_d.shape, planet, int),
            is_valid=self.is_valid & self.is_detected,
        )


class ReportPrediction(eqx.Module):
    """Expected counts, total count variance and east/north position, each (R,)."""

    counts: jax.Array = eqx.field(converter=_float64)
    variance: jax.Array = eqx.field(converter=_float64)
    ra: jax.Array = eqx.field(converter=_float64)
    dec: jax.Array = eqx.field(converter=_float64)

    @classmethod
    def from_limit(cls, counts, limit_counts, ra=None, dec=None) -> Self:
        """Combine planet-free variance and shot noise, with optional positions."""
        return cls(
            counts,
            limit_counts**2 + counts,
            jnp.zeros_like(counts) if ra is None else ra,
            jnp.zeros_like(counts) if dec is None else dec,
        )


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
    one response per mode. Build the matching quadrature on the host, with
    the system and these responses, as for PhotometryTerm.
    Diffraction scales have shape (n_modes,). ``epoch`` is the static origin of
    the report times; a predictor raises if the model's epoch differs.
    """

    sep_arcsec: jax.Array = eqx.field(converter=_float64)
    counts_per_contrast_s: jax.Array = eqx.field(converter=_float64)
    lod_arcsec: jax.Array = eqx.field(converter=_float64)
    quadrature: Quadrature
    bandpasses: tuple[Transmission, ...]
    epoch: Epoch = eqx.field(static=True)

    def __check_init__(self):
        """Require compatible mode-bank dimensions."""
        if (
            self.sep_arcsec.ndim != 2
            or self.counts_per_contrast_s.shape != self.sep_arcsec.shape
            or self.lod_arcsec.shape != (self.sep_arcsec.shape[0],)
            or len(self.quadrature.bands) != self.sep_arcsec.shape[0]
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
        if model.epoch != self.table.epoch:
            raise ValueError(
                f"report table epoch {self.table.epoch} differs from "
                f"model epoch {model.epoch}"
            )
        clean = _prediction_rows(rows)
        planet_id = jnp.full(clean.time_d.shape, self.planet, jnp.int32)
        ra, dec = _relative_positions(model, clean.time_d, planet_id)
        contrast = _band_contrasts(
            model,
            self.table.quadrature,
            self.table.bandpasses,
            clean.time_d,
            clean.mode,
            planet_id,
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
        return ReportPrediction.from_limit(counts, clean.limit_counts, ra, dec)


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


def _reflect_row(orbits, row):
    """The stacked ``orbits`` with row ``row`` reflected through the sky plane.

    The reflected orbit has the same sky track and the opposite line-of-sight sign: its
    in-plane axes are the reflections of the original ones and its plane normal follows
    from them, so a sky-only measurement cannot tell the two apart and a phase-dependent
    one can.
    """
    flip = jnp.asarray([1.0, 1.0, -1.0])
    f = rotate(orbits.quat, jnp.asarray([1.0, 0.0, 0.0])) * flip
    g = rotate(orbits.quat, jnp.asarray([0.0, 1.0, 0.0])) * flip
    quat = from_basis(f, g, jnp.cross(f, g))
    return eqx.tree_at(
        lambda o: o.quat, orbits, orbits.quat.at[..., row, :].set(quat[..., row, :])
    )


def _scale_fit(counts, limit, rows, law):
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
    det = rows.is_valid & rows.is_detected
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
    on_region = jnp.all(~det | (rows.counts >= law.detection_limit(limit)))
    return has, t_hat, sd, chi2, log_norm, on_region


class ReportTerm(AbstractConditionalTerm):
    """An exact or tabulated counts report law on the observed rows.

    The term scores the counts of every valid row: a null is the probability of counts
    below the threshold, a detection the density of its counts on the detection region.
    The position of a detection is not scored here; ``rows.astrometry()`` gives the data
    of the ``RelativeAstrometryTerm`` that scores it. ``draw`` generates the whole
    report, positions and their errors included.

    For a tabulated predictor, diffraction scales always come from
    ``predictor.table.lod_arcsec[rows.mode]``. For any other predictor,
    ``lod_arcsec`` is required and has shape (R,). It is used only for sampling the
    position errors.

    The term is the exact likelihood at the model: it reads the planet's photometric
    leaf from the model and holds no integral over it. With ``orbit_path`` it also sums
    the two line-of-sight sheets of the planet's orbit, ``logaddexp(L_plus, L_minus) -
    log 2``, which is the likelihood of a sheet-summed Thiele-Innes block (a model
    holds the orbit on one sheet; the other is its reflection through the sky plane).
    The sum is over the whole report, so a row added later enters through
    ``replace_term`` of the term holding every row, not through ``append`` of a second
    term (a product of two sheet sums is not the sheet sum of the product).

    ``leaf_conditional`` offers a proposal density for the photometric leaf from the
    detections (see :class:`~photomancy.infer.LeafConditional`); it never enters the
    likelihood.

    Attributes:
        leaf_path: Path of the positive photometric leaf (a radius, say), or ``None``
            when the term offers no leaf proposal.
        orbit_path: Path of the stacked orbits whose sheet is summed, or ``None`` for
            no sum. Needs a tabulated predictor (its ``planet`` is the row reflected).
        leaf_exponent: Power of the leaf in the expected counts (2 for a radius). It
            shapes the proposal only; the likelihood reads the model.
    """

    law: ReportLaw
    rows: ReportRows
    predictor: AbstractReportPredictor
    lod_arcsec: jax.Array | None = eqx.field(
        default=None, converter=lambda x: None if x is None else _float64(x)
    )
    leaf_path: str | None = eqx.field(static=True, default=None)
    orbit_path: str | None = eqx.field(static=True, default=None)
    leaf_exponent: float = eqx.field(static=True, default=2.0)
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require row diffraction scales for an exact predictor, and a planet row."""
        if not isinstance(self.predictor, TabulatedReportPredictor):
            if (
                self.lod_arcsec is None
                or jnp.shape(self.lod_arcsec) != self.rows.time_d.shape
            ):
                raise ValueError(
                    "a non-tabulated predictor needs lod_arcsec with shape (R,)"
                )
            if self.orbit_path is not None:
                raise ValueError(
                    "orbit_path needs a tabulated predictor, whose planet is the "
                    "orbit row summed over its two sheets"
                )
        if not self.leaf_exponent > 0:
            raise ValueError("leaf_exponent must be positive")

    def _row_density(self, model, rows):
        clean = _prediction_rows(rows)
        prediction = self.predictor(model, clean)
        detected = rows.is_valid & rows.is_detected
        counts = jnp.where(detected, rows.counts, 0.0)
        detection = self.law.log_counts_density(counts, clean.limit_counts, prediction)
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

    def _reflected(self, model):
        """The model with the planet's orbit reflected through the sky plane."""
        orbits = resolve(model, self.orbit_path)
        return assign(
            model, self.orbit_path, _reflect_row(orbits, self.predictor.planet)
        )

    def _sheet_total(self, model):
        return jnp.sum(self._row_density(model, self.rows)[0])

    def __call__(self, model) -> jax.Array:
        """Sum valid null log probabilities and detection counts log densities.

        With ``orbit_path`` the value is the log of the mean of the two sheets'
        likelihoods.
        """
        total = self._sheet_total(model)
        if self.orbit_path is None:
            return total
        return jnp.logaddexp(
            total, self._sheet_total(self._reflected(model))
        ) - jnp.log(2.0)

    def leaf_conditional(self, model) -> LeafConditional:
        """A Gaussian mixture proposal for the photometric leaf, one component a sheet.

        Each sheet's detections are Gaussian in the log of the counts scale
        (:func:`_scale_fit`); the component is that Gaussian in ``ln(leaf)``, with its
        own mass (the Gaussian integral of the likelihood) as the log weight. A report
        with no detection has no peak: every log mass is ``-inf``.

        Raises:
            ValueError: With no ``leaf_path``, or a leaf that is not a single value.
        """
        if self.leaf_path is None:
            raise ValueError("this ReportTerm has no leaf_path to propose")
        leaf = jnp.asarray(resolve(model, self.leaf_path))
        if leaf.size != 1:
            raise ValueError(f"leaf {self.leaf_path!r} must hold a single value")
        models = [model] if self.orbit_path is None else [model, self._reflected(model)]
        clean = _prediction_rows(self.rows)
        means, sds, log_pis = [], [], []
        for sheet in models:
            prediction = self.predictor(sheet, clean)
            has, t_hat, sd, chi2, log_norm, on_region = _scale_fit(
                prediction.counts, clean.limit_counts, self.rows, self.law
            )
            ok = has & on_region
            means.append(jnp.log(jnp.reshape(leaf, ())) + t_hat / self.leaf_exponent)
            sds.append(sd / self.leaf_exponent)
            log_pis.append(
                jnp.where(
                    ok,
                    log_norm - 0.5 * chi2 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(sd),
                    -jnp.inf,
                )
            )
        return LeafConditional(
            self.leaf_path,
            jnp.stack(means),
            jnp.stack(sds),
            jnp.stack(log_pis),
            jnp.asarray(-jnp.inf),
            jnp.asarray(jnp.inf),
        )

    def predict(self, model) -> ReportPrediction:
        """Return the predictor's report for every row."""
        return self._row_density(model, self.rows)[1]

    def draw(self, key, model) -> Self:
        """Draw counts then two position normals per row, preserving padding."""
        rows = self.rows
        valid = rows.is_valid & (rows.science_s > 0) & (rows.limit_counts > 0)
        clean = _prediction_rows(eqx.tree_at(lambda r: r.is_valid, rows, valid))
        prediction = self.predictor(model, clean)
        eps = jax.random.normal(key, (rows.time_d.size, 3), dtype=jnp.float64)
        noise = jnp.sqrt(jnp.where(prediction.variance > 0, prediction.variance, 1.0))
        counts = prediction.counts + noise * eps[:, 0]
        detected = valid & (counts >= self.law.detection_limit(clean.limit_counts))
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
        return ReportTerm(
            self.law,
            self.rows,
            TabulatedReportPredictor(table, planet),
            leaf_path=self.leaf_path,
            orbit_path=self.orbit_path,
            leaf_exponent=self.leaf_exponent,
        )

    def residual(self, model, other) -> jax.Array:
        """Per-row log p(this) - log p(other), both scored on this term's rows."""
        return (
            self._row_density(model, self.rows)[0]
            - other._row_density(model, self.rows)[0]
        )
