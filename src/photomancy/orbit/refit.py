"""Stateless importance refit of one planet's orbit from its record buffer.

A refit reads the records of a planet (the whole buffer, nothing carried
between calls) and returns an equal-weight cloud of physical orbit elements
with the Kish effective size of the importance sample behind it. The effective
size is the status: a refit that cannot reach ``ess_floor`` says so and the
caller decides what to do. Everything here is pure JAX with fixed shapes, so a
refit runs under ``jax.vmap`` over planets.

The declared model is :class:`PhysicalPriorTarget`: the physical prior of the
fixed-mass orbit model (``ln P`` uniform, ``e ~ Beta(0.867, 3.03)``, ``cos i``
uniform, three uniform angles) with the stellar mass uniform in its logarithm
inside a window around the catalog mass. In the Thiele-Innes chart of
:class:`~photomancy.orbit.updater.ThieleInnesTarget` that density is the
eccentricity density times ``1 / (a^2 sqrt(u^2 - 4 v^2))``: the map
``(ln a, cos i, omega, W) -> (four constants)`` has Jacobian
``a^4 (1 - cos^2 i) = a^2 sqrt(u^2 - 4 v^2)`` with ``u`` the sum of the
squared constants and ``v`` their determinant, and it is two to one (the
reflection through the sky plane, which keeps the sky track and flips the
phase angle).

Two engines share the one model:

- :func:`scale_rotate_refit`: exact importance sampling by scale-and-rotate
  with the mass drawn from its window. Used for one record, where the
  proposal carries the prior exactly.
- :func:`collapsed_refit`: the four linear constants are drawn from their
  conditional Gaussian (the marginal likelihood is the weight) and the other
  three coordinates from adaptive rounds, a defensive mixture of the prior box
  and Gaussian kernels on points resampled from the previous round. The weight
  carries the physical prior over the Gaussian reference prior of the constants.

:func:`importance_refit` picks between them by the number of valid records.
Under ``vmap`` the choice is a ``lax.cond`` that evaluates both engines, so a
batch whose entries share an engine should call that engine directly.

The helpers ``_linear_gaussian`` and ``_rest_log_prior`` come from
:mod:`photomancy.orbit.updater` unchanged; the resampler, the collapsed draw and the
kernel-fit machinery are written here and not shared with the updater.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import G, Msun2kg, Rearth2AU, two_pi
from jax.scipy.special import betainc, log_ndtr, logsumexp, ndtr
from jax.scipy.stats import beta as beta_dist
from jax.scipy.stats import norm
from orbix.equations.orbit import mean_motion
from orbix.equations.phase import lambert_phase_exact
from orbix.kepler.core import diff_solve_trig

from photomancy.orbit.few_epoch import MU_SUN, TIBasisModel, ti_semimajor_axis
from photomancy.orbit.forward import predict_photometry
from photomancy.orbit.ofti import _scale_rotate_score
from photomancy.orbit.updater import (
    ThieleInnesTarget,
    UpdaterStatus,
    _linear_gaussian,
    _rest_log_prior,
)

ECC_ALPHA = 0.867
ECC_BETA = 3.03
ELEMENT_NAMES = (
    "ln_period",
    "e",
    "cos_i",
    "W",
    "omega",
    "tau",
    "ln_mass",
    "ln_lam",
)

_TINY = 1e-300
_UNDETECTABLE_DMAG0 = -1e8
_LN10 = jnp.log(10.0)
_TAU_IMAGES = jnp.arange(-2.0, 3.0)
_MAP_BATCH = 2**14
_DENSITY_BATCH = 1024
_GL_X, _GL_W = (jnp.asarray(v) for v in np.polynomial.legendre.leggauss(24))
_N_CELLS = 512
_LADDER_FLOOR = 1000.0
_LADDER_FLOOR_MIN = 100.0
_LADDER_SHRINK = 0.5
_BISECT_STEPS = 24
_RETRIES = 2
_N_SUPPORT = 2**12
_K_NEAR = 64
_K_NEAR_MAX = 2**10
_K_SCALE = 1.5
_K_DISTINCT = 16.0


class PhysicalPriorTarget(ThieleInnesTarget):
    """Thiele-Innes chart carrying the physical prior with a stellar-mass window.

    The chart, the forward model and the linear block are those of
    :class:`~photomancy.orbit.updater.ThieleInnesTarget`. ``model.mu_lo`` and
    ``model.mu_hi`` (AU^3/day^2) are the mass window, so the mass is uniform in
    its logarithm between them; ``model.lnP_lo`` and ``model.lnP_hi`` bound the
    period and ``model.e_max`` the eccentricity. ``model.s_ti`` is only the scale
    of the Gaussian the collapsed draw uses as its reference for the constants.

    :meth:`log_prior` is the physical density in the chart, not the Gaussian
    product of the parent: it is ``Beta(0.867, 3.03)`` on ``e`` truncated to
    ``e_max`` minus ``log(a^2 sqrt(u^2 - 4 v^2))``, and ``-inf`` outside the
    model's support. The density that remains constant over the support is
    :meth:`log_prior_constant`.

    The photometric scale ``ln(Ag Rp^2)`` (AU^2, the ``Lambda`` of
    :func:`~photomancy.orbit.forward.predict_photometry`) is uniform in its
    logarithm over ``[ln_lam_lo, ln_lam_hi]`` and is integrated out of the report
    likelihood in closed form (:func:`report_log_weights`). The default range is
    an Earth twin (albedo 0.3, one Earth radius) plus or minus one in the log.

    Args:
        model: The ``TIBasisModel``. Default: its field defaults.
        dist_pc: Distance (parsec). Default 10.
        ln_lam_range: ``(low, high)`` of the log scale. Default: Earth twin
            plus or minus one.
    """

    ln_lam_lo: jax.Array
    ln_lam_hi: jax.Array

    def __init__(self, model=None, dist_pc=10.0, ln_lam_range=None):
        """Store the chart, the distance and the photometric scale range."""
        super().__init__(model, dist_pc)
        if ln_lam_range is None:
            ref = jnp.log(0.3 * Rearth2AU**2)
            ln_lam_range = (ref - 1.0, ref + 1.0)
        self.ln_lam_lo = jnp.asarray(ln_lam_range[0], dtype=float)
        self.ln_lam_hi = jnp.asarray(ln_lam_range[1], dtype=float)

    @property
    def ln_lam_ref(self):
        """Middle of the log scale range."""
        return 0.5 * (self.ln_lam_lo + self.ln_lam_hi)

    @property
    def ln_lam_half(self):
        """Half width of the log scale range."""
        return 0.5 * (self.ln_lam_hi - self.ln_lam_lo)

    def log_prior(self, theta):
        """Log physical density of the chart point, up to a constant.

        Args:
            theta: Chart point. Shape ``(7,)``.

        Returns:
            Scalar; ``-inf`` outside ``model.in_support``.
        """
        c = theta[3:]
        u = jnp.sum(c**2)
        v = c[0] * c[3] - c[2] * c[1]
        disc = jnp.sqrt(jnp.maximum(u**2 - 4.0 * v**2, _TINY))
        log_a2 = jnp.log((u + disc) / 2.0)
        e = jnp.clip(theta[0], _TINY, 1.0 - 1e-12)
        log_e = beta_dist.logpdf(e, ECC_ALPHA, ECC_BETA) - jnp.log(
            betainc(ECC_ALPHA, ECC_BETA, self.model.e_max)
        )
        return jnp.where(
            self.model.in_support(theta), log_e - log_a2 - jnp.log(disc), -jnp.inf
        )

    def log_prior_constant(self):
        """Log of the density factor :meth:`log_prior` leaves out.

        The period density ``1 / (lnP_hi - lnP_lo)``, the density of ``ln a``
        at a given period ``3 / ln(mu_hi / mu_lo)``, the angle densities
        ``1 / (2 pi)^2`` and the factor 1/2 of ``cos i`` against the two-to-one
        factor 2 of the map. With it the evidence is the marginal likelihood.
        """
        m = self.model
        return (
            -jnp.log(m.lnP_hi - m.lnP_lo)
            - jnp.log(jnp.log(m.mu_hi / m.mu_lo) / 3.0)
            - 2.0 * jnp.log(two_pi)
        )

    def log_report(self, theta, report):
        """Log of the report likelihood of a chart point, scale and mirrors summed.

        The chart point stands for two mirror orbits with equal prior weight that
        share the sky track and differ in the sign of the line-of-sight offset
        (so in the phase angle); the photometric scale is integrated over its
        range. The result is ``log( (1/2) sum_b int L(b, s) p(s) ds )``.

        Args:
            theta: Chart point. Shape ``(7,)``.
            report: A :class:`ReportData`, or None for no report.

        Returns:
            Scalar; zero when ``report`` is None.
        """
        if report is None:
            return jnp.zeros(())
        terms = _ti_report_terms(self, theta, _report_xy(self, theta, report), report)
        return logsumexp(terms) - jnp.log(2.0)

    def mass_window_msun(self):
        """The mass window ``(low, high)`` in solar masses."""
        return self.model.mu_lo / MU_SUN, self.model.mu_hi / MU_SUN


def physical_prior_target(mass_msun, dist_pc, half_width=0.10, ln_lam_range=None):
    """A :class:`PhysicalPriorTarget` with the mass inside ``mass_msun (1 +/- w)``.

    Args:
        mass_msun: Catalog stellar mass (solar masses).
        dist_pc: Distance (parsec).
        half_width: Fractional half width of the mass window. Default 0.10.
        ln_lam_range: ``(low, high)`` of ``ln(Ag Rp^2)`` in AU^2, the log scale
            being uniform between them. Default: Earth twin plus or minus one.

    Returns:
        The target. The period range and eccentricity bound are the model's
        defaults (50 to 5000 d, 0.9).
    """
    mu = MU_SUN * mass_msun
    model = TIBasisModel(mu_lo=(1.0 - half_width) * mu, mu_hi=(1.0 + half_width) * mu)
    return PhysicalPriorTarget(model=model, dist_pc=dist_pc, ln_lam_range=ln_lam_range)


class RefitCloud(eqx.Module):
    """Equal-weight orbit cloud of a refit.

    Args:
        elements: Physical elements ``(ln P, e, cos_i, W, omega, tau, ln_mass,
            ln_lam)``, shape ``(n, 8)``: period in days, node and argument of
            periastron in radians, ``tau`` the periastron phase with the mean
            anomaly ``2 pi ((t - t_ref) / P - tau)`` and ``t_ref`` the model's
            reference epoch, the mass in solar masses. ``ln_lam`` is NaN until
            a photometric term fills it.
        log_evidence: Log marginal likelihood of the records under the declared
            model (data normalization included).
        ess: Kish effective size of the last importance round.
        status: An :class:`~photomancy.orbit.updater.UpdaterStatus` value:
            ``OK``, ``LOW_ESS`` (``ess`` under the floor) or ``NO_SUPPORT``.
        rounds: Importance rounds run.
        max_weight: Largest normalized importance weight of the last round (the
            weight holds ``1 / (1 - cos_i^2)``, so face-on orbits can dominate).
    """

    elements: jax.Array
    log_evidence: jax.Array
    ess: jax.Array
    status: jax.Array
    rounds: jax.Array
    max_weight: jax.Array


class ReportData(eqx.Module):
    """Imaging reports of one planet slot: detections and nulls, one row per epoch.

    Each row is the report of one exposure under the reporting law: with the
    planet-free noise standard deviation ``limit`` (counts), the expected planet
    counts ``N`` and the total variance ``limit^2 + N`` (planet shot noise
    included), a null is the event that the measured counts fall under
    ``snr_thresh * limit``, and a detection carries its measured counts. The
    position of a detection belongs in the astrometric buffer.

    Args:
        epochs: Exposure epochs (days). Shape ``(N,)``.
        sep_grid: Separation grid per epoch (arcsec), padded. Shape ``(N, P)``.
        dmag0_grid: Delta magnitude at which the expected planet counts equal
            ``snr_thresh * limit``, per epoch on ``sep_grid``; padded columns hold
            ``-inf``. Shape ``(N, P)``.
        limit_grid: Planet-free noise standard deviation (counts) per epoch on
            ``sep_grid``; it varies with separation. Padded columns are ignored.
            Shape ``(N, P)``.
        is_detected: Detection (True) or null (False). Shape ``(N,)``.
        counts: Measured counts of a detection. Shape ``(N,)``.
        is_valid: Padding mask. Shape ``(N,)``.
        snr_thresh: Detection threshold in planet-free standard deviations
            (static). Default 5.0.
    """

    epochs: jax.Array
    sep_grid: jax.Array
    dmag0_grid: jax.Array
    limit_grid: jax.Array
    is_detected: jax.Array
    counts: jax.Array
    is_valid: jax.Array
    snr_thresh: float = eqx.field(static=True, default=5.0)

    @classmethod
    def zeros(cls, max_n, max_pts, snr_thresh=5.0):
        """An all-invalid placeholder: ``max_n`` epochs, ``max_pts`` grid points."""
        return cls(
            epochs=jnp.zeros(max_n),
            sep_grid=jnp.zeros((max_n, max_pts)),
            dmag0_grid=jnp.full((max_n, max_pts), -jnp.inf),
            limit_grid=jnp.ones((max_n, max_pts)),
            is_detected=jnp.zeros(max_n, dtype=bool),
            counts=jnp.zeros(max_n),
            is_valid=jnp.zeros(max_n, dtype=bool),
            snr_thresh=snr_thresh,
        )


def _interp_report(alpha, data):
    """Interpolate the dmag0 limit and the noise level at the predicted separations.

    The limit follows :func:`photomancy.orbit.likelihoods._interp_dmag0`: padded
    columns (``dmag0 == -inf``) become an increasing tail and separations outside
    the curve get a finite "undetectable" sentinel, so the expected counts there
    are zero. The noise level is held at its edge values outside the curve.

    Args:
        alpha: Predicted separations (arcsec). Shape ``(N,)``.
        data: A :class:`ReportData`.

    Returns:
        ``(dmag0, limit)``, each shape ``(N,)``.
    """
    is_pad = ~jnp.isfinite(data.dmag0_grid)
    real_sep = jnp.where(is_pad, -jnp.inf, data.sep_grid)
    max_real = jnp.max(real_sep, axis=1, keepdims=True)
    max_real = jnp.where(jnp.isfinite(max_real), max_real, 0.0)
    xp = jnp.where(
        is_pad,
        max_real + (jnp.arange(data.sep_grid.shape[1]) + 1.0) * 1e-6,
        data.sep_grid,
    )
    last = jnp.argmax(real_sep, axis=1)[:, None]
    edge_limit = jnp.take_along_axis(data.limit_grid, last, axis=1)
    fp_dmag0 = jnp.where(is_pad, _UNDETECTABLE_DMAG0, data.dmag0_grid)
    fp_limit = jnp.where(is_pad, edge_limit, data.limit_grid)

    def one(a, x, f0, fl):
        dmag0 = jnp.interp(
            a, x, f0, left=_UNDETECTABLE_DMAG0, right=_UNDETECTABLE_DMAG0
        )
        return dmag0, jnp.interp(a, x, fl)

    return jax.vmap(one)(alpha, xp, fp_dmag0, fp_limit)


def _log_counts_pred(dmag0, limit, ln_contrast, data):
    """Log of the expected planet counts from the contrast and the detection limit.

    ``N = snr_thresh * limit * 10^(-0.4 (dMag - dmag0))`` with the exponent
    clipped at 200, and ``ln_contrast = -0.4 ln(10) dMag``.
    """
    log_limit = jnp.log(data.snr_thresh * jnp.maximum(limit, _TINY))
    return jnp.minimum(
        log_limit + 0.4 * _LN10 * dmag0 + ln_contrast, log_limit + 200.0 * _LN10
    )


def _report_sum(counts_pred, limit, data):
    """Summed log report likelihood for the expected counts ``(N,)``."""
    variance = limit**2 + counts_pred
    s = jnp.sqrt(jnp.where(variance > 0.0, variance, 1.0))
    log_null = log_ndtr((data.snr_thresh * limit - counts_pred) / s)
    on_region = data.counts >= data.snr_thresh * limit
    log_det = jnp.where(on_region, norm.logpdf(data.counts, counts_pred, s), -jnp.inf)
    row = jnp.where(data.is_detected, log_det, log_null)
    return jnp.sum(jnp.where(data.is_valid, row, 0.0))


def loglike_report(alpha_pred, dMag_pred, data):
    """Report log likelihood: detections and nulls in the reporting-law form.

    A null contributes ``log Phi((snr_thresh * limit - N) / s)`` and a detection
    the counts density ``Normal(counts; N, s^2)`` on ``counts >= snr_thresh *
    limit`` (zero density below), with ``N`` the expected planet counts and
    ``s^2 = limit^2 + N``, which includes the planet's shot noise. ``N`` follows
    from the predicted delta magnitude and the detection limit at the predicted
    separation. With ``limit`` large against ``N`` this tends to
    :func:`~photomancy.orbit.likelihoods.loglike_null`.

    Args:
        alpha_pred: Predicted separations (arcsec). Shape ``(N,)``.
        dMag_pred: Predicted delta magnitudes. Shape ``(N,)``.
        data: A :class:`ReportData`.

    Returns:
        Scalar log likelihood; padded rows contribute zero.
    """
    dmag0, limit = _interp_report(alpha_pred, data)
    ln_contrast = -0.4 * _LN10 * dMag_pred
    counts_pred = jnp.exp(_log_counts_pred(dmag0, limit, ln_contrast, data))
    return _report_sum(counts_pred, limit, data)


def _unit_counts(target, limit, dmag0, ln_ratio, data):
    """Log expected counts at the middle of the scale range, shape ``(N,)``.

    ``ln_ratio`` is ``ln(phase / r^2)`` (the log contrast at unit ``Lambda``), so
    the counts at a scale offset ``t`` from the middle are ``exp(log_g + t)``.
    """
    log_limit = jnp.log(data.snr_thresh * jnp.maximum(limit, _TINY))
    return log_limit + 0.4 * _LN10 * dmag0 + ln_ratio + target.ln_lam_ref


def _counts_at(log_g, limit, t, data):
    """Expected counts at scale offset ``t`` with the exponent clipped as the report."""
    cap = jnp.log(data.snr_thresh * jnp.maximum(limit, _TINY)) + 200.0 * _LN10
    return jnp.exp(jnp.minimum(log_g + t, cap))


def _scale_fit(limit, log_g, data):
    """Detections as a Gaussian in the log scale offset ``t``.

    With the detection variance fixed at ``limit^2 + counts`` (the measured
    counts), the detected rows are linear in ``exp(t)``: their log likelihood is
    Gaussian in ``t`` about ``t_hat`` with standard deviation ``sd`` up to the
    constant ``-chi2 / 2`` and the normalization ``log_norm``. ``has`` is False
    when no valid row is a detection (no peak).

    Returns:
        ``(has, t_hat, sd, chi2, log_norm, on_region, g)``.
    """
    det = data.is_valid & data.is_detected
    counts = jnp.where(det, data.counts, 0.0)
    var = limit**2 + counts
    g = jnp.exp(jnp.clip(log_g, -300.0, 300.0))
    s_gg = jnp.sum(jnp.where(det, g**2 / var, 0.0))
    s_cg = jnp.sum(jnp.where(det, counts * g / var, 0.0))
    s_cc = jnp.sum(jnp.where(det, counts**2 / var, 0.0))
    has = (s_cg > 0.0) & (s_gg > 0.0)
    safe_gg = jnp.where(has, s_gg, 1.0)
    safe_cg = jnp.where(has, s_cg, 1.0)
    t_hat = jnp.log(safe_cg / safe_gg)
    sd = jnp.sqrt(safe_gg) / safe_cg
    chi2 = s_cc - jnp.where(has, s_cg**2 / safe_gg, 0.0)
    log_norm = jnp.sum(jnp.where(det, -0.5 * jnp.log(2.0 * jnp.pi * var), 0.0))
    on_region = jnp.all(~det | (data.counts >= data.snr_thresh * limit))
    return has, t_hat, sd, chi2, log_norm, on_region, g


def _has_detection(data):
    """True when any valid row of the report is a detection."""
    return jnp.any(data.is_valid & data.is_detected)


def _scale_marginal(target, limit, log_g, data):
    """Log report likelihood of one orbit with the scale integrated out.

    The scale is uniform in its logarithm over ``[ln_lam_lo, ln_lam_hi]``. With a
    detection the integral is closed form: the Gaussian of :func:`_scale_fit`
    times the share of its mass inside the range, the nulls taken at its peak
    (they change slowly over the peak's width, about ``1 / SNR``). Without a
    detection there is no peak: Gauss-Legendre quadrature over the range.

    This closed form and the Gaussian draw of :func:`_scale_draw` are two halves
    of one approximation, the joint of the scale and the nulls with the nulls
    frozen at the peak, so their errors cancel to second order. Never make one
    half exact alone: an exact draw on these weights cost 16 percent of the
    detection probability at three records at a signal to noise of 5 to 6.
    """
    half = target.ln_lam_half

    def closed():
        has, t_hat, sd, chi2, log_norm, on_region, g = _scale_fit(limit, log_g, data)
        inside = ndtr((half - t_hat) / sd) - ndtr((-half - t_hat) / sd)
        counts = jnp.exp(t_hat) * g
        null = log_ndtr(
            (data.snr_thresh * limit - counts) / jnp.sqrt(limit**2 + counts)
        )
        nulls = jnp.sum(jnp.where(data.is_valid & ~data.is_detected, null, 0.0))
        val = (
            log_norm
            - 0.5 * chi2
            + 0.5 * jnp.log(2.0 * jnp.pi)
            + jnp.log(sd)
            + jnp.log(jnp.maximum(inside, _TINY))
            - jnp.log(2.0 * half)
            + nulls
        )
        return jnp.where(has & on_region & (inside > 0.0), val, -jnp.inf)

    def node(i, acc):
        ll = _report_sum(_counts_at(log_g, limit, half * _GL_X[i], data), limit, data)
        return jnp.logaddexp(acc, jnp.log(_GL_W[i]) + ll)

    # The nodes run in a loop whose trip count is zero with a detection, so a lane
    # batch (which turns a cond on the report into a select that runs both
    # branches) still skips them; a cond would pay for all of them on every draw.
    n_nodes = jnp.where(_has_detection(data), 0, _GL_X.shape[0])
    total = jax.lax.fori_loop(0, n_nodes, node, jnp.asarray(-jnp.inf))
    return jnp.where(_has_detection(data), closed(), total - jnp.log(2.0))


def _scale_draw(key, target, limit, log_g, data):
    """Draw the log scale of one orbit from its conditional given the report.

    With a detection: the Gaussian of :func:`_scale_fit` truncated to the range.
    Without: a draw from a fine grid of the null likelihood over the range
    (uniform within the chosen cell).

    The Gaussian draw and the closed-form weights of :func:`_scale_marginal` are
    two halves of one approximation, the joint of the scale and the nulls with
    the nulls frozen at the peak, so their errors cancel to second order. Never
    make one half exact alone: an exact draw on the present weights cost 16
    percent of the detection probability at three records at a signal to noise
    of 5 to 6.
    """
    half = target.ln_lam_half
    k_cell, k_in, k_norm = jax.random.split(key, 3)

    width = 2.0 * half / _N_CELLS
    mid = -half + width * (jnp.arange(_N_CELLS) + 0.5)
    noise = jax.random.gumbel(k_cell, (_N_CELLS,))

    def best(i, carry):
        top, pick, top_flat, pick_flat = carry
        ll = _report_sum(_counts_at(log_g, limit, mid[i], data), limit, data)
        score = ll + noise[i]
        take = score > top
        take_flat = noise[i] > top_flat
        return (
            jnp.where(take, score, top),
            jnp.where(take, i, pick),
            jnp.where(take_flat, noise[i], top_flat),
            jnp.where(take_flat, i, pick_flat),
        )

    def gaussian():
        _, t_hat, sd, _, _, _, _ = _scale_fit(limit, log_g, data)
        z = jax.random.truncated_normal(
            k_norm, (-half - t_hat) / sd, (half - t_hat) / sd, ()
        )
        return jnp.clip(t_hat + sd * z, -half, half)

    # Without a detection the cells are scanned in a loop (zero trips with one),
    # so a lane batch that turns the cond into a select does not pay for them.
    n_cells = jnp.where(_has_detection(data), 0, _N_CELLS)
    init = (jnp.asarray(-jnp.inf), jnp.asarray(0)) * 2
    top, pick, _, pick_flat = jax.lax.fori_loop(0, n_cells, best, init)
    # No cell with a finite likelihood: the cells are equally likely.
    cell = jnp.where(jnp.isfinite(top), pick, pick_flat)
    cells = mid[cell] + width * (jax.random.uniform(k_in) - 0.5)
    t = jnp.where(_has_detection(data), gaussian(), cells)
    return target.ln_lam_ref + jnp.where(jnp.isfinite(t), t, 0.0)


def _report_xy(target, theta, data):
    """Orbit-plane ``(cos E - e, sqrt(1 - e^2) sin E)`` at the report epochs."""
    e, ln_period, tau = theta[0], theta[1], theta[2]
    m = target.model
    mean_anomaly = two_pi * ((data.epochs - m.t_ref) / jnp.exp(ln_period) - tau)
    sin_e, cos_e = diff_solve_trig(jnp.mod(mean_anomaly, two_pi), e)
    return cos_e - e, jnp.sqrt(1.0 - e**2) * sin_e


def _ti_geometry(target, theta, xy, data):
    """Noise level ``(N,)`` and unit-scale log counts ``(2, N)`` of a chart point.

    The constants are ``A = a P`` and ``B = a Q`` for orthonormal in-plane
    vectors, so the line-of-sight components follow from ``|A| = |B| = a`` and
    ``A . B = 0`` up to the joint sign that distinguishes the mirror orbits
    (row 0 is the orbit of ``ti_to_elements(theta, 0)``).
    """
    x, y = xy
    east_x, east_y, north_x, north_y = theta[3], theta[4], theta[5], theta[6]
    a2 = ti_semimajor_axis(east_x, east_y, north_x, north_y) ** 2
    z_a = jnp.sqrt(jnp.maximum(a2 - east_x**2 - north_x**2, 0.0))
    z_b = jnp.sqrt(jnp.maximum(a2 - east_y**2 - north_y**2, 0.0))
    z_b = jnp.where(east_x * east_y + north_x * north_y > 0.0, -z_b, z_b)
    east = east_x * x + east_y * y
    north = north_x * x + north_y * y
    z0 = z_a * x + z_b * y
    alpha = jnp.hypot(east, north) / target.dist_pc
    r2 = east**2 + north**2 + z0**2
    r = jnp.sqrt(r2)
    dmag0, limit = _interp_report(alpha, data)

    def branch(sign):
        cos_beta = jnp.clip(sign * z0 / jnp.maximum(r, 1e-30), -1.0, 1.0)
        sin_beta = jnp.sqrt(jnp.maximum(1.0 - cos_beta**2, 0.0))
        phase = lambert_phase_exact(cos_beta, sin_beta)
        ln_ratio = jnp.log(jnp.maximum(phase, _TINY)) - jnp.log(jnp.maximum(r2, 1e-60))
        return _unit_counts(target, limit, dmag0, ln_ratio, data)

    # The sign of the line-of-sight component of the first constant, ``a sin(omega)
    # sin(i)``, is that of ``sin(omega)`` of the orbit ``ti_to_elements(theta, 0)``
    # (its mirror has the opposite one); that fixes which sign is row 0.
    sigma = jnp.arctan2(east_x - north_y, north_x + east_y)
    phi = jnp.arctan2(east_x + north_y, north_x - east_y)
    s0 = jnp.where(jnp.sin((sigma - phi) / 2.0) >= 0.0, 1.0, -1.0)
    return limit, jnp.stack([branch(s0), branch(-s0)])


def _ti_report_terms(target, theta, xy, data):
    """Log report likelihood of the two mirror orbits of a chart point, ``(2,)``."""
    limit, log_g = _ti_geometry(target, theta, xy, data)
    return jax.vmap(lambda lg: _scale_marginal(target, limit, lg, data))(log_g)


def _element_geometry(target, el, data):
    """Noise level ``(N,)`` and unit-scale log counts ``(N,)`` of physical elements."""
    period = jnp.exp(el[0])
    mass = jnp.exp(el[6]) * Msun2kg
    a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
    alpha, d_mag = predict_photometry(
        data.epochs,
        a,
        el[1],
        el[2],
        el[3],
        jnp.cos(el[4]),
        jnp.sin(el[4]),
        target.model.t_ref + el[5] * period,
        mass,
        1.0,
        target.dist_pc,
    )
    dmag0, limit = _interp_report(alpha, data)
    return limit, _unit_counts(target, limit, dmag0, -0.4 * _LN10 * d_mag, data)


def _element_terms(target, el, data):
    """Log report likelihood of physical elements ``(n_el,)``, scale integrated out."""
    limit, log_g = _element_geometry(target, el, data)
    return _scale_marginal(target, limit, log_g, data)


def report_log_weights(target, elements, report):
    """Log of the scale-integrated report likelihood of orbit elements.

    The photometric scale is integrated over the target's range (uniform in its
    logarithm), so a draw of the orbit alone is weighted by the marginal. Use
    :func:`draw_report_scale` to give the resampled orbits their scale.

    Args:
        target: A :class:`PhysicalPriorTarget` (scale range and distance).
        elements: Physical elements, shape ``(n, 8)`` (``ln_lam`` is not read).
        report: A :class:`ReportData`.

    Returns:
        Shape ``(n,)``; ``-inf`` where the likelihood is zero or not finite.
    """
    out = jax.lax.map(
        lambda el: _element_terms(target, el, report),
        elements,
        batch_size=min(_MAP_BATCH, elements.shape[0]),
    )
    return jnp.where(jnp.isfinite(out), out, -jnp.inf)


def draw_report_scale(target, elements, report, key):
    """Draw each orbit's photometric scale from its conditional given the report.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        elements: Physical elements, shape ``(n, 8)``.
        report: A :class:`ReportData`.
        key: PRNG key.

    Returns:
        The elements with ``ln_lam`` (column 7) drawn per orbit.
    """
    keys = jax.random.split(key, elements.shape[0])

    def one(args):
        el, k = args
        limit, log_g = _element_geometry(target, el, report)
        return _scale_draw(k, target, limit, log_g, report)

    ln_lam = jax.lax.map(one, (elements, keys), batch_size=min(_MAP_BATCH, len(keys)))
    return elements.at[:, 7].set(ln_lam)


def _gumbel_pick(key, log_p):
    """Index drawn from the unnormalized log probabilities ``log_p`` (any shape)."""
    flat = jnp.ravel(log_p)
    safe = jnp.where(jnp.isfinite(jnp.max(flat)), flat, 0.0)
    return jnp.argmax(safe + jax.random.gumbel(key, flat.shape))


def _draw_branch_scale(key, target, theta, report):
    """Mirror orbit and log scale of a chart point from their conditional."""
    k_branch, k_scale = jax.random.split(key)
    limit, log_g = _ti_geometry(
        target, theta, _report_xy(target, theta, report), report
    )
    terms = jax.vmap(lambda lg: _scale_marginal(target, limit, lg, report))(log_g)
    branch = _gumbel_pick(k_branch, terms)
    return branch, _scale_draw(k_scale, target, limit, log_g[branch], report)


def ti_to_elements(theta, branch):
    """Physical elements of a Thiele-Innes chart point.

    The constants fix ``(a, cos i)`` and the sums ``omega + W`` and ``W - omega``
    modulo ``2 pi``, so ``(omega, W)`` is known up to a joint shift by ``pi``:
    the two mirror orbits, which share their sky track.

    Args:
        theta: Chart point ``(e, lnP, tau, east_X, east_Y, north_X, north_Y)``.
        branch: 0 or 1, the mirror orbit (adds ``pi`` to ``omega`` and ``W``).

    Returns:
        Elements in the order of ``ELEMENT_NAMES``, with ``ln_lam`` NaN.
    """
    east_x, east_y, north_x, north_y = theta[3], theta[4], theta[5], theta[6]
    r_plus = jnp.hypot(north_x + east_y, east_x - north_y)
    r_minus = jnp.hypot(north_x - east_y, east_x + north_y)
    sigma = jnp.arctan2(east_x - north_y, north_x + east_y)
    phi = jnp.arctan2(east_x + north_y, north_x - east_y)
    shift = jnp.pi * branch
    node = jnp.mod((sigma + phi) / 2.0 + shift, two_pi)
    omega = jnp.mod((sigma - phi) / 2.0 + shift, two_pi)
    a = ti_semimajor_axis(east_x, east_y, north_x, north_y)
    cos_i = (r_plus - r_minus) / (r_plus + r_minus)
    ln_mu = 2.0 * jnp.log(two_pi) - 2.0 * theta[1] + 3.0 * jnp.log(a)
    return jnp.stack(
        [
            theta[1],
            theta[0],
            cos_i,
            node,
            omega,
            theta[2],
            ln_mu - jnp.log(MU_SUN),
            jnp.nan * jnp.ones(()),
        ]
    )


def _kish(log_w):
    """Kish effective size of log weights (0 when none is finite)."""
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    return jnp.where(total > 0.0, total**2 / jnp.sum(w**2), 0.0)


def _max_weight(log_w):
    """Largest normalized weight of log weights (0 when none is finite)."""
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    return jnp.where(total > 0.0, 1.0 / jnp.where(total > 0.0, total, 1.0), 0.0)


def _status(ess, log_w, ess_floor):
    """Refit status from the last round."""
    no_support = ~jnp.isfinite(jnp.max(log_w))
    return jnp.where(
        no_support,
        UpdaterStatus.NO_SUPPORT,
        jnp.where(ess < ess_floor, UpdaterStatus.LOW_ESS, UpdaterStatus.OK),
    ).astype(jnp.int32)


def _systematic_indices(key, log_weights, n):
    """Systematic resampling indices for ``n`` picks from normalized log weights.

    A copy of the updater's helper, which clips to ``n - 1`` and so is correct
    only when the picks are as many as the weights.
    """
    cdf = jnp.cumsum(jnp.exp(log_weights))
    cdf = cdf / cdf[-1]
    u = (jax.random.uniform(key) + jnp.arange(n)) / n
    return jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, cdf.shape[0] - 1)


def _resample_elements(key, log_w, n_particles, to_elements, draw=None):
    """Systematic resample to equal weights, mirror branch drawn per particle.

    ``draw(idx, key)`` replaces the equal-probability mirror branch by the
    caller's own draw of the branch (and the photometric node) per particle.
    """
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    w = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 1.0 / w.size)
    k_idx, k_branch = jax.random.split(key)
    idx = _systematic_indices(k_idx, jnp.log(jnp.maximum(w, _TINY)), n_particles)
    if draw is not None:
        return draw(idx, k_branch)
    branch = jax.random.bernoulli(k_branch, 0.5, (n_particles,)).astype(float)
    return to_elements(idx, branch)


def _prior_box(target):
    """Lower and upper bounds of ``(e, ln P, tau)`` under the prior."""
    m = target.model
    lo = jnp.stack([jnp.zeros(()), m.lnP_lo, jnp.zeros(())])
    hi = jnp.stack([m.e_max, m.lnP_hi, jnp.ones(())])
    return lo, hi


def _kernel_fit(key, u, log_w, lo, hi, n_centers):
    """Resampled centers and the Cholesky factor of the kernel covariance.

    The covariance is the weighted covariance of ``u`` (``tau`` unwrapped around
    its circular mean) times the squared Silverman bandwidth, plus a floor of
    1e-3 of the box width.
    """
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    w = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 1.0 / w.size)
    angle = two_pi * u[:, 2]
    mean_tau = jnp.arctan2(jnp.sum(w * jnp.sin(angle)), jnp.sum(w * jnp.cos(angle)))
    mean_tau = mean_tau / two_pi
    x = u.at[:, 2].set(mean_tau + jnp.mod(u[:, 2] - mean_tau + 0.5, 1.0) - 0.5)
    mean = w @ x
    centered = x - mean
    cov = (centered * w[:, None]).T @ centered / (1.0 - jnp.sum(w**2))
    h = 0.969 * jnp.maximum(1.0 / jnp.sum(w**2), 4.0) ** (-1.0 / 7.0)
    floor = 1e-3 * (hi - lo)
    cov = h**2 * cov + jnp.diag(floor**2)
    grid = (jax.random.uniform(key) + jnp.arange(n_centers)) / n_centers
    idx = jnp.minimum(jnp.searchsorted(jnp.cumsum(w), grid), w.size - 1)
    return u[idx], jnp.linalg.cholesky(cov)


def _log_kernel_mixture(u, centers, chol):
    """Log density at each row of ``u`` of the equal-weight Gaussian kernel mixture.

    ``tau`` is periodic: the kernel images shifted by -2 to 2 are summed.
    Distances are taken in the whitened frame of the kernel covariance.
    """
    solve = jax.scipy.linalg.solve_triangular
    centers_w = solve(chol, centers.T, lower=True).T
    shift = solve(chol, jnp.array([0.0, 0.0, 1.0]), lower=True)
    log_norm = -1.5 * jnp.log(two_pi) - jnp.sum(jnp.log(jnp.diag(chol)))
    log_n = jnp.log(centers.shape[0])

    def one(block):
        y = solve(chol, block.T, lower=True).T
        d = y[:, None, :] - centers_w[None]
        dt = d @ shift
        base = jnp.sum(d**2, axis=-1)
        sq = (
            base[..., None]
            + 2.0 * dt[..., None] * _TAU_IMAGES
            + jnp.sum(shift**2) * _TAU_IMAGES**2
        )
        return logsumexp(-0.5 * sq, axis=(1, 2)) - log_n + log_norm

    blocks = u.reshape(-1, min(_DENSITY_BATCH, u.shape[0]), 3)
    return jax.lax.map(one, blocks).reshape(-1)


def _propose(key, centers, chol, lo, hi, n, defensive):
    """Mixture draws of ``(e, ln P, tau)`` and their log proposal density."""
    k = jax.random.split(key, 4)
    log_box = -jnp.sum(jnp.log(hi - lo))
    prior = lo + (hi - lo) * jax.random.uniform(k[0], (n, 3))
    pick = jax.random.randint(k[1], (n,), 0, centers.shape[0])
    kern = centers[pick] + jax.random.normal(k[2], (n, 3)) @ chol.T
    kern = kern.at[:, 2].set(jnp.mod(kern[:, 2], 1.0))
    u = jnp.where(jax.random.uniform(k[3], (n, 1)) < defensive, prior, kern)
    inside = jnp.all((u > lo) & (u < hi), axis=1)
    log_q = jnp.logaddexp(
        jnp.log(defensive) + jnp.where(inside, log_box, -jnp.inf),
        jnp.log1p(-defensive) + _log_kernel_mixture(u, centers, chol),
    )
    return u, log_q


def _draw_constants(target, records, ui, ki, n_constants):
    """Log marginal, ``n_constants`` chart points at ``ui`` and their prior factors.

    The constants come from their conditional Gaussian given the records; the
    returned ``rest`` is the physical density over the reference Gaussian (``-inf``
    outside the support). ``k_keep`` is the key a caller may use to pick a draw.
    """
    k_draw, k_keep = jax.random.split(ki)
    theta = jnp.concatenate([ui, jnp.zeros(4)])
    mean, chol, log_marginal, _, _ = _linear_gaussian(
        target, theta, records, records.is_valid.astype(float)
    )
    z = jax.random.normal(k_draw, (n_constants, 4))
    c = mean + jax.scipy.linalg.solve_triangular(chol.T, z.T, lower=False).T
    thetas = jnp.concatenate([jnp.broadcast_to(ui, (n_constants, 3)), c], axis=1)
    rest = jax.vmap(lambda th: _rest_log_prior(target, th))(thetas)
    return log_marginal, thetas, jnp.where(jnp.isfinite(rest), rest, -jnp.inf), k_keep


def _collapsed_weights(target, records, u, key, n_constants):
    """Constants from their conditional Gaussian and the log weight before proposal.

    Each point of ``u`` draws ``n_constants`` sets of constants. Its weight holds
    the mean over the draws of ``p_rest`` (the physical density over the
    reference Gaussian, zero outside the mass window) and one draw is kept with
    probability proportional to its ``p_rest``. For any function ``h`` the
    expectation of ``mean(p_rest) h(kept)`` is that of ``p_rest h`` under the
    conditional Gaussian, so the kept point is properly weighted for the joint
    target at any ``n_constants`` and the mean weight is still the evidence. The
    draws cost no further Kepler solve.

    Returns ``(theta, log_w)`` with ``log_w`` the log marginal likelihood of the
    constants plus the log mean ``p_rest``; the caller subtracts the log
    proposal density of ``u``.
    """

    def one(args):
        ui, ki = args
        log_marginal, thetas, rest, k_keep = _draw_constants(
            target, records, ui, ki, n_constants
        )
        keep = jnp.argmax(rest + jax.random.gumbel(k_keep, (n_constants,)))
        lw = log_marginal + logsumexp(rest) - jnp.log(n_constants)
        return thetas[keep], jnp.where(jnp.isfinite(lw), lw, -jnp.inf)

    keys = jax.random.split(key, u.shape[0])
    return jax.lax.map(one, (u, keys), batch_size=min(_MAP_BATCH, u.shape[0]))


def _evidence_constant(target, records):
    """Constants of the collapsed weight that make its mean the marginal likelihood.

    The data normalization with the zero-constants prediction, the
    ``(2 pi)^(p/2)`` of the conditional Gaussian and the prior constant.
    """
    theta0 = jnp.array([0.1, 0.5 * (target.model.lnP_lo + target.model.lnP_hi), 0.5])
    theta0 = jnp.concatenate([theta0, jnp.zeros(4)])
    return (
        target.log_likelihood(theta0, records)
        + 2.0 * jnp.log(two_pi)
        + target.log_prior_constant()
    )


def _score_terms(target, records, report, u, key, n_constants):
    """Per point the log marginal, per draw the prior factor and the log report.

    The report score of a draw is the log of its likelihood with the scale
    integrated out and the two mirror orbits averaged. Keeping the pieces apart
    lets a round be re-weighted at any power of the report without a second pass.
    """

    def one(args):
        ui, ki = args
        log_marginal, thetas, rest, _ = _draw_constants(
            target, records, ui, ki, n_constants
        )
        xy = _report_xy(target, ui, report)
        lr = jax.vmap(
            lambda th: (
                logsumexp(_ti_report_terms(target, th, xy, report)) - jnp.log(2.0)
            )
        )(thetas)
        return log_marginal, rest, jnp.where(jnp.isfinite(lr), lr, -jnp.inf)

    keys = jax.random.split(key, u.shape[0])
    return jax.lax.map(one, (u, keys), batch_size=min(_MAP_BATCH, u.shape[0]))


def _tempered(terms, log_q, beta):
    """Log weights of scored points with the report likelihood raised to ``beta``."""
    log_marginal, rest, lr = terms
    total = jnp.where(beta > 0.0, rest + beta * lr, rest)
    lw = log_marginal + logsumexp(total, axis=1) - jnp.log(rest.shape[1]) - log_q
    return jnp.where(jnp.isfinite(lw), lw, -jnp.inf)


def _pick_beta(terms, log_q, beta_min):
    """Largest report power at least ``beta_min`` that keeps the Kish size afloat.

    The floor is ``_LADDER_FLOOR``, lowered to half the Kish size at ``beta_min``
    (never under ``_LADDER_FLOOR_MIN``) so a round whose points cannot reach it at
    the previous power still moves. Power one is taken when it holds the floor;
    otherwise bisection on the Kish size.
    """
    full = _kish(_tempered(terms, log_q, 1.0))
    here = _kish(_tempered(terms, log_q, beta_min))
    floor = jnp.clip(_LADDER_SHRINK * here, _LADDER_FLOOR_MIN, _LADDER_FLOOR)

    def bisect(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        ok = _kish(_tempered(terms, log_q, mid)) >= floor
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

    lo, _ = jax.lax.fori_loop(0, _BISECT_STEPS, bisect, (beta_min, 1.0))
    return jnp.where(full >= floor, 1.0, jnp.where(here <= floor, beta_min, lo))


def _local_fit(key, u, log_w, lo, hi, n_centers):
    """Resampled kernel centers, each with the covariance of the points near it.

    ``_N_SUPPORT`` points are resampled by weight. A center's covariance is the
    second moment about it of its nearest support points (distance in the frame
    whitened by the weighted covariance, ``tau`` wrapped), times ``_K_SCALE``
    squared, plus the floor of :func:`_kernel_fit`. The neighbor count is
    ``_K_DISTINCT * _N_SUPPORT / Kish`` between ``_K_NEAR`` and ``_K_NEAR_MAX``,
    so that about ``_K_DISTINCT`` distinct points sit under each kernel as the
    Kish size falls. Returns ``(centers, chols)`` with ``chols`` ``(n_centers, 3, 3)``.
    """
    peak = jnp.max(log_w)
    w = jnp.exp(log_w - jnp.where(jnp.isfinite(peak), peak, 0.0))
    total = jnp.sum(w)
    w = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 1.0 / w.size)
    k_support, k_centers = jax.random.split(key)
    grid = (jax.random.uniform(k_support) + jnp.arange(_N_SUPPORT)) / _N_SUPPORT
    support = u[jnp.minimum(jnp.searchsorted(jnp.cumsum(w), grid), w.size - 1)]
    angle = two_pi * u[:, 2]
    mean_tau = jnp.arctan2(jnp.sum(w * jnp.sin(angle)), jnp.sum(w * jnp.cos(angle)))
    mean_tau = mean_tau / two_pi
    x = u.at[:, 2].set(mean_tau + jnp.mod(u[:, 2] - mean_tau + 0.5, 1.0) - 0.5)
    centered = x - w @ x
    floor = jnp.diag((1e-3 * (hi - lo)) ** 2)
    frame = jnp.linalg.cholesky((centered * w[:, None]).T @ centered + floor)
    centers = support[
        jax.random.choice(k_centers, _N_SUPPORT, (n_centers,), replace=False)
    ]
    k_use = jnp.clip(_K_DISTINCT * _N_SUPPORT * jnp.sum(w**2), _K_NEAR, _K_NEAR_MAX)
    use = (jnp.arange(_K_NEAR_MAX) < k_use).astype(float)

    def one(center):
        d = support - center
        d = d.at[:, 2].set(jnp.mod(d[:, 2] + 0.5, 1.0) - 0.5)
        y = jax.scipy.linalg.solve_triangular(frame, d.T, lower=True).T
        _, near = jax.lax.top_k(-jnp.sum(y**2, axis=1), _K_NEAR_MAX)
        dn = d[near] * use[:, None]
        return jnp.linalg.cholesky(_K_SCALE**2 * dn.T @ dn / jnp.sum(use) + floor)

    return centers, jax.vmap(one)(centers)


def _local_log_mixture(u, centers, chols):
    """Log density of the equal-weight mixture with one covariance per center.

    ``tau`` is periodic: the kernel images shifted by -2 to 2 are summed.
    """
    eye = jnp.eye(3)
    inv = jax.vmap(lambda c: jax.scipy.linalg.solve_triangular(c, eye, lower=True))(
        chols
    )
    log_det = jnp.sum(jnp.log(jnp.diagonal(chols, axis1=1, axis2=2)), axis=1)
    shift = inv[:, :, 2]
    shift_sq = jnp.sum(shift**2, axis=1)
    log_norm = -1.5 * jnp.log(two_pi) - jnp.log(centers.shape[0])

    def one(block):
        y = jnp.einsum("nij,bnj->bni", inv, block[:, None, :] - centers[None])
        dt = jnp.einsum("bni,ni->bn", y, shift)
        sq = (
            jnp.sum(y**2, axis=-1)[..., None]
            + 2.0 * dt[..., None] * _TAU_IMAGES
            + shift_sq[None, :, None] * _TAU_IMAGES**2
        )
        return logsumexp(-0.5 * sq - log_det[None, :, None], axis=(1, 2)) + log_norm

    blocks = u.reshape(-1, min(_DENSITY_BATCH, u.shape[0]), 3)
    return jax.lax.map(one, blocks).reshape(-1)


def _local_propose(key, centers, chols, lo, hi, n, defensive):
    """:func:`_propose` for the mixture with one covariance per center."""
    k = jax.random.split(key, 4)
    log_box = -jnp.sum(jnp.log(hi - lo))
    prior = lo + (hi - lo) * jax.random.uniform(k[0], (n, 3))
    pick = jax.random.randint(k[1], (n,), 0, centers.shape[0])
    step = jnp.einsum("nij,nj->ni", chols[pick], jax.random.normal(k[2], (n, 3)))
    kern = centers[pick] + step
    kern = kern.at[:, 2].set(jnp.mod(kern[:, 2], 1.0))
    u = jnp.where(jax.random.uniform(k[3], (n, 1)) < defensive, prior, kern)
    inside = jnp.all((u > lo) & (u < hi), axis=1)
    log_q = jnp.logaddexp(
        jnp.log(defensive) + jnp.where(inside, log_box, -jnp.inf),
        jnp.log1p(-defensive) + _local_log_mixture(u, centers, chols),
    )
    return u, log_q


def _ladder_rounds(
    target,
    records,
    report,
    key,
    n_round,
    n_last,
    n_centers,
    defensive,
    n_constants,
    ess_floor,
    max_rounds,
):
    """Rounds with the report power climbing from zero to one, then the last round.

    Every round but the last scores its points once and takes the largest power
    of the report likelihood, not below the previous round's, at which its Kish
    size holds a floor (:func:`_pick_beta`); the next proposal is fit to those
    tempered weights. Round one is uniform over the prior box and usually runs at
    power zero. The last round, of ``n_last`` points at power one, follows the
    first round that reaches power one with a Kish size of ``_LADDER_FLOOR``, the
    second round at power one, or round ``max_rounds - 1``. Its weights are the
    plain importance weights, so the ladder changes the proposal only. A last
    round under ``ess_floor`` is repeated, with the proposal fit to it, up to
    ``_RETRIES`` times and the final round is returned. The returned round is
    selected on its own Kish size, so the evidence of a repeated refit carries
    that selection.

    Returns:
        ``(u, log_q, terms, k_score, log_w, n_rounds)`` of the returned round.
    """
    lo, hi = _prior_box(target)
    k0, k_score0 = jax.random.split(jax.random.fold_in(key, 0))
    u0 = lo + (hi - lo) * jax.random.uniform(k0, (n_round, 3))
    log_q0 = jnp.full(n_round, -jnp.sum(jnp.log(hi - lo)))
    terms0 = _score_terms(target, records, report, u0, k_score0, n_constants)
    beta0 = _pick_beta(terms0, log_q0, 0.0)
    ess0 = _kish(_tempered(terms0, log_q0, beta0))

    def stop(state):
        r, _, _, _, beta, ess, prev = state
        return (
            (r >= max_rounds - 2)
            | ((beta >= 1.0) & (ess >= _LADDER_FLOOR))
            | ((r >= 1) & (prev >= 1.0))
        )

    def propose(state, k_r, size):
        _, u, log_q, terms, beta, _, _ = state
        k_fit, k_draw, k_next = jax.random.split(k_r, 3)
        log_w = _tempered(terms, log_q, beta)
        centers, chols = _local_fit(k_fit, u, log_w, lo, hi, n_centers)
        new_u, new_log_q = _local_propose(
            k_draw, centers, chols, lo, hi, size, defensive
        )
        return new_u, new_log_q, k_next

    def body(state):
        r, _, _, _, beta, _, _ = state
        u, log_q, k_next = propose(state, jax.random.fold_in(key, 3 * r + 3), n_round)
        terms = _score_terms(target, records, report, u, k_next, n_constants)
        new_beta = _pick_beta(terms, log_q, beta)
        ess = _kish(_tempered(terms, log_q, new_beta))
        return r + 1, u, log_q, terms, new_beta, ess, beta

    state = (jnp.asarray(0), u0, log_q0, terms0, beta0, ess0, jnp.zeros(()))
    state = jax.lax.while_loop(lambda s: ~stop(s), body, state)
    r = state[0]
    u, log_q, k_score = propose(state, jax.random.fold_in(key, 3 * r + 3), n_last)
    k_score, _ = jax.random.split(k_score)
    terms = _score_terms(target, records, report, u, k_score, n_constants)
    log_w = _tempered(terms, log_q, 1.0)

    def low(carry):
        j, *_, log_w = carry
        return (j < _RETRIES) & (_kish(log_w) < ess_floor)

    def retry(carry):
        j, u, _, _, _, log_w = carry
        k_fit, k_draw, k_next = jax.random.split(jax.random.fold_in(key, 1000 + j), 3)
        centers, chols = _local_fit(k_fit, u, log_w, lo, hi, n_centers)
        u, log_q = _local_propose(k_draw, centers, chols, lo, hi, n_last, defensive)
        terms = _score_terms(target, records, report, u, k_next, n_constants)
        return j + 1, u, log_q, terms, k_next, _tempered(terms, log_q, 1.0)

    carry = (jnp.asarray(0), u, log_q, terms, k_score, log_w)
    j, u, log_q, terms, k_score, log_w = jax.lax.while_loop(low, retry, carry)
    return u, log_q, terms, k_score, log_w, r + 2 + j


@eqx.filter_jit
def collapsed_refit(
    target,
    records,
    key,
    *,
    n_particles,
    n_round=2**16,
    n_rounds=3,
    n_last=None,
    n_centers=256,
    defensive=0.1,
    n_constants=8,
    ess_floor=400.0,
    report=None,
    max_rounds=8,
):
    """Collapsed importance refit with adaptive rounds (any number of records).

    Round one draws ``(e, ln P, tau)`` from the prior box; each later round
    from a mixture of the box (fraction ``defensive``) and Gaussian kernels on
    ``n_centers`` points resampled from the previous round. The four constants
    always come from their conditional Gaussian. Every round is an exact
    importance sample, so a poor proposal costs effective size and never
    correctness; only the last round is returned.

    With a ``report`` the rounds follow :func:`_ladder_rounds`: the report
    likelihood enters at a power that climbs from zero to one across the rounds,
    each kernel center carries its own covariance, and a last round under
    ``ess_floor`` is repeated up to twice. Without a report the rounds are the
    plain ones, ``n_rounds`` of them, with one shared kernel covariance.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer with at least one valid record.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        n_round: Proposals per round (static).
        n_rounds: Rounds without a report (static); 1 draws from the prior box
            only.
        n_last: Proposals of the last round (static); default ``n_round``. The
            Kish size is linear in it for a fixed proposal.
        n_centers: Kernels of the adaptive proposal (static); with a report at
            most 4096, the support points they are drawn from.
        defensive: Fraction of an adaptive round drawn from the prior box.
        n_constants: Sets of constants drawn at each proposal (static). More
            than one integrates part of the mass window and of the physical
            density over the constants, the main loss of size where the
            records span more than about one period.
        ess_floor: Kish size under which the status is ``LOW_ESS``; with a report
            also the size under which the last round is repeated.
        report: A :class:`ReportData` or None. With a report the weight carries
            its likelihood (photometric scale integrated out), the returned
            elements carry a mirror orbit and a photometric scale drawn from
            their conditional, and the evidence includes it.
        max_rounds: Most rounds of the report ladder (static); the last is one
            of them. At least 2.

    Returns:
        A :class:`RefitCloud`.
    """
    n_last = n_round if n_last is None else n_last
    for n in (n_round, n_last):
        if n > _DENSITY_BATCH and n % _DENSITY_BATCH:
            raise ValueError(
                f"proposals per round ({n}) must be a multiple of {_DENSITY_BATCH}"
            )
    if report is not None:
        if max_rounds < 2:
            raise ValueError(f"max_rounds ({max_rounds}) must be at least 2")
        if n_centers > _N_SUPPORT:
            raise ValueError(
                f"n_centers ({n_centers}) cannot exceed the {_N_SUPPORT} support "
                "points they are drawn from"
            )
        return _collapsed_refit_report(
            target,
            records,
            report,
            key,
            n_particles,
            n_round,
            n_last,
            n_centers,
            defensive,
            n_constants,
            ess_floor,
            max_rounds,
        )
    sizes = [n_round] * (n_rounds - 1) + [n_last]
    lo, hi = _prior_box(target)
    keys = jax.random.split(key, 3 * n_rounds + 1)
    u = lo + (hi - lo) * jax.random.uniform(keys[0], (sizes[0], 3))
    log_q = jnp.full(sizes[0], -jnp.sum(jnp.log(hi - lo)))
    log_w = None
    for r in range(n_rounds):
        if r > 0:
            centers, chol = _kernel_fit(keys[3 * r], u, log_w, lo, hi, n_centers)
            u, log_q = _propose(
                keys[3 * r + 1], centers, chol, lo, hi, sizes[r], defensive
            )
        theta, log_marginal = _collapsed_weights(
            target, records, u, keys[3 * r + 2], n_constants
        )
        log_w = jnp.where(jnp.isfinite(log_marginal), log_marginal - log_q, -jnp.inf)
    ess = _kish(log_w)
    log_evidence = (
        logsumexp(log_w) - jnp.log(sizes[-1]) + _evidence_constant(target, records)
    )
    elements = _resample_elements(
        keys[-1],
        log_w,
        n_particles,
        lambda idx, branch: jax.vmap(ti_to_elements)(theta[idx], branch),
    )
    return RefitCloud(
        elements=elements,
        log_evidence=jnp.where(jnp.isfinite(ess) & (ess > 0), log_evidence, -jnp.inf),
        ess=ess,
        status=_status(ess, log_w, ess_floor),
        rounds=jnp.asarray(n_rounds),
        max_weight=_max_weight(log_w),
    )


def _collapsed_refit_report(
    target,
    records,
    report,
    key,
    n_particles,
    n_round,
    n_last,
    n_centers,
    defensive,
    n_constants,
    ess_floor,
    max_rounds,
):
    """The report path of :func:`collapsed_refit`: ladder rounds, then the cloud."""
    k_rounds, k_pick = jax.random.split(key)
    u, _, terms, k_score, log_w, n_done = _ladder_rounds(
        target,
        records,
        report,
        k_rounds,
        n_round,
        n_last,
        n_centers,
        defensive,
        n_constants,
        ess_floor,
        max_rounds,
    )
    ess = _kish(log_w)
    log_evidence = (
        logsumexp(log_w) - jnp.log(n_last) + _evidence_constant(target, records)
    )
    point_keys = jax.random.split(k_score, n_last)
    _, rest, lr = terms

    def draw(idx, k):
        k_keep, k_scale = jax.random.split(k)
        keep = jax.vmap(
            lambda i, kk: jnp.argmax(
                rest[i] + lr[i] + jax.random.gumbel(kk, (n_constants,))
            )
        )(idx, jax.random.split(k_keep, idx.shape[0]))

        def one(i, j, kk):
            thetas = _draw_constants(target, records, u[i], point_keys[i], n_constants)[
                1
            ]
            theta = thetas[j]
            branch, ln_lam = _draw_branch_scale(kk, target, theta, report)
            return ti_to_elements(theta, branch.astype(float)).at[7].set(ln_lam)

        return jax.vmap(one)(idx, keep, jax.random.split(k_scale, idx.shape[0]))

    elements = _resample_elements(k_pick, log_w, n_particles, None, draw)
    return RefitCloud(
        elements=elements,
        log_evidence=jnp.where(jnp.isfinite(ess) & (ess > 0), log_evidence, -jnp.inf),
        ess=ess,
        status=_status(ess, log_w, ess_floor),
        rounds=n_done,
        max_weight=_max_weight(log_w),
    )


@eqx.filter_jit
def scale_rotate_draw(target, records, key, *, n_proposals, report=None):
    """Weighted draws by scale-and-rotate, the mass drawn from its window.

    The first valid record is the anchor: orbit shapes ``(e, cos i, omega,
    M_ref)`` come from their priors, the mass log-uniformly from the window and
    the anchor position from its error kernel; ``(a, W)`` are solved so the orbit
    passes through it, which induces the ``ln P`` and node priors exactly. The
    weight is the likelihood of the other valid records over the squared anchor
    separation, so the draw is exact for the physical prior at any number of
    records but is efficient only for one.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer with at least one valid record.
        key: PRNG key.
        n_proposals: Proposals (static).
        report: A :class:`ReportData` or None. With a report each weight carries
            the scale-integrated report likelihood of its orbit.

    Returns:
        ``(elements, log_w, log_offset)``: elements ``(n, 8)`` of every proposal,
        its log weights (``-inf`` for a proposal outside the support), and the
        constant so that ``logsumexp(log_w) - log(n) - log_offset`` is the log
        evidence.
    """
    m = target.model
    n = n_proposals
    anchor = jnp.argmax(records.is_valid)
    k = jax.random.split(key, 8)
    e = jax.random.beta(k[0], ECC_ALPHA, ECC_BETA, (n,))
    cos_i = jax.random.uniform(k[1], (n,), minval=-1.0, maxval=1.0)
    omega = jax.random.uniform(k[2], (n,), maxval=two_pi)
    m_ref = jax.random.uniform(k[3], (n,), maxval=two_pi)
    ln_mu = jax.random.uniform(
        k[4], (n,), minval=jnp.log(m.mu_lo), maxval=jnp.log(m.mu_hi)
    )
    mu = jnp.exp(ln_mu)
    sx, sy, rho = records.ra_err[anchor], records.dec_err[anchor], records.corr[anchor]
    z1 = jax.random.normal(k[5], (n,))
    z2 = jax.random.normal(k[6], (n,))
    tx = records.ra[anchor] + sx * z1
    ty = records.dec[anchor] + sy * (
        rho * z1 + jnp.sqrt(jnp.maximum(1.0 - rho**2, 0.0)) * z2
    )
    t_anchor = records.times[anchor]
    others = eqx.tree_at(
        lambda r: r.is_valid, records, records.is_valid.at[anchor].set(False)
    )
    log10_lo, log10_hi = m.lnP_lo / jnp.log(10.0), m.lnP_hi / jnp.log(10.0)

    def one(ei, ci, wi, mi, mui, txi, tyi):
        return _scale_rotate_score(
            ei,
            ci,
            jnp.cos(wi),
            jnp.sin(wi),
            mi,
            txi,
            tyi,
            others,
            t_anchor,
            mui,
            mui / G,
            target.dist_pc,
            log10_lo,
            log10_hi,
            m.e_max,
        )

    rows, ll = jax.lax.map(
        lambda args: one(*args),
        (e, cos_i, omega, m_ref, mu, tx, ty),
        batch_size=min(_MAP_BATCH, n),
    )
    log_w = ll - 2.0 * jnp.log(jnp.hypot(tx, ty))
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    log_offset = (
        jnp.log((2.0 / 3.0) * (m.lnP_hi - m.lnP_lo))
        + jnp.log(two_pi)
        + jnp.log(betainc(ECC_ALPHA, ECC_BETA, m.e_max))
    )
    a, ec, ci, w_node, cw, sw, tp = rows.T
    period = two_pi / mean_motion(a, mu)
    elements = jnp.stack(
        [
            jnp.log(period),
            ec,
            ci,
            w_node,
            jnp.mod(jnp.arctan2(sw, cw), two_pi),
            jnp.mod((tp - m.t_ref) / period, 1.0),
            ln_mu - jnp.log(MU_SUN),
            jnp.full(n, jnp.nan),
        ],
        axis=1,
    )
    if report is not None:
        log_w = jnp.where(
            jnp.isfinite(log_w),
            log_w + report_log_weights(target, elements, report),
            -jnp.inf,
        )
    return elements, log_w, log_offset


@eqx.filter_jit
def scale_rotate_refit(
    target,
    records,
    key,
    *,
    n_particles,
    n_proposals=2**17,
    ess_floor=400.0,
    report=None,
):
    """Exact importance refit by scale-and-rotate, the mass drawn from its window.

    See :func:`scale_rotate_draw` for the draw and its weights; the returned
    cloud is ``n_particles`` equal-weight resamples.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer with at least one valid record.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        n_proposals: Proposals (static).
        ess_floor: Kish size under which the status is ``LOW_ESS``.
        report: A :class:`ReportData` or None.

    Returns:
        A :class:`RefitCloud` with ``rounds`` 1.
    """
    k_draw, k_pick = jax.random.split(key)
    elements, log_w, log_offset = scale_rotate_draw(
        target, records, k_draw, n_proposals=n_proposals, report=report
    )
    ess = _kish(log_w)
    log_evidence = logsumexp(log_w) - jnp.log(n_proposals) - log_offset
    draw = None
    if report is not None:

        def draw(idx, k):
            return draw_report_scale(target, elements[idx], report, k)

    picked = _resample_elements(
        k_pick, log_w, n_particles, lambda idx, branch: elements[idx], draw
    )
    return RefitCloud(
        elements=picked,
        log_evidence=jnp.where(ess > 0, log_evidence, -jnp.inf),
        ess=ess,
        status=_status(ess, log_w, ess_floor),
        rounds=jnp.asarray(1),
        max_weight=_max_weight(log_w),
    )


def importance_refit(
    target,
    records,
    report,
    key,
    *,
    n_particles,
    n_round=2**16,
    n_rounds=3,
    n_last=None,
    n_centers=256,
    defensive=0.1,
    n_constants=8,
    ess_floor=400.0,
    max_rounds=8,
):
    """Stateless importance refit of an orbit from its record buffer.

    One valid record runs :func:`scale_rotate_refit` with ``n_round`` proposals;
    two or more run :func:`collapsed_refit`. With concrete records the engine is
    chosen in Python; under a transformation it is a ``lax.cond`` (both engines
    run under ``vmap``).

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer.
        report: A :class:`ReportData` or None. With a report the weights carry
            its likelihood (detections and nulls, planet shot noise included,
            the photometric scale integrated over the target's range), and the
            returned elements carry a drawn ``ln_lam``.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        n_round: Proposals per round.
        n_rounds: Rounds of the collapsed engine.
        n_last: Proposals of the last collapsed round; default ``n_round``.
        n_centers: Kernels of the adaptive proposal.
        defensive: Prior fraction of an adaptive round.
        n_constants: Sets of constants drawn at each collapsed proposal.
        ess_floor: Kish size under which the status is ``LOW_ESS``.
        max_rounds: Most rounds of the report ladder of the collapsed engine.

    Returns:
        A :class:`RefitCloud`; with no valid record the status is ``NO_SUPPORT``
        and the elements are NaN.
    """
    n_valid = jnp.sum(records.is_valid)

    def one(_):
        return scale_rotate_refit(
            target,
            records,
            key,
            n_particles=n_particles,
            n_proposals=n_round,
            ess_floor=ess_floor,
            report=report,
        )

    def many(_):
        return collapsed_refit(
            target,
            records,
            key,
            n_particles=n_particles,
            n_round=n_round,
            n_rounds=n_rounds,
            n_last=n_last,
            n_centers=n_centers,
            defensive=defensive,
            n_constants=n_constants,
            ess_floor=ess_floor,
            report=report,
            max_rounds=max_rounds,
        )

    if isinstance(n_valid, jax.core.Tracer):
        cloud = jax.lax.cond(n_valid >= 2, many, one, 0)
    else:
        cloud = many(0) if int(n_valid) >= 2 else one(0)
    supported = n_valid > 0
    cloud = eqx.tree_at(
        lambda c: c.elements, cloud, jnp.where(supported, cloud.elements, jnp.nan)
    )
    return eqx.tree_at(
        lambda c: c.status,
        cloud,
        jnp.where(supported, cloud.status, int(UpdaterStatus.NO_SUPPORT)).astype(
            jnp.int32
        ),
    )


def prior_cloud(target, key, n):
    """``n`` draws of the declared physical prior as elements ``(n, 8)``.

    ``ln P`` is uniform over the period range, ``e`` is ``Beta(0.867, 3.03)``
    truncated to ``e_max``, ``cos i`` is uniform, the node, ``omega`` and ``tau``
    are uniform, the mass is log-uniform in its window and ``ln_lam`` is uniform
    over the scale range. A fixed cloud reweighted by :func:`first_tier` is the
    belief of a slot with no detection.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        key: PRNG key.
        n: Draws (static).

    Returns:
        Elements in the order of ``ELEMENT_NAMES``.
    """
    m = target.model
    k = jax.random.split(key, 8)
    pool = jax.random.beta(k[0], ECC_ALPHA, ECC_BETA, (2 * n,))
    order = jnp.argsort(pool >= m.e_max, stable=True)
    ecc = pool[order][:n]
    ln_period = jax.random.uniform(k[1], (n,), minval=m.lnP_lo, maxval=m.lnP_hi)
    cos_i = jax.random.uniform(k[2], (n,), minval=-1.0, maxval=1.0)
    node = jax.random.uniform(k[3], (n,), maxval=two_pi)
    omega = jax.random.uniform(k[4], (n,), maxval=two_pi)
    tau = jax.random.uniform(k[5], (n,))
    ln_mass = jax.random.uniform(
        k[6], (n,), minval=jnp.log(m.mu_lo), maxval=jnp.log(m.mu_hi)
    ) - jnp.log(MU_SUN)
    ln_lam = jax.random.uniform(
        k[7], (n,), minval=target.ln_lam_lo, maxval=target.ln_lam_hi
    )
    return jnp.stack(
        [ln_period, ecc, cos_i, node, omega, tau, ln_mass, ln_lam],
        axis=1,
    )


def first_tier(target, elements, report, key, *, n_particles, ess_floor=400.0):
    """Reweight a fixed prior cloud by a report: the belief of a slot with no fit.

    The weight of each orbit is its scale-integrated report likelihood
    (:func:`report_log_weights`); the returned cloud is ``n_particles``
    equal-weight resamples with a photometric scale drawn per orbit. The cloud is
    fixed by the caller (:func:`prior_cloud`, or a prior cloud restricted to a
    window), so the same elements serve every report of a star.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        elements: Prior draws, shape ``(n, 8)``.
        report: A :class:`ReportData`.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        ess_floor: Kish size under which the status is ``LOW_ESS``.

    Returns:
        A :class:`RefitCloud` with ``rounds`` 1; ``log_evidence`` is the log mean
        report likelihood over the prior draws.
    """
    log_w = report_log_weights(target, elements, report)
    ess = _kish(log_w)
    k_pick, k_draw = jax.random.split(key)
    picked = _resample_elements(
        k_pick,
        log_w,
        n_particles,
        lambda idx, branch: elements[idx],
        lambda idx, k: draw_report_scale(target, elements[idx], report, k_draw),
    )
    return RefitCloud(
        elements=picked,
        log_evidence=jnp.where(
            ess > 0, logsumexp(log_w) - jnp.log(log_w.size), -jnp.inf
        ),
        ess=ess,
        status=_status(ess, log_w, ess_floor),
        rounds=jnp.asarray(1),
        max_weight=_max_weight(log_w),
    )
