#!/usr/bin/env python3
"""Golden reference for the windowed physical-prior refit, on the maintainer fixture.

The fixture is a 1 solar mass star at 10 pc with a planet at a = 1 AU, e = 0.1,
cos i = 0.35 and epochs measured to 3.5 mas at 0, 200 and 360 days; the stellar
mass is known to a 10 percent log-uniform window, and the quantities are read at
500 days. The reference is scale-and-rotate with the mass drawn from the window:
exact importance weights for the physical prior, no adaptation, accumulated over
chunks of proposals until the Kish size is large enough.

Per count of records ``k`` (1, 2, 3) the file ``tests/data/refit_golden.json``
stores the records, the visibility probability at the future epoch (separation
above the inner working angle), the mean and covariance of the predicted
position, the eight ``ln P`` bin masses, the log evidence and the Kish size.
This is a slow campaign, never part of the unit tier.

Usage::

    uv run python tools/refit_golden.py [MIN_KISH]
    uv run python tools/refit_golden.py report [MIN_KISH] [MAX_CHUNKS]
    uv run python tools/refit_golden.py independent [KEYS]

The ``report`` form adds the photometric report: the same fixture with the first
``k`` detections reported with their counts, a null at 410 days, a photometric
scale uniform in its logarithm over a range of 2.25, and the report likelihood in
the exact weights with the scale integrated on a fine grid. It writes
``tests/data/refit_report_golden.json``. The ``independent`` form adds to that file
a second reference built without the exact-weight chunks: astrometry-only clouds of
the refit, each orbit weighted by the report likelihood from the forward model with
the scale integrated on a grid, repeated over several keys so that the spread of
the keys gives the standard error. It is the reference the acceptance test grades
the detection probability against, since the exact reference keeps few effective
draws at three records.
"""

import itertools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import G, Msun2kg, Rearth2AU, mas2arcsec, two_pi

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.few_epoch import MU_SUN
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from photomancy.orbit.refit import (
    ECC_ALPHA,
    ECC_BETA,
    ReportData,
    _counts_at,
    _element_geometry,
    _report_sum,
    importance_refit,
    loglike_report,
    physical_prior_target,
    scale_rotate_draw,
)

jax.config.update("jax_enable_x64", True)

DIST_PC = 10.0
SIGMA_MAS = 3.5
EPOCHS_D = (0.0, 200.0, 360.0)
FUTURE_D = 500.0
IWA_ARCSEC = 0.05
# truth: a (AU), e, cos_i, W, omega, M_ref (rad) at t = 0
TRUTH = (1.0, 0.1, 0.35, 1.0, 2.0, 1.0)
CAPACITY = 5
N_BINS = 8
CHUNK = 2**20
BRUTE_CHUNK = 2**17
MAX_CHUNKS = 200
PATH = Path(__file__).parents[1] / "tests" / "data" / "refit_golden.json"


def positions(elements, times):
    """East and north offsets (arcsec) of elements ``(n, 8)`` at ``times``."""

    def one(el):
        period = jnp.exp(el[0])
        mass = jnp.exp(el[6]) * Msun2kg
        a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
        return jnp.stack(
            predict_relative_astrometry(
                times,
                a,
                el[1],
                el[2],
                el[3],
                jnp.cos(el[4]),
                jnp.sin(el[4]),
                el[5] * period,
                mass,
                DIST_PC,
            )
        )

    return jax.vmap(one)(elements)


def fixture(k):
    """The first ``k`` noisy records of the truth orbit, as a padded buffer."""
    a, e, cos_i, node, omega, m_ref = TRUTH
    mass = Msun2kg
    period = two_pi / jnp.sqrt(G * mass / a**3)
    tp = -m_ref / two_pi * period
    times = jnp.asarray(EPOCHS_D[:k])
    ra, dec = predict_relative_astrometry(
        times, a, e, cos_i, node, jnp.cos(omega), jnp.sin(omega), tp, mass, DIST_PC
    )
    sigma = SIGMA_MAS * mas2arcsec
    noise = sigma * jax.random.normal(jax.random.key(11, impl="threefry2x32"), (2, k))
    n = CAPACITY
    pad = lambda x, fill=0.0: jnp.full(n, fill).at[:k].set(x)  # noqa: E731
    return RelativeAstromData(
        times=pad(times),
        ra=pad(ra + noise[0]),
        dec=pad(dec + noise[1]),
        ra_err=pad(jnp.full(k, sigma), 1.0),
        dec_err=pad(jnp.full(k, sigma), 1.0),
        corr=pad(jnp.zeros(k)),
        planet_id=jnp.zeros(n, int),
        is_valid=jnp.arange(n) < k,
    )


def target():
    """The windowed physical-prior target of the fixture."""
    return physical_prior_target(1.0, DIST_PC)


def reference(k, min_kish, seed=0):
    """Accumulate the weighted statistics for ``k`` records."""
    tg, buf = target(), fixture(k)
    edges = np.linspace(float(tg.model.lnP_lo), float(tg.model.lnP_hi), N_BINS + 1)
    future = jnp.asarray([FUTURE_D])
    shift = None
    total = np.zeros(
        1 + 5 + 1 + N_BINS
    )  # weight, ra, dec, ra2, dec2, ra*dec, vis, bins
    sq = 0.0
    n_done = 0
    for chunk in range(MAX_CHUNKS):
        key = jax.random.fold_in(jax.random.key(seed), chunk)
        elements, log_w, offset = scale_rotate_draw(tg, buf, key, n_proposals=CHUNK)
        lw = np.asarray(log_w)
        if shift is None:
            shift = float(lw.max()) if np.isfinite(lw.max()) else 0.0
        w = np.exp(lw - shift)
        xy = np.asarray(positions(elements, future))[:, :, 0]
        ra, dec = xy[:, 0], xy[:, 1]
        ln_p = np.asarray(elements[:, 0])
        vis = (np.hypot(ra, dec) > IWA_ARCSEC).astype(float)
        bins = np.stack(
            [(ln_p >= lo) & (ln_p < hi) for lo, hi in itertools.pairwise(edges)], 1
        )
        parts = np.concatenate(
            [
                [w.sum()],
                [w @ ra, w @ dec, w @ ra**2, w @ dec**2, w @ (ra * dec), w @ vis],
                w @ bins,
            ]
        )
        total += parts
        sq += float(w @ w)
        n_done += CHUNK
        kish = total[0] ** 2 / sq
        if kish >= min_kish and chunk >= 3:
            break
    s0 = total[0]
    mean = np.array([total[1], total[2]]) / s0
    cov = np.array(
        [
            [total[3] / s0 - mean[0] ** 2, total[5] / s0 - mean[0] * mean[1]],
            [total[5] / s0 - mean[0] * mean[1], total[4] / s0 - mean[1] ** 2],
        ]
    )
    return {
        "k": k,
        "proposals": n_done,
        "kish": float(kish),
        "p_visible": float(total[6] / s0),
        "p_visible_se": float(np.sqrt(total[6] / s0 * (1 - total[6] / s0) / kish)),
        "position_mean": mean.tolist(),
        "position_cov": cov.tolist(),
        "bin_edges": edges.tolist(),
        "bin_masses": (total[7:] / s0).tolist(),
        "log_evidence": float(np.log(s0) + shift - np.log(n_done) - float(offset)),
    }


def brute_force_reference(k, n_chunks=64, seed=1):
    """Independent check of the reference: plain importance sampling from the prior.

    Orbits are drawn in physical coordinates from the declared prior (``e`` Beta
    truncated, ``ln P`` and the periastron phase uniform, the mass log-uniform
    in its window, ``cos i``, ``omega`` and the node uniform) and weighted by
    the record likelihood alone, through the forward model and none of the
    scale-and-rotate machinery. It needs about 1e3 draws per effective sample, so
    it serves one record only.
    """
    tg, buf = target(), fixture(k)
    m = tg.model
    edges = np.linspace(float(m.lnP_lo), float(m.lnP_hi), N_BINS + 1)
    future = jnp.asarray([FUTURE_D])
    times = buf.times[:k]
    data = jnp.stack([buf.ra[:k], buf.dec[:k]])
    sig = jnp.stack([buf.ra_err[:k], buf.dec_err[:k]])
    vis = np.zeros(1)
    bins = np.zeros(N_BINS)
    s0 = s1 = 0.0
    n_done = 0
    for chunk in range(n_chunks):
        ks = jax.random.split(jax.random.fold_in(jax.random.key(seed), chunk), 7)
        n = BRUTE_CHUNK
        e = jax.random.beta(ks[0], ECC_ALPHA, ECC_BETA, (n,))
        ln_p = jax.random.uniform(ks[1], (n,), minval=m.lnP_lo, maxval=m.lnP_hi)
        tau = jax.random.uniform(ks[2], (n,))
        cos_i = jax.random.uniform(ks[3], (n,), minval=-1.0, maxval=1.0)
        omega = jax.random.uniform(ks[4], (n,), maxval=two_pi)
        node = jax.random.uniform(ks[5], (n,), maxval=two_pi)
        ln_mu = jax.random.uniform(
            ks[6], (n,), minval=jnp.log(m.mu_lo), maxval=jnp.log(m.mu_hi)
        )
        elements = jnp.stack(
            [ln_p, e, cos_i, node, omega, tau, ln_mu - jnp.log(MU_SUN), 0 * e], axis=1
        )
        xy = positions(elements, times)
        resid = (xy - data[None]) / sig[None]
        log_w = -0.5 * jnp.sum(resid**2, axis=(1, 2))
        log_w = jnp.where(e < m.e_max, log_w, -jnp.inf)
        w = np.exp(np.asarray(log_w))
        xy_f = np.asarray(positions(elements, future))[:, :, 0]
        ln_p_np = np.asarray(ln_p)
        vis += np.array([w @ (np.hypot(xy_f[:, 0], xy_f[:, 1]) > IWA_ARCSEC)])
        bins += w @ np.stack(
            [(ln_p_np >= lo) & (ln_p_np < hi) for lo, hi in itertools.pairwise(edges)],
            1,
        )
        s0 += w.sum()
        s1 += float(w @ w)
        n_done += n
    kish = s0**2 / s1
    p_vis = float(vis[0] / s0)
    return {
        "k": k,
        "proposals": n_done,
        "kish": float(kish),
        "p_visible": p_vis,
        "p_visible_se": float(np.sqrt(p_vis * (1 - p_vis) / kish)),
        "bin_masses": (bins / s0).tolist(),
    }


def records_json(k):
    """The fixture buffer as plain lists."""
    buf = fixture(k)
    return {
        name: np.asarray(getattr(buf, name)).tolist()
        for name in ("times", "ra", "dec", "ra_err", "dec_err", "corr", "is_valid")
    }


# -- report fixture ----------------------------------------------------------------

THRESHOLD = 5.0
LIMIT_COUNTS = 100.0
CONTRAST_FLOOR = 5e-11
LAM_TRUE = 0.3 * Rearth2AU**2
SCALE_HALF = 1.125
N_SCALE_GRID = 512
T_NULL = 410.0
N_GRID = 128
KEEP_BELOW_MAX = 20.0
# Three records with the photometric report keep about 2 effective draws per 1e6
# proposals, so the chunk cap, not the Kish target, ends that reference.
REPORT_MAX_CHUNKS = 600
REPORT_PATH = Path(__file__).parents[1] / "tests" / "data" / "refit_report_golden.json"


def lam_range():
    """The log scale range ``ln(Ag Rp^2)``: the truth plus or minus 1.125."""
    return (np.log(LAM_TRUE) - SCALE_HALF, np.log(LAM_TRUE) + SCALE_HALF)


def report_target():
    """The windowed physical-prior target of the fixture with its scale range."""
    return physical_prior_target(1.0, DIST_PC, 0.10, lam_range())


def fine_scale_terms(tg, el, rep, row):
    """Log report marginal and detection probability of one orbit, scale on a grid.

    The scale offset ``t`` runs over ``N_SCALE_GRID`` cell midpoints of the range
    (uniform prior); the marginal is the mean of the report likelihood over them
    and the detection probability at ``row`` is its posterior average.
    """
    half = tg.ln_lam_half
    t = -half + (2.0 * half / N_SCALE_GRID) * (jnp.arange(N_SCALE_GRID) + 0.5)
    limit, log_g = _element_geometry(tg, el, rep)
    ll = jax.vmap(
        lambda ti: _report_sum(_counts_at(log_g, limit, ti, rep), limit, rep)
    )(t)
    limit_f, log_g_f = _element_geometry(tg, el, row)
    pd = jax.vmap(
        lambda ti: (
            1.0
            - jnp.exp(_report_sum(_counts_at(log_g_f, limit_f, ti, row), limit_f, row))
        )
    )(t)
    marginal = jax.scipy.special.logsumexp(ll) - jnp.log(N_SCALE_GRID)
    post = jax.nn.softmax(jnp.where(jnp.isfinite(ll), ll, -jnp.inf))
    return marginal, jnp.sum(jnp.where(post > 0.0, post * pd, 0.0))


def sensitivity(n_rows):
    """Separation, limiting delta magnitude and noise grids of ``n_rows`` epochs.

    The planet-free noise is ``LIMIT_COUNTS`` at every separation and the
    contrast limit is ``CONTRAST_FLOOR (1 + (0.05 / alpha)^6)``, so the expected
    counts are ``THRESHOLD LIMIT_COUNTS contrast / limit(alpha)``.
    """
    sep = np.geomspace(0.005, 2.0, N_GRID)
    dmag0 = -2.5 * np.log10(CONTRAST_FLOOR * (1.0 + (IWA_ARCSEC / sep) ** 6))
    tile = lambda x: jnp.asarray(np.tile(x, (n_rows, 1)))  # noqa: E731
    return tile(sep), tile(dmag0), tile(np.full(N_GRID, LIMIT_COUNTS))


def truth_counts(times):
    """Expected planet counts of the fixture's truth orbit at ``times`` (days)."""
    a, e, cos_i, node, omega, m_ref = TRUTH
    period = two_pi / jnp.sqrt(G * Msun2kg / a**3)
    tp = -m_ref / two_pi * period
    alpha, d_mag = predict_photometry(
        jnp.asarray(times),
        a,
        e,
        cos_i,
        node,
        jnp.cos(omega),
        jnp.sin(omega),
        tp,
        Msun2kg,
        LAM_TRUE,
        DIST_PC,
    )
    limit = CONTRAST_FLOOR * (1.0 + (IWA_ARCSEC / alpha) ** 6)
    return THRESHOLD * LIMIT_COUNTS * 10.0 ** (-0.4 * d_mag) / limit


def report_fixture(k):
    """Reports of the fixture: the first ``k`` detections and a null at ``T_NULL``.

    Detections carry counts drawn from the reporting law at the truth orbit
    (``Normal(N, LIMIT_COUNTS^2 + N)``, every draw above the threshold).
    """
    times = np.array([*EPOCHS_D[:k], T_NULL])
    n = CAPACITY
    expected = np.asarray(truth_counts(times))
    noise = np.asarray(jax.random.normal(jax.random.key(12, impl="threefry2x32"), (k,)))
    counts = expected[:k] + np.sqrt(LIMIT_COUNTS**2 + expected[:k]) * noise
    assert np.all(counts >= THRESHOLD * LIMIT_COUNTS)
    sep, dmag0, limit = sensitivity(n)
    detected = np.zeros(n, bool)
    detected[:k] = True
    row_counts = np.zeros(n)
    row_counts[:k] = counts
    epochs = np.zeros(n)
    epochs[: k + 1] = times
    valid = np.zeros(n, bool)
    valid[: k + 1] = True
    return ReportData(
        epochs=jnp.asarray(epochs),
        sep_grid=sep,
        dmag0_grid=dmag0,
        limit_grid=limit,
        is_detected=jnp.asarray(detected),
        counts=jnp.asarray(row_counts),
        is_valid=jnp.asarray(valid),
        snr_thresh=THRESHOLD,
    )


def future_row():
    """A one-row report at ``FUTURE_D`` whose null probability is ``1 - P(detect)``."""
    sep, dmag0, limit = sensitivity(1)
    return ReportData(
        epochs=jnp.asarray([FUTURE_D]),
        sep_grid=sep,
        dmag0_grid=dmag0,
        limit_grid=limit,
        is_detected=jnp.zeros(1, bool),
        counts=jnp.zeros(1),
        is_valid=jnp.ones(1, bool),
        snr_thresh=THRESHOLD,
    )


def report_reference(k, min_kish, max_chunks=REPORT_MAX_CHUNKS, seed=0):
    """Accumulate the weighted statistics for ``k`` records and the report rows.

    Scale-and-rotate draws are exact for the astrometric physical prior; each is
    weighted by the report likelihood with the photometric scale integrated on a
    fine grid, and the scale posterior of each orbit carries the detection
    probability.
    """
    tg, buf, rep, row = report_target(), fixture(k), report_fixture(k), future_row()
    edges = np.linspace(float(tg.model.lnP_lo), float(tg.model.lnP_hi), N_BINS + 1)
    future = jnp.asarray([FUTURE_D])
    shift = None
    total = np.zeros(1 + 5 + 2 + N_BINS)
    sq = 0.0
    n_done = 0
    for chunk in range(max_chunks):
        key = jax.random.fold_in(jax.random.key(seed), chunk)
        elements, log_w, offset = scale_rotate_draw(tg, buf, key, n_proposals=CHUNK)
        # The report weight is at most one per row, so astrometric weights far under
        # the chunk's largest cannot matter: score the report on the rest only.
        astro = np.asarray(log_w)
        keep = np.flatnonzero(astro > np.max(astro) - KEEP_BELOW_MAX)
        size = max(2**12, 2 ** int(np.ceil(np.log2(max(keep.size, 1)))))
        padded = np.concatenate([keep, np.full(size - keep.size, keep[0])])
        elements = elements[jnp.asarray(padded)]
        marginal, pdet = jax.lax.map(
            lambda el: fine_scale_terms(tg, el, rep, row), elements, batch_size=2**9
        )
        marginal, pdet = np.asarray(marginal), np.asarray(pdet)
        lw = astro[padded] + marginal
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        lw[keep.size :] = -np.inf
        if shift is None:
            shift = float(lw.max()) if np.isfinite(lw.max()) else 0.0
        w = np.exp(lw - shift)
        xy = np.asarray(positions(elements, future))[:, :, 0]
        ra, dec = xy[:, 0], xy[:, 1]
        ln_p = np.asarray(elements[:, 0])
        vis = (np.hypot(ra, dec) > IWA_ARCSEC).astype(float)
        bins = np.stack(
            [(ln_p >= lo) & (ln_p < hi) for lo, hi in itertools.pairwise(edges)], 1
        )
        parts = np.concatenate(
            [
                [w.sum()],
                [w @ ra, w @ dec, w @ ra**2, w @ dec**2, w @ (ra * dec), w @ vis],
                [w @ pdet],
                w @ bins,
            ]
        )
        total += parts
        sq += float(w @ w)
        n_done += CHUNK
        kish = total[0] ** 2 / sq
        if kish >= min_kish and chunk >= 3:
            break
    s0 = total[0]
    mean = np.array([total[1], total[2]]) / s0
    cov = np.array(
        [
            [total[3] / s0 - mean[0] ** 2, total[5] / s0 - mean[0] * mean[1]],
            [total[5] / s0 - mean[0] * mean[1], total[4] / s0 - mean[1] ** 2],
        ]
    )
    p_det = total[7] / s0
    return {
        "k": k,
        "proposals": n_done,
        "kish": float(kish),
        "p_visible": float(total[6] / s0),
        "p_detect": float(p_det),
        "p_detect_se": float(np.sqrt(p_det * (1 - p_det) / kish)),
        "position_mean": mean.tolist(),
        "position_cov": cov.tolist(),
        "bin_edges": edges.tolist(),
        "bin_masses": (total[8:] / s0).tolist(),
        "log_evidence": float(np.log(s0) + shift - np.log(n_done) - float(offset)),
    }


def report_json(k):
    """The report fixture as plain lists."""
    rep = report_fixture(k)
    return {
        name: np.asarray(getattr(rep, name)).tolist()
        for name in (
            "epochs",
            "sep_grid",
            "dmag0_grid",
            "limit_grid",
            "is_detected",
            "counts",
            "is_valid",
        )
    }


def report_main(min_kish=20_000.0, max_chunks=REPORT_MAX_CHUNKS):
    """Write the report golden file for k = 1, 2, 3."""
    out = {
        "fixture": {
            "mass_msun": 1.0,
            "mass_half_width": 0.10,
            "dist_pc": DIST_PC,
            "epochs_d": list(EPOCHS_D),
            "null_d": T_NULL,
            "future_d": FUTURE_D,
            "iwa_arcsec": IWA_ARCSEC,
            "snr_thresh": THRESHOLD,
            "ln_lam_range": list(lam_range()),
        },
        "records": {str(k): records_json(k) for k in (1, 2, 3)},
        "report": {str(k): report_json(k) for k in (1, 2, 3)},
        "reference": {},
    }
    for k in (1, 2, 3):
        out["reference"][str(k)] = report_reference(k, min_kish, int(max_chunks))
        print(k, out["reference"][str(k)], flush=True)
    REPORT_PATH.write_text(json.dumps(out, indent=1))
    print("wrote", REPORT_PATH)


# -- independent report reference ----------------------------------------------------

IND_KEYS = 8
IND_PARTICLES = 2**15
IND_SEED = 100
# proposals of the astrometry-only refit that make its cloud large enough
IND_FUSED = {
    1: dict(n_round=2**17),
    2: dict(n_round=2**16, n_last=2**18),
    3: dict(n_round=2**16, n_last=2**18),
}


def _unit_photometry(tg, el, epochs):
    """Separation (arcsec) and delta magnitude at unit scale of ``el`` at ``epochs``."""
    period = jnp.exp(el[0])
    mass = jnp.exp(el[6]) * Msun2kg
    a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
    return predict_photometry(
        epochs,
        a,
        el[1],
        el[2],
        el[3],
        jnp.cos(el[4]),
        jnp.sin(el[4]),
        tg.model.t_ref + el[5] * period,
        mass,
        1.0,
        tg.dist_pc,
    )


def _log_like_at(tg, el, rep, ln_lam):
    """Report log likelihood of one orbit at each log scale, via the forward model."""
    alpha, d_mag = _unit_photometry(tg, el, rep.epochs)
    return jax.vmap(
        lambda s: loglike_report(alpha, d_mag - 2.5 * s / np.log(10.0), rep)
    )(ln_lam)


def _forward_scale_terms(tg, el, rep, row):
    """Log report marginal and detection probability of one orbit, forward model route.

    The scale runs over ``N_SCALE_GRID`` cell midpoints of its log range. Unlike
    :func:`fine_scale_terms`, no geometry or scale helper of the refit module is
    used: every row goes through ``predict_photometry`` and ``loglike_report``.
    """
    width = tg.ln_lam_hi - tg.ln_lam_lo
    grid = tg.ln_lam_lo + width * (jnp.arange(N_SCALE_GRID) + 0.5) / N_SCALE_GRID
    ll = _log_like_at(tg, el, rep, grid)
    ll = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
    p_det = 1.0 - jnp.exp(_log_like_at(tg, el, row, grid))
    post = jax.nn.softmax(ll)
    marginal = jax.scipy.special.logsumexp(ll) - jnp.log(N_SCALE_GRID)
    return marginal, jnp.sum(jnp.where(post > 0.0, post * p_det, 0.0))


def _weighted_mean(log_w, x):
    """Self-normalized mean of ``x`` under host log weights."""
    w = np.exp(log_w - np.max(log_w))
    return float(w @ x / w.sum())


def independent_reference(k, keys=IND_KEYS):
    """Detection probability at the future epoch from astrometry-only clouds.

    Each of ``keys`` astrometry-only refits (no report) is scored with the report
    likelihood of every orbit, the scale integrated on a grid, and the clouds
    are pooled. The standard error is the spread of the per-key values over the
    root of their number. No scale code of the refit enters.
    """
    keys = int(keys)
    tg, buf, rep, row = report_target(), fixture(k), report_fixture(k), future_row()
    score = jax.jit(
        lambda els: jax.lax.map(
            lambda el: _forward_scale_terms(tg, el, rep, row), els, batch_size=2**9
        )
    )
    pooled, per_key = [], []
    for i in range(keys):
        cloud = importance_refit(
            tg,
            buf,
            None,
            jax.random.key(IND_SEED + i),
            n_particles=IND_PARTICLES,
            **IND_FUSED[k],
        )
        marginal, p_det = (np.asarray(v) for v in score(cloud.elements))
        pooled.append((marginal, p_det))
        per_key.append(_weighted_mean(marginal, p_det))
        print(k, i, per_key[-1], flush=True)
    marginal, p_det = (np.concatenate(v) for v in zip(*pooled, strict=True))
    weight = np.exp(marginal - marginal.max())
    return {
        "k": k,
        "keys": keys,
        "p_detect": _weighted_mean(marginal, p_det),
        "p_detect_se": float(np.std(per_key, ddof=1) / np.sqrt(keys)),
        "kish": float(weight.sum() ** 2 / np.sum(weight**2)),
        "per_key": per_key,
    }


def independent_main(keys=IND_KEYS):
    """Add the independent reference to the stored report golden file."""
    out = json.loads(REPORT_PATH.read_text())
    out["independent"] = {}
    for k in (1, 2, 3):
        out["independent"][str(k)] = independent_reference(k, keys)
        print(k, out["independent"][str(k)], flush=True)
    REPORT_PATH.write_text(json.dumps(out, indent=1))
    print("wrote", REPORT_PATH)


def main(min_kish=40_000.0):
    """Write the golden file for k = 1, 2, 3."""
    out = {
        "fixture": {
            "mass_msun": 1.0,
            "mass_half_width": 0.10,
            "dist_pc": DIST_PC,
            "epochs_d": list(EPOCHS_D),
            "future_d": FUTURE_D,
            "iwa_arcsec": IWA_ARCSEC,
        },
        "records": {str(k): records_json(k) for k in (1, 2, 3)},
        "reference": {},
    }
    for k in (1, 2, 3):
        out["reference"][str(k)] = reference(k, min_kish)
        print(k, out["reference"][str(k)], flush=True)
    out["independent"] = {"1": brute_force_reference(1)}
    print("independent", out["independent"], flush=True)
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(out, indent=1))
    print("wrote", PATH)


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report_main(*(float(x) for x in sys.argv[2:]))
    elif sys.argv[1:2] == ["independent"]:
        independent_main(*sys.argv[2:])
    else:
        main(*(float(x) for x in sys.argv[1:]))
