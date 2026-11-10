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

    uv run python libraries/photomancy/tools/refit_golden.py [MIN_KISH]
"""

import itertools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import G, Msun2kg, mas2arcsec, two_pi

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.few_epoch import MU_SUN
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.refit import (
    ECC_ALPHA,
    ECC_BETA,
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
    main(*(float(x) for x in sys.argv[1:]))
