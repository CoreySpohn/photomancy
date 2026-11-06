#!/usr/bin/env python3
"""Latency probe for the likelihood kernel of a fixed-budget orbit updater.

Times one jitted, vmapped evaluation of the astrometric log likelihood over a
particle cloud, with the record buffer passed as a dynamic argument, and scales
it to the cost of one update at the budget ceiling (particles x rungs x moves
likelihood evaluations). Compilation and the first call are timed separately
from the steady state, and every call blocks until the result is ready.

Two kernels are timed: the independent Thiele-Innes basis (``TIBasisModel``)
and the fixed-mass physical forward model. ``--references`` also prices one
proposal of each reference sampler (the Lambert depth fit and the OFTI
scale-and-rotate proposal) for the compute tally.

Usage::

    python tools/updater_latency_probe.py [--output docs/updater-latency-probe.json]
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import Msun2kg, mas2arcsec

jax.config.update("jax_enable_x64", True)

from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.few_epoch import TIBasisModel, lambert_depth_fit  # noqa: E402
from photomancy.orbit.forward import predict_relative_astrometry  # noqa: E402
from photomancy.orbit.likelihoods import loglike_relative_astrom  # noqa: E402
from photomancy.orbit.ofti import ScaleAndRotate  # noqa: E402

DIST_PC = 10.0
SIGMA_MAS = 3.5


def make_data(model, key, times, capacity, sigma_mas=SIGMA_MAS):
    """Synthetic astrometry of one TI prior draw, padded to ``capacity``."""
    k_truth, k_noise = jax.random.split(key)
    theta = model.sample_prior(k_truth, 1)[0]
    times = jnp.asarray(times)
    east, north = model.forward_au(theta, times)
    sigma = sigma_mas * mas2arcsec
    noise = sigma * jax.random.normal(k_noise, (2, times.shape[0]))
    n = times.shape[0]
    return RelativeAstromData.pad(
        times=times,
        ra=east / DIST_PC + noise[0],
        dec=north / DIST_PC + noise[1],
        ra_err=jnp.full(n, sigma),
        dec_err=jnp.full(n, sigma),
        corr=jnp.zeros(n),
        planet_id=jnp.zeros(n, dtype=int),
        max_n=capacity,
    )


def ti_loglike(model):
    """Per-particle TI log likelihood, vmapped over particles."""

    def one(theta, data):
        east, north = model.forward_au(theta, data.times)
        return loglike_relative_astrom(east / DIST_PC, north / DIST_PC, data)

    return eqx.filter_jit(jax.vmap(one, in_axes=(0, None)))


def physical_loglike(mass_kg):
    """Per-particle fixed-mass physical log likelihood, vmapped over particles."""

    def one(row, data):
        a, e, cos_i, W, cos_w, sin_w, tp = row
        ra, dec = predict_relative_astrometry(
            data.times, a, e, cos_i, W, cos_w, sin_w, tp, mass_kg, DIST_PC
        )
        return loglike_relative_astrom(ra, dec, data)

    return eqx.filter_jit(jax.vmap(one, in_axes=(0, None)))


def physical_particles(key, n):
    """Physical rows (a, e, cos_i, W, cos_w, sin_w, tp) spread over the prior."""
    ks = jax.random.split(key, 5)
    a = jnp.exp(
        jax.random.uniform(ks[0], (n,), minval=jnp.log(0.27), maxval=jnp.log(5.7))
    )
    e = jax.random.uniform(ks[1], (n,), maxval=0.9)
    cos_i = jax.random.uniform(ks[2], (n,), minval=-1.0, maxval=1.0)
    W = jax.random.uniform(ks[3], (n,), maxval=2.0 * jnp.pi)
    w = jax.random.uniform(ks[4], (n,), maxval=2.0 * jnp.pi)
    tp = jax.random.uniform(jax.random.fold_in(key, 9), (n,), maxval=5000.0)
    return jnp.stack([a, e, cos_i, W, jnp.cos(w), jnp.sin(w), tp], axis=1)


def time_kernel(fn, args, repeats, perturb):
    """Return (first-call seconds, steady-state per-call seconds list).

    ``perturb(i, args)`` returns fresh argument values of the same shapes for
    each steady-state call, so no call can reuse an earlier result.
    """
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    jax.block_until_ready(fn(*perturb(-1, args)))  # one more warm call
    steady = []
    for i in range(repeats):
        call_args = perturb(i, args)
        jax.block_until_ready(call_args)
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*call_args))
        steady.append(time.perf_counter() - t0)
    return first, steady


def summarize(first, steady, particles, evals_per_update):
    """Per-call, per-evaluation and implied per-update costs."""
    med = statistics.median(steady)
    q = statistics.quantiles(steady, n=4)
    per_eval = med / particles
    return {
        "first_call_s": first,
        "steady_median_s": med,
        "steady_iqr_s": [q[0], q[2]],
        "steady_min_s": min(steady),
        "per_evaluation_us": per_eval * 1e6,
        "ceiling_seconds_per_update": per_eval * evals_per_update,
    }


def perturb_data(data):
    """Return a perturbation function that shifts the measured offsets."""

    def perturb(i, args):
        particles, d = args
        shift = 1e-6 * (i + 2)
        return particles, eqx.tree_at(
            lambda x: (x.ra, x.dec), d, (d.ra + shift, d.dec - shift)
        )

    return perturb


def price_references(key, data, model, repeats):
    """Seconds per proposal of the two reference samplers (warm)."""
    out = {}
    n_pf = 4096
    n_max_rev = 2
    families = 2 + 4 * n_max_rev

    def lam(i, args):
        return (jax.random.fold_in(key, i + 3),)

    def lam_fit(k):
        return lambert_depth_fit(
            data,
            dist_pc=DIST_PC,
            model=model,
            key=k,
            n_per_family=n_pf,
            n_max_rev=n_max_rev,
            z_max_au=15.7,
        ).log_weights

    _, steady = time_kernel(lam_fit, (key,), repeats, lam)
    out["ti"] = statistics.median(steady) / (n_pf * families)

    batch = 2**17
    cond = ScaleAndRotate()

    @eqx.filter_jit
    def score(k, d):
        return cond.sample_and_score(
            k,
            batch,
            d,
            0,
            Msun2kg,
            DIST_PC,
            "kipping13",
            (float(np.log10(50.0)), float(np.log10(5000.0))),
            0.9,
        )

    def sc(i, args):
        return (jax.random.fold_in(key, i + 11), args[1])

    _, steady = time_kernel(score, (key, data), repeats, sc)
    out["physical"] = statistics.median(steady) / batch
    return out


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--rungs", type=int, default=32)
    parser.add_argument("--moves", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=3, help="record buffer length")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--references", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    key = jax.random.PRNGKey(0)
    model = TIBasisModel()
    times = [0.0, 180.0, 360.0][: args.epochs]
    data = make_data(model, jax.random.fold_in(key, 1), times, args.capacity)
    evals = args.particles * args.rungs * args.moves

    theta = model.sample_prior(jax.random.fold_in(key, 2), args.particles)
    rows = physical_particles(jax.random.fold_in(key, 3), args.particles)
    perturb = perturb_data(data)
    results = {}
    for name, fn, particles in (
        ("ti", ti_loglike(model), theta),
        ("physical", physical_loglike(Msun2kg), rows),
    ):
        first, steady = time_kernel(fn, (particles, data), args.repeats, perturb)
        results[name] = summarize(first, steady, args.particles, evals)

    report = {
        "kernel": "vmapped astrometric log likelihood, records as a dynamic argument",
        "particles": args.particles,
        "rungs": args.rungs,
        "moves_per_rung": args.moves,
        "evaluations_per_update": evals,
        "epochs": args.epochs,
        "record_capacity": args.capacity,
        "repeats": args.repeats,
        "float": "float64",
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "results": results,
        "ceiling_seconds_per_update": {
            k: v["ceiling_seconds_per_update"] for k, v in results.items()
        },
    }
    if args.references:
        report["reference_seconds_per_proposal"] = price_references(
            jax.random.fold_in(key, 4), data, model, max(5, args.repeats // 20)
        )
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
