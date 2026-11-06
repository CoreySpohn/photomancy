#!/usr/bin/env python3
"""Acceptance panel driver for the fixed-budget array orbit updater.

The protocol lives in ``docs/updater-protocol.md``; its JSON parameter block is
the only source of the numbers used here. Critical values are recomputed from
their formulas, never copied.

Usage::

    python tools/updater_panel.py --dry-run [--prices docs/updater-latency-probe.json]
    python tools/updater_panel.py --sizing
    python tools/updater_panel.py --freeze RUN_DIR [--exact EXACT.json]
    python tools/updater_panel.py --run development --out RUN_DIR [--shard i/n]
        [--parts updater,reference,certification] [--worlds 0-5]
    python tools/updater_panel.py --status RUN_DIR

``--dry-run`` prints the cells, the world and seed allocation, the model hashes,
the reference arguments, the critical values and the compute tally. ``--sizing``
prints the power of each criterion against its stated alternatives.
``--run`` executes the work units of a stage (development, smoke or panel):
updater runs (three updates of one world for one run role) and reference
fits (one world, role and epoch count), each written as one JSON line the
moment it finishes. A restart skips every unit already in the directory, and
``--shard i/n`` splits the units between processes or machines. ``--freeze``
assembles the development records, computes the alias tolerance, certifies
both references against it and reads the exact-check results; it reports
``frozen`` or ``incomplete`` with every reason. Grading the panel into the
verdict table is not implemented yet.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs" / "updater-protocol.md"
PRICES = REPO / "docs" / "updater-latency-probe.json"
MARKER = "<!-- updater-protocol-parameters -->"


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------


def load_protocol(path=PROTOCOL):
    """Return the parameter block of the protocol document as a dict.

    The block is the first fenced ``json`` code block after the marker comment.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(MARKER)
    except ValueError as err:
        raise ValueError(f"{path}: parameter marker not found") from err
    body = []
    inside = False
    for line in lines[start + 1 :]:
        if not inside:
            if line.strip() == "```json":
                inside = True
            continue
        if line.strip() == "```":
            break
        body.append(line)
    if not body:
        raise ValueError(f"{path}: empty parameter block")
    return json.loads("\n".join(body))


# ---------------------------------------------------------------------------
# Exact binomial tails and the critical values
# ---------------------------------------------------------------------------


def _binom_logpmf(k, n, p):
    """Log probability of k successes in n Bernoulli(p) trials."""
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binom_cdf(k, n, p):
    """Return P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.fsum(math.exp(_binom_logpmf(j, n, p)) for j in range(k + 1))


def binom_sf(k, n, p):
    """Return P(X > k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return math.fsum(math.exp(_binom_logpmf(j, n, p)) for j in range(k + 1, n + 1))


def clopper_pearson_upper(x, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if x >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binom_cdf(x, n, mid) > 1.0 - confidence:
            lo = mid
        else:
            hi = mid
    return hi


def coverage_critical_count(n, level, alpha):
    """Largest c with P(Bin(n, level) <= c) <= alpha (fail at c or fewer)."""
    return max(c for c in range(n + 1) if binom_cdf(c, n, level) <= alpha)


def overcoverage_count(n, level, alpha):
    """Smallest k with P(Bin(n, level) >= k) <= alpha."""
    return min(k for k in range(n + 1) if binom_sf(k - 1, n, level) <= alpha)


def dkw_epsilon(n, alpha):
    """Two-sided DKW band half-width at level alpha for n draws."""
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def upper_tail_critical_count(n, rate, alpha):
    """Smallest c with P(Bin(n, rate) > c) <= alpha (fail above c)."""
    return min(c for c in range(n + 1) if binom_sf(c, n, rate) <= alpha)


def student_t_cdf(t, dof):
    """CDF of Student's t with an integer number of degrees of freedom."""
    theta = math.atan(t / math.sqrt(dof))
    c, s = math.cos(theta), math.sin(theta)
    if dof % 2 == 1:
        term, total = c, (c if dof > 1 else 0.0)
        for k in range(3, dof - 1, 2):
            term *= c * c * (k - 1) / k
            total += term
        return 0.5 + (theta + s * total) / math.pi
    term, total = 1.0, 1.0
    for k in range(2, dof - 1, 2):
        term *= c * c * (k - 1) / k
        total += term
    return 0.5 + 0.5 * s * total


def student_t_ppf(q, dof):
    """Quantile of Student's t (integer degrees of freedom) by bisection."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, dof) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson_lower(x, n, confidence=0.95):
    """One-sided Clopper-Pearson lower bound on a binomial rate."""
    if x <= 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(n - x, n, confidence)


def group_counts(protocol):
    """Number of regimes, mass models, epoch counts, quantities and updates."""
    panel = protocol["panel"]
    return {
        "regimes": len(protocol["regimes"]),
        "mass_models": len(panel["mass_models"]),
        "epoch_counts": len(panel["epoch_counts"]),
        "quantities": len(panel["quantities"]),
        "updates": len(panel["updates"]),
        "alias_quantities": len(protocol["criteria"]["alias_accuracy"]["quantities"]),
    }


_CRITICAL_VALUES = {}


def critical_values(protocol):
    """Recompute every derived threshold of the protocol from its formula.

    Returns a dict with the exact values and their rounded forms, keyed like the
    protocol's ``stated`` block for the rounded ones. Cached per protocol
    content (the binomial critical counts take a third of a second).
    """
    key = json.dumps(protocol, sort_keys=True, default=str)
    if key not in _CRITICAL_VALUES:
        _CRITICAL_VALUES[key] = _critical_values(protocol)
    return dict(_CRITICAL_VALUES[key])


def _critical_values(protocol):
    g = group_counts(protocol)
    crit = protocol["criteria"]
    n = protocol["panel"]["worlds_per_cell"]
    groups = g["regimes"] * g["mass_models"] * g["epoch_counts"]
    n_cells = groups * g["quantities"]

    cov = crit["coverage"]
    alpha_cell = cov["family_alpha"] / n_cells
    eps = dkw_epsilon(n, crit["calibration"]["family_alpha"] / n_cells)

    alias = crit["alias_accuracy"]
    n_alias = groups * g["alias_quantities"]
    z = statistics.NormalDist().inv_cdf(1.0 - alias["family_alpha"] / n_alias)

    fu = crit["failed_updates"]
    n_fu = g["regimes"] * g["mass_models"] * g["updates"]
    fu_c = upper_tail_critical_count(
        n, fu["acceptable_rate"], fu["family_alpha"] / n_fu
    )

    damaged = alias["alternative"]["damaged_fraction"]
    cert_worlds = (
        g["regimes"] * protocol["panel"]["development_worlds_per_regime_and_mass_model"]
    )
    cert_miss = (1.0 - damaged) ** cert_worlds

    ex = protocol["reference"]["exact_checks"]
    exact_m = ex["cases"] * (ex["bins_per_case"] + ex["interval_end_points_per_case"])
    t_crit = student_t_ppf(
        1.0 - ex["family_alpha"] / (2.0 * exact_m), ex["sampler_repeats"] - 1
    )

    return {
        "cells": n_cells,
        "coverage_alpha": alpha_cell,
        "coverage_critical_count": coverage_critical_count(
            n, cov["interval"], alpha_cell
        ),
        "overcoverage_report_count": overcoverage_count(n, cov["interval"], alpha_cell),
        "dkw_epsilon": eps,
        "dkw_epsilon_5dp": round(eps, 5),
        "alias_tests": n_alias,
        "alias_z": z,
        "alias_z_3dp": round(z, 3),
        "failed_update_groups": n_fu,
        "failed_updates_critical_count": fu_c,
        "certification_worlds_per_mass_model": cert_worlds,
        "certification_miss_probability": cert_miss,
        "certification_miss_probability_3dp": round(cert_miss, 3),
        "exact_check_compared_values": exact_m,
        "exact_check_t_critical": t_crit,
        "exact_check_t_critical_3dp": round(t_crit, 3),
    }


# ---------------------------------------------------------------------------
# Cells, models and reference arguments
# ---------------------------------------------------------------------------


def cells(protocol):
    """Every graded (regime, mass model, epoch count, quantity) cell."""
    panel = protocol["panel"]
    return [
        (r["id"], m, k, q)
        for r in protocol["regimes"]
        for m in panel["mass_models"]
        for k in panel["epoch_counts"]
        for q in panel["quantities"]
    ]


def regime_bounds(protocol, regime, mass_model):
    """Period range (days) and e_max of a mass model under a regime."""
    defaults = protocol["model_pairs"][mass_model]["defaults"]
    trunc = regime["truncation"]
    p_lo, p_hi = trunc.get("period_days", defaults["period_days"])
    return p_lo, p_hi, trunc.get("e_max", defaults["e_max"])


def model_declaration(protocol, regime, mass_model):
    """The hashed declaration: the pair, its truncation and the regime design."""
    p_lo, p_hi, e_max = regime_bounds(protocol, regime, mass_model)
    return {
        "pair": protocol["model_pairs"][mass_model],
        "period_days": [p_lo, p_hi],
        "e_max": e_max,
        "cadence_days": regime["cadence_days"],
        "sigma_mas": regime["sigma_mas"],
        "distance_pc": protocol["distance_pc"],
        "astrometry": protocol["astrometry"],
    }


def declaration_hash(declaration):
    """SHA-256 of a declaration's canonical JSON (sorted keys, no whitespace)."""
    text = json.dumps(declaration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semimajor_axis_au(period_days, mu):
    """Kepler's third law: semimajor axis (AU) of a period (days) at mu."""
    return (mu * (period_days / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)


def physical_semimajor_range(protocol, regime):
    """Semimajor-axis range (AU) of the physical prior under a regime."""
    from hwoutils.constants import G, Msun2kg

    p_lo, p_hi, _ = regime_bounds(protocol, regime, "physical_fixed")
    mass = protocol["model_pairs"]["physical_fixed"]["defaults"]["stellar_mass_msun"]
    mu = mass * float(G) * float(Msun2kg)
    return semimajor_axis_au(p_lo, mu), semimajor_axis_au(p_hi, mu)


def visibility_mas(protocol):
    """Inner and outer working angles of the visibility event (mas)."""
    from hwoutils.constants import arcsec2mas
    from hwoutils.conversions import lambda_d_to_arcsec

    vis = protocol["visibility"]
    return tuple(
        float(lambda_d_to_arcsec(vis[k], vis["wavelength_nm"], vis["aperture_m"]))
        * arcsec2mas
        for k in ("iwa_lambda_over_d", "owa_lambda_over_d")
    )


def future_epoch(regime, epoch_count):
    """The next cadence epoch after the posterior's last accepted epoch."""
    times = regime["cadence_days"]
    if epoch_count < len(times):
        return times[epoch_count]
    return times[-1] + (times[-1] - times[-2])


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def root_key(protocol):
    """Typed threefry root key, independent of JAX's default generator."""
    import jax

    prng = protocol["prng"]
    return jax.random.key(prng["root_seed"], impl=prng["implementation"])


def world_key(protocol, stage, mass_model, regime_index, world_index):
    """The world key: semantic coordinates folded into the root key."""
    import jax

    prng = protocol["prng"]
    m_index = protocol["panel"]["mass_models"].index(mass_model)
    with jax.enable_x64(True):
        key = root_key(protocol)
        for value in (prng["stages"][stage], m_index, regime_index, world_index):
            key = jax.random.fold_in(key, value)
    return key


def draw_keys(protocol, stage, mass_model, regime_index, world_indices, purpose, index):
    """Keys of one run role for an array of worlds (vectorized over worlds)."""
    import jax
    import jax.numpy as jnp

    prng = protocol["prng"]
    m_index = protocol["panel"]["mass_models"].index(mass_model)
    with jax.enable_x64(True):
        block = root_key(protocol)
        for value in (prng["stages"][stage], m_index, regime_index):
            block = jax.random.fold_in(block, value)

        def one(w):
            key = jax.random.fold_in(block, w)
            key = jax.random.fold_in(key, prng["purposes"][purpose])
            return jax.random.fold_in(key, index)

        return jax.vmap(one)(jnp.asarray(world_indices))


def draw_key(protocol, stage, mass_model, regime_index, world_index, purpose, index):
    """Key of one draw: world key, then purpose, then the last index."""
    return draw_keys(
        protocol, stage, mass_model, regime_index, [world_index], purpose, index
    )[0]


def run_roles(protocol):
    """Every declared (stage, purpose, index) run role, with its world count.

    Yields ``(stage, purpose, index, worlds)``. Updater runs use the update
    index (0, 1, 2); reference, certification and PIT draws the epoch index (1
    for two epochs, 2 for three); truth and noise 0.
    """
    panel = protocol["panel"]
    updates = range(len(panel["updates"]))
    epoch_indices = [k - 1 for k in panel["epoch_counts"]]
    base = [("truth", 0), ("noise", 0)]
    base += [
        (p, u) for p in ("updater_a", "updater_b", "updater_doubled") for u in updates
    ]
    base += [
        (p, e) for p in ("reference_1", "reference_2", "pit") for e in epoch_indices
    ]
    cert = [(p, u) for p in PRECISION_ROLES for u in updates]
    cert += [
        (p, e)
        for p in (
            "reference_3",
            "reference_4",
            "cert_doubled_1",
            "cert_doubled_2",
            "cert_proposal_1",
            "cert_proposal_2",
        )
        for e in epoch_indices
    ]
    repeats = protocol["reference"]["exact_checks"]["sampler_repeats"]
    for stage in protocol["prng"]["stages"]:
        if stage == "exact":
            for r in range(repeats):
                yield stage, "exact_run", r, 1
            continue
        roles = base + (cert if stage == "development" else [])
        for purpose, index in roles:
            yield stage, purpose, index, stage_worlds(protocol, stage)


def all_role_keys(protocol, key_fn=None):
    """Raw key data (uint32 pairs) of every declared draw, through ``key_fn``."""
    import jax
    import numpy as np

    key_fn = key_fn or draw_keys
    rows = []
    for stage, purpose, index, worlds in run_roles(protocol):
        for mass_model in protocol["panel"]["mass_models"]:
            regimes = 1 if stage == "exact" else len(protocol["regimes"])
            for r_index in range(regimes):
                keys = key_fn(
                    protocol,
                    stage,
                    mass_model,
                    r_index,
                    np.arange(worlds),
                    purpose,
                    index,
                )
                rows.append(np.asarray(jax.random.key_data(keys)).reshape(-1, 2))
    return np.concatenate(rows)


def key_collisions(protocol, key_fn=None):
    """Number of declared draws whose key repeats another draw's key."""
    import numpy as np

    data = all_role_keys(protocol, key_fn)
    return len(data) - len(np.unique(data, axis=0))


# ---------------------------------------------------------------------------
# Truths (JAX rejection sampling of the declared priors)
# ---------------------------------------------------------------------------

TRUTH_BLOCK = 1024
TRUTH_MAX_BLOCKS = 256


def ti_model(protocol, regime):
    """The ``TIBasisModel`` of a regime (code defaults plus the truncation)."""
    from photomancy.orbit.few_epoch import TIBasisModel

    p_lo, p_hi, e_max = regime_bounds(protocol, regime, "ti_window")
    return TIBasisModel(e_max=e_max, lnP_lo=math.log(p_lo), lnP_hi=math.log(p_hi))


def ti_truth(model, truth_key):
    """First in-support candidate of the TI prior, drawn block by block."""
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        for block in range(TRUTH_MAX_BLOCKS):
            k = jax.random.split(jax.random.fold_in(truth_key, block), 4)
            n = (TRUTH_BLOCK,)
            cand = jnp.column_stack(
                [
                    jax.random.uniform(k[0], n, maxval=model.e_max),
                    jax.random.uniform(
                        k[1], n, minval=model.lnP_lo, maxval=model.lnP_hi
                    ),
                    jax.random.uniform(k[2], n),
                    model.s_ti * jax.random.normal(k[3], (TRUTH_BLOCK, 4)),
                ]
            )
            ok = jax.vmap(model.in_support)(cand)
            if bool(ok.any()):
                return cand[int(jnp.argmax(ok))]
    raise RuntimeError("TI truth draw exhausted its blocks")


def physical_truth(protocol, regime, truth_key):
    """One draw of the physical prior: (a, e, cos_i, W, omega, M_ref)."""
    import jax
    import jax.numpy as jnp

    a_lo, a_hi = physical_semimajor_range(protocol, regime)
    _, _, e_max = regime_bounds(protocol, regime, "physical_fixed")
    with jax.enable_x64(True):
        k = jax.random.split(truth_key, 6)
        e = None
        for block in range(TRUTH_MAX_BLOCKS):
            draws = jax.random.beta(
                jax.random.fold_in(k[1], block), 0.867, 3.03, (TRUTH_BLOCK,)
            )
            ok = draws < e_max
            if bool(ok.any()):
                e = draws[int(jnp.argmax(ok))]
                break
        if e is None:
            raise RuntimeError("physical eccentricity draw exhausted its blocks")
        ln_a = jax.random.uniform(k[0], minval=math.log(a_lo), maxval=math.log(a_hi))
        two_pi = 2.0 * math.pi
        return jnp.array(
            [
                jnp.exp(ln_a),
                e,
                jax.random.uniform(k[2], minval=-1.0, maxval=1.0),
                jax.random.uniform(k[3], maxval=two_pi),
                jax.random.uniform(k[4], maxval=two_pi),
                jax.random.uniform(k[5], maxval=two_pi),
            ]
        )


def stage_worlds(protocol, stage):
    """Worlds per (regime, mass model) on a stage."""
    panel = protocol["panel"]
    per_cell = {
        "panel": panel["worlds_per_cell"],
        "development": panel["development_worlds_per_regime_and_mass_model"],
    }
    if stage in per_cell:
        return per_cell[stage]
    if stage == "exact":
        return 1
    blocks = len(protocol["regimes"]) * len(panel["mass_models"])
    return panel["smoke_worlds"] // blocks


# ---------------------------------------------------------------------------
# Compute tally
# ---------------------------------------------------------------------------


def compute_tally(protocol):
    """Update-equivalents and reference fits per model pair."""
    g = group_counts(protocol)
    panel = protocol["panel"]
    # Two independent keys plus one doubled-budget run that costs two; the
    # development stage adds the precision replicates.
    run_units = 4
    upd_panel = g["regimes"] * panel["worlds_per_cell"] * g["updates"] * run_units
    dev_worlds = panel["development_worlds_per_regime_and_mass_model"]
    dev_units = run_units + len(PRECISION_ROLES)
    upd_dev = g["regimes"] * dev_worlds * g["updates"] * dev_units
    ref_runs = 2
    fits = g["regimes"] * panel["alias_subset_worlds"] * g["epoch_counts"] * ref_runs
    # Certification: repeat-noise runs, two runs at twice the budget (cost four)
    # and two other-proposal runs (at twice the budget for the TI pair, whose
    # independent reference is the less efficient one; amendment A7), priced
    # at the reference's cost per fit.
    dev_fits = {}
    for pair in panel["mass_models"]:
        proposal_units = sum(
            reference_budget_factor(pair, r)
            for r in ("cert_proposal_1", "cert_proposal_2")
        )
        cert_units = protocol["reference"]["repeat_noise"]["runs"] + 4 + proposal_units
        dev_fits[pair] = g["regimes"] * dev_worlds * g["epoch_counts"] * cert_units
    return {
        "panel_update_equivalents_per_pair": upd_panel,
        "development_update_equivalents_per_pair": upd_dev,
        "panel_reference_fits_per_pair": fits,
        "development_reference_fit_equivalents_per_pair": dev_fits,
    }


def _hours(seconds):
    return seconds / 3600.0


def probe_matches_budget(protocol, prices):
    """Whether the probe timed the protocol's budget; returns (ok, reasons)."""
    budget = protocol["updater_budget"]
    reasons = [
        f"{probe_key}={prices.get(probe_key)} but protocol {budget[key]}"
        for probe_key, key in (
            ("particles", "particles"),
            ("rungs", "rungs"),
            ("moves_per_rung", "moves_per_rung"),
        )
        if prices.get(probe_key) != budget[key]
    ]
    if prices.get("float") != budget["float"]:
        reasons.append(f"float={prices.get('float')} but protocol {budget['float']}")
    return not reasons, reasons


PAIR_PROBE_KEYS = {"ti_window": "ti", "physical_fixed": "physical"}


def priced_tally(protocol, prices):
    """Hours per model pair from the probe, or None when it does not match."""
    ok, reasons = probe_matches_budget(protocol, prices)
    if not ok:
        return None, reasons
    tally = compute_tally(protocol)
    out = {}
    for pair, key in PAIR_PROBE_KEYS.items():
        s_upd = prices["ceiling_seconds_per_update"][key]
        row = {
            "seconds_per_update": s_upd,
            "updater_panel_h": _hours(
                tally["panel_update_equivalents_per_pair"] * s_upd
            ),
        }
        per_prop = prices.get("reference_seconds_per_proposal", {}).get(key)
        if per_prop is not None:
            per_fit = per_prop * protocol["reference"][pair]["proposals_per_fit"]
            row["reference_seconds_per_fit"] = per_fit
            row["reference_panel_h"] = _hours(
                tally["panel_reference_fits_per_pair"] * per_fit
            )
            row["reference_development_h"] = _hours(
                tally["development_reference_fit_equivalents_per_pair"][pair] * per_fit
            )
        out[pair] = row
    return out, []


def print_tally(protocol, prices):
    """Print the compute tally, priced per pair when probe data match."""
    tally = compute_tally(protocol)
    print("Compute tally")
    for k, v in tally.items():
        if isinstance(v, dict):
            v = ", ".join(f"{pair} {n:,}" for pair, n in v.items())
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:,}")
    if prices is None:
        upd = tally["panel_update_equivalents_per_pair"]
        for s in (0.1, 0.3, 1.0):
            print(f"  updater panel per pair at {s} s/update: {_hours(upd * s):.1f} h")
        print("  (no probe prices given; pass --prices to price the tally)")
        return
    priced, reasons = priced_tally(protocol, prices)
    if priced is None:
        print(f"  probe does not match the protocol budget: {'; '.join(reasons)}")
        return
    total = 0.0
    for pair, row in priced.items():
        total += row["updater_panel_h"] + row.get("reference_panel_h", 0.0)
        line = (
            f"  {pair}: {row['seconds_per_update']:.3g} s/update, "
            f"updater panel {row['updater_panel_h']:.2f} h"
        )
        if "reference_seconds_per_fit" in row:
            line += (
                f"; reference {row['reference_seconds_per_fit']:.2f} s/fit, "
                f"panel {row['reference_panel_h']:.1f} h, "
                f"development {row['reference_development_h']:.1f} h"
            )
        print(line)
    print(f"  panel total {total:.1f} h on one slot (likelihood-only updater price)")
    measured = measured_tally(protocol)
    if measured is None:
        return
    print(f"Repriced at measured cost ({measured['source']})")
    for pair, row in measured["pairs"].items():
        print(
            f"  {pair}: {row['seconds_per_update']:.3g} s/update, "
            f"{row['reference_seconds_per_fit']:.3g} s/fit; updater panel "
            f"{row['updater_panel_h']:.1f} h, updater development "
            f"{row['updater_development_h']:.1f} h, reference panel "
            f"{row['reference_panel_h']:.1f} h, reference development "
            f"{row['reference_development_h']:.1f} h"
        )
    print(
        f"  development {measured['development_h']:.1f} h, panel "
        f"{measured['panel_h']:.1f} h, on one slot"
    )


def measured_tally(protocol):
    """Hours per pair at the measured full-kernel and reference costs.

    The protocol's ``compute`` block holds warm seconds per update (the whole
    tempered kernel at the budget ceiling, not the likelihood alone) and
    seconds per reference fit at the protocol budget, measured on one slot.
    Returns None when the block is absent.
    """
    block = protocol.get("compute")
    if not block:
        return None
    tally = compute_tally(protocol)
    pairs = {}
    dev_h = panel_h = 0.0
    for pair in protocol["panel"]["mass_models"]:
        s_upd = block["measured_seconds_per_update"][pair]
        s_fit = block["measured_reference_seconds_per_fit"][pair]
        row = {
            "seconds_per_update": s_upd,
            "reference_seconds_per_fit": s_fit,
            "updater_panel_h": _hours(
                tally["panel_update_equivalents_per_pair"] * s_upd
            ),
            "updater_development_h": _hours(
                tally["development_update_equivalents_per_pair"] * s_upd
            ),
            "reference_panel_h": _hours(tally["panel_reference_fits_per_pair"] * s_fit),
            "reference_development_h": _hours(
                tally["development_reference_fit_equivalents_per_pair"][pair] * s_fit
            ),
        }
        dev_h += row["updater_development_h"] + row["reference_development_h"]
        panel_h += row["updater_panel_h"] + row["reference_panel_h"]
        pairs[pair] = row
    return {
        "source": block["source"],
        "pairs": pairs,
        "development_h": dev_h,
        "panel_h": panel_h,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def protocol_commit(path=PROTOCOL):
    """Latest commit that changed the protocol file, or None outside git."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def dry_run(protocol, prices=None):
    """Print the full allocation of the panel without running it."""
    import jax

    print(
        f"Protocol {protocol['protocol']} v{protocol['version']} ({protocol['status']})"
    )
    print(f"Protocol commit: {protocol_commit() or 'uncommitted'}")
    print(f"Provisional: {', '.join(protocol['provisional'])}")

    cell_list = cells(protocol)
    print(
        f"\nCells ({len(cell_list)}), "
        f"each {protocol['panel']['worlds_per_cell']} worlds"
    )
    for regime, mass_model, epochs, quantity in cell_list:
        print(f"  {regime} {mass_model:15s} epochs={epochs} {quantity}")

    print("\nModel declarations (SHA-256 of canonical JSON)")
    for regime in protocol["regimes"]:
        for mass_model in protocol["panel"]["mass_models"]:
            decl = model_declaration(protocol, regime, mass_model)
            print(f"  {regime['id']} {mass_model:15s} {declaration_hash(decl)}")

    print("\nRegime designs")
    iwa, owa = visibility_mas(protocol)
    for regime in protocol["regimes"]:
        a_lo, a_hi = physical_semimajor_range(protocol, regime)
        t_f = [future_epoch(regime, k) for k in protocol["panel"]["epoch_counts"]]
        print(
            f"  {regime['id']} {regime['name']}: cadence {regime['cadence_days']} d, "
            f"sigma {regime['sigma_mas']} mas, "
            f"truncation {regime['truncation'] or 'none'}, "
            f"future epochs {t_f} d, physical a {a_lo:.4f} to {a_hi:.4f} AU"
        )
    print(f"  visibility: {iwa:.1f} to {owa:.1f} mas")
    sched = protocol["criteria"]["schedule_stability"]
    print(
        f"  schedule tolerance: d > {sched['tolerance']} in more than "
        f"{sched['max_count']} of {protocol['panel']['worlds_per_cell']} worlds fails"
    )

    ti_ref = protocol["reference"]["ti_window"]
    print(
        f"\nTI reference (amendment A7): {ti_ref['method']}, "
        f"{ti_ref['proposals_per_fit']:,} proposals per fit (at most "
        f"{ti_ref['max_proposals_per_fit']:,}); independent-proposal check: "
        f"{ti_ref['independent_method']} at twice the budget"
    )
    ref = protocol["reference"]
    phys = ref["physical_fixed"]
    print(
        f"Physical reference: {phys['proposals_per_fit']:,} proposals per fit "
        f"(at most {phys['max_proposals_per_fit']:,}), anchor {phys['anchor']}"
    )
    print(
        "Reference certification: repeat noise "
        f"<= {ref['repeat_noise']['bound_fraction_of_delta2']} delta2, convergence "
        f"<= {ref['convergence']['bound_fraction_of_delta2']} delta2, independent "
        f"proposal <= {ref['cross_method']['bound_fraction_of_delta2']} delta2, "
        "one-epoch exactness"
    )

    print(
        f"\nWorld and seed allocation (typed {protocol['prng']['implementation']} "
        f"keys, root seed {protocol['prng']['root_seed']})"
    )
    for stage in protocol["prng"]["stages"]:
        n = stage_worlds(protocol, stage)
        for m in protocol["panel"]["mass_models"]:
            for r_index, regime in enumerate(protocol["regimes"]):
                key = world_key(protocol, stage, m, r_index, 0)
                print(
                    f"  {stage:11s} {m:15s} {regime['id']} worlds 0..{n - 1}; "
                    f"world 0 key {list(map(int, jax.random.key_data(key)))}"
                )
    print(f"  alias worlds: 0..{protocol['panel']['alias_subset_worlds'] - 1}")
    print(f"  simulation seed streams: {protocol['prng']['simulation_seeds']}")
    print(f"  purposes: {protocol['prng']['purposes']}")

    print("\nCritical values (from the formulas)")
    values = critical_values(protocol)
    for k, v in values.items():
        print(f"  {k}: {v}")
    stated = protocol["stated"]
    mismatched = [k for k, v in stated.items() if values[k] != v]
    print(
        f"  stated values match: {not mismatched}"
        + (f" {mismatched}" if mismatched else "")
    )

    print()
    print_tally(protocol, prices)
    return values


def d2_pass_rates(
    rng,
    damaged,
    ess_u,
    ess_r=4400,
    worlds=200,
    panels=200,
    delta2=5e-4,
    z=3.078,
    bins=8,
    moved=0.2,
):
    """Seeded multinomial simulation of the alias test pass probability."""
    import numpy as np

    passes = 0
    for _ in range(panels):
        p = rng.dirichlet(np.full(bins, 0.7), size=worlds)
        q = p.copy()
        hit = np.flatnonzero(rng.random(worlds) < damaged)
        order = np.argsort(q[hit], axis=1)[:, ::-1]
        rows = hit[:, None]
        take = np.minimum(moved, q[rows, order[:, 1:2]])
        q[rows, order[:, 1:2]] -= take
        q[rows, order[:, 0:1]] += take

        def est(x, ess):
            return rng.multinomial(ess, x) / ess

        d2 = ((est(q, ess_u) - est(p, ess_r)) * (est(q, ess_u) - est(p, ess_r))).sum(1)
        mean, se = d2.mean(), d2.std(ddof=1) / np.sqrt(worlds)
        passes += mean + z * se <= delta2
    return passes / panels


def canonical_groups(protocol):
    """Every (regime, mass model, epoch count) group, in parameter-block order."""
    return [
        (r["id"], m, k)
        for r in protocol["regimes"]
        for m in protocol["panel"]["mass_models"]
        for k in protocol["panel"]["epoch_counts"]
    ]


def validate_summary(protocol, summary):
    """Map each canonical group to its entry, worlds sorted by id.

    Refuses a missing, duplicate or unknown group, and a group whose world ids
    are not exactly the development ids (0 to n - 1, each once).
    """
    expected = canonical_groups(protocol)
    n_dev = protocol["panel"]["development_worlds_per_regime_and_mass_model"]
    expected_ids = set(range(n_dev))
    seen = {}
    for entry in summary["groups"]:
        ident = (entry["regime"], entry["mass_model"], entry["epoch_count"])
        if ident not in expected:
            raise ValueError(f"unknown group {ident}")
        if ident in seen:
            raise ValueError(f"duplicate group {ident}")
        ids = [w["world"] for w in entry["worlds"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"group {ident} repeats a development world id")
        if set(ids) != expected_ids:
            missing_ids = sorted(expected_ids - set(ids))
            unknown_ids = sorted(set(ids) - expected_ids)
            raise ValueError(
                f"group {ident} development worlds: missing {missing_ids}, "
                f"unknown {unknown_ids}"
            )
        seen[ident] = dict(
            entry, worlds=sorted(entry["worlds"], key=lambda w: w["world"])
        )
    missing = [g for g in expected if g not in seen]
    if missing:
        raise ValueError(f"missing groups {missing}")
    return seen


def _updater_ess(world, quantity):
    """A world's updater precision for one quantity (a number or a dict)."""
    value = world["updater_ess"]
    return value[quantity] if isinstance(value, dict) else value


def freeze_alias(protocol, summary, panels=None):
    """Alias tolerance from development summaries, with its bound and power floor.

    ``summary["groups"]`` holds exactly one entry per canonical group with
    ``regime``, ``mass_model``, ``epoch_count`` and ``worlds``; each world has
    its id ``world``, ``updater_ess`` (one number, or a dict from alias
    quantity to the key-to-key precision of ``key_to_key_ess``),
    ``reference_ess`` and ``bin_masses`` (a dict from alias quantity to an
    eight-vector). Whole worlds are resampled,
    so masses and both ESS values stay together. Returns the candidate
    ``delta2``, per-group standard errors, power with its one-sided 95 percent
    lower bound, the exact-updater pass rate with its 95 percent interval, and
    ``status`` (``frozen`` or ``incomplete``) with the reasons.
    """
    import numpy as np

    alias = protocol["criteria"]["alias_accuracy"]
    alt = alias["alternative"]
    n_worlds = protocol["panel"]["alias_subset_worlds"]
    panels = panels or alias["freeze_panels"]
    z = critical_values(protocol)["alias_z"]
    stream = protocol["prng"]["simulation_seeds"]["alias_freeze"]
    root = protocol["prng"]["root_seed"]
    entries = validate_summary(protocol, summary)
    regimes = [r["id"] for r in protocol["regimes"]]
    masses_order = protocol["panel"]["mass_models"]
    epochs_order = protocol["panel"]["epoch_counts"]

    def panel_stats(rng, masses, ess_u, ess_r, damaged):
        pick = rng.integers(0, len(masses), n_worlds)
        p, nu, nr = masses[pick], ess_u[pick, None], ess_r[pick, None]
        q = p.copy()
        hit = np.flatnonzero(rng.random(n_worlds) < damaged)
        order = np.argsort(q[hit], axis=1)[:, ::-1]
        rows = hit[:, None]
        take = np.minimum(alt["moved_mass"], q[rows, order[:, 1:2]])
        q[rows, order[:, 1:2]] -= take
        q[rows, order[:, 0:1]] += take
        u_a, u_b = (rng.multinomial(nu[:, 0], q) / nu for _ in range(2))
        r_a, r_b = (rng.multinomial(nr[:, 0], p) / nr for _ in range(2))
        d2 = ((u_a - r_a) * (u_b - r_b)).sum(1)
        return d2.mean(), d2.std(ddof=1) / np.sqrt(n_worlds)

    rows = []
    for ident in canonical_groups(protocol):
        worlds = entries[ident]["worlds"]
        ess_r = np.array([max(round(w["reference_ess"]), 1) for w in worlds])
        canon = [
            regimes.index(ident[0]),
            masses_order.index(ident[1]),
            epochs_order.index(ident[2]),
        ]
        for q_index, quantity in enumerate(alias["quantities"]):
            ess_u = np.array([max(round(_updater_ess(w, quantity)), 1) for w in worlds])
            masses = np.asarray([w["bin_masses"][quantity] for w in worlds], float)
            masses = masses / masses.sum(axis=1, keepdims=True)
            rng = np.random.default_rng([root, stream, *canon, q_index])
            null = [panel_stats(rng, masses, ess_u, ess_r, 0.0) for _ in range(panels)]
            damaged = [
                panel_stats(rng, masses, ess_u, ess_r, alt["damaged_fraction"])
                for _ in range(panels)
            ]
            rows.append(
                {
                    "regime": ident[0],
                    "mass_model": ident[1],
                    "epoch_count": ident[2],
                    "quantity": quantity,
                    "se0": float(np.mean([se for _, se in null])),
                    "null": null,
                    "damaged": damaged,
                }
            )
    delta2 = max(alias["delta2"], 2.0 * z * max(r["se0"] for r in rows))
    reasons = []
    if delta2 > alias["delta2_max"]:
        reasons.append(f"delta2 {delta2:.3g} exceeds its bound {alias['delta2_max']}")
    graded = min(delta2, alias["delta2_max"])
    for r in rows:
        null, dmg = r.pop("null"), r.pop("damaged")
        fails = sum(m + z * se > graded for m, se in dmg)
        passes = sum(m + z * se <= graded for m, se in null)
        r["power"] = fails / panels
        r["power_lower_95"] = clopper_pearson_lower(fails, panels)
        r["exact_pass_rate"] = passes / panels
        r["exact_pass_interval_95"] = [
            clopper_pearson_lower(passes, panels, 0.975),
            clopper_pearson_upper(passes, panels, 0.975),
        ]
        if r["power_lower_95"] < alias["min_power"]:
            reasons.append(
                f"{r['regime']} {r['mass_model']} {r['epoch_count']} {r['quantity']}: "
                f"power lower bound {r['power_lower_95']:.3f} "
                f"below {alias['min_power']}"
            )
    return {
        "delta2": delta2,
        "panels": panels,
        "status": "incomplete" if reasons else "frozen",
        "reasons": reasons,
        "groups": rows,
    }


def sizing(protocol):
    """Print the power of each criterion against its stated alternatives."""
    import numpy as np

    values = critical_values(protocol)
    n = protocol["panel"]["worlds_per_cell"]
    c = values["coverage_critical_count"]
    print("Coverage: P(cell fails) at true coverage")
    for p in (0.88, 0.86, 0.85, 0.84, 0.82, 0.80):
        print(f"  {p}: {binom_cdf(c, n, p):.3f}")
    fu = values["failed_updates_critical_count"]
    print("Failed updates: P(group fails) at true rate")
    for r in (0.03, 0.04, 0.05, 0.06, 0.08):
        print(f"  {r}: {binom_sf(fu, n, r):.3f}")
    sched = protocol["criteria"]["schedule_stability"]
    tol, cap = sched["tolerance"], sched["max_count"]
    print(f"Schedule tolerance: fail when more than {cap} of {n} worlds have d > {tol}")
    for r in (0.03, 0.05, 0.08, 0.10):
        print(
            f"  {r} of worlds exceed: P(comparison fails) = {binom_sf(cap, n, r):.4f}"
        )
    print("  orientation, estimates like m independent draws at p = 0.5:")
    for m in (300, 500, 1000, 2000):
        q = 2.0 * (1.0 - statistics.NormalDist().cdf(tol / math.sqrt(0.5 / m)))
        print(
            f"    m={m}: P(d > {tol}) = {q:.4f}, P(fails) = {binom_sf(cap, n, q):.4f}"
        )
    alias = protocol["criteria"]["alias_accuracy"]
    worlds = protocol["panel"]["alias_subset_worlds"]
    frac = alias["alternative"]["damaged_fraction"]
    print(
        f"Alias accuracy: {worlds} worlds; P(no damaged world | fraction {frac}) = "
        f"{(1.0 - frac) ** worlds:.2e} "
        f"(per-test level {0.05 / values['alias_tests']:.2e})"
    )
    print("  P(group passes), seeded simulation, 200 panels per case")
    root = protocol["prng"]["root_seed"]
    stream = protocol["prng"]["simulation_seeds"]["sizing"]
    cases = [
        (d2, e, f)
        for d2 in (alias["delta2"], alias["delta2_max"])
        for e in (1000, 300)
        for f in (0.0, 0.01, frac, 0.04)
    ]
    for i, (d2, ess, f) in enumerate(cases):
        rng = np.random.default_rng([root, stream, i])
        rate = d2_pass_rates(
            rng,
            f,
            ess,
            worlds=worlds,
            delta2=d2,
            z=values["alias_z"],
            bins=alias["bins"],
            moved=alias["alternative"]["moved_mass"],
        )
        print(f"  delta2 {d2:g}, updater ESS {ess}, damaged fraction {f}: {rate:.3f}")
    rate, lo, hi = exact_check_control(protocol)
    print(
        f"Exact checks: m = {values['exact_check_compared_values']}, "
        f"t* = {values['exact_check_t_critical']:.4f}; control family rejection "
        f"{rate:.4f} (95 percent interval {lo:.4f} to {hi:.4f})"
    )


def exact_check_control(protocol, sims=20000, ess=1000):
    """Seeded control of the exact-check family false-rejection rate.

    A correct sampler is simulated: per case, three eight-bin marginals with
    Dirichlet(0.7) masses estimated by multinomials at ``ess``, and four
    Gaussian interval end points; ``sampler_repeats`` runs each. Returns the
    fraction of simulated sets with any value outside its tolerance (zero
    quadrature error, the bin-mass floor applied) and a two-sided 95 percent
    Clopper-Pearson interval.
    """
    import numpy as np

    ex = protocol["reference"]["exact_checks"]
    values = critical_values(protocol)
    t_crit = values["exact_check_t_critical"]
    reps = ex["sampler_repeats"]
    n_marg = ex["cases"] * ex["bins_per_case"] // 8
    n_end = ex["cases"] * ex["interval_end_points_per_case"]
    root = protocol["prng"]["root_seed"]
    rng = np.random.default_rng(
        [root, protocol["prng"]["simulation_seeds"]["sizing"], 99]
    )
    masses = rng.dirichlet(np.full(8, 0.7), size=n_marg)
    bins = rng.multinomial(ess, masses, size=(sims, reps, n_marg)) / ess
    bins = bins.reshape(sims, reps, -1)
    ends = rng.standard_normal((sims, reps, n_end))
    x = np.concatenate([bins, ends], axis=2)
    truth = np.concatenate([masses.reshape(-1), np.zeros(n_end)])
    dev = np.abs(x.mean(axis=1) - truth)
    tol = t_crit * x.std(axis=1, ddof=1) / math.sqrt(reps)
    n_bins = bins.shape[2]
    tol[:, :n_bins] = np.maximum(tol[:, :n_bins], ex["bin_mass_floor"])
    fails = int((dev > tol).any(axis=1).sum())
    return (
        fails / sims,
        clopper_pearson_lower(fails, sims, 0.975),
        clopper_pearson_upper(fails, sims, 0.975),
    )


# ---------------------------------------------------------------------------
# Worlds: targets, truths and records
# ---------------------------------------------------------------------------

QUANTITIES = ("e", "lnP", "future_east", "future_north")
INTERVALS = {"90": (0.05, 0.95), "50": (0.25, 0.75)}


def enable_x64():
    """Switch JAX to 64-bit mode (the protocol's float)."""
    import jax

    jax.config.update("jax_enable_x64", True)


def stellar_mass_kg(protocol):
    """The physical pair's fixed stellar mass (kg)."""
    from hwoutils.constants import Msun2kg

    msun = protocol["model_pairs"]["physical_fixed"]["defaults"]["stellar_mass_msun"]
    return msun * float(Msun2kg)


def update_target(protocol, regime, mass_model):
    """The updater target of a mass model under a regime."""
    from photomancy.orbit.updater import FixedMassTarget, ThieleInnesTarget

    if mass_model == "ti_window":
        return ThieleInnesTarget(ti_model(protocol, regime), protocol["distance_pc"])
    p_lo, p_hi, e_max = regime_bounds(protocol, regime, "physical_fixed")
    return FixedMassTarget(
        Ms_kg=stellar_mass_kg(protocol),
        dist_pc=protocol["distance_pc"],
        period_lo_day=p_lo,
        period_hi_day=p_hi,
        e_max=e_max,
        t_ref_day=regime["cadence_days"][0],
    )


def world_data(protocol, stage, mass_model, r_index, world):
    """Truth (in the updater's chart) and noisy records of one world.

    Returns a dict with ``target``, ``truth`` (chart point), ``times``,
    ``ra`` and ``dec`` (arcsec) and ``sigma`` (arcsec).
    """
    import jax
    import numpy as np
    from hwoutils.constants import mas2arcsec

    regime = protocol["regimes"][r_index]
    target = update_target(protocol, regime, mass_model)
    truth_key = draw_key(protocol, stage, mass_model, r_index, world, "truth", 0)
    noise_key = draw_key(protocol, stage, mass_model, r_index, world, "noise", 0)
    if mass_model == "ti_window":
        truth = ti_truth(ti_model(protocol, regime), truth_key)
    else:
        truth = target.from_physical(physical_truth(protocol, regime, truth_key))
    times = np.asarray(regime["cadence_days"], float)
    sigma = regime["sigma_mas"] * mas2arcsec
    with jax.enable_x64(True):
        ra, dec = target.predict(truth, times)
        noise = sigma * jax.random.normal(noise_key, (2, len(times)))
    return {
        "target": target,
        "truth": truth,
        "times": times,
        "ra": np.asarray(ra + noise[0]),
        "dec": np.asarray(dec + noise[1]),
        "sigma": sigma,
    }


def astrometry(world, epochs, mask=None):
    """The first ``epochs`` records as ``RelativeAstromData`` (optional mask)."""
    import jax.numpy as jnp

    from photomancy.orbit.data import RelativeAstromData

    valid = jnp.ones(epochs, bool) if mask is None else jnp.asarray(mask)
    return RelativeAstromData(
        times=jnp.asarray(world["times"][:epochs]),
        ra=jnp.asarray(world["ra"][:epochs]),
        dec=jnp.asarray(world["dec"][:epochs]),
        ra_err=jnp.full(epochs, world["sigma"]),
        dec_err=jnp.full(epochs, world["sigma"]),
        corr=jnp.zeros(epochs),
        planet_id=jnp.zeros(epochs, dtype=int),
        is_valid=valid,
    )


def _chart_quantities_fn(mass_model):
    import jax
    import jax.numpy as jnp
    from hwoutils.constants import G, two_pi
    from orbix.equations.orbit import mean_motion

    @jax.jit
    def fn(target, particles, t_f):
        times = jnp.reshape(t_f, (1,))
        east, north = jax.vmap(lambda th: target.predict(th, times))(particles)
        if mass_model == "ti_window":
            e, ln_p = particles[:, 0], particles[:, 1]
        else:
            phys = jax.vmap(target.to_physical)(particles)
            e = phys[:, 1]
            ln_p = jnp.log(two_pi / mean_motion(phys[:, 0], G * target.Ms_kg))
        return e, ln_p, east[:, 0], north[:, 0]

    return fn


_QUANTITY_FNS = {}


def chart_quantities(mass_model, target, particles, t_f):
    """Graded quantities of chart points: e, lnP and the future position."""
    import numpy as np

    if mass_model not in _QUANTITY_FNS:
        _QUANTITY_FNS[mass_model] = _chart_quantities_fn(mass_model)
    out = _QUANTITY_FNS[mass_model](target, particles, float(t_f))
    return dict(zip(QUANTITIES, (np.asarray(x) for x in out), strict=True))


def finite_chart_quantities(mass_model, target, particles, log_weights, t_f, chunk):
    """Graded quantities and log weights of the finite-weight particles.

    The quantities are evaluated in blocks of ``chunk`` particles (the last
    block padded), so the compiled quantity function sees one shape and the
    memory stays bounded; selecting the finite-weight particles first would
    compile once per distinct count.
    """
    import numpy as np

    particles = np.asarray(particles)
    log_weights = np.asarray(log_weights)
    n = len(particles)
    parts = {name: [] for name in QUANTITIES}
    for start in range(0, n, chunk):
        fin = np.isfinite(log_weights[start : start + chunk])
        if not fin.any():
            continue
        block = particles[start : start + chunk]
        if len(block) < chunk:
            block = np.concatenate(
                [block, np.repeat(block[-1:], chunk - len(block), 0)]
            )
        q = chart_quantities(mass_model, target, block, t_f)
        for name in QUANTITIES:
            parts[name].append(np.asarray(q[name])[: len(fin)][fin])
    lw = log_weights[np.isfinite(log_weights)]
    if not len(lw):
        return {name: np.zeros(0) for name in QUANTITIES}, lw
    return {name: np.concatenate(v) for name, v in parts.items()}, lw


def _physical_rows_fn():
    import jax
    import jax.numpy as jnp
    from hwoutils.constants import G, two_pi
    from orbix.equations.orbit import mean_motion

    from photomancy.orbit.forward import predict_relative_astrometry

    @jax.jit
    def fn(rows, times, ms_kg, dist_pc):
        def one(row):
            a, e, cos_i, w_node, cos_w, sin_w, tp = row
            return predict_relative_astrometry(
                times, a, e, cos_i, w_node, cos_w, sin_w, tp, ms_kg, dist_pc
            )

        east, north = jax.vmap(one)(rows)
        ln_p = jnp.log(two_pi / mean_motion(rows[:, 0], G * ms_kg))
        return rows[:, 1], ln_p, east, north

    return fn


def truth_quantities(protocol, mass_model, world, epoch_count):
    """The truth's graded quantities at the future epoch of ``epoch_count``."""
    import numpy as np

    regime_tf = future_epoch({"cadence_days": list(world["times"])}, epoch_count)
    q = chart_quantities(
        mass_model, world["target"], np.asarray(world["truth"])[None], regime_tf
    )
    return {k: float(v[0]) for k, v in q.items()}


# ---------------------------------------------------------------------------
# Weighted posterior summaries
# ---------------------------------------------------------------------------


def alias_edges(protocol, regime, mass_model):
    """Eight-bin edges of each alias quantity over the regime's support."""
    import numpy as np

    p_lo, p_hi, e_max = regime_bounds(protocol, regime, mass_model)
    n_bins = protocol["criteria"]["alias_accuracy"]["bins"]
    return {
        "lnP": np.linspace(math.log(p_lo), math.log(p_hi), n_bins + 1),
        "e": np.linspace(0.0, e_max, n_bins + 1),
        "future_pa": np.linspace(0.0, 2.0 * math.pi, n_bins + 1),
    }


def weighted_quantile(values, weights, level):
    """First sorted value whose cumulative normalized weight reaches ``level``."""
    import numpy as np

    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    cum /= cum[-1]
    idx = int(np.searchsorted(cum, level * (1.0 - 1e-12), side="left"))
    return float(values[order][min(idx, len(order) - 1)])


def summarize_posterior(q, log_weights, edges, visibility, truth=None, pit_v=None):
    """Graded summary of one weighted posterior.

    Args:
        q: Dict of per-particle quantities (``QUANTITIES``; positions in arcsec).
        log_weights: Unnormalized log weights; non-finite entries are dropped.
        edges: Bin edges of the alias quantities (``alias_edges``).
        visibility: Inner and outer working angles (mas).
        truth: Truth value per quantity, for the randomized PIT.
        pit_v: Uniform draw per quantity for the randomized PIT.

    Returns:
        Dict with ``kish_ess``, ``bins`` (alias quantity to masses),
        ``intervals`` (quantity to ``{"90": [lo, hi], "50": [lo, hi]}``),
        ``schedule_p`` and, with a truth, ``pit``.
    """
    import numpy as np

    lw = np.asarray(log_weights, float)
    keep = np.isfinite(lw)
    if not keep.any():
        return {"kish_ess": 0.0, "particles": 0}
    lw = lw[keep]
    w = np.exp(lw - lw.max())
    w /= w.sum()
    vals = {k: np.asarray(v, float)[keep] for k, v in q.items()}
    vals["future_pa"] = np.mod(
        np.arctan2(vals["future_east"], vals["future_north"]), 2 * math.pi
    )
    sep_mas = 1000.0 * np.hypot(vals["future_east"], vals["future_north"])
    out = {
        "kish_ess": float(1.0 / np.sum(w**2)),
        "particles": int(keep.sum()),
        "bins": {},
        "intervals": {},
        "schedule_p": float(
            w[(sep_mas >= visibility[0]) & (sep_mas <= visibility[1])].sum()
        ),
    }
    for name, e in edges.items():
        idx = np.clip(np.searchsorted(e, vals[name], side="right") - 1, 0, len(e) - 2)
        out["bins"][name] = np.bincount(idx, weights=w, minlength=len(e) - 1).tolist()
    for name in QUANTITIES:
        out["intervals"][name] = {
            label: [
                weighted_quantile(vals[name], w, lo),
                weighted_quantile(vals[name], w, hi),
            ]
            for label, (lo, hi) in INTERVALS.items()
        }
    if truth is not None:
        out["pit"] = {}
        for i, name in enumerate(QUANTITIES):
            below = w[vals[name] < truth[name]].sum()
            at_or_below = w[vals[name] <= truth[name]].sum()
            out["pit"][name] = float(below + pit_v[i] * (at_or_below - below))
    return out


# ---------------------------------------------------------------------------
# Work units, sharding and resume
# ---------------------------------------------------------------------------

UPDATER_ROLES = ("updater_a", "updater_b", "updater_doubled")
# Six more nominal-budget updater runs per development world: with updater_a
# and updater_b they give eight replicates for the freeze's precision estimate.
PRECISION_ROLES = tuple(f"updater_rep_{i}" for i in range(1, 7))
ALL_UPDATER_ROLES = UPDATER_ROLES + PRECISION_ROLES
REFERENCE_ROLES = ("reference_1", "reference_2")
CERTIFICATION_ROLES = (
    "reference_3",
    "reference_4",
    "cert_doubled_1",
    "cert_doubled_2",
    "cert_proposal_1",
    "cert_proposal_2",
)
PARTS = {
    "updater": UPDATER_ROLES,
    "precision": PRECISION_ROLES,
    "reference": REFERENCE_ROLES,
    "certification": CERTIFICATION_ROLES,
}
DEVELOPMENT_ONLY_PARTS = ("precision", "certification")


def unit_id(stage, mass_model, regime_id, world, role, epoch_count=None):
    """Canonical identity of one work unit."""
    tail = f"/k{epoch_count}" if epoch_count is not None else ""
    return f"{stage}/{mass_model}/{regime_id}/w{world:03d}/{role}{tail}"


def plan_units(protocol, stage, parts, mass_models=None, regimes=None, worlds=None):
    """Every work unit of a stage, in canonical order.

    An updater unit runs the three updates of one world and one run role; a
    reference unit runs one reference fit for one world, role and epoch
    count. Precision and certification roles exist on the development stage
    only.
    """
    panel = protocol["panel"]
    mass_models = mass_models or panel["mass_models"]
    regimes = regimes or [r["id"] for r in protocol["regimes"]]
    worlds = range(stage_worlds(protocol, stage)) if worlds is None else worlds
    roles = []
    for part in parts:
        if part in DEVELOPMENT_ONLY_PARTS and stage != "development":
            continue
        roles += PARTS[part]
    out = []
    for mass_model in panel["mass_models"]:
        if mass_model not in mass_models:
            continue
        for r_index, regime in enumerate(protocol["regimes"]):
            if regime["id"] not in regimes:
                continue
            for world in worlds:
                for role in roles:
                    counts = (
                        [None] if role in ALL_UPDATER_ROLES else panel["epoch_counts"]
                    )
                    for k in counts:
                        out.append(
                            {
                                "id": unit_id(
                                    stage, mass_model, regime["id"], world, role, k
                                ),
                                "stage": stage,
                                "mass_model": mass_model,
                                "regime": regime["id"],
                                "r_index": r_index,
                                "world": world,
                                "role": role,
                                "epoch_count": k,
                            }
                        )
    return out


def parse_shard(text):
    """``"i/n"`` with ``0 <= i < n`` to ``(i, n)``."""
    i, n = (int(x) for x in text.split("/"))
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"shard {text!r}: need i/n with 0 <= i < n")
    return i, n


def shard_units(units, shard):
    """Units of shard ``i`` of ``n``: every ``n``-th unit from ``i``."""
    i, n = shard
    return units[i::n]


def parse_worlds(text):
    """World selection such as ``0-5`` or ``0,3,7``."""
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out += range(lo, hi + 1)
        else:
            out.append(int(part))
    return sorted(set(out))


def read_records(out_dir):
    """Every complete JSON line of every unit file in ``out_dir``.

    A truncated last line (a process killed mid-write) is skipped. Returns
    the records in file order.
    """
    records = []
    for path in sorted(Path(out_dir).glob("units-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_units(out_dir):
    """Map unit id to its first complete record, and the duplicate ids."""
    units, duplicates = {}, []
    for rec in read_records(out_dir):
        if rec["id"] in units:
            duplicates.append(rec["id"])
        else:
            units[rec["id"]] = rec
    return units, duplicates


def append_record(path, record):
    """Append one record as a JSON line, flushed and synced.

    A file whose last byte is not a newline (a killed writer) gets one first,
    so the new record starts its own line.
    """
    import os

    path = Path(path)
    with open(path, "a+b") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())


class IncompatibleRunError(RuntimeError):
    """A run directory holds records that the current run cannot reuse."""


def run_units(units, out_dir, shard, execute, log=print, compatible=None):
    """Run the units of one shard that no unit file in ``out_dir`` holds yet.

    Each finished unit is appended to ``units-{i}of{n}.jsonl`` at once, so a
    killed run loses at most the unit in progress, and a restart skips every
    unit any shard has completed. With ``compatible`` (a predicate on a
    record), a directory holding any record it rejects is refused before
    anything runs (``IncompatibleRunError``), so an earlier test override, a
    budget change or a code change cannot be skipped over silently.

    Returns:
        Dict with the counts ``planned``, ``skipped`` and ``ran``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mine = shard_units(units, shard)
    existing = load_units(out_dir)[0]
    if compatible is not None:
        bad = sorted(i for i, rec in existing.items() if not compatible(rec))
        if bad:
            raise IncompatibleRunError(
                f"{out_dir} holds {len(bad)} records this run cannot reuse "
                "(another code or protocol content, or other budgets); first: "
                f"{bad[:3]}. Use a fresh directory or rerun with the same "
                "code, protocol and budget arguments."
            )
    done = set(existing)
    todo = [u for u in mine if u["id"] not in done]
    path = out_dir / f"units-{shard[0]}of{shard[1]}.jsonl"
    for n, unit in enumerate(todo, 1):
        record = execute(unit)
        record["id"] = unit["id"]
        append_record(path, record)
        log(f"[{n}/{len(todo)}] {unit['id']} {record.get('seconds', 0.0):.1f} s")
    return {"planned": len(mine), "skipped": len(mine) - len(todo), "ran": len(todo)}


def code_commit():
    """HEAD of the repository and whether the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(dirty)


def write_manifest(out_dir, shard, entry):
    """Append an invocation entry to the shard's manifest, or update it.

    An entry whose ``started`` time matches the last entry replaces it, so a
    run writes its entry when it starts (a killed run still leaves its
    provenance) and completes it when it finishes.
    """
    path = Path(out_dir) / f"manifest-{shard[0]}of{shard[1]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {"invocations": []}
    runs = data["invocations"]
    if runs and runs[-1].get("started") == entry.get("started"):
        runs[-1] = entry
    else:
        runs.append(entry)
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def atomic_write_text(path, text):
    """Replace ``path`` with ``text`` through a synced temporary file.

    The rename is atomic, so a kill at any point leaves either the old file
    or the new one, never a truncated one.
    """
    import os
    import tempfile

    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_entry(protocol, args, budgets):
    """Provenance of one invocation: commits, versions, host and budgets."""
    import datetime
    import os
    import platform

    import jax
    import jaxlib

    head, dirty = code_commit()
    return {
        "started": datetime.datetime.now(datetime.UTC).isoformat(),
        "protocol_commit": protocol_commit(),
        "protocol_status": protocol["status"],
        "protocol_hash": declaration_hash(protocol),
        "run_hash": run_hash(protocol),
        "code_commit": head,
        "code_dirty": dirty,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "platform": platform.platform(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "argv": sys.argv[1:] if args is None else args,
        "budgets": budgets,
    }


# ---------------------------------------------------------------------------
# Executing units
# ---------------------------------------------------------------------------


def protocol_budgets(
    protocol, particles=None, ti_proposals=None, physical_proposals=None
):
    """Updater particles and reference proposals, with optional overrides."""
    ref = protocol["reference"]
    return {
        "particles": particles or protocol["updater_budget"]["particles"],
        "rungs": protocol["updater_budget"]["rungs"],
        "moves_per_rung": protocol["updater_budget"]["moves_per_rung"],
        "ti_window": ti_proposals or ref["ti_window"]["proposals_per_fit"],
        "physical_fixed": physical_proposals
        or ref["physical_fixed"]["proposals_per_fit"],
    }


# Library modules whose source determines what a unit computes (the updater
# and its targets, the reference samplers and the forward model).
RECORD_MODULES = (
    "photomancy.orbit.updater",
    "photomancy.orbit.few_epoch",
    "photomancy.orbit.ofti",
    "photomancy.orbit.forward",
    "photomancy.orbit.data",
    "photomancy.orbit.likelihoods",
    "photomancy.orbit.thiele_innes",
    "photomancy.posterior",
)


def _record_definitions():
    """Functions of this tool whose source determines what a unit computes."""
    return (
        regime_bounds,
        model_declaration,
        semimajor_axis_au,
        TI_FREQUENCY,
        _ti_prior_u,
        _collapsed_draws,
        ti_collapsed_reference,
        ti_frequency_reference,
        frequency_mixture_ln_period,
        frequency_mixture_log_ratio,
        inflated_conditional_draw,
        _jitted,
        reference_budget_factor,
        physical_semimajor_range,
        visibility_mas,
        future_epoch,
        root_key,
        world_key,
        draw_keys,
        draw_key,
        ti_model,
        ti_truth,
        physical_truth,
        stellar_mass_kg,
        update_target,
        world_data,
        astrometry,
        _chart_quantities_fn,
        finite_chart_quantities,
        chart_quantities,
        _physical_rows_fn,
        truth_quantities,
        alias_edges,
        weighted_quantile,
        summarize_posterior,
        unit_id,
        UnitExecutor,
        _concat,
    )


def source_fingerprint(extra=()):
    """Hash of the implementation that determines what a work unit computes.

    Covers the source text of ``RECORD_MODULES``, of this tool's
    record-defining functions and of ``extra`` (functions, classes, or plain
    values hashed by ``repr``). Freeze and grading code is outside it, so a
    fix there does not orphan finished records; any edit to the updater, the
    targets, the reference samplers, the world draws or the summaries does.
    """
    digest = hashlib.sha256()
    for name in RECORD_MODULES:
        origin = importlib.util.find_spec(name).origin
        digest.update(name.encode() + b"\0" + Path(origin).read_bytes())
    for obj in (*_record_definitions(), *extra):
        digest.update(_source_text(obj).encode() + b"\0")
    return digest.hexdigest()[:16]


def _source_text(obj):
    """Source of a function, of every function of a class, or a value's repr.

    A class is read method by method because ``inspect.getsource`` of a class
    needs its module registered in ``sys.modules``, which a tool loaded from
    its file path is not.
    """
    import inspect

    if isinstance(obj, type):
        parts = [obj.__qualname__]
        for name, member in sorted(vars(obj).items()):
            func = getattr(member, "__func__", member)
            if inspect.isfunction(func):
                parts.append(inspect.getsource(func))
            elif not name.startswith("__"):
                parts.append(f"{name}={member!r}")
        return "\n".join(parts)
    if inspect.isfunction(obj):
        return inspect.getsource(obj)
    return repr(obj)


def run_hash(protocol):
    """Hash of what determines a work unit's result: protocol content and code.

    The protocol part excludes the status, the grading criteria (but keeps
    the alias bins and quantities, which define the summaries), the
    provisional list, the stated values, the limits, the compute prices and
    the reference budgets (each record carries its own budget, checked
    separately against the protocol), so freezing or re-sizing the block does
    not orphan development records, while a change to the regimes, the model
    pairs, the seeds, the kernel, the reference method or the summary bins
    does. The code part is ``source_fingerprint``.
    """
    import copy

    body = copy.deepcopy(protocol)
    alias = body.get("criteria", {}).get("alias_accuracy", {})
    summary_definition = {k: alias.get(k) for k in ("bins", "quantities")}
    for key in ("status", "criteria", "provisional", "stated", "limits", "compute"):
        body.pop(key, None)
    for pair in ("ti_window", "physical_fixed"):
        for key in ("proposals_per_fit", "max_proposals_per_fit", "ess_floor"):
            body.get("reference", {}).get(pair, {}).pop(key, None)
    return declaration_hash(
        {
            "protocol": body,
            "summary_definition": summary_definition,
            "source": source_fingerprint(),
        }
    )


def budgets_match_protocol(protocol, budgets):
    """Whether a run's budgets are the protocol's (overrides make it a test run)."""
    return budgets == protocol_budgets(protocol)


class UnitExecutor:
    """Runs updater and reference work units; caches compiled programs.

    Args:
        protocol: The parameter block.
        budgets: ``protocol_budgets`` output.
        chunk: Reference proposals per compiled batch (per family for the
            Lambert fit).
    """

    def __init__(self, protocol, budgets, chunk=2**16):
        """Store the protocol and budgets; switch JAX to 64-bit mode."""
        enable_x64()
        self.protocol = protocol
        self.budgets = budgets
        self.chunk = chunk
        self.visibility = visibility_mas(protocol)
        self._compiled = set()
        self._physical_rows = None
        self.run_hash = run_hash(protocol)

    def __call__(self, unit):
        """Run one unit and return its record (``seconds`` included)."""
        import time

        world = world_data(
            self.protocol,
            unit["stage"],
            unit["mass_model"],
            unit["r_index"],
            unit["world"],
        )
        start = time.perf_counter()
        if unit["role"] in ALL_UPDATER_ROLES:
            record = self.updater_unit(unit, world)
        else:
            record = self.reference_unit(unit, world)
        record["seconds"] = time.perf_counter() - start
        record["run_hash"] = self.run_hash
        return record

    def _pit_v(self, unit, epoch_count):
        import jax
        import numpy as np

        key = draw_key(
            self.protocol,
            unit["stage"],
            unit["mass_model"],
            unit["r_index"],
            unit["world"],
            "pit",
            epoch_count - 1,
        )
        with jax.enable_x64(True):
            return np.asarray(jax.random.uniform(key, (len(QUANTITIES),)))

    def updater(self, mass_model, particles):
        """The protocol's updater for a mass model at a particle count."""
        from photomancy.orbit.updater import OrbitUpdater

        kernel = self.protocol["updater_kernel"]

        def per_model(value):
            return value[mass_model] if isinstance(value, dict) else value

        return OrbitUpdater(
            n_particles=particles,
            n_rungs=self.budgets["rungs"],
            n_moves=self.budgets["moves_per_rung"],
            record_capacity=len(self.protocol["regimes"][0]["cadence_days"]),
            low_ess_fraction=per_model(kernel["low_ess_fraction"]),
            target_cess_fraction=per_model(kernel["target_cess_fraction"]),
            proposal_scale=per_model(kernel["proposal_scale"]),
            alias_moves=per_model(kernel["alias_moves"]),
        )

    def updater_unit(self, unit, world):
        """Three updates of one world; summaries after epochs 2 and 3."""
        import time

        import jax
        import numpy as np

        from photomancy.orbit.updater import UpdaterRecord, UpdaterStatus

        particles = self.budgets["particles"] * (
            2 if unit["role"] == "updater_doubled" else 1
        )
        updater = self.updater(unit["mass_model"], particles)
        regime = self.protocol["regimes"][unit["r_index"]]
        edges = alias_edges(self.protocol, regime, unit["mass_model"])
        compile_key = (unit["mass_model"], particles)
        first = compile_key not in self._compiled
        self._compiled.add(compile_key)
        updates, summaries, truths = [], {}, {}
        belief = None
        for u in range(len(world["times"])):
            record = UpdaterRecord(
                time=world["times"][u],
                ra=world["ra"][u],
                dec=world["dec"][u],
                ra_err=world["sigma"],
                dec_err=world["sigma"],
                record_id=u + 1,
            )
            key = draw_key(
                self.protocol,
                unit["stage"],
                unit["mass_model"],
                unit["r_index"],
                unit["world"],
                unit["role"],
                u,
            )
            t0 = time.perf_counter()
            if u == 0:
                belief, diag = updater.init(world["target"], record, key)
            else:
                belief, diag = updater.update(belief, record, key)
            jax.block_until_ready(belief)
            seconds = time.perf_counter() - t0
            lw = np.asarray(belief.log_weights)
            fin = np.isfinite(lw)
            w = np.exp(lw[fin] - lw[fin].max()) if fin.any() else np.zeros(0)
            updates.append(
                {
                    "update": u,
                    "status": UpdaterStatus(int(diag.status)).name,
                    "kernel_ess": float(diag.ess),
                    "kish_ess": float(w.sum() ** 2 / np.sum(w**2)) if w.size else 0.0,
                    "n_unique": int(diag.n_unique),
                    "rungs_used": int(diag.rungs_used),
                    "temperature": float(diag.temperature),
                    "acceptance_rate": float(diag.acceptance_rate),
                    "alias_acceptance_rate": float(diag.alias_acceptance_rate),
                    "log_evidence_increment": float(diag.log_evidence_increment),
                    "seconds": seconds,
                    "includes_compile": first,
                }
            )
            k = u + 1
            if k in self.protocol["panel"]["epoch_counts"]:
                t_f = future_epoch(regime, k)
                truth = truth_quantities(self.protocol, unit["mass_model"], world, k)
                q = chart_quantities(
                    unit["mass_model"], world["target"], belief.particles, t_f
                )
                summaries[str(k)] = summarize_posterior(
                    q, lw, edges, self.visibility, truth, self._pit_v(unit, k)
                )
                truths[str(k)] = truth
        return {
            "kind": "updater",
            "particles": particles,
            "updates": updates,
            "summaries": summaries,
            "truth": truths,
        }

    def reference_unit(self, unit, world):
        """One reference fit (any certification variant) for one epoch count."""
        mass_model, k = unit["mass_model"], unit["epoch_count"]
        n_prop = self.budgets[mass_model] * reference_budget_factor(
            mass_model, unit["role"]
        )
        proposal = unit["role"].startswith("cert_proposal")
        regime = self.protocol["regimes"][unit["r_index"]]
        key = draw_key(
            self.protocol,
            unit["stage"],
            mass_model,
            unit["r_index"],
            unit["world"],
            unit["role"],
            k - 1,
        )
        t_f = future_epoch(regime, k)
        if mass_model == "ti_window":
            q, lw, method = self.ti_reference(
                regime, world, k, key, n_prop, proposal, t_f
            )
        else:
            q, lw, method = self.physical_reference(
                world, k, key, n_prop, proposal, t_f
            )
        truth = truth_quantities(self.protocol, mass_model, world, k)
        summary = summarize_posterior(
            q,
            lw,
            alias_edges(self.protocol, regime, mass_model),
            self.visibility,
            truth,
            self._pit_v(unit, k),
        )
        return {
            "kind": "reference",
            "method": method,
            "proposals": n_prop,
            "summary": summary,
            "truth": {str(k): truth},
        }

    def ti_reference(self, regime, world, k, key, n_prop, proposal, t_f):
        """TI reference fit (amendment A7).

        The collapsed prior sampler, or for the independent-proposal check the
        frequency-mixture sampler.
        """
        data = astrometry(world, k)
        target = world["target"]
        if proposal:
            config = self.protocol["reference"]["ti_window"]["independent"]
            theta, lw = ti_frequency_reference(
                target, data, key, n_prop, self.chunk, config
            )
            method = "ti_frequency_mixture_joint_weights"
        else:
            theta, lw = ti_collapsed_reference(target, data, key, n_prop, self.chunk)
            method = "ti_collapsed_prior"
        q, lw = finite_chart_quantities(
            "ti_window", target, theta, lw, t_f, min(self.chunk, n_prop)
        )
        parts = {name: [q[name]] if len(lw) else [] for name in QUANTITIES}
        weights = [lw] if len(lw) else []
        return (*_concat(parts, weights), method)

    def physical_reference(self, world, k, key, n_prop, lowest_snr, t_f):
        """Physical reference: scale-and-rotate proposal with exact weights.

        The weight is the likelihood of every epoch but the anchor times
        ``rho^-2`` (``rho`` the anchor-position separation), with the support
        indicator; the anchor is the highest signal-to-noise epoch, or the
        lowest for the independent-proposal check.
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np

        from photomancy.orbit.ofti import ScaleAndRotate

        data = astrometry(world, k)
        snr = np.hypot(world["ra"][:k], world["dec"][:k]) / world["sigma"]
        anchor = int(np.argmin(snr) if lowest_snr else np.argmax(snr))
        mask = np.ones(k, bool)
        mask[anchor] = False
        scored = astrometry(world, k, mask)
        defaults = self.protocol["model_pairs"]["physical_fixed"]["defaults"]
        regime_bounds_ = (world["target"].period_lo_day, world["target"].period_hi_day)
        log_p = tuple(float(jnp.log10(b)) for b in regime_bounds_)
        e_max = float(world["target"].e_max)
        ms = stellar_mass_kg(self.protocol)
        dist = self.protocol["distance_pc"]
        if self._physical_rows is None:
            self._physical_rows = _physical_rows_fn()
        batch = min(self.chunk, n_prop)

        @eqx.filter_jit
        def score(kb):
            rows, ll = ScaleAndRotate().sample_and_score(
                kb, batch, scored, anchor, ms, dist, defaults["ecc_prior"], log_p, e_max
            )
            return rows, ll

        parts = {name: [] for name in QUANTITIES}
        weights = []
        times = jnp.asarray([data.times[anchor], t_f])
        for j in range(-(-n_prop // batch)):
            rows, ll = score(jax.random.fold_in(key, j))
            take = min(batch, n_prop - j * batch)
            e, ln_p, east, north = self._physical_rows(rows[:take], times, ms, dist)
            rho = jnp.hypot(east[:, 0], north[:, 0])
            lw = np.asarray(ll[:take] - 2.0 * jnp.log(rho))
            fin = np.isfinite(lw)
            for name, v in zip(
                QUANTITIES, (e, ln_p, east[:, 1], north[:, 1]), strict=True
            ):
                parts[name].append(np.asarray(v)[fin])
            weights.append(lw[fin])
        method = (
            f"scale_and_rotate_exact_{'lowest' if lowest_snr else 'highest'}_snr_anchor"
        )
        return (*_concat(parts, weights), method)


# TI references (amendment A7). Both are exact self-normalized importance
# samplers of the TI target's posterior (prior, implied-mass window and every
# epoch's likelihood); they differ in their proposals and in how the weight is
# computed.
TI_FREQUENCY = {"prior_fraction": 0.5, "inflation": 1.25}


def _ti_prior_u(target, key, n):
    """Prior draws of the nonlinear TI coordinates (e, ln P, tau)."""
    return target.sample_prior(key, n)[:, :3]


def _collapsed_draws(target, data, u, z):
    """TI constants from their conditional Gaussian; log Z(u) times the window.

    Returns ``(theta, log_w)`` with ``log_w = log Z(u)`` (up to a constant)
    where the completed point lies in the support, else ``-inf``.
    """
    import jax
    import jax.numpy as jnp

    from photomancy.orbit.updater import _linear_gaussian

    weights = data.is_valid.astype(float)

    def one(ui, zi):
        theta = jnp.concatenate([ui, jnp.zeros(4)])
        mean, chol, log_z, _, _ = _linear_gaussian(target, theta, data, weights)
        c = mean + jax.scipy.linalg.solve_triangular(chol.T, zi, lower=False)
        theta = theta.at[3:].set(c)
        inside = jnp.isfinite(target.log_prior(theta)) & jnp.isfinite(log_z)
        return theta, jnp.where(inside, log_z, -jnp.inf)

    return jax.vmap(one)(u, z)


def ti_collapsed_reference(target, data, key, n_prop, chunk):
    """Primary TI reference: the collapsed prior importance sampler.

    ``(e, ln P, tau)`` come from the prior, the TI constants from their
    conditional Gaussian given every epoch, and the weight is the marginal
    likelihood ``Z(u)`` times the support (implied-mass window) indicator:
    the proposal ``p(u) N(c; m(u), V(u))`` against the target ``p(u) N(c;
    0, s^2) L(u, c) 1_W(u, c)`` leaves exactly that ratio.

    Returns:
        Tuple ``(theta, log_w)`` of NumPy arrays, ``n_prop`` rows.
    """
    import jax
    import numpy as np

    batch = min(chunk, n_prop)
    run = _jitted("collapsed")
    thetas, log_ws = [], []
    for j in range(-(-n_prop // batch)):
        theta, log_w = run(target, data, jax.random.fold_in(key, j), batch)
        take = min(batch, n_prop - j * batch)
        thetas.append(np.asarray(theta)[:take])
        log_ws.append(np.asarray(log_w)[:take])
    return np.concatenate(thetas), np.concatenate(log_ws)


def ti_frequency_reference(target, data, key, n_prop, chunk, config=None):
    """Independent TI reference for certification check 3 (amendment A7).

    Its proposal differs from the primary's in both blocks. ``ln P`` comes
    from a defensive mixture: with probability ``prior_fraction`` the prior
    (uniform in ln P), otherwise uniform in frequency over the same range,
    the data-independent limit of a period-alias comb whose offset is
    unknown; ``e`` and ``tau`` come from the prior. The TI constants come
    from the conditional Gaussian with its covariance inflated by
    ``inflation^2``, and the weight is the full posterior density (the
    target's prior, window included, and likelihood evaluated directly) over
    the joint proposal density, so it never uses the primary's
    marginal-likelihood algebra. Both proposal densities are exact, so the
    sampler is exact.

    Returns:
        Tuple ``(theta, log_w)`` of NumPy arrays, ``n_prop`` rows.
    """
    import jax
    import numpy as np

    cfg = dict(TI_FREQUENCY, **(config or {}))
    cfg.pop("budget_factor", None)
    batch = min(chunk, n_prop)
    run = _jitted("frequency")
    static = (batch, float(cfg["prior_fraction"]), float(cfg["inflation"]))
    thetas, log_ws = [], []
    for j in range(-(-n_prop // batch)):
        theta, log_w = run(target, data, jax.random.fold_in(key, j), *static)
        take = min(batch, n_prop - j * batch)
        thetas.append(np.asarray(theta)[:take])
        log_ws.append(np.asarray(log_w)[:take])
    return np.concatenate(thetas), np.concatenate(log_ws)


def frequency_mixture_ln_period(target, k_mix, k_f, prior_ln_period, prior_fraction):
    """The check-3 mixture's ln P: the prior draw, or uniform in frequency.

    With probability ``prior_fraction`` the given prior draw is kept;
    otherwise ``ln P = -ln f`` with ``f`` uniform on the prior's frequency
    range.
    """
    import jax
    import jax.numpy as jnp

    m = target.model
    n = prior_ln_period.shape[0]
    f = jax.random.uniform(
        k_f, (n,), minval=jnp.exp(-m.lnP_hi), maxval=jnp.exp(-m.lnP_lo)
    )
    use_prior = jax.random.uniform(k_mix, (n,)) < prior_fraction
    return jnp.where(use_prior, prior_ln_period, -jnp.log(f))


def frequency_mixture_log_ratio(target, ln_period, prior_fraction):
    """``log q(ln P) / p(ln P)`` of the check-3 mixture, ``p`` the prior.

    ``p = 1 / (lnP_hi - lnP_lo)``; uniform frequency has density ``f / (f_hi
    - f_lo)`` in ln P, so ``q / p = a + (1 - a) span f / (f_hi - f_lo)``.
    """
    import jax.numpy as jnp

    m = target.model
    f_lo, f_hi = jnp.exp(-m.lnP_hi), jnp.exp(-m.lnP_lo)
    span = m.lnP_hi - m.lnP_lo
    f = jnp.exp(-ln_period)
    return jnp.log(prior_fraction + (1.0 - prior_fraction) * span * f / (f_hi - f_lo))


def inflated_conditional_draw(mean, chol, z, kappa):
    """A draw from ``N(mean, kappa^2 P^-1)`` and its normalized log density.

    ``chol`` is the lower Cholesky factor of the precision ``P`` and ``z`` a
    standard normal vector: ``c = mean + kappa L^-T z``, whose density is
    ``-|z|^2 / 2 - d log(kappa) + log det L - d log(2 pi) / 2``.
    """
    import jax
    import jax.numpy as jnp

    d = mean.shape[0]
    c = mean + kappa * jax.scipy.linalg.solve_triangular(chol.T, z, lower=False)
    log_q = (
        -0.5 * jnp.sum(z**2)
        - d * jnp.log(kappa)
        + jnp.sum(jnp.log(jnp.diag(chol)))
        - 0.5 * d * jnp.log(2.0 * jnp.pi)
    )
    return c, log_q


_JITTED = {}


def _jitted(name):
    """Module-level compiled TI reference batches, built once per process.

    Built lazily so that importing the tool does not import JAX; the batch
    size, grid shape, defensive weight and inflation are static arguments.
    """
    if name in _JITTED:
        return _JITTED[name]
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from photomancy.orbit.updater import _linear_gaussian

    @eqx.filter_jit
    def collapsed(target, data, key, batch):
        k_u, k_z = jax.random.split(key)
        u = _ti_prior_u(target, k_u, batch)
        return _collapsed_draws(target, data, u, jax.random.normal(k_z, (batch, 4)))

    @eqx.filter_jit
    def frequency(target, data, key, batch, prior_fraction, kappa):
        weights = data.is_valid.astype(float)
        k_mix, k_prior, k_f, k_z = jax.random.split(key, 4)
        u = _ti_prior_u(target, k_prior, batch)
        u = u.at[:, 1].set(
            frequency_mixture_ln_period(target, k_mix, k_f, u[:, 1], prior_fraction)
        )
        log_ratio = frequency_mixture_log_ratio(target, u[:, 1], prior_fraction)
        z = jax.random.normal(k_z, (batch, 4))

        def one(ui, zi):
            theta = jnp.concatenate([ui, jnp.zeros(4)])
            mean, chol, _, _, _ = _linear_gaussian(target, theta, data, weights)
            c, log_q_c = inflated_conditional_draw(mean, chol, zi, kappa)
            theta = theta.at[3:].set(c)
            log_post = target.log_prior(theta) + target.log_likelihood(theta, data)
            return theta, log_post - log_q_c

        theta, log_w = jax.vmap(one)(u, z)
        log_w = log_w - log_ratio
        return theta, jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)

    _JITTED.update(collapsed=collapsed, frequency=frequency)
    return _JITTED[name]


def _concat(parts, weights):
    import numpy as np

    if not weights:
        return {name: np.zeros(0) for name in parts}, np.zeros(0)
    return {name: np.concatenate(v) for name, v in parts.items()}, np.concatenate(
        weights
    )


# ---------------------------------------------------------------------------
# Aggregation: freeze summary, effective precision and certification
# ---------------------------------------------------------------------------


def key_to_key_ess(bins_a, bins_b, cap):
    """Effective multinomial size of binned estimates from two independent runs.

    With ``pbar`` the mean of the two runs' bin masses, ``s2 = sum (a - b)^2 /
    2`` estimates the summed per-run variance, and a multinomial of size
    ``n`` has summed variance ``sum pbar (1 - pbar) / n``; so ``n = sum pbar
    (1 - pbar) / s2``, capped at ``cap`` (the particle count) and floored at
    1. This is the precision the panel's D2 sees; the kernel's own ESS is
    not used.
    """
    import numpy as np

    a, b = np.asarray(bins_a, float), np.asarray(bins_b, float)
    pbar = 0.5 * (a + b)
    s2 = 0.5 * np.sum((a - b) ** 2)
    spread = float(np.sum(pbar * (1.0 - pbar)))
    if s2 <= 0.0:
        return float(cap)
    return float(min(max(spread / s2, 1.0), cap))


def replicate_ess(runs, cap):
    """Effective multinomial size from K independent runs, matched in 1 / n^2.

    The panel's D2 variance is quadratic in the per-run covariance, so the
    freeze needs an unbiased ``1 / n^2``, not ``1 / n``. With ``p`` the mean
    of the runs' bin masses, ``M = diag(p) - p p^T`` (so a multinomial of
    size ``n`` has covariance ``M / n``) and ``T`` the trace of the sample
    covariance of the runs on ``nu = K - 1`` degrees of freedom, a Gaussian
    approximation gives ``E[T^2] = ((tr M)^2 + 2 tr(M^2) / nu) / n^2``; so
    ``n = sqrt((tr M)^2 + 2 tr(M^2) / nu) / T`` has ``E[1 / n^2] = 1 / n^2``.
    (``tr M / T``, unbiased in ``1 / n`` only, inflates ``1 / n^2`` by up
    to ``1 + 2 / nu``: threefold for one pair of two-bin runs.) Capped at
    ``cap`` (the particle count) and floored at 1.
    """
    import numpy as np

    x = np.asarray(runs, float)
    nu = x.shape[0] - 1
    p = x.mean(axis=0)
    t = float(np.sum(x.var(axis=0, ddof=1)))
    if t <= 0.0:
        return float(cap)
    m = np.diag(p) - np.outer(p, p)
    scale = math.sqrt(float(np.trace(m)) ** 2 + 2.0 * float(np.sum(m * m)) / nu)
    return float(min(max(scale / t, 1.0), cap))


def _bins(record, k, quantity):
    if record is None:
        return None
    if record["kind"] == "updater":
        summary = record["summaries"].get(str(k))
    else:
        summary = record["summary"]
    if summary is None or "bins" not in summary:
        return None
    return summary["bins"][quantity]


def assemble_freeze_summary(protocol, units, stage="development"):
    """Freeze input from development unit records, and what is missing.

    Each world carries its reference bin masses (the mean of ``reference_1``
    and ``reference_2``), its reference ESS (the smaller Kish ESS of the
    two) and its updater precision per alias quantity from the spread of its
    eight nominal-budget updater runs (``updater_a``, ``updater_b`` and the
    precision replicates; ``replicate_ess``).

    A world whose reference fit kept no weighted draw (an empty fit, where the
    budget is too small for that world) cannot be scored; it is left out and
    its unit id is listed in ``summary["empty_references"]``.

    Returns:
        Tuple ``(summary, missing)``; ``missing`` lists absent unit ids.
    """
    import numpy as np

    alias_q = protocol["criteria"]["alias_accuracy"]["quantities"]
    n_dev = stage_worlds(protocol, stage)
    groups, missing, empty = [], [], []
    for regime_id, mass_model, k in canonical_groups(protocol):
        worlds = []
        for w in range(n_dev):
            ids = {
                role: unit_id(
                    stage,
                    mass_model,
                    regime_id,
                    w,
                    role,
                    k if role.startswith("ref") else None,
                )
                for role in (
                    "updater_a",
                    "updater_b",
                    *PRECISION_ROLES,
                    "reference_1",
                    "reference_2",
                )
            }
            recs = {role: units.get(i) for role, i in ids.items()}
            absent = [ids[r] for r, rec in recs.items() if rec is None]
            if absent:
                missing += absent
                continue
            hollow = [
                ids[r]
                for r in ("reference_1", "reference_2")
                if _bins(recs[r], k, alias_q[0]) is None
            ]
            if hollow:
                empty += hollow
                continue
            cap = recs["updater_a"]["particles"]
            worlds.append(
                {
                    "world": w,
                    "updater_ess": {
                        q: replicate_ess(
                            [
                                _bins(recs[r], k, q)
                                for r in ("updater_a", "updater_b", *PRECISION_ROLES)
                            ],
                            cap,
                        )
                        for q in alias_q
                    },
                    "kernel_ess": recs["updater_a"]["updates"][k - 1]["kernel_ess"],
                    "reference_ess": min(
                        recs["reference_1"]["summary"]["kish_ess"],
                        recs["reference_2"]["summary"]["kish_ess"],
                    ),
                    "bin_masses": {
                        q: (
                            0.5
                            * (
                                np.asarray(_bins(recs["reference_1"], k, q))
                                + np.asarray(_bins(recs["reference_2"], k, q))
                            )
                        ).tolist()
                        for q in alias_q
                    },
                }
            )
        groups.append(
            {
                "regime": regime_id,
                "mass_model": mass_model,
                "epoch_count": k,
                "worlds": worlds,
            }
        )
    return {"groups": groups, "empty_references": sorted(empty)}, sorted(set(missing))


CERTIFICATION_CHECKS = {
    "repeat_noise": (("reference_1", "reference_3"), ("reference_2", "reference_4")),
    "budget_agreement": (
        ("reference_1", "cert_doubled_1"),
        ("reference_2", "cert_doubled_2"),
    ),
    "proposal_agreement": (
        ("reference_1", "cert_proposal_1"),
        ("reference_2", "cert_proposal_2"),
    ),
}


def certification_fractions(protocol):
    """Fraction of delta2 that bounds each certification check."""
    ref = protocol["reference"]
    return {
        "repeat_noise": ref["repeat_noise"]["bound_fraction_of_delta2"],
        "budget_agreement": ref["convergence"]["bound_fraction_of_delta2"],
        "proposal_agreement": ref["cross_method"]["bound_fraction_of_delta2"],
    }


def certify_references(protocol, units, delta2, stage="development"):
    """Certification checks 1 to 3 per mass model, from development records.

    For each mass model, epoch count and alias quantity the statistic pools
    every development world of every regime: ``D2_w = sum_b (X_b - Y_b)(X'_b -
    Y'_b)`` with the run pairs of ``CERTIFICATION_CHECKS``, and the check
    passes when ``mean + z SE`` (``z`` the alias critical value) is at most
    its fraction of ``delta2``. A mass model with any missing record, or any
    empty reference fit (no weighted draw), is ``incomplete``.
    """
    import numpy as np

    z = critical_values(protocol)["alias_z"]
    fractions = certification_fractions(protocol)
    alias_q = protocol["criteria"]["alias_accuracy"]["quantities"]
    n_dev = stage_worlds(protocol, stage)
    out = {}
    for mass_model in protocol["panel"]["mass_models"]:
        rows, missing, empty = [], [], set()
        for k in protocol["panel"]["epoch_counts"]:
            for check, (left, right) in CERTIFICATION_CHECKS.items():
                for q in alias_q:
                    d2 = []
                    for regime in protocol["regimes"]:
                        for w in range(n_dev):
                            roles = (*left, *right)
                            ids = [
                                unit_id(stage, mass_model, regime["id"], w, r, k)
                                for r in roles
                            ]
                            recs = [units.get(i) for i in ids]
                            if any(r is None for r in recs):
                                missing += [
                                    i
                                    for i, r in zip(ids, recs, strict=True)
                                    if r is None
                                ]
                                continue
                            bins = [_bins(r, k, q) for r in recs]
                            if any(b is None for b in bins):
                                empty.update(
                                    i
                                    for i, b in zip(ids, bins, strict=True)
                                    if b is None
                                )
                                continue
                            x, y, x2, y2 = (np.asarray(b) for b in bins)
                            d2.append(float(np.sum((x - y) * (x2 - y2))))
                    if not d2:
                        continue
                    d2 = np.asarray(d2)
                    se = (
                        d2.std(ddof=1) / math.sqrt(len(d2)) if len(d2) > 1 else math.inf
                    )
                    upper = float(d2.mean() + z * se)
                    bound = fractions[check] * delta2
                    rows.append(
                        {
                            "check": check,
                            "epoch_count": k,
                            "quantity": q,
                            "worlds": len(d2),
                            "d2_mean": float(d2.mean()),
                            "d2_upper": upper,
                            "bound": bound,
                            "pass": upper <= bound,
                        }
                    )
        missing = sorted(set(missing))
        if missing or empty:
            status = "incomplete"
        elif all(r["pass"] for r in rows):
            status = "certified"
        else:
            status = "not certified"
        out[mass_model] = {
            "status": status,
            "missing_units": len(missing),
            "empty_units": len(empty),
            "checks": rows,
        }
    return out


def freeze_report(protocol, path, exact_path=None):
    """The freeze decision from a summary file or a development run directory.

    A directory is checked against the complete canonical development plan:
    only records run under the current code and protocol content (run hash)
    at the protocol budgets count, every planned unit must have one, and
    duplicates and unknown ids are reasons. The compatible records are then
    assembled into a summary. The freeze is ``frozen`` only when the plan is
    complete, the alias tolerance meets its bound and power floor, both
    references pass certification checks 1 to 3 against it, and the exact
    checks, re-assessed from their run directory (``exact_path``), pass for
    both mass models. Otherwise the report is ``incomplete`` and lists every
    reason.
    """
    path = Path(path)
    reasons = []
    report = {}
    incomplete = False
    units = None
    over_cap = budget_cap_violations(protocol)
    if over_cap:
        incomplete = True
        reasons.append(
            "protocol budgets exceed the proposal cap after role multiples "
            f"(the method needs an amendment, not another doubling): {over_cap}"
        )
    if path.is_dir():
        records, duplicates = load_units(path)
        if duplicates:
            reasons.append(f"{len(duplicates)} duplicate unit records (first kept)")
        current = run_hash(protocol)
        stale = sorted(i for i, r in records.items() if r.get("run_hash") != current)
        if stale:
            hashes = sorted({str(records[i].get("run_hash")) for i in stale})
            reasons.append(
                f"{len(stale)} records were run under other code or protocol "
                f"content (run hashes {hashes}, current {current}); rerun them "
                "into a fresh directory"
            )
        off_budget = sorted(
            {
                (r.get("kind"), r.get("particles", r.get("proposals")))
                for r in records.values()
                if not _record_at_protocol_budget(protocol, r)
            }
        )
        if off_budget:
            reasons.append(f"records at non-protocol budgets: {off_budget}")
        units = {
            i: r
            for i, r in records.items()
            if r.get("run_hash") == current and _record_at_protocol_budget(protocol, r)
        }
        planned = [u["id"] for u in plan_units(protocol, "development", list(PARTS))]
        unknown = sorted(set(records) - set(planned))
        if unknown:
            reasons.append(
                f"{len(unknown)} records are not development units: {unknown[:3]}"
            )
        absent = [i for i in planned if i not in units]
        if absent:
            incomplete = True
            reasons.append(
                f"{len(absent)} of {len(planned)} planned development units "
                f"missing or incompatible; first: {absent[:3]}"
            )
        reasons.extend(ess_floor_reasons(protocol, units))
        summary, _ = assemble_freeze_summary(protocol, units)
        empty = summary["empty_references"]
        if empty:
            incomplete = True
            reasons.append(
                f"{len(empty)} reference fits kept no weighted draw (the proposal "
                "budget is too small for those worlds; double it); first empty: "
                f"{empty[:3]}"
            )
        if incomplete:
            report["worlds_complete_per_group"] = {
                "{regime} {mass_model} k={epoch_count}".format(**g): len(g["worlds"])
                for g in summary["groups"]
            }
            # For information only: the certification statistics of whatever
            # compatible records exist, bounded by the provisional delta2.
            provisional = protocol["criteria"]["alias_accuracy"]["delta2"]
            report["certification_partial_vs_provisional_delta2"] = certify_references(
                protocol, units, provisional
            )
    else:
        summary = json.loads(path.read_text())
    if not incomplete:
        try:
            alias = freeze_alias(protocol, summary)
        except ValueError as err:
            reasons.append(f"summary refused: {err}")
            alias = None
        if alias is not None:
            report["alias"] = alias
            reasons += alias["reasons"]
            delta2 = min(
                alias["delta2"], protocol["criteria"]["alias_accuracy"]["delta2_max"]
            )
            if units is not None:
                cert = certify_references(protocol, units, delta2)
                report["certification"] = cert
                for mass_model, c in cert.items():
                    if c["status"] != "certified":
                        reasons.append(f"{mass_model} reference {c['status']}")
            else:
                reasons.append(
                    "certification needs the run directory, not a summary file"
                )
    reasons += _exact_reasons(protocol, exact_path, report)
    report["status"] = "incomplete" if reasons else "frozen"
    report["reasons"] = reasons
    return report


REFERENCE_GROUPS = {
    "primary": ("reference_1", "reference_2", "reference_3", "reference_4"),
    "doubled": ("cert_doubled_1", "cert_doubled_2"),
    "independent": ("cert_proposal_1", "cert_proposal_2"),
}


def _reference_group(role):
    return next(g for g, roles in REFERENCE_GROUPS.items() if role in roles)


def ess_floor_violations(protocol, units):
    """Reference fits below their pair's ``ess_floor``: ``(id, proposals, ess)``.

    Every reference role counts (primary, doubled and independent-proposal
    runs); a pair without a floor in the parameter block is not checked.
    """
    out = []
    for unit_id_, rec in units.items():
        if rec.get("kind") != "reference":
            continue
        pair = unit_id_.split("/")[1]
        floor = protocol["reference"].get(pair, {}).get("ess_floor")
        ess = rec.get("summary", {}).get("kish_ess", 0.0)
        if floor is not None and ess < floor:
            out.append((unit_id_, rec.get("proposals"), round(float(ess), 1)))
    return sorted(out)


def ess_table(protocol, units):
    """Worst and median Kish ESS per pair, regime, epoch count and reference group.

    Returns a dict keyed ``"pair regime k=K group"`` with ``fits``, ``worst``,
    ``median``, the ``floor`` and the ids below it.
    """
    import statistics

    rows = {}
    for unit_id_, rec in units.items():
        if rec.get("kind") != "reference":
            continue
        _, pair, regime, _, role, k = unit_id_.split("/")
        key = f"{pair} {regime} {k.replace('k', 'k=')} {_reference_group(role)}"
        rows.setdefault(key, []).append((rec["summary"]["kish_ess"], unit_id_))
    out = {}
    for key in sorted(rows):
        ess = [e for e, _ in rows[key]]
        floor = protocol["reference"].get(key.split()[0], {}).get("ess_floor")
        out[key] = {
            "fits": len(ess),
            "worst": round(min(ess), 1),
            "median": round(statistics.median(ess), 1),
            "floor": floor,
            "below_floor": sorted(i for e, i in rows[key] if floor and e < floor),
        }
    return out


def _exact_reasons(protocol, exact_path, report):
    """Re-assess the exact checks from their run directory; return reasons.

    The verdicts are recomputed from the unit records (provenance, budgets,
    uniqueness and the required levels and repeats are validated there), never
    read from a results file, which could be stale or edited.
    """
    if exact_path is None:
        return ["exact-check results not supplied (--exact EXACT_RUN_DIR)"]
    exact_path = Path(exact_path)
    if not exact_path.is_dir():
        return [
            f"--exact {exact_path} is not a run directory; the freeze re-assesses "
            "the exact checks from their unit records"
        ]
    spec = importlib.util.spec_from_file_location(
        "updater_exact_checks",
        Path(__file__).resolve().parent / "updater_exact_checks.py",
    )
    exact_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact_tool)
    exact = exact_tool.report(protocol, exact_path)
    report["exact_checks"] = {
        "verdicts": exact["verdicts"],
        "reasons": {c: e["reasons"] for c, e in exact["cases"].items()},
    }
    # Every case is required; the binding-window case feeds the TI verdict.
    return [
        f"{exact_tool.CASE_MASS_MODEL[case]} exact checks ({case}): "
        f"{exact['verdicts'].get(case, 'missing')}"
        for case in exact_tool.CASES
        if exact["verdicts"].get(case) != "pass"
    ]


def _record_at_protocol_budget(protocol, record):
    return record_at_budgets(protocol_budgets(protocol), record) and record_within_cap(
        protocol, record
    )


def record_at_budgets(budgets, record):
    """Whether a unit record was run at ``budgets`` (doubled roles at twice)."""
    if record.get("kind") == "updater":
        doubled = record["id"].split("/")[4] == "updater_doubled"
        return record["particles"] == budgets["particles"] * (2 if doubled else 1)
    if record.get("kind") != "reference":
        return False
    mass_model, role = record["id"].split("/")[1], record["id"].split("/")[4]
    factor = reference_budget_factor(mass_model, role)
    return record.get("proposals") == budgets[mass_model] * factor


def reference_budget_factor(mass_model, role):
    """Budget multiple of a reference role.

    2 for the doubled-budget runs and, for the TI pair, for the
    independent-proposal runs (amendment A7); 1 otherwise.
    """
    if role.startswith("cert_doubled"):
        return 2
    if role.startswith("cert_proposal") and mass_model == "ti_window":
        return 2
    return 1


# A role of each budget multiple, for the cap check.
_BUDGET_ROLES = ("reference_1", "cert_doubled_1", "cert_proposal_1")


def budget_cap_violations(protocol, budgets=None):
    """Effective reference budgets above their pair's ``max_proposals_per_fit``.

    The effective budget is the run budget times the role's multiple
    (``reference_budget_factor``), so a primary budget at the cap leaves the
    doubled and, for the TI pair, the check-3 runs above it.
    """
    budgets = budgets or protocol_budgets(protocol)
    out = []
    for pair in ("ti_window", "physical_fixed"):
        cap = protocol["reference"][pair]["max_proposals_per_fit"]
        for role in _BUDGET_ROLES:
            proposals = budgets[pair] * reference_budget_factor(pair, role)
            if proposals > cap:
                out.append(f"{pair} {role[:-2]}: {proposals} > cap {cap}")
    return out


def record_within_cap(protocol, record):
    """Whether a reference record's proposals are within its pair's cap."""
    if record.get("kind") != "reference":
        return True
    pair = record["id"].split("/")[1]
    cap = protocol["reference"][pair]["max_proposals_per_fit"]
    return record.get("proposals", 0) <= cap


def ess_floor_reasons(protocol, units):
    """Freeze reasons for the reference fits below their ESS floor.

    A fit whose doubled budget stays within the cap is remedied by doubling
    that method's budget; a fit already at a budget whose doubling would
    exceed the cap cannot be, and its method needs an amendment.
    """
    low = ess_floor_violations(protocol, units)
    caps = {
        p: protocol["reference"][p]["max_proposals_per_fit"]
        for p in ("ti_window", "physical_fixed")
    }
    double, amend = [], []
    for row in low:
        pair = row[0].split("/")[1]
        (double if 2 * row[1] <= caps[pair] else amend).append(row)
    reasons = []
    if double:
        reasons.append(
            f"{len(double)} reference fits below their ESS floor (double that "
            "method's budget and rerun them; id, proposals, ESS): "
            f"{double[:5]}"
        )
    if amend:
        reasons.append(
            f"{len(amend)} reference fits below their ESS floor whose budget "
            "cannot be doubled within the proposal cap (the method needs an "
            f"amendment; id, proposals, ESS): {amend[:5]}"
        )
    return reasons


def progress(protocol, out_dir, stage="development"):
    """Done and planned unit counts per part, for a run directory."""
    units, duplicates = load_units(out_dir)
    lines = []
    for part in PARTS:
        planned = plan_units(protocol, stage, [part])
        done = sum(u["id"] in units for u in planned)
        lines.append(f"{part}: {done} of {len(planned)} units")
    lines.append(f"duplicates: {len(duplicates)}")
    return lines


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the allocation")
    parser.add_argument("--sizing", action="store_true", help="print criterion power")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--prices",
        type=Path,
        default=None,
        help="latency probe output that prices the tally (default: the committed one)",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=None,
        help="development run directory (or summary JSON); prints the freeze decision",
    )
    parser.add_argument(
        "--exact", type=Path, default=None, help="exact-check results JSON"
    )
    parser.add_argument(
        "--run",
        choices=["development", "smoke", "panel"],
        default=None,
        help="run the work units of a stage into --out",
    )
    parser.add_argument("--out", type=Path, default=None, help="run directory")
    parser.add_argument(
        "--parts",
        default="updater,precision,reference,certification",
        help="comma list of updater, precision, reference, certification",
    )
    parser.add_argument(
        "--shard", default="0/1", help="i/n: run every n-th unit from i"
    )
    parser.add_argument("--worlds", default=None, help="world subset, e.g. 0-5 or 0,3")
    parser.add_argument("--regimes", default=None, help="regime subset, e.g. R1,R2")
    parser.add_argument("--mass-models", default=None, help="mass-model subset")
    parser.add_argument("--particles", type=int, default=None, help="test override")
    parser.add_argument("--ti-proposals", type=int, default=None, help="test override")
    parser.add_argument(
        "--physical-proposals", type=int, default=None, help="test override"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=2**16,
        help="reference proposals per compiled batch",
    )
    parser.add_argument("--status", type=Path, default=None, help="print run progress")
    parser.add_argument(
        "--roles", default=None, help="comma list: run only these run roles"
    )
    parser.add_argument(
        "--ess",
        type=Path,
        default=None,
        help="run directory: print the reference ESS table against the floors",
    )
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.freeze is not None:
        print(json.dumps(freeze_report(protocol, args.freeze, args.exact), indent=2))
    if args.dry_run:
        path = args.prices or (PRICES if PRICES.exists() else None)
        prices = json.loads(path.read_text()) if path else None
        dry_run(protocol, prices)
    if args.sizing:
        sizing(protocol)
    if args.status is not None:
        print("\n".join(progress(protocol, args.status)))
    if args.ess is not None:
        print(json.dumps(ess_table(protocol, load_units(args.ess)[0]), indent=1))
    if args.run is not None:
        if args.out is None:
            parser.error("--run needs --out")
        run_stage(protocol, args)
    if not (
        args.dry_run
        or args.sizing
        or args.freeze
        or args.run
        or args.status
        or args.ess
    ):
        parser.error("nothing to do")
    return 0


def run_stage(protocol, args):
    """Plan, shard and run the units of ``args.run`` into ``args.out``."""
    budgets = protocol_budgets(
        protocol, args.particles, args.ti_proposals, args.physical_proposals
    )
    over_cap = budget_cap_violations(protocol, budgets)
    if over_cap:
        raise SystemExit(
            "refusing to run: reference budgets exceed the proposal cap after "
            f"role multiples (amend the method instead): {over_cap}"
        )
    shard = parse_shard(args.shard)
    units = plan_units(
        protocol,
        args.run,
        args.parts.split(","),
        mass_models=args.mass_models.split(",") if args.mass_models else None,
        regimes=args.regimes.split(",") if args.regimes else None,
        worlds=parse_worlds(args.worlds) if args.worlds else None,
    )
    if args.roles:
        roles = set(args.roles.split(","))
        units = [u for u in units if u["role"] in roles]
    entry = manifest_entry(protocol, None, budgets)
    entry["protocol_budgets"] = budgets_match_protocol(protocol, budgets)
    entry["shard"] = args.shard
    entry["finished"] = None
    executor = UnitExecutor(protocol, budgets, chunk=args.chunk)

    def compatible(record):
        return record.get("run_hash") == executor.run_hash and record_at_budgets(
            budgets, record
        )

    try:
        counts = run_units(
            units,
            args.out,
            shard,
            _ManifestFirst(executor, args.out, shard, entry),
            log=lambda m: print(m, flush=True),
            compatible=compatible,
        )
    except IncompatibleRunError as err:
        raise SystemExit(f"refusing to resume: {err}") from None
    import datetime

    entry.update(counts, finished=datetime.datetime.now(datetime.UTC).isoformat())
    write_manifest(args.out, shard, entry)
    print(json.dumps(counts))


class _ManifestFirst:
    """Executor wrapper that writes the manifest entry once, before any unit.

    The entry goes in after the compatibility check (a refused directory is
    left untouched) and before the first unit, so a killed run still leaves
    its provenance.
    """

    def __init__(self, execute, out_dir, shard, entry):
        """Wrap ``execute``; write ``entry`` on the first call."""
        self.execute, self.out_dir, self.shard, self.entry = (
            execute,
            out_dir,
            shard,
            entry,
        )
        self.written = False

    def __call__(self, unit):
        """Write the manifest entry if not yet written, then run ``unit``."""
        if not self.written:
            write_manifest(self.out_dir, self.shard, self.entry)
            self.written = True
        return self.execute(unit)


if __name__ == "__main__":
    sys.exit(main())
