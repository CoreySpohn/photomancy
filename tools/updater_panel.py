#!/usr/bin/env python3
"""Acceptance panel driver for the fixed-budget array orbit updater.

The protocol lives in ``docs/updater-protocol.md``; its JSON parameter block is
the only source of the numbers used here. Critical values are recomputed from
their formulas, never copied.

Usage::

    python tools/updater_panel.py --dry-run [--prices docs/updater-latency-probe.json]
    python tools/updater_panel.py --sizing
    python tools/updater_panel.py --freeze SUMMARY.json

``--dry-run`` prints the cells, the world and seed allocation, the model hashes,
the reference arguments, the critical values and the compute tally. ``--sizing``
prints the power of each criterion against its stated alternatives.
``--freeze`` computes the alias tolerance from development summaries and checks
it against its bound and power floor. Running the panel itself is not
implemented yet.
"""

import argparse
import hashlib
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


def schedule_null_probability(margin_k):
    """Null exceedance probability 2 (1 - Phi(k)) under Gaussian errors."""
    return 2.0 * (1.0 - statistics.NormalDist().cdf(margin_k))


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
        "schedule_comparisons": len(
            protocol["criteria"]["schedule_stability"]["comparisons"]
        ),
    }


def critical_values(protocol):
    """Recompute every derived threshold of the protocol from its formula.

    Returns a dict with the exact values and their rounded forms, keyed like the
    protocol's ``stated`` block for the rounded ones.
    """
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

    sched = crit["schedule_stability"]
    q0 = schedule_null_probability(sched["margin_k"])
    n_sched = groups * g["schedule_comparisons"]
    sched_c = upper_tail_critical_count(n, q0, sched["family_alpha"] / n_sched)
    sched_cap = max(
        c
        for c in range(n + 1)
        if binom_sf(c, n, sched["alternative_exceedance"]) >= sched["min_power"]
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
        "schedule_tests": n_sched,
        "schedule_null_probability": q0,
        "schedule_critical_count": sched_c,
        "schedule_max_critical_count": sched_cap,
    }


# ---------------------------------------------------------------------------
# Schedule null (per world, binomial at the claimed ESS)
# ---------------------------------------------------------------------------


def schedule_null_probabilities(p_a, p_b, ess_a, ess_b, margin_k, sims, rng):
    """Per-world null exceedance probabilities under the claimed ESS.

    Each world's two runs are replaced by binomial estimates of their mean
    probability at their own ESS (rounded, at least 1); the null probability is
    the exceedance frequency over ``sims`` simulated pairs.
    """
    import numpy as np

    p_bar = 0.5 * (np.asarray(p_a, float) + np.asarray(p_b, float))
    n_a = np.maximum(np.rint(np.asarray(ess_a, float)), 1).astype(np.int64)
    n_b = np.maximum(np.rint(np.asarray(ess_b, float)), 1).astype(np.int64)
    ha = rng.binomial(n_a, p_bar, size=(sims, p_bar.size)) / n_a
    hb = rng.binomial(n_b, p_bar, size=(sims, p_bar.size)) / n_b
    se_d = np.sqrt(ha * (1 - ha) / n_a + hb * (1 - hb) / n_b)
    return (np.abs(ha - hb) > margin_k * se_d).mean(axis=0)


def poisson_binomial_pmf(probs):
    """Distribution of the count of independent Bernoulli(probs) successes."""
    dist = [1.0]
    for q in probs:
        nxt = [0.0] * (len(dist) + 1)
        for j, mass in enumerate(dist):
            nxt[j] += mass * (1.0 - q)
            nxt[j + 1] += mass * q
        dist = nxt
    return dist


def schedule_cutoff(probs, alpha):
    """Smallest c with P(count > c) <= alpha under the per-world null."""
    dist = poisson_binomial_pmf(probs)
    return min(c for c in range(len(dist)) if math.fsum(dist[c + 1 :]) <= alpha)


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


def ti_reference_arguments(protocol, regime, epoch_count):
    """Arguments of the TI reference fit for one regime and epoch count."""
    from hwoutils.constants import G, Msun2kg

    p_lo, p_hi, e_max = regime_bounds(protocol, regime, "ti_window")
    window = protocol["model_pairs"]["ti_window"]["defaults"]["mass_window_msun"]
    mu_hi = window[1] * float(G) * float(Msun2kg)
    z_max = (1.0 + e_max) * semimajor_axis_au(p_hi, mu_hi)
    times = regime["cadence_days"][:epoch_count]
    n_max_rev = math.ceil((times[-1] - times[0]) / p_lo)
    families = 2 + 4 * n_max_rev
    n_prop = protocol["reference"]["ti_window"]["proposals_per_fit"]
    return {
        "z_max_au": z_max,
        "n_max_rev": n_max_rev,
        "families": families,
        "n_per_family": n_prop // families,
    }


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


def draw_key(protocol, stage, mass_model, regime_index, world_index, purpose, update):
    """Key of one draw: world key, then purpose, then update index."""
    import jax

    with jax.enable_x64(True):
        key = world_key(protocol, stage, mass_model, regime_index, world_index)
        key = jax.random.fold_in(key, protocol["prng"]["purposes"][purpose])
        return jax.random.fold_in(key, update)


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
    blocks = len(protocol["regimes"]) * len(panel["mass_models"])
    return panel["smoke_worlds"] // blocks


# ---------------------------------------------------------------------------
# Compute tally
# ---------------------------------------------------------------------------


def compute_tally(protocol):
    """Update-equivalents and reference fits per model pair."""
    g = group_counts(protocol)
    panel = protocol["panel"]
    # Two independent keys plus one doubled-budget run that costs two.
    run_units = 4
    upd_panel = g["regimes"] * panel["worlds_per_cell"] * g["updates"] * run_units
    dev_worlds = panel["development_worlds_per_regime_and_mass_model"]
    upd_dev = g["regimes"] * dev_worlds * g["updates"] * run_units
    ref_runs = 2
    fits = g["regimes"] * panel["alias_subset_worlds"] * g["epoch_counts"] * ref_runs
    # Certification: repeat-noise runs, two runs at twice the budget (cost four)
    # and two independent-proposal runs.
    cert_units = protocol["reference"]["repeat_noise"]["runs"] + 4 + 2
    dev_fits = g["regimes"] * dev_worlds * g["epoch_counts"] * cert_units
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
                tally["development_reference_fit_equivalents_per_pair"] * per_fit
            )
        out[pair] = row
    return out, []


def print_tally(protocol, prices):
    """Print the compute tally, priced per pair when probe data match."""
    tally = compute_tally(protocol)
    print("Compute tally")
    for k, v in tally.items():
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

    print("\nTI reference arguments (lambert_depth_fit)")
    for regime in protocol["regimes"]:
        for k in protocol["panel"]["epoch_counts"]:
            args = ti_reference_arguments(protocol, regime, k)
            print(
                f"  {regime['id']} epochs={k}: z_max_au={args['z_max_au']:.3f} "
                f"n_max_rev={args['n_max_rev']} families={args['families']} "
                f"n_per_family={args['n_per_family']}"
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


def freeze_alias(protocol, summary, panels=None):
    """Alias tolerance from development summaries, with its bound and power floor.

    ``summary["groups"]`` holds one entry per group with ``regime``,
    ``mass_model``, ``epoch_count``, ``updater_ess``, ``reference_ess`` and
    ``reference_bin_masses`` (a dict from alias quantity to a list of per-world
    bin-mass vectors). Returns a dict with the candidate ``delta2``, the
    per-group standard errors and powers, and ``status`` (``frozen`` or
    ``incomplete``) with the reasons.
    """
    import numpy as np

    alias = protocol["criteria"]["alias_accuracy"]
    alt = alias["alternative"]
    n_worlds = protocol["panel"]["alias_subset_worlds"]
    panels = panels or alias["freeze_panels"]
    z = critical_values(protocol)["alias_z"]
    stream = protocol["prng"]["simulation_seeds"]["alias_freeze"]
    root = protocol["prng"]["root_seed"]

    def panel_stats(rng, masses, ess_u, ess_r, damaged):
        p = masses[rng.integers(0, len(masses), n_worlds)]
        q = p.copy()
        hit = np.flatnonzero(rng.random(n_worlds) < damaged)
        order = np.argsort(q[hit], axis=1)[:, ::-1]
        rows = hit[:, None]
        take = np.minimum(alt["moved_mass"], q[rows, order[:, 1:2]])
        q[rows, order[:, 1:2]] -= take
        q[rows, order[:, 0:1]] += take
        u_a, u_b = (rng.multinomial(ess_u, q) / ess_u for _ in range(2))
        r_a, r_b = (rng.multinomial(ess_r, p) / ess_r for _ in range(2))
        d2 = ((u_a - r_a) * (u_b - r_b)).sum(1)
        return d2.mean(), d2.std(ddof=1) / np.sqrt(n_worlds)

    rows = []
    for g_index, group in enumerate(summary["groups"]):
        ess_u = max(round(group["updater_ess"]), 1)
        ess_r = max(round(group["reference_ess"]), 1)
        for q_index, quantity in enumerate(alias["quantities"]):
            masses = np.asarray(group["reference_bin_masses"][quantity], float)
            masses = masses / masses.sum(axis=1, keepdims=True)
            rng = np.random.default_rng([root, stream, g_index, q_index])
            null = [panel_stats(rng, masses, ess_u, ess_r, 0.0) for _ in range(panels)]
            damaged = [
                panel_stats(rng, masses, ess_u, ess_r, alt["damaged_fraction"])
                for _ in range(panels)
            ]
            rows.append(
                {
                    "regime": group["regime"],
                    "mass_model": group["mass_model"],
                    "epoch_count": group["epoch_count"],
                    "quantity": quantity,
                    "se0": float(np.mean([se for _, se in null])),
                    "damaged": damaged,
                }
            )
    delta2 = max(alias["delta2"], 2.0 * z * max(r["se0"] for r in rows))
    reasons = []
    if delta2 > alias["delta2_max"]:
        reasons.append(f"delta2 {delta2:.3g} exceeds its bound {alias['delta2_max']}")
    graded = min(delta2, alias["delta2_max"])
    for r in rows:
        dmg = r.pop("damaged")
        r["power"] = float(np.mean([m + z * se > graded for m, se in dmg]))
        if r["power"] < alias["min_power"]:
            reasons.append(
                f"{r['regime']} {r['mass_model']} {r['epoch_count']} {r['quantity']}: "
                f"power {r['power']:.3f} below {alias['min_power']}"
            )
    return {
        "delta2": delta2,
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
    print("Schedule: P(test fails) at true exceedance probability")
    for label, sc in (
        ("large-ESS cutoff", values["schedule_critical_count"]),
        ("power-floor cap", values["schedule_max_critical_count"]),
    ):
        for q in (0.08, 0.10, 0.15):
            print(f"  {label} {sc}, q={q}: {binom_sf(sc, n, q):.3f}")
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
        help="development summary JSON; prints the alias tolerance freeze",
    )
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.freeze is not None:
        summary = json.loads(args.freeze.read_text())
        print(json.dumps(freeze_alias(protocol, summary), indent=2))
    if args.dry_run:
        path = args.prices or (PRICES if PRICES.exists() else None)
        prices = json.loads(path.read_text()) if path else None
        dry_run(protocol, prices)
    if args.sizing:
        sizing(protocol)
    if not (args.dry_run or args.sizing or args.freeze):
        raise NotImplementedError("running the panel is not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
