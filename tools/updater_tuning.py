#!/usr/bin/env python3
"""Kernel tuning of the orbit updater on the tuning worlds (0 to 5).

The protocol allows the proposal scales, the tempering ladder and the moves to
be tuned on the development panel, and only on its tuning worlds. This script
runs the two independent-key updater roles (``updater_a``, ``updater_b``) at
the protocol budget for each kernel configuration in ``CONFIGS`` and, for the
worlds that have them, compares the binned posteriors with two independent
reference runs from ``tools/updater_panel.py``.

Usage::

    python tools/updater_tuning.py --run CONFIG --out DIR [--mass-models ...]
    python tools/updater_tuning.py --report DIR [--references REF_DIR]

``--run`` writes one unit file per configuration under ``DIR/CONFIG`` with the
panel runner's resumable work units. ``--report`` prints, per configuration,
mass model, regime and epoch count, the key-to-key precision of each alias
quantity (``key_to_key_ess``), the per-world total variation between the two
keys, the debiased squared distance to the references when they exist, the
rungs used, the statuses and the latency per update.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "updater_panel", REPO / "tools" / "updater_panel.py"
)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

TUNING_WORLDS = range(6)
CONFIGS = {
    "cess50_s030_noalias": {"target_cess_fraction": 0.5, "scale": 0.3, "alias": False},
    "cess90_s100_noalias": {"target_cess_fraction": 0.9, "scale": 1.0, "alias": False},
    "cess90_s100_alias": {"target_cess_fraction": 0.9, "scale": 1.0, "alias": True},
    "cess90_s238_alias": {"target_cess_fraction": 0.9, "scale": 2.38, "alias": True},
    "cess70_s100_alias": {"target_cess_fraction": 0.7, "scale": 1.0, "alias": True},
    "cess80_s100_alias": {"target_cess_fraction": 0.8, "scale": 1.0, "alias": True},
    "cess85_s100_alias": {"target_cess_fraction": 0.85, "scale": 1.0, "alias": True},
    "cess50_s030": {"target_cess_fraction": 0.5, "scale": 0.3, "alias": True},
    "cess70_s030": {"target_cess_fraction": 0.7, "scale": 0.3, "alias": True},
    "cess90_s030": {"target_cess_fraction": 0.9, "scale": 0.3, "alias": True},
    "cess90_s060": {"target_cess_fraction": 0.9, "scale": 0.6, "alias": True},
    "cess85_s030": {"target_cess_fraction": 0.85, "scale": 0.3, "alias": True},
}


def configured_protocol(protocol, config, mass_models):
    """The protocol with its updater kernel block replaced by ``config``."""
    kernel = dict(protocol["updater_kernel"])
    kernel["target_cess_fraction"] = {
        m: config["target_cess_fraction"] for m in protocol["panel"]["mass_models"]
    }
    kernel["proposal_scale"] = {m: config["scale"] for m in mass_models}
    kernel["alias_moves"] = config["alias"]
    return dict(protocol, updater_kernel=kernel)


def run(config_name, out_dir, mass_models, regimes):
    """Run the two independent-key updater roles on the tuning worlds."""
    protocol = panel.load_protocol()
    proto = configured_protocol(protocol, CONFIGS[config_name], mass_models)
    units = [
        u
        for u in panel.plan_units(
            proto,
            "development",
            ["updater"],
            mass_models=mass_models,
            regimes=regimes,
            worlds=TUNING_WORLDS,
        )
        if u["role"] != "updater_doubled"
    ]
    executor = panel.UnitExecutor(proto, panel.protocol_budgets(proto))
    target = Path(out_dir) / config_name
    counts = panel.run_units(
        units, target, (0, 1), executor, log=lambda m: print(m, flush=True)
    )
    entry = panel.manifest_entry(proto, None, panel.protocol_budgets(proto))
    entry["config"] = CONFIGS[config_name]
    panel.write_manifest(target, (0, 1), entry)
    print(json.dumps(counts))


def _tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b, strict=True))


def report(out_dir, ref_dir=None):
    """Per-configuration tuning metrics, as printed lines and a dict."""
    protocol = panel.load_protocol()
    refs = panel.load_units(ref_dir)[0] if ref_dir else {}
    alias_q = protocol["criteria"]["alias_accuracy"]["quantities"]
    result = {}
    for config_dir in sorted(p for p in Path(out_dir).iterdir() if p.is_dir()):
        units = panel.load_units(config_dir)[0]
        rows = {}
        for rec in units.values():
            _, mass_model, regime, world, role = rec["id"].split("/")
            if role != "updater_a":
                continue
            other = units.get(rec["id"].replace("updater_a", "updater_b"))
            if other is None:
                continue
            for k in protocol["panel"]["epoch_counts"]:
                key = (mass_model, regime, k)
                row = rows.setdefault(
                    key,
                    {
                        "ess": {q: [] for q in alias_q},
                        "tv_keys": [],
                        "d2_ref": [],
                        "tv_ref": [],
                        "rungs": [],
                        "rungs_init": [],
                        "not_ok": 0,
                        "not_ok_init": 0,
                        "seconds": [],
                    },
                )
                upd = rec["updates"][k - 1]
                row["rungs_init"] += [
                    r["updates"][0]["rungs_used"] for r in (rec, other)
                ]
                row["not_ok_init"] += sum(
                    r["updates"][0]["status"] != "OK" for r in (rec, other)
                )
                row["rungs"] += [
                    upd["rungs_used"],
                    other["updates"][k - 1]["rungs_used"],
                ]
                row["not_ok"] += sum(
                    r["updates"][k - 1]["status"] != "OK" for r in (rec, other)
                )
                row["seconds"] += [
                    r["updates"][k - 1]["seconds"]
                    for r in (rec, other)
                    if not r["updates"][k - 1]["includes_compile"]
                    and not r.get("cache_cold", False)
                ]
                for q in alias_q:
                    a = rec["summaries"][str(k)]["bins"][q]
                    b = other["summaries"][str(k)]["bins"][q]
                    row["ess"][q].append(panel.key_to_key_ess(a, b, rec["particles"]))
                a = rec["summaries"][str(k)]["bins"]["lnP"]
                b = other["summaries"][str(k)]["bins"]["lnP"]
                row["tv_keys"].append(_tv(a, b))
                r1 = refs.get(
                    panel.unit_id(
                        "development",
                        mass_model,
                        regime,
                        int(world[1:]),
                        "reference_1",
                        k,
                    )
                )
                r2 = refs.get(
                    panel.unit_id(
                        "development",
                        mass_model,
                        regime,
                        int(world[1:]),
                        "reference_2",
                        k,
                    )
                )
                if r1 and r2:
                    x1 = r1["summary"]["bins"]["lnP"]
                    x2 = r2["summary"]["bins"]["lnP"]
                    row["d2_ref"].append(
                        sum(
                            (u - v) * (w - y)
                            for u, v, w, y in zip(a, x1, b, x2, strict=True)
                        )
                    )
                    ref_mean = [0.5 * (u + v) for u, v in zip(x1, x2, strict=True)]
                    row["tv_ref"] += [_tv(a, ref_mean), _tv(b, ref_mean)]
        out = {}
        for (mass_model, regime, k), row in sorted(rows.items()):
            summary = {
                "worlds": len(row["tv_keys"]),
                "key_ess_median": {
                    q: statistics.median(v) for q, v in row["ess"].items()
                },
                "key_ess_min": {q: min(v) for q, v in row["ess"].items()},
                "tv_keys_lnP_max": max(row["tv_keys"]),
                "tv_keys_lnP_median": statistics.median(row["tv_keys"]),
                "rungs_max": max(row["rungs"]),
                "rungs_init_max": max(row["rungs_init"]),
                "not_ok_init": row["not_ok_init"],
                "not_ok": row["not_ok"],
                "seconds_per_update_median": statistics.median(row["seconds"])
                if row["seconds"]
                else None,
            }
            if row["d2_ref"]:
                summary["d2_ref_lnP_mean"] = statistics.fmean(row["d2_ref"])
                summary["tv_ref_lnP_range"] = [min(row["tv_ref"]), max(row["tv_ref"])]
            out[f"{mass_model} {regime} k={k}"] = summary
        result[config_dir.name] = out
    return result


def lane_study(config_names, mass_model, regime_id, lanes, ref_dir, out_path):
    """Lane spread and reference agreement over many keys per tuning world.

    For each configuration and tuning world, ``lanes`` independent keys (the
    ``updater_a`` key of each update folded with the lane index) run the three
    updates. Per world and epoch count it records the lane-mean ln P masses,
    the mean pairwise total variation between lanes, each lane's total
    variation from the mean of the two references (when present), and the
    effective multinomial size implied by the across-lane variance of the
    bins (``sum pbar (1 - pbar) / sum var``).
    """
    import jax
    import numpy as np

    from photomancy.orbit.updater import UpdaterRecord

    protocol = panel.load_protocol()
    refs = panel.load_units(ref_dir)[0] if ref_dir else {}
    r_index = [r["id"] for r in protocol["regimes"]].index(regime_id)
    regime = protocol["regimes"][r_index]
    edges = panel.alias_edges(protocol, regime, mass_model)["lnP"]
    result = {}
    for name in config_names:
        proto = configured_protocol(protocol, CONFIGS[name], [mass_model])
        executor = panel.UnitExecutor(proto, panel.protocol_budgets(proto))
        updater = executor.updater(mass_model, proto["updater_budget"]["particles"])
        rows = {}
        for w in TUNING_WORLDS:
            world = panel.world_data(proto, "development", mass_model, r_index, w)
            masses = {2: [], 3: []}
            seconds = []
            for lane in range(lanes):
                belief = None
                for u in range(3):
                    rec = UpdaterRecord(
                        time=world["times"][u],
                        ra=world["ra"][u],
                        dec=world["dec"][u],
                        ra_err=world["sigma"],
                        dec_err=world["sigma"],
                        record_id=u + 1,
                    )
                    key = jax.random.fold_in(
                        panel.draw_key(
                            proto, "development", mass_model, r_index, w, "updater_a", u
                        ),
                        lane,
                    )
                    t0 = time.perf_counter()
                    if u == 0:
                        belief, _ = updater.init(world["target"], rec, key)
                    else:
                        belief, _ = updater.update(belief, rec, key)
                    jax.block_until_ready(belief)
                    seconds.append(time.perf_counter() - t0)
                    if u + 1 in masses:
                        t_f = panel.future_epoch(regime, u + 1)
                        q = panel.chart_quantities(
                            mass_model, world["target"], belief.particles, t_f
                        )
                        lw = np.asarray(belief.log_weights)
                        wts = np.exp(lw - lw[np.isfinite(lw)].max())
                        wts = np.where(np.isfinite(lw), wts, 0.0)
                        idx = np.clip(
                            np.searchsorted(edges, q["lnP"], "right") - 1, 0, 7
                        )
                        m = np.bincount(idx, weights=wts, minlength=8)
                        masses[u + 1].append(m / m.sum())
            for k, ms in masses.items():
                ms = np.asarray(ms)
                pbar = ms.mean(0)
                pair_tv = [
                    _tv(ms[i], ms[j]) for i in range(lanes) for j in range(i + 1, lanes)
                ]
                var = ms.var(0, ddof=1).sum()
                row = {
                    "lane_mean_lnP": pbar.round(4).tolist(),
                    "pairwise_tv_mean": float(np.mean(pair_tv)),
                    "pairwise_tv_max": float(np.max(pair_tv)),
                    "lane_ess": float(np.sum(pbar * (1 - pbar)) / var)
                    if var > 0
                    else None,
                }
                ids = [
                    panel.unit_id("development", mass_model, regime_id, w, r, k)
                    for r in ("reference_1", "reference_2")
                ]
                if all(i in refs for i in ids):
                    ref = np.mean(
                        [refs[i]["summary"]["bins"]["lnP"] for i in ids], axis=0
                    )
                    lane_tv = [_tv(m, ref) for m in ms]
                    row["ref_lnP"] = ref.round(4).tolist()
                    row["ref_kish_ess"] = [refs[i]["summary"]["kish_ess"] for i in ids]
                    row["lane_tv_ref"] = [float(min(lane_tv)), float(max(lane_tv))]
                    row["lane_mean_tv_ref"] = _tv(pbar, ref)
                rows[f"w{w} k={k}"] = row
            rows[f"w{w} seconds_median"] = float(np.median(seconds[3:]))
        result[name] = rows
        print(name, flush=True)
    Path(out_path).write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=sorted(CONFIGS), default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mass-models", default="ti_window,physical_fixed")
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument(
        "--lanes", type=int, default=0, help="lane study: keys per tuning world"
    )
    parser.add_argument("--configs", default=None, help="lane study configurations")
    args = parser.parse_args(argv)
    if args.lanes:
        panel.enable_x64()
        lane_study(
            args.configs.split(","),
            args.mass_models.split(",")[0],
            args.regimes,
            args.lanes,
            args.references,
            args.out,
        )
    if args.run:
        panel.enable_x64()
        run(
            args.run,
            args.out,
            args.mass_models.split(","),
            args.regimes.split(",") if args.regimes else None,
        )
    if args.report:
        print(json.dumps(report(args.report, args.references), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
