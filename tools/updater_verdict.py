#!/usr/bin/env python3
r"""Grade an updater acceptance panel into the verdict table.

This module holds the grading half of the panel tooling. It reads the unit
records and manifests that ``tools/updater_panel.py --run panel`` wrote into one
directory (``units-{i}of{n}.jsonl`` and ``manifest-{i}of{n}.json``), checks that
the directory is the complete, consistent panel the protocol describes, applies
every criterion of ``docs/updater-protocol.md`` and writes
``docs/updater-verdict.csv``.

Nothing here is part of a work unit's record: the run hash covers the library
modules and the record-defining functions of ``updater_panel.py``, never this
file, so a fix here does not orphan finished records.

Usage, through the panel tool::

    python tools/updater_panel.py --status RUN_DIR --stage panel
    python tools/updater_panel.py --check-panel RUN_DIR
    python tools/updater_panel.py --grade RUN_DIR --mechanics pass \\
        [--verdict-out docs/updater-verdict.csv] [--memory-mb MB]

A directory is first audited (``audit_panel``). Missing units leave the groups
that need them ``incomplete`` and the rest graded; any inconsistency (stale
run hash, off-budget, duplicate or unknown records, manifests that disagree,
a dirty tree, a draft protocol) refuses to grade and writes nothing.
"""

import csv
import importlib.util
import io
import json
import math
import re
from pathlib import Path

COLUMNS = (
    "regime",
    "mass_model",
    "epoch_count",
    "quantity",
    "worlds",
    "covered_count",
    "covered_50_count",
    "pit_sup_distance",
    "d2_mean",
    "d2_upper",
    "exhausted_count",
    "exhausted_cp_upper",
    "schedule_deviations_keys",
    "schedule_deviations_budget",
    "tv_median",
    "tv_p95",
    "latency_ms_per_update",
    "memory_mb",
    "seeds",
    "verdict_coverage",
    "verdict_calibration",
    "verdict_alias",
    "verdict_failed_updates",
    "verdict_schedule",
    "verdict_mechanics",
    "protocol_commit",
    "code_commit",
)
CRITERIA = (
    "coverage",
    "calibration",
    "alias",
    "failed_updates",
    "schedule",
    "mechanics",
)
VERDICTS = ("pass", "fail", "incomplete", "na")

# Roles each criterion reads. Coverage, calibration and failed updates grade
# the nominal-budget run of the first key; the alias statistic pairs the two
# independent keys with the two independent references; schedule stability
# compares the two keys and the doubled budget.
GRADED_ROLE = "updater_a"
ALIAS_PAIRS = (("updater_a", "reference_1"), ("updater_b", "reference_2"))
SCHEDULE_ROLES = ("updater_a", "updater_b", "updater_doubled")
ARCHITECTURES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "x86_64": "x86_64",
    "amd64": "x86_64",
}
DEFAULT_CHUNK = "default"

_TOOL = {}


def load_tool():
    """The sibling ``updater_panel.py`` loaded from its file (cached)."""
    if "module" not in _TOOL:
        path = Path(__file__).with_name("updater_panel.py")
        spec = importlib.util.spec_from_file_location("updater_panel", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TOOL["module"] = module
    return _TOOL["module"]


# ---------------------------------------------------------------------------
# Decisions: one small function per criterion, so each is tested on its own
# ---------------------------------------------------------------------------


def decide_coverage(covered, critical):
    """Fail at ``critical`` covered worlds or fewer."""
    return "pass" if covered > critical else "fail"


def decide_calibration(sup_distance, epsilon):
    """Fail when the PIT sup-distance exceeds ``epsilon`` (NaN fails)."""
    return "pass" if sup_distance <= epsilon else "fail"


def decide_alias(upper, delta2):
    """Fail when the upper bound exceeds ``delta2`` (NaN fails)."""
    return "pass" if upper <= delta2 else "fail"


def decide_failed_updates(failed, critical):
    """Fail when more than ``critical`` worlds have a non-OK update."""
    return "pass" if failed <= critical else "fail"


def decide_schedule(deviations_keys, deviations_budget, max_count):
    """Fail when either comparison has more than ``max_count`` deviations."""
    ok = deviations_keys <= max_count and deviations_budget <= max_count
    return "pass" if ok else "fail"


# ---------------------------------------------------------------------------
# Directory audit
# ---------------------------------------------------------------------------


def _load_directory(tool, path):
    """Records, malformed lines, duplicates and unit files of a directory."""
    path = Path(path)
    records, duplicates, malformed = {}, [], 0
    for rec in tool.read_records(path):
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str):
            malformed += 1
        elif rec["id"] in records:
            duplicates.append(rec["id"])
        else:
            records[rec["id"]] = rec
    return records, sorted(set(duplicates)), malformed


def _arch(platform_text):
    lowered = str(platform_text).lower()
    for token, name in ARCHITECTURES.items():
        if token in lowered:
            return name
    return str(platform_text)


def _chunk(argv):
    argv = [str(a) for a in (argv or [])]
    for i, arg in enumerate(argv):
        if arg == "--chunk" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--chunk="):
            return arg.split("=", 1)[1]
    return DEFAULT_CHUNK


def _manifests(path):
    """Invocation entries per manifest file, and unreadable manifest names."""
    entries, unreadable = [], []
    for file in sorted(Path(path).glob("manifest-*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            runs = data["invocations"]
            if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs):
                raise TypeError("invocations")
        except (OSError, ValueError, KeyError, TypeError):
            unreadable.append(file.name)
            continue
        entries += [(file.name, run) for run in runs]
    return entries, unreadable


def _one_value(entries, field, key=None):
    """Distinct values of a manifest field, in a stable order."""
    key = key or (lambda entry: entry.get(field))
    return sorted({str(key(e)) for _, e in entries})


def audit_manifests(tool, protocol, path, require_frozen=True, allow_dirty=False):
    """Provenance of a panel directory: reasons that forbid grading, and the commits.

    Every invocation of every shard must have run the current code and
    protocol content at the protocol budgets, and all must agree on the JAX
    version, host architecture, recorded chunk and commits.
    """
    path = Path(path)
    entries, unreadable = _manifests(path)
    reasons = []
    if unreadable:
        reasons.append(f"unreadable manifests: {unreadable}")
    if not entries:
        reasons.append("no manifest entries: the run's provenance is unknown")
    shards = {
        re.search(r"units-(\d+of\d+)", f.name).group(1)
        for f in path.glob("units-*.jsonl")
        if re.search(r"units-(\d+of\d+)", f.name)
    }
    manifest_shards = {
        re.search(r"manifest-(\d+of\d+)", name).group(1)
        for name, _ in entries
        if re.search(r"manifest-(\d+of\d+)", name)
    }
    if shards - manifest_shards:
        reasons.append(
            f"unit files without a manifest: {sorted(shards - manifest_shards)}"
        )
    current = tool.run_hash(protocol)
    stale = _one_value(entries, "run_hash")
    if entries and stale != [current]:
        reasons.append(
            f"manifests record run hashes {stale}, current {current} "
            "(another code or protocol content)"
        )
    declared = tool.declaration_hash(protocol)
    hashes = _one_value(entries, "protocol_hash")
    if entries and hashes != [declared]:
        reasons.append(
            f"manifests record protocol hashes {hashes}, current {declared} "
            "(the protocol changed since the panel ran)"
        )
    if require_frozen and protocol.get("status") != "frozen":
        reasons.append(f"protocol status is {protocol.get('status')!r}, not 'frozen'")
    statuses = _one_value(entries, "protocol_status")
    if require_frozen and entries and statuses != ["frozen"]:
        reasons.append(
            f"manifests record protocol statuses {statuses}, not only 'frozen'"
        )
    if any(e.get("protocol_budgets") is not True for _, e in entries):
        reasons.append("an invocation ran at non-protocol budgets (a test run)")
    for label, values in (
        ("jax versions", _one_value(entries, "jax")),
        ("jaxlib versions", _one_value(entries, "jaxlib")),
        (
            "host architectures",
            _one_value(entries, None, lambda e: _arch(e.get("platform", ""))),
        ),
        ("recorded chunks", _one_value(entries, None, lambda e: _chunk(e.get("argv")))),
        ("code commits", _one_value(entries, "code_commit")),
        ("protocol commits", _one_value(entries, "protocol_commit")),
    ):
        if len(values) > 1:
            reasons.append(f"manifests disagree on {label}: {values}")
    dirty = sorted({name for name, e in entries if e.get("code_dirty") is not False})
    if dirty and not allow_dirty:
        reasons.append(f"invocations with a dirty or unknown code tree in: {dirty}")
    interrupted = sum(1 for _, e in entries if not e.get("finished"))
    commits = _one_value(entries, "code_commit")
    protocol_commits = _one_value(entries, "protocol_commit")
    return {
        "reasons": reasons,
        "invocations": len(entries),
        "interrupted_invocations": interrupted,
        "code_commit": commits[0] if len(commits) == 1 else "",
        "code_dirty": bool(dirty),
        "protocol_commit": protocol_commits[0] if len(protocol_commits) == 1 else "",
        "jax": _one_value(entries, "jax"),
        "architecture": _one_value(
            entries, None, lambda e: _arch(e.get("platform", ""))
        ),
        "chunk": _one_value(entries, None, lambda e: _chunk(e.get("argv"))),
    }


def audit_panel(
    tool, protocol, path, stage="panel", require_frozen=True, allow_dirty=False
):
    """Completeness and consistency of a panel run directory.

    Plans every unit of ``stage`` and reports the missing, duplicate and
    unknown unit ids, records under another run hash, records at other
    budgets, and the manifest provenance (``audit_manifests``). ``usable``
    holds the records with the current run hash at the protocol budgets.

    Returns:
        Dict with ``status`` (``refused`` when ``refusals`` is not empty,
        ``incomplete`` when units are missing, else ``complete``),
        ``refusals``, ``missing``, ``planned``, ``usable`` and the provenance.
    """
    records, duplicates, malformed = _load_directory(tool, path)
    refusals = []
    if malformed:
        refusals.append(f"{malformed} records have no string id")
    if duplicates:
        refusals.append(f"{len(duplicates)} duplicate unit ids: {duplicates[:3]}")
    current = tool.run_hash(protocol)
    stale = sorted(i for i, r in records.items() if r.get("run_hash") != current)
    if stale:
        hashes = sorted({str(records[i].get("run_hash")) for i in stale})
        refusals.append(
            f"{len(stale)} records were run under other code or protocol content "
            f"(run hashes {hashes}, current {current}); first: {stale[:3]}"
        )

    def at_budget(record):
        try:
            return bool(tool._record_at_protocol_budget(protocol, record))
        except (KeyError, IndexError, TypeError):
            return False

    off_budget = sorted(i for i, r in records.items() if not at_budget(r))
    if off_budget:
        refusals.append(
            f"{len(off_budget)} records at non-protocol budgets: {off_budget[:3]}"
        )
    planned = [u["id"] for u in tool.plan_units(protocol, stage, list(tool.PARTS))]
    unknown = sorted(set(records) - set(planned))
    if unknown:
        refusals.append(f"{len(unknown)} records are not {stage} units: {unknown[:3]}")
    usable = {
        i: r
        for i, r in records.items()
        if r.get("run_hash") == current and i not in off_budget
    }
    provenance = audit_manifests(tool, protocol, path, require_frozen, allow_dirty)
    refusals += provenance["reasons"]
    missing = [i for i in planned if i not in usable]
    status = "refused" if refusals else ("incomplete" if missing else "complete")
    return {
        "status": status,
        "refusals": refusals,
        "missing": missing,
        "planned": len(planned),
        "present": len(planned) - len(missing),
        "usable": usable,
        "provenance": provenance,
    }


def audit_summary(audit):
    """The audit without its records, for printing."""
    out = {k: v for k, v in audit.items() if k != "usable"}
    out["missing"] = {"count": len(audit["missing"]), "first": audit["missing"][:5]}
    return out


# ---------------------------------------------------------------------------
# Statistics of one group
# ---------------------------------------------------------------------------


def pit_sup_distance(pits):
    """Sup-distance between the empirical CDF of ``pits`` and the uniform CDF."""
    import numpy as np

    u = np.sort(np.asarray(pits, float))
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def _finite(values):
    import numpy as np

    return bool(np.all(np.isfinite(np.asarray(values, float))))


class _Group:
    """The records of one (regime, mass model) block, by world and role."""

    def __init__(self, tool, units, stage, mass_model, regime, n):
        self.tool, self.units, self.stage = tool, units, stage
        self.mass_model, self.regime, self.n = mass_model, regime, n

    def record(self, world, role, k=None):
        ident = self.tool.unit_id(
            self.stage, self.mass_model, self.regime, world, role, k
        )
        return self.units.get(ident)

    def records(self, role, k=None):
        """One record per world, ``None`` where absent."""
        return [self.record(w, role, k) for w in range(self.n)]

    def complete(self, roles, k=None):
        """Worlds that have every role in ``roles``, and whether all do."""
        have = [
            all(
                self.record(w, role, k if role.startswith("ref") else None) is not None
                for role in roles
            )
            for w in range(self.n)
        ]
        return sum(have), all(have)


def _summary(record, k):
    if record is None:
        return {}
    return record.get("summaries", {}).get(str(k)) or {}


def _incomplete_row(row, columns):
    for c in columns:
        row[f"verdict_{c}"] = "incomplete"


def cell_statistics(group, k, quantity, tool, protocol, crit):
    """Coverage and calibration of one cell: counts, statistic and verdicts."""
    import numpy as np

    out = {"coverage": "incomplete", "calibration": "incomplete"}
    stats = {}
    worlds, complete = group.complete((GRADED_ROLE,))
    stats["worlds"] = worlds
    if not complete:
        return out, stats
    covered = covered_50 = 0
    pits, hollow = [], 0
    for rec in group.records(GRADED_ROLE):
        s = _summary(rec, k)
        truth = rec.get("truth", {}).get(str(k), {}).get(quantity)
        ends = s.get("intervals", {}).get(quantity)
        pit = s.get("pit", {}).get(quantity)
        if truth is None or ends is None or not math.isfinite(float(truth)):
            hollow += 1
            continue
        lo, hi = ends["90"]
        covered += bool(lo <= truth <= hi)
        lo, hi = ends["50"]
        covered_50 += bool(lo <= truth <= hi)
        if pit is None or not math.isfinite(float(pit)):
            hollow += 1
        else:
            pits.append(float(pit))
    stats["covered_count"], stats["covered_50_count"] = covered, covered_50
    out["coverage"] = decide_coverage(covered, crit["coverage_critical_count"])
    if hollow:
        # A world whose belief has no scorable interval or PIT counts against
        # calibration; its CDF value cannot be placed.
        out["calibration"] = "fail"
        return out, stats
    sup = pit_sup_distance(np.asarray(pits))
    stats["pit_sup_distance"] = sup
    out["calibration"] = decide_calibration(sup, crit["dkw_epsilon"])
    return out, stats


def alias_statistics(group, k, quantity, protocol, crit):
    """D2 statistic and TV of one group and alias quantity, or None if unscorable.

    Returns:
        ``(verdict, stats)``; a hollow or non-finite bin vector, or a missing
        unit, makes the verdict ``incomplete``.
    """
    import numpy as np

    alias = protocol["criteria"]["alias_accuracy"]
    stats = {}
    roles = [r for pair in ALIAS_PAIRS for r in pair]
    worlds, complete = group.complete(roles, k)
    stats["worlds"] = worlds
    if not complete:
        return "incomplete", stats
    nbins = alias["bins"]
    d2, tv = [], []
    for w in range(group.n):
        vec = {}
        for role in roles:
            rec = group.record(w, role, k if role.startswith("ref") else None)
            s = (
                _summary(rec, k)
                if rec["kind"] == "updater"
                else rec.get("summary") or {}
            )
            bins = s.get("bins", {}).get(quantity)
            if bins is None or len(bins) != nbins or not _finite(bins):
                return "incomplete", stats
            vec[role] = np.asarray(bins, float)
        (ua, ra), (ub, rb) = ALIAS_PAIRS
        d2.append(float((vec[ua] - vec[ra]) @ (vec[ub] - vec[rb])))
        tv.append(0.5 * float(np.abs(vec[ua] - vec[ra]).sum()))
    d2 = np.asarray(d2)
    se = float(d2.std(ddof=1) / math.sqrt(len(d2)))
    stats["d2_mean"] = float(d2.mean())
    stats["d2_upper"] = stats["d2_mean"] + crit["alias_z"] * se
    stats["tv_median"] = float(np.median(tv))
    stats["tv_p95"] = float(np.percentile(tv, 95))
    if alias["delta2"] > alias["delta2_max"]:
        return "incomplete", stats
    return decide_alias(stats["d2_upper"], alias["delta2"]), stats


def update_statistics(group, update, protocol, crit, tool):
    """Failed-update count of one (regime, mass model, update) group."""
    stats = {}
    worlds, complete = group.complete((GRADED_ROLE,))
    stats["worlds"] = worlds
    warm = []
    for role in ("updater_a", "updater_b"):
        for rec in group.records(role):
            try:
                u = rec["updates"][update]
                if u.get("includes_compile") is False:
                    warm.append(float(u["seconds"]))
            except (TypeError, KeyError, IndexError):
                continue
    if warm:
        stats["latency_ms_per_update"] = 1000.0 * sum(warm) / len(warm)
    if not complete:
        return "incomplete", stats
    failed = 0
    for rec in group.records(GRADED_ROLE):
        try:
            status = rec["updates"][update]["status"]
        except (TypeError, KeyError, IndexError):
            status = None
        failed += status != "OK"
    stats["exhausted_count"] = failed
    conf = protocol["criteria"]["failed_updates"]["upper_bound_confidence"]
    stats["exhausted_cp_upper"] = tool.clopper_pearson_upper(failed, group.n, conf)
    return decide_failed_updates(failed, crit["failed_updates_critical_count"]), stats


def schedule_statistics(group, k, protocol):
    """Schedule deviations of one group: keys, and nominal against doubled."""
    import numpy as np

    sched = protocol["criteria"]["schedule_stability"]
    stats = {}
    worlds, complete = group.complete(SCHEDULE_ROLES)
    stats["worlds"] = worlds
    if not complete:
        return "incomplete", stats
    p = {
        role: np.array(
            [
                float(_summary(rec, k).get("schedule_p", math.nan))
                for rec in group.records(role)
            ]
        )
        for role in SCHEDULE_ROLES
    }
    tol = sched["tolerance"]
    # A comparison that cannot be evaluated (NaN) counts as a deviation.
    keys = int(np.sum(~(np.abs(p["updater_a"] - p["updater_b"]) <= tol)))
    budget = int(np.sum(~(np.abs(p["updater_a"] - p["updater_doubled"]) <= tol)))
    stats["schedule_deviations_keys"], stats["schedule_deviations_budget"] = (
        keys,
        budget,
    )
    return decide_schedule(keys, budget, sched["max_count"]), stats


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def _blank_row(regime, mass_model, epoch_count, quantity, seeds, commits):
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        regime=regime,
        mass_model=mass_model,
        epoch_count=epoch_count,
        quantity=quantity,
        seeds=seeds,
        protocol_commit=commits.get("protocol_commit", ""),
        code_commit=commits.get("code_commit", ""),
    )
    for c in CRITERIA:
        row[f"verdict_{c}"] = "na"
    return row


def thresholds(tool, protocol, n):
    """Critical values of the criteria at ``n`` worlds per group.

    The same formulas as the protocol's derived values (``--dry-run``), kept
    to the four the grader applies: the coverage critical count (``-1`` when
    ``n`` is too small for any count to have a tail below its level, so that
    no cell can fail), the DKW bound, the alias z quantile and the
    failed-update critical count. A test checks them against the dry run at
    the protocol's 400 worlds.
    """
    import statistics

    g = tool.group_counts(protocol)
    crit = protocol["criteria"]
    groups = g["regimes"] * g["mass_models"] * g["epoch_counts"]
    cells = groups * g["quantities"]
    cov = crit["coverage"]
    try:
        coverage = tool.coverage_critical_count(
            n, cov["interval"], cov["family_alpha"] / cells
        )
    except ValueError:
        coverage = -1
    alias = crit["alias_accuracy"]
    fu = crit["failed_updates"]
    return {
        "coverage_critical_count": coverage,
        "dkw_epsilon": tool.dkw_epsilon(n, crit["calibration"]["family_alpha"] / cells),
        "alias_z": statistics.NormalDist().inv_cdf(
            1.0 - alias["family_alpha"] / (groups * g["alias_quantities"])
        ),
        "failed_updates_critical_count": tool.upper_tail_critical_count(
            n,
            fu["acceptable_rate"],
            fu["family_alpha"] / (g["regimes"] * g["mass_models"] * g["updates"]),
        ),
    }


def build_rows(
    tool, protocol, units, commits=None, stage="panel", mechanics=None, memory_mb=None
):
    """The verdict rows of a set of usable unit records.

    Args:
        tool: The loaded ``updater_panel`` module.
        protocol: The parameter block.
        units: Map of unit id to record (current run hash, protocol budgets).
        commits: ``protocol_commit`` and ``code_commit`` for every row.
        stage: Stage of the unit ids.
        mechanics: ``"pass"`` or ``"fail"``, the result of the mechanics tests
            at the graded commit; any other value leaves the criterion
            ``incomplete``.
        memory_mb: Peak memory of the intended batch, written on the
            failed-update rows when given.

    Returns:
        List of row dicts in column order: the 64 cells, the position-angle
        alias rows, the failed-update groups and the schedule groups.
    """
    commits = commits or {}
    panel = protocol["panel"]
    n = tool.stage_worlds(protocol, stage)
    crit = thresholds(tool, protocol, n)
    seeds = f"{protocol['prng']['root_seed']}/{stage}"
    mech = mechanics if mechanics in ("pass", "fail") else "incomplete"
    alias_q = protocol["criteria"]["alias_accuracy"]["quantities"]
    rows = []

    def new(*args):
        row = _blank_row(*args, seeds, commits)
        row["verdict_mechanics"] = mech
        return row

    def fill(row, stats):
        for key, value in stats.items():
            row[key] = value

    blocks = [
        (r["id"], m, _Group(tool, units, stage, m, r["id"], n))
        for r in protocol["regimes"]
        for m in panel["mass_models"]
    ]
    for regime, mass_model, group in blocks:
        for k in panel["epoch_counts"]:
            for q in panel["quantities"]:
                row = new(regime, mass_model, k, q)
                verdicts, stats = cell_statistics(group, k, q, tool, protocol, crit)
                row["verdict_coverage"] = verdicts["coverage"]
                row["verdict_calibration"] = verdicts["calibration"]
                fill(row, stats)
                if q in alias_q:
                    row["verdict_alias"], stats = alias_statistics(
                        group, k, q, protocol, crit
                    )
                    stats.pop("worlds", None)
                    fill(row, stats)
                rows.append(row)
    for regime, mass_model, group in blocks:
        for k in panel["epoch_counts"]:
            row = new(regime, mass_model, k, "future_pa")
            row["verdict_alias"], stats = alias_statistics(
                group, k, "future_pa", protocol, crit
            )
            fill(row, stats)
            rows.append(row)
    for regime, mass_model, group in blocks:
        for u in range(len(panel["updates"])):
            row = new(regime, mass_model, u + 1, "updates")
            row["verdict_failed_updates"], stats = update_statistics(
                group, u, protocol, crit, tool
            )
            fill(row, stats)
            if memory_mb is not None:
                row["memory_mb"] = memory_mb
            rows.append(row)
    for regime, mass_model, group in blocks:
        for k in panel["epoch_counts"]:
            row = new(regime, mass_model, k, "schedule")
            row["verdict_schedule"], stats = schedule_statistics(group, k, protocol)
            fill(row, stats)
            rows.append(row)
    return rows


def panel_verdict(rows):
    """``fail`` if any verdict fails, else ``incomplete`` if any is, else ``pass``."""
    found = {row[f"verdict_{c}"] for row in rows for c in CRITERIA}
    if "fail" in found:
        return "fail"
    if "incomplete" in found or not rows:
        return "incomplete"
    return "pass"


def verdict_counts(rows):
    """Count of each verdict per criterion."""
    return {
        c: {v: sum(row[f"verdict_{c}"] == v for row in rows) for v in VERDICTS}
        for c in CRITERIA
    }


def _format(value):
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def rows_to_csv(rows):
    """The verdict table as CSV text, columns in protocol order."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([_format(row[c]) for c in COLUMNS])
    return buffer.getvalue()


def grade_panel(
    tool,
    protocol,
    path,
    stage="panel",
    require_frozen=True,
    allow_dirty=False,
    mechanics=None,
    memory_mb=None,
):
    """Audit a panel directory and grade it.

    Returns:
        Dict with ``status`` (``refused``, ``incomplete``, ``fail`` or
        ``pass``), the ``audit`` (without records) and, unless refused, the
        ``rows`` and per-criterion ``counts``. A refused directory has no rows.
    """
    audit = audit_panel(tool, protocol, path, stage, require_frozen, allow_dirty)
    if audit["status"] == "refused":
        return {"status": "refused", "audit": audit_summary(audit), "rows": None}
    prov = audit["provenance"]
    code = prov["code_commit"] + ("-dirty" if prov["code_dirty"] else "")
    rows = build_rows(
        tool,
        protocol,
        audit["usable"],
        {"protocol_commit": prov["protocol_commit"], "code_commit": code},
        stage,
        mechanics,
        memory_mb,
    )
    verdict = panel_verdict(rows)
    if audit["status"] == "incomplete" and verdict == "pass":
        verdict = "incomplete"
    return {
        "status": verdict,
        "audit": audit_summary(audit),
        "rows": rows,
        "counts": verdict_counts(rows),
    }


def write_verdict(tool, result, out_path):
    """Write the table of a graded result atomically."""
    tool.atomic_write_text(out_path, rows_to_csv(result["rows"]))


def report_text(result):
    """Printable report: the status, the audit and the verdict counts."""
    body = {k: v for k, v in result.items() if k != "rows"}
    return json.dumps(body, indent=2, sort_keys=True, default=str)
