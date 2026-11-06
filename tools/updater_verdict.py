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
import subprocess
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


HEX_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


def provenance_gaps(entry):
    """Fields of a manifest entry that are absent or not populated.

    The commits must look like commit ids, the versions and platform must be
    non-empty strings, the arguments a non-empty list, the dirty flag a
    boolean and the budgets a dict. Agreement between shards on an absent
    field is not provenance.
    """
    gaps = []
    for key in ("protocol_commit", "code_commit"):
        if not isinstance(entry.get(key), str) or not HEX_COMMIT.match(entry[key]):
            gaps.append(key)
    for key in ("jax", "jaxlib", "platform", "run_hash", "protocol_hash"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            gaps.append(key)
    if not isinstance(entry.get("argv"), list) or not entry["argv"]:
        gaps.append("argv")
    if not isinstance(entry.get("code_dirty"), bool):
        gaps.append("code_dirty")
    if not isinstance(entry.get("budgets"), dict):
        gaps.append("budgets")
    return gaps


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
    reasons, notes = [], []
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
    expected = tool.protocol_budgets(protocol)
    wrong = sorted({name for name, e in entries if e.get("budgets") != expected})
    if wrong:
        reasons.append(
            f"recorded budgets differ from the protocol's {expected} in: {wrong}"
        )
    for name, entry in entries:
        missing_fields = provenance_gaps(entry)
        if missing_fields:
            reasons.append(f"unpopulated provenance in {name}: {missing_fields}")
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
    elif dirty:
        notes.append(f"exploratory: dirty or unknown code tree in {dirty}")
    interrupted = sum(1 for _, e in entries if not e.get("finished"))
    commits = _one_value(entries, "code_commit")
    protocol_commits = _one_value(entries, "protocol_commit")
    return {
        "reasons": reasons,
        "notes": notes,
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


STATUSES = ("OK", "NO_SUPPORT", "BUDGET_EXHAUSTED", "LOW_ESS", "INCOMPLETE_LADDER")
NORMALIZATION_TOLERANCE = 1e-6


def _number(x):
    """``x`` as a finite float, else None (booleans and non-numbers included)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def valid_probability_vector(values, size):
    """Whether ``values`` is ``size`` finite, nonnegative numbers summing to one."""
    if not isinstance(values, (list, tuple)) or len(values) != size:
        return False
    nums = [_number(v) for v in values]
    if any(v is None or v < 0.0 for v in nums):
        return False
    return abs(sum(nums) - 1.0) <= NORMALIZATION_TOLERANCE


def valid_unit_interval(value):
    """Whether ``value`` is a finite number in [0, 1]."""
    x = _number(value)
    return x is not None and 0.0 <= x <= 1.0


def valid_intervals(ends):
    """Whether an interval pair is finite, ordered and nested (50 inside 90)."""
    try:
        lo90, hi90 = (_number(v) for v in ends["90"])
        lo50, hi50 = (_number(v) for v in ends["50"])
    except (KeyError, TypeError, ValueError):
        return False
    if None in (lo90, hi90, lo50, hi50):
        return False
    return lo90 <= lo50 <= hi50 <= hi90


def _empty_fit(summary):
    """Whether a fit lacks valid retained-particle and Kish ESS diagnostics."""
    if not isinstance(summary, dict):
        return True
    particles = summary.get("particles")
    ess = _number(summary.get("kish_ess"))
    return type(particles) is not int or particles <= 0 or ess is None or ess <= 0.0


def record_problems(protocol, record):
    """What is missing or invalid in a unit record, as a list of strings.

    An updater record needs a declared status for every update, and for each
    epoch count a finite truth, finite ordered nested intervals and PITs in
    [0, 1] for every graded quantity, normalized bin masses for every alias
    quantity and a schedule probability in [0, 1]. A reference record needs
    normalized bin masses for every alias quantity from a fit that kept a
    weighted draw. The grader scores a record only when it has no problems;
    missing evidence is never counted as a failure or a tolerated deviation.
    """
    panel = protocol["panel"]
    alias = protocol["criteria"]["alias_accuracy"]
    nbins = alias["bins"]
    problems = []
    kind = record.get("kind")
    if kind == "updater":
        updates = record.get("updates")
        if not isinstance(updates, list) or len(updates) != len(panel["updates"]):
            problems.append("updates")
        else:
            for u, entry in enumerate(updates):
                if not isinstance(entry, dict) or entry.get("status") not in STATUSES:
                    problems.append(f"status of update {u}")
        for k in panel["epoch_counts"]:
            problems += _summary_problems(protocol, record, k, nbins)
    elif kind == "reference":
        summary = record.get("summary")
        if _empty_fit(summary):
            problems.append("reference fit has missing or invalid fit diagnostics")
        else:
            problems += _bin_problems(summary, alias["quantities"], nbins)
    else:
        problems.append("kind")
    return problems


def _bin_problems(summary, quantities, nbins):
    bins = summary.get("bins") if isinstance(summary, dict) else None
    if not isinstance(bins, dict):
        return ["bins"]
    return [
        f"bins {q}"
        for q in quantities
        if not valid_probability_vector(bins.get(q), nbins)
    ]


def _summary_problems(protocol, record, k, nbins):
    summaries = record.get("summaries")
    summary = summaries.get(str(k)) if isinstance(summaries, dict) else None
    if _empty_fit(summary):
        return [f"summary at {k} epochs is empty"]
    problems = _bin_problems(
        summary, protocol["criteria"]["alias_accuracy"]["quantities"], nbins
    )
    truth = (record.get("truth") or {}).get(str(k))
    intervals = summary.get("intervals")
    pits = summary.get("pit")
    for q in protocol["panel"]["quantities"]:
        if not isinstance(truth, dict) or _number(truth.get(q)) is None:
            problems.append(f"truth {q} at {k}")
        if not isinstance(intervals, dict) or not valid_intervals(intervals.get(q)):
            problems.append(f"intervals {q} at {k}")
        if not isinstance(pits, dict) or not valid_unit_interval(pits.get(q)):
            problems.append(f"pit {q} at {k}")
    if not valid_unit_interval(summary.get("schedule_p")):
        problems.append(f"schedule_p at {k}")
    return problems


def valid_units(protocol, units):
    """The records without problems, and the problems of the others by id."""
    good, bad = {}, {}
    for ident, record in units.items():
        problems = record_problems(protocol, record)
        if problems:
            bad[ident] = problems
        else:
            good[ident] = record
    return good, bad


class _Group:
    """The valid records of one (regime, mass model) block, by world and role."""

    def __init__(self, tool, units, stage, mass_model, regime, n):
        self.tool, self.units, self.stage = tool, units, stage
        self.mass_model, self.regime, self.n = mass_model, regime, n

    def record(self, world, role, k=None):
        ident = self.tool.unit_id(
            self.stage, self.mass_model, self.regime, world, role, k
        )
        return self.units.get(ident)

    def records(self, role, k=None):
        """One record per world, ``None`` where absent or invalid."""
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
    return record["summaries"][str(k)]


def cell_statistics(group, k, quantity, crit):
    """Coverage and calibration of one cell: counts, statistic and verdicts."""
    out = {"coverage": "incomplete", "calibration": "incomplete"}
    stats = {}
    worlds, complete = group.complete((GRADED_ROLE,))
    stats["worlds"] = worlds
    if not complete:
        return out, stats
    covered = covered_50 = 0
    pits = []
    for rec in group.records(GRADED_ROLE):
        s = _summary(rec, k)
        truth = rec["truth"][str(k)][quantity]
        lo, hi = s["intervals"][quantity]["90"]
        covered += bool(lo <= truth <= hi)
        lo, hi = s["intervals"][quantity]["50"]
        covered_50 += bool(lo <= truth <= hi)
        pits.append(float(s["pit"][quantity]))
    stats["covered_count"], stats["covered_50_count"] = covered, covered_50
    out["coverage"] = decide_coverage(covered, crit["coverage_critical_count"])
    sup = pit_sup_distance(pits)
    stats["pit_sup_distance"] = sup
    out["calibration"] = decide_calibration(sup, crit["dkw_epsilon"])
    return out, stats


def alias_statistics(group, k, quantity, protocol, crit):
    """D2 statistic and TV of one group and alias quantity.

    Returns:
        ``(verdict, stats)``. ``verdict`` is ``pass`` or ``fail`` from the
        statistics alone, or ``incomplete`` when a run is missing or invalid
        or the tolerance exceeds its cap. The caller blocks a ``pass`` that
        lacks certification evidence.
    """
    import numpy as np

    alias = protocol["criteria"]["alias_accuracy"]
    stats = {}
    roles = [r for pair in ALIAS_PAIRS for r in pair]
    worlds, complete = group.complete(roles, k)
    stats["worlds"] = worlds
    if not complete:
        return "incomplete", stats
    d2, tv = [], []
    for w in range(group.n):
        vec = {}
        for role in roles:
            rec = group.record(w, role, k if role.startswith("ref") else None)
            s = _summary(rec, k) if rec["kind"] == "updater" else rec["summary"]
            vec[role] = np.asarray(s["bins"][quantity], float)
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
    if not complete:
        return "incomplete", stats
    failed = sum(
        rec["updates"][update]["status"] != "OK" for rec in group.records(GRADED_ROLE)
    )
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
        role: np.array([_summary(rec, k)["schedule_p"] for rec in group.records(role)])
        for role in SCHEDULE_ROLES
    }
    tol = sched["tolerance"]
    keys = int(np.sum(np.abs(p["updater_a"] - p["updater_b"]) > tol))
    budget = int(np.sum(np.abs(p["updater_a"] - p["updater_doubled"]) > tol))
    stats["schedule_deviations_keys"], stats["schedule_deviations_budget"] = (
        keys,
        budget,
    )
    return decide_schedule(keys, budget, sched["max_count"]), stats


# ---------------------------------------------------------------------------
# Evidence from outside the panel: certification and mechanics
# ---------------------------------------------------------------------------


def freeze_binding(tool, protocol):
    """Binding block that ``--freeze`` adds to its report.

    The freeze report is evidence for a later grade only through what it was
    computed for: the run hash (protocol content and implementation), the
    budgets and the alias tolerance in force.
    """
    return {
        "run_hash": tool.run_hash(protocol),
        "budgets": tool.protocol_budgets(protocol),
        "alias_delta2": protocol["criteria"]["alias_accuracy"]["delta2"],
    }


EXACT_CASES = {
    "ti_window": "ti_window",
    "physical_fixed": "physical_fixed",
    "ti_binding": "ti_window",
}


def freeze_evidence(tool, protocol, report):
    """Validate report structure and resolve eligibility per mass model and group.

    ``valid`` describes the whole freeze. ``usable`` requires a valid binding
    and schema but allows scoped certification, exact-check and power failures.
    Those failures make only their affected aliases incomplete, even when the
    numerical alias bound fails. File integrity is checked by ``grade_panel``.
    """
    alias = protocol["criteria"]["alias_accuracy"]
    reasons = []
    out = {
        "valid": False,
        "usable": False,
        "reasons": reasons,
        "certified": {},
        "exact_ok": {},
        "power_ok": set(),
    }
    if not isinstance(report, dict):
        reasons.append("freeze evidence is not a freeze report")
        return out
    structural = []
    if report.get("status") not in ("frozen", "incomplete"):
        structural.append("freeze report has an invalid status")
    if not isinstance(report.get("reasons"), list):
        structural.append("freeze report's reasons are not a list")
    if report.get("status") != "frozen" or report.get("reasons"):
        reasons.append("freeze report is not frozen with no reasons")
    binding = report.get("binding")
    binding = binding if isinstance(binding, dict) else {}
    if binding.get("run_hash") != tool.run_hash(protocol):
        structural.append("freeze report is bound to another run hash")
    if binding.get("budgets") != tool.protocol_budgets(protocol):
        structural.append("freeze report is bound to other budgets")
    alias_report = report.get("alias")
    alias_report = alias_report if isinstance(alias_report, dict) else {}
    frozen = _number(alias_report.get("delta2"))
    bound = _number(binding.get("alias_delta2"))
    if not (
        frozen is not None
        and frozen > 0.0
        and bound is not None
        and math.isclose(
            min(frozen, alias["delta2_max"]), alias["delta2"], rel_tol=1e-9
        )
        and math.isclose(bound, alias["delta2"], rel_tol=1e-9)
    ):
        structural.append("freeze report's alias tolerance is not the protocol's")
    certification = report.get("certification")
    certification = certification if isinstance(certification, dict) else {}
    for mass_model in protocol["panel"]["mass_models"]:
        cert = certification.get(mass_model)
        out["certified"][mass_model] = (
            isinstance(cert, dict) and cert.get("status") == "certified"
        )
        if not out["certified"][mass_model]:
            reasons.append(f"{mass_model} reference is not certified in the report")
    exact_report = report.get("exact_checks")
    exact = exact_report.get("verdicts") if isinstance(exact_report, dict) else None
    if not isinstance(exact, dict) or set(exact) != set(EXACT_CASES):
        structural.append("exact checks must contain the canonical cases exactly once")
        exact = {}
    elif any(v not in ("pass", "fail", "incomplete") for v in exact.values()):
        structural.append("exact checks have invalid verdicts")
    for mass_model in protocol["panel"]["mass_models"]:
        out["exact_ok"][mass_model] = all(
            exact.get(case) == "pass"
            for case, model in EXACT_CASES.items()
            if model == mass_model
        )
        if not out["exact_ok"][mass_model]:
            reasons.append(f"{mass_model} exact checks are absent or not all passed")
    expected = {
        (r, m, k, q)
        for r, m, k in tool.canonical_groups(protocol)
        for q in alias["quantities"]
    }
    groups = alias_report.get("groups")
    entries = {key: [] for key in expected}
    if not isinstance(groups, list):
        structural.append("freeze report's alias groups are not a list")
        groups = []
    for row in groups:
        if not isinstance(row, dict):
            structural.append("freeze report has a malformed alias group")
            continue
        key = tuple(
            row.get(k) for k in ("regime", "mass_model", "epoch_count", "quantity")
        )
        if (
            not all(isinstance(x, str) for x in (key[0], key[1], key[3]))
            or type(key[2]) is not int
            or key not in expected
        ):
            structural.append("freeze report has an unknown alias group")
            continue
        entries[key].append(row.get("power_lower_95"))
    for key, values in sorted(entries.items()):
        if len(values) != 1:
            reasons.append(f"alias group {key} is missing or duplicated")
        elif not valid_unit_interval(values[0]):
            reasons.append(f"alias group {key} has an invalid power probability")
        elif values[0] < alias["min_power"]:
            reasons.append(f"alias group {key} is below the power floor")
        else:
            out["power_ok"].add(key)
    reasons += structural
    out["usable"] = not structural
    out["valid"] = not reasons
    return out


def read_freeze_evidence(path, repo):
    """Read only a tracked regular file whose bytes and index agree with HEAD.

    Return the parsed report and Git identity, or no report and a reason. Git
    supplies the commit and blob hashes; this is a review binding, not a claim
    that the committed evidence was computed honestly.
    """
    identity = {"valid": False, "commit": None, "blob_hash": None, "path": None}
    try:
        if path is None:
            raise ValueError("freeze evidence file is required")
        repo = Path(repo).resolve()
        path = Path(path).absolute()
        relative = path.relative_to(repo).as_posix()
        # Resolving as well prevents a symlink from escaping the repository.
        path.resolve().relative_to(repo)
        identity["path"] = relative

        def git(*args):
            return subprocess.check_output(
                ["git", "--literal-pathspecs", "-C", str(repo), *args],
                stderr=subprocess.PIPE,
            )

        if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
            raise ValueError("freeze evidence must belong to the photomancy repository")
        commit = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        identity["commit"] = commit
        git("ls-files", "--error-unmatch", "--", relative)
        tree = git("ls-tree", commit, "--", relative).decode().split("\t", 1)[0].split()
        if len(tree) != 3 or tree[0] not in ("100644", "100755") or tree[1] != "blob":
            raise ValueError("freeze evidence must be a committed regular file")
        identity["blob_hash"] = tree[2]
        git(
            "diff",
            "--cached",
            "--quiet",
            "--no-ext-diff",
            "--no-textconv",
            commit,
            "--",
            relative,
        )
        git("diff", "--quiet", "--no-ext-diff", "--no-textconv", commit, "--", relative)
        if path.is_symlink():
            raise ValueError("freeze evidence must be a regular file, not a symlink")
        committed = git("cat-file", "blob", tree[2])
        if path.read_bytes() != committed:
            raise ValueError("freeze evidence bytes differ from HEAD")
        report = json.loads(committed, object_pairs_hook=_unique_json_object)
        if not isinstance(report, dict):
            raise ValueError("freeze evidence is not a JSON object")
    except (OSError, ValueError, subprocess.CalledProcessError) as err:
        identity["reason"] = (
            "freeze evidence must be tracked and unmodified relative to HEAD "
            f"in the photomancy repository ({err})"
        )
        return None, identity
    identity["valid"] = True
    return report, identity


def _unique_json_object(pairs):
    """Reject repeated JSON keys rather than silently keeping the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def mechanics_verdict(evidence, code_commit):
    """Verdict of the mechanics criterion from an attestation, with a reason.

    The attestation is a dict with ``result`` (``pass`` or ``fail``), the
    ``code_commit`` the tests ran at and the ``log_sha256`` of the test log
    (64 hex digits). It counts only when its commit equals the commit the
    panel's manifests record. Anything else is ``incomplete``.
    """
    if not isinstance(evidence, dict):
        return "incomplete", "no mechanics attestation"
    if evidence.get("result") not in ("pass", "fail"):
        return "incomplete", "mechanics attestation has no pass or fail result"
    digest = evidence.get("log_sha256")
    if not isinstance(digest, str) or not re.match(r"^[0-9a-f]{64}$", digest):
        return "incomplete", "mechanics attestation has no test-log hash"
    if not code_commit or evidence.get("code_commit") != code_commit:
        return "incomplete", "mechanics attestation is for another code commit"
    return evidence["result"], ""


def read_mechanics_attestation(path):
    """Load an attestation file; a ``log_path`` entry must match ``log_sha256``."""
    import hashlib

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    log = data.get("log_path")
    if log is not None:
        try:
            digest = hashlib.sha256((path.parent / log).read_bytes()).hexdigest()
        except OSError:
            return {}
        if digest != data.get("log_sha256"):
            return {}
    return data


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
    tool,
    protocol,
    units,
    commits=None,
    stage="panel",
    mechanics=None,
    memory_mb=None,
    latency_ms=None,
    freeze=None,
):
    """The verdict rows of a set of unit records.

    Args:
        tool: The loaded ``updater_panel`` module.
        protocol: The parameter block.
        units: Map of unit id to record (current run hash, protocol budgets).
            Records with missing or invalid contents are not scored; the
            criteria that read them are ``incomplete``.
        commits: ``protocol_commit`` and ``code_commit`` for every row.
        stage: Stage of the unit ids.
        mechanics: ``"pass"`` or ``"fail"`` (``mechanics_verdict``); any other
            value leaves the criterion ``incomplete``.
        memory_mb: Measured peak memory of the intended batch, written on the
            failed-update rows when given.
        latency_ms: Measured warm batched amortized cost per active update for
            the intended batch, written on the failed-update rows when given.
        freeze: ``freeze_evidence`` output. Scoped certification, exact-check
            or power deficiencies make affected aliases ``incomplete``, even
            numerical failures. Unbound evidence cannot support a pass.

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
    units, _ = valid_units(protocol, units)
    rows = []

    def new(*args):
        row = _blank_row(*args, seeds, commits)
        row["verdict_mechanics"] = mech
        return row

    def fill(row, stats):
        for key, value in stats.items():
            row[key] = value

    def alias(group, row, k, q):
        verdict, stats = alias_statistics(group, k, q, protocol, crit)
        key = (group.regime, group.mass_model, k, q)
        if freeze is not None and freeze.get("usable"):
            eligible = (
                freeze["certified"].get(group.mass_model)
                and freeze["exact_ok"].get(group.mass_model)
                and key in freeze["power_ok"]
            )
            if not eligible:
                verdict = "incomplete"
        elif verdict == "pass":
            verdict = "incomplete"
        row["verdict_alias"] = verdict
        return stats

    blocks = [
        (r["id"], m, _Group(tool, units, stage, m, r["id"], n))
        for r in protocol["regimes"]
        for m in panel["mass_models"]
    ]
    for regime, mass_model, group in blocks:
        for k in panel["epoch_counts"]:
            for q in panel["quantities"]:
                row = new(regime, mass_model, k, q)
                verdicts, stats = cell_statistics(group, k, q, crit)
                row["verdict_coverage"] = verdicts["coverage"]
                row["verdict_calibration"] = verdicts["calibration"]
                fill(row, stats)
                if q in alias_q:
                    stats = alias(group, row, k, q)
                    stats.pop("worlds", None)
                    fill(row, stats)
                rows.append(row)
    for regime, mass_model, group in blocks:
        for k in panel["epoch_counts"]:
            row = new(regime, mass_model, k, "future_pa")
            fill(row, alias(group, row, k, "future_pa"))
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
            if latency_ms is not None:
                row["latency_ms_per_update"] = latency_ms
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


def _positive(value, name):
    if value is None:
        return None
    if _number(value) is None or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def grade_panel(
    tool,
    protocol,
    path,
    stage="panel",
    require_frozen=True,
    allow_dirty=False,
    mechanics=None,
    memory_mb=None,
    latency_ms=None,
    freeze_evidence_path=None,
):
    """Audit a panel directory and grade it.

    Args:
        tool: The loaded ``updater_panel`` module.
        protocol: The parameter block.
        path: Panel run directory.
        stage: Stage of the units.
        require_frozen: Refuse a protocol whose status is not ``frozen``.
        mechanics: Attestation dict for ``mechanics_verdict``.
        memory_mb: Measured peak memory of the intended batch.
        latency_ms: Measured warm batched amortized ms per active update.
        freeze_evidence_path: Tracked ``--freeze`` JSON, unmodified from HEAD.
        allow_dirty: Grade a panel whose manifests flag a dirty tree. The
            result is exploratory: every ``pass`` is written as ``incomplete``
            so the table cannot satisfy a gate, and ``exploratory`` lists why.

    Returns:
        Dict with ``status`` (``refused``, ``incomplete``, ``fail`` or
        ``pass``), ``reasons`` (why a panel that could pass does not), the
        ``audit`` (without records) and, unless refused, the ``rows`` and
        per-criterion ``counts``. A refused directory has no rows.
    """
    memory_mb = _positive(memory_mb, "memory_mb")
    latency_ms = _positive(latency_ms, "latency_ms")
    audit = audit_panel(tool, protocol, path, stage, require_frozen, allow_dirty)
    if audit["status"] == "refused":
        return {
            "status": "refused",
            "reasons": audit["refusals"],
            "audit": audit_summary(audit),
            "rows": None,
        }
    prov = audit["provenance"]
    commit = prov["code_commit"]
    mech, mech_reason = mechanics_verdict(mechanics, commit)
    freeze_report, identity = read_freeze_evidence(freeze_evidence_path, tool.REPO)
    freeze = freeze_evidence(tool, protocol, freeze_report)
    rows = build_rows(
        tool,
        protocol,
        audit["usable"],
        {
            "protocol_commit": prov["protocol_commit"],
            "code_commit": commit + ("-dirty" if prov["code_dirty"] else ""),
        },
        stage,
        mech,
        memory_mb,
        latency_ms,
        freeze,
    )
    reasons = []
    if not identity["valid"]:
        reasons.append(identity["reason"])
    if mech_reason:
        reasons.append(mech_reason)
    if not freeze["valid"]:
        reasons += freeze["reasons"]
    exploratory = list(prov["notes"])
    if exploratory:
        reasons += exploratory
        for row in rows:
            for c in CRITERIA:
                if row[f"verdict_{c}"] == "pass":
                    row[f"verdict_{c}"] = "incomplete"
    counts = verdict_counts(rows)
    verdict = panel_verdict(rows)
    if (audit["status"] == "incomplete" or not freeze["valid"]) and verdict == "pass":
        verdict = "incomplete"
    return {
        "status": verdict,
        "reasons": reasons,
        "exploratory": bool(exploratory),
        "freeze_evidence": identity,
        "audit": audit_summary(audit),
        "rows": rows,
        "counts": counts,
    }


def write_verdict(tool, result, out_path):
    """Write the table of a graded result atomically."""
    tool.atomic_write_text(out_path, rows_to_csv(result["rows"]))


def report_text(result):
    """Printable report: the status, the audit and the verdict counts."""
    body = {k: v for k, v in result.items() if k != "rows"}
    return json.dumps(body, indent=2, sort_keys=True, default=str)
