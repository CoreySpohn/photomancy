"""Tests for grading an updater panel into the verdict table.

The records are synthetic: a panel of ``N`` worlds per block whose summaries
are built to pass every criterion, then damaged one criterion at a time. Only
the 24-world smoke test runs the real updater.
"""

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).parents[1] / "tools"
N = 12
BINS = [1.0 / 8] * 8


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    """The panel tool module."""
    return _load("updater_panel")


@pytest.fixture(scope="module")
def verdict():
    """The grading module."""
    return _load("updater_verdict")


@pytest.fixture(scope="module")
def protocol(tool):
    """The protocol, frozen, at N worlds per cell."""
    p = copy.deepcopy(tool.load_protocol())
    p["status"] = "frozen"
    p["panel"]["worlds_per_cell"] = N
    # The count of 20 of 400 worlds, scaled to N worlds.
    p["criteria"]["schedule_stability"]["max_count"] = 2
    return p


def _updater(tool, protocol, unit, world_fn):
    """A synthetic updater record at the protocol budget."""
    budgets = tool.protocol_budgets(protocol)
    doubled = 2 if unit["role"] == "updater_doubled" else 1
    k_summaries, truths = {}, {}
    for k in (2, 3):
        spec = world_fn(unit, k)
        k_summaries[str(k)] = spec["summary"]
        truths[str(k)] = spec["truth"]
    return {
        "kind": "updater",
        "particles": budgets["particles"] * doubled,
        "updates": [
            {
                "update": u,
                "status": world_fn(unit, 0)["status"][u],
                "seconds": 0.5,
                "includes_compile": False,
            }
            for u in range(3)
        ],
        "summaries": k_summaries,
        "truth": truths,
    }


def make_units(tool, protocol, edit=None):
    """Synthetic panel records; ``edit(unit, k, spec)`` damages a record."""
    quantities = protocol["panel"]["quantities"]
    budgets = tool.protocol_budgets(protocol)
    run_hash = tool.run_hash(protocol)

    def world_fn(unit, k):
        w = unit["world"]
        summary = {
            "intervals": {
                q: {"90": [-1.0, 1.0], "50": [-1.0, 1.0] if w % 2 else [2.0, 3.0]}
                for q in quantities
            },
            "pit": {q: (w + 0.5) / N for q in quantities},
            "bins": {q: list(BINS) for q in ("lnP", "e", "future_pa")},
            "schedule_p": 0.5,
        }
        spec = {
            "summary": summary,
            "truth": dict.fromkeys(quantities, 0.0),
            "status": ["OK"] * 3,
        }
        if edit is not None:
            edit(unit, k, spec)
        return spec

    out = {}
    for unit in tool.plan_units(protocol, "panel", list(tool.PARTS)):
        if unit["role"] in tool.ALL_UPDATER_ROLES:
            rec = _updater(tool, protocol, unit, world_fn)
        else:
            k = unit["epoch_count"]
            rec = {
                "kind": "reference",
                "proposals": budgets[unit["mass_model"]]
                * tool.reference_budget_factor(unit["mass_model"], unit["role"]),
                "summary": {
                    "kish_ess": 900.0,
                    "bins": {q: list(BINS) for q in ("lnP", "e", "future_pa")},
                },
            }
            if edit is not None:
                spec = world_fn(unit, k)
                if "reference_bins" in spec:
                    rec["summary"]["bins"] = spec["reference_bins"](unit)
        out[unit["id"]] = dict(rec, id=unit["id"], run_hash=run_hash)
    return out


@pytest.fixture(scope="module")
def clean(tool, protocol):
    """A panel that passes every criterion."""
    return make_units(tool, protocol)


def grade(tool, verdict, protocol, units, mechanics="pass"):
    """Verdict rows of in-memory records."""
    return verdict.build_rows(tool, protocol, units, {}, "panel", mechanics)


def by(rows, **match):
    """Rows matching every given column value."""
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in match.items())]


def verdicts(rows, criterion):
    """Distinct verdicts of one criterion."""
    return {r[f"verdict_{criterion}"] for r in rows}


# --- schema ----------------------------------------------------------------


def test_clean_panel_passes_with_the_protocol_schema(tool, verdict, protocol, clean):
    """A panel built to satisfy every criterion passes, in the stated schema.

    Fails if a column is missing or misordered, a row family has the wrong
    count (64 cells, 16 position-angle alias rows, 24 failed-update groups,
    16 schedule groups), a verdict is outside the four values, or a criterion
    that does not apply to a row is not ``na``.
    """
    rows = grade(tool, verdict, protocol, clean)
    assert len(rows) == 120
    assert len(by(rows, quantity="updates")) == 24
    assert len(by(rows, quantity="schedule")) == 16
    assert len(by(rows, quantity="future_pa")) == 16
    text = verdict.rows_to_csv(rows)
    table = list(csv.DictReader(io.StringIO(text)))
    assert (
        list(table[0])
        == (
            "regime mass_model epoch_count quantity worlds covered_count "
            "covered_50_count "
            "pit_sup_distance d2_mean d2_upper exhausted_count exhausted_cp_upper "
            "schedule_deviations_keys schedule_deviations_budget tv_median tv_p95 "
            "latency_ms_per_update memory_mb seeds verdict_coverage "
            "verdict_calibration "
            "verdict_alias verdict_failed_updates verdict_schedule verdict_mechanics "
            "protocol_commit code_commit"
        ).split()
    )
    for r in table:
        for c in verdict.CRITERIA:
            assert r[f"verdict_{c}"] in verdict.VERDICTS
    assert verdicts(rows, "coverage") == {"pass", "na"}
    assert verdict.panel_verdict(rows) == "pass"
    assert {r["verdict_mechanics"] for r in rows} == {"pass"}
    cell = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    assert cell["covered_count"] == N and cell["covered_50_count"] == N // 2
    assert cell["verdict_alias"] == "pass" and cell["verdict_failed_updates"] == "na"
    assert cell["seeds"] == "271828/panel"
    upd = by(rows, quantity="updates", epoch_count=1)[0]
    assert upd["exhausted_count"] == 0 and upd[
        "latency_ms_per_update"
    ] == pytest.approx(500.0)
    assert by(rows, quantity="future_east")[0]["verdict_alias"] == "na"


def test_mechanics_is_incomplete_unless_supplied(tool, verdict, protocol, clean):
    """Without the mechanics result no panel can pass.

    Fails if a missing or unrecognized mechanics input yields pass.
    """
    for value in (None, "green"):
        rows = grade(tool, verdict, protocol, clean, mechanics=value)
        assert verdict.panel_verdict(rows) == "incomplete"
    rows = grade(tool, verdict, protocol, clean, mechanics="fail")
    assert verdict.panel_verdict(rows) == "fail"


# --- each criterion fails on its own ----------------------------------------


def _is(unit, **kw):
    return all(unit.get(k) == v for k, v in kw.items())


def _undercover(unit, k, spec):
    """Truth outside the 90 percent interval in 8 worlds of one cell."""
    # One cell: ti_window R2, epoch 3, quantity e; the truth is outside in 8 worlds.
    if (
        _is(unit, role="updater_a", mass_model="ti_window", regime="R2")
        and k == 3
        and unit["world"] < 8
    ):
        spec["truth"]["e"] = 5.0


def _bad_pits(unit, k, spec):
    """PITs piled near one in one cell."""
    if _is(unit, role="updater_a", mass_model="physical_fixed", regime="R3") and k == 2:
        spec["summary"]["pit"]["lnP"] = 0.99


def _bad_alias(unit, k, spec):
    """Both keys shifted off the reference in one group."""
    if _is(unit, role="updater_a", regime="R4", mass_model="ti_window") or _is(
        unit, role="updater_b", regime="R4", mass_model="ti_window"
    ):
        spec["summary"]["bins"]["lnP"] = [0.225, 0.025] + [0.125] * 6


def _failed_updates(unit, k, spec):
    """Four worlds with a failed second update in one group."""
    if (
        _is(unit, role="updater_a", regime="R1", mass_model="physical_fixed")
        and unit["world"] < 4
    ):
        spec["status"][1] = "LOW_ESS"


def _unstable(unit, k, spec):
    """The doubled run disagrees in five worlds of one group."""
    if (
        _is(unit, role="updater_doubled", regime="R2", mass_model="physical_fixed")
        and unit["world"] < 5
    ):
        spec["summary"]["schedule_p"] = 0.9


SCENARIOS = {
    "coverage": (
        _undercover,
        {"regime": "R2", "mass_model": "ti_window", "epoch_count": 3, "quantity": "e"},
    ),
    "calibration": (
        _bad_pits,
        {
            "regime": "R3",
            "mass_model": "physical_fixed",
            "epoch_count": 2,
            "quantity": "lnP",
        },
    ),
    "alias": (
        _bad_alias,
        {"regime": "R4", "mass_model": "ti_window", "quantity": "lnP"},
    ),
    "failed_updates": (
        _failed_updates,
        {
            "regime": "R1",
            "mass_model": "physical_fixed",
            "epoch_count": 2,
            "quantity": "updates",
        },
    ),
    "schedule": (
        _unstable,
        {"regime": "R2", "mass_model": "physical_fixed", "quantity": "schedule"},
    ),
}


def check_each_criterion_fails_alone(tool, verdict, protocol):
    """Damage one block per criterion; only the damaged rows may fail."""
    for criterion, (edit, where) in SCENARIOS.items():
        rows = grade(tool, verdict, protocol, make_units(tool, protocol, edit))
        failed = [r for r in rows if r[f"verdict_{criterion}"] == "fail"]
        assert failed, f"{criterion}: nothing failed"
        assert all(
            all(str(r[k]) == str(v) for k, v in where.items()) for r in failed
        ), criterion
        others = [c for c in verdict.CRITERIA if c != criterion]
        assert all("fail" not in verdicts(rows, c) for c in others), criterion
        assert verdict.panel_verdict(rows) == "fail"


def test_each_criterion_fails_on_its_own_and_only_in_its_cell(tool, verdict, protocol):
    """A criterion that fails in one cell fails that row and no other.

    Fails if a criterion is not graded (its damaged cell passes), the verdict
    is pooled over cells (the damaged cell is hidden by the passing ones), or
    a failure spills into another criterion or cell. The 8 of 12 covered
    worlds fall below the critical count 6 for 12 worlds only when the cell
    is graded alone.
    """
    check_each_criterion_fails_alone(tool, verdict, protocol)


def test_alias_margin_comes_from_the_standard_error(tool, verdict, protocol):
    """A mean D2 under the tolerance fails when its upper bound exceeds it.

    A quarter of the worlds carry D2 of 1.8e-3, so the mean (4.5e-4) is under
    5e-4 and the mean plus 3.08 standard errors is over it. Fails if the
    bound ignores the standard error (z = 0), or doubles or halves it.
    """

    def edit(unit, k, spec):
        if (
            _is(unit, regime="R1", mass_model="ti_window")
            and unit["role"] in ("updater_a", "updater_b")
            and unit["world"] % 4 == 0
        ):
            spec["summary"]["bins"]["e"] = [0.125 + 0.03, 0.125 - 0.03] + [0.125] * 6

    rows = grade(tool, verdict, protocol, make_units(tool, protocol, edit))
    row = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    assert (
        row["d2_mean"]
        < tool.load_protocol()["criteria"]["alias_accuracy"]["delta2"]
        < row["d2_upper"]
    )
    assert row["verdict_alias"] == "fail"
    assert row["tv_p95"] >= row["tv_median"] >= 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_values_never_pass(tool, verdict, protocol, bad):
    """NaN or inf in any graded input is a fail or incomplete, never a pass.

    Fails if a NaN interval, PIT, schedule probability or bin vector slips
    through a comparison that is false for NaN.
    """

    def edit(unit, k, spec):
        if (
            _is(unit, role="updater_a", regime="R1", mass_model="ti_window")
            and k == 2
            and unit["world"] == 0
        ):
            spec["truth"]["e"] = bad
            spec["summary"]["pit"]["lnP"] = bad
            spec["summary"]["schedule_p"] = bad
            spec["summary"]["bins"]["future_pa"] = [bad] + [0.0] * 7

    rows = grade(tool, verdict, protocol, make_units(tool, protocol, edit))
    assert (
        by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="lnP")[0][
            "verdict_calibration"
        ]
        == "fail"
    )
    assert (
        by(
            rows,
            regime="R1",
            mass_model="ti_window",
            epoch_count=2,
            quantity="future_pa",
        )[0]["verdict_alias"]
        == "incomplete"
    )
    sched = by(
        rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="schedule"
    )[0]
    assert sched["schedule_deviations_keys"] == 1
    assert verdict.panel_verdict(rows) != "pass"


def test_thresholds_match_the_protocol_values(tool, verdict):
    """The grader's thresholds equal the protocol's stated values at 400 worlds.

    Fails if a formula drifts from the dry run's derived values (339 covered,
    DKW 0.09904, z 3.078, 17 failed updates).
    """
    protocol = tool.load_protocol()
    got = verdict.thresholds(tool, protocol, 400)
    stated = protocol["stated"]
    assert got["coverage_critical_count"] == stated["coverage_critical_count"] == 339
    assert round(got["dkw_epsilon"], 5) == stated["dkw_epsilon_5dp"]
    assert round(got["alias_z"], 3) == stated["alias_z_3dp"]
    assert (
        got["failed_updates_critical_count"]
        == stated["failed_updates_critical_count"]
        == 17
    )
    assert verdict.thresholds(tool, protocol, 3)["coverage_critical_count"] == -1


def test_decisions_at_their_boundaries(verdict):
    """Each decision flips exactly at its stated boundary, and NaN never passes.

    Fails if a boundary moves by one (fail at the critical count or fewer for
    coverage, strictly above for the others).
    """
    assert (
        verdict.decide_coverage(6, 6) == "fail"
        and verdict.decide_coverage(7, 6) == "pass"
    )
    assert (
        verdict.decide_failed_updates(17, 17) == "pass"
        and verdict.decide_failed_updates(18, 17) == "fail"
    )
    assert (
        verdict.decide_schedule(20, 20, 20) == "pass"
        and verdict.decide_schedule(20, 21, 20) == "fail"
    )
    assert (
        verdict.decide_alias(5e-4, 5e-4) == "pass"
        and verdict.decide_alias(5.01e-4, 5e-4) == "fail"
    )
    assert (
        verdict.decide_calibration(0.2, 0.2) == "pass"
        and verdict.decide_calibration(0.21, 0.2) == "fail"
    )
    nan = float("nan")
    assert (
        verdict.decide_alias(nan, 5e-4) == "fail"
        and verdict.decide_calibration(nan, 0.2) == "fail"
    )


def test_pit_sup_distance_is_the_ecdf_distance(verdict):
    """The sup-distance of PITs matches hand values.

    Fails if the distance uses only one side of each step: a uniform grid of
    n midpoints gives 1 / (2n), and n copies of 0.5 give 0.5.
    """
    assert verdict.pit_sup_distance(
        [(i + 0.5) / 10 for i in range(10)]
    ) == pytest.approx(0.05)
    assert verdict.pit_sup_distance([0.5] * 10) == pytest.approx(0.5)
    assert verdict.pit_sup_distance([0.0, 0.0]) == pytest.approx(1.0)


def test_failed_update_rate_bound_is_reported(tool, verdict, protocol):
    """The failed-update row carries the count and its Clopper-Pearson bound.

    Fails if the count is taken from another update, or the bound is not the
    protocol's one-sided 95 percent value.
    """
    rows = grade(tool, verdict, protocol, make_units(tool, protocol, _failed_updates))
    row = by(
        rows,
        regime="R1",
        mass_model="physical_fixed",
        epoch_count=2,
        quantity="updates",
    )[0]
    assert row["exhausted_count"] == 4
    assert row["exhausted_cp_upper"] == pytest.approx(
        tool.clopper_pearson_upper(4, N, 0.95)
    )
    other = by(
        rows,
        regime="R1",
        mass_model="physical_fixed",
        epoch_count=1,
        quantity="updates",
    )[0]
    assert other["exhausted_count"] == 0


# --- incomplete panels -------------------------------------------------------


def test_missing_worlds_leave_only_their_groups_incomplete(
    tool, verdict, protocol, clean
):
    """A missing unit makes the criteria that read it incomplete, never pass.

    Fails if the group is graded on the remaining worlds, if an unrelated
    group changes, or if the panel verdict is pass or fail.
    """
    victim = "panel/ti_window/R1/w003/updater_b"
    units = {i: r for i, r in clean.items() if i != victim}
    rows = grade(tool, verdict, protocol, units)
    hit = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    assert hit["verdict_alias"] == "incomplete" and hit["verdict_coverage"] == "pass"
    assert hit["covered_count"] == N
    sched = by(rows, regime="R1", mass_model="ti_window", quantity="schedule")
    assert verdicts(sched, "schedule") == {"incomplete"} and sched[0]["worlds"] == N - 1
    assert sched[0]["schedule_deviations_keys"] == ""
    assert all(
        r["verdict_schedule"] == "pass"
        for r in by(rows, regime="R2", quantity="schedule")
    )
    assert verdict.panel_verdict(rows) == "incomplete"


def test_failure_outranks_incomplete(tool, verdict, protocol, clean):
    """A failed group fails the panel even while another group is incomplete.

    Fails if incomplete masks a fail (the protocol: fail if any group fails).
    """
    units = make_units(tool, protocol, _undercover)
    units = {i: r for i, r in units.items() if i != "panel/ti_window/R1/w003/updater_b"}
    assert verdict.panel_verdict(grade(tool, verdict, protocol, units)) == "fail"
    assert verdict.panel_verdict([]) == "incomplete"


def test_hollow_reference_makes_the_alias_group_incomplete(tool, verdict, protocol):
    """A reference fit with no weighted draw leaves its group unscorable.

    Fails if a world with an empty reference is dropped or scored as zeros.
    """
    units = make_units(tool, protocol)
    unit = "panel/physical_fixed/R3/w002/reference_1/k3"
    units[unit] = dict(units[unit], summary={"kish_ess": 0.0, "particles": 0})
    rows = grade(tool, verdict, protocol, units)
    hit = by(
        rows, regime="R3", mass_model="physical_fixed", epoch_count=3, quantity="lnP"
    )[0]
    assert hit["verdict_alias"] == "incomplete" and hit["d2_mean"] == ""


def test_alias_bound_above_its_cap_is_incomplete(tool, verdict, protocol, clean):
    """A delta2 above delta2_max leaves every alias verdict incomplete.

    Fails if the grader applies an out-of-bounds tolerance.
    """
    loose = copy.deepcopy(protocol)
    loose["criteria"]["alias_accuracy"]["delta2"] = 2e-3
    rows = grade(tool, verdict, loose, clean)
    assert verdicts(by(rows, quantity="future_pa"), "alias") == {"incomplete"}


# --- directory audit ---------------------------------------------------------


def write_panel(tool, protocol, units, path, shards=2, edit_manifest=None):
    """Write records into ``units-{i}of{n}.jsonl`` and matching manifests."""
    path.mkdir(parents=True, exist_ok=True)
    ordered = [
        units[u["id"]]
        for u in tool.plan_units(protocol, "panel", list(tool.PARTS))
        if u["id"] in units
    ]
    for i in range(shards):
        with open(path / f"units-{i}of{shards}.jsonl", "w") as fh:
            for rec in ordered[i::shards]:
                fh.write(json.dumps(rec) + "\n")
        entry = {
            "started": f"2026-11-0{i + 1}T00:00:00+00:00",
            "finished": f"2026-11-0{i + 1}T01:00:00+00:00",
            "protocol_commit": "p" * 40,
            "protocol_status": protocol["status"],
            "protocol_hash": tool.declaration_hash(protocol),
            "run_hash": tool.run_hash(protocol),
            "code_commit": "c" * 40,
            "code_dirty": False,
            "jax": "0.9.0",
            "jaxlib": "0.9.0",
            "platform": "Linux-6.8-x86_64-with-glibc2.39",
            "argv": ["--run", "panel", "--shard", f"{i}/{shards}", "--chunk", "65536"],
            "protocol_budgets": True,
            "budgets": tool.protocol_budgets(protocol),
        }
        if edit_manifest:
            edit_manifest(i, entry)
        (path / f"manifest-{i}of{shards}.json").write_text(
            json.dumps({"invocations": [entry]})
        )


@pytest.fixture
def panel_dir(tool, protocol, clean, tmp_path):
    """A complete panel directory of two shards."""
    write_panel(tool, protocol, clean, tmp_path)
    return tmp_path


def audit(tool, verdict, protocol, path, **kw):
    """Audit a directory."""
    return verdict.audit_panel(tool, protocol, path, **kw)


def test_complete_panel_directory_grades_and_records_commits(
    tool, verdict, protocol, panel_dir, tmp_path
):
    """A complete consistent directory grades to pass and stamps the commits.

    Fails if the audit refuses a good directory, or the table does not carry
    the manifests' protocol and code commits.
    """
    result = verdict.grade_panel(
        tool, protocol, panel_dir, mechanics="pass", memory_mb=1234.5
    )
    assert result["status"] == "pass", result["audit"]
    assert result["audit"]["present"] == result["audit"]["planned"] == 8 * N * 7
    assert {r["code_commit"] for r in result["rows"]} == {"c" * 40}
    assert {r["protocol_commit"] for r in result["rows"]} == {"p" * 40}
    assert {r["memory_mb"] for r in by(result["rows"], quantity="updates")} == {1234.5}
    out = tmp_path / "out" / "verdict.csv"
    out.parent.mkdir()
    verdict.write_verdict(tool, result, out)
    with out.open() as fh:
        assert len(list(csv.DictReader(fh))) == 120


def test_missing_unit_is_reported_and_graded_incomplete(
    tool, verdict, protocol, clean, tmp_path
):
    """A directory missing a unit reports its id and grades incomplete.

    Fails if the audit does not name the missing unit, or the status is
    complete, or the graded panel is a pass.
    """
    victim = "panel/physical_fixed/R4/w011/reference_2/k2"
    write_panel(
        tool, protocol, {i: r for i, r in clean.items() if i != victim}, tmp_path
    )
    a = audit(tool, verdict, protocol, tmp_path)
    assert (
        a["status"] == "incomplete" and a["missing"] == [victim] and not a["refusals"]
    )
    result = verdict.grade_panel(tool, protocol, tmp_path, mechanics="pass")
    assert result["status"] == "incomplete" and result["rows"] is not None
    hit = by(
        result["rows"],
        regime="R4",
        mass_model="physical_fixed",
        epoch_count=2,
        quantity="lnP",
    )[0]
    assert hit["verdict_alias"] == "incomplete"


def test_completeness_check_that_skips_a_unit_is_caught(
    tool, verdict, protocol, clean, tmp_path, monkeypatch
):
    """Mutant: an audit whose plan drops one unit no longer reports it missing.

    The assertions of the missing-unit test fail under the mutant.
    """
    victim = "panel/physical_fixed/R4/w011/reference_2/k2"
    write_panel(
        tool, protocol, {i: r for i, r in clean.items() if i != victim}, tmp_path
    )
    real = tool.plan_units
    monkeypatch.setattr(
        tool,
        "plan_units",
        lambda *a, **k: [u for u in real(*a, **k) if u["id"] != victim],
    )
    a = audit(tool, verdict, protocol, tmp_path)
    assert a["missing"] != [victim]
    assert a["status"] == "complete"


@pytest.mark.parametrize(
    "case",
    ["duplicate", "unknown", "stale", "off_budget", "malformed"],
)
def test_inconsistent_records_refuse_to_grade(
    tool, verdict, protocol, clean, tmp_path, case
):
    """A duplicate, unknown, stale, off-budget or id-less record refuses grading.

    Fails if the grader writes a table from such a directory, or the refusal
    does not say why. Stale and off-budget records must not be silently
    treated as missing units.
    """
    units = dict(clean)
    victim = "panel/ti_window/R1/w000/updater_a"
    write_panel(tool, protocol, units, tmp_path)
    extra = None
    if case == "duplicate":
        extra = units[victim]
    elif case == "unknown":
        extra = dict(units[victim], id="panel/ti_window/R1/w999/updater_a")
    elif case == "stale":
        for f in tmp_path.glob("units-*.jsonl"):
            rows = [json.loads(line) for line in f.read_text().splitlines()]
            for r in rows:
                if r["id"] == victim:
                    r["run_hash"] = "0" * 64
            f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif case == "off_budget":
        for f in tmp_path.glob("units-*.jsonl"):
            rows = [json.loads(line) for line in f.read_text().splitlines()]
            for r in rows:
                if r["id"] == victim:
                    r["particles"] = 128
            f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif case == "malformed":
        extra = {"kind": "updater"}
    if extra is not None:
        with open(tmp_path / "units-0of2.jsonl", "a") as fh:
            fh.write(json.dumps(extra) + "\n")
    result = verdict.grade_panel(tool, protocol, tmp_path, mechanics="pass")
    assert result["status"] == "refused" and result["rows"] is None
    needle = {
        "duplicate": "duplicate",
        "unknown": "not panel",
        "stale": "run hashes",
        "off_budget": "non-protocol budgets",
        "malformed": "no string id",
    }[case]
    assert any(needle in reason for reason in result["audit"]["refusals"])


@pytest.mark.parametrize(
    ("field", "edit", "needle"),
    [
        ("jax", lambda i, e: e.update(jax="0.9.1") if i == 1 else None, "jax versions"),
        (
            "arch",
            lambda i, e: (
                e.update(platform="macOS-15-arm64-arm-64bit") if i == 1 else None
            ),
            "host architectures",
        ),
        (
            "chunk",
            lambda i, e: (
                e.update(argv=["--run", "panel", "--chunk=1024"]) if i == 1 else None
            ),
            "recorded chunks",
        ),
        (
            "code",
            lambda i, e: e.update(code_commit="d" * 40) if i == 1 else None,
            "code commits",
        ),
        ("dirty", lambda i, e: e.update(code_dirty=True) if i == 1 else None, "dirty"),
        (
            "hash",
            lambda i, e: e.update(run_hash="1" * 64) if i == 1 else None,
            "run hashes",
        ),
        (
            "protocol",
            lambda i, e: e.update(protocol_hash="2" * 64) if i == 1 else None,
            "protocol hashes",
        ),
        (
            "budgets",
            lambda i, e: e.update(protocol_budgets=False) if i == 1 else None,
            "non-protocol budgets",
        ),
        ("draft", lambda i, e: e.update(protocol_status="draft"), "statuses"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_manifests_that_disagree_refuse_to_grade(
    tool, verdict, protocol, clean, tmp_path, field, edit, needle
):
    """Manifest disagreement or a flag refuses grading and names the field.

    Fails if shards run under different JAX versions, architectures, chunks or
    commits, a dirty tree, another run or protocol hash, a test-budget
    invocation or a draft protocol is graded. The aarch64 and arm64 spellings
    are one architecture.
    """
    write_panel(tool, protocol, clean, tmp_path, edit_manifest=edit)
    result = verdict.grade_panel(tool, protocol, tmp_path, mechanics="pass")
    assert result["status"] == "refused"
    assert any(needle in reason for reason in result["audit"]["refusals"]), result[
        "audit"
    ]["refusals"]


def test_dirty_flag_can_be_accepted_and_is_marked(
    tool, verdict, protocol, clean, tmp_path
):
    """An accepted dirty tree is marked in the code commit column.

    Fails if the override is ignored, or the table hides the dirty flag.
    """
    write_panel(
        tool,
        protocol,
        clean,
        tmp_path,
        edit_manifest=lambda i, e: e.update(code_dirty=True),
    )
    assert (
        verdict.grade_panel(tool, protocol, tmp_path, mechanics="pass")["status"]
        == "refused"
    )
    result = verdict.grade_panel(
        tool, protocol, tmp_path, allow_dirty=True, mechanics="pass"
    )
    assert result["status"] == "pass"
    assert {r["code_commit"] for r in result["rows"]} == {"c" * 40 + "-dirty"}


def test_equivalent_architecture_spellings_agree(
    tool, verdict, protocol, clean, tmp_path
):
    """arm64 and aarch64 are one architecture; mixed spellings do not refuse.

    Fails if the architecture comparison is a string comparison.
    """
    plat = {0: "macOS-15-arm64-arm-64bit", 1: "Linux-6.8-aarch64-with-glibc2.39"}
    write_panel(
        tool,
        protocol,
        clean,
        tmp_path,
        edit_manifest=lambda i, e: e.update(platform=plat[i]),
    )
    assert verdict.audit_panel(tool, protocol, tmp_path)["status"] == "complete"


def test_draft_protocol_and_missing_manifests_refuse(
    tool, verdict, protocol, clean, tmp_path
):
    """A draft protocol, or a directory with no manifest, is not graded.

    Fails if either is graded.
    """
    write_panel(tool, protocol, clean, tmp_path)
    draft = copy.deepcopy(protocol)
    draft["status"] = "draft"
    a = verdict.audit_panel(tool, draft, tmp_path)
    assert a["status"] == "refused"
    for f in tmp_path.glob("manifest-*.json"):
        f.unlink()
    a = verdict.audit_panel(tool, protocol, tmp_path)
    assert a["status"] == "refused" and any("no manifest" in r for r in a["refusals"])


def test_skipped_criterion_is_caught(tool, verdict, protocol, monkeypatch):
    """Mutants: a grader that ignores any one criterion fails the criterion test.

    Each decision replaced by an always-pass (or, for stale statistics, a
    constant) is detected by the same check the real grader passes.
    """
    for name in (
        "decide_coverage",
        "decide_calibration",
        "decide_alias",
        "decide_failed_updates",
        "decide_schedule",
    ):
        with monkeypatch.context() as m:
            m.setattr(verdict, name, lambda *a, **k: "pass")
            with pytest.raises(AssertionError):
                check_each_criterion_fails_alone(tool, verdict, protocol)


def test_pooled_or_shifted_statistics_are_caught(tool, verdict, protocol, monkeypatch):
    """Mutants: a doubled-variance guard, a zero z and a shifted threshold are caught.

    Fails if the alias SE test and the coverage threshold would survive a
    stale value. The alias mutant sets the upper bound to the mean.
    """
    real = verdict.thresholds

    def shifted(tool_, protocol_, n):
        out = real(tool_, protocol_, n)
        out["coverage_critical_count"] = 0
        return out

    def no_se(tool_, protocol_, n):
        out = real(tool_, protocol_, n)
        out["alias_z"] = 0.0
        return out

    with monkeypatch.context() as m:
        m.setattr(verdict, "thresholds", shifted)
        with pytest.raises(AssertionError):
            check_each_criterion_fails_alone(tool, verdict, protocol)
    with monkeypatch.context() as m:
        m.setattr(verdict, "thresholds", no_se)
        with pytest.raises(AssertionError):
            test_alias_margin_comes_from_the_standard_error(tool, verdict, protocol)


def test_cancelling_strata_do_not_hide_a_failure(tool, verdict, protocol):
    """Over-coverage in one regime does not offset under-coverage in another.

    Fails if coverage or failed updates are pooled across blocks: R1 always
    covers and R2 undercovers, and the R2 row must still fail.
    """
    rows = grade(tool, verdict, protocol, make_units(tool, protocol, _undercover))
    assert verdicts(by(rows, regime="R1"), "coverage") == {"pass", "na"}
    assert (
        by(rows, regime="R2", mass_model="ti_window", epoch_count=3, quantity="e")[0][
            "verdict_coverage"
        ]
        == "fail"
    )


# --- command line -------------------------------------------------------------


def test_status_counts_the_requested_stage(tool, protocol, clean, tmp_path):
    """``progress`` counts panel units when given the panel stage.

    Fails if the stage is ignored (the development plan would show 0 of 304
    units and the panel plan 0 of 8 * N units).
    """
    write_panel(tool, protocol, clean, tmp_path)
    lines = tool.progress(protocol, tmp_path, "panel")
    assert lines[0] == f"updater: {8 * N * 3} of {8 * N * 3} units"
    assert lines[2] == f"reference: {8 * N * 4} of {8 * N * 4} units"
    assert lines[1] == "precision: 0 of 0 units"
    dev = tool.progress(protocol, tmp_path, "development")
    assert dev[0].startswith("updater: 0 of")


def test_command_line_grades_and_exits_by_verdict(
    tool, protocol, clean, tmp_path, capsys, monkeypatch
):
    """``--check-panel`` and ``--grade`` run through ``main`` with exit codes.

    Fails if the table is not written for a complete panel, or a refused
    directory writes one, or the exit code is not 0 for pass and 2 for
    refused.
    """
    proto_path = tmp_path / "protocol.md"
    text = tool.PROTOCOL.read_text()
    proto_path.write_text(text)
    monkeypatch.setattr(tool, "load_protocol", lambda path=None: protocol)
    run = tmp_path / "run"
    write_panel(tool, protocol, clean, run)
    out = tmp_path / "verdict.csv"
    code = tool.main(["--check-panel", str(run)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    code = tool.main(
        ["--grade", str(run), "--mechanics", "pass", "--verdict-out", str(out)]
    )
    assert code == 0 and out.exists()
    out.unlink()
    for f in run.glob("manifest-*.json"):
        f.unlink()
    code = tool.main(
        ["--grade", str(run), "--mechanics", "pass", "--verdict-out", str(out)]
    )
    assert code == 2 and not out.exists()


# --- smoke --------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_panel_runs_the_real_updater_and_grades(tool, verdict, tmp_path):
    """The 24-world smoke stage runs the real updater and grades end to end.

    Small budgets keep it fast (256 particles, 4,096 and 32,768 reference
    proposals), so the verdicts are not meaningful; the test checks that
    every planned unit is produced, the audit passes and the table has the
    protocol's 120 rows with a verdict in every graded column. Fails if the
    runner's records, the manifests and the grader disagree on any field.
    """
    import argparse

    protocol = copy.deepcopy(tool.load_protocol())
    protocol["status"] = "frozen"
    protocol["updater_budget"]["particles"] = 256
    protocol["reference"]["ti_window"]["proposals_per_fit"] = 4096
    protocol["reference"]["physical_fixed"]["proposals_per_fit"] = 2**15
    protocol["panel"]["worlds_per_cell"] = 3
    args = argparse.Namespace(
        particles=None,
        ti_proposals=None,
        physical_proposals=None,
        shard="0/1",
        run="smoke",
        parts="updater,reference",
        mass_models=None,
        regimes=None,
        worlds=None,
        roles=None,
        out=tmp_path,
        chunk=2**13,
    )
    tool.run_stage(protocol, args)
    # The runner writes a dirty-tree flag when the checkout has edits.
    result = verdict.grade_panel(
        tool, protocol, tmp_path, stage="smoke", allow_dirty=True, mechanics="pass"
    )
    assert result["audit"]["status"] != "refused", result["audit"]["refusals"]
    assert result["audit"]["missing"]["count"] == 0
    assert len(result["rows"]) == 120
    assert {r["worlds"] for r in result["rows"]} == {3}
    assert verdict.rows_to_csv(result["rows"]).count("\n") == 121
    assert result["status"] in {"pass", "fail", "incomplete"}
