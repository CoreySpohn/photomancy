"""Tests for grading an updater panel into the verdict table.

The records are synthetic: a panel of ``N`` worlds per block whose summaries
are built to pass every criterion, then damaged one criterion at a time. Only
the 24-world smoke test runs the real updater.
"""

import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).parents[1] / "tools"
N = 12
BINS = [1.0 / 8] * 8
ALIAS_QUANTITIES = ("lnP", "e", "future_pa")
CODE = "a" * 40
PROTOCOL_COMMIT = "b" * 40
LOG_HASH = "c" * 64


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


# --- synthetic records ---


def _updater(tool, protocol, unit, world_fn):
    """A synthetic updater record at the protocol budget."""
    budgets = tool.protocol_budgets(protocol)
    doubled = 2 if unit["role"] == "updater_doubled" else 1
    summaries, truths = {}, {}
    for k in (2, 3):
        spec = world_fn(unit, k)
        summaries[str(k)] = spec["summary"]
        truths[str(k)] = spec["truth"]
    statuses = world_fn(unit, 0)["status"]
    return {
        "kind": "updater",
        "particles": budgets["particles"] * doubled,
        "updates": [
            {
                "update": u,
                "status": statuses[u],
                "seconds": 0.5,
                "includes_compile": False,
            }
            for u in range(3)
        ],
        "summaries": summaries,
        "truth": truths,
    }


def make_units(tool, protocol, edit=None, bins=None, post=None):
    """Synthetic panel records.

    ``edit(unit, k, spec)`` damages an updater's summary, truth or statuses;
    ``bins(unit, k)`` returns a dict of alias-quantity bin masses for any
    record; ``post(unit, record)`` damages a finished record.
    """
    quantities = protocol["panel"]["quantities"]
    budgets = tool.protocol_budgets(protocol)
    run_hash = tool.run_hash(protocol)

    def bin_masses(unit, k):
        if bins is not None:
            return bins(unit, k)
        return {q: list(BINS) for q in ALIAS_QUANTITIES}

    def world_fn(unit, k):
        w = unit["world"]
        half = [-1.0, 1.0] if w % 2 else [0.5, 0.9]
        summary = {
            "particles": 1024,
            "kish_ess": 900.0,
            "intervals": {q: {"90": [-1.0, 1.0], "50": list(half)} for q in quantities},
            "pit": {q: (w + 0.5) / N for q in quantities},
            "bins": bin_masses(unit, k),
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
                    "particles": 1024,
                    "kish_ess": 900.0,
                    "bins": bin_masses(unit, k),
                },
            }
        rec = dict(rec, id=unit["id"], run_hash=run_hash)
        if post is not None:
            post(unit, rec)
        out[unit["id"]] = rec
    return out


def freeze_report(tool, verdict, protocol, mass_models=None):
    """A synthetic committed freeze report that satisfies every requirement."""
    alias = protocol["criteria"]["alias_accuracy"]
    groups = [
        {
            "regime": r,
            "mass_model": m,
            "epoch_count": k,
            "quantity": q,
            "power_lower_95": 0.9,
        }
        for r, m, k in tool.canonical_groups(protocol)
        for q in alias["quantities"]
    ]
    return {
        "status": "frozen",
        "reasons": [],
        "binding": verdict.freeze_binding(tool, protocol),
        "alias": {"delta2": alias["delta2"], "groups": groups},
        "certification": {
            m: {"status": "certified"}
            for m in (mass_models or protocol["panel"]["mass_models"])
        },
        "exact_checks": {
            "verdicts": dict.fromkeys(
                ("ti_window", "physical_fixed", "ti_binding"), "pass"
            )
        },
    }


@pytest.fixture(scope="module")
def clean(tool, protocol):
    """A panel that passes every criterion."""
    return make_units(tool, protocol)


@pytest.fixture(scope="module")
def evidence(tool, verdict, protocol):
    """Valid freeze evidence for the test protocol."""
    return verdict.freeze_evidence(
        tool, protocol, freeze_report(tool, verdict, protocol)
    )


def grade(tool, verdict, protocol, units, evidence, mechanics="pass", **kw):
    """Verdict rows of in-memory records."""
    return verdict.build_rows(
        tool, protocol, units, {}, "panel", mechanics, freeze=evidence, **kw
    )


def by(rows, **match):
    """Rows matching every given column value."""
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in match.items())]


def verdicts(rows, criterion):
    """Distinct verdicts of one criterion."""
    return {r[f"verdict_{criterion}"] for r in rows}


# --- schema ---


def test_clean_panel_passes_with_the_protocol_schema(
    tool, verdict, protocol, clean, evidence
):
    """A panel built to satisfy every criterion passes, in the stated schema.

    Fails if a column is missing or misordered, a row family has the wrong
    count (64 cells, 16 position-angle alias rows, 24 failed-update groups,
    16 schedule groups), a verdict is outside the four values, or a criterion
    that does not apply to a row is not ``na``.
    """
    rows = grade(tool, verdict, protocol, clean, evidence)
    assert len(rows) == 120
    assert len(by(rows, quantity="updates")) == 24
    assert len(by(rows, quantity="schedule")) == 16
    assert len(by(rows, quantity="future_pa")) == 16
    table = list(csv.DictReader(io.StringIO(verdict.rows_to_csv(rows))))
    assert (
        list(table[0])
        == (
            "regime mass_model epoch_count quantity worlds covered_count "
            "covered_50_count pit_sup_distance d2_mean d2_upper exhausted_count "
            "exhausted_cp_upper schedule_deviations_keys schedule_deviations_budget "
            "tv_median tv_p95 latency_ms_per_update memory_mb seeds verdict_coverage "
            "verdict_calibration verdict_alias verdict_failed_updates verdict_schedule "
            "verdict_mechanics protocol_commit code_commit"
        ).split()
    )
    for r in table:
        for c in verdict.CRITERIA:
            assert r[f"verdict_{c}"] in verdict.VERDICTS
    assert verdicts(rows, "coverage") == {"pass", "na"}
    assert verdict.panel_verdict(rows) == "pass"
    cell = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    assert cell["covered_count"] == N and cell["covered_50_count"] == N // 2
    assert cell["verdict_alias"] == "pass" and cell["verdict_failed_updates"] == "na"
    assert cell["seeds"] == "271828/panel"
    upd = by(rows, quantity="updates", epoch_count=1)[0]
    assert upd["exhausted_count"] == 0
    assert upd["latency_ms_per_update"] == "" and upd["memory_mb"] == ""
    assert by(rows, quantity="future_east")[0]["verdict_alias"] == "na"


def test_latency_and_memory_are_only_the_supplied_measurements(
    tool, verdict, protocol, clean, evidence
):
    """Latency and memory are written as given on the update rows and nowhere else.

    Fails if latency is derived from the records' per-world times, or either
    value leaks onto other rows or changes a verdict.
    """
    rows = grade(
        tool, verdict, protocol, clean, evidence, latency_ms=12.5, memory_mb=300.0
    )
    upd = by(rows, quantity="updates")
    assert {r["latency_ms_per_update"] for r in upd} == {12.5}
    assert {r["memory_mb"] for r in upd} == {300.0}
    assert {r["latency_ms_per_update"] for r in rows if r["quantity"] != "updates"} == {
        ""
    }
    assert verdict.panel_verdict(rows) == "pass"


# --- mechanics attestation ---


def test_mechanics_needs_an_attestation_bound_to_the_commit(verdict, tmp_path):
    """Only a complete attestation for the recorded commit counts.

    Fails if a bare flag, a missing log hash, a result other than pass or fail,
    or another commit's attestation yields a verdict other than incomplete, or
    if a recorded failure is not a fail.
    """
    good = {"result": "pass", "code_commit": CODE, "log_sha256": LOG_HASH}
    assert verdict.mechanics_verdict(good, CODE)[0] == "pass"
    assert verdict.mechanics_verdict(dict(good, result="fail"), CODE)[0] == "fail"
    for bad in (
        None,
        "pass",
        {},
        dict(good, result="green"),
        dict(good, log_sha256="zz"),
        dict(good, log_sha256=None),
        dict(good, code_commit="d" * 40),
    ):
        assert verdict.mechanics_verdict(bad, CODE)[0] == "incomplete"
    assert verdict.mechanics_verdict(good, "")[0] == "incomplete"
    log = tmp_path / "mechanics.log"
    log.write_text("42 passed\n")
    digest = hashlib.sha256(log.read_bytes()).hexdigest()
    path = tmp_path / "attest.json"
    entry = {
        "result": "pass",
        "code_commit": CODE,
        "log_sha256": digest,
        "log_path": "mechanics.log",
    }
    path.write_text(json.dumps(entry))
    assert verdict.read_mechanics_attestation(path)["result"] == "pass"
    log.write_text("42 passed, edited\n")
    assert verdict.read_mechanics_attestation(path) == {}


# --- each criterion fails on its own ---


def _is(unit, **kw):
    return all(unit.get(k) == v for k, v in kw.items())


def _undercover(unit, k, spec):
    """Truth outside the 90 percent interval in 8 worlds of one cell."""
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
    if unit["role"] in ("updater_a", "updater_b") and _is(
        unit, regime="R4", mass_model="ti_window"
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


def check_each_criterion_fails_alone(tool, verdict, protocol, evidence):
    """Damage one block per criterion; only the damaged rows may fail."""
    for criterion, (edit, where) in SCENARIOS.items():
        rows = grade(
            tool, verdict, protocol, make_units(tool, protocol, edit), evidence
        )
        failed = [r for r in rows if r[f"verdict_{criterion}"] == "fail"]
        assert failed, f"{criterion}: nothing failed"
        assert all(
            all(str(r[k]) == str(v) for k, v in where.items()) for r in failed
        ), criterion
        others = [c for c in verdict.CRITERIA if c != criterion]
        assert all("fail" not in verdicts(rows, c) for c in others), criterion
        assert verdict.panel_verdict(rows) == "fail"


def test_each_criterion_fails_on_its_own_and_only_in_its_cell(
    tool, verdict, protocol, evidence
):
    """A criterion that fails in one cell fails that row and no other.

    Fails if a criterion is not graded (its damaged cell passes), the verdict
    is pooled over cells (the damaged cell is hidden by the passing ones), or
    a failure spills into another criterion or cell.
    """
    check_each_criterion_fails_alone(tool, verdict, protocol, evidence)


def test_cancelling_strata_do_not_hide_a_failure(tool, verdict, protocol, evidence):
    """Over-coverage in one regime does not offset under-coverage in another.

    Fails if coverage is pooled across blocks: R1 always covers and R2
    undercovers, and the R2 row must still fail.
    """
    rows = grade(
        tool, verdict, protocol, make_units(tool, protocol, _undercover), evidence
    )
    assert verdicts(by(rows, regime="R1"), "coverage") == {"pass", "na"}
    row = by(rows, regime="R2", mass_model="ti_window", epoch_count=3, quantity="e")[0]
    assert row["verdict_coverage"] == "fail"


# --- numeric expectations ---


def _dirichlet(unit, k, scale=60.0):
    seed = zlib.crc32(
        f"{unit['mass_model']}{unit['regime']}{unit['world']}{unit['role']}{k}".encode()
    )
    rng = np.random.default_rng(seed)
    return {q: rng.dirichlet(np.full(8, scale)).tolist() for q in ALIAS_QUANTITIES}


def test_alias_statistic_matches_an_independent_computation(
    tool, verdict, protocol, evidence
):
    """D2, its standard error, the bound and TV equal a hand computation.

    Four distinct runs per world (the two keys and the two references differ
    and are asymmetric), expected values from explicit loops and
    ``statistics.stdev``. Fails if one difference is squared, a reference is
    reused on both sides, the standard deviation uses ddof=0, the standard
    error is halved or doubled, or the z quantile is wrong.
    """

    def bins(unit, k):
        return _dirichlet(unit, k)

    units = make_units(tool, protocol, bins=bins)
    rows = grade(tool, verdict, protocol, units, evidence)
    row = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="lnP")[
        0
    ]
    d2, tv = [], []
    for w in range(N):

        def vec(role, k=None, w=w):
            ident = f"panel/ti_window/R1/w{w:03d}/{role}" + (f"/k{k}" if k else "")
            rec = units[ident]
            s = rec["summaries"]["2"] if rec["kind"] == "updater" else rec["summary"]
            return s["bins"]["lnP"]

        ua, ub, r1, r2 = (
            vec("updater_a"),
            vec("updater_b"),
            vec("reference_1", 2),
            vec("reference_2", 2),
        )
        d2.append(sum((ua[i] - r1[i]) * (ub[i] - r2[i]) for i in range(8)))
        tv.append(0.5 * sum(abs(ua[i] - r1[i]) for i in range(8)))
    z = statistics.NormalDist().inv_cdf(1 - 0.05 / 48)
    se = statistics.stdev(d2) / math.sqrt(N)
    assert row["d2_mean"] == pytest.approx(statistics.fmean(d2), rel=1e-9)
    assert row["d2_upper"] == pytest.approx(statistics.fmean(d2) + z * se, rel=1e-9)
    assert row["tv_median"] == pytest.approx(statistics.median(tv), rel=1e-9)
    assert row["tv_p95"] == pytest.approx(float(np.percentile(tv, 95)), rel=1e-9)
    delta2 = protocol["criteria"]["alias_accuracy"]["delta2"]
    assert (row["verdict_alias"] == "pass") == (row["d2_upper"] <= delta2)


def test_alias_margin_comes_from_the_standard_error(tool, verdict, protocol, evidence):
    """A mean D2 under the tolerance fails when its upper bound exceeds it.

    A quarter of the worlds carry D2 of 1.8e-3, so the mean (4.5e-4) is under
    5e-4 and the mean plus 3.08 standard errors is over it. Fails if the
    bound ignores the standard error (z = 0).
    """

    def edit(unit, k, spec):
        if (
            _is(unit, regime="R1", mass_model="ti_window")
            and unit["role"] in ("updater_a", "updater_b")
            and unit["world"] % 4 == 0
        ):
            spec["summary"]["bins"]["e"] = [0.125 + 0.03, 0.125 - 0.03] + [0.125] * 6

    rows = grade(tool, verdict, protocol, make_units(tool, protocol, edit), evidence)
    row = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    delta2 = protocol["criteria"]["alias_accuracy"]["delta2"]
    assert row["d2_mean"] < delta2 < row["d2_upper"]
    assert row["verdict_alias"] == "fail"


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
    assert verdict.decide_failed_updates(17, 17) == "pass"
    assert verdict.decide_failed_updates(18, 17) == "fail"
    assert verdict.decide_schedule(20, 20, 20) == "pass"
    assert verdict.decide_schedule(20, 21, 20) == "fail"
    assert verdict.decide_alias(5e-4, 5e-4) == "pass"
    assert verdict.decide_alias(5.01e-4, 5e-4) == "fail"
    assert verdict.decide_calibration(0.2, 0.2) == "pass"
    assert verdict.decide_calibration(0.21, 0.2) == "fail"
    nan = float("nan")
    assert verdict.decide_alias(nan, 5e-4) == "fail"
    assert verdict.decide_calibration(nan, 0.2) == "fail"


def _brute_force_sup(pits):
    """Sup over the sample points of the ECDF distance, both sides of each step."""
    n = len(pits)
    worst = 0.0
    for u in pits:
        below = sum(x < u for x in pits) / n
        at_or_below = sum(x <= u for x in pits) / n
        worst = max(worst, abs(at_or_below - u), abs(below - u))
    return worst


def test_pit_sup_distance_matches_the_ecdf_definition(verdict):
    """The sup-distance of PITs equals a brute-force ECDF computation.

    Includes sets where only the step below a point matters (all PITs at one:
    distance 1, which only the lower side of the step gives) and sets where
    only the step above does (all at zero). Fails if either side is dropped.
    """
    assert verdict.pit_sup_distance([1.0] * 10) == pytest.approx(1.0)
    assert verdict.pit_sup_distance([0.0] * 10) == pytest.approx(1.0)
    assert verdict.pit_sup_distance(
        [(i + 0.5) / 10 for i in range(10)]
    ) == pytest.approx(0.05)
    rng = np.random.default_rng(0)
    for _ in range(20):
        sample = np.round(rng.random(rng.integers(3, 15)), 1).tolist()
        assert verdict.pit_sup_distance(sample) == pytest.approx(
            _brute_force_sup(sample)
        )


def test_pits_piled_at_one_fail_calibration(tool, verdict, protocol, evidence):
    """A cell whose PITs are all one fails calibration through its lower step.

    Fails if the row's sup-distance is not 1.
    """

    def edit(unit, k, spec):
        if _is(unit, role="updater_a", regime="R1", mass_model="ti_window"):
            spec["summary"]["pit"]["e"] = 1.0

    rows = grade(tool, verdict, protocol, make_units(tool, protocol, edit), evidence)
    row = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    assert row["pit_sup_distance"] == pytest.approx(1.0)
    assert row["verdict_calibration"] == "fail"


def test_failed_update_bound_matches_beta_quantiles(tool, verdict, protocol, evidence):
    """The failed-update row's count and bound match independent beta quantiles.

    Fails if the count is taken from another update, or the bound is not the
    one-sided 95 percent Clopper-Pearson value (beta quantile of x + 1, n - x).
    """
    from scipy.stats import beta

    rows = grade(
        tool, verdict, protocol, make_units(tool, protocol, _failed_updates), evidence
    )
    row = by(
        rows,
        regime="R1",
        mass_model="physical_fixed",
        epoch_count=2,
        quantity="updates",
    )[0]
    assert row["exhausted_count"] == 4
    assert row["exhausted_cp_upper"] == pytest.approx(
        beta.ppf(0.95, 5, N - 4), rel=1e-9
    )
    other = by(
        rows,
        regime="R1",
        mass_model="physical_fixed",
        epoch_count=1,
        quantity="updates",
    )[0]
    assert other["exhausted_count"] == 0
    assert other["exhausted_cp_upper"] == pytest.approx(beta.ppf(0.95, 1, N), rel=1e-9)


def test_clopper_pearson_matches_beta_quantiles_for_every_count(tool):
    """The one-sided bound equals the beta quantile for every count up to n.

    Fails if an edge count (n - 1, n) or the confidence level is mishandled.
    """
    from scipy.stats import beta

    for n in (3, 12, 50):
        for x in range(n):
            got = tool.clopper_pearson_upper(x, n, 0.95)
            assert got == pytest.approx(beta.ppf(0.95, x + 1, n - x), abs=1e-9)
        assert tool.clopper_pearson_upper(n, n, 0.95) == 1.0


def test_criteria_follow_the_protocol_parameters(tool, verdict, protocol, evidence):
    """Changing the protocol's tolerance, count or delta2 changes the verdict.

    The doubled run differs by 0.4 in five worlds. Fails if the schedule
    tolerance or count, or the alias tolerance, is hardcoded.
    """
    units = make_units(tool, protocol, _unstable)

    def schedule_row(p):
        rows = grade(tool, verdict, p, units, evidence)
        return by(
            rows,
            regime="R2",
            mass_model="physical_fixed",
            epoch_count=2,
            quantity="schedule",
        )[0]

    base = schedule_row(protocol)
    assert (
        base["schedule_deviations_budget"] == 5 and base["verdict_schedule"] == "fail"
    )
    loose = copy.deepcopy(protocol)
    loose["criteria"]["schedule_stability"]["tolerance"] = 0.5
    assert schedule_row(loose)["schedule_deviations_budget"] == 0
    assert schedule_row(loose)["verdict_schedule"] == "pass"
    tight = copy.deepcopy(protocol)
    tight["criteria"]["schedule_stability"]["tolerance"] = 0.01
    assert schedule_row(tight)["schedule_deviations_budget"] == 5
    for count, expected in ((4, "fail"), (5, "pass")):
        counted = copy.deepcopy(protocol)
        counted["criteria"]["schedule_stability"]["max_count"] = count
        assert schedule_row(counted)["verdict_schedule"] == expected
    alias_units = make_units(tool, protocol, _bad_alias)
    wide = copy.deepcopy(protocol)
    wide["criteria"]["alias_accuracy"]["delta2"] = 1.0
    wide["criteria"]["alias_accuracy"]["delta2_max"] = 2.0
    wide_evidence = verdict.freeze_evidence(
        tool, wide, freeze_report(tool, verdict, wide)
    )
    assert wide_evidence["valid"]
    rows = grade(tool, verdict, wide, alias_units, wide_evidence)
    row = by(rows, regime="R4", mass_model="ti_window", epoch_count=2, quantity="lnP")[
        0
    ]
    assert row["verdict_alias"] == "pass"


# --- missing and invalid evidence is incomplete, never a pass ----------------------


def _drop(path):
    def apply(rec):
        node = rec
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return apply


def _set(path, value):
    def apply(rec):
        node = rec
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


def _only(role, mass_model="ti_window", regime="R1", world=None):
    def match(unit):
        return (
            unit["role"] in (role if isinstance(role, tuple) else (role,))
            and unit["mass_model"] == mass_model
            and unit["regime"] == regime
            and (world is None or unit["world"] < world)
        )

    return match


def _zero_bins(rec):
    summary = rec["summary"] if rec["kind"] == "reference" else None
    targets = [summary] if summary else list(rec["summaries"].values())
    for s in targets:
        s["bins"] = {q: [0.0] * 8 for q in ALIAS_QUANTITIES}


INVALID = {
    "missing_status": (
        _only("updater_a", world=3),
        _drop(["updates", 0, "status"]),
        {"quantity": "updates", "epoch_count": 1},
        "failed_updates",
    ),
    "unknown_status": (
        _only("updater_a", world=1),
        _set(["updates", 1, "status"], "WEIRD"),
        {"quantity": "updates", "epoch_count": 2},
        "failed_updates",
    ),
    "no_updates_field": (
        _only(("updater_b", "updater_doubled")),
        _drop(["updates"]),
        {"quantity": "schedule", "epoch_count": 2},
        "schedule",
    ),
    "zero_updater_bins": (
        _only("updater_b"),
        _zero_bins,
        {"quantity": "lnP", "epoch_count": 2},
        "alias",
    ),
    "empty_reference": (
        _only("reference_1", world=2),
        lambda rec: rec.update(
            summary={
                "kish_ess": 0.0,
                "particles": 0,
                "bins": {q: [0.0] * 8 for q in ALIAS_QUANTITIES},
            }
        ),
        {"quantity": "lnP", "epoch_count": 3},
        "alias",
    ),
    "unnormalized_bins": (
        _only("updater_a", world=1),
        _set(["summaries", "2", "bins", "e"], [0.2] * 8),
        {"quantity": "e", "epoch_count": 2},
        "alias",
    ),
    "negative_bins": (
        _only("updater_a", world=1),
        _set(["summaries", "2", "bins", "e"], [-0.125, 0.375] + [0.125] * 6),
        {"quantity": "e", "epoch_count": 2},
        "alias",
    ),
    "infinite_interval": (
        _only("updater_a", world=1),
        _set(["summaries", "2", "intervals", "e", "90"], [-math.inf, math.inf]),
        {"quantity": "e", "epoch_count": 2},
        "coverage",
    ),
    "unordered_interval": (
        _only("updater_a", world=1),
        _set(["summaries", "2", "intervals", "e", "90"], [1.0, -1.0]),
        {"quantity": "e", "epoch_count": 2},
        "coverage",
    ),
    "nan_truth": (
        _only("updater_a", world=1),
        _set(["truth", "2", "e"], math.nan),
        {"quantity": "e", "epoch_count": 2},
        "coverage",
    ),
    "pit_above_one": (
        _only("updater_a", world=1),
        _set(["summaries", "2", "pit", "e"], 1.5),
        {"quantity": "e", "epoch_count": 2},
        "calibration",
    ),
    "nan_pit": (
        _only("updater_a", world=1),
        _set(["summaries", "2", "pit", "e"], math.nan),
        {"quantity": "e", "epoch_count": 2},
        "calibration",
    ),
    "schedule_above_one": (
        _only(("updater_a", "updater_b")),
        _set(["summaries", "2", "schedule_p"], 2.0),
        {"quantity": "schedule", "epoch_count": 2},
        "schedule",
    ),
    "nan_schedule": (
        _only("updater_doubled", world=2),
        _set(["summaries", "2", "schedule_p"], math.nan),
        {"quantity": "schedule", "epoch_count": 2},
        "schedule",
    ),
}


def check_invalid_evidence_is_incomplete(tool, verdict, protocol, evidence):
    """Each kind of missing or invalid evidence makes its criterion incomplete."""
    for name, (match, damage, where, criterion) in INVALID.items():

        def post(unit, rec, match=match, damage=damage):
            if match(unit):
                damage(rec)

        rows = grade(
            tool, verdict, protocol, make_units(tool, protocol, post=post), evidence
        )
        hit = by(rows, regime="R1", mass_model="ti_window", **where)[0]
        assert hit[f"verdict_{criterion}"] == "incomplete", name
        assert verdict.panel_verdict(rows) != "pass", name
        assert not any(r[f"verdict_{criterion}"] == "fail" for r in rows), name


def test_missing_or_invalid_evidence_is_incomplete(tool, verdict, protocol, evidence):
    """Absent or invalid record contents leave the criterion incomplete.

    Covers missing, unknown and absent update statuses, all-zero and
    unnormalized or negative bin masses, empty reference fits, infinite or
    unordered intervals, a NaN truth or PIT, a PIT above one and schedule
    probabilities above one or NaN. Fails if any is counted as a failure, a
    tolerated deviation, a zero distance or a pass.
    """
    check_invalid_evidence_is_incomplete(tool, verdict, protocol, evidence)


def test_all_zero_panel_never_passes(tool, verdict, protocol, evidence):
    """Zero bin masses everywhere, references marked empty, do not pass alias.

    Fails if D2 of zero vectors (0) is graded as an exact match.
    """

    def post(unit, rec):
        if rec["kind"] == "reference":
            rec["summary"] = {"kish_ess": 0.0, "particles": 0}
        else:
            _zero_bins(rec)

    rows = grade(
        tool, verdict, protocol, make_units(tool, protocol, post=post), evidence
    )
    assert verdicts(rows, "alias") == {"incomplete", "na"}
    assert verdict.panel_verdict(rows) == "incomplete"


def test_statuses_are_the_updater_statuses(verdict):
    """The declared statuses are exactly the updater's status names.

    Fails if a status is added to or removed from the updater without the
    grader following.
    """
    from photomancy.orbit.updater import UpdaterStatus

    assert set(verdict.STATUSES) == {s.name for s in UpdaterStatus}


def test_non_ok_statuses_all_count(tool, verdict, protocol, evidence):
    """Every declared non-OK status counts as a failed update.

    Fails if a status is skipped: one world each of the four, three allowed
    (critical count 2), so a count of four fails.
    """

    def edit(unit, k, spec):
        if (
            _is(unit, role="updater_a", regime="R1", mass_model="ti_window")
            and unit["world"] < 4
        ):
            spec["status"][0] = [
                "NO_SUPPORT",
                "BUDGET_EXHAUSTED",
                "LOW_ESS",
                "INCOMPLETE_LADDER",
            ][unit["world"]]

    rows = grade(tool, verdict, protocol, make_units(tool, protocol, edit), evidence)
    row = by(
        rows, regime="R1", mass_model="ti_window", epoch_count=1, quantity="updates"
    )[0]
    assert row["exhausted_count"] == 4 and row["verdict_failed_updates"] == "fail"


def test_missing_worlds_leave_only_their_groups_incomplete(
    tool, verdict, protocol, clean, evidence
):
    """A missing unit makes the criteria that read it incomplete, never pass.

    Fails if the group is graded on the remaining worlds, if an unrelated
    group changes, or if the panel verdict is pass or fail.
    """
    victim = "panel/ti_window/R1/w003/updater_b"
    units = {i: r for i, r in clean.items() if i != victim}
    rows = grade(tool, verdict, protocol, units, evidence)
    hit = by(rows, regime="R1", mass_model="ti_window", epoch_count=2, quantity="e")[0]
    assert hit["verdict_alias"] == "incomplete" and hit["verdict_coverage"] == "pass"
    sched = by(rows, regime="R1", mass_model="ti_window", quantity="schedule")
    assert verdicts(sched, "schedule") == {"incomplete"} and sched[0]["worlds"] == N - 1
    assert sched[0]["schedule_deviations_keys"] == ""
    assert all(
        r["verdict_schedule"] == "pass"
        for r in by(rows, regime="R2", quantity="schedule")
    )
    assert verdict.panel_verdict(rows) == "incomplete"


def test_failure_outranks_incomplete(tool, verdict, protocol, evidence):
    """A failed group fails the panel even while another group is incomplete.

    Fails if incomplete masks a fail (the protocol: fail if any group fails).
    """
    units = make_units(tool, protocol, _undercover)
    units = {i: r for i, r in units.items() if i != "panel/ti_window/R1/w003/updater_b"}
    assert (
        verdict.panel_verdict(grade(tool, verdict, protocol, units, evidence)) == "fail"
    )
    assert verdict.panel_verdict([]) == "incomplete"


def test_alias_bound_above_its_cap_is_incomplete(tool, verdict, protocol, clean):
    """A delta2 above delta2_max leaves every alias verdict incomplete.

    Fails if the grader applies an out-of-bounds tolerance.
    """
    loose = copy.deepcopy(protocol)
    loose["criteria"]["alias_accuracy"]["delta2"] = 2e-3
    evid = verdict.freeze_evidence(tool, loose, freeze_report(tool, verdict, loose))
    rows = grade(tool, verdict, loose, clean, evid)
    assert verdicts(by(rows, quantity="future_pa"), "alias") == {"incomplete"}


# --- certification evidence ---


def test_freeze_requires_canonical_exact_cases(tool, verdict, protocol):
    """Probe: replace all exact verdicts with {invented: pass}, or omit one.

    Fails if an invented or partial set of passing checks is accepted.
    """
    for cases in (
        {"invented": "pass"},
        {"ti_window": "pass", "physical_fixed": "pass"},
    ):
        report = freeze_report(tool, verdict, protocol)
        report["exact_checks"]["verdicts"] = cases
        assert not verdict.freeze_evidence(tool, protocol, report)["valid"]


def test_freeze_rejects_invalid_or_ambiguous_group_power(tool, verdict, protocol):
    """Probe: power 2.0, absent groups, or repeated good and bad group rows.

    Fails if out-of-range bounds, missing groups, duplicates (even identical),
    contradictory duplicates or unknown groups remain valid freeze evidence.
    """
    base = freeze_report(tool, verdict, protocol)
    for value in (2.0, -0.1, math.nan, math.inf, None, True):
        report = copy.deepcopy(base)
        for row in report["alias"]["groups"]:
            row["power_lower_95"] = value
        assert not verdict.freeze_evidence(tool, protocol, report)["valid"]
    for mutant in ("missing", "duplicate", "contradictory", "unknown"):
        report = copy.deepcopy(base)
        groups = report["alias"]["groups"]
        key = tuple(
            groups[0][k] for k in ("regime", "mass_model", "epoch_count", "quantity")
        )
        if mutant == "missing":
            groups.pop(0)
        elif mutant == "unknown":
            groups[0]["regime"] = "invented"
        else:
            groups.append(
                dict(
                    groups[0], power_lower_95=0.0 if mutant == "contradictory" else 0.9
                )
            )
        evid = verdict.freeze_evidence(tool, protocol, report)
        assert not evid["valid"], mutant
        assert key not in evid["power_ok"], mutant


@pytest.mark.parametrize("diagnostic", [None, -1.0, math.nan, math.inf])
def test_reference_diagnostics_cannot_be_missing_or_nonpositive(
    tool, verdict, protocol, clean, evidence, diagnostic
):
    """Probe: normalized reference bins with absent/negative/NaN/infinite ESS.

    Fails if invalid Kish ESS can contribute an alias pass (replaces the
    zero-only empty-fit guard with coverage of populated fit diagnostics).
    """
    ident = "panel/ti_window/R1/w000/reference_1/k2"
    units = dict(clean)
    record = copy.deepcopy(units[ident])
    if diagnostic is None:
        del record["summary"]["kish_ess"]
    else:
        record["summary"]["kish_ess"] = diagnostic
    assert verdict.record_problems(protocol, record)
    units[ident] = record
    row = by(
        grade(tool, verdict, protocol, units, evidence),
        regime="R1",
        mass_model="ti_window",
        epoch_count=2,
        quantity="lnP",
    )[0]
    assert row["verdict_alias"] == "incomplete"


@pytest.mark.parametrize("particles", [None, -1, 1.5, True])
def test_reference_requires_a_positive_integer_retained_count(
    verdict, protocol, clean, particles
):
    """Probe: valid ESS and bins with absent, negative or noninteger particles.

    Fails if a reference with no valid retained-particle count can be scored.
    """
    record = copy.deepcopy(next(r for r in clean.values() if r["kind"] == "reference"))
    if particles is None:
        del record["summary"]["particles"]
    else:
        record["summary"]["particles"] = particles
    assert verdict.record_problems(protocol, record)


def test_grading_document_cites_operative_rules():
    """Probe: the round-2 document cited precision/certification for D2 roles.

    Fails if D2, failures or mechanics lack their operative citations, role
    selection is presented as protocol text, or scoped certification is lost.
    """
    doc = (TOOLS.parent / "docs" / "updater-panel-grading.md").read_text()
    for text in (
        "931-940",
        "966-992",
        "1027-1031",
        "1216-1220",
        "implementation ruling fixed before results",
        "affected mass model",
    ):
        assert text in doc
    assert "every mass model certified" not in doc


def test_alias_cannot_pass_without_freeze_evidence(
    tool, verdict, protocol, clean, evidence
):
    """No evidence, or invalid evidence, blocks every alias pass; fails stay fails.

    Fails if the alias criterion passes on statistics alone, or if a computed
    failure is hidden as incomplete when evidence is missing.
    """
    for none in (
        None,
        {"valid": False, "reasons": ["x"], "certified": {}, "power_ok": set()},
    ):
        rows = grade(tool, verdict, protocol, clean, none)
        assert verdicts(rows, "alias") == {"incomplete", "na"}
        assert verdict.panel_verdict(rows) == "incomplete"
        assert verdicts(rows, "coverage") == {"pass", "na"}
    damaged = make_units(tool, protocol, _bad_alias)
    rows = grade(tool, verdict, protocol, damaged, None)
    row = by(rows, regime="R4", mass_model="ti_window", epoch_count=2, quantity="lnP")[
        0
    ]
    assert row["verdict_alias"] == "fail"


def _binding_mutants(tool, verdict, protocol):
    base = freeze_report(tool, verdict, protocol)
    out = {}
    for name, edit in {
        "status": lambda r: r.update(status="incomplete", reasons=["power"]),
        "run_hash": lambda r: r["binding"].update(run_hash="0" * 64),
        "budgets": lambda r: r["binding"]["budgets"].update(rungs=1),
        "delta2": lambda r: r["alias"].update(delta2=7e-4),
        "binding_delta2": lambda r: r["binding"].update(alias_delta2=7e-4),
        "no_binding": lambda r: r.pop("binding"),
        "exact_failed": lambda r: r["exact_checks"]["verdicts"].update(
            ti_window="fail"
        ),
        "exact_missing": lambda r: r.pop("exact_checks"),
    }.items():
        report = copy.deepcopy(base)
        edit(report)
        out[name] = verdict.freeze_evidence(tool, protocol, report)
    return out


def test_freeze_evidence_must_be_bound_to_the_protocol(tool, verdict, protocol):
    """A freeze report counts only if bound to this run hash, budgets and delta2.

    Fails if a report that is not frozen, is bound to another run hash,
    budgets or alias tolerance, lacks its binding, or has failed or absent
    exact checks is accepted.
    """
    assert verdict.freeze_evidence(
        tool, protocol, freeze_report(tool, verdict, protocol)
    )["valid"]
    for name, evid in _binding_mutants(tool, verdict, protocol).items():
        assert not evid["valid"], name
        assert evid["reasons"], name
    assert not verdict.freeze_evidence(tool, protocol, None)["valid"]


def test_certification_and_power_are_per_mass_model_and_group(
    tool, verdict, protocol, clean
):
    """An uncertified mass model blocks alias; an unpowered group blocks only itself.

    Probe: mark only TI uncertified, including its numerically failing row.
    Fails if physical aliases lose their passes, TI can fail the panel, or
    the power floor is not scoped to its group and quantity.
    """
    report = freeze_report(tool, verdict, protocol)
    report["certification"]["ti_window"]["status"] = "not certified"
    evid = verdict.freeze_evidence(tool, protocol, report)
    rows = grade(tool, verdict, protocol, clean, evid)
    assert verdicts(
        by(rows, mass_model="ti_window", quantity="future_pa"), "alias"
    ) == {"incomplete"}
    assert not evid["valid"] and evid["certified"]["physical_fixed"]
    assert verdicts(
        by(rows, mass_model="physical_fixed", quantity="future_pa"), "alias"
    ) == {"pass"}
    report.update(status="incomplete", reasons=["ti_window reference not certified"])
    scoped = verdict.freeze_evidence(tool, protocol, report)
    damaged = make_units(tool, protocol, _bad_alias)
    scoped_rows = grade(tool, verdict, protocol, damaged, scoped)
    assert verdict.panel_verdict(scoped_rows) == "incomplete"
    assert verdicts(by(scoped_rows, mass_model="ti_window"), "alias") == {
        "incomplete",
        "na",
    }
    assert verdicts(
        by(scoped_rows, mass_model="physical_fixed", quantity="future_pa"), "alias"
    ) == {"pass"}
    report = freeze_report(tool, verdict, protocol)
    for row in report["alias"]["groups"]:
        if (row["regime"], row["mass_model"], row["epoch_count"], row["quantity"]) == (
            "R3",
            "physical_fixed",
            3,
            "e",
        ):
            row["power_lower_95"] = 0.5
    evid = verdict.freeze_evidence(tool, protocol, report)
    rows = grade(tool, verdict, protocol, clean, evid)
    weak = by(
        rows, regime="R3", mass_model="physical_fixed", epoch_count=3, quantity="e"
    )[0]
    assert weak["verdict_alias"] == "incomplete"
    strong = by(
        rows, regime="R3", mass_model="physical_fixed", epoch_count=3, quantity="lnP"
    )[0]
    assert strong["verdict_alias"] == "pass"
    report = freeze_report(tool, verdict, protocol)
    for row in report["alias"]["groups"]:
        if (row["regime"], row["mass_model"], row["epoch_count"], row["quantity"]) == (
            "R4",
            "ti_window",
            2,
            "lnP",
        ):
            row["power_lower_95"] = 0.5
    weak_rows = grade(
        tool,
        verdict,
        protocol,
        damaged,
        verdict.freeze_evidence(tool, protocol, report),
    )
    assert (
        by(
            weak_rows,
            regime="R4",
            mass_model="ti_window",
            epoch_count=2,
            quantity="lnP",
        )[0]["verdict_alias"]
        == "incomplete"
    )


def test_binding_from_the_tool_matches_the_protocol(tool, verdict, protocol):
    """The binding ``--freeze`` writes names the run hash, budgets and delta2.

    Fails if the binding omits a field the grader requires.
    """
    binding = verdict.freeze_binding(tool, protocol)
    assert binding == {
        "run_hash": tool.run_hash(protocol),
        "budgets": tool.protocol_budgets(protocol),
        "alias_delta2": protocol["criteria"]["alias_accuracy"]["delta2"],
    }


# --- directory audit ---


def write_panel(tool, protocol, units, path, shards=2, edit_manifest=None):
    """Write records into ``units-{i}of{n}.jsonl`` and matching manifests."""
    path.mkdir(parents=True, exist_ok=True)
    plan = tool.plan_units(protocol, "panel", list(tool.PARTS))
    ordered = [units[u["id"]] for u in plan if u["id"] in units]
    for i in range(shards):
        with open(path / f"units-{i}of{shards}.jsonl", "w") as fh:
            for rec in ordered[i::shards]:
                fh.write(json.dumps(rec) + "\n")
        entry = {
            "started": f"2026-11-0{i + 1}T00:00:00+00:00",
            "finished": f"2026-11-0{i + 1}T01:00:00+00:00",
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_status": protocol["status"],
            "protocol_hash": tool.declaration_hash(protocol),
            "run_hash": tool.run_hash(protocol),
            "code_commit": CODE,
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


def attestation(commit=CODE):
    """A mechanics attestation."""
    return {"result": "pass", "code_commit": commit, "log_sha256": LOG_HASH}


def committed_freeze(path, report):
    """Create an isolated Git fixture; never commit in the working repository."""
    path.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.PIPE
        )

    git("init", "-q")
    evidence = path / "freeze.json"
    if evidence.exists() and evidence.read_text() == json.dumps(report):
        return evidence
    evidence.write_text(json.dumps(report))
    git("add", "freeze.json")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Synthetic freeze evidence",
    )
    return evidence


def test_freeze_file_must_be_tracked_unchanged_and_unambiguous(
    tool, verdict, protocol, tmp_path
):
    """Probe: use an untracked/outside file, edit it, stage it, or repeat a key.

    Fails if Git membership, HEAD bytes, index changes or duplicate JSON exact
    verdicts can be bypassed. No previous test covered file integrity.
    """
    repo = tmp_path / "repo"
    path = committed_freeze(repo, freeze_report(tool, verdict, protocol))
    original = path.read_text()
    report, identity = verdict.read_freeze_evidence(path, repo)
    assert identity["valid"] and report["status"] == "frozen"
    for other in (
        repo / "untracked.json",
        tmp_path / "outside.json",
        repo / "fre*.json",
        repo / ":(glob)*.json",
    ):
        other.write_text(original)
        assert not verdict.read_freeze_evidence(other, repo)[1]["valid"]
    path.write_text(original + "\n")
    assert not verdict.read_freeze_evidence(path, repo)[1]["valid"]
    subprocess.run(["git", "-C", str(repo), "add", "freeze.json"], check=True)
    path.write_text(original)
    assert not verdict.read_freeze_evidence(path, repo)[1]["valid"]
    subprocess.run(["git", "-C", str(repo), "add", "freeze.json"], check=True)
    assert verdict.read_freeze_evidence(path, repo)[1]["valid"]
    duplicate = original.replace(
        '"ti_binding": "pass"', '"ti_binding": "fail", "ti_binding": "pass"'
    )
    path.write_text(duplicate)
    subprocess.run(["git", "-C", str(repo), "add", "freeze.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Synthetic ambiguous evidence",
        ],
        check=True,
    )
    assert not verdict.read_freeze_evidence(path, repo)[1]["valid"]


def grade_dir(tool, verdict, protocol, path, evidence_report=True, **kw):
    """Grade a directory with valid mechanics and freeze evidence."""
    kw.setdefault("mechanics", attestation())
    if evidence_report:
        kw.setdefault("freeze_report", freeze_report(tool, verdict, protocol))
        report = kw.pop("freeze_report")
        repo = path / "evidence-repo"
        kw.setdefault("freeze_evidence_path", committed_freeze(repo, report))
        tool = SimpleNamespace(**dict(vars(tool), REPO=repo))
    return verdict.grade_panel(tool, protocol, path, **kw)


def test_complete_panel_directory_grades_and_records_commits(
    tool, verdict, protocol, clean, tmp_path
):
    """A complete consistent directory with evidence grades to pass and stamps commits.

    Fails if the audit refuses a good directory, or the table does not carry
    the manifests' protocol and code commits.
    """
    write_panel(tool, protocol, clean, tmp_path / "run")
    result = grade_dir(tool, verdict, protocol, tmp_path / "run", memory_mb=1234.5)
    assert result["status"] == "pass", result["reasons"]
    assert result["audit"]["present"] == result["audit"]["planned"] == 8 * N * 7
    assert {r["code_commit"] for r in result["rows"]} == {CODE}
    assert {r["protocol_commit"] for r in result["rows"]} == {PROTOCOL_COMMIT}
    assert {r["memory_mb"] for r in by(result["rows"], quantity="updates")} == {1234.5}
    out = tmp_path / "out" / "verdict.csv"
    out.parent.mkdir()
    verdict.write_verdict(tool, result, out)
    with out.open() as fh:
        assert len(list(csv.DictReader(fh))) == 120


def test_missing_evidence_inputs_cap_the_panel_at_incomplete(
    tool, verdict, protocol, clean, tmp_path
):
    """Without a mechanics attestation or freeze report the panel is incomplete.

    Fails if either input is optional for a pass, or the reason is not given.
    """
    write_panel(tool, protocol, clean, tmp_path)
    no_mech = grade_dir(tool, verdict, protocol, tmp_path, mechanics=None)
    assert no_mech["status"] == "incomplete" and any(
        "mechanics" in r for r in no_mech["reasons"]
    )
    wrong = grade_dir(
        tool, verdict, protocol, tmp_path, mechanics=attestation("d" * 40)
    )
    assert wrong["status"] == "incomplete"
    no_freeze = grade_dir(tool, verdict, protocol, tmp_path, evidence_report=False)
    assert no_freeze["status"] == "incomplete" and no_freeze["reasons"]
    failed = grade_dir(
        tool, verdict, protocol, tmp_path, mechanics=dict(attestation(), result="fail")
    )
    assert failed["status"] == "fail"


def test_invalid_measurements_are_rejected(tool, verdict, protocol, clean, tmp_path):
    """Latency and memory must be positive finite numbers.

    Fails if a negative, zero, NaN or infinite measurement is written.
    """
    write_panel(tool, protocol, clean, tmp_path)
    for bad in (-1.0, 0.0, math.nan, math.inf):
        for key in ("memory_mb", "latency_ms"):
            with pytest.raises(ValueError):
                grade_dir(tool, verdict, protocol, tmp_path, **{key: bad})


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
    a = verdict.audit_panel(tool, protocol, tmp_path)
    assert (
        a["status"] == "incomplete" and a["missing"] == [victim] and not a["refusals"]
    )
    result = grade_dir(tool, verdict, protocol, tmp_path)
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
    a = verdict.audit_panel(tool, protocol, tmp_path)
    assert a["missing"] != [victim]
    assert a["status"] == "complete"


def _rewrite(path, victim, change):
    for f in path.glob("units-*.jsonl"):
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        for r in rows:
            if r["id"] == victim:
                change(r)
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize(
    "case", ["duplicate", "unknown", "stale", "off_budget", "malformed"]
)
def test_inconsistent_records_refuse_to_grade(
    tool, verdict, protocol, clean, tmp_path, case
):
    """A duplicate, unknown, stale, off-budget or id-less record refuses grading.

    Fails if the grader writes a table from such a directory, or the refusal
    does not say why. Stale and off-budget records must not be silently
    treated as missing units.
    """
    victim = "panel/ti_window/R1/w000/updater_a"
    write_panel(tool, protocol, clean, tmp_path)
    extra = None
    if case == "duplicate":
        extra = clean[victim]
    elif case == "unknown":
        extra = dict(clean[victim], id="panel/ti_window/R1/w999/updater_a")
    elif case == "stale":
        _rewrite(tmp_path, victim, lambda r: r.update(run_hash="0" * 64))
    elif case == "off_budget":
        _rewrite(tmp_path, victim, lambda r: r.update(particles=128))
    elif case == "malformed":
        extra = {"kind": "updater"}
    if extra is not None:
        with open(tmp_path / "units-0of2.jsonl", "a") as fh:
            fh.write(json.dumps(extra) + "\n")
    result = grade_dir(tool, verdict, protocol, tmp_path)
    assert result["status"] == "refused" and result["rows"] is None
    needle = {
        "duplicate": "duplicate",
        "unknown": "not panel",
        "stale": "run hashes",
        "off_budget": "non-protocol budgets",
        "malformed": "no string id",
    }[case]
    assert any(needle in reason for reason in result["audit"]["refusals"])


def _drop_fields(*fields):
    def edit(i, entry):
        for f in fields:
            entry.pop(f, None)

    return edit


def _rungs(i, entry):
    entry["budgets"] = dict(entry["budgets"], rungs=1)


MANIFEST_CASES = [
    ("jax", lambda i, e: e.update(jax="0.9.1") if i == 1 else None, "jax versions"),
    (
        "arch",
        lambda i, e: e.update(platform="macOS-15-arm64-arm-64bit") if i == 1 else None,
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
        "test_budgets",
        lambda i, e: e.update(protocol_budgets=False) if i == 1 else None,
        "non-protocol budgets",
    ),
    ("draft", lambda i, e: e.update(protocol_status="draft"), "statuses"),
    ("rungs", _rungs, "recorded budgets differ"),
    (
        "no_commits",
        _drop_fields("code_commit", "protocol_commit"),
        "unpopulated provenance",
    ),
    ("no_jax", _drop_fields("jax", "jaxlib"), "unpopulated provenance"),
    ("no_platform", _drop_fields("platform"), "unpopulated provenance"),
    ("no_argv", _drop_fields("argv"), "unpopulated provenance"),
    ("bad_commit", lambda i, e: e.update(code_commit="None"), "unpopulated provenance"),
    ("no_budgets", _drop_fields("budgets"), "recorded budgets differ"),
    ("no_dirty_flag", _drop_fields("code_dirty"), "unpopulated provenance"),
]


@pytest.mark.parametrize(
    ("edit", "needle"),
    [(c[1], c[2]) for c in MANIFEST_CASES],
    ids=[c[0] for c in MANIFEST_CASES],
)
def test_manifests_that_disagree_or_lack_provenance_refuse(
    tool, verdict, protocol, clean, tmp_path, edit, needle
):
    """Manifest disagreement, gaps or a flag refuses grading and names the cause.

    Fails if shards under different JAX versions, architectures, chunks or
    commits, a dirty tree, another run or protocol hash, a test-budget
    invocation, a budget dictionary that is not the protocol's, absent
    provenance fields (agreeing with themselves) or a draft protocol is
    graded.
    """
    write_panel(tool, protocol, clean, tmp_path, edit_manifest=edit)
    result = grade_dir(tool, verdict, protocol, tmp_path)
    assert result["status"] == "refused"
    assert any(needle in r for r in result["audit"]["refusals"]), result["audit"][
        "refusals"
    ]


def test_dirty_override_is_exploratory_and_never_passes(
    tool, verdict, protocol, clean, tmp_path
):
    """An accepted dirty tree grades at best incomplete, with no pass in the table.

    Probe: dirty manifests with --allow-dirty convert every pass to incomplete.
    Fails if counts retain pre-conversion passes or disagree with the rows,
    or the override yields a pass, a pass row, or an unmarked commit.
    """
    write_panel(
        tool,
        protocol,
        clean,
        tmp_path,
        edit_manifest=lambda i, e: e.update(code_dirty=True),
    )
    assert grade_dir(tool, verdict, protocol, tmp_path)["status"] == "refused"
    result = grade_dir(tool, verdict, protocol, tmp_path, allow_dirty=True)
    assert result["status"] == "incomplete" and result["exploratory"]
    assert any("exploratory" in r for r in result["reasons"])
    assert all(
        r[f"verdict_{c}"] != "pass" for r in result["rows"] for c in verdict.CRITERIA
    )
    assert {r["code_commit"] for r in result["rows"]} == {CODE + "-dirty"}
    assert result["counts"] == verdict.verdict_counts(result["rows"])
    assert result["counts"]["coverage"]["pass"] == 0
    assert result["counts"]["coverage"]["incomplete"] == 64


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
    assert verdict.audit_panel(tool, draft, tmp_path)["status"] == "refused"
    for f in tmp_path.glob("manifest-*.json"):
        f.unlink()
    a = verdict.audit_panel(tool, protocol, tmp_path)
    assert a["status"] == "refused" and any("no manifest" in r for r in a["refusals"])


# --- mutants ---


def test_skipped_criterion_is_caught(tool, verdict, protocol, evidence, monkeypatch):
    """Mutants: a grader that ignores any one criterion fails the criterion test.

    Each decision replaced by an always-pass is detected by the same check
    the real grader passes.
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
                check_each_criterion_fails_alone(tool, verdict, protocol, evidence)


def test_skipped_validation_is_caught(tool, verdict, protocol, evidence, monkeypatch):
    """Mutant: a grader that does not validate record contents fails the evidence test.

    With ``record_problems`` returning nothing, missing and invalid evidence
    is graded and the incomplete-evidence checks fail.
    """
    with monkeypatch.context() as m:
        m.setattr(verdict, "record_problems", lambda *a, **k: [])
        with pytest.raises((AssertionError, KeyError, TypeError)):
            check_invalid_evidence_is_incomplete(tool, verdict, protocol, evidence)


def test_shifted_statistics_are_caught(tool, verdict, protocol, evidence, monkeypatch):
    """Mutants: a shifted coverage cutoff and a zero z are caught.

    Fails if the coverage threshold or the alias standard-error term could be
    stale or dropped without a test noticing.
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
            check_each_criterion_fails_alone(tool, verdict, protocol, evidence)
    with monkeypatch.context() as m:
        m.setattr(verdict, "thresholds", no_se)
        with pytest.raises(AssertionError):
            test_alias_margin_comes_from_the_standard_error(
                tool, verdict, protocol, evidence
            )
        with pytest.raises(AssertionError):
            test_alias_statistic_matches_an_independent_computation(
                tool, verdict, protocol, evidence
            )


# --- command line ---


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
    assert tool.progress(protocol, tmp_path, "development")[0].startswith(
        "updater: 0 of"
    )


def test_command_line_grades_and_exits_by_verdict(
    tool, verdict, protocol, clean, tmp_path, capsys, monkeypatch
):
    """``--check-panel`` and ``--grade`` run through ``main`` with exit codes.

    Probe: append whitespace to a committed freeze JSON after a passing grade.
    Fails if the edited evidence returns exit 0, Git identity is absent,
    the table is not written for a complete panel with its evidence,
    a refused directory writes one, an exploratory grade overwrites the
    default table, or the exit code is not 0 for pass and 2 otherwise.
    """
    monkeypatch.setattr(tool, "load_protocol", lambda path=None: protocol)
    run = tmp_path / "run"
    write_panel(tool, protocol, clean, run)
    mech = tmp_path / "mechanics.json"
    mech.write_text(json.dumps(attestation()))
    repo = tmp_path / "repo"
    freeze = committed_freeze(repo, freeze_report(tool, verdict, protocol))
    monkeypatch.setattr(tool, "REPO", repo)
    monkeypatch.setattr(tool, "_self_module", lambda verdict: tool)
    out = tmp_path / "verdict.csv"
    assert tool.main(["--check-panel", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    args = [
        "--grade",
        str(run),
        "--mechanics-evidence",
        str(mech),
        "--freeze-evidence",
        str(freeze),
    ]
    assert tool.main([*args, "--verdict-out", str(out)]) == 0 and out.exists()
    printed = capsys.readouterr().out
    result = json.JSONDecoder().raw_decode(printed)[0]
    original = freeze.read_text()
    freeze.write_text(original + "\n")
    assert tool.main([*args, "--verdict-out", str(out)]) == 2
    assert "unmodified" in capsys.readouterr().out
    freeze.write_text(original)
    assert (
        result["freeze_evidence"]["commit"]
        == subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    )
    assert (
        result["freeze_evidence"]["blob_hash"]
        == subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD:freeze.json"], text=True
        ).strip()
    )
    out.unlink()
    assert tool.main(["--grade", str(run), "--verdict-out", str(out)]) == 2
    assert out.exists() and "incomplete" in out.read_text()
    out.unlink()
    for f in run.glob("manifest-*.json"):
        f.unlink()
    assert tool.main([*args, "--verdict-out", str(out)]) == 2 and not out.exists()
    capsys.readouterr()
    dirty = tmp_path / "dirty"
    write_panel(
        tool,
        protocol,
        clean,
        dirty,
        edit_manifest=lambda i, e: e.update(code_dirty=True),
    )
    assert tool.main([*args[:1], str(dirty), *args[2:], "--allow-dirty"]) == 2
    assert "no table written" in capsys.readouterr().out
    assert not (repo / "docs" / "updater-verdict.csv").exists()


# --- smoke ---


@pytest.mark.slow
def test_smoke_panel_runs_the_real_updater_and_grades(tool, verdict, tmp_path):
    """The 24-world smoke stage runs the real updater and grades end to end.

    Small budgets keep it fast (256 particles, 4,096 and 32,768 reference
    proposals), so the verdicts are not meaningful; the test checks that
    every planned unit is produced, the audit does not refuse, the manifests
    carry the provenance the grader requires and the table has the protocol's
    120 rows. A dirty checkout makes the grade exploratory, so the verdict is
    never a pass. Fails if the runner's records, the manifests and the grader
    disagree on any field.
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
    result = verdict.grade_panel(
        tool, protocol, tmp_path, stage="smoke", allow_dirty=True
    )
    assert result["audit"]["status"] != "refused", result["audit"]["refusals"]
    assert result["audit"]["missing"]["count"] == 0
    assert len(result["rows"]) == 120
    assert {r["worlds"] for r in result["rows"]} <= {0, 1, 2, 3}
    assert verdict.rows_to_csv(result["rows"]).count("\n") == 121
    assert result["status"] in {"fail", "incomplete"}
