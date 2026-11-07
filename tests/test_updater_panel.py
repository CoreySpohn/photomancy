"""Tests for the updater acceptance panel driver (tools/updater_panel.py)."""

import importlib.util
import math
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "updater_panel.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("updater_panel", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_critical_values_match_protocol(capsys):
    """The dry run's critical values equal the thresholds the protocol states.

    Fails if the protocol's stated thresholds drift from their formulas (a
    re-sized family without recomputed counts), if a binomial tail is off by
    one, or if the DKW bound is computed from a rounded literal.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    values = tool.dry_run(protocol)
    out = capsys.readouterr().out
    assert "stated values match: True" in out

    for key, stated in protocol["stated"].items():
        assert values[key] == stated, key
    assert values["cells"] == 64
    assert values["coverage_critical_count"] == 339
    assert values["overcoverage_report_count"] == 379
    assert values["failed_updates_critical_count"] == 17
    alpha, n = 0.05 / 64, 400
    assert values["dkw_epsilon"] == math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    assert values["dkw_epsilon_5dp"] == 0.09904
    sched = protocol["criteria"]["schedule_stability"]
    assert sched["tolerance"] <= sched["tolerance_max"] == 0.05
    assert sched["max_count"] <= sched["max_count_cap"] == 20


def test_production_keys_distinct_across_run_roles():
    """Every declared run role gets its own key from the production function.

    Fails if two draws (stage, mass model, regime, world, purpose, index) share
    a key, if a broken key function (one truth key for every world) would go
    unnoticed, or if the keys follow JAX's default generator instead of the
    pinned threefry2x32.
    """
    import jax
    import numpy as np

    tool = _load_tool()
    protocol = tool.load_protocol()
    assert tool.key_collisions(protocol) == 0

    def same_truth_key(protocol, stage, mass_model, r_index, worlds, purpose, index):
        if purpose == "truth":
            worlds = np.zeros_like(worlds)
        return tool.draw_keys(
            protocol, stage, mass_model, r_index, worlds, purpose, index
        )

    assert tool.key_collisions(protocol, same_truth_key) > 0

    pinned = jax.random.key_data(
        tool.draw_key(protocol, "panel", "ti_window", 0, 0, "truth", 0)
    )
    previous = jax.config.jax_default_prng_impl
    try:
        jax.config.update("jax_default_prng_impl", "rbg")
        ambient = jax.random.key_data(
            tool.draw_key(protocol, "panel", "ti_window", 0, 0, "truth", 0)
        )
    finally:
        jax.config.update("jax_default_prng_impl", previous)
    np.testing.assert_array_equal(pinned, ambient)


def test_truths_respect_the_declared_truncation():
    """JAX truth draws are deterministic and inside each regime's support.

    Fails if a truth draw ignores the regime truncation (R3: e at most 0.05),
    escapes the TI mass window, or is not a function of its key alone.
    """
    import jax

    tool = _load_tool()
    protocol = tool.load_protocol()
    r3 = protocol["regimes"][2]
    model = tool.ti_model(protocol, r3)
    for w in range(3):
        key = tool.draw_key(protocol, "smoke", "ti_window", 2, w, "truth", 0)
        theta = tool.ti_truth(model, key)
        with jax.enable_x64(True):
            assert bool(model.in_support(theta))
        assert (tool.ti_truth(model, key) == theta).all()
        key = tool.draw_key(protocol, "smoke", "physical_fixed", 2, w, "truth", 0)
        row = tool.physical_truth(protocol, r3, key)
        a_lo, a_hi = tool.physical_semimajor_range(protocol, r3)
        assert 0.0 <= float(row[1]) < 0.05
        assert a_lo <= float(row[0]) <= a_hi


MASSES = [0.4, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]


def _summary(tool, protocol, updater_ess, ids=None):
    ids = range(len(updater_ess)) if ids is None else ids
    worlds = [
        {
            "world": i,
            "updater_ess": updater_ess[i],
            "reference_ess": 20000,
            "bin_masses": {q: MASSES for q in ("lnP", "e", "future_pa")},
        }
        for i in ids
    ]
    return {
        "groups": [
            {"regime": r, "mass_model": m, "epoch_count": k, "worlds": worlds}
            for r, m, k in tool.canonical_groups(protocol)
        ]
    }


def test_freeze_never_loosens_the_alias_gate():
    """Imprecise or partial development summaries never yield a looser gate.

    Fails if the freeze drops a group, drops or reorders development worlds
    (12 worlds at ESS 30 among 26 at 5,000 need a delta2 above the cap, and
    removing those 12 must be refused, not frozen), or freezes a tolerance whose
    power lower bound is below the floor.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    alias = protocol["criteria"]["alias_accuracy"]
    mixed_ess = [30] * 12 + [5000] * 26

    partial = _summary(tool, protocol, [5000] * 38)
    partial["groups"] = partial["groups"][:1]
    with pytest.raises(ValueError, match="missing groups"):
        tool.freeze_alias(protocol, partial, panels=24)

    dropped = _summary(tool, protocol, mixed_ess, ids=range(12, 38))
    with pytest.raises(ValueError, match="missing"):
        tool.freeze_alias(protocol, dropped, panels=24)
    duplicated = _summary(tool, protocol, mixed_ess, ids=[*range(38), 37])
    with pytest.raises(ValueError, match="repeats"):
        tool.freeze_alias(protocol, duplicated, panels=24)

    shuffled = _summary(tool, protocol, mixed_ess, ids=reversed(range(38)))
    in_order = _summary(tool, protocol, mixed_ess)
    mixed = tool.freeze_alias(protocol, shuffled, panels=4)
    assert mixed == tool.freeze_alias(protocol, in_order, panels=4)
    assert mixed["status"] == "incomplete"
    assert mixed["delta2"] > alias["delta2_max"]

    tight = tool.freeze_alias(
        protocol, _summary(tool, protocol, [5000] * 38), panels=24
    )
    assert tight["status"] == "frozen", tight["reasons"]
    assert tight["delta2"] == alias["delta2"]
    assert all(g["power_lower_95"] >= alias["min_power"] for g in tight["groups"])


def test_exact_check_critical_value():
    """The exact-check critical value is t_15 at 1 - 0.01 / (2 m), m = 140.

    Fails if the compared-value count, the repeat count or the Student t
    quantile drift from the protocol (a normal quantile, 3.75, would reject a
    correct sampler far more often than 1 percent with 16 repeats).
    """
    from scipy import stats

    tool = _load_tool()
    protocol = tool.load_protocol()
    ex = protocol["reference"]["exact_checks"]
    m = ex["cases"] * (ex["bins_per_case"] + ex["interval_end_points_per_case"])
    assert m == protocol["stated"]["exact_check_compared_values"] == 140
    expected = stats.t.ppf(
        1.0 - ex["family_alpha"] / (2 * m), ex["sampler_repeats"] - 1
    )
    values = tool.critical_values(protocol)
    assert values["exact_check_t_critical"] == pytest.approx(expected, abs=1e-9)
    assert round(expected, 3) == protocol["stated"]["exact_check_t_critical_3dp"]


def test_development_plan_shards_disjointly_and_matches_the_tally():
    """Shards partition the development units; the plan costs what the tally says.

    Fails if two shards share a unit or a unit falls in no shard (the worker
    would duplicate or drop work), if unit ids collide, or if the planned
    updater runs and reference fits (doubled roles costing two) drift from
    the protocol's compute tally.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    units = tool.plan_units(protocol, "development", list(tool.PARTS))
    ids = [u["id"] for u in units]
    assert len(set(ids)) == len(ids)
    for n in (2, 3):
        parts = [{u["id"] for u in tool.shard_units(units, (i, n))} for i in range(n)]
        assert sum(len(p) for p in parts) == len(ids)
        assert set().union(*parts) == set(ids)
    tally = tool.compute_tally(protocol)

    def cost(u):
        if u["role"] in tool.ALL_UPDATER_ROLES:
            return 2 if "doubled" in u["role"] else 1
        return tool.reference_budget_factor(u["mass_model"], u["role"])

    upd = sum(cost(u) for u in units if u["role"] in tool.ALL_UPDATER_ROLES)
    n_mass = len(protocol["panel"]["mass_models"])
    n_updates = len(protocol["panel"]["updates"])
    assert upd * n_updates == n_mass * tally["development_update_equivalents_per_pair"]
    for pair in protocol["panel"]["mass_models"]:
        ref = sum(
            cost(u)
            for u in units
            if u["role"] not in tool.ALL_UPDATER_ROLES and u["mass_model"] == pair
        )
        assert ref == tally["development_reference_fit_equivalents_per_pair"][pair]
    with pytest.raises(ValueError):
        tool.parse_shard("2/2")


def test_run_resumes_after_a_kill_without_duplicate_work(tmp_path):
    """A killed run restarts where it stopped; a torn last line is skipped.

    The first run dies after three units, leaving half a record on the last
    line. Fails if the restart reruns a finished unit, loses the unfinished
    ones, appends its first record onto the torn line, or if two shards
    writing to one directory skip each other's finished units incorrectly.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    units = tool.plan_units(
        protocol, "smoke", ["updater"], regimes=["R1"], mass_models=["ti_window"]
    )
    calls = []

    class Killed(Exception):
        pass

    def dying(unit):
        if len(calls) == 3:
            raise Killed
        calls.append(unit["id"])
        return {"kind": "fake", "seconds": 0.0}

    with pytest.raises(Killed):
        tool.run_units(units, tmp_path, (0, 1), dying, log=lambda m: None)
    path = tmp_path / "units-0of1.jsonl"
    with open(path, "a") as fh:
        fh.write('{"id": "torn')
    first = list(calls)

    def finishing(unit):
        calls.append(unit["id"])
        return {"kind": "fake", "seconds": 0.0}

    counts = tool.run_units(units, tmp_path, (0, 1), finishing, log=lambda m: None)
    assert counts == {"planned": len(units), "skipped": 3, "ran": len(units) - 3}
    assert sorted(calls) == sorted(u["id"] for u in units)
    assert set(calls[3:]).isdisjoint(first)
    records, duplicates = tool.load_units(tmp_path)
    assert set(records) == {u["id"] for u in units} and not duplicates
    again = tool.run_units(units, tmp_path, (0, 1), finishing, log=lambda m: None)
    assert again["ran"] == 0


def _fake_records(tool, protocol, roles, worlds, masses=None, run_hash=None):
    """Development records at the protocol budgets for the given roles."""
    budgets = tool.protocol_budgets(protocol)
    flat = masses or [1.0 / 8] * 8
    bins = {q: flat for q in ("lnP", "e", "future_pa")}
    out = {}
    for unit in tool.plan_units(protocol, "development", list(tool.PARTS)):
        if unit["role"] not in roles or unit["world"] not in worlds:
            continue
        doubled = 2 if "doubled" in unit["role"] else 1
        if unit["role"] in tool.ALL_UPDATER_ROLES:
            rec = {
                "kind": "updater",
                "particles": budgets["particles"] * doubled,
                "updates": [{"kernel_ess": 7372.8}] * 3,
                "summaries": {str(k): {"bins": bins} for k in (2, 3)},
            }
        else:
            rec = {
                "kind": "reference",
                "proposals": budgets[unit["mass_model"]]
                * tool.reference_budget_factor(unit["mass_model"], unit["role"]),
                "summary": {"kish_ess": 900.0, "bins": bins},
            }
        out[unit["id"]] = dict(rec, id=unit["id"], run_hash=run_hash)
    return out


def test_freeze_precision_is_unbiased_in_inverse_square_size(tmp_path):
    """Freeze precision comes from replicate spread, unbiased where D2 needs it.

    The D2 variance is quadratic in the per-run covariance, so the estimate
    must be unbiased in 1 / n^2. For eight multinomial runs at size n = 500
    over Dirichlet(0.7) masses, n^2 mean(1 / size^2) is within 6 percent of
    1 (2,000 worlds; standard error about 1.5 percent); for one pair of
    two-bin runs it is within 15 percent (4,000 worlds, standard error about
    3 percent), where ``tr M / s^2`` (``key_to_key_ess``) is three times too
    large. In assembled records the kernel ESS (0.9 N by construction) never
    enters: runs that disagree get a small precision whatever the kernel
    reports. A partial run directory freezes to ``incomplete`` and names the
    records run under other code or protocol content; the run hash ignores
    the status but changes with the kernel, the summary bins and the source
    of a record-defining function. Fails if the freeze reads the kernel ESS,
    estimates 1 / n rather than 1 / n^2, freezes a partial panel, or reuses
    records after a change to the kernel, the bins or the code.
    """
    import numpy as np

    tool = _load_tool()
    protocol = tool.load_protocol()
    rng = np.random.default_rng(0)
    n = 500
    inv2 = []
    for _ in range(2000):
        p = rng.dirichlet(np.full(8, 0.7))
        runs = rng.multinomial(n, p, size=8) / n
        inv2.append(tool.replicate_ess(runs, cap=10**6) ** -2.0)
    assert abs(n**2 * np.mean(inv2) - 1.0) < 0.06
    pair, linear = [], []
    for _ in range(4000):
        p = np.zeros(8)
        p[:2] = rng.dirichlet([2.0, 2.0])
        a, b = rng.multinomial(n, p, size=2) / n
        pair.append(tool.replicate_ess([a, b], cap=10**6) ** -2.0)
        linear.append(tool.key_to_key_ess(a, b, cap=10**6) ** -2.0)
    assert abs(n**2 * np.mean(pair) - 1.0) < 0.15
    assert n**2 * np.mean(linear) > 2.0

    skew = [0.5, 0.5] + [0.0] * 6
    units = _fake_records(
        tool,
        protocol,
        set(tool.PARTS["updater"])
        | set(tool.PRECISION_ROLES)
        | {"reference_1", "reference_2"},
        {0, 1},
        run_hash=tool.run_hash(protocol),
    )
    for rid, rec in units.items():
        if any(f"/{r}" in rid for r in ("updater_b", *tool.PRECISION_ROLES[:3])):
            rec["summaries"] = {
                str(k): {"bins": {q: skew for q in ("lnP", "e", "future_pa")}}
                for k in (2, 3)
            }
    group_id = tool.unit_id("development", "ti_window", "R1", 1, "reference_1", 2)
    units[group_id]["summary"] = {"kish_ess": 0.0, "particles": 0}
    summary, missing = tool.assemble_freeze_summary(protocol, units)
    assert summary["empty_references"] == [group_id]
    world = next(g for g in summary["groups"] if g["worlds"])["worlds"][0]
    assert world["kernel_ess"] == 7372.8
    assert all(v < 10 for v in world["updater_ess"].values())
    assert missing

    stale = tool.unit_id("development", "physical_fixed", "R4", 0, "updater_a")
    units[stale]["run_hash"] = None
    for rec in units.values():
        tool.append_record(tmp_path / "units-0of1.jsonl", rec)
    report = tool.freeze_report(protocol, tmp_path)
    assert report["status"] == "incomplete"
    assert any("planned development units missing" in r for r in report["reasons"])
    assert any("kept no weighted draw" in r for r in report["reasons"])
    assert any("exact-check results not supplied" in r for r in report["reasons"])
    # One record carries no run hash, as if run under other code.
    assert any("other code or protocol content" in r for r in report["reasons"])
    current = tool.run_hash(protocol)
    assert tool.run_hash(dict(protocol, status="frozen")) == current
    kernel = dict(protocol["updater_kernel"], alias_moves=False)
    assert tool.run_hash(dict(protocol, updater_kernel=kernel)) != current
    alias = dict(protocol["criteria"]["alias_accuracy"], bins=16)
    criteria = dict(protocol["criteria"], alias_accuracy=alias)
    assert tool.run_hash(dict(protocol, criteria=criteria)) != current
    definitions = tool._record_definitions
    tool._record_definitions = lambda: (*definitions(), _fake_records)
    assert tool.run_hash(protocol) != current


def test_estimated_precision_does_not_loosen_the_alias_tolerance():
    """The freeze with estimated precision is no looser than with known precision.

    Reproduction of the case that exposed the one-pair estimate: 38 two-bin
    worlds per group, true updater size 300, reference size 20,000. With
    precision estimated from eight replicates the largest and the mean
    per-group SE0 stay within 15 percent of their values at the known size
    (measured +7 and -2 percent); one pair through ``key_to_key_ess``
    inflates the largest by more than 30 percent (measured +156). Fails if
    estimation noise in the updater precision can widen the frozen
    tolerance.
    """
    import numpy as np

    tool = _load_tool()
    protocol = tool.load_protocol()
    rng = np.random.default_rng(0)
    n = 300
    worlds = {"known": [], "replicates": [], "pair": []}
    for _ in tool.canonical_groups(protocol):
        rows = {key: [] for key in worlds}
        for w in range(38):
            masses, est = {}, {key: {} for key in worlds}
            for q in ("lnP", "e", "future_pa"):
                p = np.zeros(8)
                p[:2] = rng.dirichlet([2.0, 2.0])
                runs = rng.multinomial(n, p, size=8) / n
                masses[q] = p.tolist()
                est["known"][q] = n
                est["replicates"][q] = tool.replicate_ess(runs, 8192)
                est["pair"][q] = tool.key_to_key_ess(runs[0], runs[1], 8192)
            for key in worlds:
                rows[key].append(
                    {
                        "world": w,
                        "updater_ess": est[key],
                        "reference_ess": 20000,
                        "bin_masses": masses,
                    }
                )
        for key in worlds:
            worlds[key].append(rows[key])

    def se0(key):
        summary = {
            "groups": [
                {"regime": r, "mass_model": m, "epoch_count": k, "worlds": ws}
                for (r, m, k), ws in zip(
                    tool.canonical_groups(protocol), worlds[key], strict=True
                )
            ]
        }
        out = tool.freeze_alias(protocol, summary, panels=25)
        values = np.array([g["se0"] for g in out["groups"]])
        return values.max(), values.mean()

    known, replicates, pair = se0("known"), se0("replicates"), se0("pair")
    assert replicates[0] < 1.15 * known[0] and replicates[1] < 1.15 * known[1]
    assert pair[0] > 1.3 * known[0]


def test_freeze_requires_every_planned_development_unit(tmp_path):
    """A run missing one planned role, or one reference below its floor, is incomplete.

    Every planned development unit is present at the protocol budgets under
    the current run hash except the 304 ``updater_doubled`` runs, which no
    freeze statistic reads. Then with every unit present, one primary TI
    reference at ESS 199 and one check-3 reference at 150 must each be named
    with its budget against the floor of 200. Fails if the freeze grades a
    directory that does not hold the complete canonical development plan, or
    accepts a reference fit below the ESS floor.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    roles = {r for part in tool.PARTS.values() for r in part} - {"updater_doubled"}
    current = tool.run_hash(protocol)
    units = _fake_records(tool, protocol, roles, set(range(38)), run_hash=current)
    path = tmp_path / "units-0of1.jsonl"
    path.write_text("".join(tool.json.dumps(r) + "\n" for r in units.values()))
    report = tool.freeze_report(protocol, tmp_path)
    assert report["status"] == "incomplete"
    reason = next(r for r in report["reasons"] if "planned development" in r)
    assert reason.startswith("304 of ") and "updater_doubled" in reason
    assert "alias" not in report
    assert not any("ESS floor" in r for r in report["reasons"])

    # Every planned unit present, one primary TI fit at ESS 199 and one
    # check-3 fit at 150: the floor of 200 rejects both, by id and budget.
    units.update(
        _fake_records(
            tool, protocol, {"updater_doubled"}, set(range(38)), run_hash=current
        )
    )
    low = tool.unit_id("development", "ti_window", "R2", 7, "reference_1", 3)
    cert = tool.unit_id("development", "ti_window", "R3", 0, "cert_proposal_2", 2)
    units[low]["summary"] = dict(units[low]["summary"], kish_ess=199.0)
    units[cert]["summary"] = dict(units[cert]["summary"], kish_ess=150.0)
    path.write_text("".join(tool.json.dumps(r) + "\n" for r in units.values()))
    # Few simulated panels: only the reasons are under test here.
    alias = dict(protocol["criteria"]["alias_accuracy"], freeze_panels=2)
    quick = dict(protocol, criteria=dict(protocol["criteria"], alias_accuracy=alias))
    report = tool.freeze_report(quick, tmp_path)
    assert not any("planned development" in r for r in report["reasons"])
    floor = [r for r in report["reasons"] if "ESS floor" in r]
    assert len(floor) == 2
    ti_budget = protocol["reference"]["ti_window"]["proposals_per_fit"]
    assert f"('{low}', {ti_budget}, 199.0)" in floor[0]
    assert f"('{cert}', {2 * ti_budget}, 150.0)" in floor[1]
    table = tool.ess_table(protocol, units)
    assert table["ti_window R2 k=3 primary"]["below_floor"] == [low]
    assert table["ti_window R3 k=2 independent"]["below_floor"] == [cert]


@pytest.mark.parametrize(
    "role",
    [f"reference_{i}" for i in range(1, 5)]
    + ["cert_doubled_1", "cert_doubled_2", "cert_proposal_1", "cert_proposal_2"],
)
def test_every_required_ti_reference_role_is_held_to_the_floor(role):
    """Each required TI reference role below the ESS floor is named.

    One fit of the role at ESS 199 among fits at 900 must appear in the
    freeze's floor reason with its budget. A primary fit (budget 2^22) is
    told to double; a doubled or check-3 fit (2^23, the cap) is told its
    method needs an amendment, never to double past the cap. Fails if any
    role is left out of the floor check, or if the remedy ignores the cap.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    roles = {r for part in tool.PARTS.values() for r in part}
    units = _fake_records(tool, protocol, roles, {0})
    low = tool.unit_id("development", "ti_window", "R2", 0, role, 3)
    units[low]["summary"] = dict(units[low]["summary"], kish_ess=199.0)
    reasons = tool.ess_floor_reasons(protocol, units)
    assert len(reasons) == 1
    proposals = units[low]["proposals"]
    assert f"('{low}', {proposals}, 199.0)" in reasons[0]
    cap = protocol["reference"]["ti_window"]["max_proposals_per_fit"]
    if 2 * proposals <= cap:
        assert "double that method's budget" in reasons[0]
    else:
        assert "needs an amendment" in reasons[0]
        assert "double that method" not in reasons[0]


def test_budgets_above_the_proposal_cap_are_refused(tmp_path):
    """A TI budget at the cap is refused before running and at the freeze.

    With the TI primary budget doubled to 2^23 (the cap), the doubled and
    check-3 runs would be at 2^24. The development runner and the exact-check
    runner must refuse to start, the freeze and the exact report must give
    the cap as a reason, and a record at 2^24 must not count as a protocol-
    budget record. Fails if the cap on effective budgets is only descriptive.
    """
    tool = _load_tool()
    spec = importlib.util.spec_from_file_location(
        "updater_exact_checks", TOOL.parent / "updater_exact_checks.py"
    )
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    text = tool.PROTOCOL.read_text()
    budget = '"ti_window": {"proposals_per_fit": 4194304,'
    assert text.count(budget) == 1
    path = tmp_path / "protocol.md"
    path.write_text(text.replace(budget, '"ti_window": {"proposals_per_fit": 8388608,'))
    protocol = tool.load_protocol(path)
    assert tool.budget_cap_violations(protocol)
    assert exact.budget_cap_violations(protocol)
    with pytest.raises(SystemExit, match="refusing to run"):
        tool.main(
            [
                "--protocol",
                str(path),
                "--run",
                "development",
                "--out",
                str(tmp_path / "dev"),
                "--worlds",
                "0",
            ]
        )
    with pytest.raises(SystemExit, match="refusing to run"):
        exact.main(["--protocol", str(path), "--out", str(tmp_path / "exact")])
    assert not (tmp_path / "dev").exists() and not (tmp_path / "exact").exists()
    (tmp_path / "empty").mkdir()
    reasons = tool.freeze_report(protocol, tmp_path / "empty", None)["reasons"]
    assert any("exceed the proposal cap" in r for r in reasons)
    report = exact.report(protocol, tmp_path / "empty")
    assert all(
        any("exceed the proposal cap" in r for r in e["reasons"])
        for e in report["cases"].values()
    )
    uid = tool.unit_id("development", "ti_window", "R2", 0, "cert_proposal_1", 3)
    record = {"id": uid, "kind": "reference", "proposals": 2 * 8388608}
    assert not tool._record_at_protocol_budget(protocol, record)


def test_resume_refuses_incompatible_records_and_manifests_survive_kills(tmp_path):
    """Resume refuses a directory it cannot reuse; manifests are crash-safe.

    A directory holding one updater record at 64 particles is refused by a
    protocol-budget run before anything runs or any manifest is written. A
    manifest write killed before its rename leaves the previous manifest
    intact and no temporary file; the run writes its manifest before the
    first unit (a run killed in that unit leaves its provenance), and so does
    the exact-check runner. Fails if a stale record is skipped as done, a
    kill can truncate a manifest, or a killed run leaves no provenance.
    """
    import os

    tool = _load_tool()
    protocol = tool.load_protocol()
    out = tmp_path / "dev"
    out.mkdir()
    stale = _fake_records(tool, protocol, {"updater_a"}, {0}, run_hash=None)
    rid = tool.unit_id("development", "ti_window", "R1", 0, "updater_a")
    rec = dict(stale[rid], particles=64, run_hash=tool.run_hash(protocol))
    tool.append_record(out / "units-0of1.jsonl", rec)
    args = ["--run", "development", "--out", str(out), "--worlds", "0"]
    args += ["--regimes", "R1", "--mass-models", "ti_window", "--parts", "updater"]
    with pytest.raises(SystemExit, match="refusing to resume"):
        tool.main(args)
    assert not list(out.glob("manifest-*"))

    tool.write_manifest(out, (0, 1), {"started": "a"})
    before = (out / "manifest-0of1.json").read_text()
    real_fsync = os.fsync

    def killed(fd):
        raise KeyboardInterrupt

    os.fsync = killed
    try:
        with pytest.raises(KeyboardInterrupt):
            tool.write_manifest(out, (0, 1), {"started": "b"})
    finally:
        os.fsync = real_fsync
    assert (out / "manifest-0of1.json").read_text() == before
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest-0of1.json",
        "units-0of1.jsonl",
    ]

    class Killed(Exception):
        pass

    def dying(unit):
        raise Killed

    entry = {"started": "c", "finished": None}
    units = tool.plan_units(protocol, "smoke", ["updater"], regimes=["R2"])
    with pytest.raises(Killed):
        tool.run_units(
            units,
            tmp_path / "smoke",
            (0, 1),
            tool._ManifestFirst(dying, tmp_path / "smoke", (0, 1), entry),
        )
    written = tool.json.loads((tmp_path / "smoke" / "manifest-0of1.json").read_text())
    assert written["invocations"][-1] == entry

    spec = importlib.util.spec_from_file_location(
        "updater_exact_checks", TOOL.parent / "updater_exact_checks.py"
    )
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)

    class DyingExact:
        run_hash = "x"

        def __init__(self, *args):
            pass

        def __call__(self, unit):
            raise Killed

    exact.ExactExecutor = DyingExact
    with pytest.raises(Killed):
        exact.main(["--out", str(tmp_path / "exact"), "--levels", "0"])
    manifest = exact.panel.json.loads(
        (tmp_path / "exact" / "manifest-0of1.json").read_text()
    )
    assert manifest["invocations"][-1]["finished"] is None


@pytest.mark.medium
def test_units_write_complete_summaries_and_exact_physical_weights():
    """One updater unit and one physical reference produce gradable records.

    The updater record carries every update's status, kernel and Kish ESS
    and timing, and per epoch count normalized bin masses, ordered interval
    end points, a schedule probability and PITs in [0, 1]. The physical
    reference conditioned on its one epoch (anchor weight rho^-2, anchor
    likelihood excluded) must scatter the anchor position by sigma per axis
    (2^15 proposals, ESS in the thousands; the measured ratio is within 3
    percent of 1). Fails if a summary field is missing or unnormalized, the
    interval rule inverts, or the reference counts the anchor epoch twice
    (scatter sigma / sqrt 2, a ratio of 0.71).
    """
    import jax
    import numpy as np

    from photomancy.orbit.updater import UpdaterStatus

    tool = _load_tool()
    protocol = tool.load_protocol()
    budgets = tool.protocol_budgets(
        protocol, particles=256, ti_proposals=4096, physical_proposals=2**15
    )
    executor = tool.UnitExecutor(protocol, budgets, chunk=2**13)
    unit = tool.plan_units(
        protocol,
        "smoke",
        ["updater"],
        mass_models=["physical_fixed"],
        regimes=["R3"],
        worlds=[0],
    )[0]
    record = executor(unit)
    assert [u["update"] for u in record["updates"]] == [0, 1, 2]
    for u in record["updates"]:
        assert u["status"] in {s.name for s in UpdaterStatus}
        assert u["kernel_ess"] > 0 and u["kish_ess"] > 0 and u["seconds"] > 0
    for k in ("2", "3"):
        s = record["summaries"][k]
        for masses in s["bins"].values():
            assert abs(sum(masses) - 1.0) < 1e-12
        for ends in s["intervals"].values():
            assert ends["90"][0] <= ends["50"][0] <= ends["50"][1] <= ends["90"][1]
        assert 0.0 <= s["schedule_p"] <= 1.0
        assert all(0.0 <= u <= 1.0 for u in s["pit"].values())
        assert set(record["truth"][k]) == set(tool.QUANTITIES)

    world = tool.world_data(protocol, "smoke", "physical_fixed", 2, 0)
    key = jax.random.key(0, impl="threefry2x32")
    q, lw, _ = executor.physical_reference(world, 1, key, 2**15, False, 0.0)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    assert 1.0 / np.sum(w**2) > 1000
    for name, obs in (
        ("future_east", world["ra"][0]),
        ("future_north", world["dec"][0]),
    ):
        mean = w @ q[name]
        sd = np.sqrt(w @ (q[name] - mean) ** 2)
        assert abs(mean - obs) < 0.1 * world["sigma"]
        assert abs(sd / world["sigma"] - 1.0) < 0.06


# Slow: six-world, four-role certification retains full precision and both mutants.
@pytest.mark.slow
def test_ti_independent_reference_catches_an_injected_bias():
    """Certification check 3 for TI passes two exact references, fails a biased one.

    Six R3 development worlds at two epochs (the implied-mass window binds
    there): the collapsed prior reference (``reference_1``, ``reference_2``,
    2^15 proposals) against the frequency-mixture reference
    (``cert_proposal_1``, ``cert_proposal_2``, 2^16). The check statistic is
    the protocol's: per world ``sum_b (R1 - C1)(R2 - C2)``, pooled, with
    ``mean + z SE`` (z = 3.078) against half the provisional delta2. The two
    exact references pass for every alias quantity (largest upper bound
    8e-5 against 2.5e-4). Each injected bias makes some quantity fail: the
    primary reference without the implied-mass window (ln P upper bound
    0.2), and the independent reference without the proposal density of its
    TI-constant draw (its Jacobian). Fails if either TI reference is biased,
    or if check 3 cannot see a window or proposal-density error in one
    method.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from photomancy.orbit.updater import ThieleInnesTarget, _linear_gaussian

    tool = _load_tool()
    protocol = tool.load_protocol()
    budgets = tool.protocol_budgets(protocol, particles=64, ti_proposals=2**15)
    roles = {"reference_1", "reference_2", "cert_proposal_1", "cert_proposal_2"}
    units = [
        u
        for u in tool.plan_units(
            protocol,
            "development",
            ["reference", "certification"],
            mass_models=["ti_window"],
            regimes=["R3"],
            worlds=range(6),
        )
        if u["role"] in roles and u["epoch_count"] == 2
    ]

    class NoWindowTI(ThieleInnesTarget):
        def log_prior(self, theta):
            m = self.model
            box = (theta[0] > 0) & (theta[0] < m.e_max)
            box &= (theta[1] > m.lnP_lo) & (theta[1] < m.lnP_hi)
            return jnp.where(box, m.log_prior_ti(theta), -jnp.inf)

    collapsed, frequency = tool.ti_collapsed_reference, tool.ti_frequency_reference

    def no_window(target, data, key, n_prop, chunk):
        mutant = NoWindowTI(target.model, target.dist_pc)
        return collapsed(mutant, data, key, n_prop, chunk)

    def no_constants_density(target, data, key, n_prop, chunk, config=None):
        # Undo the weight's division by the TI constants' proposal density.
        theta, lw = frequency(target, data, key, n_prop, chunk, config)
        kappa, weights = config["inflation"], data.is_valid.astype(float)

        def log_q_c(t):
            mean, chol, _, _, _ = _linear_gaussian(target, t, data, weights)
            r = chol.T @ (t[3:] - mean) / kappa
            return -0.5 * r @ r + jnp.sum(jnp.log(jnp.diag(chol)))

        return theta, lw + np.asarray(jax.vmap(log_q_c)(jnp.asarray(theta)))

    def check3(**patch):
        for name, fn in patch.items():
            setattr(tool, name, fn)
        try:
            executor = tool.UnitExecutor(protocol, budgets, chunk=2**16)
            records = {u["id"]: dict(executor(u), id=u["id"]) for u in units}
        finally:
            tool.ti_collapsed_reference = collapsed
            tool.ti_frequency_reference = frequency
        cert = tool.certify_references(protocol, records, 5e-4)["ti_window"]
        rows = [r for r in cert["checks"] if r["check"] == "proposal_agreement"]
        assert len(rows) == 3 and all(r["worlds"] == 6 for r in rows)
        return {r["quantity"]: r["pass"] for r in rows}

    assert all(check3().values())
    assert not all(check3(ti_collapsed_reference=no_window).values())
    assert not all(check3(ti_frequency_reference=no_constants_density).values())


def test_check3_proposal_densities_match_their_draws():
    """The check-3 sampler's proposal densities are the densities it draws from.

    ln P: 2^18 draws of the mixture (half prior, half uniform frequency) over
    40 equal ln P bins match the density ``p(ln P) exp(log_ratio)`` within 5
    standard errors per bin; the same draws against the prior density alone
    (the mixture correction removed) miss by more than 20. TI constants: the
    log density returned with an inflated conditional draw equals the
    multivariate normal ``N(m, kappa^2 P^-1)`` log density to 1e-10 for
    kappa 1.25 and 1; dropping its kappa terms is off by ``4 log kappa``.
    Fails if either proposal correction is wrong or removed.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import multivariate_normal

    from photomancy.orbit.few_epoch import TIBasisModel
    from photomancy.orbit.updater import ThieleInnesTarget

    tool = _load_tool()
    tool.enable_x64()
    target = ThieleInnesTarget(TIBasisModel(lnP_hi=math.log(500.0)))
    m = target.model
    n = 2**18
    k_p, k_mix, k_f = jax.random.split(jax.random.key(0), 3)
    prior = jax.random.uniform(k_p, (n,), minval=m.lnP_lo, maxval=m.lnP_hi)
    draws = np.asarray(tool.frequency_mixture_ln_period(target, k_mix, k_f, prior, 0.5))
    edges = np.linspace(m.lnP_lo, m.lnP_hi, 41)
    counts = np.histogram(draws, edges)[0]
    fine = np.linspace(m.lnP_lo, m.lnP_hi, 40 * 64 + 1)
    centers = 0.5 * (fine[1:] + fine[:-1])
    ratio = np.exp(np.asarray(tool.frequency_mixture_log_ratio(target, centers, 0.5)))
    span = m.lnP_hi - m.lnP_lo
    expected = n * (ratio / span * np.diff(fine)).reshape(40, 64).sum(axis=1)
    prior_only = np.full(40, n / 40.0)
    assert np.max(np.abs(counts - expected) / np.sqrt(expected)) < 5.0
    assert np.max(np.abs(counts - prior_only) / np.sqrt(prior_only)) > 20.0

    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    precision = a @ a.T + 4.0 * np.eye(4)
    chol = np.linalg.cholesky(precision)
    mean = rng.normal(size=4)
    for kappa in (1.25, 1.0):
        for _ in range(3):
            z = rng.normal(size=4)
            c, log_q = tool.inflated_conditional_draw(
                jnp.asarray(mean), jnp.asarray(chol), jnp.asarray(z), kappa
            )
            cov = kappa**2 * np.linalg.inv(precision)
            exact = multivariate_normal(mean, cov).logpdf(np.asarray(c))
            assert abs(float(log_q) - exact) < 1e-10
            if kappa != 1.0:
                assert abs(float(log_q) + 4.0 * math.log(kappa) - exact) > 0.5


@pytest.mark.parametrize("threshold,cleared", [(None, []), (0, [0, 1, 2]), (128, [1])])
def test_run_units_releases_records_and_clears_after_logging(
    tmp_path, monkeypatch, threshold, cleared
):
    """Fails if cleanup precedes persistence/release, ignores RSS, or runs on resume.

    No existing test covers the completed-unit cache lifetime. Replace RSS
    and global cleanup operations here; the isolated identity probe below
    exercises real cache clearing without collecting other tests' programs.
    """
    import gc
    import weakref

    import equinox as eqx
    import jax

    tool = _load_tool()
    units = [{"id": str(i)} for i in range(3)]
    events, refs = [], []
    rss = iter([128, 129, 120])
    monkeypatch.setattr(tool, "_resident_memory_mb", lambda: next(rss), raising=False)
    monkeypatch.setattr(tool, "_cgroup_memory_bytes", lambda: None)

    class Record(dict):
        pass

    def execute(unit):
        assert all(ref() is None for ref in refs)
        record = Record(seconds=0.0, value=int(unit["id"]))
        refs.append(weakref.ref(record))
        return record

    def log(message):
        assert len(tool.read_records(tmp_path)) == len(refs)
        events.append((len(refs) - 1, "log"))

    for owner, name, label in (
        (eqx, "clear_caches", "eqx"),
        (jax, "clear_caches", "jax"),
        (gc, "collect", "gc"),
    ):

        def observe(*args, _label=label, **kwargs):
            assert all(ref() is None for ref in refs)
            events.append((len(refs) - 1, _label))

        monkeypatch.setattr(owner, name, observe)
    counts = tool.run_units(
        units, tmp_path, (0, 1), execute, log=log, clear_caches_above_mb=threshold
    )
    assert counts == {"planned": 3, "skipped": 0, "ran": 3}
    assert events == [
        (i, event)
        for i in range(3)
        for event in (["log", "eqx", "jax", "gc"] if i in cleared else ["log"])
    ]
    assert all(ref() is None for ref in refs)
    before = list(events)
    assert tool.run_units(
        units, tmp_path, (0, 1), execute, log=log, clear_caches_above_mb=threshold
    ) == {"planned": 3, "skipped": 3, "ran": 0}
    assert events == before


def test_runner_cleanup_is_outside_panel_and_exact_fingerprints():
    """Fails if runner orchestration enters either record fingerprint."""
    tool = _load_tool()
    spec = importlib.util.spec_from_file_location(
        "updater_exact_checks", TOOL.parent / "updater_exact_checks.py"
    )
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    for definitions in (tool._record_definitions(), exact._exact_definitions()):
        assert all(getattr(obj, "__name__", None) != "run_units" for obj in definitions)
    protocol = tool.load_protocol()
    before = tool.run_hash(protocol)
    exact_before = exact.exact_run_hash(protocol)
    tool.run_units = lambda *a, **kw: None
    exact.panel.run_units = tool.run_units
    assert tool.run_hash(protocol) == before
    assert exact.exact_run_hash(protocol) == exact_before


@pytest.mark.medium
def test_cleanup_preserves_written_unit_bytes(tmp_path):
    """Fails if clearing changes any physical reference/certification record bytes.

    Replaces the tiny square-closure probe with real UnitExecutor sampling,
    summaries and certification variants. Medium tier: six physical fits
    with forced recompilation exceed the five-second cold unit budget.
    Only wall-clock seconds are excluded, using a byte substitution.
    """
    import subprocess
    import sys

    # A fresh process prevents this test's real global cache clearing from
    # paying for compiled programs retained by unrelated earlier tests.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from test_updater_panel import _cache_identity_probe; "
            "_cache_identity_probe(Path(sys.argv[2]))",
            str(Path(__file__).parent),
            str(tmp_path),
        ],
        check=True,
    )


def _cache_identity_probe(tmp_path):
    """Physical sampling and real global cache clearing in a fresh process."""
    tool = _load_tool()
    protocol = tool.load_protocol()
    budgets = tool.protocol_budgets(protocol, physical_proposals=256)
    roles = {"reference_1", "cert_doubled_1", "cert_proposal_1"}
    units = [
        u
        for u in tool.plan_units(
            protocol,
            "development",
            ["reference", "certification"],
            mass_models=["physical_fixed"],
            regimes=["R3"],
            worlds=[0],
        )
        if u["role"] in roles and u["epoch_count"] == 2
    ]
    assert len(units) == 3
    written = []
    for policy, threshold in (("disabled", None), ("every", 0)):
        out = tmp_path / policy
        tool.run_units(
            units,
            out,
            (0, 1),
            tool.UnitExecutor(protocol, budgets, chunk=256),
            log=lambda m: None,
            clear_caches_above_mb=threshold,
        )
        records = tool.read_records(out)
        assert [r["proposals"] for r in records] == [256, 512, 256]
        assert [r["method"] for r in records] == [
            "scale_and_rotate_exact_highest_snr_anchor",
            "scale_and_rotate_exact_highest_snr_anchor",
            "scale_and_rotate_exact_lowest_snr_anchor",
        ]
        assert all(r["summary"]["kish_ess"] > 0 for r in records)
        lines = []
        for line in (out / "units-0of1.jsonl").read_bytes().splitlines(keepends=True):
            # Remove just the timing member from the written bytes. Preserve
            # every other byte, including the newline and JSON separators.
            import re

            lines.append(re.sub(rb'"seconds": [^,}]+, ', b"", line))
        written.append(b"".join(lines))
    assert written[0] == written[1]


def test_cleanup_marks_cold_configurations_and_tuning_excludes_them(
    tmp_path, monkeypatch
):
    """Fails if post-clear recompiles enter warm medians or stay cold on reuse.

    No existing tuning test covers the runner's cleanup-to-report boundary.
    Fake numerical work keeps the test below the unit budget; persistence,
    cleanup decisions and the tuning report are real.
    """
    import gc

    import equinox as eqx
    import jax

    spec = importlib.util.spec_from_file_location(
        "updater_tuning", TOOL.parent / "updater_tuning.py"
    )
    tuning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuning)
    tool = tuning.panel
    protocol = tool.load_protocol()
    units = tool.plan_units(
        protocol,
        "development",
        ["updater"],
        mass_models=["ti_window"],
        regimes=["R1"],
        worlds=[0, 1],
    )
    units = [u for u in units if u["role"] != "updater_doubled"]
    rss = iter([129, 0, 0, 0, 0, 0])
    monkeypatch.setattr(tool, "_resident_memory_mb", lambda: next(rss))
    for owner, name in ((eqx, "clear_caches"), (jax, "clear_caches"), (gc, "collect")):
        monkeypatch.setattr(owner, name, lambda: None)
    monkeypatch.setattr(tool, "_cgroup_memory_bytes", lambda: None, raising=False)
    seen = set()

    def execute(unit):
        key = (unit["mass_model"], unit.get("particles", 16))
        first = key not in seen
        seen.add(key)
        bins = {q: [0.25, 0.75] for q in ("lnP", "e", "future_pa")}
        return {
            "kind": "updater",
            "particles": key[1],
            "updates": [
                {
                    "includes_compile": first,
                    "seconds": 100
                    if unit["world"] == 0
                    else (1 if unit["role"] == "updater_a" else 3),
                    "status": "OK",
                    "rungs_used": 1,
                }
                for _ in range(3)
            ],
            "summaries": {str(k): {"bins": bins} for k in (2, 3)},
        }

    # Distinct model and particle configurations must each be cold once.
    extra = dict(
        units[0],
        role="updater_doubled",
        particles=32,
        id=units[0]["id"].replace("updater_a", "updater_doubled"),
    )
    physical = dict(
        extra,
        mass_model="physical_fixed",
        particles=16,
        id=extra["id"].replace("ti_window", "physical_fixed"),
    )
    tool.run_units(
        [units[0], extra, physical, *units[1:]],
        tmp_path / "config",
        (0, 1),
        execute,
        log=lambda m: None,
        clear_caches_above_mb=128,
    )
    records = tool.read_records(tmp_path / "config")
    assert [r.get("cache_cold", False) for r in records] == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert not records[3]["updates"][0]["includes_compile"]
    report = tuning.report(tmp_path)["config"]
    assert all(row["seconds_per_update_median"] == 2 for row in report.values())


@pytest.fixture
def cgroup_fs(tmp_path):
    """A private proc membership file and cgroup tree, never the host's files."""
    membership = tmp_path / "proc" / "self" / "cgroup"
    membership.parent.mkdir(parents=True)
    membership.write_text("4:cpu:/legacy\n0::/slice/shard\n")
    root = tmp_path / "sys" / "fs" / "cgroup"
    group = root / "slice" / "shard"
    group.mkdir(parents=True)
    (group / "memory.current").write_text("601\n")
    (group / "memory.max").write_text("1000\n")
    return membership, root, group


def test_cgroup_memory_parsing_and_fallbacks(cgroup_fs, monkeypatch):
    """Fails if path selection or absent limits break RSS fallback."""
    tool = _load_tool()
    membership, root, group = cgroup_fs
    monkeypatch.setattr(tool.sys, "platform", "linux")
    assert tool._cgroup_memory_bytes(membership, root) == (601, 1000)
    (group / "memory.max").write_text("max\n")
    assert tool._cgroup_memory_bytes(membership, root) is None
    (group / "memory.max").write_text("invalid\n")
    assert tool._cgroup_memory_bytes(membership, root) is None
    (group / "memory.max").unlink()
    assert tool._cgroup_memory_bytes(membership, root) is None
    membership.write_text("0::/\n")
    (root / "memory.current").write_text("600\n")
    (root / "memory.max").write_text("1000\n")
    assert tool._cgroup_memory_bytes(membership, root) == (600, 1000)
    membership.write_text("4:memory:/legacy\n")
    assert tool._cgroup_memory_bytes(membership, root) is None
    membership.write_text("0::/../../outside\n")
    assert tool._cgroup_memory_bytes(membership, root) is None
    membership.unlink()
    assert tool._cgroup_memory_bytes(membership, root) is None
    membership.write_text("0::/\n")
    monkeypatch.setattr(tool.sys, "platform", "darwin")
    assert tool._cgroup_memory_bytes(membership, root) is None


@pytest.mark.parametrize(
    "argv,clears",
    [
        ([], 1),
        (["--clear-caches-above-fraction", "0.7"], 0),
        (["--no-clear-caches"], 0),
    ],
)
def test_cgroup_pressure_clears_below_rss_threshold(
    tmp_path, monkeypatch, cgroup_fs, argv, clears
):
    """Fails if cgroup pressure is ignored, uses >=, or overrides disabled cleanup."""
    import gc

    import equinox as eqx
    import jax

    tool = _load_tool()
    membership, root, group = cgroup_fs
    monkeypatch.setattr(tool.sys, "platform", "linux")
    read_memory = tool._cgroup_memory_bytes
    monkeypatch.setattr(
        tool, "_cgroup_memory_bytes", lambda: read_memory(membership, root)
    )
    monkeypatch.setattr(tool, "_resident_memory_mb", lambda: 100)
    events = []
    monkeypatch.setattr(eqx, "clear_caches", lambda: events.append("eqx"))
    monkeypatch.setattr(jax, "clear_caches", lambda: events.append("jax"))
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))

    def execute(unit):
        (group / "memory.current").write_text("600" if unit["id"] == "0" else "601")
        return {"value": unit["id"]}

    parser = tool.argparse.ArgumentParser()
    tool._cache_arguments(parser)
    policy = vars(parser.parse_args(argv))
    tool.run_units(
        [{"id": "0"}, {"id": "1"}],
        tmp_path / "run",
        (0, 1),
        execute,
        log=lambda m: None,
        **policy,
    )
    assert events == ["eqx", "jax", "gc"] * clears
    assert [r["value"] for r in tool.read_records(tmp_path / "run")] == ["0", "1"]


@pytest.mark.parametrize("fraction", [0, 1, float("nan")])
def test_invalid_cache_fraction_is_refused_before_writing(tmp_path, fraction):
    """Fails if a missing headroom fraction permits work or writes output."""
    tool = _load_tool()
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="clear_caches_above_fraction"):
        tool.run_units(
            [],
            out,
            (0, 1),
            lambda unit: {},
            clear_caches_above_fraction=fraction,
        )
    assert not out.exists()
