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
    """The exact-check critical value is t_15 at 1 - 0.01 / (2 m), m = 56.

    Fails if the compared-value count, the repeat count or the Student t
    quantile drift from the protocol (a normal quantile, 3.75, would reject a
    correct sampler far more often than 1 percent with 16 repeats).
    """
    from scipy import stats

    tool = _load_tool()
    protocol = tool.load_protocol()
    ex = protocol["reference"]["exact_checks"]
    m = ex["cases"] * (ex["bins_per_case"] + ex["interval_end_points_per_case"])
    assert m == protocol["stated"]["exact_check_compared_values"] == 56
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
        return 2 if "doubled" in u["role"] else 1

    upd = sum(cost(u) for u in units if u["role"] in tool.ALL_UPDATER_ROLES)
    ref = sum(cost(u) for u in units if u["role"] not in tool.ALL_UPDATER_ROLES)
    n_mass = len(protocol["panel"]["mass_models"])
    n_updates = len(protocol["panel"]["updates"])
    assert upd * n_updates == n_mass * tally["development_update_equivalents_per_pair"]
    assert ref == n_mass * tally["development_reference_fit_equivalents_per_pair"]
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
                "proposals": budgets[unit["mass_model"]] * doubled,
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
    """A run directory missing one planned role is incomplete, not frozen.

    Every planned development unit is present at the protocol budgets under
    the current run hash except the 304 ``updater_doubled`` runs, which no
    freeze statistic reads. Fails if the freeze grades a directory that does
    not hold the complete canonical development plan.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    roles = {r for part in tool.PARTS.values() for r in part} - {"updater_doubled"}
    units = _fake_records(
        tool, protocol, roles, set(range(38)), run_hash=tool.run_hash(protocol)
    )
    path = tmp_path / "units-0of1.jsonl"
    path.write_text("".join(tool.json.dumps(r) + "\n" for r in units.values()))
    report = tool.freeze_report(protocol, tmp_path)
    assert report["status"] == "incomplete"
    reason = next(r for r in report["reasons"] if "planned development" in r)
    assert reason.startswith("304 of ") and "updater_doubled" in reason
    assert "alias" not in report


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
