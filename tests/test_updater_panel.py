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


def _summary(tool, protocol, updater_ess):
    worlds = [
        {
            "updater_ess": ess,
            "reference_ess": 20000,
            "bin_masses": {q: MASSES for q in ("lnP", "e", "future_pa")},
        }
        for ess in updater_ess
    ]
    return {
        "groups": [
            {"regime": r, "mass_model": m, "epoch_count": k, "worlds": worlds}
            for r, m, k in tool.canonical_groups(protocol)
        ]
    }


def test_freeze_never_loosens_the_alias_gate():
    """Imprecise or partial development summaries never yield a looser gate.

    Fails if the freeze drops a group, summarizes ESS by its median (two worlds
    at ESS 30 among four at 5,000 need a delta2 above the cap), or freezes a
    tolerance whose power lower bound is below the floor.
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    alias = protocol["criteria"]["alias_accuracy"]

    partial = _summary(tool, protocol, [5000] * 6)
    partial["groups"] = partial["groups"][:1]
    with pytest.raises(ValueError, match="missing groups"):
        tool.freeze_alias(protocol, partial, panels=24)

    mixed = tool.freeze_alias(
        protocol, _summary(tool, protocol, [30, 30, 5000, 5000, 5000, 5000]), panels=24
    )
    assert mixed["status"] == "incomplete"
    assert mixed["delta2"] > alias["delta2_max"]

    tight = tool.freeze_alias(protocol, _summary(tool, protocol, [5000] * 6), panels=24)
    assert tight["status"] == "frozen", tight["reasons"]
    assert tight["delta2"] == alias["delta2"]
    assert all(g["power_lower_95"] >= alias["min_power"] for g in tight["groups"])
