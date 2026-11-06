"""Tests for the updater acceptance panel driver (tools/updater_panel.py)."""

import importlib.util
import math
from pathlib import Path

TOOL = Path(__file__).parents[1] / "tools" / "updater_panel.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("updater_panel", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_critical_values_match_protocol(capsys):
    """The dry run's critical values equal the thresholds the protocol states.

    Fails if the protocol's stated thresholds drift from their formulas (a
    re-sized margin or family without recomputed counts), if a binomial tail is
    off by one, or if the DKW bound is computed from a rounded literal.
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
    assert values["schedule_critical_count"] == 32
    alpha, n = 0.05 / 64, 400
    assert values["dkw_epsilon"] == math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    assert values["dkw_epsilon_5dp"] == 0.09904


def test_keys_are_pinned_and_streams_distinct():
    """Every declared key coordinate gives its own key, whatever the JAX default.

    Fails if two draws (stage, mass model, regime, world, purpose, update) share
    a key, if the development and panel streams overlap, or if the keys follow
    the ambient default generator instead of the pinned threefry2x32.
    """
    import jax
    import numpy as np

    tool = _load_tool()
    protocol = tool.load_protocol()
    prng = protocol["prng"]
    purposes = jax.numpy.asarray(list(prng["purposes"].values()))
    updates = jax.numpy.arange(len(protocol["panel"]["updates"]))
    worlds_by_stage = {s: tool.stage_worlds(protocol, s) for s in prng["stages"]}

    def per_world(block, w):
        world = jax.random.fold_in(block, w)
        return jax.vmap(
            lambda p: jax.vmap(
                lambda u: jax.random.fold_in(jax.random.fold_in(world, p), u)
            )(updates)
        )(purposes)

    rows = []
    with jax.enable_x64(True):
        for stage, n in worlds_by_stage.items():
            for m_index, mass_model in enumerate(protocol["panel"]["mass_models"]):
                for r_index in range(len(protocol["regimes"])):
                    block = tool.root_key(protocol)
                    for v in (prng["stages"][stage], m_index, r_index):
                        block = jax.random.fold_in(block, v)
                    keys = jax.vmap(lambda w, b=block: per_world(b, w))(
                        jax.numpy.arange(n)
                    )
                    rows.append(np.asarray(jax.random.key_data(keys)).reshape(-1, 2))
                    ref = tool.draw_key(
                        protocol, stage, mass_model, r_index, n - 1, "pit", 2
                    )
                    np.testing.assert_array_equal(
                        jax.random.key_data(ref), jax.random.key_data(keys[-1, -1, -1])
                    )
    data = np.concatenate(rows)
    assert len(np.unique(data, axis=0)) == len(data)

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


def test_schedule_cutoff_large_ess_limit():
    """At large ESS the per-world null reproduces the stated cutoff of 32.

    Fails if the simulated per-world null or the Poisson-binomial tail is off
    (the large-ESS limit is the Gaussian rate 0.0455 at 400 worlds).
    """
    import numpy as np

    tool = _load_tool()
    protocol = tool.load_protocol()
    sched = protocol["criteria"]["schedule_stability"]
    rng = np.random.default_rng(0)
    p = np.full(400, 0.5)
    q = tool.schedule_null_probabilities(
        p, p, 1e6, 1e6, sched["margin_k"], sched["null_simulations_per_world"], rng
    )
    assert abs(q.mean() - 0.0455) < 0.002
    cutoff = tool.schedule_cutoff(np.full(400, q.mean()), 0.05 / 32)
    assert cutoff == protocol["stated"]["schedule_critical_count"]


def test_freeze_never_loosens_the_alias_gate():
    """An imprecise development summary yields incomplete, not a larger delta2.

    Fails if the frozen alias tolerance can exceed its declared maximum or be
    accepted with power below the floor (updater ESS 30 needs about 5e-3).
    """
    tool = _load_tool()
    protocol = tool.load_protocol()
    masses = [[0.125] * 8] * 6
    group = {
        "regime": "R1",
        "mass_model": "ti_window",
        "epoch_count": 2,
        "reference_ess": 20000,
        "reference_bin_masses": {"lnP": masses, "e": masses, "future_pa": masses},
    }
    loose = tool.freeze_alias(
        protocol, {"groups": [dict(group, updater_ess=30)]}, panels=40
    )
    assert loose["status"] == "incomplete"
    assert loose["delta2"] > protocol["criteria"]["alias_accuracy"]["delta2_max"]
    tight = tool.freeze_alias(
        protocol, {"groups": [dict(group, updater_ess=5000)]}, panels=40
    )
    assert tight["status"] == "frozen", tight["reasons"]
    assert tight["delta2"] == protocol["criteria"]["alias_accuracy"]["delta2"]
