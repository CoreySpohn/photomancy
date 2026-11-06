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
