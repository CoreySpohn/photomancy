"""The declared floors on sibling libraries keep old and new releases apart.

photomancy 0.4.0 reads orbix and skyscapes states in the (east, north, away)
sky chart, so it must not install beside releases that use the old chart. The
pins live in ``pyproject.toml``; this test reads them without installing anything.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# First release of each sibling that uses the (east, north, away) sky chart.
SKY_FLOORS = {
    # Next releases (not yet on PyPI; see .github/sibling-pins.txt).
    "orbix": Version("6.0.0"),  # sky design and Thiele-Innes block APIs
    "skyscapes": Version("6.0.0"),  # scene sky frame and local-zodi models
    "coronagraphoto": Version("7.1.0"),  # renders on optixstuff 7 types; observe_data
}


def _requirements():
    """Yield (group, Requirement) for every declared dependency."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    for text in project["dependencies"]:
        yield "dependencies", Requirement(text)
    for extra, texts in project.get("optional-dependencies", {}).items():
        for text in texts:
            yield f"extra:{extra}", Requirement(text)


def _floor(requirement):
    lowers = [
        Version(s.version) for s in requirement.specifier if s.operator in (">=", "==")
    ]
    return max(lowers) if lowers else None


def test_runtime_floors_are_declared():
    """The runtime dependencies on the sibling libraries carry the sky-chart floors."""
    floors = {
        req.name: _floor(req)
        for group, req in _requirements()
        if group == "dependencies" and req.name in SKY_FLOORS
    }
    assert floors == {
        "coronagraphoto": SKY_FLOORS["coronagraphoto"],
        "orbix": SKY_FLOORS["orbix"],
        "skyscapes": SKY_FLOORS["skyscapes"],
    }


def test_every_sibling_pin_is_at_or_above_the_sky_chart_release():
    """No dependency, extra or docs group names an older sibling than the train."""
    seen = set()
    for group, req in _requirements():
        if req.name in SKY_FLOORS:
            seen.add(req.name)
            floor = _floor(req)
            assert floor is not None, f"{req.name} is unpinned in {group}"
            assert floor >= SKY_FLOORS[req.name], f"{req.name} floor in {group}"
    assert {"orbix", "skyscapes", "coronagraphoto"} <= seen


def test_python_floor_matches_coronagraphoto():
    """The package must not admit an interpreter coronagraphoto cannot run on."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    spec = SpecifierSet(project["requires-python"])
    assert "3.11" not in spec, f"requires-python {spec} admits Python 3.11"
    assert "3.12" in spec
