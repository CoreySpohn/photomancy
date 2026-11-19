"""``photomancy.infer`` stays free of the astronomy stack.

Every module under ``photomancy.infer`` is parsed, so imports inside functions
count too. The subpackage may import the numerical stack and its own modules,
never an astronomy library or the rest of photomancy.
"""

import ast
import sys
from pathlib import Path

import photomancy.infer

FORBIDDEN = {
    "orbix",
    "skyscapes",
    "coronagraphoto",
    "optixstuff",
    "yippy",
    "physicaloptix",
    "coronalyze",
}
ROOT = Path(photomancy.infer.__file__).parent


def _imported_modules(path):
    """Absolute names of every module ``path`` imports, at any depth."""
    relative = path.relative_to(ROOT.parent.parent).with_suffix("")
    package = list(relative.parts[:-1])
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            # ``from package import name`` may bind a submodule.
            names += [module, *(f"{module}.{alias.name}" for alias in node.names)]
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", ""))
            in ("import_module", "import_extra", "__import__")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names.append(node.args[0].value)
    return names


def _violations(path):
    found = []
    for name in _imported_modules(path):
        root = name.split(".")[0]
        outside = root == "photomancy" and not (
            name == "photomancy.infer" or name.startswith("photomancy.infer.")
        )
        if root in FORBIDDEN or outside:
            found.append(name)
    return found


def test_infer_imports_no_astronomy_and_no_other_photomancy_module():
    """No module under ``photomancy.infer`` reaches outside the subpackage."""
    modules = sorted(ROOT.rglob("*.py"))
    assert len(modules) > 10
    violations = {
        str(path.relative_to(ROOT)): found
        for path in modules
        if (found := _violations(path))
    }
    assert not violations, violations


def test_scan_catches_function_level_and_relative_imports(tmp_path, monkeypatch):
    """The scan sees a deferred import, a relative import and a bare package."""
    package = tmp_path / "src" / "photomancy" / "infer"
    package.mkdir(parents=True)
    module = package / "leak.py"
    module.write_text(
        "from . import model\n"
        "from .. import priors\n"
        "from photomancy import detection\n"
        "def run():\n"
        "    import orbix.system\n"
        "    from skyscapes.scene import Scene\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", package)
    assert _violations(module) == [
        "photomancy",
        "photomancy.priors",
        "photomancy",
        "photomancy.detection",
        "orbix.system",
        "skyscapes.scene",
        "skyscapes.scene.Scene",
    ]
