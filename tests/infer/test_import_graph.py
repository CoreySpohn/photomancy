"""``photomancy.infer`` stays free of the astronomy stack; deleted names stay gone.

Every module under ``photomancy.infer`` is parsed, so imports inside functions
count too. The subpackage may import the numerical stack and its own modules,
never an astronomy library or the rest of photomancy.
"""

import ast
import re
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


# Names removed from the package with no replacement under the same name.
DELETED = (
    "OrbitUpdater",
    "UpdaterBelief",
    "UpdaterRecord",
    "UpdaterDiagnostics",
    "ConjugateGaussianTarget",
    "FixedMassTarget",
    "jitter_m_per_s",
    "migrate_sky_convention",
    "photomancy.system_density",
    "PhysicalPriorTarget",
    "physical_prior_target",
    "RefitCloud",
    "loglike_report",
    "report_log_weights",
    "draw_report_scale",
    "ti_to_elements",
    "collapsed_refit",
    "scale_rotate_draw",
    "scale_rotate_refit",
    "importance_refit",
    "prior_cloud",
    "UpdaterStatus",
    "AbstractUpdateTarget",
    "ThieleInnesTarget",
    "AbstractConditioner",
    "chart_scale_rotate",
    "scored_orbit",
    "ScaleAndRotate",
    "ofti",
    "MU_SUN",
    "TI_PARAM_NAMES",
    "TIBasisModel",
    "lambert_families",
    "chart_in_support",
    "chart_log_prior",
    "lambert_depth_fit",
    "admissible_region_fit",
    "ParamBounds",
    "chart_box_bounds",
    "AbstractGridStrategy",
    "AdaptiveImportanceSampler",
    "batched_loglike",
    "grid_search",
    "OrbitNestedPosterior",
    "OrbitPriorTransform",
    "build_chart_nested_model",
    "orbit_tempered_smc",
    "orbit_nested_sampling",
    "roberts_sequence",
    "map_laplace_fit",
    "map_laplace_mixture_fit",
    "ORBIT_ROWS_CHART",
    "orbit_chart_log_volume",
    "build_chart_logdensity",
    "ScalarCoordinate",
    "SystemLogDensity",
    "find_init",
    "TI_CHART_PARAM_NAMES",
    "TIFitResult",
    "thiele_innes_fit",
    "thiele_innes_grid",
    "thiele_innes_grid_search",
    "fixed_mass_orbits",
    "fixed_mass_coordinates",
    "astrometric_loss",
    "make_astrometric_evaluator",
    "RV_AU_PER_DAY_TO_M_PER_S",
    "LOG_HAAR",
    "orbit_log_prior",
    "FactorizedOrbitPrior",
    "PopulationIntensity",
    "PopulationPrior",
    "mode_observables",
    "sample_physical",
    "orbits_from_samples",
    "mode_scalars",
    "mode_summary",
    "photomancy.orbit.refit",
    "photomancy.orbit.updater",
    "photomancy.orbit.ofti",
    "photomancy.orbit.few_epoch",
    "photomancy.orbit.grid_search",
    "photomancy.orbit.nested",
    "photomancy.orbit.quasi_random",
    "photomancy.orbit.laplace",
    "photomancy.orbit.inference",
    "photomancy.orbit.system_density",
    "photomancy.orbit.init",
    "photomancy.orbit.thiele_innes",
    "photomancy.orbit.projected",
    "photomancy.orbit.units",
    "photomancy.priors",
    "ChartOrbitProblem",
    "Updater",
    "NullData",
    "loglike_null",
    "ImagingData",
    "loglike_imaging",
    "area_AU2",
    "RelativeAstromData",
    "StellarAstromData",
    "RVData",
    "PMAnomalyData",
    "ReportRows",
    "ReportLaw",
    "ReportTable",
    "TabulatedReportPredictor",
)


def test_no_deleted_name_is_left_in_the_package():
    """Fails if a deleted class, keyword, function or module path reappears in src.

    Every ``.py`` file of the package is searched for each name as a whole word, in
    code, docstrings and comments alike, so a stale cross-reference fails too.
    """
    package = ROOT.parent
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, DELETED)) + r")\b")
    found = {
        str(path.relative_to(package)): sorted(set(pattern.findall(path.read_text())))
        for path in sorted(package.rglob("*.py"))
        if pattern.search(path.read_text())
    }
    assert not found, found


def _private_reaches(path):
    """Imports of ``path`` that are legacy or private to another subpackage."""
    found = set()
    for name in _imported_modules(path):
        parts = name.split(".")
        if parts[0] != "photomancy" or len(parts) < 2 or parts[1] == "orbit":
            continue
        legacy = name.startswith("photomancy.infer.legacy_posterior")
        if legacy or any(part.startswith("_") for part in parts[1:]):
            found.add(name)
    return found


def test_orbit_reaches_no_private_or_legacy_name_of_another_subpackage():
    """Fails if ``photomancy.orbit`` imports a private module or name of a sibling.

    A leading underscore anywhere in the imported path, or the legacy posterior
    module, counts.
    """
    orbit = ROOT.parent / "orbit"
    found = {
        path.name: reaches
        for path in sorted(orbit.rglob("*.py"))
        if (reaches := _private_reaches(path))
    }
    assert not found, found
