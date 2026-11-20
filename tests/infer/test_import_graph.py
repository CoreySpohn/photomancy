"""``photomancy.infer`` stays free of the astronomy stack; deleted names stay gone.

Every module under ``photomancy.infer`` is parsed, so imports inside functions
count too. The subpackage may import the numerical stack and its own modules,
never an astronomy library or the rest of photomancy.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

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
    "SMC",
    "SMCInfo",
    "SMCBackend",
    "photomancy.infer.backends.smc",
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
    "ThetaPrior",
    "theta_idx",
    "theta_periods",
    "c_idx",
    "n_constants",
    "n_proposals",
    "posterior_physical",
    "n_round",
    "kish_per_round",
    "TermLogDensity",
    "LegacyBackend",
    "laplace_fit",
    "laplace_covariance",
    "LaplaceBackend",
    "LaplaceMixtureBackend",
    "PathfinderBackend",
    "PathfinderMixtureBackend",
    "NUTSBackend",
    "MCLMCBackend",
    "JaxnsBackend",
    "jaxns_prior",
    "build_scene_nested_model",
    "AbstractLogDensity",
    "AbstractPosterior",
    "GaussianPosterior",
    "MixturePosterior",
    "PointPosterior",
    "SamplePosterior",
    "IndependentPrior",
    "build_gaussian_fit",
    "build_scene_logdensity",
    "project_samples",
    "sample_capped",
    "photomancy.infer.legacy_posterior",
    "photomancy.infer.model",
    "photomancy.infer.priors",
    "photomancy.infer.posterior_utils",
    # The time field of the measurement data classes is ``time_d``. The old field
    # names are ordinary prose words, so they are guarded as an attribute read and as
    # a keyword.
    ".times",
    ".epochs",
    "times=",
    "epochs=",
    "det_epochs",
    "null_epochs",
    # Renamed public names: the old spellings are not importable and not mentioned.
    "LeafConditional",
    "AbstractConditionalTerm",
    "leaf_conditional",
    "LinearForm",
    "linear_form",
    "AbstractLinearTerm",
    "evaluate_orbit_candidates",
    "evaluate_candidates",
    "FitStage",
    "n_tune",
    "gauss_legendre",
    "geometric_eig",
    "alias_breaking_eig",
    "alias_breaking_eig_from_covariances",
    "detectability_eig",
    "detection_class_eig",
    "detection_channel_eig",
    "class_eig",
    "planet_id",
    "_flat_layout",
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
    """Imports of ``path`` that are private to another subpackage."""
    found = set()
    for name in _imported_modules(path):
        parts = name.split(".")
        if parts[0] != "photomancy" or len(parts) < 2 or parts[1] == "orbit":
            continue
        if any(part.startswith("_") for part in parts[1:]):
            found.add(name)
    return found


def test_orbit_reaches_no_private_or_legacy_name_of_another_subpackage():
    """Fails if ``photomancy.orbit`` imports a private module or name of a sibling.

    A leading underscore anywhere in the imported path counts.
    """
    orbit = ROOT.parent / "orbit"
    found = {
        path.name: reaches
        for path in sorted(orbit.rglob("*.py"))
        if (reaches := _private_reaches(path))
    }
    assert not found, found


def test_measurement_data_name_their_time_time_d_and_hide_the_padding_sizes():
    """Fails if a data class keeps a ``times`` or ``epochs`` field, or a capacity leaks.

    Every ``*Data`` class of ``photomancy.measurements`` with a time field names it
    ``time_d``; the padding capacities (``MAX_RV`` and the rest) are the layout of the
    caller and are not importable from the package.
    """
    import dataclasses

    import photomancy.measurements as measurements

    data = [
        getattr(measurements, name)
        for name in measurements.__all__
        if name.endswith("Data")
        and dataclasses.is_dataclass(getattr(measurements, name))
    ]
    assert len(data) >= 6
    for cls in data:
        fields = {field.name for field in dataclasses.fields(cls)}
        assert not fields & {"times", "epochs"}, cls.__name__
    for name in (
        "MAX_CC_PTS",
        "MAX_IMG",
        "MAX_REL_ASTROM",
        "MAX_RV",
        "MAX_STELLAR_ASTROM",
    ):
        assert not hasattr(measurements, name), name


def test_update_info_is_exported_from_infer():
    """Fails if ``UpdateInfo`` leaves ``photomancy.infer`` or its ``__all__``."""
    assert "UpdateInfo" in photomancy.infer.__all__
    assert photomancy.infer.UpdateInfo.__name__ == "UpdateInfo"


PUBLIC_NAMES = (
    "MAP",
    "MCLMC",
    "NUTS",
    "AbstractBackend",
    "AbstractDensityProposal",
    "AbstractLinearGaussianTerm",
    "AbstractProposal",
    "AbstractProposingTerm",
    "AbstractTerm",
    "AdaptiveImportance",
    "DefensiveProposal",
    "EIGEstimate",
    "EarlyStopping",
    "Fit",
    "Forecast",
    "ForwardTerm",
    "Grid",
    "Importance",
    "ImportanceInfo",
    "Laplace",
    "LaplaceInfo",
    "LeafProposal",
    "LinearGaussian",
    "LinearizedEIG",
    "MAPInfo",
    "MAPStage",
    "MCLMCInfo",
    "MixtureProposal",
    "NUTSInfo",
    "NestedInfo",
    "NestedSampling",
    "Pathfinder",
    "PathfinderInfo",
    "Posterior",
    "PriorProposal",
    "ReweightInfo",
    "UpdateInfo",
    "cluster_to_mixture",
    "eig_contrastive",
    "eig_linearized",
    "eig_nested",
    "gauss_legendre_rule",
    "group_labels",
    "importance_evidence",
    "khat_threshold",
    "kish_ess",
    "kish_temperature",
    "mc_se",
    "midpoint_rule",
    "normalize_log_weights",
    "pareto_khat",
    "ratio_mc_se",
    "ravel_samples",
    "rescaled_transform",
    "run",
    "sigma_from_log_bayes_factor",
    "update",
    "weighted_cov",
    "weighted_mean",
    "weighted_quantile",
)


def test_infer_all_is_the_pinned_public_surface():
    """Fails if a name enters or leaves ``photomancy.infer.__all__`` unannounced.

    ``infer.__all__`` holds every name needed to state a fit, run it and read the
    result; each name resolves, and ``backends`` does not export the staged engine.
    """
    assert sorted(photomancy.infer.__all__) == sorted(PUBLIC_NAMES)
    for name in PUBLIC_NAMES:
        assert hasattr(photomancy.infer, name), name
    from photomancy.infer import backends

    assert "map_fit" not in backends.__all__
    assert "MAPResult" not in backends.__all__


def test_the_two_public_submodules_declare_an_all():
    """Fails if ``infer.eig`` or ``infer.likelihoods`` loses its ``__all__``.

    Every listed name resolves, every public (non-underscore) function or class
    defined there is listed, and every estimator of ``eig`` follows ``eig_<method>``.
    """
    import inspect

    from photomancy.infer import eig, likelihoods

    for module in (eig, likelihoods):
        listed = set(module.__all__)
        assert all(hasattr(module, name) for name in listed)
        defined = {
            name
            for name, obj in vars(module).items()
            if (inspect.isfunction(obj) or inspect.isclass(obj))
            and obj.__module__ == module.__name__
            and not name.startswith("_")
        }
        assert defined == listed, defined ^ listed
    functions = [name for name in eig.__all__ if inspect.isfunction(getattr(eig, name))]
    assert functions and all(name.startswith("eig_") for name in functions)


def test_docs_import_no_private_module_of_infer():
    """Fails if a page, notebook or the README imports below ``photomancy.infer``.

    Documentation imports from ``photomancy.infer`` itself, except the two public
    submodules ``eig`` and ``likelihoods``; ``photomancy.infer.backends`` and the
    private modules never appear in an import or a cross-reference.
    """
    root = ROOT.parent.parent.parent
    pages = [
        *root.joinpath("docs").rglob("*.md"),
        *root.joinpath("docs").rglob("*.ipynb"),
        root / "README.md",
    ]
    pages = [page for page in pages if page.is_file()]
    if not pages:
        pytest.skip("the documentation is not part of this installation")
    pattern = re.compile(r"photomancy\.infer\.(\w+)")
    allowed = {"eig", "likelihoods"}
    found = {}
    for page in pages:
        text = page.read_text()
        bad = {
            name
            for name in pattern.findall(text)
            if name not in allowed and name not in set(photomancy.infer.__all__)
        }
        if bad:
            found[str(page.relative_to(root))] = sorted(bad)
    assert not found, found
