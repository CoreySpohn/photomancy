"""Import mechanics for photomancy.viz: lazy exports, clean base install."""

import subprocess
import sys

import pytest

BLOCK_EYEPIECE = "import sys; sys.modules['eyepiece'] = None; "


def _run(code):
    """Run a code snippet in a fresh interpreter and return the result."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_photomancy_imports_without_eyepiece():
    """The base install imports clean with eyepiece blocked."""
    result = _run(BLOCK_EYEPIECE + "import photomancy")
    assert result.returncode == 0, result.stderr


def test_labels_work_without_eyepiece():
    """PARAM_LABELS is plain data and needs no plotting stack."""
    result = _run(
        BLOCK_EYEPIECE
        + "import photomancy.viz; "
        + "assert 'T (days)' == photomancy.viz.PARAM_LABELS['T']; "
        + "assert photomancy.viz.default_corner_params({'T': [1.0], 'e': [0.1]})"
    )
    assert result.returncode == 0, result.stderr


def test_viz_function_without_eyepiece_names_the_extra():
    """Touching a plot function without eyepiece names the viz extra."""
    result = _run(
        BLOCK_EYEPIECE
        + "import photomancy.viz\n"
        + "try:\n"
        + "    photomancy.viz.plot_corner\n"
        + "except ImportError as err:\n"
        + "    assert 'photomancy[viz]' in str(err), str(err)\n"
        + "else:\n"
        + "    raise SystemExit('expected ImportError')\n"
    )
    assert result.returncode == 0, result.stderr


def test_importing_viz_package_imports_no_plotting_stack():
    """Import photomancy.viz pulls neither eyepiece nor matplotlib."""
    result = _run(
        "import sys; import photomancy.viz; "
        "assert 'eyepiece' not in sys.modules, 'eyepiece imported'; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    )
    assert result.returncode == 0, result.stderr


def test_dir_lists_lazy_exports():
    """dir(photomancy.viz) advertises the lazily provided names."""
    import photomancy.viz

    listed = dir(photomancy.viz)
    for name in ("plot_corner", "plot_eig", "plot_rv", "PARAM_LABELS"):
        assert name in listed


def test_unknown_attribute_raises_attribute_error():
    """A name outside the lazy tables raises AttributeError."""
    import photomancy.viz

    with pytest.raises(AttributeError, match="no attribute"):
        _ = photomancy.viz.plot_nonexistent
