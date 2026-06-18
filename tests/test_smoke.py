"""Smoke tests: the package imports and reports a version."""

import photomancy


def test_import_and_version():
    """Photomancy imports and exposes a non-empty version string."""
    assert isinstance(photomancy.__version__, str)
    assert photomancy.__version__
