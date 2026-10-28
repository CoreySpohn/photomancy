"""Static checks on the documentation notebooks."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "docs/examples/orbit_fitting.ipynb"


def test_orbit_fitting_passes_cos_i_not_the_inclination():
    """predict_relative_astrometry takes cos_i, so the angle must go through cos."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "ts, a, o.e, jnp.cos(o.inc), o.Omega" in source
    assert "o.e, o.inc," not in source
    assert all(c.get("outputs", []) == [] for c in cells if c["cell_type"] == "code")
