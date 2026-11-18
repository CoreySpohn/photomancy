"""Static checks on the documentation notebooks."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "docs/examples/orbit_fitting.ipynb"


def test_orbit_fitting_passes_the_inclination_angle_to_orbix():
    """Fails if the notebook feeds orbix a cosine or keeps stored outputs.

    FrameOrbit.from_classical takes the inclination in radians, so the notebook
    passes the angle itself, never its cosine.
    """
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "o.e, o.inc, o.Omega, o.omega, o.M0" in source
    assert "jnp.cos(o.inc)" not in source
    assert all(c.get("outputs", []) == [] for c in cells if c["cell_type"] == "code")
