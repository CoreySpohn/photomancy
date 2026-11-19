"""The three golden systems of the density golden, with their measurement terms."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from hwoutils.model_paths import assign
from infer.test_fit import (
    EARTH_MASS_MSUN,
    GOLDEN,
    TIMES,
    _block,
    _coords,
    _photometry,
    _pma,
    _relative,
    _rv,
    _scene,
    _stellar,
)

from photomancy.measurements import DMagTerm, ImagingData, NullData


def imaging_data():
    """Two detections and two nulls on a flat 24 mag contrast curve."""
    sep = jnp.tile(jnp.linspace(0.02, 0.5, 20), (4, 1))
    curve = jnp.full((4, 20), 24.0)
    return ImagingData.pad(
        epochs=TIMES,
        sep_grid=sep,
        dmag0_grid=curve,
        is_detected=[True, False, True, False],
        dmag_obs=[22.0, 0.0, 21.5, 0.0],
        dmag_err=[0.3, 1.0, 0.3, 1.0],
        max_n=4,
        max_pts=20,
    ), NullData.pad(epochs=TIMES, sep_grid=sep, dmag0_grid=curve, max_n=4, max_pts=20)


def _g2_entry(i):
    mass = jnp.exp(GOLDEN["g2"]["ln_mass_Mearth"][i]) * EARTH_MASS_MSUN
    return {
        "dynamics.planets.mass_Msun": jnp.array([mass]),
        "dynamics.orbits": _coords(GOLDEN["g2"]["points"][i]),
    }


def _g3_entry(i):
    golden = GOLDEN["g3"]
    return {
        "planets[0].Rp_Rearth": jnp.exp(golden["ln_radius_AU"][i])
        / Rearth2AU
        * jnp.ones(1),
        "planets[0].physical_model.Ag": jnp.exp(golden["ln_albedo"][i]) * jnp.ones(1),
        "dynamics.orbits": _coords(golden["points"][i]),
    }


def systems():
    """Yield ``(name, scene, entry(i), n_points, terms)`` for g1, g2 and g3."""
    yield (
        "g1",
        _scene(),
        lambda i: {"dynamics.orbits": _coords(GOLDEN["g1"]["points"][i])},
        (_relative(),),
    )
    yield (
        "g2",
        _scene(planet_mass_Msun=0.1),
        _g2_entry,
        (_stellar(), _rv(), _pma()),
    )
    yield (
        "g3",
        _scene(),
        _g3_entry,
        (_relative(), _photometry(), DMagTerm(imaging_data()[0])),
    )


def model_at(scene, entry):
    """The scene with a golden point written through its leaves and orbit block."""
    model = scene
    for path, value in entry.items():
        if path != "dynamics.orbits":
            model = assign(model, path, value)
    block = _block()
    return assign(
        model,
        "dynamics.orbits",
        block.forward(entry["dynamics.orbits"], model)["dynamics.orbits"],
    )
