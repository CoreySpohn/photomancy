"""Achromatic Lambertian predictions for single-planet orbit measurements."""

import jax
import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import contrast_to_delta_mag
from orbix.system import System


def lambertian_from_area(area_AU2):
    """Map Lambda = Ag Rp^2 to a unit-albedo model and radius in Earth radii.

    Reflected light identifies only the area product. Fix Ag=1 and put
    the sampled scale into radius; both returned arrays have shape (K,).
    """
    # Delay scene imports, keeping their module constants concrete under first JIT.
    with jax.ensure_compile_time_eval():
        from skyscapes.physical_model import LambertianPhysicalModel

    area = jnp.atleast_1d(area_AU2)
    return LambertianPhysicalModel(Ag=jnp.ones_like(area)), jnp.sqrt(area) / Rearth2AU


def lambertian_dmag(phase_rad, distance_AU, model, radius_Rearth):
    """Convert a Lambertian reflector's contrast to dMag at the orbit-fit floor."""
    contrast = model.contrast(
        phase_rad[None, :], distance_AU[None, :], None, radius_Rearth
    )[0]
    return contrast_to_delta_mag(jnp.maximum(contrast, 1e-300))


def system_separation_dmag(system, time_d, *, area_AU2=None):
    """Predict separation (arcsec) and dMag at days from a one-planet epoch.

    A scene supplies its Lambertian reflector and radius. An explicit
    geometric-albedo times radius squared (AU^2) supplies the reflector
    for a bare dynamical system or a known photometric scale.
    """
    bare = isinstance(system, System)
    count = system.orbits.a_AU.size if bare else system.n_planets
    if count != 1:
        raise ValueError(
            "Multi-planet null and imaging require a detection model per planet"
        )
    if area_AU2 is None:
        if bare:
            raise ValueError(
                "A dynamical system needs area_AU2 for Lambertian photometry"
            )
        with jax.ensure_compile_time_eval():
            from skyscapes.physical_model import LambertianPhysicalModel

        planet = system.planets[0]
        model = planet.physical_model
        if not isinstance(model, LambertianPhysicalModel):
            raise ValueError(
                "Imaging and null orbit terms require a Lambertian reflector"
            )
        radius_Rearth = planet.Rp_Rearth
    else:
        if jnp.size(area_AU2) != 1:
            raise ValueError("area_AU2 must specify one planet's photometric area")
        model, radius_Rearth = lambertian_from_area(jnp.reshape(area_AU2, (1,)))
    if not bare:
        radius_Rearth = jnp.where(system.valid, radius_Rearth, 0.0)
    ephemeris = system.solve(time_d) if bare else system.ephemeris(time_d)
    return ephemeris.separation_arcsec(1, 0), lambertian_dmag(
        ephemeris.phase_angle_rad(1, 0),
        ephemeris.distance_AU(1, 0),
        model,
        radius_Rearth,
    )
