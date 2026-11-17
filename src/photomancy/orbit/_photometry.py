"""Map the inferred photometric area into a physical reflector."""

import jax
import jax.numpy as jnp
from hwoutils.constants import Rearth2AU


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
