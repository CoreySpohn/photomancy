"""Physical orbital elements read from the draws of a fitted ``Posterior``."""

import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve


def posterior_physical(posterior, key, n, *, orbit_path, row=0):
    """Draw ``n`` samples of a ``Posterior`` and read one orbit in physical units.

    Each draw builds its model through the posterior's ``Fit``; the orbits at
    ``orbit_path`` give the row's period, size, shape and orientation.

    Args:
        posterior: A :class:`~photomancy.infer.Posterior` of one slice.
        key: PRNG key.
        n: Number of samples.
        orbit_path: Model path of the batched orbits, for example
            ``"dynamics.orbits"``.
        row: Planet row of those orbits.

    Returns:
        Dict ``{name: (n,)}`` with ``T`` (days), ``a`` (AU), ``e``, ``cos_i``, and
        ``W`` and ``M0`` (radians, the mean anomaly at the orbit's epoch).
    """
    draws = posterior.sample(key, (n,))

    def one(x):
        orbits = resolve(posterior.fit.model_at(x), orbit_path)
        orbit = jax.tree.map(lambda leaf: leaf[row], orbits)
        classical = orbit.to_classical()
        return {
            "T": orbit.period_d,
            "a": classical["a_AU"],
            "e": classical["e"],
            "cos_i": jnp.cos(classical["i_rad"]),
            "W": classical["W_rad"],
            "M0": classical["M0_rad"],
        }

    return jax.vmap(one)(draws)
