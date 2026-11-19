"""Classical orbital elements read from the draws of a fitted ``Posterior``."""

import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve


def classical_elements(posterior, key, n, *, orbit_path, row=0):
    """Draw ``n`` samples of a ``Posterior`` and read one orbit's classical elements.

    Each draw builds its model through the posterior's ``Fit``; the orbits at
    ``orbit_path`` give the row's period, size, shape and orientation. The keys follow
    ``FrameOrbit.to_classical``.

    Args:
        posterior: A :class:`~photomancy.infer.Posterior` of one slice.
        key: PRNG key.
        n: Number of samples.
        orbit_path: Model path of the batched orbits, for example
            ``"dynamics.orbits"``.
        row: Planet row of those orbits.

    Returns:
        Dict ``{name: (n,)}`` with ``period_d``, ``a_AU``, ``e``, ``cos_i``, ``W_rad``
        and ``M0_rad`` (the mean anomaly at the orbit's epoch).
    """
    draws = posterior.sample(key, (n,))

    def one(x):
        orbits = resolve(posterior.fit.model_at(x), orbit_path)
        orbit = jax.tree.map(lambda leaf: leaf[row], orbits)
        classical = orbit.to_classical()
        return {
            "period_d": orbit.period_d,
            "a_AU": classical["a_AU"],
            "e": classical["e"],
            "cos_i": jnp.cos(classical["i_rad"]),
            "W_rad": classical["W_rad"],
            "M0_rad": classical["M0_rad"],
        }

    return jax.vmap(one)(draws)
