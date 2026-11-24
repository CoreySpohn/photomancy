"""Classical orbital elements read from the draws of a fitted ``Posterior``."""

import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve


def classical_elements(posterior, key, n, *, orbit_path, planet=0):
    """Draw ``n`` samples of a ``Posterior`` and read one orbit's classical elements.

    Each draw builds its model through the posterior's ``Fit``; the orbits at
    ``orbit_path`` give the planet's period, size, shape and orientation. The keys are
    those of ``FrameOrbit.to_classical``, with ``cos_i`` in place of ``i_rad``, plus
    ``period_d``.

    Args:
        posterior: A :class:`~photomancy.infer.Posterior` of one slice.
        key: PRNG key.
        n: Number of samples.
        orbit_path: Model path of the batched orbits, for example
            ``"dynamics.orbits"``.
        planet: Planet index of those orbits, in ``[0, n_planets)``.

    Raises:
        ValueError: When ``planet`` is outside the orbit rows.

    Returns:
        Dict ``{name: (n,)}`` with ``period_d``, ``a_AU``, ``e``, ``cos_i``,
        ``W_rad``, ``w_rad`` (the argument of periapsis) and ``M0_rad`` (the mean
        anomaly at the orbit's epoch).
    """
    draws = posterior.sample(key, (n,))

    def one(x):
        orbits = resolve(posterior.fit.model_at(x), orbit_path)
        count = jax.tree.leaves(orbits)[0].shape[0]
        if not 0 <= planet < count:
            raise ValueError(
                f"planet {planet} is outside the {count} orbit row(s) at "
                f"{orbit_path!r}; indices lie in [0, {count})"
            )
        orbit = jax.tree.map(lambda leaf: leaf[planet], orbits)
        classical = orbit.to_classical()
        return {
            "period_d": orbit.period_d,
            "a_AU": classical["a_AU"],
            "e": classical["e"],
            "cos_i": jnp.cos(classical["i_rad"]),
            "W_rad": classical["W_rad"],
            "w_rad": classical["w_rad"],
            "M0_rad": classical["M0_rad"],
        }

    return jax.vmap(one)(draws)
