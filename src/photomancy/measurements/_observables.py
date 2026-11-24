"""Observables read from a system or scene, shared by terms and reports."""

import jax.numpy as jnp
import numpy as np
from orbix.system import System


def _sky_positions(system, time_d):
    """Star-relative east/north offsets in arcsec, shape (2, K, T)."""
    if isinstance(system, System):
        ephemeris = system.solve(time_d)
        return jnp.stack(
            ephemeris.offset_arcsec(jnp.arange(1, system.orbits.a_AU.size + 1))
        )
    return system.positions(time_d=time_d)


def _planet_count(system):
    """Number of planets of a system or scene, read on the host."""
    if isinstance(system, System):
        return int(system.orbits.a_AU.size)
    if hasattr(system, "n_planets"):
        return int(system.n_planets)
    return int(system.positions(time_d=jnp.zeros(1)).shape[1])


def _check_planets(system, planet, valid, name):
    """Raise when a valid row selects a planet outside ``[0, n_planets)``.

    Negative indices are refused, not wrapped. Padded rows are not read. Host only:
    ``planet`` and ``valid`` must be concrete.
    """
    selected = np.asarray(planet)[np.asarray(valid)]
    count = _planet_count(system)
    if np.any((selected < 0) | (selected >= count)):
        raise ValueError(
            f"{name}: a valid row selects planet {np.unique(selected).tolist()} but "
            f"the system has {count} planet(s); indices lie in [0, {count})"
        )


def _relative_positions(system, time_d, planet):
    """Read east/north offsets (2, T) for each row's planet and epoch."""
    return _sky_positions(system, time_d)[:, planet, jnp.arange(time_d.size)]


def _stellar_reflex(system, time_d):
    """Stellar barycentric east/north offsets (arcsec), excluding frame motion."""
    if isinstance(system, System):
        return jnp.stack(system.solve(time_d).offset_arcsec(0, "barycenter"))
    return system.star_reflex(time_d=time_d)


def _band_contrasts(
    system, quadrature, transmissions, time_d, band_id, planet, resolution
):
    """Select each row's filter, planet and epoch from a single scene evaluation."""

    def response(nodes_nm):
        return jnp.stack(
            [curve(row) for curve, row in zip(transmissions, nodes_nm, strict=True)]
        )

    bands = system.band_contrasts(
        quadrature, time_d, resolution=resolution, transmission=response
    )
    return bands[band_id, planet, jnp.arange(time_d.size)]
