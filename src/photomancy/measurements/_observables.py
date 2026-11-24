"""Observables read from a system or scene, shared by terms and reports."""

import jax
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


def _concrete(*values):
    """True when no value is a tracer, so a host range check can read it."""
    return not any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(values))


def _planet_count(system, forward=None):
    """Number of planets of a system or scene, read on the host.

    A bare ``System`` and an object with ``n_planets`` answer directly; any other model
    gives the count from the planet axis of ``forward()``, the term's own forward
    result.
    """
    if isinstance(system, System):
        return int(system.orbits.a_AU.size)
    if hasattr(system, "n_planets"):
        return int(system.n_planets)
    return int(forward())


def _check_indices(selected, valid, count, name, what):
    """Raise when a valid row's index lies outside ``[0, count)``.

    Negative indices are refused, not wrapped. Padded rows are not read. The check runs
    on concrete values only: under a trace the indices are unknown and nothing is
    tested.
    """
    if _concrete(selected, valid):
        chosen = np.asarray(selected)[np.asarray(valid)]
        if np.any((chosen < 0) | (chosen >= count)):
            raise ValueError(
                f"{name}: a valid row selects {what} {np.unique(chosen).tolist()} "
                f"but there are {count}; indices lie in [0, {count})"
            )


def _check_planets(system, planet, valid, name, forward=None):
    """Raise when a valid row selects a planet outside ``[0, n_planets)``."""
    if _concrete(planet, valid):
        _check_indices(planet, valid, _planet_count(system, forward), name, "planet")


def _relative_positions(system, time_d, planet):
    """Read east/north offsets (2, T) for each row's planet and epoch."""
    return _sky_positions(system, time_d)[:, planet, jnp.arange(time_d.size)]


def _stellar_reflex(system, time_d):
    """Stellar barycentric east/north offsets (arcsec), excluding frame motion."""
    if isinstance(system, System):
        return jnp.stack(system.solve(time_d).offset_arcsec(0, "barycenter"))
    return system.star_reflex(time_d=time_d)


def _band_table(system, quadrature, transmissions, time_d, resolution):
    """Band contrasts ``(W, K, T)`` of a system, one filter response per band."""

    def response(nodes_nm):
        return jnp.stack(
            [curve(row) for curve, row in zip(transmissions, nodes_nm, strict=True)]
        )

    return system.band_contrasts(
        quadrature, time_d, resolution=resolution, transmission=response
    )


def _band_contrasts(
    system, quadrature, transmissions, time_d, band_id, planet, resolution
):
    """Select each row's filter, planet and epoch from a single scene evaluation."""
    bands = _band_table(system, quadrature, transmissions, time_d, resolution)
    return bands[band_id, planet, jnp.arange(time_d.size)]
