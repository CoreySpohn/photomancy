"""Observables read from a system or scene, shared by terms and reports."""

import jax.numpy as jnp
from orbix.system import System


def _sky_positions(system, time_d):
    """Star-relative east/north offsets in arcsec, shape (2, K, T)."""
    if isinstance(system, System):
        ephemeris = system.solve(time_d)
        return jnp.stack(
            ephemeris.offset_arcsec(jnp.arange(1, system.orbits.a_AU.size + 1))
        )
    return system.positions(time_d=time_d)


def _relative_positions(system, time_d, planet):
    """Read east/north offsets (2, T) for each row's planet and epoch."""
    return _sky_positions(system, time_d)[:, planet, jnp.arange(time_d.size)]


def _stellar_reflex(system, time_d):
    """Stellar barycentric east/north offsets (arcsec), excluding frame motion."""
    if isinstance(system, System):
        return jnp.stack(system.solve(time_d).offset_arcsec(0, "barycenter"))
    return system.star_reflex(time_d=time_d)


def _band_contrasts(
    system, quadrature, transmissions, time_d, band_id, planet, resolution=None
):
    """Select each row's filter, planet and epoch from a single scene evaluation."""

    def response(nodes_nm):
        return jnp.stack(
            [curve(row) for curve, row in zip(transmissions, nodes_nm, strict=True)]
        )

    options = {} if resolution is None else {"resolution": resolution}
    bands = system.band_contrasts(quadrature, time_d, transmission=response, **options)
    return bands[band_id, planet, jnp.arange(time_d.size)]
