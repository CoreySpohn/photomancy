"""Orbit-domain Expected Information Gain (a thin wrapper over the generic layer).

Chart-space mixtures use the system terms' sky positions and Lambertian dMag
predictions. Candidate scoring applies the contrast
curve and delegates the analytic EIG to :mod:`photomancy.eig`.
"""

import jax
import jax.numpy as jnp

from photomancy.eig import (
    alias_breaking_eig,
    alias_breaking_eig_from_covariances,
    detectability_eig,
    geometric_eig,
)
from photomancy.eig import evaluate_candidates as _eval_candidates
from photomancy.orbit._contrast import interpolate_contrast_curve
from photomancy.orbit._photometry import system_separation_dmag

__all__ = [
    "alias_breaking_eig",
    "alias_breaking_eig_from_covariances",
    "detectability_eig",
    "evaluate_orbit_candidates",
    "geometric_eig",
]


def _astrometry(system, t, planet):
    # The system terms import this orbit package through their data containers.
    from photomancy.system_terms import _sky_positions

    positions = _sky_positions(system, jnp.atleast_1d(t))
    count = positions.shape[1]
    if not 0 <= planet < count:
        raise ValueError(f"Planet index {planet} outside system of {count} planets")
    return positions[:, planet, 0]


def _photometry(system, t, area_AU2):
    sep, dmag = system_separation_dmag(system, jnp.atleast_1d(t), area_AU2=area_AU2)
    return jnp.stack((sep[0], dmag[0]))


# ---------------------------------------------------------------------------
# High-level batch evaluation (delegates to photomancy.eig)
# ---------------------------------------------------------------------------


def _orbit_evaluate(
    posterior,
    candidate_epochs,
    obs_variance,
    forward,
    photometry,
    contrast_curve,
    iwa_arcsec,
):
    """Score east/north measurements with optional dMag selection at each mode."""
    detectable = None
    detectable_args = ()
    if contrast_curve is not None:
        detectable_args = tuple(map(jnp.asarray, contrast_curve))

        def detectable(z, t, csep, cdmag):
            sep, dmag = photometry(z, t)
            dmag_limit = interpolate_contrast_curve(
                sep[None], csep[None, :], cdmag[None, :]
            )[0]
            return ((sep > iwa_arcsec) & (dmag < dmag_limit)).astype(sep.dtype)

    epochs = jnp.asarray(candidate_epochs)
    res = _eval_candidates(
        posterior,
        epochs,
        forward,
        obs_variance,
        detectable=detectable,
        detectable_args=detectable_args,
    )

    if contrast_curve is not None:
        sep_dmag = jax.vmap(
            lambda t: jax.vmap(photometry, in_axes=(0, None))(posterior.means, t)
        )(epochs)
        res["separation"] = sep_dmag[..., 0]
        res["dMag"] = sep_dmag[..., 1]
    else:
        res["separation"] = jnp.zeros_like(res["detectability"])
        res["dMag"] = jnp.zeros_like(res["detectability"])
    return res


def evaluate_orbit_candidates(
    posterior,
    problem,
    time_d,
    obs_variance,
    *,
    planet=0,
    contrast_curve=None,
    area_AU2=None,
    iwa_arcsec=0.0,
):
    """Score a planet's relative astrometry at days from the system epoch.

    The mixture means and covariances are in the flat chart coordinates of the
    fitted ``ChartOrbitProblem``. Its density builds the physical system, including
    sampled body properties.
    The zero-based ``planet`` index follows scene order for a scene system and
    Jacobi chain order for a bare orbix system.

    Each reported measurement has variance ``obs_variance`` in arcsec squared.
    An optional ``(sep_arcsec, dMag_limit)`` curve enables mode-wise
    detection for a single planet above the inner working angle.
    The Lambert scale defaults to the scene's albedo times squared radius.
    Supply ``area_AU2`` to use a fixed scale, including with a bare orbix system.

    The posterior must declare the problem's chart and epoch (``problem.run`` and
    ``problem.declare`` stamp them); ``time_d`` is days from that epoch. A posterior
    in another chart, or an undeclared one, raises ``ValueError``.
    """
    problem.require(posterior)

    def forward(z, t):
        return _astrometry(
            problem.density.build_system(problem.unflatten(z)), t, planet
        )

    def photometry(z, t):
        return _photometry(
            problem.density.build_system(problem.unflatten(z)), t, area_AU2
        )

    return _orbit_evaluate(
        posterior, time_d, obs_variance, forward, photometry, contrast_curve, iwa_arcsec
    )
