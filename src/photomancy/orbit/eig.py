"""Orbit-domain Expected Information Gain (a thin wrapper over the generic layer).

Both chart-space mixtures and NumPyro orbit problems use the system terms' sky
positions and Lambertian dMag predictions. Candidate scoring applies the contrast
curve and delegates the analytic EIG to :mod:`photomancy.eig`.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from hwoutils.constants import Mearth2kg, Msun2kg, two_pi
from hwoutils.epoch import Epoch
from orbix.system import Body, SkyFrame, System

from photomancy.eig import (
    alias_breaking_eig,
    alias_breaking_eig_from_covariances,
    detectability_eig,
    geometric_eig,
)
from photomancy.eig import evaluate_candidates as _eval_candidates
from photomancy.orbit._contrast import interpolate_contrast_curve
from photomancy.orbit._photometry import system_separation_dmag
from photomancy.orbit.diagnostics import orbits_from_samples

__all__ = [
    "alias_breaking_eig",
    "alias_breaking_eig_from_covariances",
    "detectability_eig",
    "evaluate_numpyro_candidates",
    "evaluate_orbit_candidates",
    "geometric_eig",
]


# ---------------------------------------------------------------------------
# Pure-JAX prediction from unconstrained z -- the orbit forward model
# ---------------------------------------------------------------------------


def _system_from_sites(z_flat, unflatten, constrain, Ms, dist_pc):
    """Build a single star/companion pair in the sample sites' day coordinate."""
    phys = constrain(unflatten(z_flat))
    period_d = 10.0 ** phys["log_P"]
    mass_kg = 10.0 ** phys["log_Mp"] * Mearth2kg if "log_Mp" in phys else 0.0
    samples = {
        "T": period_d,
        "Mp": mass_kg,
        "e": phys["e_raw"],
        "cos_i": phys["cos_i"],
        "W": phys["W"],
        "cos_w": jnp.cos(phys["w_raw"]),
        "sin_w": jnp.sin(phys["w_raw"]),
        "tp": -phys["M0"] * period_d / two_pi,
    }
    orbit = orbits_from_samples(samples, Ms, reference_day=0.0)
    return System(
        Body(Ms / Msun2kg),
        Body(jnp.atleast_1d(mass_kg / Msun2kg)),
        jax.tree.map(lambda x: x[None], orbit),
        SkyFrame(dist_pc, Epoch()),
    )


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


def _predict_astrom_pure(
    z_flat, t, unflatten: Callable, constrain: Callable, Ms, dist_pc
):
    """Predict the first planet's east/north offsets (arcsec) from a site vector."""
    return _astrometry(
        _system_from_sites(z_flat, unflatten, constrain, Ms, dist_pc), t, 0
    )


def _predict_sep_dmag_pure(
    z_flat, t, unflatten: Callable, constrain: Callable, Ms, dist_pc, Lambda
):
    """Predict angular separation and dMag with the fitted orbit and fixed area."""
    return _photometry(
        _system_from_sites(z_flat, unflatten, constrain, Ms, dist_pc), t, Lambda
    )


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
    *,
    cache_key=None,
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
        cache_key=cache_key,
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


def evaluate_numpyro_candidates(
    posterior,
    problem,
    candidate_epochs,
    obs_variance,
    Ms,
    dist_pc,
    *,
    Lambda=None,
    contrast_curve=None,
    iwa=0.0,
):
    """EIG over candidate epochs for a generic z-space MixturePosterior + OrbitProblem.

    Lets OFTI / grid_search (via ``to_unconstrained`` -> ``cluster_to_mixture``) drive
    the same analytic EIG as the Laplace mixture. ``problem`` supplies the model
    ``unflatten`` and ``constrain``. Returns the per-candidate EIG dict of
    :func:`photomancy.eig.evaluate_candidates`, an approximate score.

    Without ``Lambda`` and ``contrast_curve`` every epoch is assumed to yield
    reported relative astrometry with fixed measurement noise. With both, each mode
    is reported according to its mean orbit (outside ``iwa``, within the curve's
    separation range and brighter than its limit). This gives the per-mode reporting
    probability in :func:`photomancy.eig.evaluate_candidates`. The
    threshold is not applied within a mode, so selection that cuts through a mode's
    predicted position or brightness is not represented.
    """
    unflatten, constrain = problem.unflatten, problem.constrain

    def forward(z, t):
        return _predict_astrom_pure(z, t, unflatten, constrain, Ms, dist_pc)

    def photometry(z, t):
        return _predict_sep_dmag_pure(z, t, unflatten, constrain, Ms, dist_pc, Lambda)

    curve = contrast_curve if Lambda is not None else None
    # The model cache retains one unflatten per config (including prior bounds).
    cache_key = (id(unflatten), Ms, dist_pc, curve is not None, Lambda, iwa)
    return _orbit_evaluate(
        posterior,
        candidate_epochs,
        obs_variance,
        forward,
        photometry,
        curve,
        iwa,
        cache_key=cache_key,
    )


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
    """

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
