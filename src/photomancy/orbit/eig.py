"""Expected information gain of a planet's next astrometric epoch.

The posterior's draws are clustered into Gaussian modes over the flat vector of its
entries (:func:`~photomancy.infer.cluster_to_mixture`); each mode builds the model
through the posterior's ``Fit`` and is linearized about its mean. Candidate scoring
applies an optional contrast curve and delegates the analytic EIG to
:func:`photomancy.infer.eig.evaluate_candidates`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve
from orbix.system import System

from photomancy.infer import cluster_to_mixture, ravel_samples
from photomancy.infer.eig import evaluate_candidates
from photomancy.measurements import (
    AstrometryData,
    RelativeAstrometryTerm,
    interpolate_contrast_curve,
    system_separation_dmag,
)

__all__ = ["evaluate_orbit_candidates"]


class _Modes(NamedTuple):
    """Gaussian modes of a posterior in the layout ``evaluate_candidates`` reads."""

    means: jnp.ndarray
    covs: jnp.ndarray
    log_weights: jnp.ndarray


def _astrometry(system, t, planet):
    """East and north offsets (arcsec) of one planet at one epoch, as a row predicts."""
    count = system.orbits.a_AU.size if isinstance(system, System) else system.n_planets
    if not 0 <= planet < count:
        raise ValueError(f"Planet index {planet} outside system of {count} planets")
    one = jnp.ones(1)
    row = AstrometryData(
        times=jnp.atleast_1d(t),
        ra=0.0 * one,
        dec=0.0 * one,
        ra_err=one,
        dec_err=one,
        corr=0.0 * one,
        planet_id=jnp.array([planet]),
        is_valid=jnp.ones(1, dtype=bool),
    )
    return RelativeAstrometryTerm(row).predict(system)[:, 0]


def _photometry(system, t, albedo_area_AU2):
    sep, dmag = system_separation_dmag(
        system, jnp.atleast_1d(t), albedo_area_AU2=albedo_area_AU2
    )
    return jnp.stack((sep[0], dmag[0]))


def _modes(posterior, n_modes, key):
    """Cluster the draws into ``n_modes`` Gaussian modes over the flat entries."""
    mixture = cluster_to_mixture(posterior, n_modes, key)
    components = mixture.component_distribution
    tril = components.scale_tril
    return _Modes(
        means=components.loc,
        covs=tril @ jnp.swapaxes(tril, -1, -2),
        log_weights=jax.nn.log_softmax(mixture.mixing_logits),
    )


def evaluate_orbit_candidates(
    posterior,
    time_d,
    obs_variance,
    *,
    key,
    n_modes=4,
    system_path="",
    planet=0,
    contrast_curve=None,
    area_path=None,
    iwa_arcsec=0.0,
):
    """Score a planet's relative astrometry at candidate days from the system epoch.

    The draws of ``posterior`` are clustered into ``n_modes`` Gaussian modes over the
    flat vector of all its entries. At each mode mean the model is built through
    ``posterior.fit`` and the system at ``system_path`` predicts the east and north
    offsets of planet ``planet``, linearized about the mean. The zero-based ``planet``
    index follows scene order for a scene and Jacobi chain order for a bare orbix
    system.

    Each reported measurement has variance ``obs_variance`` in arcsec squared. An
    optional ``(sep_arcsec, dMag_limit)`` curve makes a mode report only when its
    planet is above the inner working angle and brighter than the curve (one-planet
    systems). The Lambertian scale is the scene's albedo times squared radius; a bare
    orbix system needs it as a model leaf (AU^2) at ``area_path``.

    Args:
        posterior: An unbatched :class:`~photomancy.infer.Posterior`.
        time_d: Candidate epochs, days from the system's epoch, shape ``(N,)``.
        obs_variance: Measurement variance (arcsec squared).
        key: PRNG key of the clustering.
        n_modes: Number of Gaussian modes.
        system_path: Model path of the scored system; the empty path is the model.
        planet: Planet index.
        contrast_curve: Optional ``(sep_arcsec, dMag_limit)`` arrays.
        area_path: Model path of the geometric albedo times squared radius (AU^2),
            or None to read the scene's.
        iwa_arcsec: Inner working angle (arcsec) of the detection model.

    Returns:
        The dict of :func:`~photomancy.infer.eig.evaluate_candidates` (``total_eig``,
        ``geometric_eig``, ``alias_eig``, ``predictions``, ``detectability``) plus
        ``separation`` and ``dMag``, shape ``(N, n_modes)``, at each mode mean (zero
        without a contrast curve).
    """
    modes = _modes(posterior, n_modes, key)
    _, unravel = ravel_samples(posterior)

    def system_at(z):
        return resolve(posterior.fit.model_at(unravel(z)), system_path)

    def forward(z, t):
        return _astrometry(system_at(z), t, planet)

    def photometry(z, t):
        model = posterior.fit.model_at(unravel(z))
        area = None if area_path is None else resolve(model, area_path)
        return _photometry(resolve(model, system_path), t, area)

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

    epochs = jnp.asarray(time_d)
    res = evaluate_candidates(
        modes,
        epochs,
        forward,
        obs_variance,
        detectable=detectable,
        detectable_args=detectable_args,
    )
    if contrast_curve is not None:
        sep_dmag = jax.vmap(
            lambda t: jax.vmap(photometry, in_axes=(0, None))(modes.means, t)
        )(epochs)
        res["separation"] = sep_dmag[..., 0]
        res["dMag"] = sep_dmag[..., 1]
    else:
        res["separation"] = jnp.zeros_like(res["detectability"])
        res["dMag"] = jnp.zeros_like(res["detectability"])
    return res
