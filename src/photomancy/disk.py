"""Fit a skyscapes System's disk through the scene-as-PyTree engine.

photomancy reimplements no disk physics -- the forward is skyscapes' own ``System`` /
disk render. Only the inference glue lives here: select the fitted leaves, build a
Gaussian-image likelihood, and delegate to ``build_scene_logdensity``. The forward is an
injected callable, so the surface-brightness render now and a coronagraph render later
share one code path.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.core import build_scene_logdensity


def surface_brightness_forward(wavelength_nm, time_jd):
    """Build the default disk forward: ``system -> surface-brightness image``.

    Reads ``midplane_inc_deg`` / ``midplane_pa_deg`` off the ``System`` so the shared
    orientation is the live, fitted geometry. The coronagraph forward has the same
    signature (``system -> image``) and drops in without touching the rest of the fit.

    Args:
        wavelength_nm: Wavelength to render at (within the disk's grid).
        time_jd: Epoch (ignored by static disks; part of the disk interface).

    Returns:
        ``forward(system) -> image`` of shape ``(ny, nx)`` (contrast per pixel).
    """
    wl = jnp.asarray(wavelength_nm)
    t = jnp.asarray(time_jd)

    def forward(system):
        return system.disk.surface_brightness(
            wl, t, system.midplane_inc_deg, system.midplane_pa_deg
        )

    return forward


def build_disk_logdensity(
    system,
    image,
    *,
    fit_leaves,
    noise_sigma,
    forward,
    prior=None,
):
    """Flat logdensity fitting selected ``System`` leaves to a disk image.

    Args:
        system: A skyscapes ``System`` (the scene; its selected leaves are the params).
        image: Observed surface-brightness image, shape ``(ny, nx)``.
        fit_leaves: ``system -> list[leaf]`` selecting the leaves to fit (e.g.
            ``lambda s: [s.disk.sma_AU, s.midplane_inc_deg, ...]``).
        noise_sigma: Per-pixel Gaussian noise sigma (scalar).
        forward: ``system -> predicted image`` -- the swappable seam.
        prior: Optional ``system -> scalar`` log-prior; ``None`` is flat (improper).

    Returns:
        ``(logdensity, z0, unravel)`` from ``build_scene_logdensity``: ``logdensity(z)``
        scores a flat position over the fitted leaves, ``z0`` is the scene's initial
        position; ``unravel(z)`` rebuilds the fitted-leaf PyTree.
    """
    n_fit = len(fit_leaves(system))
    mask = jax.tree_util.tree_map(lambda _: False, system)
    mask = eqx.tree_at(fit_leaves, mask, [True] * n_fit)

    image = jnp.asarray(image)
    inv_var = 1.0 / jnp.asarray(noise_sigma) ** 2

    def likelihood(predicted):
        return -0.5 * jnp.sum((predicted - image) ** 2 * inv_var)

    if prior is None:

        def prior(_system):
            return 0.0

    return build_scene_logdensity(system, forward, likelihood, prior, filter_spec=mask)
