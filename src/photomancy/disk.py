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
from jax.flatten_util import ravel_pytree

from photomancy.core import build_gaussian_fit
from photomancy.priors import IndependentPrior, LogNormal, Normal


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
        prior: Optional prior, forwarded to ``build_scene_logdensity`` -- either a
            ``photomancy.priors.AbstractPrior`` (scored in z-space) or a
            ``system -> scalar`` callable. ``None`` is flat (improper).

    Returns:
        ``(logdensity, z0, unravel)`` from ``build_scene_logdensity``: ``logdensity(z)``
        scores a flat position over the fitted leaves, ``z0`` is the scene's initial
        position; ``unravel(z)`` rebuilds the fitted-leaf PyTree.
    """
    return build_gaussian_fit(
        system,
        image,
        fit_leaves=fit_leaves,
        noise_sigma=noise_sigma,
        forward=forward,
        prior=prior,
    )


# The canonical disk fit-leaf prior specs, aligned to ``disk_fit_leaves`` order:
# (kind, scale) per leaf -- LogNormal for positive scales, Normal for slopes / angles.
_DISK_PRIOR_SPECS = (
    ("lognormal", 0.5),  # sma_AU -- positive radial scale
    ("normal", 3.0),  # alpha_in -- inner radial slope
    ("normal", 3.0),  # alpha_out -- outer radial slope
    ("lognormal", 0.5),  # ksi0_AU -- positive vertical scale
    ("normal", 20.0),  # midplane_inc_deg -- orientation angle (deg)
    ("normal", 20.0),  # midplane_pa_deg -- orientation angle (deg)
)


def disk_fit_leaves(system):
    """The canonical disk fit-leaves: radial / vertical shape + shared orientation.

    Returns the six leaves ``sma_AU, alpha_in, alpha_out, ksi0_AU, midplane_inc_deg,
    midplane_pa_deg`` -- the selection ``default_disk_prior`` is aligned to. Pass it as
    ``build_disk_logdensity``'s ``fit_leaves``.
    """
    return [
        system.disk.sma_AU,
        system.disk.alpha_in,
        system.disk.alpha_out,
        system.disk.ksi0_AU,
        system.midplane_inc_deg,
        system.midplane_pa_deg,
    ]


def default_disk_prior(system):
    """A weakly-informative ``IndependentPrior`` for the canonical disk fit.

    Aligned to ``disk_fit_leaves`` and centered on the system's current geometry with
    broad scales -- LogNormal for the positive scales (``sma_AU``, ``ksi0_AU``), Normal
    for the radial slopes and the orientation angles. No absolute physics is hardcoded:
    each prior sits around whatever value ``system`` currently holds. Pass it together
    with ``disk_fit_leaves`` so the prior's z matches the fit's z.

    Args:
        system: A skyscapes ``System`` whose current disk geometry centers the prior.

    Returns:
        An ``IndependentPrior`` over the raveled ``disk_fit_leaves`` selection (ndim 6).
    """
    n = len(_DISK_PRIOR_SPECS)
    mask = jax.tree_util.tree_map(lambda _: False, system)
    mask = eqx.tree_at(disk_fit_leaves, mask, [True] * n)
    z0 = ravel_pytree(eqx.partition(system, mask)[0])[0]  # current values, z order

    # Tag each fitted leaf with its spec index, then ravel so specs line up with z0
    # (the ravel order need not match the disk_fit_leaves list order).
    tagged = eqx.tree_at(
        disk_fit_leaves, system, [jnp.asarray(float(i)) for i in range(n)]
    )
    spec_z = ravel_pytree(eqx.partition(tagged, mask)[0])[0]

    components = []
    for j in range(n):
        kind, scale = _DISK_PRIOR_SPECS[int(spec_z[j])]
        value = z0[j]
        if kind == "lognormal":
            components.append(
                LogNormal(loc=jnp.log(value)[None], scale=jnp.asarray([scale]))
            )
        else:
            components.append(Normal(loc=value[None], scale=jnp.asarray([scale])))
    return IndependentPrior(tuple(components))
