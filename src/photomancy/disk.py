"""Fit a skyscapes Scene's disk through the scene-as-PyTree engine.

photomancy reimplements no disk physics -- the forward is skyscapes' own ``Scene`` /
disk render. Only the inference glue lives here: the forward module and the default
prior over the fitted leaves. A fit is ``Fit(system, default_disk_prior(system),
(ForwardTerm(radiance_forward(...), likelihood),))``.
The forward is an injected module, so the radiance render now and a coronagraph render
later share one code path.
"""

import equinox as eqx
import jax.numpy as jnp
from hwoutils.model_paths import resolve
from priorities import LogNormal, Normal

__all__ = ["default_disk_prior", "radiance_forward"]


class _RadianceForward(eqx.Module):
    """``system -> photon radiance image`` at one wavelength and time.

    Attributes:
        wavelength_nm: Wavelength to render at, in nm.
        time_d: Time in days from the system's epoch.
        grid: The sky ``PixelGrid`` (arcsec) the radiance is sampled on.
    """

    wavelength_nm: jnp.ndarray
    time_d: jnp.ndarray
    grid: object

    def __call__(self, system):
        return system.photon_radiance(
            self.wavelength_nm, self.time_d, grid=self.grid
        ).data


def radiance_forward(wavelength_nm, time_d, *, grid):
    """Build the default disk forward: ``system -> photon radiance image``.

    Calls ``system.photon_radiance`` so the shared midplane orientation is the live,
    fitted geometry. A coronagraph forward has the same signature (``system -> image``)
    and drops in without touching the rest of the fit.

    Args:
        wavelength_nm: Wavelength to render at (within the disk's grid).
        time_d: Time in days from the system's epoch (ignored by static disks).
        grid: The sky ``PixelGrid`` (arcsec, with an orientation) to sample on.

    Returns:
        An ``eqx.Module`` ``forward(system) -> image`` of shape ``grid.shape``: the
        disk photon radiance in ph s^-1 m^-2 nm^-1 sr^-1 at the pixel centres (a
        density, independent of the pixel size).
    """
    return _RadianceForward(jnp.asarray(wavelength_nm), jnp.asarray(time_d), grid)


# The canonical disk free entries: (path, kind, scale) -- LogNormal for positive scales,
# Normal for slopes and angles.
_DISK_PRIOR_SPECS = (
    ("disk.sma_AU", "lognormal", 0.5),  # positive radial scale
    ("disk.alpha_in", "normal", 3.0),  # inner radial slope
    ("disk.alpha_out", "normal", 3.0),  # outer radial slope
    ("disk.ksi0_AU", "lognormal", 0.5),  # positive vertical scale
    ("midplane_inc_deg", "normal", 20.0),  # orientation angle (deg)
    ("midplane_position_angle_deg", "normal", 20.0),  # orientation angle (deg)
)


def default_disk_prior(system):
    """A weakly-informative free dict for the canonical disk fit.

    Keyed by the six model paths ``disk.sma_AU``, ``disk.alpha_in``,
    ``disk.alpha_out``, ``disk.ksi0_AU``, ``midplane_inc_deg`` and
    ``midplane_position_angle_deg``, and centered on the system's current geometry with
    broad scales -- LogNormal for the positive scales (``sma_AU``, ``ksi0_AU``), Normal
    for the radial slopes and the orientation angles. No absolute physics is hardcoded:
    each law sits around whatever value ``system`` currently holds.

    Args:
        system: A skyscapes ``Scene`` whose current disk geometry centers the laws.

    Returns:
        ``{path: law}`` for the ``free`` field of a ``Fit`` over ``system``.
    """
    free = {}
    for path, kind, scale in _DISK_PRIOR_SPECS:
        value = jnp.asarray(resolve(system, path))
        if kind == "lognormal":
            free[path] = LogNormal(jnp.log(value), jnp.asarray(scale))
        else:
            free[path] = Normal(value, jnp.asarray(scale))
    return free
