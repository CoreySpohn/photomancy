"""Fit a skyscapes Scene's disk through the scene-as-PyTree engine.

photomancy reimplements no disk physics -- the forward is skyscapes' own ``Scene`` /
disk render. Only the inference glue lives here: the forward module and the default
prior over the fitted leaves. A fit is ``Fit(system, default_disk_prior(system),
(ForwardTerm(radiance_forward(...), likelihood),))``.
The forward is an injected module, so the radiance render now and a coronagraph render
later share one code path.
"""

import math
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from hwoutils.constants import deg2rad, rad2deg
from orbix.system import IsotropicInclination
from priorities import (
    AbstractDistribution,
    LogNormal,
    Normal,
    TruncatedDistribution,
    Uniform,
)

__all__ = ["default_disk_prior", "radiance_forward"]


class _IsotropicInclinationDeg(AbstractDistribution):
    """The orbix isotropic inclination law in the scene's degree coordinate.

    Attributes:
        radians: The isotropic inclination law in radians it delegates to.
        event_dim: Number of event dimensions, zero for a scalar.
        bounds: Kind of support, "interval".
        has_icdf: True: the inverse CDF is available.
    """

    radians: IsotropicInclination
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "interval"
    has_icdf: ClassVar[bool] = True

    def support(self):
        low, high = self.radians.support()
        return low * rad2deg, high * rad2deg

    def log_prob(self, value):
        return self.radians.log_prob(jnp.asarray(value) * deg2rad) + math.log(deg2rad)

    def sample(self, key, sample_shape=(), *, dtype=None):
        return self.radians.sample(key, sample_shape, dtype=dtype) * rad2deg

    def cdf(self, value):
        return self.radians.cdf(jnp.asarray(value) * deg2rad)

    def icdf(self, q):
        return self.radians.icdf(q) * rad2deg


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


def default_disk_prior(system):
    """Return the six independent leaf laws for a Grater disk fit.

    The disk orientation is isotropic. Inclination i in [0, 180] degrees has
    density (pi/360) * sin(i * pi/180) per degree. The position angle is
    uniform over one 360-degree turn centered on the scene's current angle.
    At fixed inclination, a half-turn reverses the near side seen in anisotropic
    scattered light. Reflecting the inclination as well gives the same image,
    and both labels have the same prior density when inside these intervals.

    The radial and vertical scales have LogNormal laws with the current values
    as medians and log standard deviation 0.5. The slope laws are Normals of
    width 3 centered on the current slopes, conditioned on a positive inner
    slope and a negative outer slope. The continuous laws assign zero mass to zero.

    Fit maps the scalar support bounds to unconstrained coordinates, so these
    laws introduce no walls for NUTS or MCLMC. Use interior values for an
    explicit sampler start.

    Args:
        system: A skyscapes Scene with a Grater disk.

    Returns:
        A dict from the six free model paths to their normalized laws.
    """
    disk = system.disk
    pa = system.midplane_position_angle_deg
    return {
        "disk.sma_AU": LogNormal(jnp.log(disk.sma_AU), 0.5),
        "disk.alpha_in": TruncatedDistribution(Normal(disk.alpha_in, 3.0), low=0.0),
        "disk.alpha_out": TruncatedDistribution(Normal(disk.alpha_out, 3.0), high=0.0),
        "disk.ksi0_AU": LogNormal(jnp.log(disk.ksi0_AU), 0.5),
        "midplane_inc_deg": _IsotropicInclinationDeg(IsotropicInclination()),
        "midplane_position_angle_deg": Uniform(pa - 180.0, pa + 180.0),
    }
