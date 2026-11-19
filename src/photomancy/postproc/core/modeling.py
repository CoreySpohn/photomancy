"""Forward modeling tools for injecting synthetic sources.

Provides utilities for fake planet injection and simple disk generation
for throughput estimation and contrast curve calculations.
"""

import functools

import jax
import jax.numpy as jnp
from hwoutils.radial import get_center
from hwoutils.transforms import shift_image


@jax.jit
def inject_planet(
    image: jnp.ndarray,
    psf_template: jnp.ndarray,
    flux: float,
    pos: tuple[float, float],
    order: int = 3,
) -> jnp.ndarray:
    """Inject a fake planet into an image with sub-pixel precision.

    The PSF template is shifted from the image center to the target position
    with sub-pixel precision, scaled by the flux, and added to the image.

    Args:
        image: 2D image to inject into, shape (ny, nx).
        psf_template: 2D PSF template centered at image center, same shape as image.
        flux: Flux scaling factor for the injected planet.
        pos: Target position (y, x) in pixels.
        order: Interpolation order for sub-pixel shifting. Default 3 uses
            the Keys cubic convolution kernel.

    Returns:
        Image with injected planet, same shape as input.
    """
    ny, nx = image.shape
    cy, cx = get_center((ny, nx))

    # Shift from center to target position
    dy = pos[0] - cy
    dx = pos[1] - cx

    planet_signal = shift_image(psf_template, dy, dx, order=order)
    return image + planet_signal * flux


@functools.partial(jax.jit, static_argnames=["shape"])
def make_simple_disk(
    shape: tuple[int, int],
    radius: float,
    inclination_deg: float,
    width: float,
    flux: float = 1.0,
    pa_deg: float = 0.0,
) -> jnp.ndarray:
    """Generate a simple, optically thin Gaussian ring/disk.

    Analytically projects the disk to avoid interpolation artifacts.
    Flux is normalized so the total integrated flux equals the specified value.

    Image arrays follow the FITS display convention: row index increases north
    and column index increases west. With imshow(origin="lower") and no axis
    inversion, north is up and east is left. Astronomical position angle is
    measured from north through east, counterclockwise on this image.

    Args:
        shape: Output image shape (ny, nx).
        radius: Ring radius in pixels.
        inclination_deg: Disk inclination (0 = face-on, 90 = edge-on).
        width: Gaussian width (sigma) of the ring in pixels.
        flux: Total integrated flux of the disk. Default 1.0.
        pa_deg: Position angle of major axis, from north through east (degrees).

    Returns:
        2D disk image with total flux normalized to the specified value.
    """
    ny, nx = shape
    cy, cx = get_center((ny, nx))
    y, x = jnp.mgrid[:ny, :nx]

    # Shift to center
    y_c = y - cy
    x_c = x - cx

    # Rotate by position angle
    # PA=0 aligns with north (+row); positive PA turns toward east (-column).
    pa_rad = jnp.deg2rad(90.0 - pa_deg)
    cos_pa = jnp.cos(pa_rad)
    sin_pa = jnp.sin(pa_rad)
    y_rot = y_c * cos_pa + x_c * sin_pa
    x_rot = -y_c * sin_pa + x_c * cos_pa

    # De-project inclination (stretch along minor axis)
    inc_rad = jnp.deg2rad(inclination_deg)
    cos_inc = jnp.cos(inc_rad)
    # Avoid division by zero for edge-on disks
    cos_inc = jnp.maximum(cos_inc, 1e-6)
    y_deproj = y_rot / cos_inc

    # Radial distance in disk plane
    r_deproj = jnp.sqrt(y_deproj**2 + x_rot**2)

    # Gaussian profile
    profile = jnp.exp(-0.5 * ((r_deproj - radius) / width) ** 2)

    # Normalize to specified flux
    total = jnp.sum(profile)
    return jnp.where(total > 0, profile / total * flux, profile)


# =============================================================================
# Model Subtraction (inverse of injection)
# =============================================================================


@jax.jit
def subtract_star(
    science: jnp.ndarray,
    star_model: jnp.ndarray,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Subtract stellar PSF model from observation.

    This is the fundamental operation for "perfect" RDI when you have
    a noiseless stellar PSF expectation (e.g., from coronagraphoto).

    Args:
        science: Observed image (electrons).
        star_model: Noiseless stellar PSF expectation (electrons).
        scale: Multiplicative scaling factor for the model before subtraction.
            Use values != 1.0 when the reference brightness differs from
            the science image (e.g., different exposure times or stellar flux).

    Returns:
        Residual image containing noise + planet signal.

    Example::

        residual = subtract_star(observation, star_expectation)
        # With scaling for brightness mismatch:
        residual = subtract_star(observation, star_expectation, scale=0.95)
    """
    return science - scale * star_model


@jax.jit
def subtract_disk(
    residual: jnp.ndarray,
    disk_model: jnp.ndarray,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Subtract disk model from residual image.

    Disk subtraction is typically a separate modeling task from stellar
    speckle subtraction. Call this after subtract_star when analyzing
    systems with circumstellar disks.

    Args:
        residual: Image after stellar subtraction (from subtract_star).
        disk_model: Disk model expectation (electrons).
        scale: Multiplicative scaling factor for the disk model.
            Adjust when disk model brightness doesn't match observation.

    Returns:
        Residual image with disk contribution removed.

    Example::

        # Two-step subtraction
        residual = subtract_star(observation, star_model)
        residual = subtract_disk(residual, disk_model)
    """
    return residual - scale * disk_model
