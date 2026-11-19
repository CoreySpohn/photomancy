"""Convolution-based aperture photometry for coronagraphic images.

Provides differentiable aperture kernels (soft and hard) and convolution-based
flux map generation for efficient photometry calculations.
"""

import functools

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d


@functools.lru_cache(maxsize=32)
def make_aperture_kernel(
    radius: float,
    soft: bool = False,
    sharpness: float = 10.0,
) -> jnp.ndarray:
    """Create a circular aperture kernel for convolution-based photometry.

    This function is cached (up to 32 configurations) to avoid repeated
    kernel creation overhead when called with the same parameters.

    Args:
        radius: Aperture radius in pixels.
        soft: If True, use sigmoid-based soft edge for differentiability.
              If False, use hard binary mask.
        sharpness: Steepness of sigmoid transition (only used if soft=True).

    Returns:
        2D aperture kernel. Not normalized (sum of counts, not average).
    """
    # Kernel size with padding
    size = int(2 * radius + 3)
    half = size // 2

    # Create coordinate grid centered on kernel
    y, x = jnp.ogrid[-half : half + 1, -half : half + 1]
    dist = jnp.sqrt(x**2 + y**2)

    if soft:
        # Sigmoid for differentiable edge: 1 / (1 + e^(sharpness*(dist-radius)))
        kernel = jax.nn.sigmoid(sharpness * (radius - dist))
    else:
        # Hard binary mask
        kernel = (dist <= radius).astype(jnp.float32)

    return kernel


@jax.jit
def flux_map(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Generate a flux map via convolution with an aperture kernel.

    Each pixel in the output represents the integrated flux that would be
    measured if an aperture were centered at that position.

    Args:
        image: 2D image array.
        kernel: Aperture kernel from make_aperture_kernel.

    Returns:
        2D flux map with same shape as input image.
    """
    return convolve2d(image, kernel, mode="same")


# ==============================================================================
# Aperture Mask Functions (consolidated from top-level photometry.py)
# ==============================================================================


def circular_aperture_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> jnp.ndarray:
    """Create a circular aperture mask.

    Args:
        shape: Image shape (ny, nx).
        center: Center of aperture (y, x) in pixels.
        radius: Radius of aperture in pixels.

    Returns:
        Boolean mask array with True inside the aperture.
    """
    ny, nx = shape
    y, x = jnp.ogrid[:ny, :nx]
    cy, cx = center
    distance = jnp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return distance <= radius


def soft_aperture_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    sharpness: float = 10.0,
) -> jnp.ndarray:
    """Create a soft (differentiable) circular aperture mask.

    Uses a sigmoid function to create a smooth transition at the aperture
    edge, enabling gradient-based optimization through the mask.

    Args:
        shape: Image shape (ny, nx).
        center: Center of aperture (y, x) in pixels.
        radius: Radius of aperture in pixels.
        sharpness: Steepness of the sigmoid transition.

    Returns:
        Soft mask array with values in [0, 1].
    """
    ny, nx = shape
    y, x = jnp.ogrid[:ny, :nx]
    cy, cx = center
    distance = jnp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return jax.nn.sigmoid(sharpness * (radius - distance))


@jax.jit
def aperture_photometry(
    image: jnp.ndarray,
    center: tuple[float, float],
    radius: float,
) -> float:
    """Perform circular aperture photometry on an image.

    Args:
        image: 2D image array.
        center: Center of aperture (y, x) in pixels.
        radius: Radius of aperture in pixels.

    Returns:
        Total flux within the aperture.
    """
    mask = circular_aperture_mask(image.shape, center, radius)
    return jnp.sum(image * mask)


def aperture_solid_angle(
    radius_pixels: float,
    pixel_scale_arcsec: float,
) -> float:
    """Calculate the solid angle of a circular aperture.

    Args:
        radius_pixels: Aperture radius in pixels.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.

    Returns:
        Solid angle in arcsec^2.
    """
    radius_arcsec = radius_pixels * pixel_scale_arcsec
    return jnp.pi * radius_arcsec**2
