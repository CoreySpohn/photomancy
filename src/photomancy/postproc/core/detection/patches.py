"""Differentiable fixed-size patch extraction for template filtering."""

import jax.numpy as jnp
from hwoutils.map_coordinates import map_coordinates


def extract_patch(
    image: jnp.ndarray,
    center_yx: jnp.ndarray,
    size: int,
    order: int = 3,
    cval: float = 0.0,
) -> jnp.ndarray:
    """Extract a (size, size) patch centered on a sub-pixel position.

    The patch is sampled with spline interpolation on a regular grid of unit
    pixel spacing centered at center_yx, so the extraction is differentiable
    with respect to the center position. Samples falling outside the image
    fill with cval.

    Args:
        image: 2D source image.
        center_yx: (2,) patch center (y, x) in pixels; may be sub-pixel and
            may be a traced value.
        size: Patch edge length in pixels (static; odd sizes center exactly
            on a pixel).
        order: Interpolation order passed to map_coordinates.
        cval: Fill value for samples outside the image.

    Returns:
        (size, size) patch.
    """
    offsets = jnp.arange(size) - (size - 1) / 2.0
    dy, dx = jnp.meshgrid(offsets, offsets, indexing="ij")
    coords = jnp.stack([center_yx[0] + dy, center_yx[1] + dx])
    return map_coordinates(image, coords, order=order, cval=cval)
