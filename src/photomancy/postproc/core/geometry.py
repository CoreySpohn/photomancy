"""Geometric utilities for coronagraphic image analysis.

Functions for coordinate transforms, distance calculations, and generating
aperture positions with static array shapes for JAX compatibility.
"""

import jax.numpy as jnp


def n_reference_apertures(
    radius: jnp.ndarray,
    fwhm: float,
    exclusion_buffer: float = 0.5,
) -> jnp.ndarray:
    """Traceable count of reference apertures at a given radius.

    Identical formula to calculate_n_apertures (Mawet et al. 2014 aperture
    packing with an exclusion buffer on both sides of the candidate), but
    expressed in jnp ops end to end so it can be evaluated with a traced
    radius inside jit/vmap. This is the single source of truth for the
    count used by the detection samplers.

    Args:
        radius: Radial distance from center in pixels (may be traced).
        fwhm: Full width at half maximum in pixels.
        exclusion_buffer: Gap between candidate and first/last reference
            aperture in units of angular step.

    Returns:
        Integer scalar array of valid reference apertures (at least 1).
    """
    half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(radius, 0.1), 1.0))
    d_theta = 2.0 * half_angle
    n_theoretical = jnp.floor(2 * jnp.pi / jnp.maximum(d_theta, 0.01))
    return jnp.maximum((n_theoretical - 1 - 2 * exclusion_buffer).astype(int), 1)


def calculate_n_apertures(
    radius: float,
    fwhm: float,
    exclusion_buffer: float = 0.5,
) -> int:
    """Calculate the number of reference apertures at a given radius.

    Uses the Mawet et al. (2014) formula with exclusion buffer correction
    to ensure no overlap with the planet aperture on either side.

    This function provides the canonical calculation used by the SNR module,
    ensuring consistency between visualization and computation.

    Args:
        radius: Radial distance from center in pixels.
        fwhm: Full width at half maximum in pixels.
        exclusion_buffer: Gap between planet and first/last reference aperture
            in units of angular step (default 0.5). Creates a gap on both sides.

    Returns:
        Number of valid reference apertures.

    Example::

        from photomancy.postproc.core.geometry import calculate_n_apertures
        n = calculate_n_apertures(radius=20, fwhm=5.0)
        print(f"{n} reference apertures at r=20px")
    """
    return int(n_reference_apertures(radius, fwhm, exclusion_buffer))


def generate_aperture_coords(
    center: tuple[float, float],
    radius: float,
    planet_angle: float,
    n_apertures: int,
    max_apertures: int = 200,
    fwhm: float | None = None,
    exclusion_buffer: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate coordinates for reference apertures at a given radius.

    Uses a fixed-size array with masking for JAX compatibility (static shapes).
    Apertures are distributed evenly around the annulus, excluding the planet position.

    Matches VIP's clockwise rotation and angle formula from planet position.

    Args:
        center: Image center (cy, cx) in pixels.
        radius: Radial distance from center in pixels.
        planet_angle: Angle of the planet position in radians.
        n_apertures: Actual number of valid apertures to use.
        max_apertures: Maximum buffer size for static array shape.
        fwhm: Full width half maximum for VIP-style angle calculation. If None,
              uses uniform distribution.
        exclusion_buffer: Gap between test and first reference aperture in
            units of angular step (default 0.5). Prevents PSF wing leakage.

    Returns:
        Tuple of (y_coords, x_coords, mask) where:
            - y_coords: Y coordinates of aperture centers (size max_apertures)
            - x_coords: X coordinates of aperture centers (size max_apertures)
            - mask: Boolean mask indicating valid apertures
    """
    cy, cx = center
    idx_grid = jnp.arange(max_apertures)

    # Angular step calculation
    # VIP formula: angle = 2 * arcsin(fwhm/2/radius)
    if fwhm is not None:
        half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(radius, 0.1), 1.0))
        d_theta = 2.0 * half_angle
    else:
        # Fallback: uniform distribution
        d_theta = 2 * jnp.pi / jnp.maximum(n_apertures + 1, 1)

    # Angles starting from one step past the planet (with optional buffer)
    # Use NEGATIVE rotation (clockwise) to match VIP's convention
    angles = planet_angle - (idx_grid + 1 + exclusion_buffer) * d_theta

    # Compute coordinates
    y_coords = cy + radius * jnp.sin(angles)
    x_coords = cx + radius * jnp.cos(angles)

    # Mask: valid if index < n_apertures
    mask = idx_grid < n_apertures

    return y_coords, x_coords, mask
