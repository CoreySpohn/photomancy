"""Yield simulation pipelines for fast SNR calculation.

Provides high-level workflows for yield estimation:

- calculate_yield_snr: End-to-end subtraction + SNR calculation
- klip_subtract: PCA/KLIP PSF subtraction

For subtraction primitives, see photomancy.postproc.core.modeling:
- subtract_star, subtract_disk

All functions are JIT-compiled and differentiable.
"""

import functools

import jax
import jax.numpy as jnp

from photomancy.postproc.core.detection.estimator import DetectionEstimator
from photomancy.postproc.core.modeling import subtract_disk, subtract_star
from photomancy.postproc.core.pca import get_pca_basis, pca_subtract


@functools.partial(jax.jit, static_argnames=["n_modes"])
def klip_subtract(
    science: jnp.ndarray,
    reference_cube: jnp.ndarray,
    n_modes: int = 5,
) -> jnp.ndarray:
    """KLIP/PCA PSF subtraction.

    Uses PCA to build a stellar PSF model from a reference cube
    (e.g., images at different roll angles) and subtracts it.

    This is the most physically realistic stellar subtraction mode
    but also the slowest.

    Args:
        science: Science observation (electrons).
        reference_cube: Reference library of shape (n_frames, ny, nx).
            Typically star-only images at different roll angles.
        n_modes: Number of PCA modes to use for subtraction.

    Returns:
        Residual image after KLIP subtraction.

    Example::

        # ADI with KLIP
        residual = klip_subtract(science, roll_cube, n_modes=10)
        # If you also need disk subtraction, do it after:
        residual = subtract_disk(residual, disk_model)
    """
    basis, mean_ref = get_pca_basis(reference_cube, n_modes)
    return pca_subtract(science, basis, mean_ref)


def calculate_yield_snr(
    science: jnp.ndarray,
    planet_positions: jnp.ndarray,
    fwhm: float,
    star_model: jnp.ndarray = None,
    disk_model: jnp.ndarray = None,
    reference_cube: jnp.ndarray = None,
    n_modes: int = 5,
    method: str = "star",
    star_scale: float = 1.0,
    disk_scale: float = 1.0,
    exclusion_buffer: float = 0.5,
    validity_map: jnp.ndarray = None,
    estimator: DetectionEstimator | None = None,
) -> jnp.ndarray:
    """End-to-end yield SNR calculation.

    Convenience function that performs subtraction and SNR calculation
    in one call. Selects the appropriate subtraction method based on
    the `method` argument.

    Args:
        science: Observed image (electrons).
        planet_positions: Planet positions as (N, 2) array of (y, x) coords.
        fwhm: PSF FWHM in pixels.
        star_model: Noiseless star expectation (required for 'star' method).
        disk_model: Optional noiseless disk expectation.
        reference_cube: Reference library for KLIP (required for 'klip'/'rdi').
        n_modes: Number of PCA modes (for klip method only).
        method: Subtraction method - "star", "rdi", or "klip".
        star_scale: Scaling factor for star model (default 1.0).
        disk_scale: Scaling factor for disk model (default 1.0).
        exclusion_buffer: Gap between test and reference apertures in units
            of angular step (default 0.5). Prevents PSF wing leakage.
        validity_map: Optional 2D mask (1=valid, 0=invalid) to exclude
            known companions, bad pixels, or edge regions.
        estimator: Optional DetectionEstimator. The default builds a Mawet
            estimator with fwhm and exclusion_buffer. A supplied estimator
            uses its own configuration.

    Returns:
        Detection statistic values for each planet position.

    Example::

        # Fast yield calculation with static PSF
        snrs = calculate_yield_snr(
            science_image,
            planet_positions,
            fwhm=4.5,
            star_model=star_expectation,
            disk_model=disk_expectation,
            method="star"
        )
    """
    if method == "star":
        if star_model is None:
            raise ValueError("star_model required for 'star' method")
        residual = subtract_star(science, star_model, star_scale)
    elif method == "rdi":
        if reference_cube is None:
            raise ValueError("reference_cube required for 'rdi' method")
        # Handle both 2D (single reference) and 3D (cube) inputs
        if reference_cube.ndim == 2:
            reference = reference_cube
        else:
            reference = reference_cube[0]
        # Use subtract_star - it's the same operation (science - scale * ref)
        residual = subtract_star(science, reference, star_scale)
    elif method == "klip":
        if reference_cube is None:
            raise ValueError("reference_cube required for 'klip' method")
        residual = klip_subtract(science, reference_cube, n_modes)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'star', 'rdi', or 'klip'")

    # Apply disk subtraction if provided
    if disk_model is not None:
        residual = subtract_disk(residual, disk_model, disk_scale)

    if estimator is None:
        estimator = DetectionEstimator.mawet(fwhm, exclusion_buffer=exclusion_buffer)
    return estimator(residual, planet_positions, validity_map).statistic
