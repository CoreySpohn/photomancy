"""Frozen v1.1.1 fused detection cores, kept verbatim as parity oracles.

These are byte-level copies (modulo the golden_ rename) of _snr_batch_core
and _matched_filter_snr_batch_core as of the former standalone v1.1.1 release.
They are the numerical ground truth the composed DetectionEstimator must
reproduce; do not modernize, refactor, or "fix" them.
"""

import functools

import jax
import jax.numpy as jnp
from hwoutils.map_coordinates import map_coordinates

from photomancy.postproc.core.photometry import flux_map


@functools.partial(jax.jit, static_argnums=(4, 5))
def golden_snr_batch_core(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    kernel: jnp.ndarray,
    fwhm: float,
    max_apertures: int,
    order: int,
    exclusion_buffer: float = 0.5,
    validity_map: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled batch SNR calculation (Mawet method).

    Args:
        image: 2D science image.
        positions: (N, 2) array of (y, x) coordinates.
        kernel: Pre-computed aperture kernel.
        fwhm: Full width at half maximum in pixels.
        max_apertures: Maximum buffer size for static shapes.
        order: Interpolation order (1=bilinear, 3=cubic).
        exclusion_buffer: Angular gap between test and first reference.
        validity_map: Optional 2D mask (1=valid, 0=invalid). Off-chip
            locations automatically get 0 via cval boundary handling.

    Returns:
        Tuple of (snr values (N,), valid reference-aperture counts (N,)).
    """
    # Compute flux map ONCE
    fmap = flux_map(image, kernel)
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Default to all-valid if no mask provided
    if validity_map is None:
        validity_map = jnp.ones((ny, nx))

    def _single_snr(planet_pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        py, px = planet_pos[0], planet_pos[1]

        # Extract planet flux
        planet_flux = map_coordinates(fmap, jnp.array([[py], [px]]), order=order)[0]

        # Geometry
        r_pix = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)
        planet_angle = jnp.arctan2(py - cy, px - cx)

        # Number of apertures (VIP formula, with exclusion buffer on both sides)
        half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(r_pix, 0.1), 1.0))
        d_theta = 2.0 * half_angle
        n_theoretical = jnp.floor(2 * jnp.pi / jnp.maximum(d_theta, 0.01))
        # Subtract: 1 for planet position + 2*buffer for gap on each side
        n_actual = jnp.maximum(
            (n_theoretical - 1 - 2 * exclusion_buffer).astype(int), 1
        )

        # Generate reference aperture coordinates with exclusion buffer
        idx_grid = jnp.arange(max_apertures)
        angles = planet_angle - (idx_grid + 1 + exclusion_buffer) * d_theta
        ref_y = cy + r_pix * jnp.sin(angles)
        ref_x = cx + r_pix * jnp.cos(angles)

        # Sample validity map (cval=0.0 auto-excludes off-chip apertures)
        ref_validity = map_coordinates(
            validity_map,
            jnp.stack([ref_y, ref_x]),
            order=0,  # Nearest neighbor for speed
            cval=0.0,  # Off-chip = invalid
        )

        # Unified mask: index valid AND spatially valid
        mask = (idx_grid < n_actual) & (ref_validity > 0.5)

        # Sample background fluxes
        ref_fluxes = map_coordinates(fmap, jnp.stack([ref_y, ref_x]), order=order)

        # Masked statistics using actual valid count
        n_valid = jnp.sum(mask)
        bg_mean = jnp.sum(ref_fluxes * mask) / jnp.maximum(n_valid, 1.0)
        residuals = (ref_fluxes - bg_mean) * mask
        bg_std = jnp.sqrt(jnp.sum(residuals**2) / jnp.maximum(n_valid - 1, 1.0))

        # Small-sample penalty using actual valid count
        penalty = jnp.sqrt(1 + 1 / jnp.maximum(n_valid, 1.0))

        # SNR calculation
        signal = planet_flux - bg_mean
        noise = bg_std * penalty
        snr_val = signal / jnp.maximum(noise, 1e-10)

        # Return NaN for unreliable measurements:
        # - Radius smaller than FWHM (can't fit reference apertures)
        # - Fewer than 3 valid reference apertures (insufficient statistics)
        is_valid = (r_pix >= fwhm) & (n_valid >= 3)
        return jnp.where(is_valid, snr_val, jnp.nan), n_valid.astype(int)

    return jax.vmap(_single_snr)(positions)  # -> (snr (N,), n_valid (N,))


@functools.partial(jax.jit, static_argnums=(4,))
def golden_matched_filter_snr_batch_core(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    fwhm: float,
    kernel_1d: jnp.ndarray,
    order: int,
    annulus_inner: float,
    annulus_outer: float,
) -> jnp.ndarray:
    """JIT-compiled batch matched-filter SNR calculation."""
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    max_radius = jnp.minimum(ny, nx) / 2 - 1

    # Compute filtered image ONCE
    filtered = golden_gaussian_filter_2d(image, kernel_1d)

    # Pre-compute coordinate grids
    y_coords, x_coords = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    r_grid = jnp.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    def _single_snr(planet_pos: jnp.ndarray) -> float:
        py, px = planet_pos[0], planet_pos[1]
        r_planet = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)

        # Default annulus bounds
        default_inner = jnp.maximum(r_planet - fwhm, fwhm)
        default_outer = jnp.minimum(r_planet + fwhm, max_radius)
        default_inner = jnp.minimum(default_inner, default_outer - 1.0)

        inner_r = jnp.where(annulus_inner >= 0, annulus_inner, default_inner)
        outer_r = jnp.where(annulus_outer >= 0, annulus_outer, default_outer)

        # Extract raw signal
        raw_signal = map_coordinates(filtered, jnp.array([[py], [px]]), order=order)[0]

        # Annulus mask
        planet_dist = jnp.sqrt((y_coords - py) ** 2 + (x_coords - px) ** 2)
        annulus_mask = (
            (r_grid >= inner_r) & (r_grid <= outer_r) & (planet_dist > fwhm * 1.5)
        )

        # Background statistics
        masked_vals = jnp.where(annulus_mask, filtered, jnp.nan)
        bg_mean = jnp.nanmean(masked_vals)
        variance = jnp.nanmean(
            jnp.where(annulus_mask, (filtered - bg_mean) ** 2, jnp.nan)
        )
        bg_std = jnp.sqrt(variance)

        bg_mean = jnp.nan_to_num(bg_mean, nan=0.0)
        bg_std = jnp.nan_to_num(bg_std, nan=1.0)

        signal = raw_signal - bg_mean
        return signal / jnp.maximum(bg_std, 1e-10)

    return jax.vmap(_single_snr)(positions)


@jax.jit
def golden_gaussian_filter_2d(
    image: jnp.ndarray, kernel_1d: jnp.ndarray
) -> jnp.ndarray:
    """Apply 2D Gaussian filter using separable convolution."""
    pad_size = len(kernel_1d) // 2

    # Row convolution
    padded = jnp.pad(image, ((0, 0), (pad_size, pad_size)), mode="reflect")
    row_conv = jax.vmap(lambda row: jnp.convolve(row, kernel_1d, mode="valid"))(padded)

    # Column convolution
    padded = jnp.pad(row_conv, ((pad_size, pad_size), (0, 0)), mode="reflect")
    col_conv = jax.vmap(
        lambda col: jnp.convolve(col, kernel_1d, mode="valid"), in_axes=1, out_axes=1
    )(padded)

    return col_conv
