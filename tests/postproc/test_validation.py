"""test_validation.py - Geometric and physical validation tests.

Validates geometric conventions, centering, and unit consistency.
Run with: pytest tests/test_validation.py -v
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.transforms import resample_flux

from photomancy.postproc.core.modeling import inject_planet, make_simple_disk


class TestRotationDirection:
    """Verify rotation follows coronagraphoto convention."""

    def test_rotation_matches_coronagraphoto(self):
        """Positive rotation should match coronagraphoto's behavior.

        With origin="lower", positive rotation_deg rotates the image content
        counterclockwise. This matches coronagraphoto exactly.
        """
        # Create image with a dot on the right (west, +column).
        img = jnp.zeros((33, 33))
        img = img.at[16, 26].set(1.0)  # Center at (16,16), dot at x=+10

        # Apply rotation - should match coronagraphoto
        rotated = resample_flux(img, 1.0, 1.0, (33, 33), rotation_deg=90.0)

        # With origin="lower", +90 deg moves right (west) to up (north).
        peak = jnp.unravel_index(jnp.argmax(rotated), rotated.shape)
        # Peak should be at row=26 (north), column=16 (center).
        assert peak == (26, 16), f"Expected (26,16), got {peak}"


class TestInjectionCentering:
    """Verify planet injection uses correct geometric center."""

    def test_center_injection_no_shift(self):
        """Injecting at geometric center should cause zero shift."""
        # 11x11 image. Geometric center is index 5.0 (0-indexed)
        img = jnp.zeros((11, 11))
        psf = jnp.zeros((11, 11))
        psf = psf.at[5, 5].set(1.0)

        # Inject at exact geometric center (5.0, 5.0)
        result = inject_planet(img, psf, flux=1.0, pos=(5.0, 5.0))

        # Peak should remain exactly at (5, 5) with value 1.0
        peak_pos = jnp.unravel_index(jnp.argmax(result), result.shape)
        assert peak_pos == (5, 5), f"Expected (5, 5), got {peak_pos}"
        assert jnp.isclose(jnp.max(result), 1.0, atol=1e-4)

    def test_even_sized_image_center(self):
        """Even-sized image: center is between pixels."""
        # 10x10 image. Geometric center is 4.5
        img = jnp.zeros((10, 10))
        psf = jnp.zeros((10, 10))
        # PSF at (4, 4) and (5, 5) equally weighted
        psf = psf.at[4, 4].set(0.5)
        psf = psf.at[5, 5].set(0.5)

        # Inject at center (4.5, 4.5) - should not shift
        result = inject_planet(img, psf, flux=1.0, pos=(4.5, 4.5))

        # Both peaks should be preserved
        assert result[4, 4] > 0.4
        assert result[5, 5] > 0.4


class TestDiskOrientation:
    """Verify disk position angle follows astronomical convention."""

    @pytest.mark.parametrize("pa_deg", [30.0, 45.0, 90.0, 120.0])
    def test_major_axis_sky_pa(self, pa_deg):
        """Measure sky PA from brightness moments with east = -column.

        The 90-degree ring lies along a constant-row line, but its mirror
        symmetry cannot distinguish the PA sign; the oblique cases can.
        """
        disk = np.asarray(
            make_simple_disk(
                (161, 161), radius=30, inclination_deg=60, width=4, pa_deg=pa_deg
            )
        )
        row, col = np.indices(disk.shape)
        total = disk.sum()
        east = -(col - (disk * col).sum() / total)
        north = row - (disk * row).sum() / total
        moments = (
            np.array(
                [
                    [(disk * east**2).sum(), (disk * east * north).sum()],
                    [(disk * east * north).sum(), (disk * north**2).sum()],
                ]
            )
            / total
        )
        _, axes = np.linalg.eigh(moments)
        major_east, major_north = axes[:, -1]
        measured_pa = np.degrees(np.arctan2(major_east, major_north)) % 180
        error_deg = (measured_pa - pa_deg + 90) % 180 - 90

        # The projected width is >= 2 pixels and the image extends > 12
        # radial sigmas beyond the ring. A 0.1-degree tolerance allows pixel
        # sampling and float32 roundoff while resolving the mirrored angles.
        assert abs(error_deg) < 0.1, (
            f"PA={pa_deg}: measured sky PA={measured_pa:.6f}, "
            f"error={error_deg:.6f} degrees"
        )

    def test_pa_zero_is_north(self):
        """PA=0 should produce a vertical (North-South) disk."""
        shape = (101, 101)
        # Edge-on disk (needle) with PA=0
        disk = make_simple_disk(shape, radius=20, inclination_deg=85, width=2, pa_deg=0)

        # Calculate variance along each axis relative to center
        y, x = jnp.indices(shape)
        center = 50
        total = jnp.sum(disk)
        var_y = jnp.sum(disk * (y - center) ** 2) / total
        var_x = jnp.sum(disk * (x - center) ** 2) / total

        # If PA=0 is North (vertical), variance in Y > variance in X
        assert var_y > var_x, (
            f"PA=0 produced horizontal disk (var_y={var_y:.1f}, var_x={var_x:.1f})"
        )

    def test_pa_90_is_east(self):
        """PA=90 should produce a horizontal (East-West) disk."""
        shape = (101, 101)
        disk = make_simple_disk(
            shape, radius=20, inclination_deg=85, width=2, pa_deg=90
        )

        y, x = jnp.indices(shape)
        center = 50
        total = jnp.sum(disk)
        var_y = jnp.sum(disk * (y - center) ** 2) / total
        var_x = jnp.sum(disk * (x - center) ** 2) / total

        # If PA=90 is East (horizontal), variance in X > variance in Y
        assert var_x > var_y, (
            f"PA=90 produced vertical disk (var_y={var_y:.1f}, var_x={var_x:.1f})"
        )
