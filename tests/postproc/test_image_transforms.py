"""test_image_transforms.py - Tests for image transformation utilities.

Run with: pytest tests/test_image_transforms.py -v
"""

import jax
import jax.numpy as jnp
from hwoutils.transforms import (
    ccw_rotation_matrix,
    resample_flux,
    shift_image,
)


class TestShiftImage:
    """Tests for the shift_image function."""

    def test_zero_shift_identity(self):
        """Shifting by (0, 0) should return the original image."""
        image = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
        shifted = shift_image(image, 0.0, 0.0)
        assert jnp.allclose(image, shifted, atol=1e-5)

    def test_roundtrip_shift(self):
        """Shifting by +dx then -dx should approximately recover original.

        Note: Cubic spline interpolation accumulates error, so we test
        correlation rather than exact match. ~0.90 correlation is expected.
        """
        image = jax.random.normal(jax.random.PRNGKey(1), (64, 64))
        shifted = shift_image(image, 1.5, -0.8)
        recovered = shift_image(shifted, -1.5, 0.8)
        # Focus on center region to avoid boundary effects
        center = slice(10, 54)
        orig = image[center, center].ravel()
        rec = recovered[center, center].ravel()
        # Interpolation adds smoothing - check structure is preserved
        correlation = jnp.corrcoef(orig, rec)[0, 1]
        assert correlation > 0.85, f"Poor correlation: {correlation}"

    def test_flux_conservation(self):
        """Total flux should be approximately conserved for small shifts."""
        # Create positive image with padding to reduce boundary effects
        image = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (64, 64)))
        shifted = shift_image(image, 2.3, -1.7)
        # Allow some loss at boundaries - cubic spline can have ringing
        ratio = jnp.sum(shifted) / jnp.sum(image)
        assert 0.90 < ratio < 1.10

    def test_integer_shift_exact(self):
        """Integer pixel shifts should be exact (within interpolation)."""
        image = jnp.zeros((32, 32))
        image = image.at[16, 16].set(1.0)  # Single bright pixel at center
        shifted = shift_image(image, 3, 2)
        # Peak should be at (19, 18)
        peak_pos = jnp.unravel_index(jnp.argmax(shifted), shifted.shape)
        assert peak_pos == (19, 18)

    def test_gradient_flow(self):
        """Verify gradients flow through shift_image."""

        def loss_fn(shift_y, shift_x):
            image = jnp.ones((32, 32))
            shifted = shift_image(image, shift_y, shift_x)
            return jnp.sum(shifted**2)

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grads = grad_fn(1.0, 1.0)
        # Should return finite gradients
        assert all(jnp.isfinite(g) for g in grads)


class TestResampleFlux:
    """Tests for flux-conserving resampling."""

    def test_flux_conservation_upsample(self):
        """Total flux should be conserved when upsampling."""
        image = jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (32, 32)))
        resampled = resample_flux(image, 1.0, 0.5, (64, 64))
        orig_flux = jnp.sum(image)
        new_flux = jnp.sum(resampled)
        assert jnp.abs(orig_flux - new_flux) / orig_flux < 0.05

    def test_flux_conservation_downsample(self):
        """Total flux should be conserved when downsampling."""
        image = jnp.abs(jax.random.normal(jax.random.PRNGKey(4), (64, 64)))
        resampled = resample_flux(image, 1.0, 2.0, (32, 32))
        orig_flux = jnp.sum(image)
        new_flux = jnp.sum(resampled)
        assert jnp.abs(orig_flux - new_flux) / orig_flux < 0.05

    def test_identity_resample(self):
        """Same pixscale and shape should return ~original."""
        image = jax.random.normal(jax.random.PRNGKey(5), (32, 32))
        resampled = resample_flux(image, 1.0, 1.0, (32, 32))
        assert jnp.allclose(image, resampled, atol=1e-4)


class TestRotationMatrix:
    """Tests for rotation matrix generation."""

    def test_identity_rotation(self):
        """0 degree rotation should give identity matrix."""
        R = ccw_rotation_matrix(0.0)
        expected = jnp.eye(2)
        assert jnp.allclose(R, expected, atol=1e-6)

    def test_90_degree_rotation(self):
        """90 degree CCW rotation matrix."""
        R = ccw_rotation_matrix(90.0)
        expected = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        assert jnp.allclose(R, expected, atol=1e-6)

    def test_rotation_orthogonal(self):
        """Rotation matrix should be orthogonal."""
        R = ccw_rotation_matrix(37.5)
        should_be_identity = R @ R.T
        assert jnp.allclose(should_be_identity, jnp.eye(2), atol=1e-6)
