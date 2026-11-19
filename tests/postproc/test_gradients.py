"""Tests for JAX differentiability of the SNR functions.

Ensures that gradients can be computed through the Mawet SNR calculation
for optimization applications.
"""

import jax
import jax.numpy as jnp

from photomancy.postproc.core.detection.estimator import DetectionEstimator


def make_simple_image(
    planet_flux: float = 500.0,
    background: float = 100.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a simple test image for gradient testing.

    Returns:
        Tuple of (image, planet_position as (1, 2) array).
    """
    shape = (51, 51)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0

    image = jnp.ones(shape) * background

    # Planet at fixed position
    planet_y, planet_x = cy + 15, cx
    y, x = jnp.ogrid[: shape[0], : shape[1]]
    sigma = 2.0
    planet = planet_flux * jnp.exp(
        -((y - planet_y) ** 2 + (x - planet_x) ** 2) / (2 * sigma**2)
    )
    image = image + planet

    return image, jnp.array([[planet_y, planet_x]])


class TestGradients:
    """Test suite for gradient computation."""

    def test_gradient_wrt_image(self):
        """Gradients should be computable w.r.t. image pixels."""
        image, planet_pos = make_simple_image()
        fwhm = 5.0

        def loss_fn(img):
            return -DetectionEstimator.mawet(fwhm)(img, planet_pos).statistic[0]

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(image)

        # Gradients should exist and be finite
        assert grads.shape == image.shape
        assert jnp.all(jnp.isfinite(grads))

        # Gradients should be non-zero (at least somewhere)
        assert jnp.any(grads != 0), "Gradients should be non-zero"

    def test_gradient_wrt_position(self):
        """Gradients should be computable w.r.t. planet position."""
        image, _ = make_simple_image()
        planet_pos = jnp.array([[25.0 + 15, 25.0]])  # y, x
        fwhm = 5.0

        def loss_fn(pos):
            return -DetectionEstimator.mawet(fwhm)(image, pos).statistic[0]

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(planet_pos)

        # Gradients should exist and be finite
        assert grads.shape == (1, 2)
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_compilation(self):
        """Function should be JIT-compilable."""
        image, planet_pos = make_simple_image()
        fwhm = 5.0

        jitted_fn = jax.jit(
            lambda img: DetectionEstimator.mawet(fwhm)(img, planet_pos).statistic[0]
        )

        # Should not raise
        result = jitted_fn(image)
        assert jnp.isfinite(result)

    def test_vmap_over_images(self):
        """Should support vectorization over multiple images."""
        # Create a batch of images
        batch_size = 3
        images = jnp.stack([make_simple_image()[0] for _ in range(batch_size)])
        planet_pos = jnp.array([[25.0 + 15, 25.0]])
        fwhm = 5.0

        vmapped_fn = jax.vmap(
            lambda img: DetectionEstimator.mawet(fwhm)(img, planet_pos).statistic[0]
        )
        results = vmapped_fn(images)

        assert results.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(results))

    def test_gradient_descent_reduces_negative_snr(self):
        """Simple gradient descent should increase SNR (decrease -SNR)."""
        # Start with lower flux
        image, planet_pos = make_simple_image(planet_flux=200.0)
        fwhm = 5.0

        def loss_fn(img):
            return -DetectionEstimator.mawet(fwhm)(img, planet_pos).statistic[0]

        initial_loss = loss_fn(image)

        # One gradient step
        grads = jax.grad(loss_fn)(image)
        step_size = 0.01
        new_image = image - step_size * grads

        new_loss = loss_fn(new_image)

        # Loss should decrease (SNR should increase)
        assert new_loss < initial_loss, (
            f"Gradient step should reduce loss: {initial_loss} -> {new_loss}"
        )
