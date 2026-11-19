"""test_snr_gradients.py - Gradient/differentiability tests for SNR methods.

Verifies that SNR functions are differentiable (JAX-compatible) for use
in optimization, such as wavefront control or orbit fitting.

Run with: pytest tests/test_snr_gradients.py -v
"""

import jax
import jax.numpy as jnp

from photomancy.postproc.core.detection.estimator import DetectionEstimator


class TestSNRGradients:
    """Gradient verification tests for SNR methods."""

    shape = (100, 100)
    fwhm = 5.0
    pos = jnp.array([[70.0, 50.0]])  # (N, 2) format for new API

    def _make_test_image(self):
        """Create a test image with a Gaussian planet."""
        y, x = jnp.ogrid[: self.shape[0], : self.shape[1]]
        sigma = self.fwhm / 2.355
        return jnp.exp(-((y - 70) ** 2 + (x - 50) ** 2) / (2 * sigma**2))

    def test_matched_filter_snr_is_differentiable(self):
        """Matched-filter SNR should be differentiable for optimization."""
        img = self._make_test_image()

        def loss(image):
            return -DetectionEstimator.matched_filter(self.fwhm)(
                image, self.pos
            ).statistic[0]

        grad = jax.grad(loss)(img)
        gnorm = float(jnp.linalg.norm(grad))

        assert gnorm > 1e-6, f"Matched-filter SNR gradient is zero (norm={gnorm:.2e})"
        assert jnp.isfinite(gnorm), "Matched-filter SNR gradient contains NaN/Inf"

    def test_mawet_snr_is_differentiable(self):
        """Mawet SNR should be differentiable for optimization."""
        img = self._make_test_image()

        def loss(image):
            return -DetectionEstimator.mawet(self.fwhm)(image, self.pos).statistic[0]

        grad = jax.grad(loss)(img)
        gnorm = float(jnp.linalg.norm(grad))

        assert gnorm > 1e-6, f"Mawet SNR gradient is zero (norm={gnorm:.2e})"
        assert jnp.isfinite(gnorm), "Mawet SNR gradient contains NaN/Inf"

    def test_gradient_direction_matched_filter(self):
        """Matched-filter SNR gradient should point toward increasing planet flux."""
        img = self._make_test_image()

        def snr_value(image):
            return DetectionEstimator.matched_filter(self.fwhm)(
                image, self.pos
            ).statistic[0]

        grad = jax.grad(snr_value)(img)
        # Gradient at planet center should be positive (increasing flux increases SNR)
        assert grad[70, 50] > 0, "Gradient at planet center should be positive"

    def test_gradient_direction_mawet(self):
        """Mawet SNR gradient should point toward increasing planet flux."""
        img = self._make_test_image()

        def snr_value(image):
            return DetectionEstimator.mawet(self.fwhm)(image, self.pos).statistic[0]

        grad = jax.grad(snr_value)(img)
        # Gradient at planet center should be positive
        assert grad[70, 50] > 0, "Gradient at planet center should be positive"
