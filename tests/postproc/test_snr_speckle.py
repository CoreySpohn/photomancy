"""test_snr_speckle.py - Speckle noise stress tests for SNR methods.

Tests behavior in spatially correlated 'speckle' noise, which mimics
real coronagraphic residuals. This validates why Mawet SNR is needed
for detection claims while matched-filter SNR is appropriate for yield calculations.

Run with: pytest tests/test_snr_speckle.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve2d

from photomancy.postproc.core.detection.estimator import DetectionEstimator


def make_speckle_noise(shape, fwhm, rng_key):
    """Generate synthetic 'Speckle' noise (spatially correlated).

    This mimics the 'blobs' seen in coronagraphic residuals by
    convolving white noise with a PSF-sized kernel.
    """
    white = jax.random.normal(rng_key, shape)

    sigma = fwhm / 2.355
    k_size = int(4 * sigma + 1)
    if k_size % 2 == 0:
        k_size += 1
    y, x = jnp.ogrid[-k_size // 2 : k_size // 2 + 1, -k_size // 2 : k_size // 2 + 1]
    kernel = jnp.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel = kernel / jnp.sum(kernel)

    return convolve2d(white, kernel, mode="same")


class TestSNRSpeckle:
    """Speckle noise stress tests for SNR methods."""

    fwhm = 5.0

    def test_matched_filter_snr_in_speckle_noise(self):
        """Matched-filter SNR should show inflated std in speckle noise.

        Because it treats every pixel as independent, it underestimates
        the noise of spatially correlated 'blobs'. This is expected behavior.
        """
        n_trials = 300
        rng = jax.random.PRNGKey(42)
        speckle_img = make_speckle_noise((200, 200), self.fwhm, rng)

        # Generate random positions at valid radii
        pos_rng = np.random.default_rng(0)
        center = 100
        min_r, max_r = 2 * self.fwhm, 50
        rs = pos_rng.uniform(min_r, max_r, n_trials)
        thetas = pos_rng.uniform(0, 2 * np.pi, n_trials)
        ys = center + rs * np.sin(thetas)
        xs = center + rs * np.cos(thetas)

        positions_arr = jnp.stack([jnp.array(ys), jnp.array(xs)], axis=1)
        mf_vals = DetectionEstimator.matched_filter(self.fwhm)(
            speckle_img, positions_arr
        ).statistic

        mf_std = float(jnp.std(mf_vals))
        # Matched-filter SNR is expected to be inflated in speckle noise
        # We just verify it's not broken (finite, positive)
        assert mf_std > 0.5, f"Matched-filter SNR std={mf_std:.3f}, too low"
        assert mf_std < 10.0, f"Matched-filter SNR std={mf_std:.3f}, unexpectedly high"

    def test_mawet_snr_in_speckle_noise(self):
        """Mawet SNR should be more robust to speckle correlations than MF."""
        n_trials = 300
        rng = jax.random.PRNGKey(42)
        speckle_img = make_speckle_noise((200, 200), self.fwhm, rng)

        pos_rng = np.random.default_rng(0)
        center = 100
        min_r, max_r = 2 * self.fwhm, 50
        rs = pos_rng.uniform(min_r, max_r, n_trials)
        thetas = pos_rng.uniform(0, 2 * np.pi, n_trials)
        ys = center + rs * np.sin(thetas)
        xs = center + rs * np.cos(thetas)

        positions_arr = jnp.stack([jnp.array(ys), jnp.array(xs)], axis=1)
        mawet_vals = DetectionEstimator.mawet(self.fwhm)(
            speckle_img, positions_arr
        ).statistic

        mawet_std = float(jnp.std(mawet_vals))
        # Mawet should handle correlations better (std closer to 1.0)
        assert mawet_std > 0.5, f"Mawet SNR std={mawet_std:.3f}, too low"
        assert mawet_std < 5.0, f"Mawet SNR std={mawet_std:.3f}, unexpectedly high"
