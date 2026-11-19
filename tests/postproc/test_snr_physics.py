"""test_snr_physics.py - Physics-based verification suite for SNR methods.

Tests four critical failure modes:
1. Background Invariance (The Bug Fix Check)
2. Linearity (Flux scaling)
3. White Noise Calibration (Theory check)
4. Empty Field Statistics (False positive rate)

Run with: pytest tests/test_snr_physics.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np

from photomancy.postproc.core.detection.estimator import DetectionEstimator


def make_gaussian_psf(shape, center, fwhm, total_flux):
    """Generates a synthetic Gaussian PSF normalized so its sum equals total_flux."""
    y, x = jnp.ogrid[: shape[0], : shape[1]]
    sigma = fwhm / 2.355
    r2 = (y - center[0]) ** 2 + (x - center[1]) ** 2
    psf = jnp.exp(-0.5 * r2 / sigma**2)
    normalization = 2 * jnp.pi * sigma**2
    return (psf / normalization) * total_flux


class TestSNRPhysics:
    """Physics-based tests for SNR methods."""

    # Common test parameters
    shape = (100, 100)
    fwhm = 5.0
    pos = jnp.array([[70.0, 50.0]])  # (N, 2) format for new API

    def test_background_invariance(self):
        """Matched-filter SNR should not change when a constant background is added."""
        noise_sigma = 10.0
        rng = jax.random.PRNGKey(0)
        noise = jax.random.normal(rng, self.shape) * noise_sigma

        img_clean = (
            make_gaussian_psf(self.shape, (70.0, 50.0), self.fwhm, total_flux=1000.0)
            + noise
        )
        img_bg = img_clean + 5000.0  # Add constant background

        snr_clean = float(
            DetectionEstimator.matched_filter(self.fwhm)(img_clean, self.pos).statistic[
                0
            ]
        )
        snr_bg = float(
            DetectionEstimator.matched_filter(self.fwhm)(img_bg, self.pos).statistic[0]
        )

        rel_diff = abs(snr_clean - snr_bg) / max(abs(snr_clean), 1e-10)
        assert rel_diff < 0.01, f"SNR changed by {rel_diff * 100:.2f}% with background"

    def test_linearity(self):
        """SNR should scale linearly with flux (double flux = double SNR)."""
        base_flux = 500.0
        noise_sigma = 10.0

        rng = jax.random.PRNGKey(1)
        noise_map = jax.random.normal(rng, self.shape) * noise_sigma

        img_1x = noise_map + make_gaussian_psf(
            self.shape, (70.0, 50.0), self.fwhm, base_flux
        )
        img_2x = noise_map + make_gaussian_psf(
            self.shape, (70.0, 50.0), self.fwhm, base_flux * 2
        )

        snr_1x = float(
            DetectionEstimator.matched_filter(self.fwhm)(img_1x, self.pos).statistic[0]
        )
        snr_2x = float(
            DetectionEstimator.matched_filter(self.fwhm)(img_2x, self.pos).statistic[0]
        )

        ratio = snr_2x / snr_1x
        assert 1.8 < ratio < 2.2, f"Linearity broken: ratio={ratio:.3f}, expected ~2.0"

    def test_white_noise_calibration(self):
        """MF and Mawet SNR should agree within expected bounds in white noise."""
        flux = 500.0
        noise_sigma = 10.0

        rng = jax.random.PRNGKey(2)
        img = jax.random.normal(rng, self.shape) * noise_sigma
        img = img + make_gaussian_psf(self.shape, (70.0, 50.0), self.fwhm, flux)

        mf_snr = float(
            DetectionEstimator.matched_filter(self.fwhm)(img, self.pos).statistic[0]
        )
        m_snr = float(DetectionEstimator.mawet(self.fwhm)(img, self.pos).statistic[0])

        ratio = mf_snr / m_snr
        assert 0.8 <= ratio <= 2.0, (
            f"MatchedFilter/Mawet ratio={ratio:.2f}, expected 0.8-2.0"
        )

    def test_empty_field_matched_filter_snr(self):
        """Matched-filter SNR on pure noise should have std ~1.0 (calibrated FPR)."""
        n_trials = 500
        rng = jax.random.PRNGKey(3)
        noise_img = jax.random.normal(rng, (200, 200))

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
            noise_img, positions_arr
        ).statistic

        mf_std = float(jnp.std(mf_vals))
        assert 0.7 < mf_std < 1.5, f"Matched-filter SNR std={mf_std:.3f}, expected ~1.0"

    def test_empty_field_mawet_snr(self):
        """Mawet SNR on pure noise should have std ~1.0 (calibrated FPR)."""
        n_trials = 500
        rng = jax.random.PRNGKey(4)
        noise_img = jax.random.normal(rng, (200, 200))

        # Generate random positions at valid radii
        pos_rng = np.random.default_rng(0)
        center = 100
        min_r, max_r = 2 * self.fwhm, 50
        rs = pos_rng.uniform(min_r, max_r, n_trials)
        thetas = pos_rng.uniform(0, 2 * np.pi, n_trials)
        ys = center + rs * np.sin(thetas)
        xs = center + rs * np.cos(thetas)

        positions_arr = jnp.stack([jnp.array(ys), jnp.array(xs)], axis=1)
        mawet_vals = DetectionEstimator.mawet(self.fwhm)(
            noise_img, positions_arr
        ).statistic

        mawet_std = float(jnp.std(mawet_vals))
        assert 0.7 < mawet_std < 1.5, f"Mawet SNR std={mawet_std:.3f}, expected ~1.0"
