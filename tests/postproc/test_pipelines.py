"""test_pipelines.py - Tests for yield simulation pipelines.

Run with: pytest tests/test_pipelines.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.postproc.core.modeling import subtract_disk, subtract_star
from photomancy.postproc.pipelines import (
    calculate_yield_snr,
    klip_subtract,
)


class TestSubtractStar:
    """Tests for star subtraction primitive."""

    def test_exact_subtraction(self):
        """Should exactly subtract star model."""
        science = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        star_model = jnp.array([[5.0, 10.0], [15.0, 20.0]])

        residual = subtract_star(science, star_model)
        expected = jnp.array([[5.0, 10.0], [15.0, 20.0]])

        assert jnp.allclose(residual, expected)

    def test_preserves_planet_signal(self):
        """Planet signal not in models should remain."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        star_model = 100 * jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 200)
        planet = 10 * jnp.exp(-((y - 40) ** 2 + (x - 50) ** 2) / 10)
        science = star_model + planet

        residual = subtract_star(science, star_model)

        assert jnp.allclose(residual, planet, atol=1e-5)


class TestSubtractDisk:
    """Tests for disk subtraction primitive."""

    def test_with_disk_model(self):
        """Should subtract disk model."""
        science = jnp.array([[100.0]])
        disk_model = jnp.array([[30.0]])

        residual = subtract_disk(science, disk_model)
        assert jnp.isclose(residual[0, 0], 70.0)


class TestKLIPSubtract:
    """Tests for KLIP subtraction pipeline."""

    def test_removes_common_structure(self):
        """KLIP should remove structure present in reference cube."""
        ny, nx = 32, 32
        n_frames = 10

        # Common structure
        y, x = jnp.mgrid[:ny, :nx]
        structure = jnp.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 50)

        # Reference cube = structure + noise
        key = jax.random.PRNGKey(0)
        ref_cube = structure[None, :, :] + 0.1 * jax.random.normal(
            key, (n_frames, ny, nx)
        )

        # Science = structure + planet
        planet = 0.5 * jnp.exp(-((y - 20) ** 2 + (x - 25) ** 2) / 10)
        science = structure + planet

        residual = klip_subtract(science, ref_cube, n_modes=3)

        # Planet should be preserved
        planet_region = residual[17:24, 22:28]
        assert jnp.max(planet_region) > 0.1


class TestCalculateYieldSNR:
    """Tests for end-to-end yield SNR calculation."""

    def test_star_method(self):
        """Test star method (formerly faux_rdi)."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        star_model = 1000 * jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 100)
        planet_pos = (40.0, 45.0)
        planet = 50 * jnp.exp(
            -((y - planet_pos[0]) ** 2 + (x - planet_pos[1]) ** 2) / 5
        )

        # Add noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (ny, nx)) * 2
        science = star_model + planet + noise

        # Calculate SNR
        positions = jnp.array([[40.0, 45.0]])
        snrs = calculate_yield_snr(
            science, positions, fwhm=4.0, star_model=star_model, method="star"
        )

        # Should detect planet
        assert snrs.shape == (1,)
        assert snrs[0] > 5  # Should be clearly detected

    def test_klip_method(self):
        """Test KLIP method through convenience wrapper."""
        ny, nx = 32, 32
        n_frames = 10

        key = jax.random.PRNGKey(1)
        ref_cube = jax.random.normal(key, (n_frames, ny, nx))
        science = jax.random.normal(jax.random.PRNGKey(2), (ny, nx))

        # Position must be at radius >= fwhm from center for valid SNR
        # Center is (15.5, 15.5), so (20, 20) gives radius ~ 6.4 > fwhm=3
        positions = jnp.array([[20.0, 20.0]])
        snrs = calculate_yield_snr(
            science,
            positions,
            fwhm=3.0,
            reference_cube=ref_cube,
            method="klip",
            n_modes=3,
        )

        assert snrs.shape == (1,)
        assert jnp.isfinite(snrs[0])

    def test_invalid_method_raises(self):
        """Should raise for unknown method."""
        science = jnp.zeros((32, 32))
        positions = jnp.array([[16.0, 16.0]])

        with pytest.raises(ValueError, match="Unknown method"):
            calculate_yield_snr(science, positions, fwhm=3.0, method="invalid")

    def test_missing_star_model_raises(self):
        """Star method should require star_model."""
        science = jnp.zeros((32, 32))
        positions = jnp.array([[16.0, 16.0]])

        with pytest.raises(ValueError, match="star_model required"):
            calculate_yield_snr(science, positions, fwhm=3.0, method="star")


class TestYieldEstimatorArgument:
    """calculate_yield_snr accepts an optional detection estimator."""

    def _scene(self):
        """Science image, star model, positions, fwhm for the star method."""
        key = jax.random.PRNGKey(11)
        star = 100.0 * jnp.exp(
            -(
                (jnp.arange(101)[:, None] - 50.0) ** 2
                + (jnp.arange(101)[None, :] - 50.0) ** 2
            )
            / (2 * 8.0**2)
        )
        science = star + jax.random.normal(key, (101, 101))
        positions = jnp.array([[68.0, 50.0], [50.0, 30.0]])
        return science, star, positions, 5.0

    def test_default_matches_explicit_none(self):
        """Omitting estimator and passing None are identical."""
        science, star, positions, fwhm = self._scene()
        a = calculate_yield_snr(science, positions, fwhm, star_model=star)
        b = calculate_yield_snr(
            science, positions, fwhm, star_model=star, estimator=None
        )
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_mawet_estimator_object_matches_default(self):
        """A Mawet estimator with the same config reproduces the default path."""
        from photomancy.postproc import DetectionEstimator

        science, star, positions, fwhm = self._scene()
        default = calculate_yield_snr(science, positions, fwhm, star_model=star)
        est = DetectionEstimator.mawet(fwhm=fwhm)
        via_est = calculate_yield_snr(
            science, positions, fwhm, star_model=star, estimator=est
        )
        np.testing.assert_array_equal(np.asarray(default), np.asarray(via_est))

    def test_detection_estimator_returns_statistic_array(self):
        """A composed DetectionEstimator yields its statistic array."""
        from photomancy.postproc.core.detection.estimator import DetectionEstimator
        from photomancy.postproc.core.detection.filters import ApertureFilter
        from photomancy.postproc.core.detection.samplers import ApertureSampler
        from photomancy.postproc.core.detection.significance import GrubbsTest
        from photomancy.postproc.core.photometry import make_aperture_kernel

        science, star, positions, fwhm = self._scene()
        composed = DetectionEstimator(
            filter=ApertureFilter(
                kernel=make_aperture_kernel(
                    radius=fwhm / 2.0, soft=True, sharpness=10.0
                ),
                order=3,
            ),
            sampler=ApertureSampler(fwhm=fwhm),
            test=GrubbsTest(),
        )
        out = calculate_yield_snr(
            science, positions, fwhm, star_model=star, estimator=composed
        )
        assert out.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_matched_filter_estimator_uses_validity_map(self):
        """Estimator output is the oracle; footprint NaN gates and parity are exact."""
        from photomancy.postproc import DetectionEstimator

        science, star, positions, fwhm = self._scene()
        validity = jnp.ones_like(science).at[:, :50].set(0.0)
        estimator = DetectionEstimator.matched_filter(
            fwhm, annulus_inner=10.0, annulus_outer=30.0
        )
        out = calculate_yield_snr(
            science,
            positions,
            fwhm,
            star_model=star,
            validity_map=validity,
            estimator=estimator,
        )
        residual = science - star
        expected = estimator(residual, positions, validity).statistic
        unmasked = estimator(residual, positions).statistic
        np.testing.assert_array_equal(out, expected)
        assert bool(jnp.all(jnp.isnan(out)))
        # A narrower excluded region affects the annular noise without touching
        # either candidate footprint, so this case must remain measurable.
        safe_validity = jnp.ones_like(science).at[:, :20].set(0.0)
        safe_out = calculate_yield_snr(
            science,
            positions,
            fwhm,
            star_model=star,
            validity_map=safe_validity,
            estimator=estimator,
        )
        np.testing.assert_array_equal(
            safe_out, estimator(residual, positions, safe_validity).statistic
        )
        assert bool(jnp.all(jnp.isfinite(safe_out)))
        assert not np.array_equal(safe_out, unmasked)
        assert not np.array_equal(out, unmasked)
