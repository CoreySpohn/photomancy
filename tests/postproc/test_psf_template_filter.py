"""Tests for the PSF-template matched filter and per-candidate binding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.postproc.core.detection.estimator import DetectionEstimator
from photomancy.postproc.core.detection.filters import PSFTemplateFilter
from photomancy.postproc.core.detection.samplers import ApertureSampler
from photomancy.postproc.core.detection.significance import TwoSampleTTest
from photomancy.postproc.core.geometry import n_reference_apertures
from photomancy.postproc.templates import ArrayTemplateProvider

STAMP_SIZE = 15
FWHM = 5.0


def _gaussian_stamp(size: int = STAMP_SIZE, fwhm: float = FWHM) -> jnp.ndarray:
    """Peak-normalized Gaussian template stamp of shape (size, size)."""
    c = (size - 1) / 2.0
    y, x = jnp.mgrid[:size, :size]
    sigma = fwhm / 2.355
    return jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))


class TestUnboundFilterRaises:
    """Evaluating a filter before bind() is a usage error, not a silent NaN."""

    def test_unbound_evaluate_raises_valueerror(self):
        """evaluate() on an unbound filter raises ValueError naming unbound."""
        provider = ArrayTemplateProvider(_gaussian_stamp())
        filt = PSFTemplateFilter(provider=provider)
        image = jnp.zeros((64, 64))
        with pytest.raises(ValueError, match="unbound"):
            filt.evaluate(image, jnp.array([32.0, 32.0]))


class TestAmplitudeRecovery:
    """The bound filter recovers an injected template's amplitude exactly."""

    def test_recovers_injected_amplitude(self):
        """3.7x the stamp pasted on a zero background recovers 3.7."""
        stamp = _gaussian_stamp()
        half = stamp.shape[0] // 2
        position_yx = jnp.array([32.0, 32.0])
        image = jnp.zeros((64, 64))
        image = image.at[32 - half : 32 + half + 1, 32 - half : 32 + half + 1].set(
            3.7 * stamp
        )
        provider = ArrayTemplateProvider(stamp)
        filt = PSFTemplateFilter(provider=provider).bind(position_yx)
        flux = filt.evaluate(image, position_yx)
        np.testing.assert_allclose(np.asarray(flux), 3.7, rtol=1e-5)


class TestBackgroundInvariance:
    """A uniform additive background does not move the filter response."""

    def test_unwhitened_response_is_background_invariant(self):
        """Adding a constant background leaves the unwhitened response fixed."""
        stamp = _gaussian_stamp()
        position_yx = jnp.array([32.0, 32.0])
        image = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
        provider = ArrayTemplateProvider(stamp)
        filt = PSFTemplateFilter(provider=provider).bind(position_yx)
        baseline = filt.evaluate(image, position_yx)
        shifted = filt.evaluate(image + 12.3, position_yx)
        np.testing.assert_allclose(np.asarray(shifted), np.asarray(baseline), rtol=1e-6)

    def test_whitened_response_is_background_invariant(self):
        """A uniform-variance whitened filter is also background invariant."""
        stamp = _gaussian_stamp()
        position_yx = jnp.array([32.0, 32.0])
        image = jax.random.normal(jax.random.PRNGKey(1), (64, 64))
        noise_variance = jnp.full((64, 64), 2.5)
        provider = ArrayTemplateProvider(stamp)
        filt = PSFTemplateFilter(provider=provider, noise_variance=noise_variance).bind(
            position_yx
        )
        baseline = filt.evaluate(image, position_yx)
        shifted = filt.evaluate(image + 12.3, position_yx)
        np.testing.assert_allclose(np.asarray(shifted), np.asarray(baseline), rtol=1e-6)


class TestUniformWhiteningEquivalence:
    """A uniform noise_variance reduces exactly to the unwhitened response."""

    def test_uniform_variance_matches_unwhitened(self):
        """A constant variance map gives the same response as unwhitened."""
        stamp = _gaussian_stamp()
        position_yx = jnp.array([32.0, 32.0])
        image = jax.random.normal(jax.random.PRNGKey(2), (64, 64))
        provider = ArrayTemplateProvider(stamp)
        unwhitened = PSFTemplateFilter(provider=provider).bind(position_yx)
        whitened = PSFTemplateFilter(
            provider=provider, noise_variance=jnp.full((64, 64), 2.5)
        ).bind(position_yx)
        expected = unwhitened.evaluate(image, position_yx)
        actual = whitened.evaluate(image, position_yx)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


class TestWhiteningDownweightsHotRegion:
    """Inverse-variance weighting discounts a locally corrupted region."""

    def test_whitened_response_closer_to_clean_than_unwhitened(self):
        """A huge-variance mask on a hot half-patch favors the clean signal."""
        stamp = _gaussian_stamp()
        size = stamp.shape[0]
        half = size // 2
        position_yx = jnp.array([32.0, 32.0])
        y0, y1 = 32 - half, 32 + half + 1
        x0, x1 = 32 - half, 32 + half + 1
        hot_rows = slice(y0, 32)  # upper half of the patch, excluding the center row

        clean_image = jnp.zeros((64, 64)).at[y0:y1, x0:x1].set(stamp)
        hot_image = clean_image.at[hot_rows, x0:x1].add(1000.0)

        provider = ArrayTemplateProvider(stamp)
        unwhitened = PSFTemplateFilter(provider=provider).bind(position_yx)
        clean_response = float(unwhitened.evaluate(clean_image, position_yx))
        unwhitened_response = float(unwhitened.evaluate(hot_image, position_yx))

        variance = jnp.ones((64, 64)).at[hot_rows, x0:x1].set(1e8)
        whitened = PSFTemplateFilter(provider=provider, noise_variance=variance).bind(
            position_yx
        )
        whitened_response = float(whitened.evaluate(hot_image, position_yx))

        unwhitened_error = abs(unwhitened_response - clean_response)
        whitened_error = abs(whitened_response - clean_response)
        assert whitened_error < unwhitened_error


class TestComposedEndToEnd:
    """PSFTemplateFilter composes with the standard sampler/test pairing."""

    def test_finite_statistics_on_a_candidate_ring(self):
        """Six same-radius candidates give finite stats and consistent dof."""
        provider = ArrayTemplateProvider(_gaussian_stamp())
        estimator = DetectionEstimator(
            filter=PSFTemplateFilter(provider=provider),
            sampler=ApertureSampler(fwhm=FWHM),
            test=TwoSampleTTest(),
        )
        image = jax.random.normal(jax.random.PRNGKey(3), (101, 101))
        center = jnp.array([50.0, 50.0])
        radius = 20.0
        theta = jnp.linspace(0.0, 2 * jnp.pi, 6, endpoint=False)
        positions = jnp.stack(
            [
                center[0] + radius * jnp.sin(theta),
                center[1] + radius * jnp.cos(theta),
            ],
            axis=1,
        )
        stats = estimator(image, positions)
        assert bool(jnp.all(jnp.isfinite(stats.statistic)))
        assert bool(jnp.all(jnp.isfinite(stats.fpf)))
        expected_dof = int(n_reference_apertures(jnp.asarray(radius), FWHM, 0.5)) - 1
        np.testing.assert_array_equal(
            np.asarray(stats.dof), np.full(6, float(expected_dof))
        )


class TestBindPurity:
    """bind returns a new filter and never mutates the original."""

    def test_bind_returns_provider_stamp_and_leaves_original_unbound(self):
        """Binding two positions yields independently templated filters."""
        stamp = _gaussian_stamp()
        provider = ArrayTemplateProvider(stamp)
        filt = PSFTemplateFilter(provider=provider)
        pos1 = jnp.array([10.0, 12.0])
        pos2 = jnp.array([40.0, 30.0])
        bound1 = filt.bind(pos1)
        bound2 = filt.bind(pos2)
        np.testing.assert_array_equal(
            np.asarray(bound1.template), np.asarray(provider(pos1))
        )
        np.testing.assert_array_equal(
            np.asarray(bound2.template), np.asarray(provider(pos2))
        )
        assert filt.template is None
