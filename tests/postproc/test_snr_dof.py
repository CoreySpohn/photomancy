"""Tests for the statistic map and Mawet reference-sample degrees of freedom."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from photomancy.postproc.core.detection.estimator import DetectionEstimator


def make_image(key, n=64):
    """Generate a random Gaussian noise image for testing."""
    return jr.normal(key, (n, n))


def test_statistic_map_matches_all_pixel_coordinates():
    """The full map preserves pixel ordering and the coordinate call's statistic."""
    key = jr.PRNGKey(0)
    image = make_image(key, n=24)[:, :20]
    positions = jnp.indices(image.shape).reshape(2, -1).T
    estimator = DetectionEstimator.mawet(fwhm=4.0)
    validity = jnp.ones_like(image).at[:, :8].set(0.0)
    center = jnp.array([10.5, 9.0])
    result = estimator.map(image, validity, center)
    expected = estimator(image, positions, validity, center).statistic.reshape(
        image.shape
    )
    assert result.shape == image.shape
    np.testing.assert_array_equal(
        np.asarray(result).view(np.uint8), np.asarray(expected).view(np.uint8)
    )


def test_dof_decreases_toward_small_separation():
    """Fewer reference apertures yield fewer dof at smaller separations."""
    key = jr.PRNGKey(1)
    image = make_image(key)
    estimator = DetectionEstimator.mawet(fwhm=4.0)
    # center is (31.5, 31.5); r ~ 6 px vs r ~ 25 px
    positions = jnp.array([[31.5, 37.5], [31.5, 56.5]])
    dof = estimator(image, positions).dof
    assert int(dof[0]) < int(dof[1])
    assert int(dof[0]) >= 2


def test_dof_respects_validity_map():
    """Masking half the field via validity_map reduces the degrees of freedom."""
    key = jr.PRNGKey(2)
    image = make_image(key)
    estimator = DetectionEstimator.mawet(fwhm=4.0)
    positions = jnp.array([[31.5, 51.5]])
    dof_all = estimator(image, positions).dof
    validity = jnp.ones_like(image).at[:, :32].set(0.0)  # kill half the field
    dof_masked = estimator(image, positions, validity).dof
    assert int(dof_masked[0]) < int(dof_all[0])


def test_dof_reported_when_statistic_nan():
    """The degrees of freedom survive the invalid-geometry gate."""
    key = jr.PRNGKey(5)
    image = make_image(key)
    inside = jnp.array([[33.0, 31.5]])  # r < fwhm -> NaN statistic
    est = DetectionEstimator.mawet(fwhm=5.0)
    stats = est(image, inside)
    assert jnp.isnan(stats.statistic[0])
    assert int(stats.dof[0]) >= 0
