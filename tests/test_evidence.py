"""Tests for log-evidence calibration to Gaussian detection significance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erfc
from scipy.optimize import brentq
from scipy.special import erfcinv, log_ndtr, ndtri_exp

from photomancy import sigma_from_log_bayes_factor


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_benneke_seager_table(dtype):
    """Recover the rounded Bayes-factor/sigma pairs in Table 2 (2013)."""
    with jax.enable_x64():
        bayes_factors = jnp.array([2.5, 2.9, 8, 12, 21, 53, 150, 43000], dtype=dtype)
        expected = [2.0, 2.1, 2.6, 2.7, 3.0, 3.3, 3.6, 5.0]
        actual = sigma_from_log_bayes_factor(jnp.log(bayes_factors))
        np.testing.assert_allclose(actual, expected, atol=0.06, rtol=0)
        assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_sigma_round_trip(dtype):
    """Recover 1-20 sigma from Bayes factors built independently in float64."""
    with jax.enable_x64():
        sigma = jnp.linspace(1, 20, 191, dtype=jnp.float64)
        log_p = jnp.log(erfc(sigma / jnp.sqrt(2.0)))
        log_b = -1 - log_p - jnp.log(-log_p)
        actual = sigma_from_log_bayes_factor(log_b.astype(dtype))
        np.testing.assert_allclose(actual, sigma, atol=1e-3, rtol=0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_monotonic_and_large_log_bayes_factors(dtype):
    """Significance increases through log B = 1000 without overflowing."""
    with jax.enable_x64():
        log_b = jnp.concatenate(
            [
                jnp.geomspace(1e-12, 1, 500, dtype=dtype),
                jnp.linspace(1.01, 1000, 2000, dtype=dtype),
            ]
        )
        actual = np.asarray(sigma_from_log_bayes_factor(log_b))
        assert np.isfinite(actual).all()
        # Roundoff may give identical float32 values near the branch endpoint.
        assert (np.diff(actual) >= 0).all()
        assert (np.diff(actual[500:]) > 0).all()
        # Independent SciPy inversion checks accuracy beyond the p underflow limit.
        log_p = log_ndtr(-actual.astype(np.float64)) + np.log(2)
        recovered_log_b = -1 - log_p - np.log(-log_p)
        np.testing.assert_allclose(recovered_log_b, log_b, atol=1e-3, rtol=0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_domain_and_branch_endpoint(dtype):
    """Invalid inputs are NaN and B = 1 maps to p = 1/e without a floor."""
    with jax.enable_x64():
        log_b = jnp.array(
            [-jnp.inf, -2, -1e-12, 0, 1e-12, jnp.inf, jnp.nan], dtype=dtype
        )
        actual = np.asarray(sigma_from_log_bayes_factor(log_b))
        assert np.isnan(actual[:3]).all()
        np.testing.assert_allclose(actual[3], np.sqrt(2) * erfcinv(1 / np.e), atol=1e-6)
        assert actual[4] >= actual[3]
        assert np.isposinf(actual[5])
        assert np.isnan(actual[6])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_vmap_and_array_shapes(dtype):
    """Scalar, matrix, jit and vmap calls agree in both floating dtypes."""
    with jax.enable_x64():
        log_b = jnp.array([[-1, 0, 1e-8], [10, 700, 1000]], dtype=dtype)
        eager = sigma_from_log_bayes_factor(log_b)
        compiled = jax.jit(sigma_from_log_bayes_factor)(log_b)
        mapped = jax.jit(jax.vmap(jax.vmap(sigma_from_log_bayes_factor)))(log_b)
        for actual in (compiled, mapped):
            assert actual.shape == log_b.shape
            assert actual.dtype == dtype
            np.testing.assert_allclose(
                actual, eager, atol=1e-6, rtol=1e-6, equal_nan=True
            )
        scalar = sigma_from_log_bayes_factor(0.0)
        assert scalar.shape == ()
        np.testing.assert_allclose(scalar, eager[0, 1], atol=1e-6)


def test_extreme_tail_reference():
    """Remain accurate without a display cap far beyond 20 sigma."""
    with jax.enable_x64():
        log_b = jnp.array([700, 1000, 1e4, 1e6], dtype=jnp.float64)
        actual = sigma_from_log_bayes_factor(log_b)
        log_p = np.array(
            [
                -brentq(lambda t, b=b: t - np.log(t) - 1 - b, 1, 2 * (b + 1))
                for b in np.asarray(log_b)
            ]
        )
        expected = -ndtri_exp(log_p - np.log(2))
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)
