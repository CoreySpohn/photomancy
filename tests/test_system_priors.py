"""Normalization of the scalar priors of the synthetic systems.

All pointwise log-density comparisons use absolute tolerance 1e-10, the
float64 floor for the small sums here. Support and masks are exact.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priorities import HalfNormal, LogUniform
from scipy import integrate, stats
from system_helpers import BOUNDS

ATOL = 1e-10


def _check(label, actual, expected):
    error = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
    print(f"{label}: max_error={error:.4g} atol={ATOL:g}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


@pytest.mark.float64_only(reason="quadrature of inference prior densities")
def test_scalar_cdfs_and_support():
    """Integrate in physical units; check a partial CDF as well as total mass."""
    for name, (lo, hi) in BOUNDS.items():
        law = jax.jit(lambda x, lo=lo, hi=hi: LogUniform(lo, hi).log_prob(x))
        for end in (np.sqrt(lo * hi), hi):
            area, _ = integrate.quad(
                lambda x, law=law: np.exp(law(x)), lo, end, epsabs=1e-12, epsrel=1e-12
            )
            _check(name, area, stats.loguniform.cdf(end, lo, hi))
        assert np.all(np.isneginf(law(jnp.array([0.0, lo / 2, hi * 2]))))
    sigma = 0.8
    for end in (0.37, np.inf):
        area, _ = integrate.quad(
            lambda x: np.exp(HalfNormal(sigma).log_prob(x)),
            0,
            end,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        _check("jitter CDF", area, stats.halfnorm.cdf(end, scale=sigma))
    assert np.isneginf(HalfNormal(sigma).log_prob(-0.1))
