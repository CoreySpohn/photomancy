"""Tests for photomancy.postproc.core.statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.postproc.core.statistics import (
    grubbs_fpf,
    masked_mean,
    masked_std,
    nanmasked_mean,
    nanmasked_population_std,
    normal_sf,
    student_t_sf,
)

# Reference values computed with scipy.stats.t.sf (scipy 1.16).
SCIPY_T_SF_CASES = [
    (0.0, 5.0, 0.50000000),
    (1.0, 5.0, 0.18160873),
    (2.5, 3.0, 0.04385332),
    (5.0, 2.0, 0.01887478),
    (5.0, 20.0, 0.00003437),
    (-1.5, 7.0, 0.91135076),
    (7.0, 6.0, 0.00021174),
]


@pytest.mark.parametrize(("t", "df", "expected"), SCIPY_T_SF_CASES)
def test_student_t_sf_matches_scipy_reference(t, df, expected):
    """Test student_t_sf against scipy reference values."""
    result = float(student_t_sf(jnp.asarray(t), jnp.asarray(df)))
    assert result == pytest.approx(expected, abs=1e-5)


def test_student_t_sf_is_vectorized():
    """Test that student_t_sf handles vectorized inputs."""
    t = jnp.array([0.0, 1.0, 2.5])
    df = jnp.array([5.0, 5.0, 3.0])
    result = student_t_sf(t, df)
    assert result.shape == (3,)
    assert float(result[0]) == pytest.approx(0.5, abs=1e-6)


def test_student_t_sf_monotone_decreasing_in_t():
    """Test that student_t_sf is monotone decreasing in t."""
    df = jnp.asarray(8.0)
    ts = jnp.linspace(-4.0, 6.0, 21)
    vals = student_t_sf(ts, df)
    assert bool(jnp.all(jnp.diff(vals) < 0.0))


def test_student_t_sf_invalid_dof_gives_nan():
    """Test that invalid degrees of freedom produce NaN."""
    assert bool(jnp.isnan(student_t_sf(jnp.asarray(2.0), jnp.asarray(0.0))))
    assert bool(jnp.isnan(student_t_sf(jnp.asarray(2.0), jnp.asarray(-3.0))))


def test_student_t_sf_propagates_nan_statistic():
    """Test that NaN in the statistic produces NaN output."""
    assert bool(jnp.isnan(student_t_sf(jnp.asarray(jnp.nan), jnp.asarray(5.0))))


def test_student_t_sf_jit_and_grad():
    """Test that student_t_sf is jitable and differentiable."""
    f = jax.jit(student_t_sf)
    val = f(jnp.asarray(2.0), jnp.asarray(10.0))
    assert jnp.isfinite(val)
    g = jax.grad(lambda t: student_t_sf(t, jnp.asarray(10.0)))(2.0)
    assert jnp.isfinite(g)
    assert g < 0.0  # survival function decreases in t


def test_masked_mean_ignores_masked_entries():
    """Test that masked_mean correctly ignores masked entries."""
    values = jnp.array([1.0, 2.0, 100.0])
    mask = jnp.array([1.0, 1.0, 0.0])
    assert float(masked_mean(values, mask)) == pytest.approx(1.5)


def test_masked_std_uses_bessel_correction():
    """Test that masked_std uses Bessel's correction (N-1 denominator)."""
    values = jnp.array([1.0, 3.0, 100.0])
    mask = jnp.array([1.0, 1.0, 0.0])
    # mean 2.0, residuals +/-1, variance = 2/(2-1) = 2
    assert float(masked_std(values, mask)) == pytest.approx(2.0**0.5, rel=1e-6)


class TestNormalSf:
    """Gaussian upper-tail probability against scipy references."""

    def test_reference_values(self):
        """Match scipy.stats.norm.sf at fixed points to 1e-6."""
        # scipy.stats.norm.sf([0, 1, 2, 3, -1.5]) ->
        # 0.5, 0.15865525, 0.02275013, 0.00134990, 0.93319280
        z = jnp.array([0.0, 1.0, 2.0, 3.0, -1.5])
        expected = np.array([0.5, 0.15865525, 0.02275013, 0.00134990, 0.93319280])
        np.testing.assert_allclose(np.asarray(normal_sf(z)), expected, atol=1e-6)

    def test_nan_propagates(self):
        """A NaN statistic yields a NaN survival probability."""
        assert jnp.isnan(normal_sf(jnp.nan))


class TestNanMaskedStats:
    """Nan-masked mean/std reproduce the matched-filter core idioms."""

    def test_matches_plain_stats_on_full_mask(self):
        """Full mask reduces to plain mean and population std."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.ones(4, dtype=bool)
        np.testing.assert_allclose(float(nanmasked_mean(values, mask)), 2.5, atol=1e-6)
        np.testing.assert_allclose(
            float(nanmasked_population_std(values, mask)),
            float(jnp.std(values)),
            atol=1e-6,
        )

    def test_partial_mask_ignores_excluded(self):
        """Masked-out entries do not affect the statistics."""
        values = jnp.array([1.0, 2.0, 100.0])
        mask = jnp.array([True, True, False])
        assert float(nanmasked_mean(values, mask)) == 1.5

    def test_empty_mask_gives_nan(self):
        """An empty mask yields NaN (the caller applies nan_to_num policy)."""
        values = jnp.array([1.0, 2.0])
        mask = jnp.zeros(2, dtype=bool)
        assert jnp.isnan(nanmasked_mean(values, mask))
        assert jnp.isnan(nanmasked_population_std(values, mask))


class TestGrubbsFpf:
    """One-sided ESD (Grubbs) p-value transform."""

    def test_critical_value_roundtrip(self):
        """Fpf at the one-sided critical value recovers alpha.

        References derived once with scipy:
            t = scipy.stats.t.ppf(1 - alpha / N, N - 2)
            G_crit = (N - 1) / sqrt(N) * sqrt(t**2 / (N - 2 + t**2))
        (N=10, alpha=0.05) -> G_crit = 2.17606839
        (N=20, alpha=0.05) -> G_crit = 2.55658133
        (N=10, alpha=0.01) -> G_crit = 2.40972459
        """
        cases = [
            (10.0, 2.17606839, 0.05),
            (20.0, 2.55658133, 0.05),
            (10.0, 2.40972459, 0.01),
        ]
        for n, g_crit, alpha in cases:
            fpf = float(grubbs_fpf(jnp.asarray(g_crit), jnp.asarray(n)))
            np.testing.assert_allclose(fpf, alpha, rtol=2e-3)

    def test_monotone_decreasing_in_g(self):
        """Larger outliers are less probable under the null."""
        n = jnp.asarray(12.0)
        g = jnp.array([1.0, 1.5, 2.0, 2.5])
        fpf = np.asarray(grubbs_fpf(g, n))
        assert np.all(np.diff(fpf) < 0)

    def test_edges(self):
        """Beyond-max G gives 0, negative G gives 1, tiny/NaN inputs give NaN."""
        # Attainable maximum is (N-1)/sqrt(N); just above it the p-value is 0.
        n = 10.0
        g_max = (n - 1.0) / jnp.sqrt(n)
        assert float(grubbs_fpf(g_max + 0.1, jnp.asarray(n))) == 0.0
        assert float(grubbs_fpf(jnp.asarray(-0.5), jnp.asarray(n))) == 1.0
        assert jnp.isnan(grubbs_fpf(jnp.asarray(2.0), jnp.asarray(2.0)))
        assert jnp.isnan(grubbs_fpf(jnp.asarray(jnp.nan), jnp.asarray(n)))
        for n_i in (5.0, 10.0, 12.0, 20.0, 50.0):
            g_max_i = (n_i - 1.0) / jnp.sqrt(jnp.asarray(n_i))
            assert float(grubbs_fpf(g_max_i, jnp.asarray(n_i))) == 0.0
            assert float(grubbs_fpf(g_max_i * 0.9, jnp.asarray(n_i))) > 0.0
