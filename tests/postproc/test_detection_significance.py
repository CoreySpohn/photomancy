"""Tests for detection significance tests."""

import jax.numpy as jnp
import numpy as np

from photomancy.postproc.core.detection.significance import (
    AnnulusSigmaTest,
    GrubbsTest,
    TwoSampleTTest,
)
from photomancy.postproc.core.statistics import grubbs_fpf, student_t_sf


def _samples(values, n_pad=8):
    """Pad values into a static buffer with a matching mask."""
    values = jnp.asarray(values, dtype=float)
    pad = n_pad - values.shape[0]
    samples = jnp.concatenate([values, jnp.zeros(pad)])
    mask = jnp.concatenate([jnp.ones_like(values, dtype=bool), jnp.zeros(pad, bool)])
    return samples, mask


class TestTwoSampleTTest:
    """Mawet small-sample t statistic."""

    def test_matches_hand_computation(self):
        """Statistic equals the Mawet 2014 formula on a known sample."""
        refs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        samples, mask = _samples(refs)
        signal = jnp.asarray(9.0)
        statistic, fpf, dof = TwoSampleTTest()(signal, samples, mask)
        n = 5.0
        bg_mean = 3.0
        bg_std = float(jnp.std(refs, ddof=1))
        expected = (9.0 - bg_mean) / (bg_std * np.sqrt(1 + 1 / n))
        np.testing.assert_allclose(float(statistic), expected, rtol=1e-6)
        np.testing.assert_allclose(float(dof), n - 1.0, rtol=0)
        np.testing.assert_allclose(
            float(fpf), float(student_t_sf(statistic, dof)), rtol=0
        )

    def test_insufficient_samples_gate(self):
        """Fewer than 3 valid references NaN the statistic, dof still reported."""
        samples, mask = _samples(jnp.array([1.0, 2.0]))
        statistic, fpf, dof = TwoSampleTTest()(jnp.asarray(5.0), samples, mask)
        assert jnp.isnan(statistic)
        assert jnp.isnan(fpf)
        assert float(dof) == 1.0

    def test_kind_label(self):
        """The statistic kind is a class-level static string."""
        assert TwoSampleTTest.kind == "two_sample_t"


class TestAnnulusSigmaTest:
    """Gaussian sigma test with the matched-filter fallbacks."""

    def test_matches_fused_core_ops(self):
        """Statistic reproduces the v1.1.1 matched-filter op sequence."""
        vals = jnp.array([0.5, -0.3, 0.8, 0.1, -0.6, 0.2])
        samples, mask = _samples(vals, n_pad=10)
        signal = jnp.asarray(1.7)
        statistic, _fpf, dof = AnnulusSigmaTest()(signal, samples, mask)

        masked = jnp.where(mask, samples, jnp.nan)
        bg_mean = jnp.nanmean(masked)
        variance = jnp.nanmean(jnp.where(mask, (samples - bg_mean) ** 2, jnp.nan))
        bg_std = jnp.sqrt(variance)
        expected = (1.7 - jnp.nan_to_num(bg_mean, nan=0.0)) / jnp.maximum(
            jnp.nan_to_num(bg_std, nan=1.0), 1e-10
        )
        np.testing.assert_array_equal(np.asarray(statistic), np.asarray(expected))
        assert jnp.isinf(dof)

    def test_empty_mask_fallbacks(self):
        """Empty annulus reproduces the nan_to_num(0, 1) fallback statistic."""
        samples = jnp.zeros(6)
        mask = jnp.zeros(6, dtype=bool)
        statistic, _fpf, _dof = AnnulusSigmaTest()(jnp.asarray(2.0), samples, mask)
        np.testing.assert_allclose(float(statistic), 2.0, rtol=1e-6)

    def test_fpf_is_gaussian_tail(self):
        """FPF equals the standard normal survival function."""
        vals = jnp.array([0.0, 1.0, -1.0, 0.5, -0.5])
        samples, mask = _samples(vals, n_pad=8)
        statistic, fpf, _ = AnnulusSigmaTest()(jnp.asarray(2.0), samples, mask)
        # normal_sf at the statistic
        from photomancy.postproc.core.statistics import normal_sf

        np.testing.assert_array_equal(np.asarray(fpf), np.asarray(normal_sf(statistic)))


class TestGrubbsTest:
    """Extreme studentized deviate over the pooled sample."""

    def test_pooled_statistic(self):
        """G uses the pooled candidate + references sample."""
        refs = jnp.array([1.0, 2.0, 3.0, 4.0])
        samples, mask = _samples(refs)
        signal = jnp.asarray(10.0)
        statistic, fpf, dof = GrubbsTest()(signal, samples, mask)
        pooled = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        expected_g = (pooled.max() - pooled.mean()) / pooled.std(ddof=1)
        np.testing.assert_allclose(float(statistic), expected_g, rtol=1e-6)
        np.testing.assert_allclose(float(dof), 3.0, rtol=0)
        np.testing.assert_allclose(
            float(fpf), float(grubbs_fpf(statistic, jnp.asarray(5.0))), rtol=0
        )

    def test_reference_can_be_the_outlier(self):
        """A reference larger than the candidate drives the statistic."""
        refs = jnp.array([1.0, 2.0, 20.0, 3.0])
        samples, mask = _samples(refs)
        statistic, _, _ = GrubbsTest()(jnp.asarray(4.0), samples, mask)
        pooled = np.array([1.0, 2.0, 20.0, 3.0, 4.0])
        expected_g = (pooled.max() - pooled.mean()) / pooled.std(ddof=1)
        np.testing.assert_allclose(float(statistic), expected_g, rtol=1e-6)

    def test_insufficient_pool_gate(self):
        """A pool smaller than 3 NaNs the statistic."""
        samples, mask = _samples(jnp.array([1.0]))
        statistic, fpf, _dof = GrubbsTest()(jnp.asarray(2.0), samples, mask)
        assert jnp.isnan(statistic)
        assert jnp.isnan(fpf)
