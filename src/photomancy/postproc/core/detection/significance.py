"""Significance tests: statistic, false positive fraction, degrees of freedom."""

from typing import ClassVar

import jax.numpy as jnp

from photomancy.postproc.core.detection.base import AbstractTest
from photomancy.postproc.core.statistics import (
    grubbs_fpf,
    masked_mean,
    masked_std,
    nanmasked_mean,
    nanmasked_population_std,
    normal_sf,
    small_sample_penalty,
    student_t_sf,
)


class TwoSampleTTest(AbstractTest):
    """Small-sample two-sample t-test (Mawet et al. 2014).

    The statistic is (signal - mean(refs)) / (std(refs) * sqrt(1 + 1/n)) with
    Bessel-corrected std, distributed as Student-t with n - 1 degrees of
    freedom under the null. Fewer than 3 valid references NaN the statistic;
    dof is reported regardless.
    """

    kind: ClassVar[str] = "two_sample_t"

    def __call__(
        self,
        signal: jnp.ndarray,
        samples: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Small-sample-corrected t statistic, its FPF, and dof."""
        n_valid = jnp.sum(mask)
        bg_mean = masked_mean(samples, mask)
        bg_std = masked_std(samples, mask, mean=bg_mean)
        penalty = small_sample_penalty(n_valid)
        statistic = (signal - bg_mean) / jnp.maximum(bg_std * penalty, 1e-10)
        statistic = jnp.where(n_valid >= 3, statistic, jnp.nan)
        dof = jnp.asarray(n_valid, dtype=jnp.result_type(float)) - 1.0
        fpf = student_t_sf(statistic, dof)
        return statistic, fpf, dof


class AnnulusSigmaTest(AbstractTest):
    """Gaussian sigma test over annulus pixels (matched-filter convention).

    The statistic is (signal - mean) / std with population std over the
    masked samples and the v1.1.1 fallbacks (mean 0, std 1) when the annulus
    is empty; the null is standard normal (large-sample convention), so the
    degrees of freedom are infinite.
    """

    kind: ClassVar[str] = "annulus_sigma"

    def __call__(
        self,
        signal: jnp.ndarray,
        samples: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sigma statistic against the annulus, its Gaussian FPF, and dof."""
        bg_mean = nanmasked_mean(samples, mask)
        bg_std = nanmasked_population_std(samples, mask, mean=bg_mean)
        bg_mean = jnp.nan_to_num(bg_mean, nan=0.0)
        bg_std = jnp.nan_to_num(bg_std, nan=1.0)
        statistic = (signal - bg_mean) / jnp.maximum(bg_std, 1e-10)
        fpf = normal_sf(statistic)
        dof = jnp.asarray(jnp.inf, dtype=jnp.result_type(float))
        return statistic, fpf, dof


class GrubbsTest(AbstractTest):
    """One-sided Grubbs extreme studentized deviate over the pooled sample.

    Pools the candidate signal with the valid reference samples and asks
    whether the pooled MAXIMUM is an outlier (Grubbs 1950): a blind-search
    statistic free of aperture-placement bias. dof is N - 2 for pooled size
    N; a pool smaller than 3 NaNs the statistic.
    """

    kind: ClassVar[str] = "grubbs_g"

    def __call__(
        self,
        signal: jnp.ndarray,
        samples: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Pooled-maximum Grubbs statistic, its ESD FPF, and dof."""
        n_refs = jnp.sum(mask)
        n_pooled = n_refs + 1
        pooled_sum = jnp.sum(samples * mask) + signal
        pooled_mean = pooled_sum / jnp.maximum(n_pooled, 1.0)
        residuals = (samples - pooled_mean) * mask
        pooled_var = (jnp.sum(residuals**2) + (signal - pooled_mean) ** 2) / (
            jnp.maximum(n_pooled - 1, 1.0)
        )
        pooled_std = jnp.sqrt(pooled_var)
        pooled_max = jnp.maximum(jnp.max(jnp.where(mask, samples, -jnp.inf)), signal)
        statistic = (pooled_max - pooled_mean) / jnp.maximum(pooled_std, 1e-10)
        statistic = jnp.where(n_pooled >= 3, statistic, jnp.nan)
        n_float = jnp.asarray(n_pooled, dtype=jnp.result_type(float))
        dof = n_float - 2.0
        fpf = grubbs_fpf(statistic, n_float)
        return statistic, fpf, dof
