"""Empirical FPF calibration of the composed detection estimators.

The FPF-honesty gate: on planet-free white noise the fraction of candidates
with fpf < alpha must track alpha (exactly calibrated for the t pairing,
approximately for the Gaussian pairing on a correlated filtered field, and
conservatively bounded for the Grubbs pooled-maximum test).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from photomancy.postproc import (
    AnnulusSampler,
    AnnulusSigmaTest,
    ApertureFilter,
    ApertureSampler,
    DetectionEstimator,
    GaussianFilter,
    GrubbsTest,
    TwoSampleTTest,
    gaussian_kernel_1d,
    make_aperture_kernel,
)

FWHM = 5.0
N_IMAGES = 12
N_PER_IMAGE = 40


def _candidate_ring(n: int, n_candidates: int, key) -> jnp.ndarray:
    """Random candidate positions between 2 and 8 resolution elements."""
    k1, k2 = jax.random.split(key)
    c = (n - 1) / 2.0
    r = jax.random.uniform(k1, (n_candidates,), minval=2.0 * FWHM, maxval=8.0 * FWHM)
    theta = jax.random.uniform(k2, (n_candidates,)) * 2 * jnp.pi
    return jnp.stack([c + r * jnp.sin(theta), c + r * jnp.cos(theta)], axis=1)


def _collect_fpf(estimator: DetectionEstimator) -> np.ndarray:
    """Pool FPF values over planet-free white-noise realizations."""
    values = []
    for seed in range(N_IMAGES):
        key = jax.random.PRNGKey(1000 + seed)
        k_img, k_pos = jax.random.split(key)
        image = jax.random.normal(k_img, (101, 101))
        positions = _candidate_ring(101, N_PER_IMAGE, k_pos)
        stats = estimator(image, positions)
        fpf = np.asarray(stats.fpf)
        values.append(fpf[np.isfinite(fpf)])
    return np.concatenate(values)


def _mawet() -> DetectionEstimator:
    """Aperture + aperture + two-sample-t pairing."""
    return DetectionEstimator(
        filter=ApertureFilter(
            kernel=make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0),
            order=3,
        ),
        sampler=ApertureSampler(fwhm=FWHM),
        test=TwoSampleTTest(),
    )


class TestFpfCalibration:
    """Empirical false-positive fractions track nominal alpha."""

    def test_two_sample_t_calibrated(self):
        """t-pairing: fraction(fpf < alpha) ~ alpha at 0.1 and 0.01."""
        fpf = _collect_fpf(_mawet())
        n = fpf.size
        assert n > 300
        for alpha, slack in ((0.1, 4.0), (0.01, 4.0)):
            frac = float((fpf < alpha).mean())
            sigma = np.sqrt(alpha * (1 - alpha) / n)
            assert abs(frac - alpha) < slack * sigma + 0.005, (alpha, frac, n)

    def test_annulus_sigma_approximately_calibrated(self):
        """Gaussian pairing on a filtered (correlated) field: loose bounds."""
        est = DetectionEstimator(
            filter=GaussianFilter(kernel_1d=gaussian_kernel_1d(FWHM), order=3),
            sampler=AnnulusSampler(fwhm=FWHM),
            test=AnnulusSigmaTest(),
        )
        fpf = _collect_fpf(est)
        frac = float((fpf < 0.1).mean())
        assert 0.03 < frac < 0.22, frac

    def test_grubbs_conservative(self):
        """Grubbs pooled-maximum FPF is a conservative (Bonferroni) bound."""
        est = DetectionEstimator(
            filter=ApertureFilter(
                kernel=make_aperture_kernel(
                    radius=FWHM / 2.0, soft=True, sharpness=10.0
                ),
                order=3,
            ),
            sampler=ApertureSampler(fwhm=FWHM),
            test=GrubbsTest(),
        )
        fpf = _collect_fpf(est)
        for alpha in (0.1, 0.05):
            frac = float((fpf < alpha).mean())
            assert frac < 1.6 * alpha + 0.01, (alpha, frac)


class TestComposedProperties:
    """jit / vmap round-trips of the composed estimator."""

    def test_jit_roundtrip_matches_eager(self):
        """filter_jit output equals the direct call."""
        est = _mawet()
        image = jax.random.normal(jax.random.PRNGKey(7), (101, 101))
        positions = _candidate_ring(101, 8, jax.random.PRNGKey(8))
        direct = est(image, positions)
        jitted = eqx.filter_jit(lambda e, i, p: e(i, p))(est, image, positions)
        np.testing.assert_array_equal(
            np.asarray(direct.statistic), np.asarray(jitted.statistic)
        )

    def test_vmap_over_image_batch(self):
        """The estimator vmaps over a frame axis (statistic field checked)."""
        est = _mawet()
        images = jax.random.normal(jax.random.PRNGKey(9), (3, 101, 101))
        positions = _candidate_ring(101, 5, jax.random.PRNGKey(10))
        batched = jax.vmap(lambda img: est(img, positions).statistic)(images)
        assert batched.shape == (3, 5)
        single = est(images[1], positions).statistic
        # NOTE: rtol/atol (not assert_array_equal) -- vmap over a batch axis
        # recompiles a different-shaped XLA program than the single-image
        # call, and XLA's fusion/kernel selection is not guaranteed bit-exact
        # across differently-shaped compiled programs (float32 FP
        # non-associativity); observed here as a 1-ULP (1.1920929e-07)
        # difference in 2/5 elements, deterministic across repeated runs.
        # Same tolerance convention as tests/test_detection_parity.py:80.
        np.testing.assert_allclose(
            np.asarray(batched[1]), np.asarray(single), rtol=1e-6, atol=1e-6
        )
