"""FPF calibration, physics, and position-gradient checks for the template composition.

Covers the template filter + aperture sampler + two-sample t composition:
the same FPF-honesty gate as tests/test_detection_calibration.py (on
planet-free white noise the fraction of candidates with fpf < alpha must
track alpha), plus the composition's physics (linearity in injected
amplitude, background invariance), its differentiability with respect to
candidate position, a smoke test against the annulus sampler/test pairing,
and a check that a genuinely position-dependent template provider composes
correctly through AbstractFilter.bind.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from photomancy.postproc import (
    AnnulusSampler,
    AnnulusSigmaTest,
    ApertureSampler,
    DetectionEstimator,
    TwoSampleTTest,
)
from photomancy.postproc.core.detection import PSFTemplateFilter
from photomancy.postproc.templates import (
    AbstractTemplateProvider,
    ArrayTemplateProvider,
)

FWHM = 5.0
N_IMAGES = 12
N_PER_IMAGE = 40
STAMP_SIZE = 15


def _gaussian_stamp(size: int = STAMP_SIZE, fwhm: float = FWHM) -> jnp.ndarray:
    """Peak-normalized Gaussian template stamp of shape (size, size)."""
    c = (size - 1) / 2.0
    y, x = jnp.mgrid[:size, :size]
    sigma = fwhm / 2.355
    return jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))


def _candidate_ring(n: int, n_candidates: int, key) -> jnp.ndarray:
    """Random candidate positions between 2 and 8 resolution elements."""
    k1, k2 = jax.random.split(key)
    c = (n - 1) / 2.0
    r = jax.random.uniform(k1, (n_candidates,), minval=2.0 * FWHM, maxval=8.0 * FWHM)
    theta = jax.random.uniform(k2, (n_candidates,)) * 2 * jnp.pi
    return jnp.stack([c + r * jnp.sin(theta), c + r * jnp.cos(theta)], axis=1)


def _template_estimator() -> DetectionEstimator:
    """Template filter + aperture sampler + two-sample t composition."""
    provider = ArrayTemplateProvider(_gaussian_stamp())
    return DetectionEstimator(
        filter=PSFTemplateFilter(provider=provider),
        sampler=ApertureSampler(fwhm=FWHM),
        test=TwoSampleTTest(),
    )


def _collect_null(estimator: DetectionEstimator) -> tuple[np.ndarray, np.ndarray]:
    """Pool statistic and FPF values over planet-free white-noise realizations."""
    statistics = []
    fpfs = []
    for seed in range(N_IMAGES):
        key = jax.random.PRNGKey(4000 + seed)
        k_img, k_pos = jax.random.split(key)
        image = jax.random.normal(k_img, (101, 101))
        positions = _candidate_ring(101, N_PER_IMAGE, k_pos)
        stats = estimator(image, positions)
        statistic = np.asarray(stats.statistic)
        fpf = np.asarray(stats.fpf)
        finite = np.isfinite(fpf)
        statistics.append(statistic[finite])
        fpfs.append(fpf[finite])
    return np.concatenate(statistics), np.concatenate(fpfs)


class TestTemplateFpfCalibration:
    """Empirical false-positive fraction and null scale for the template composition."""

    def test_template_t_calibrated(self):
        """fraction(fpf < alpha) tracks alpha at 0.1 and 0.01 (t-pairing slack)."""
        _, fpf = _collect_null(_template_estimator())
        n = fpf.size
        assert n > 300
        for alpha, slack in ((0.1, 4.0), (0.01, 4.0)):
            frac = float((fpf < alpha).mean())
            sigma = np.sqrt(alpha * (1 - alpha) / n)
            assert abs(frac - alpha) < slack * sigma + 0.005, (alpha, frac, n)

    def test_null_statistic_scale(self):
        """Pooled null statistics have sample std in [0.9, 1.3].

        t(n-1) with n-1 around 20-30 has std around 1.03-1.05; the band is
        deliberately loose since the candidate ring pools a range of radii,
        and therefore a range of degrees of freedom, into one sample.
        """
        statistic, _ = _collect_null(_template_estimator())
        std = float(np.std(statistic, ddof=1))
        assert 0.9 < std < 1.3, std


class TestLinearity:
    """The bound filter's response is linear in an injected template amplitude."""

    def test_linearity(self):
        """response(a) - response(0) matches a for a in {0, 2, 4, 8}."""
        stamp = _gaussian_stamp()
        half = stamp.shape[0] // 2
        position_yx = jnp.array([40.0, 55.0])
        y0, y1 = 40 - half, 40 + half + 1
        x0, x1 = 55 - half, 55 + half + 1
        base_image = 0.1 * jax.random.normal(jax.random.PRNGKey(5001), (101, 101))
        provider = ArrayTemplateProvider(stamp)
        filt = PSFTemplateFilter(provider=provider).bind(position_yx)
        response0 = filt.evaluate(base_image, position_yx)
        for amplitude in (0.0, 2.0, 4.0, 8.0):
            injected = base_image.at[y0:y1, x0:x1].add(amplitude * stamp)
            response = filt.evaluate(injected, position_yx)
            np.testing.assert_allclose(
                float(response - response0), amplitude, rtol=1e-3, atol=1e-6
            )


class TestBackgroundInvarianceComposed:
    """The composed statistic is invariant to a uniform additive background."""

    def test_background_invariance_composed(self):
        """A constant added to the whole image leaves the composed statistic fixed.

        Both the filter (zero-meaned patches) and the two-sample test
        (difference of means, standardized by a std of differences) are
        shift-invariant, so the composed statistic should be unaffected by a
        uniform additive background.
        """
        est = _template_estimator()
        key = jax.random.PRNGKey(5002)
        k_img, k_pos = jax.random.split(key)
        image = jax.random.normal(k_img, (101, 101))
        positions = _candidate_ring(101, N_PER_IMAGE, k_pos)
        baseline = est(image, positions)
        shifted = est(image + 50.0, positions)
        # NOTE: rtol=1e-4, atol=1e-4 (not the tighter 1e-5 of a single-position
        # check) -- the zero-mean patch step subtracts a large ~50 mean from
        # every pixel before the two-sample t-test's own mean/std pass, and
        # that catastrophic-cancellation error scales with the background
        # magnitude (float32): observed max abs difference ~4e-5 at +50 and
        # ~7e-5 at +100 background, vs ~4e-6 at +5, confirmed across 20 seeds.
        np.testing.assert_allclose(
            np.asarray(shifted.statistic),
            np.asarray(baseline.statistic),
            rtol=1e-4,
            atol=1e-4,
        )


class TestGradThroughPosition:
    """The composed statistic is differentiable with respect to candidate position."""

    def test_grad_through_position(self):
        """Grad of the statistic w.r.t. position is finite and nonzero."""
        est = _template_estimator()
        image = jax.random.normal(jax.random.PRNGKey(5003), (101, 101))
        position = jnp.array([70.0, 50.0])  # mid-ring: radius 20 from center (50, 50)
        grad = jax.grad(lambda pos: est(image, pos[None, :]).statistic[0])(position)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad != 0.0))


class TestAnnulusSamplerCompositionConservative:
    """The template filter composes conservatively with the annulus sampler pairing."""

    def test_annulus_sampler_composition_conservative(self):
        """Finite statistics and a conservative fpf fraction, pooled over seeds.

        The annulus noise sample is raw prepared-image pixels; the template
        filter's prepare is identity, so the sigma test's scale is set by
        the template normalization. This stamp is peak-normalized (sum of
        squared zero-meaned template well above 1), so its fitted-amplitude
        null spread is far smaller than the raw-pixel spread and the test
        is strongly conservative HERE; a small-normalization template (e.g.
        raw coronagraph intensity stamps) flips it anti-conservative.
        Same-scale references come from ApertureSampler instead.
        """
        provider = ArrayTemplateProvider(_gaussian_stamp())
        est = DetectionEstimator(
            filter=PSFTemplateFilter(provider=provider),
            sampler=AnnulusSampler(fwhm=FWHM),
            test=AnnulusSigmaTest(),
        )
        statistics = []
        fpfs = []
        for seed in range(N_IMAGES):
            key = jax.random.PRNGKey(6000 + seed)
            k_img, k_pos = jax.random.split(key)
            image = jax.random.normal(k_img, (101, 101))
            positions = _candidate_ring(101, N_PER_IMAGE, k_pos)
            stats = est(image, positions)
            statistics.append(np.asarray(stats.statistic))
            fpfs.append(np.asarray(stats.fpf))
        statistic = np.concatenate(statistics)
        fpf = np.concatenate(fpfs)
        assert bool(np.all(np.isfinite(statistic)))
        frac = float((fpf < 0.1).mean())
        assert frac <= 0.02, frac


class _RadiusDependentProvider(AbstractTemplateProvider):
    """Toy position-dependent provider: a Gaussian stamp whose width grows with radius.

    Exercises AbstractFilter.bind's per-candidate specialization without a
    real coronagraph PSF model: the stamp returned for a query position is a
    peak-normalized Gaussian whose sigma grows smoothly with the position's
    radius from a fixed reference center, so candidates at different radii
    receive visibly different templates (loosely analogous to a coronagraph
    PSF whose shape changes with separation).
    """

    center_yx: jnp.ndarray
    stamp_size: int = eqx.field(static=True)
    base_sigma: float
    sigma_per_pixel: float

    def __init__(
        self,
        center_yx: jnp.ndarray,
        stamp_size: int = STAMP_SIZE,
        base_sigma: float = 1.0,
        sigma_per_pixel: float = 0.15,
    ):
        """Configure the radius-dependent Gaussian stamp.

        Args:
            center_yx: (2,) fixed reference position (y, x) in pixels that
                the query radius is measured from.
            stamp_size: Stamp edge length in pixels.
            base_sigma: Gaussian sigma in pixels at zero radius.
            sigma_per_pixel: Growth rate of sigma per pixel of radius.
        """
        self.center_yx = jnp.asarray(center_yx)
        self.stamp_size = int(stamp_size)
        self.base_sigma = base_sigma
        self.sigma_per_pixel = sigma_per_pixel

    def __call__(self, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Peak-normalized Gaussian stamp; sigma grows with the query radius."""
        radius = jnp.sqrt(jnp.sum((position_yx - self.center_yx) ** 2))
        sigma = self.base_sigma + self.sigma_per_pixel * radius
        c = (self.stamp_size - 1) / 2.0
        y, x = jnp.mgrid[: self.stamp_size, : self.stamp_size]
        return jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2.0 * sigma**2))


class TestBindPositionDependentProvider:
    """A genuinely position-dependent provider composes end-to-end via bind."""

    def test_bind_position_dependent_provider(self):
        """Candidates at different radii get finite stats and different templates."""
        center_yx = jnp.array([50.0, 50.0])
        provider = _RadiusDependentProvider(center_yx)
        estimator = DetectionEstimator(
            filter=PSFTemplateFilter(provider=provider),
            sampler=ApertureSampler(fwhm=FWHM),
            test=TwoSampleTTest(),
        )
        radii = jnp.array([10.0, 20.0, 35.0])
        theta = jnp.array([0.3, 1.7, 4.2])
        positions = jnp.stack(
            [
                center_yx[0] + radii * jnp.sin(theta),
                center_yx[1] + radii * jnp.cos(theta),
            ],
            axis=1,
        )
        image = jax.random.normal(jax.random.PRNGKey(5005), (101, 101))
        stats = estimator(image, positions)

        assert bool(jnp.all(jnp.isfinite(stats.statistic)))
        assert bool(jnp.all(jnp.isfinite(stats.fpf)))

        stamp_near = provider(positions[0])
        stamp_far = provider(positions[2])
        assert not np.allclose(np.asarray(stamp_near), np.asarray(stamp_far))
