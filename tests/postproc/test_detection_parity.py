"""Golden parity: composed estimators reproduce the fused v1.1.1 cores."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden_reference import (
    golden_matched_filter_snr_batch_core,
    golden_snr_batch_core,
)

from photomancy.postproc.core.detection.estimator import DetectionEstimator
from photomancy.postproc.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_kernel_1d,
)
from photomancy.postproc.core.detection.samplers import AnnulusSampler, ApertureSampler
from photomancy.postproc.core.detection.significance import (
    AnnulusSigmaTest,
    TwoSampleTTest,
)
from photomancy.postproc.core.modeling import inject_planet
from photomancy.postproc.core.photometry import make_aperture_kernel

FWHM = 5.0
MAX_APERTURES = 200
EXCLUSION_BUFFER = 0.5

# For low-SNR candidates in _scene(4) and _scene(9), 2D versus flat float32
# annulus reductions shift means near 10 by up to two ULPs. Their std is > 0.2, so
# cancellation in (signal - mean) / std needs 2 * spacing(10) / 0.2 absolute
# tolerance. Keep the relative tolerance for std/division rounding unchanged.
MATCHED_FILTER_SCENE_ATOL = 2 * float(np.spacing(np.float32(10.0))) / 0.2


def _gaussian_psf(n: int, fwhm: float = FWHM) -> jnp.ndarray:
    """Center-referenced Gaussian PSF template spanning the image."""
    c = (n - 1) / 2.0
    y, x = jnp.mgrid[:n, :n]
    sigma = fwhm / 2.355
    return jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))


def _scene(seed: int, n: int = 101) -> jnp.ndarray:
    """Speckled scene with two injected planets (the standard fixture)."""
    key = jax.random.PRNGKey(seed)
    image = 10.0 + 3.0 * jax.random.normal(key, (n, n))
    image = inject_planet(image, _gaussian_psf(n), 80.0, (50.0 + 18.0, 50.0))
    image = inject_planet(image, _gaussian_psf(n), 50.0, (50.0, 50.0 - 27.0))
    return image


def _positions(n: int = 101) -> jnp.ndarray:
    """Candidate grid: planets, empty field, small radius, off chip."""
    c = (n - 1) / 2.0
    return jnp.array(
        [
            [c + 18.0, c],
            [c, c - 27.0],
            [c - 22.0, c + 9.0],
            [c + 3.0, c],  # r < fwhm -> NaN via geometry gate
            [c + 2.0, c - 1.0],  # r < fwhm
            [c, c + 44.0],
            [c + 33.3, c - 12.7],
        ]
    )


def _assert_parity(actual, expected, *, atol=1e-6):
    """Bitwise equality first, then allow documented arithmetic roundoff."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.array_equal(actual, expected, equal_nan=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=atol)


class TestMawetParity:
    """Composed Aperture+Aperture+TwoSampleT equals the fused Mawet core."""

    def test_statistic_parity_on_scene(self):
        """Statistics match the golden core on the standard scene."""
        image = _scene(0)
        positions = _positions()
        kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
        golden_snr, golden_n = golden_snr_batch_core(
            image, positions, kernel, FWHM, MAX_APERTURES, 3, EXCLUSION_BUFFER, None
        )
        stats = DetectionEstimator.mawet(FWHM)(image, positions)
        _assert_parity(stats.statistic, golden_snr)
        _assert_parity(stats.dof + 1.0, golden_n.astype(float))

    def test_validity_map_requires_full_footprints(self):
        """Box and cubic-stencil bounds are the oracle; masks and dof are exact."""
        from photomancy.postproc.core.geometry import (
            generate_aperture_coords,
            n_reference_apertures,
        )

        image = _scene(1)
        positions = jnp.concatenate([_positions(), jnp.array([[50.0, 35.0]])])
        n = image.shape[0]
        boundary = n // 3
        validity = jnp.ones((n, n)).at[:, :boundary].set(0.0)
        estimator = DetectionEstimator.mawet(FWHM)
        prepared = estimator.filter.prepare_validity(validity)
        filtered = estimator.filter.prepare(jnp.where(validity > 0.5, image, 0.0))
        stats = estimator(image, positions, validity)
        half = estimator.filter.kernel.shape[0] // 2
        for index, pos in enumerate(positions):
            radius = jnp.linalg.norm(pos - 50.0)
            count = n_reference_apertures(radius, FWHM)
            _ry, rx, active = generate_aperture_coords(
                (50.0, 50.0),
                radius,
                jnp.arctan2(pos[0] - 50, pos[1] - 50),
                count,
                fwhm=FWHM,
            )
            _, mask, _ = estimator.sampler(
                filtered, pos, estimator.filter, prepared, jnp.array([50.0, 50.0])
            )
            # These rings remain on-chip. The leftmost cubic node is floor(x)-1.
            expected = np.asarray(active) & (
                np.floor(np.asarray(rx)) - 1 - half >= boundary
            )
            np.testing.assert_array_equal(mask, expected)
            assert stats.dof[index] == expected.sum() - 1
        assert np.isnan(stats.statistic[-1])
        assert np.isnan(stats.fpf[-1])

    def test_statistic_parity_randomized(self):
        """Property sweep: random scenes and positions stay at parity."""
        kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
        composed = DetectionEstimator.mawet(FWHM)
        for seed in range(8):
            key = jax.random.PRNGKey(100 + seed)
            k1, k2 = jax.random.split(key)
            image = jax.random.normal(k1, (101, 101))
            r = jax.random.uniform(k2, (5,), minval=2.0, maxval=45.0)
            theta = jax.random.uniform(jax.random.PRNGKey(seed), (5,)) * 2 * jnp.pi
            positions = jnp.stack(
                [50.0 + r * jnp.sin(theta), 50.0 + r * jnp.cos(theta)], axis=1
            )
            golden_snr, _ = golden_snr_batch_core(
                image, positions, kernel, FWHM, MAX_APERTURES, 3, EXCLUSION_BUFFER, None
            )
            stats = composed(image, positions)
            _assert_parity(stats.statistic, golden_snr)

    def test_fpf_and_kind_populated(self):
        """The composed path adds finite FPF where the statistic is finite."""
        stats = DetectionEstimator.mawet(FWHM)(_scene(2), _positions())
        finite = ~jnp.isnan(stats.statistic)
        assert bool(jnp.all(jnp.isfinite(stats.fpf[finite])))
        assert bool(jnp.all(jnp.isnan(stats.fpf[~finite])))
        assert stats.statistic_kind == "two_sample_t"

    def test_jit_and_grad_through_composed(self):
        """The composed estimator jits and differentiates through the image."""
        composed = DetectionEstimator.mawet(FWHM)
        image = _scene(3)
        positions = _positions()[:2]

        @eqx.filter_jit
        def loss(img):
            """Negative statistic of the first candidate."""
            return -composed(img, positions).statistic[0]

        grad = jax.grad(loss)(image)
        assert grad.shape == image.shape
        assert bool(jnp.any(grad != 0.0))
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestMatchedFilterParity:
    """Composed Gaussian+Annulus+Sigma equals the fused matched-filter core."""

    def test_statistic_parity_on_scene(self):
        """Statistics match the golden matched-filter core (auto annulus)."""
        image = _scene(4)
        positions = _positions()
        golden = golden_matched_filter_snr_batch_core(
            image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, -1.0, -1.0
        )
        stats = DetectionEstimator.matched_filter(FWHM)(image, positions)
        _assert_parity(stats.statistic, golden, atol=MATCHED_FILTER_SCENE_ATOL)
        assert bool(jnp.all(jnp.isinf(stats.dof)))

    def test_statistic_parity_explicit_annulus(self):
        """Explicit annulus bounds match the golden core."""
        image = _scene(5)
        positions = _positions()[:4]
        golden = golden_matched_filter_snr_batch_core(
            image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, 12.0, 20.0
        )
        stats = DetectionEstimator.matched_filter(
            FWHM, annulus_inner=12.0, annulus_outer=20.0
        )(image, positions)
        _assert_parity(stats.statistic, golden)

    def test_statistic_parity_randomized(self):
        """Property sweep for the matched-filter pairing."""
        composed = DetectionEstimator.matched_filter(FWHM)
        for seed in range(6):
            image = jax.random.normal(jax.random.PRNGKey(200 + seed), (101, 101))
            r = jax.random.uniform(
                jax.random.PRNGKey(seed), (4,), minval=8.0, maxval=40.0
            )
            theta = (
                jax.random.uniform(jax.random.PRNGKey(300 + seed), (4,)) * 2 * jnp.pi
            )
            positions = jnp.stack(
                [50.0 + r * jnp.sin(theta), 50.0 + r * jnp.cos(theta)], axis=1
            )
            golden = golden_matched_filter_snr_batch_core(
                image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, -1.0, -1.0
            )
            stats = composed(image, positions)
            _assert_parity(stats.statistic, golden)


class TestCenterThreading:
    """center_yx=None equals an explicit geometric center, and shifts move it."""

    def test_none_equals_explicit_geometric(self):
        """Passing the geometric center explicitly is bit-identical to None."""
        image = _scene(6)
        positions = _positions()
        composed = DetectionEstimator.mawet(FWHM)
        c = (image.shape[0] - 1) / 2.0
        a = composed(image, positions)
        b = composed(image, positions, None, jnp.array([c, c]))
        np.testing.assert_array_equal(np.asarray(a.statistic), np.asarray(b.statistic))

    def test_offset_center_changes_statistics(self):
        """A shifted star center changes the reference geometry."""
        image = _scene(7)
        positions = _positions()[:3]
        composed = DetectionEstimator.mawet(FWHM)
        a = composed(image, positions)
        b = composed(image, positions, None, jnp.array([40.0, 55.0]))
        assert not np.array_equal(
            np.asarray(a.statistic), np.asarray(b.statistic), equal_nan=True
        )


def _assert_bitwise_stats(actual, expected):
    """Compare raw float bits, including any NaNs."""
    for field in ("statistic", "fpf", "dof"):
        actual_field = np.asarray(getattr(actual, field))
        expected_field = np.asarray(getattr(expected, field))
        assert actual_field.dtype == expected_field.dtype
        np.testing.assert_array_equal(
            actual_field.view(np.uint8), expected_field.view(np.uint8), err_msg=field
        )


class TestNamedConstructorParity:
    """Named configurations preserve the previous detection outputs."""

    @pytest.mark.parametrize(
        "options",
        [
            {},
            {
                "soft": True,
                "sharpness": 5.0,
                "fast": True,
                "max_apertures": 32,
                "exclusion_buffer": 1.0,
            },
            {"soft": False},
        ],
    )
    def test_mawet_constructor_preserves_detection_stats(self, options):
        """Mawet preserves statistics, FPF, and dof at candidate positions."""
        image = _scene(8)
        positions = _positions()
        validity = jnp.ones_like(image).at[:, :30].set(0.0) if options else None
        expected = DetectionEstimator(
            filter=ApertureFilter(
                kernel=make_aperture_kernel(
                    radius=FWHM / 2.0,
                    soft=options.get("soft", True),
                    sharpness=options.get("sharpness", 10.0),
                ),
                order=1 if options.get("fast", False) else 3,
            ),
            sampler=ApertureSampler(
                fwhm=FWHM,
                max_apertures=options.get("max_apertures", MAX_APERTURES),
                exclusion_buffer=options.get("exclusion_buffer", EXCLUSION_BUFFER),
            ),
            test=TwoSampleTTest(),
        )(image, positions, validity)
        actual = DetectionEstimator.mawet(FWHM, **options)(image, positions, validity)
        _assert_bitwise_stats(actual, expected)

    @pytest.mark.parametrize(
        "inner, outer, fast", [(None, None, False), (12.0, 20.0, True)]
    )
    def test_matched_filter_constructor_preserves_detection_stats(
        self, inner, outer, fast
    ):
        """The matched filter preserves auto and explicitly configured annuli."""
        image = _scene(9)
        positions = _positions()
        expected = DetectionEstimator(
            filter=GaussianFilter(
                kernel_1d=gaussian_kernel_1d(FWHM), order=1 if fast else 3
            ),
            sampler=AnnulusSampler(
                fwhm=FWHM,
                annulus_inner=-1.0 if inner is None else inner,
                annulus_outer=-1.0 if outer is None else outer,
            ),
            test=AnnulusSigmaTest(),
        )(image, positions)
        actual = DetectionEstimator.matched_filter(
            FWHM, fast=fast, annulus_inner=inner, annulus_outer=outer
        )(image, positions)
        _assert_bitwise_stats(actual, expected)
