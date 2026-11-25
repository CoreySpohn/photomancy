"""Tests for the calibration tools: MapFilter, candidates, null, injection."""

import math
from fractions import Fraction

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.postproc import DetectionEstimator
from photomancy.postproc.calibration import (
    contrast_curve,
    false_alarm_band,
    find_candidates,
    n_resolution_elements,
    throughput_profile,
)
from photomancy.postproc.core.detection import (
    ApertureSampler,
    MapFilter,
    TwoSampleTTest,
)


def _binomial_cdf_exact(n, p, k):
    """P(X <= k) for X ~ Binomial(n, p), summed exactly in rationals."""
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))


class TestMapFilter:
    """The statistic map goes through DetectionEstimator unchanged."""

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_estimator_matches_hand_computation(self):
        """Basis exact-algebra: a plane is reproduced exactly by bilinear sampling.

        The map is a plane a + b y + c x, so every interpolated reference
        value is the plane at the reference position, computed here from the
        ring geometry (angular step 2 arcsin(fwhm / 2r), n references
        starting one and a half steps clockwise of the candidate), and the
        statistic is the two-sample t of those values.
        """
        ny = nx = 81
        a, b, c = 3.0, 0.7, -0.4
        yy, xx = np.mgrid[:ny, :nx]
        stat_map = a + b * yy + c * xx
        center = np.array([40.0, 40.0])
        fwhm, buffer = 4.0, 0.5
        position = np.array([40.0 + 18.0, 40.0 + 9.0])

        estimator = DetectionEstimator(
            MapFilter(order=1),
            ApertureSampler(fwhm=fwhm, max_apertures=100, exclusion_buffer=buffer),
            TwoSampleTTest(),
        )
        stats = estimator(
            jnp.asarray(stat_map),
            jnp.asarray(position[None]),
            center_yx=jnp.asarray(center),
        )

        r = np.hypot(*(position - center))
        phi = np.arctan2(position[0] - center[0], position[1] - center[1])
        dtheta = 2.0 * np.arcsin(fwhm / 2.0 / r)
        n = int(np.floor(2 * np.pi / dtheta) - 1 - 2 * buffer)
        angles = phi - (np.arange(n) + 1 + buffer) * dtheta
        ref = (
            a
            + b * (center[0] + r * np.sin(angles))
            + c * (center[1] + r * np.cos(angles))
        )
        signal = a + b * position[0] + c * position[1]
        expected = (signal - ref.mean()) / (ref.std(ddof=1) * np.sqrt(1 + 1 / n))

        assert n > 10  # guard: the reference ring is populated
        np.testing.assert_allclose(stats.statistic[0], expected, rtol=1e-10)
        np.testing.assert_allclose(stats.dof[0], n - 1)

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_nan_pixels_do_not_leak(self):
        """Basis exact-algebra: a NaN pixel marked invalid drops out cleanly.

        Setting a pixel on the reference ring to NaN and marking it invalid
        must equal the statistic of the same ring with that reference
        dropped; a NaN that leaked would make the result NaN.
        """
        n = 81
        rng = np.random.default_rng(3)
        stat_map = rng.normal(size=(n, n))
        center = jnp.array([40.0, 40.0])
        position = jnp.array([[40.0, 40.0 + 15.0]])
        estimator = DetectionEstimator(
            MapFilter(order=0),
            ApertureSampler(fwhm=4.0, max_apertures=100),
            TwoSampleTTest(),
        )
        clean = estimator(jnp.asarray(stat_map), position, center_yx=center)
        # Invalidate a block that holds part of the reference ring.
        validity = np.ones((n, n))
        validity[44:56, 20:34] = 0.0
        holed = stat_map.copy()
        holed[validity == 0] = np.nan
        with_nan = estimator(
            jnp.asarray(holed),
            position,
            validity_map=jnp.asarray(validity),
            center_yx=center,
        )
        with_zero = estimator(
            jnp.asarray(np.where(validity > 0, stat_map, 0.0)),
            position,
            validity_map=jnp.asarray(validity),
            center_yx=center,
        )
        assert np.isfinite(with_nan.statistic[0])
        assert with_nan.dof[0] < clean.dof[0]  # guard: the hole removed references
        np.testing.assert_allclose(with_nan.statistic, with_zero.statistic, rtol=1e-12)


class TestFindCandidates:
    """Planted peaks come back exactly."""

    def _map(self):
        stat = np.zeros((64, 64))
        for (y, x), v in {
            (32, 48): 7.0,
            (20, 20): 6.0,
            (32, 52): 5.5,
            (45, 30): 4.0,
        }.items():
            stat[y, x] = v
        return stat

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_planted_peaks_ordered_by_statistic(self):
        """Basis exact-algebra: isolated peaks are the only maxima above threshold."""
        out = find_candidates(self._map(), (32.0, 32.0), 5.0, 25.0, 3.0, min_sep_px=1.0)
        assert len(out.statistic) == 4  # guard: peaks were found at all
        # (32, 48) beats (20, 20) beats (32, 52) beats (45, 30); the zero
        # plateau never exceeds the threshold.
        expected = [(32, 48), (20, 20), (32, 52), (45, 30)]
        got = [tuple(p) for p in out.positions_yx.astype(int)]
        assert got == expected

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_min_separation_suppresses_weaker_neighbor(self):
        """Basis exact-algebra: the weaker of two close peaks is dropped."""
        stat = self._map()
        near = find_candidates(stat, (32.0, 32.0), 5.0, 30.0, 3.0, min_sep_px=5.0)
        far = find_candidates(stat, (32.0, 32.0), 5.0, 30.0, 3.0, min_sep_px=3.0)
        got_near = [tuple(p) for p in near.positions_yx.astype(int)]
        got_far = [tuple(p) for p in far.positions_yx.astype(int)]
        assert got_near == [(32, 48), (20, 20), (45, 30)]
        assert got_far == [(32, 48), (20, 20), (32, 52), (45, 30)]
        np.testing.assert_array_equal(near.statistic, [7.0, 6.0, 4.0])
        np.testing.assert_allclose(near.radius_px[0], 16.0)

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_validity_mask_and_annulus(self):
        """Basis exact-algebra: invalid, NaN and out-of-annulus peaks are excluded."""
        stat = self._map()
        stat[10, 10] = np.nan
        validity = np.ones_like(stat)
        validity[32, 48] = 0.0
        out = find_candidates(
            stat, (32.0, 32.0), 5.0, 25.0, 3.0, min_sep_px=1.0, validity=validity
        )
        got = {tuple(p) for p in out.positions_yx.astype(int)}
        # (32, 48) is invalid, (32, 52) is 20 px out so inside, (45, 30) r=13.2.
        assert got == {(20, 20), (32, 52), (45, 30)}
        inner = find_candidates(stat, (32.0, 32.0), 14.0, 25.0, 3.0, min_sep_px=1.0)
        assert (45, 30) not in {tuple(p) for p in inner.positions_yx.astype(int)}

    def test_nothing_above_threshold_is_empty(self):
        """No pixel passes: empty, correctly shaped arrays."""
        out = find_candidates(self._map(), (32.0, 32.0), 5.0, 25.0, 50.0, 1.0)
        assert out.positions_yx.shape == (0, 2)
        assert out.statistic.shape == (0,)


class TestNull:
    """Resolution-element count and binomial band."""

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_n_resolution_elements_hand_count(self):
        """Basis exact-algebra: hand-summed floors at ring centers.

        r_min = 6, r_max = 18, fwhm = 3 gives four rings centered at 7.5,
        10.5, 13.5 and 16.5; 2 pi r / 3 is 15.7, 22.0 (21.99), 28.3 and 34.6.
        """
        assert n_resolution_elements(6.0, 18.0, 3.0) == 15 + 21 + 28 + 34
        # A remainder narrower than one ring is dropped: 6 to 19.9 is still four.
        assert n_resolution_elements(6.0, 19.9, 3.0) == 98
        assert n_resolution_elements(6.0, 8.0, 3.0) == 0

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    @pytest.mark.parametrize(
        ("n", "p", "level"),
        [
            (5000, Fraction(1, 100), Fraction(99, 100)),
            (5000, Fraction(1, 1000), Fraction(99, 100)),
            (570, Fraction(1, 100), Fraction(95, 100)),
            (40, Fraction(1, 10), Fraction(9, 10)),
        ],
    )
    def test_band_matches_exact_binomial_quantiles(self, n, p, level):
        """Basis exact-algebra: band edges satisfy the quantile definition.

        lo is the smallest k with CDF(k) >= tail and hi the smallest k with
        CDF(k) >= 1 - tail, checked by a direct rational sum of binomial
        terms, independent of the implementation's log-space accumulation.
        """
        lo, hi = false_alarm_band(n, float(p), float(level))
        tail = (1 - level) / 2
        assert _binomial_cdf_exact(n, p, hi) >= 1 - tail
        assert _binomial_cdf_exact(n, p, hi - 1) < 1 - tail
        assert _binomial_cdf_exact(n, p, lo) >= tail
        if lo > 0:
            assert _binomial_cdf_exact(n, p, lo - 1) < tail

    def test_band_matches_spec_numbers(self):
        """N of about 5000 at the 1e-2 and 1e-3 levels gives the known bands."""
        lo, hi = false_alarm_band(5000, 1e-2)
        assert lo <= 33 <= 35 <= hi <= 70  # about [33, 69]
        lo3, hi3 = false_alarm_band(5000, 1e-3)
        assert lo3 == 0
        assert 11 <= hi3 <= 13  # about [0, 12]

    def test_band_edge_cases(self):
        """Degenerate fpf and bad arguments."""
        assert false_alarm_band(100, 0.0) == (0, 0)
        assert false_alarm_band(100, 1.0) == (100, 100)
        with pytest.raises(ValueError, match="fpf"):
            false_alarm_band(100, 1.5)
        with pytest.raises(ValueError, match="level"):
            false_alarm_band(100, 0.1, level=1.0)


class TestInjection:
    """Throughput statistics and the contrast curve."""

    @vv.case("system-postproc", "code-verification", basis="floating-point")
    def test_throughput_profile_matches_numpy(self):
        """Basis floating-point: per-group NumPy mean, std (ddof 1) and count."""
        rng = np.random.default_rng(1)
        seps = np.repeat([0.4, 0.8, 1.2], 12)
        injected = rng.uniform(1.0, 2.0, seps.size)
        recovered = injected * rng.normal(0.9, 0.07, seps.size)
        out = throughput_profile(seps, injected, recovered)
        ratio = recovered / injected
        np.testing.assert_array_equal(out.separation, [0.4, 0.8, 1.2])
        for i, s in enumerate(out.separation):
            sel = ratio[seps == s]
            assert out.count[i] == 12
            np.testing.assert_allclose(out.mean[i], np.mean(sel), rtol=1e-14)
            np.testing.assert_allclose(out.std[i], np.std(sel, ddof=1), rtol=1e-12)

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_binned_profile_and_empty_bins(self):
        """Basis exact-algebra: bin membership; NaN for empty or single bins."""
        seps = np.array([0.1, 0.5, 0.5, 1.0, 5.0])
        out = throughput_profile(
            seps,
            np.ones(5),
            np.array([0.5, 0.9, 0.7, 0.8, 9.0]),
            edges=[0.0, 0.5, 1.0, 2.0],
        )
        np.testing.assert_array_equal(out.count, [1, 2, 1])
        np.testing.assert_allclose(out.mean, [0.5, 0.8, 0.8])
        assert np.isnan(out.std[0]) and np.isnan(out.std[2])
        np.testing.assert_allclose(out.std[1], np.std([0.9, 0.7], ddof=1))
        empty = throughput_profile(seps, np.ones(5), np.ones(5), edges=[2.0, 3.0, 4.0])
        np.testing.assert_array_equal(empty.count, [0, 0])
        assert np.all(np.isnan(empty.mean))

    @vv.case("system-postproc", "code-verification", basis="exact-algebra")
    def test_contrast_curve_constructed_case(self):
        """Basis exact-algebra: 5 sigma of noise 1e-7 at throughput 0.5 is 1e-6."""
        out = contrast_curve(5.0, [1e-7, 2e-7, 4e-7], [0.5, 0.25, 0.0])
        np.testing.assert_allclose(out[:2], [1e-6, 4e-6], rtol=1e-12)
        assert np.isinf(out[2])
