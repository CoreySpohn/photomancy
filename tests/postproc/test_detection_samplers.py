"""Tests for detection noise samplers."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates

from photomancy.postproc.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_kernel_1d,
)
from photomancy.postproc.core.detection.samplers import AnnulusSampler, ApertureSampler
from photomancy.postproc.core.geometry import n_reference_apertures
from photomancy.postproc.core.photometry import make_aperture_kernel


def _setup(n: int = 101, seed: int = 0):
    """Image, flux map, filter, and an all-valid map for sampler tests."""
    image = jax.random.normal(jax.random.PRNGKey(seed), (n, n))
    kernel = make_aperture_kernel(radius=2.5, soft=True, sharpness=10.0)
    filt = ApertureFilter(kernel=kernel, order=3)
    fmap = filt.prepare(image)
    validity = jnp.ones((n, n))
    center = jnp.array([(n - 1) / 2.0, (n - 1) / 2.0])
    return image, fmap, filt, validity, center


class TestApertureSampler:
    """Same-radius reference apertures (Mawet geometry)."""

    def test_reproduces_fused_core_reference_fluxes(self):
        """Samples equal the v1.1.1 inline geometry sampled off the flux map."""
        _, fmap, filt, validity, center = _setup()
        fwhm, buffer, max_ap = 5.0, 0.5, 200
        pos = jnp.array([50.0 + 15.0, 50.0])  # r = 15 from center

        sampler = ApertureSampler(fwhm=fwhm, max_apertures=max_ap)
        samples, mask, geom_ok = sampler(fmap, pos, filt, validity, center)

        # Inline v1.1.1 reference computation (_snr_batch_core lines).
        cy, cx = center[0], center[1]
        py, px = pos[0], pos[1]
        r_pix = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)
        planet_angle = jnp.arctan2(py - cy, px - cx)
        half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(r_pix, 0.1), 1.0))
        d_theta = 2.0 * half_angle
        idx_grid = jnp.arange(max_ap)
        angles = planet_angle - (idx_grid + 1 + buffer) * d_theta
        ref_y = cy + r_pix * jnp.sin(angles)
        ref_x = cx + r_pix * jnp.cos(angles)
        expected = map_coordinates(fmap, jnp.stack([ref_y, ref_x]), order=3)

        np.testing.assert_allclose(
            np.asarray(samples), np.asarray(expected), rtol=1e-6, atol=1e-6
        )
        n_expected = int(n_reference_apertures(float(r_pix), fwhm, buffer))
        assert int(jnp.sum(mask)) == n_expected
        assert bool(geom_ok)

    def test_validity_map_excludes_apertures(self):
        """Zeroed validity sectors drop reference apertures from the mask."""
        _, fmap, filt, validity, center = _setup()
        pos = jnp.array([50.0, 50.0 + 20.0])
        sampler = ApertureSampler(fwhm=5.0)
        _, mask_all, _ = sampler(fmap, pos, filt, validity, center)
        blocked = validity.at[:, :50].set(0.0)  # kill the left half-plane
        _, mask_blocked, _ = sampler(fmap, pos, filt, blocked, center)
        assert int(jnp.sum(mask_blocked)) < int(jnp.sum(mask_all))

    def test_geom_ok_false_inside_fwhm(self):
        """Candidates inside one resolution element are geometry-invalid."""
        _, fmap, filt, validity, center = _setup()
        pos = jnp.array([50.0 + 2.0, 50.0])  # r = 2 < fwhm = 5
        sampler = ApertureSampler(fwhm=5.0)
        _, _, geom_ok = sampler(fmap, pos, filt, validity, center)
        assert not bool(geom_ok)

    def test_center_shift_moves_geometry(self):
        """An off-center star center changes the reference ring."""
        _, fmap, filt, validity, center = _setup()
        pos = jnp.array([65.0, 50.0])
        sampler = ApertureSampler(fwhm=5.0)
        s_geo, _, _ = sampler(fmap, pos, filt, validity, center)
        s_off, _, _ = sampler(fmap, pos, filt, validity, jnp.array([40.0, 50.0]))
        assert not np.allclose(np.asarray(s_geo), np.asarray(s_off))


class TestAnnulusSampler:
    """Full-annulus pixel sampling (matched-filter geometry)."""

    def test_reproduces_fused_core_annulus_mask(self):
        """Mask equals the v1.1.1 matched-filter inline annulus."""
        n = 101
        image = jax.random.normal(jax.random.PRNGKey(1), (n, n))
        filt = GaussianFilter(kernel_1d=gaussian_kernel_1d(5.0), order=3)
        filtered = filt.prepare(image)
        validity = jnp.ones((n, n))
        center = jnp.array([(n - 1) / 2.0, (n - 1) / 2.0])
        fwhm = 5.0
        pos = jnp.array([50.0, 50.0 + 18.0])

        sampler = AnnulusSampler(fwhm=fwhm)
        samples, mask, geom_ok = sampler(filtered, pos, filt, validity, center)

        # Inline v1.1.1 reference (_matched_filter_snr_batch_core lines).
        cy, cx = center[0], center[1]
        py, px = pos[0], pos[1]
        max_radius = jnp.minimum(n, n) / 2 - 1
        y_coords, x_coords = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        r_grid = jnp.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        r_planet = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)
        default_inner = jnp.maximum(r_planet - fwhm, fwhm)
        default_outer = jnp.minimum(r_planet + fwhm, max_radius)
        default_inner = jnp.minimum(default_inner, default_outer - 1.0)
        planet_dist = jnp.sqrt((y_coords - py) ** 2 + (x_coords - px) ** 2)
        expected_mask = (
            (r_grid >= default_inner)
            & (r_grid <= default_outer)
            & (planet_dist > fwhm * 1.5)
        )

        np.testing.assert_array_equal(
            np.asarray(mask), np.asarray(expected_mask.ravel())
        )
        np.testing.assert_array_equal(np.asarray(samples), np.asarray(filtered.ravel()))
        assert bool(geom_ok)

    def test_explicit_annulus_bounds_override_defaults(self):
        """Non-negative annulus bounds replace the automatic ones."""
        n = 101
        image = jax.random.normal(jax.random.PRNGKey(2), (n, n))
        filt = GaussianFilter(kernel_1d=gaussian_kernel_1d(4.0), order=3)
        filtered = filt.prepare(image)
        validity = jnp.ones((n, n))
        center = jnp.array([(n - 1) / 2.0, (n - 1) / 2.0])
        pos = jnp.array([50.0, 70.0])
        auto = AnnulusSampler(fwhm=4.0)
        fixed = AnnulusSampler(fwhm=4.0, annulus_inner=10.0, annulus_outer=14.0)
        _, mask_auto, _ = auto(filtered, pos, filt, validity, center)
        _, mask_fixed, _ = fixed(filtered, pos, filt, validity, center)
        assert int(jnp.sum(mask_fixed)) != int(jnp.sum(mask_auto))

    def test_validity_map_thins_annulus(self):
        """Invalid pixels drop out of the annulus mask."""
        n = 101
        image = jax.random.normal(jax.random.PRNGKey(3), (n, n))
        filt = GaussianFilter(kernel_1d=gaussian_kernel_1d(4.0), order=3)
        filtered = filt.prepare(image)
        center = jnp.array([(n - 1) / 2.0, (n - 1) / 2.0])
        pos = jnp.array([50.0, 70.0])
        sampler = AnnulusSampler(fwhm=4.0)
        _, mask_full, _ = sampler(filtered, pos, filt, jnp.ones((n, n)), center)
        _, mask_half, _ = sampler(
            filtered, pos, filt, jnp.ones((n, n)).at[:50, :].set(0.0), center
        )
        assert int(jnp.sum(mask_half)) < int(jnp.sum(mask_full))


def test_annulus_uses_propagated_gaussian_validity():
    """NumPy reflected box erosion defines the mask; pixel membership is exact."""
    from photomancy.postproc import DetectionEstimator

    image = jax.random.normal(jax.random.PRNGKey(18), (33, 33))
    valid = np.ones((33, 33))
    valid[16, 5] = 0.0
    estimator = DetectionEstimator.matched_filter(2.5, fast=True)
    width = len(estimator.filter.kernel_1d)
    padded = np.pad(valid > 0.5, width // 2, mode="reflect")
    expected = np.lib.stride_tricks.sliding_window_view(padded, (width, width)).all(
        axis=(-2, -1)
    )
    prepared_validity = estimator.filter.prepare_validity(jnp.asarray(valid))
    filtered = estimator.filter.prepare(jnp.where(jnp.asarray(valid) > 0.5, image, 0.0))
    position = jnp.array([16.0, 27.0])
    center = jnp.array([16.0, 16.0])
    samples, mask, _ = estimator.sampler(
        filtered, position, estimator.filter, prepared_validity, center
    )
    _, all_mask, _ = estimator.sampler(
        filtered, position, estimator.filter, jnp.ones_like(valid), center
    )
    np.testing.assert_array_equal(mask, np.asarray(all_mask) & expected.ravel())
    assert mask.sum() < all_mask.sum()
    signal = estimator.filter.evaluate(filtered, position)
    statistic, fpf, _ = estimator.test(signal, samples, mask)
    actual = estimator(image, position[None], jnp.asarray(valid), center)
    np.testing.assert_allclose(actual.statistic[0], statistic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.fpf[0], fpf, rtol=1e-12, atol=1e-12)


def test_gaussian_references_use_detector_bounds():
    """Nearest reference centres must lie on the 33-pixel detector, exactly."""
    image = jnp.ones((33, 33))
    filt = GaussianFilter(jnp.ones(4) / 4, order=1)
    position = jnp.array([16.0, 33.0])
    center = jnp.array([16.0, 16.0])
    sampler = ApertureSampler(fwhm=5.0, max_apertures=32)
    _, mask, _ = sampler(
        filt.prepare(image), position, filt, filt.prepare_validity(image), center
    )
    step = 2 * np.arcsin(5 / 34)
    angles = -(np.arange(32) + 1.5) * step
    refs = 16 + 17 * np.stack([np.sin(angles), np.cos(angles)], axis=1)
    nearest = np.sign(refs) * np.floor(np.abs(refs) + 0.5)
    active = np.arange(32) < int(2 * np.pi / step) - 2
    expected = active & ((0 <= nearest) & (nearest < 33)).all(axis=1)
    np.testing.assert_array_equal(mask, expected)
