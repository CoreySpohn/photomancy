"""Tests for detection filter primitives."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.map_coordinates import map_coordinates

from photomancy.postproc.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)
from photomancy.postproc.core.photometry import flux_map, make_aperture_kernel


def _test_image(seed: int = 0, n: int = 64) -> jnp.ndarray:
    """Deterministic random image."""
    return jax.random.normal(jax.random.PRNGKey(seed), (n, n))


class TestApertureFilter:
    """Aperture filter reproduces flux-map sampling."""

    def test_prepare_is_flux_map(self):
        """prepare() equals photometry.flux_map with the same kernel."""
        image = _test_image()
        kernel = make_aperture_kernel(radius=2.5, soft=True, sharpness=10.0)
        filt = ApertureFilter(kernel=kernel, order=3)
        np.testing.assert_array_equal(
            np.asarray(filt.prepare(image)), np.asarray(flux_map(image, kernel))
        )

    def test_evaluate_matches_map_coordinates(self):
        """evaluate() equals a direct map_coordinates sample."""
        image = _test_image(1)
        kernel = make_aperture_kernel(radius=2.5, soft=True, sharpness=10.0)
        filt = ApertureFilter(kernel=kernel, order=3)
        fmap = filt.prepare(image)
        pos = jnp.array([31.3, 40.7])
        expected = map_coordinates(fmap, jnp.array([[31.3], [40.7]]), order=3)[0]
        np.testing.assert_array_equal(
            np.asarray(filt.evaluate(fmap, pos)), np.asarray(expected)
        )

    def test_is_jit_and_vmap_compatible(self):
        """Filter passes through jit as a pytree and vmaps over positions."""
        image = _test_image(2)
        kernel = make_aperture_kernel(radius=2.0, soft=True, sharpness=10.0)
        filt = ApertureFilter(kernel=kernel, order=1)

        @eqx.filter_jit
        def run(f, img, positions):
            """Evaluate the filter at many positions."""
            fmap = f.prepare(img)
            return jax.vmap(lambda p: f.evaluate(fmap, p))(positions)

        out = run(filt, image, jnp.array([[20.0, 20.0], [30.5, 41.5]]))
        assert out.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(out)))


class TestGaussianFilter:
    """Gaussian filter reproduces the matched-filter preprocessing."""

    def test_kernel_is_centered_and_normalized(self):
        """The Gaussian kernel has the expected support and preserves flux."""
        kernel = np.asarray(gaussian_kernel_1d(4.0))
        assert kernel.shape == (11,)
        assert np.argmax(kernel) == 5
        np.testing.assert_array_equal(kernel, kernel[::-1])
        np.testing.assert_allclose(kernel.sum(), 1.0)

    def test_prepare_uses_separable_convolution(self):
        """prepare() equals gaussian_filter_2d on the same kernel."""
        image = _test_image(3)
        kernel_1d = gaussian_kernel_1d(5.0)
        filt = GaussianFilter(kernel_1d=kernel_1d, order=3)
        np.testing.assert_array_equal(
            np.asarray(filt.prepare(image)),
            np.asarray(gaussian_filter_2d(image, kernel_1d)),
        )

    def test_filtered_image_smooths_noise(self):
        """Gaussian filtering reduces pixel-to-pixel variance."""
        image = _test_image(4)
        filtered = gaussian_filter_2d(image, gaussian_kernel_1d(5.0))
        assert float(jnp.std(filtered)) < float(jnp.std(image))


@pytest.mark.parametrize("kind", ["aperture", "gaussian", "map", "template"])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_validity_box_and_stencil(kind, order):
    """NumPy padded box counts and explicit interpolation nodes are exact oracles."""
    from photomancy.postproc import ArrayTemplateProvider, MapFilter, PSFTemplateFilter

    filters = {
        "aperture": ApertureFilter(jnp.ones((5, 5)), order),
        "gaussian": GaussianFilter(jnp.ones(5) / 5, order),
        "map": MapFilter(order),
        "template": PSFTemplateFilter(ArrayTemplateProvider(jnp.ones((5, 5))), order),
    }
    filt = filters[kind]
    valid = np.ones((11, 13))
    valid[1, 2] = 0.0
    valid[6, 7] = 0.5
    width = 1 if kind == "map" else 5
    padded = np.pad(
        valid > 0.5,
        width // 2,
        mode="reflect" if kind == "gaussian" else "constant",
        **({} if kind == "gaussian" else {"constant_values": True}),
    )
    expected = np.lib.stride_tricks.sliding_window_view(padded, (width, width)).all(
        axis=(-2, -1)
    )
    prepared = eqx.filter_jit(lambda v: filt.prepare_validity(v))(jnp.asarray(valid))
    np.testing.assert_array_equal(prepared, expected)
    assert np.asarray(filt.prepare_validity(jnp.ones(valid.shape))).all()
    positions = np.array(
        [[0.0, 0.0], [-0.5, 2.0], [2.25, 3.75], [7.5, 8.5], [10.5, 12.5], [20.0, 20.0]]
    )
    for position in positions:
        if order == 0:
            nodes = (np.sign(position) * np.floor(np.abs(position) + 0.5)).astype(int)[
                :, None
            ]
        else:
            offsets = np.arange(-1, 3) if order == 3 else np.arange(2)
            nodes = np.floor(position).astype(int)[:, None] + offsets
        usable = all(
            expected[y, x]
            for y in nodes[0]
            for x in nodes[1]
            if 0 <= y < valid.shape[0] and 0 <= x < valid.shape[1]
        )
        actual = eqx.filter_jit(filt.evaluate_validity)(prepared, jnp.asarray(position))
        assert bool(actual) == usable


def test_aperture_validity_even_box_matches_convolution():
    """SciPy integer convolution defines even-box alignment; validity is exact."""
    from scipy.signal import convolve2d

    valid = np.ones((11, 13))
    valid[5, 6] = 0.0
    filt = ApertureFilter(jnp.ones((4, 6)), order=1)
    expected = (
        convolve2d((valid <= 0.5).astype(int), np.ones((4, 6), dtype=int), mode="same")
        == 0
    )
    np.testing.assert_array_equal(filt.prepare_validity(jnp.asarray(valid)), expected)
