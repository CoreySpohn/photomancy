"""Tests for the PSF-template matched-filter post-processing arm."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from hwoutils.constants import J2000_JD
from hwoutils.epoch import Epoch
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band

from photomancy.postproc import ApertureSampler, DetectionEstimator, TwoSampleTTest
from photomancy.postproc.arms import MatchedFilterArm
from photomancy.postproc.contracts import DetectionStats, FrameSet
from photomancy.postproc.core.detection import PSFTemplateFilter
from photomancy.postproc.templates import ArrayTemplateProvider

FWHM_PX = 5.0
STAMP_SIZE = 15


def _gaussian_stamp(size: int = STAMP_SIZE, fwhm: float = FWHM_PX) -> jnp.ndarray:
    """Peak-normalized Gaussian template stamp of shape (size, size)."""
    c = (size - 1) / 2.0
    y, x = jnp.mgrid[:size, :size]
    sigma = fwhm / 2.355
    return jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))


def make_frameset(
    key,
    n_frames: int = 3,
    shape: tuple[int, int] = (64, 64),
    frame_time_s: float = 1.0,
    noise_variance: jnp.ndarray | None = None,
) -> FrameSet:
    """Build a FrameSet of independent seeded white-noise frames.

    Args:
        key: PRNG key for the frame noise.
        n_frames: Number of frames to generate.
        shape: (ny, nx) frame shape.
        frame_time_s: Per-frame exposure time, seconds (uniform).
        noise_variance: Optional (n_frames, ny, nx) per-pixel variance.

    Returns:
        A FrameSet of n_frames independent standard-normal frames.
    """
    frames = jr.normal(key, (n_frames, *shape))
    return FrameSet(
        frames=frames,
        time_d=jnp.arange(n_frames, dtype=float),
        frame_time_s=jnp.full((n_frames,), frame_time_s),
        telescope_pa_deg=jnp.zeros((n_frames,)),
        fwhm_px=FWHM_PX,
        band=Band(550.0 - 110.0 / 2, 550.0 + 110.0 / 2),
        epoch=Epoch(J2000_JD),
        grid=PixelGrid.regular(shape, 0.0218, "arcsec"),
        noise_variance=noise_variance,
    )


def _manual_composition(provider, noise_variance=None) -> DetectionEstimator:
    """Hand-build the exact composition MatchedFilterArm.detect wires up."""
    return DetectionEstimator(
        filter=PSFTemplateFilter(
            provider=provider, order=3, noise_variance=noise_variance
        ),
        sampler=ApertureSampler(fwhm=FWHM_PX, max_apertures=200, exclusion_buffer=0.5),
        test=TwoSampleTTest(),
    )


def test_arm_matches_manual_composition_bitwise():
    """The arm's statistic and fpf match a hand-built DetectionEstimator exactly."""
    fs = make_frameset(jr.PRNGKey(100))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    provider = ArrayTemplateProvider(_gaussian_stamp())
    arm = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX)
    result = arm.detect(fs, positions)

    manual = _manual_composition(provider)
    expected = manual(
        fs.coadd(),
        positions,
        fs.valid,
        jnp.array([fs.grid.center_y, fs.grid.center_x]),
    )

    np.testing.assert_array_equal(
        np.asarray(result.statistic), np.asarray(expected.statistic)
    )
    np.testing.assert_array_equal(np.asarray(result.fpf), np.asarray(expected.fpf))


def test_statistic_kind_and_result_type():
    """statistic_kind is psf_template_t and detect returns a DetectionStats."""
    fs = make_frameset(jr.PRNGKey(101))
    positions = jnp.array([[32.0, 48.0]])
    provider = ArrayTemplateProvider(_gaussian_stamp())
    arm = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX)
    result = arm.detect(fs, positions)
    assert isinstance(result, DetectionStats)
    assert result.statistic_kind == "psf_template_t"


def test_grid_center_consulted():
    """An offset star center changes the statistic vs a geometric-center FrameSet."""
    fs = make_frameset(jr.PRNGKey(102))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    fs_off = eqx.tree_at(
        lambda f: f.grid,
        fs,
        replace(fs.grid, center_y=20.0, center_x=40.0),
    )
    provider = ArrayTemplateProvider(_gaussian_stamp())
    arm = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX)
    default_center = arm.detect(fs, positions)
    offset_center = arm.detect(fs_off, positions)
    assert not np.array_equal(
        np.asarray(default_center.statistic),
        np.asarray(offset_center.statistic),
        equal_nan=True,
    )


def test_whitening_differs_with_heteroscedastic_variance():
    """A heteroscedastic noise_variance makes whiten=True differ from whiten=False."""
    ny, nx = 64, 64
    _, x = jnp.mgrid[:ny, :nx]
    variance_map = 0.5 + 4.5 * (x / nx)  # strictly positive, varies across columns
    noise_variance = jnp.broadcast_to(variance_map, (3, ny, nx))
    fs = make_frameset(jr.PRNGKey(103), shape=(ny, nx), noise_variance=noise_variance)
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    provider = ArrayTemplateProvider(_gaussian_stamp())

    whitened = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX, whiten=True)
    unwhitened = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX, whiten=False)
    with_whitening = whitened.detect(fs, positions)
    without_whitening = unwhitened.detect(fs, positions)

    assert not np.allclose(
        np.asarray(with_whitening.statistic),
        np.asarray(without_whitening.statistic),
        equal_nan=True,
    )


def test_whitening_matches_unwhitened_for_uniform_variance():
    """A spatially uniform noise_variance makes whiten=True match whiten=False.

    The arm computes variance = sum(noise_variance, axis=0) /
    sum(frame_time_s)**2; a per-pixel-uniform noise_variance stays
    uniform through that propagation (a constant summed over frames,
    divided by a scalar), so the inverse-variance GLS weights are the same
    constant at every pixel. A constant weight factors out of every
    weighted mean and weighted sum in PSFTemplateFilter.evaluate, so the
    whitened response reduces algebraically to the unwhitened response.
    """
    ny, nx = 64, 64
    noise_variance = jnp.full((3, ny, nx), 2.5)
    fs = make_frameset(jr.PRNGKey(104), shape=(ny, nx), noise_variance=noise_variance)
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    provider = ArrayTemplateProvider(_gaussian_stamp())

    whitened = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX, whiten=True)
    unwhitened = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX, whiten=False)
    with_whitening = whitened.detect(fs, positions)
    without_whitening = unwhitened.detect(fs, positions)

    # The statistic is built from float32 weighted sums, whose rounding
    # depends on the XLA reduction order: jax 0.11.2 and 0.10.2 differ by
    # 1.4e-6 in relative terms here, so the comparison uses 1e-5.
    np.testing.assert_allclose(
        np.asarray(with_whitening.statistic),
        np.asarray(without_whitening.statistic),
        rtol=1e-5,
    )
    # The false-positive fraction is a float32 survival function of the
    # statistic, which amplifies a one-ulp statistic difference to a few
    # 1e-6 in relative terms, so it is compared at 1e-5.
    np.testing.assert_allclose(
        np.asarray(with_whitening.fpf), np.asarray(without_whitening.fpf), rtol=1e-5
    )


def test_detect_is_jit_safe():
    """Detect under eqx.filter_jit matches the direct call."""
    fs = make_frameset(jr.PRNGKey(105))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    provider = ArrayTemplateProvider(_gaussian_stamp())
    arm = MatchedFilterArm(provider=provider, fwhm_px=FWHM_PX)

    direct = arm.detect(fs, positions)
    jitted = eqx.filter_jit(lambda a, f, p: a.detect(f, p))(arm, fs, positions)

    np.testing.assert_array_equal(
        np.asarray(direct.statistic), np.asarray(jitted.statistic)
    )
    np.testing.assert_array_equal(np.asarray(direct.fpf), np.asarray(jitted.fpf))
