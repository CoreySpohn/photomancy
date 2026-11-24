"""Tests for the PSF-template matched-filter post-processing arm."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
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
    arm = MatchedFilterArm(provider=provider)
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
    arm = MatchedFilterArm(provider=provider)
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
    arm = MatchedFilterArm(provider=provider)
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

    whitened = MatchedFilterArm(provider=provider, whiten=True)
    unwhitened = MatchedFilterArm(provider=provider, whiten=False)
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

    whitened = MatchedFilterArm(provider=provider, whiten=True)
    unwhitened = MatchedFilterArm(provider=provider, whiten=False)
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
    arm = MatchedFilterArm(provider=provider)

    direct = arm.detect(fs, positions)
    jitted = eqx.filter_jit(lambda a, f, p: a.detect(f, p))(arm, fs, positions)

    np.testing.assert_array_equal(
        np.asarray(direct.statistic), np.asarray(jitted.statistic)
    )
    np.testing.assert_array_equal(np.asarray(direct.fpf), np.asarray(jitted.fpf))


def test_excluded_variance_never_enters_nearest_neighbor_fit():
    """Exclusion is the oracle: an excluded pixel's variance changes nothing, bitwise.

    With order=0 the image patch is read nearest-neighbor but the variance
    patch bilinearly, so the bilinear node one pixel past the rounded stamp
    edge must count toward the candidate's footprint.
    """
    shape = (64, 64)
    position = jnp.array([[32.2, 48.2]])
    far = jnp.array([[32.0, 16.0]])
    # Rounded stamp rows reach 39; the bilinear variance read reaches row 40.
    pixel = (40, 48)
    valid = jnp.ones(shape).at[pixel].set(0.0)
    base = make_frameset(jr.PRNGKey(11), noise_variance=jnp.ones((3, *shape)))
    base = replace(base, valid=valid)
    provider = ArrayTemplateProvider(_gaussian_stamp())
    arm = MatchedFilterArm(provider=provider, order=0, whiten=True)
    positions = jnp.concatenate([position, far])
    results = [
        arm.detect(
            replace(base, noise_variance=base.noise_variance.at[:, *pixel].set(value)),
            positions,
        )
        for value in (1.0, 1e-6, np.nan)
    ]
    for other in results[1:]:
        for field in ("statistic", "fpf", "dof"):
            assert np.array_equal(
                np.asarray(getattr(results[0], field)),
                np.asarray(getattr(other, field)),
                equal_nan=True,
            ), field
    assert np.isnan(results[0].statistic[0])
    assert np.isfinite(results[0].statistic[1])


def _edge_estimator(order=0):
    """Four-pixel stamp with an aperture ring that crosses the detector edge."""
    return DetectionEstimator(
        PSFTemplateFilter(ArrayTemplateProvider(_gaussian_stamp(4)), order),
        ApertureSampler(FWHM_PX),
        TwoSampleTTest(),
    )


def test_template_candidate_at_detector_edge():
    """An excluded nearest node gates t and FPF exactly, for either pixel value."""
    image = jr.normal(jr.PRNGKey(12), (33, 33))
    position = jnp.array([[32.1, 16.2]])
    pixel = (31, 18)
    valid = jnp.ones(image.shape).at[pixel].set(0)
    estimator = _edge_estimator()
    assert np.isfinite(estimator(image, position).statistic[0])
    result = estimator(image, position, valid)
    assert np.isnan(result.statistic[0]) and np.isnan(result.fpf[0])
    changed = estimator(image.at[pixel].set(1e6), position, valid)
    for field in ("statistic", "fpf", "dof"):
        np.testing.assert_array_equal(getattr(result, field), getattr(changed, field))


def _numpy_ring_responses(image, position, stamp, order):
    """Fit SciPy patches on the ring of a candidate due east of (16, 16)."""
    from scipy.ndimage import map_coordinates

    radius = position[1] - 16
    step = 2 * np.arcsin(FWHM_PX / (2 * radius))
    angles = -(np.arange(int(2 * np.pi / step) - 2) + 1.5) * step
    refs = 16 + radius * np.stack([np.sin(angles), np.cos(angles)], axis=1)
    offsets = np.indices(stamp.shape) - (len(stamp) - 1) / 2
    centers = np.vstack([position, refs])
    coords = centers.T[:, :, None, None] + offsets[:, None]
    patches = map_coordinates(image, coords, order=order, mode="grid-constant")
    template = stamp - stamp.mean()
    amplitudes = (patches * template).sum(axis=(-2, -1)) / (template**2).sum()
    return amplitudes[0], amplitudes[1:]


def _numpy_t(signal, samples):
    """Small-sample t with the Bessel-corrected reference spread."""
    return (signal - samples.mean()) / (
        samples.std(ddof=1) * np.sqrt(1 + 1 / len(samples))
    )


@pytest.mark.parametrize("pixel, dropped", [((31, 19), True), ((30, 16), False)])
def test_template_edge_reference_footprint(pixel, dropped):
    """SciPy nearest patches set the exact dof and the float64 t within 1e-10."""
    position = np.array([16.0, 32.2])
    image = np.random.default_rng(13).normal(size=(33, 33))
    estimator = _edge_estimator()
    signal, refs = _numpy_ring_responses(
        image, position, np.asarray(_gaussian_stamp(4)), 0
    )
    all_valid = estimator(jnp.asarray(image), jnp.asarray(position[None]))
    assert all_valid.dof[0] == len(refs) - 1
    # Reference 14 at (32.133, 17.468) reads rows 31..34 and columns 16..19.
    # Its nearest centre is on the detector. Only this patch reads (31, 19).
    keep = np.delete(refs, 14) if dropped else refs
    valid = jnp.ones(image.shape).at[pixel].set(0)
    result = estimator(jnp.asarray(image), jnp.asarray(position[None]), valid)
    assert result.dof[0] == len(keep) - 1
    np.testing.assert_allclose(
        result.statistic[0], _numpy_t(signal, keep), rtol=0, atol=1e-10
    )


def test_template_integer_image_interpolation():
    """Fits to SciPy-rounded integer patches agree within float64 roundoff (1e-10)."""
    image = np.random.default_rng(14).integers(-20, 20, size=(33, 33), dtype=np.int32)
    position = np.array([16.0, 27.25])
    signal, refs = _numpy_ring_responses(
        image, position, np.asarray(_gaussian_stamp(4)), 1
    )
    result = _edge_estimator(order=1)(jnp.asarray(image), jnp.asarray(position[None]))
    np.testing.assert_allclose(
        result.statistic[0], _numpy_t(signal, refs), rtol=0, atol=1e-10
    )


def test_whitened_edge_reference_variance():
    """Dropping one bilinear variance footprint gives exact fill-value invariance."""
    variance = jnp.ones((1, 33, 33))
    science = make_frameset(
        jr.PRNGKey(15), n_frames=1, shape=(33, 33), noise_variance=variance
    )
    # Only the bilinear variance footprint of reference 14 reaches this pixel.
    pixel = (31, 15)
    positions = jnp.array([[16.0, 32.2], [33.1, 16.2]])
    arm = MatchedFilterArm(ArrayTemplateProvider(_gaussian_stamp(4)), order=0)
    all_valid = arm.detect(science, positions)
    science = replace(science, valid=jnp.ones((33, 33)).at[pixel].set(0))
    expected = arm.detect(science, positions)
    assert expected.dof[0] == all_valid.dof[0] - 1
    assert np.isfinite(expected.statistic[0])
    assert np.isfinite(all_valid.statistic[1])
    assert np.isnan(expected.statistic[1]) and np.isnan(expected.fpf[1])
    dirty = replace(science, noise_variance=variance.at[:, *pixel].set(np.nan))
    nonfinite = arm.detect(dirty, positions)
    # The arm also keeps the discarded fit finite in the sampler's output buffer.
    composed = arm.estimator(dirty)
    filt = composed.filter.bind(positions[0])
    samples, mask, _ = composed.sampler(
        jnp.where(dirty.valid > 0.5, dirty.coadd(), 0.0),
        positions[0],
        filt,
        filt.prepare_validity(dirty.valid),
        jnp.array([16.0, 16.0]),
    )
    assert not mask[14] and np.isfinite(samples[14])
    # Change the actual replacement in the composed estimator after arm preparation.
    estimator = arm.estimator(science)
    tiny = eqx.tree_at(
        lambda e: e.filter.noise_variance,
        estimator,
        estimator.filter.noise_variance.at[pixel].set(1e-12),
    )
    replaced = tiny(science.coadd(), positions, science.valid)
    for field in ("statistic", "fpf", "dof"):
        for other in (nonfinite, replaced):
            np.testing.assert_array_equal(
                getattr(expected, field), getattr(other, field)
            )
