"""Tests for the post-processing seam ABC and the Mawet arm."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from hwoutils.constants import J2000_JD
from hwoutils.epoch import Epoch
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band

from photomancy.postproc.arms import AbstractDetectionArm, MawetArm
from photomancy.postproc.contracts import DetectionStats, FrameSet
from photomancy.postproc.core.detection.estimator import DetectionEstimator
from photomancy.postproc.core.modeling import inject_planet

FWHM_PX = 4.0


def make_frameset(image, n_frames=1, frame_time_s=1.0):
    """Build a single-image FrameSet broadcast to n_frames identical frames."""
    frames = jnp.broadcast_to(image, (n_frames, *image.shape))
    return FrameSet(
        frames=frames,
        time_d=jnp.arange(n_frames, dtype=float),
        frame_time_s=jnp.full((n_frames,), frame_time_s),
        telescope_pa_deg=jnp.zeros((n_frames,)),
        fwhm_px=FWHM_PX,
        band=Band(550.0 - 110.0 / 2, 550.0 + 110.0 / 2),
        epoch=Epoch(J2000_JD),
        grid=PixelGrid.regular(image.shape, 0.0218, "arcsec"),
    )


def gaussian_psf(n=15, sigma=FWHM_PX / 2.355):
    """Build a normalized 2D Gaussian PSF template of size (n, n)."""
    y, x = jnp.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    psf = jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))
    return psf / jnp.sum(psf)


def test_abstract_postprocessing_cannot_instantiate():
    """Test that the abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AbstractDetectionArm()


def test_mawet_arm_statistic_matches_snr_function():
    """Test that the Mawet arm's statistic matches the estimator on the same image."""
    key = jr.PRNGKey(0)
    image = jr.normal(key, (64, 64))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    fs = make_frameset(image)
    arm = MawetArm()
    stats = arm.detect(fs, positions)
    expected = DetectionEstimator.mawet(FWHM_PX)(image, positions).statistic
    assert isinstance(stats, DetectionStats)
    assert jnp.allclose(stats.statistic, expected, equal_nan=True)
    assert stats.statistic_kind == "mawet_t"


def test_mawet_arm_bright_planet_has_small_fpf():
    """Test that a bright injected planet yields a high statistic and small FPF."""
    key = jr.PRNGKey(1)
    image = jr.normal(key, (64, 64))
    planet_pos = jnp.array([32.0, 50.0])
    image = inject_planet(image, gaussian_psf(n=64), flux=200.0, pos=planet_pos)
    fs = make_frameset(image)
    arm = MawetArm()
    stats = arm.detect(fs, planet_pos[None, :])
    assert float(stats.statistic[0]) > 5.0
    assert float(stats.fpf[0]) < 1e-3
    assert float(stats.dof[0]) >= 2.0


def test_mawet_arm_empty_field_fpf_is_calibrated():
    """Test that FPF values over an empty noise field are ~Uniform(0, 1)."""
    key = jr.PRNGKey(2)
    image = jr.normal(key, (128, 128))
    fs = make_frameset(image)
    arm = MawetArm()
    # 50 random positions in an annulus well inside the field
    pos_key, angle_key = jr.split(jr.PRNGKey(3))
    radii = jr.uniform(pos_key, (50,), minval=15.0, maxval=45.0)
    angles = jr.uniform(angle_key, (50,), minval=0.0, maxval=2 * jnp.pi)
    positions = jnp.stack(
        [63.5 + radii * jnp.sin(angles), 63.5 + radii * jnp.cos(angles)], axis=1
    )
    stats = arm.detect(fs, positions)
    finite = jnp.isfinite(stats.fpf)
    assert int(jnp.sum(finite)) > 40
    median_fpf = jnp.median(stats.fpf[finite])
    # Under H0 the FPF is ~Uniform(0, 1); its median should be central.
    assert 0.2 < float(median_fpf) < 0.8


def test_mawet_arm_nan_statistic_gives_nan_fpf():
    """Test that a too-close-in position gives a NaN statistic and NaN FPF."""
    key = jr.PRNGKey(4)
    image = jr.normal(key, (64, 64))
    fs = make_frameset(image)
    arm = MawetArm()
    stats = arm.detect(fs, jnp.array([[31.5, 32.5]]))  # r < fwhm -> NaN
    assert bool(jnp.isnan(stats.statistic[0]))
    assert bool(jnp.isnan(stats.fpf[0]))


def test_mawet_arm_ignores_references_and_uses_validity():
    """Test that the Mawet arm ignores references and honors science.valid."""
    key = jr.PRNGKey(5)
    image = jr.normal(key, (64, 64))
    validity = jnp.ones((64, 64)).at[:, :20].set(0.0)
    fs = eqx.tree_at(
        lambda f: f.valid,
        make_frameset(image),
        validity,
        is_leaf=lambda x: x is None,
    )
    arm = MawetArm()
    positions = jnp.array([[32.0, 48.0]])
    with_refs = arm.detect(fs, positions, references=make_frameset(image))
    without = arm.detect(fs, positions)
    assert jnp.allclose(with_refs.statistic, without.statistic, equal_nan=True)
    expected = DetectionEstimator.mawet(FWHM_PX)(image, positions, validity).statistic
    assert jnp.allclose(without.statistic, expected, equal_nan=True)


def test_mawet_arm_multiframe_uses_coadd():
    """Test that a multi-frame FrameSet is detected on its exposure-weighted coadd."""
    key = jr.PRNGKey(6)
    image = jr.normal(key, (64, 64)) + 10.0
    fs = make_frameset(image, n_frames=4, frame_time_s=25.0)
    arm = MawetArm()
    positions = jnp.array([[32.0, 48.0]])
    stats = arm.detect(fs, positions)
    expected = DetectionEstimator.mawet(FWHM_PX)(fs.coadd(), positions).statistic
    assert jnp.allclose(stats.statistic, expected, equal_nan=True)


def test_mawet_arm_detect_works_under_jit():
    """Direct evaluation is the oracle; closed resolution gives exact JIT parity."""
    key = jr.PRNGKey(7)
    image = jr.normal(key, (64, 64))
    fs = make_frameset(image)
    arm = MawetArm()
    positions = jnp.array([[32.0, 48.0]])

    @jax.jit
    def run(frames, p):
        f = eqx.tree_at(lambda s: s.frames, fs, frames)
        return arm.detect(f, p).statistic

    np.testing.assert_array_equal(
        run(fs.frames, positions), arm.detect(fs, positions).statistic
    )


def test_mawet_arm_coadd_uses_content_differing_frames():
    """Test that the arm coadds content-differing frames rather than using frames[0].

    Frame 0 carries the planet at double amplitude; frame 1 is planet-free.
    The exposure-weighted coadd reproduces the Mawet statistic on the explicit coadd and
    differs from the frames[0]-only statistic. (An unweighted mean is
    scale-proportional to the coadd, so scale invariance makes those two
    indistinguishable BY CONSTRUCTION; frame selection is the failure mode
    this guards.)
    """
    n = 64
    planet_pos = jnp.array([32.0, 48.0])
    key = jr.PRNGKey(9)
    base = 5.0 + jr.normal(key, (n, n))
    planet = inject_planet(
        jnp.zeros((n, n)), gaussian_psf(n=n), flux=300.0, pos=planet_pos
    )
    frames = jnp.stack([base + 2.0 * planet, base])
    exposure = jnp.array([1.0, 3.0])
    fs = FrameSet(
        frames=frames,
        time_d=jnp.array([0.0, 1.0]),
        frame_time_s=exposure,
        telescope_pa_deg=jnp.zeros(2),
        fwhm_px=FWHM_PX,
        band=Band(550.0 - 100.0 / 2, 550.0 + 100.0 / 2),
        epoch=Epoch(J2000_JD),
        grid=PixelGrid.regular((n, n), 0.020, "arcsec"),
    )
    arm = MawetArm()
    stats = arm.detect(fs, planet_pos[None, :])
    explicit_coadd = jnp.sum(frames, axis=0) / jnp.sum(exposure)
    expected = DetectionEstimator.mawet(FWHM_PX)(
        explicit_coadd, planet_pos[None, :]
    ).statistic
    np.testing.assert_array_equal(np.asarray(stats.statistic), np.asarray(expected))
    # frames[0]-only and unweighted-mean statistics differ from the coadd's.
    from_frame0 = DetectionEstimator.mawet(FWHM_PX)(
        frames[0], planet_pos[None, :]
    ).statistic
    assert not np.array_equal(np.asarray(stats.statistic), np.asarray(from_frame0))


def test_mawet_arm_grid_center_matches_geometric():
    """The default grid center equals an explicit geometric center exactly."""
    key = jr.PRNGKey(10)
    image = jr.normal(key, (64, 64))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    fs = make_frameset(image)
    n = fs.frames.shape[1]
    c = (n - 1) / 2.0
    fs_center = eqx.tree_at(
        lambda f: f.grid,
        fs,
        replace(fs.grid, center_x=c, center_y=c),
    )
    arm = MawetArm()
    a = arm.detect(fs, positions)
    b = arm.detect(fs_center, positions)
    np.testing.assert_array_equal(np.asarray(a.statistic), np.asarray(b.statistic))
    np.testing.assert_array_equal(np.asarray(a.fpf), np.asarray(b.fpf))


def test_mawet_arm_consults_off_center():
    """An off-center star center changes the reference geometry."""
    key = jr.PRNGKey(11)
    image = jr.normal(key, (64, 64))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    fs = make_frameset(image)
    fs_off = eqx.tree_at(
        lambda f: f.grid,
        fs,
        replace(fs.grid, center_y=20.0, center_x=40.0),
    )
    arm = MawetArm()
    a = arm.detect(fs, positions)
    b = arm.detect(fs_off, positions)
    assert not np.array_equal(
        np.asarray(a.statistic), np.asarray(b.statistic), equal_nan=True
    )


def _small_science():
    """CPU-float64 fixture with references passing within 2 px of the edge."""
    frames = jnp.asarray(np.random.default_rng(120).normal(size=(2, 33, 33)))
    return replace(
        make_frameset(frames[0], 2),
        frames=frames,
        fwhm_px=jnp.asarray(2.5),
    )


def _arm_for(kind, **kwargs):
    """Build either arm with a compact fixed template for shared contract tests."""
    from photomancy.postproc import ArrayTemplateProvider, MatchedFilterArm

    if kind == "mawet":
        return MawetArm(**kwargs)
    y, x = jnp.mgrid[:5, :5]
    provider = ArrayTemplateProvider(jnp.exp(-((y - 2) ** 2 + (x - 2) ** 2) / 2.0))
    return MatchedFilterArm(provider, **kwargs)


@pytest.mark.parametrize("value", [100.0, np.nan, np.inf])
def test_excluded_candidate_value_is_irrelevant(value):
    """Exclusion is the oracle; every result field must match bitwise, NaNs equal."""
    science = replace(
        make_frameset(jnp.asarray(np.random.default_rng(120).normal(size=(33, 33)))),
        fwhm_px=jnp.asarray(2.5),
    )
    valid = jnp.ones((33, 33)).at[16, 26].set(0.0)
    science = replace(science, valid=valid, frames=science.frames.at[:, 16, 26].set(0))
    positions = jnp.array([[16.0, 26.0], [6.0, 16.0]])
    arm = MawetArm()
    before = arm.detect(science, positions)
    after = arm.detect(
        replace(science, frames=science.frames.at[:, 16, 26].set(value)), positions
    )
    for field in ("statistic", "fpf", "dof", "positions_yx", "fwhm_px"):
        assert np.array_equal(
            np.asarray(getattr(before, field)),
            np.asarray(getattr(after, field)),
            equal_nan=True,
        ), field
    assert np.isnan(before.statistic[0])
    assert np.isnan(before.fpf[0])
    assert np.isfinite(before.statistic[1])


@pytest.mark.parametrize("kind", ["mawet", "template"])
def test_all_valid_mask_matches_no_mask_and_the_recorded_base(kind):
    """An all-ones mask equals no mask bitwise; both match the base within 1e-12.

    The reference ring at radius 14 on a 33-pixel detector crosses the kernel
    half-width from the edge, so an implementation that erodes a fully valid
    frame at the image edge drops references there (the first candidate's dof
    falls from 32 to 17 in both arms).

    Oracles: valid=None in the same run (exact algebra: an all-ones mask must
    take the same arithmetic path, so equality is bitwise on any platform), and
    statistics recorded from the pre-change implementation on an x86-64 Linux
    CPU in float64. Tolerance basis for the recorded values: floating-point;
    other CPU math libraries differ in the last few units in the last place
    (about 1e-16 relative), so rtol is 1e-12. The dof are integers and exact.
    """
    science = _small_science()
    positions = jnp.array([[16.0, 30.0], [9.25, 21.5]])
    recorded = {
        "mawet": (
            ("0x1.ba885dbba411ap-1", "0x1.448de7fdf5715p-1"),
            ("0x1.934be552481cap-3", "0x1.117881170fc1ap-2"),
        ),
        "template": (
            ("0x1.e05fd1b453f22p-2", "0x1.d6e68db3a9973p-3"),
            ("0x1.48ca322481de2p-2", "0x1.a437b0b3cdc6dp-2"),
        ),
    }
    arm = _arm_for(kind)
    unmasked = arm.detect(science, positions)
    masked = arm.detect(replace(science, valid=jnp.ones((33, 33))), positions)
    for field in ("statistic", "fpf", "dof"):
        assert np.array_equal(
            np.asarray(getattr(masked, field)), np.asarray(getattr(unmasked, field))
        ), field
    for field, literals in zip(("statistic", "fpf"), recorded[kind], strict=True):
        expected = np.array([float.fromhex(v) for v in literals])
        np.testing.assert_allclose(
            np.asarray(getattr(masked, field)), expected, rtol=1e-12, atol=0
        )
    np.testing.assert_array_equal(np.asarray(masked.dof), [32.0, 18.0])


@pytest.mark.parametrize("kind", ["mawet", "template"])
def test_arm_single_orientation_concrete_and_traced(kind):
    """Modulo-360 equality is the oracle; exact parity and boolean NaN gates apply."""
    science = _small_science()
    science = replace(
        science, frames=jnp.broadcast_to(science.frames[0], science.frames.shape)
    )
    positions = jnp.array([[16.0, 26.0], [7.0, 16.0]])
    arm = _arm_for(kind)
    with pytest.raises(ValueError, match=r"0.*90.*spread.*degrees"):
        arm.detect(replace(science, telescope_pa_deg=jnp.array([0.0, 90.0])), positions)
    expected = arm.detect(science, positions)
    wrapped = arm.detect(
        replace(science, telescope_pa_deg=jnp.array([0.0, 360.0])), positions
    )
    assert np.array_equal(np.asarray(expected.statistic), np.asarray(wrapped.statistic))

    run = eqx.filter_jit(
        lambda pa: arm.detect(
            eqx.tree_at(lambda s: s.telescope_pa_deg, science, pa), positions
        )
    )
    different = run(jnp.array([0.0, 90.0]))
    assert np.isnan(different.statistic).all()
    assert np.isnan(different.fpf).all()
    same = run(jnp.array([0.0, 0.0]))
    assert np.isfinite(same.statistic).all()
    assert np.isfinite(same.fpf).all()


@pytest.mark.parametrize("kind", ["mawet", "template"])
def test_arm_rejects_private_resolution(kind):
    """The public constructor contract forbids fwhm_px; TypeError is exact."""
    with pytest.raises(TypeError, match="fwhm_px"):
        _arm_for(kind, fwhm_px=2.5)


@pytest.mark.parametrize("kind", ["mawet", "template"])
def test_arm_resolution_comes_from_science(kind):
    """Science resolution and aperture scaling are algebraic; compare exactly."""
    from photomancy.postproc import make_aperture_kernel

    science = _small_science()
    positions = jnp.array([[16.0, 26.0]])
    wider_science = replace(science, fwhm_px=jnp.asarray(5.0))
    arm = _arm_for(kind)
    wide = _arm_for(kind, aperture_fwhm=2.0)
    assert arm.estimator(science).sampler.fwhm == science.fwhm_px
    assert (
        wide.estimator(science).sampler.fwhm == 2 * arm.estimator(science).sampler.fwhm
    )
    assert (
        arm.estimator(wider_science).sampler.fwhm
        == 2 * arm.estimator(science).sampler.fwhm
    )
    for configured in (arm, wide):
        assert configured.detect(science, positions).fwhm_px == science.fwhm_px
        assert (
            configured.detect(wider_science, positions).fwhm_px == wider_science.fwhm_px
        )
    field = type(arm).__dataclass_fields__["aperture_fwhm"]
    # MawetArm's kernel shape depends on the width, so it is static there.
    assert field.metadata.get("static", False) == (kind == "mawet")
    if kind == "mawet":
        expected = make_aperture_kernel(radius=2.5, soft=True, sharpness=10.0)
        assert np.array_equal(
            np.asarray(wide.estimator(science).filter.kernel), expected
        )


@pytest.mark.parametrize("where", ["near_center", "stencil_fringe"])
def test_reference_footprint_exclusion_matches_numpy_t(where):
    """NumPy box support and Bessel t formula are the oracle; roundoff is <1e-10.

    The excluded pixel is either one column from the reference centre (caught
    only by a footprint test, not by the raw centre pixel) or on the outer edge
    of the reference's bilinear stencil (caught only when every interpolation
    node's footprint is checked, not by the footprint at the nearest centre).
    """
    from scipy.ndimage import map_coordinates as numpy_sample
    from scipy.signal import convolve2d

    from photomancy.postproc import generate_aperture_coords, n_reference_apertures

    image = np.random.default_rng(90).normal(size=(101, 101))
    science = replace(make_frameset(jnp.asarray(image)), fwhm_px=jnp.asarray(5.0))
    arm = MawetArm(fast=True)
    estimator = arm.estimator(science)
    position = jnp.array([50.0, 80.0])
    n = int(n_reference_apertures(jnp.asarray(30.0), 5.0))
    ry, rx, _ = generate_aperture_coords((50.0, 50.0), 30.0, 0.0, n, fwhm=5.0)
    refs = np.stack([ry[:n], rx[:n]], axis=1)
    half = estimator.filter.kernel.shape[0] // 2
    lower = np.floor(refs).astype(int) - half
    upper = np.floor(refs).astype(int) + half + 1  # bilinear's second node
    # One column from the first reference's nearest center, away from the
    # candidate and all other reference footprints on this radius-30 ring.
    if where == "near_center":
        pixel = np.floor(refs[0] + 0.5).astype(int) + np.array([0, 1])
    else:
        pixel = np.array([43, 84])
    touched = np.all((lower <= pixel) & (pixel <= upper), axis=1)
    centers = np.floor(refs + 0.5).astype(int)
    assert touched.sum() == 1
    assert not np.any(np.all(centers == pixel, axis=1))
    valid = jnp.ones((101, 101)).at[tuple(pixel)].set(0.0)
    all_valid = arm.detect(science, position[None])
    result = arm.detect(replace(science, valid=valid), position[None])
    assert float(result.dof[0]) == float(all_valid.dof[0]) - 1
    clean = image.copy()
    clean[tuple(pixel)] = 0.0
    flux = convolve2d(clean, np.asarray(estimator.filter.kernel), mode="same")
    samples = numpy_sample(flux, refs[~touched].T, order=1)
    signal = numpy_sample(flux, np.asarray(position)[:, None], order=1)[0]
    expected = (signal - samples.mean()) / (
        samples.std(ddof=1) * np.sqrt(1 + 1 / len(samples))
    )
    np.testing.assert_allclose(result.statistic[0], expected, rtol=0, atol=1e-10)


def test_mawet_arm_jit_with_uncached_kernel():
    """Direct CPU float64 is the oracle; first-call JIT must give exact parity."""
    from photomancy.postproc import make_aperture_kernel

    science = replace(_small_science(), fwhm_px=jnp.asarray(2.75))
    arm = MawetArm()
    positions = jnp.array([[16.0, 26.0]])
    make_aperture_kernel.cache_clear()
    run = eqx.filter_jit(
        lambda frames: arm.detect(replace(science, frames=frames), positions)
    )
    jitted = run(science.frames)
    direct = arm.detect(science, positions)
    assert np.isfinite(jitted.statistic).all()
    assert np.array_equal(np.asarray(jitted.statistic), np.asarray(direct.statistic))
    assert np.array_equal(np.asarray(jitted.fpf), np.asarray(direct.fpf))


def test_mawet_arm_jits_as_an_argument():
    """Direct evaluation is the oracle; same arithmetic, so equality is exact."""
    science = _small_science()
    positions = jnp.array([[16.0, 26.0]])
    arm = MawetArm(aperture_fwhm=1.5)
    jitted = jax.jit(lambda a: a.detect(science, positions).statistic)(arm)
    np.testing.assert_array_equal(jitted, arm.detect(science, positions).statistic)


@pytest.mark.parametrize("kind", ["mawet", "template"])
def test_excluded_pixel_next_to_valid_candidate(kind):
    """A zero-filled image is the oracle for bitwise t, FPF and dof with infinity."""
    science = _small_science()
    # The hard aperture has a zero-weight corner here. The template's cubic
    # stencil ends one row before its excluded pixel.
    pixel = (14, 24) if kind == "mawet" else (12, 26)
    science = replace(science, valid=jnp.ones((33, 33)).at[pixel].set(0))
    positions = jnp.array([[16.0, 26.0]])
    arm = MawetArm(soft=False, fast=True) if kind == "mawet" else _arm_for(kind)
    results = [
        arm.detect(
            replace(science, frames=science.frames.at[:, *pixel].set(value)), positions
        )
        for value in (0.0, np.inf)
    ]
    for field in ("statistic", "fpf", "dof"):
        expected, actual = (np.asarray(getattr(r, field)) for r in results)
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(actual, expected)
