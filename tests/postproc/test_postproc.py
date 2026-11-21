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
    arm = MawetArm(fwhm_px=FWHM_PX)
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
    arm = MawetArm(fwhm_px=FWHM_PX)
    stats = arm.detect(fs, planet_pos[None, :])
    assert float(stats.statistic[0]) > 5.0
    assert float(stats.fpf[0]) < 1e-3
    assert float(stats.dof[0]) >= 2.0


def test_mawet_arm_empty_field_fpf_is_calibrated():
    """Test that FPF values over an empty noise field are ~Uniform(0, 1)."""
    key = jr.PRNGKey(2)
    image = jr.normal(key, (128, 128))
    fs = make_frameset(image)
    arm = MawetArm(fwhm_px=FWHM_PX)
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
    arm = MawetArm(fwhm_px=FWHM_PX)
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
    arm = MawetArm(fwhm_px=FWHM_PX)
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
    arm = MawetArm(fwhm_px=FWHM_PX)
    positions = jnp.array([[32.0, 48.0]])
    stats = arm.detect(fs, positions)
    expected = DetectionEstimator.mawet(FWHM_PX)(fs.coadd(), positions).statistic
    assert jnp.allclose(stats.statistic, expected, equal_nan=True)


def test_mawet_arm_detect_works_under_jit():
    """Test that arm.detect can be called from inside a jitted function."""
    key = jr.PRNGKey(7)
    image = jr.normal(key, (64, 64))
    fs = make_frameset(image)
    arm = MawetArm(fwhm_px=FWHM_PX)
    positions = jnp.array([[32.0, 48.0]])

    @jax.jit
    def run(f: FrameSet, p):
        return arm.detect(f, p).statistic

    assert jnp.allclose(run(fs, positions), arm.detect(fs, positions).statistic)


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
    arm = MawetArm(fwhm_px=FWHM_PX)
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
    arm = MawetArm(fwhm_px=FWHM_PX)
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
    arm = MawetArm(fwhm_px=FWHM_PX)
    a = arm.detect(fs, positions)
    b = arm.detect(fs_off, positions)
    assert not np.array_equal(
        np.asarray(a.statistic), np.asarray(b.statistic), equal_nan=True
    )
