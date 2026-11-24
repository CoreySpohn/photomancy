"""Tests for the post-processing seam contract types (FrameSet, DetectionStats)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hwoutils.constants import J2000_JD
from hwoutils.epoch import Epoch
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band

from photomancy.postproc.contracts import DetectionStats, FrameSet


def make_frameset(n_frames=3, ny=8, nx=8, **overrides):
    """Build a FrameSet with valid defaults, applying any overrides."""
    kwargs = dict(
        frames=jnp.ones((n_frames, ny, nx)),
        time_d=jnp.arange(n_frames, dtype=float),
        frame_time_s=jnp.full((n_frames,), 100.0),
        telescope_pa_deg=jnp.zeros((n_frames,)),
        fwhm_px=3.0,
        band=Band(550.0 - 110.0 / 2, 550.0 + 110.0 / 2),
        epoch=Epoch(J2000_JD),
        grid=PixelGrid.regular((ny, nx), 0.0218, "arcsec"),
    )
    kwargs.update(overrides)
    return FrameSet(**kwargs)


def test_frameset_constructs_with_required_fields():
    """Test construction with required fields only; optionals default to None."""
    fs = make_frameset()
    assert fs.frames.shape == (3, 8, 8)
    assert fs.noise_variance is None
    assert fs.valid is None
    assert (fs.grid.center_y, fs.grid.center_x) == (3.5, 3.5)
    assert fs.tau_c_s is None


def test_frameset_converts_inputs_to_jax_arrays():
    """Test that scalar and array inputs are converted to JAX arrays."""
    fs = make_frameset(fwhm_px=3.0)
    assert isinstance(fs.fwhm_px, jnp.ndarray)
    assert isinstance(fs.frames, jnp.ndarray)


def test_frameset_rejects_wrong_frames_ndim():
    """Test that frames without a (n_frames, ny, nx) shape raise ValueError."""
    with pytest.raises(ValueError, match="frames"):
        make_frameset(frames=jnp.ones((8, 8)))


def test_frameset_rejects_mismatched_per_frame_metadata():
    """Test that per-frame metadata of the wrong length raises ValueError."""
    with pytest.raises(ValueError, match="time_d"):
        make_frameset(time_d=jnp.arange(2, dtype=float))


def test_frameset_rejects_mismatched_noise_variance_shape():
    """Test that noise_variance not matching frames shape raises ValueError."""
    with pytest.raises(ValueError, match="noise_variance"):
        make_frameset(noise_variance=jnp.ones((3, 4, 4)))


def test_frameset_rate_frames_divides_by_exposure():
    """Test that rate_frames divides counts by per-frame exposure time."""
    fs = make_frameset()
    assert fs.rate_frames.shape == (3, 8, 8)
    assert float(fs.rate_frames[0, 0, 0]) == pytest.approx(0.01)


def test_frameset_coadd_is_total_electrons_over_total_time():
    """Test that coadd returns total electrons over total exposure time."""
    frames = jnp.stack([jnp.full((4, 4), 10.0), jnp.full((4, 4), 30.0)])
    fs = make_frameset(
        n_frames=2,
        ny=4,
        nx=4,
        frames=frames,
        time_d=jnp.array([0.0, 1.0]),
        frame_time_s=jnp.array([100.0, 300.0]),
        telescope_pa_deg=jnp.zeros((2,)),
    )
    coadd = fs.coadd()
    assert coadd.shape == (4, 4)
    assert float(coadd[0, 0]) == pytest.approx(40.0 / 400.0)


def test_frameset_is_a_pytree():
    """Test that FrameSet flattens and unflattens as a JAX pytree."""
    fs = make_frameset()
    leaves, treedef = jax.tree_util.tree_flatten(fs)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, FrameSet)
    assert jnp.array_equal(rebuilt.frames, fs.frames)


def test_frameset_passes_through_jit():
    """Test that a FrameSet can be passed through a jitted function."""
    fs = make_frameset()

    @jax.jit
    def total_electrons(f: FrameSet):
        return jnp.sum(f.frames)

    assert float(total_electrons(fs)) == pytest.approx(3 * 8 * 8)


def test_frameset_equinox_partition_roundtrip():
    """Test that eqx.partition and eqx.combine round-trip a FrameSet."""
    fs = make_frameset()
    params, static = eqx.partition(fs, eqx.is_array)
    rebuilt = eqx.combine(params, static)
    assert jnp.array_equal(rebuilt.frames, fs.frames)


def make_stats(n=4, **overrides):
    """Build a DetectionStats with valid defaults, applying any overrides."""
    kwargs = dict(
        positions_yx=jnp.zeros((n, 2)),
        statistic=jnp.ones((n,)),
        fpf=jnp.full((n,), 0.05),
        dof=jnp.full((n,), 10.0),
        fwhm_px=3.0,
        statistic_kind="mawet_t",
    )
    kwargs.update(overrides)
    return DetectionStats(**kwargs)


def test_detection_stats_constructs():
    """Test construction with valid fields and correct shape."""
    stats = make_stats()
    assert stats.statistic.shape == (4,)
    assert stats.statistic_kind == "mawet_t"


def test_detection_stats_rejects_bad_positions_shape():
    """Test that positions_yx not (n, 2) raises ValueError."""
    with pytest.raises(ValueError, match="positions_yx"):
        make_stats(positions_yx=jnp.zeros((4, 3)))


def test_detection_stats_rejects_mismatched_lengths():
    """Test that per-candidate arrays of wrong length raise ValueError."""
    with pytest.raises(ValueError, match="statistic"):
        make_stats(statistic=jnp.ones((3,)))


def test_detection_stats_statistic_kind_is_static():
    """Test that statistic_kind does not appear in JAX tree leaves."""
    stats = make_stats()
    leaves = jax.tree_util.tree_leaves(stats)
    assert all(not isinstance(leaf, str) for leaf in leaves)


def test_detection_stats_passes_through_jit():
    """Test that DetectionStats can be passed through a jitted function."""
    stats = make_stats()

    @jax.jit
    def min_fpf(s: DetectionStats):
        return jnp.min(s.fpf)

    assert float(min_fpf(stats)) == pytest.approx(0.05)


def _frameset_kwargs(n_frames=2, ny=8, nx=8):
    """Minimal valid FrameSet constructor arguments."""
    return dict(
        frames=jnp.ones((n_frames, ny, nx)),
        time_d=jnp.arange(n_frames, dtype=float),
        frame_time_s=jnp.ones(n_frames),
        telescope_pa_deg=jnp.zeros(n_frames),
        fwhm_px=4.0,
        band=Band(550.0 - 100.0 / 2, 550.0 + 100.0 / 2),
        epoch=Epoch(J2000_JD),
        grid=PixelGrid.regular((ny, nx), 0.020, "arcsec"),
    )


class TestOptionalFieldRejection:
    """__check_init__ rejects malformed optional fields (P1 carry-over)."""

    def test_bad_validity_shape_rejected(self):
        """A valid mask not matching (ny, nx) raises ValueError."""
        kwargs = _frameset_kwargs()
        with pytest.raises(ValueError, match="valid"):
            FrameSet(valid=jnp.ones((3, 3)), **kwargs)

    def test_bad_grid_shape_rejected(self):
        """A grid not matching the detector frames raises ValueError."""
        kwargs = _frameset_kwargs()
        kwargs["grid"] = PixelGrid.regular((3, 3), 0.020, "arcsec")
        with pytest.raises(ValueError, match=r"grid\.shape"):
            FrameSet(**kwargs)


def test_band_and_epoch_are_host_metadata_with_precise_offsets():
    """Exact algebra: two-part dates preserve 1/65536 day in float32 offsets."""
    from types import SimpleNamespace

    epoch = Epoch(J2000_JD)
    date = SimpleNamespace(jd1=J2000_JD, jd2=1 / 65536, scale="tdb")
    fs = make_frameset(
        n_frames=1, epoch=epoch, time_d=epoch.days(date)[None].astype("float32")
    )
    leaves = jax.tree_util.tree_leaves(fs)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert fs.band == Band(495.0, 605.0)
    assert fs.epoch is epoch
    assert float(jax.jit(lambda f: f.time_d[0])(fs)) == 1 / 65536


def test_frameset_single_orientation():
    """Wrapped differences define equality; the contract tolerance is 1e-6 deg."""
    assert bool(make_frameset(n_frames=1).single_orientation())
    for angles in ([10.0, 370.0], [359.9999999, 0.0]):
        science = make_frameset(n_frames=2, telescope_pa_deg=jnp.asarray(angles))
        assert bool(science.single_orientation())
    with pytest.raises(ValueError, match=r"0.*0.001.*spread.*degrees"):
        make_frameset(
            n_frames=2, telescope_pa_deg=jnp.array([0.0, 1e-3])
        ).single_orientation()
