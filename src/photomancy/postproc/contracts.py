"""Data contracts for photomancy post-processing.

These types are the stable interchange surface between photomancy.postproc and any
caller that renders frames (image simulators, mission simulators, real
pipelines). They are plain Equinox PyTrees over JAX arrays: callers construct
them from raw arrays; postproc algorithms consume and produce them.

Conventions:
    - Frames are detector electrons per frame (counts), shape
      (n_frames, ny, nx). Rate views divide by per-frame exposure time.
    - Field names carry unit suffixes (_d, _s, _deg, _nm, _px, _mas).
    - Positions are (y, x) pixel coordinates ("_yx"), matching the image
      indexing convention used across the library.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.epoch import Epoch
from hwoutils.leaves import HostGrid, as_float_leaf
from hwoutils.numerics import COARSEST
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band


def _asarray(value):
    return jnp.asarray(value, dtype=jnp.result_type(float))


def _asarray_or_none(value):
    return None if value is None else _asarray(value)


# A host roll series is static metadata: keep its original values through jit.
# A JAX series stays a dynamic leaf so callers can scan or vmap over rolls.
class _HostRolls(np.ndarray):
    """Read-only roll angles with float64 host precision."""


jax.tree_util.register_pytree_node(
    _HostRolls,
    lambda angles: ((), HostGrid(angles)),
    lambda grid, _: grid.values.view(_HostRolls),
)


def _roll_angles(value):
    if any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(value)):
        return as_float_leaf(jnp.asarray(value))
    return HostGrid(value).values.view(_HostRolls)


_ROLL_TOLERANCE_DEG = 1e-6  # tol: physical, degrees of roll for coaddition.


class FrameSet(eqx.Module):
    """A time series of detector frames plus the metadata detection needs.

    Attributes:
        frames: (n_frames, ny, nx) detector electrons per frame.
        time_d: (n_frames,) frame start offsets in days from epoch.
        frame_time_s: (n_frames,) per-frame exposure times, seconds.
        telescope_pa_deg: (n_frames,) telescope position angles in degrees.
            Host input is immutable float64 metadata. JAX input stays dynamic.
        fwhm_px: Scalar resolution element (lambda/D) in pixels.
        band: Static wavelength interval in nanometers.
        epoch: Static host epoch defining the zero of time_d.
        grid: Shared PixelGrid carrying detector sampling and star center.
        noise_variance: Optional (n_frames, ny, nx) per-pixel total noise
            variance in electrons^2 (e.g. a detector model's variance budget);
            reserved for inverse-variance weighting (first consumer: the
            PSF-template matched filter).
        valid: Optional (ny, nx) mask (1 = usable pixel, 0 = excluded).
        tau_c_s: Optional speckle decorrelation time, seconds; scalar or
            (ny, nx) map. Reserved for correlated-noise-aware calibration.
    """

    frames: jnp.ndarray = eqx.field(converter=_asarray)
    time_d: jnp.ndarray = eqx.field(converter=_asarray)
    frame_time_s: jnp.ndarray = eqx.field(converter=_asarray)
    telescope_pa_deg: jax.Array | np.ndarray = eqx.field(converter=_roll_angles)
    fwhm_px: jnp.ndarray = eqx.field(converter=_asarray)
    band: Band = eqx.field(static=True)
    epoch: Epoch = eqx.field(static=True)
    grid: PixelGrid
    noise_variance: jnp.ndarray | None = eqx.field(
        default=None, converter=_asarray_or_none
    )
    valid: jnp.ndarray | None = eqx.field(default=None, converter=_asarray_or_none)
    tau_c_s: jnp.ndarray | None = eqx.field(default=None, converter=_asarray_or_none)

    def __check_init__(self):
        """Validate that field shapes are mutually consistent."""
        if not isinstance(self.band, Band):
            raise TypeError("band must be a Band")
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch")
        if self.frames.ndim != 3:
            raise ValueError(
                f"frames must be (n_frames, ny, nx); got shape {self.frames.shape}"
            )
        n_frames, ny, nx = self.frames.shape
        for name in ("time_d", "frame_time_s", "telescope_pa_deg"):
            value = getattr(self, name)
            if value.shape != (n_frames,):
                raise ValueError(
                    f"{name} must have shape ({n_frames},); got {value.shape}"
                )
        if self.noise_variance is not None and self.noise_variance.shape != (
            n_frames,
            ny,
            nx,
        ):
            raise ValueError(
                "noise_variance must match frames shape "
                f"({n_frames}, {ny}, {nx}); got {self.noise_variance.shape}"
            )
        if self.valid is not None and self.valid.shape != (ny, nx):
            raise ValueError(
                f"valid must have shape ({ny}, {nx}); got {self.valid.shape}"
            )
        if self.grid.shape != (ny, nx):
            raise ValueError("grid.shape must match frames spatial shape")

    @property
    def rate_frames(self) -> jnp.ndarray:
        """Per-frame count rates, electrons per second, (n_frames, ny, nx)."""
        return self.frames / self.frame_time_s[:, None, None]

    def single_orientation(self) -> jnp.ndarray:
        """Check for a common telescope orientation modulo 360 degrees.

        The permitted span is 1e-6 degrees. Both host and dynamic roll series
        use the float32 decision grid, with signed offsets about zero or the
        half turn to preserve separations across the wrap. Host metadata keeps
        its original values.

        Returns:
            Scalar bool indicating whether all frames share one orientation.

        Raises:
            ValueError: Concrete angles span more than 1e-6 degrees modulo
                360; the message includes the angles and their spread.
        """
        angles = self.telescope_pa_deg
        host = isinstance(angles, np.ndarray)
        xp = np if host else jnp
        with jax.ensure_compile_time_eval():
            values = xp.asarray(angles, dtype=COARSEST)
            reduced = xp.fmod(values, 360.0)
            signed = xp.where(
                reduced > 180,
                reduced - 360,
                xp.where(reduced < -180, reduced + 360, reduced),
            )
            # Near +/-180, subtract each endpoint before taking differences.
            half_turn = xp.where(reduced < 0, reduced + 180, reduced - 180)
            offsets = xp.where(xp.abs(signed[0]) <= 90, signed, half_turn)
            spread = xp.max(offsets) - xp.min(offsets)
            same = spread <= COARSEST(_ROLL_TOLERANCE_DEG)
        if not isinstance(angles, jax.core.Tracer) and not bool(same):
            raise ValueError(
                f"telescope_pa_deg angles {np.asarray(angles).tolist()} must share "
                f"one orientation; spread {float(spread)} degrees"
            )
        return jnp.asarray(same)

    def coadd(self) -> jnp.ndarray:
        """Detector-frame sum: total electrons / total time, (ny, nx) e-/s.

        Pixels are added at equal detector indices with no derotation. Frames
        must share one orientation for a sky-fixed source to add coherently.

        Returns:
            Coadded count rate in electrons per second, shape (ny, nx).
        """
        return jnp.sum(self.frames, axis=0) / jnp.sum(self.frame_time_s)


class DetectionStats(eqx.Module):
    """Per-candidate detection statistics produced by a post-processing arm.

    Every arm reports both the raw test statistic and its false positive
    fraction (FPF) under that arm's own null distribution, so downstream
    consumers can threshold on probability rather than raw sigma (a fixed
    sigma threshold is not a fixed false-alarm probability; Mawet et al.
    2014).

    Attributes:
        positions_yx: (n, 2) candidate positions, (y, x) pixels.
        statistic: (n,) test statistic values (NaN where unmeasurable).
        fpf: (n,) false positive fraction of each statistic under the arm's
            null distribution (NaN where the statistic is NaN).
        dof: (n,) effective degrees of freedom backing each statistic's null
            (t-tests: number of reference samples minus one).
        fwhm_px: Scalar resolution element of the science frames, pixels.
        statistic_kind: Static label of the statistic/null pairing, e.g.
            "mawet_t".
    """

    positions_yx: jnp.ndarray = eqx.field(converter=_asarray)
    statistic: jnp.ndarray = eqx.field(converter=_asarray)
    fpf: jnp.ndarray = eqx.field(converter=_asarray)
    dof: jnp.ndarray = eqx.field(converter=_asarray)
    fwhm_px: jnp.ndarray = eqx.field(converter=_asarray)
    statistic_kind: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate that field shapes are mutually consistent."""
        if self.positions_yx.ndim != 2 or self.positions_yx.shape[-1] != 2:
            raise ValueError(
                f"positions_yx must be (n, 2); got {self.positions_yx.shape}"
            )
        n = self.positions_yx.shape[0]
        for name in ("statistic", "fpf", "dof"):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},); got {value.shape}")
