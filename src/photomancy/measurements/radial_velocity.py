"""Stellar radial velocities: data container, marginalized kernel and term.

Times are days from the system's epoch; velocities are m/s, positive receding.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf, canonical
from hwoutils.model_paths import resolve
from hwoutils.precision import require_float64
from orbix.system import System

from photomancy.infer.fit import AbstractTerm
from photomancy.measurements._observables import _concrete
from photomancy.measurements._padding import _pad_1d, _valid_mask

__all__ = ["RadialVelocityData", "RadialVelocityTerm", "loglike_rv_marginalized"]


def _instrument_names(instruments):
    if isinstance(instruments, str):
        raise TypeError(
            "Received a single string for instruments. Pass (name,) for one instrument."
        )
    return tuple(instruments)


def _cast_references(references, dtype):
    original = np.asarray(references, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        stored = original.astype(dtype)
    if np.any(~np.isfinite(stored) | ((original != 0) & (stored == 0))):
        raise ValueError(f"Absolute RVs do not fit {dtype}; rescale the velocities")
    return stored


def _instrument_ids_and_mask(ids, valid, n_inst):
    message = f"inst_ids of valid rows must lie in [0, {n_inst})"
    if _concrete(ids, valid):
        kept = np.asarray(ids)[np.asarray(valid, dtype=bool)]
        if np.any((kept < 0) | (kept >= n_inst)):
            raise ValueError(f"{message}; got {np.unique(kept).tolist()}")
        with jax.ensure_compile_time_eval():
            return jnp.asarray(ids), jnp.asarray(valid)
    ids, valid = jnp.asarray(ids), jnp.asarray(valid)
    bad = jnp.any(valid & ((ids < 0) | (ids >= n_inst)))
    return eqx.error_if(ids, bad, message), valid


def _shift_host(values, origins, shifts=0.0):
    with np.errstate(over="ignore"):
        shifted = (values - origins) + shifts
    if np.any(np.isfinite(values) & np.isfinite(shifts) & ~np.isfinite(shifted)):
        raise ValueError("Finite RV offsets overflow; rescale the velocities")
    return shifted


class RadialVelocityData(eqx.Module):
    """Radial velocity observations from one or more named instruments.

    An instrument is one calibration: its constant offset is a single latent across
    the whole fit. Two sets of rows that share a name share the offset; a separate
    calibration of the same spectrograph takes a name of its own.

    Attributes:
        time_d: Observation epochs (days). Shape ``(N,)``.
        rv_offset: Measured velocities relative to each instrument reference (m/s).
        rv_err: RV measurement uncertainties (m/s). Shape ``(N,)``.
        inst_ids: Integer index into ``instruments`` per observation. Shape ``(N,)``.
            Every valid row lies in ``[0, n_inst)``; padded rows keep their sentinel.
        valid: Boolean validity mask. Shape ``(N,)``.
        instruments: Unique instrument names (static); ``n_inst`` is its length.
        rv_reference: Host reference velocities, one per instrument (m/s).
    """

    time_d: jnp.ndarray
    rv_offset: jnp.ndarray
    rv_err: jnp.ndarray
    inst_ids: jnp.ndarray
    valid: jnp.ndarray
    instruments: tuple[str, ...] = eqx.field(static=True)
    rv_reference: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        time_d,
        rv=None,
        rv_err=None,
        inst_ids=None,
        valid=None,
        instruments=None,
        *,
        rv_offset=None,
        rv_reference=None,
    ):
        """Anchor each instrument to its first finite, valid host observation.

        Args:
            time_d: Observation epochs in days.
            rv: Absolute velocities in m/s. Pass host observations before narrowing.
                Device arrays already have their working precision.
            rv_err: Measurement uncertainties in m/s.
            inst_ids: Instrument indices per observation.
            valid: Boolean validity mask per observation.
            instruments: Tuple of unique calibration names.
            rv_offset: Already conditioned velocities for rebuilding the data.
            rv_reference: Host velocity origin in m/s for each instrument when
                rebuilding from offsets. A new absolute ``rv`` replaces both.

        Raises:
            ValueError: If inputs for conditioning are missing, or an observation
                or offset cannot be represented in the storage precision.
        """
        self.time_d = jnp.asarray(time_d)
        self.rv_err = jnp.asarray(rv_err)
        self.inst_ids, self.valid = _instrument_ids_and_mask(
            inst_ids, valid, len(instruments)
        )
        self.instruments = instruments
        if rv is None:
            if rv_offset is None or rv_reference is None:
                raise ValueError(
                    "Supply absolute RVs or offsets and instrument references"
                )
            self.rv_reference = tuple(rv_reference)
            with jax.ensure_compile_time_eval():
                self.rv_offset = as_float_leaf(rv_offset)
        elif isinstance(rv, jax.Array) or not _concrete(rv):
            self.rv_reference = (0.0,) * len(instruments)
            self.rv_offset = as_float_leaf(rv)
        else:
            if not _concrete(valid, inst_ids):
                raise ValueError(
                    "Condition host RVs with concrete instrument IDs and validity"
                )
            values = np.asarray(rv, dtype=np.float64)
            offsets = values.copy()
            references = []
            ids, keep = np.asarray(inst_ids), np.asarray(valid, dtype=bool)
            for instrument in range(len(instruments)):
                rows = keep & (ids == instrument)
                finite = rows & np.isfinite(values)
                origin = float(values[finite][0]) if np.any(finite) else 0.0
                offsets[rows] = _shift_host(values[rows], origin)
                references.append(origin)
            self.rv_reference = tuple(references)
            with jax.ensure_compile_time_eval():
                self.rv_offset = as_float_leaf(offsets)

    @property
    def rv(self):
        """Absolute velocities in m/s including padding, at storage precision."""
        dtype = self.rv_offset.dtype
        if not any(self.rv_reference):
            return self.rv_offset
        origins = jnp.asarray(_cast_references(self.rv_reference, dtype))
        absolute = self.rv_offset + jnp.where(self.valid, origins[self.inst_ids], 0.0)
        return eqx.error_if(
            absolute,
            jnp.any(jnp.isfinite(self.rv_offset) & ~jnp.isfinite(absolute)),
            f"Absolute RVs do not fit {dtype}; rescale the velocities",
        )

    def __check_init__(self):
        """Require unique string names and, when concrete, in-range valid-row ids."""
        names = self.instruments
        if (
            not isinstance(names, tuple)
            or not names
            or not all(isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"instruments must be a nonempty tuple of unique strings: {names!r}"
            )
        if len(self.rv_reference) != len(names):
            raise ValueError("RV references must match the instrument names")
        dtype = self.rv_offset.dtype
        origins = _cast_references(self.rv_reference, dtype)
        if any(self.rv_reference) and _concrete(
            self.inst_ids, self.valid, self.rv_offset
        ):
            valid = np.asarray(self.valid)
            offsets = np.asarray(self.rv_offset)[valid]
            ids = np.asarray(self.inst_ids)[valid]
            with np.errstate(over="ignore"):
                absolute = offsets + origins[ids]
            if np.any(np.isfinite(offsets) & ~np.isfinite(absolute)):
                raise ValueError(
                    f"Absolute RVs do not fit {dtype}; rescale the velocities"
                )

    @property
    def n_inst(self) -> int:
        """Number of instruments, the length of ``instruments``."""
        return len(self.instruments)

    @classmethod
    def pad(cls, *, time_d, rv, rv_err, inst_ids, instruments, max_n):
        """Create a padded RadialVelocityData with ``valid`` mask.

        Args:
            time_d: Observation epochs (days). Shape ``(N,)``.
            rv: Measured radial velocities (m/s). Shape ``(N,)``.
            rv_err: Velocity uncertainties (m/s). Shape ``(N,)``. Padded rows hold 1.0.
            inst_ids: Index into ``instruments`` per row. Shape ``(N,)``.
            instruments: Unique instrument names.
            max_n: Row capacity of the result.

        Returns:
            The data padded to shape ``(max_n,)``, with ``valid`` true for the first N
            rows.
        """
        n = len(time_d)
        host = not isinstance(rv, jax.Array) and _concrete(rv)
        if host and not _concrete(inst_ids):
            raise ValueError("Condition host RVs with concrete instrument IDs")
        concrete_ids = _concrete(inst_ids)
        return cls(
            time_d=_pad_1d(time_d, max_n),
            rv=np.pad(np.asarray(rv), (0, max_n - n)) if host else _pad_1d(rv, max_n),
            rv_err=_pad_1d(rv_err, max_n, fill=1.0),
            inst_ids=(
                np.pad(np.asarray(inst_ids, dtype=int), (0, max_n - n))
                if concrete_ids
                else _pad_1d(inst_ids, max_n).astype(int)
            ),
            valid=np.arange(max_n) < n,
            instruments=_instrument_names(instruments),
        )

    @classmethod
    def zeros(cls, max_n, instruments):
        """Create an all-invalid placeholder (for model tracing).

        Args:
            max_n: Row capacity of the placeholder.
            instruments: Unique instrument names.

        Returns:
            Data of shape ``(max_n,)`` with every row invalid and unit errors.
        """
        return cls(
            time_d=jnp.zeros(max_n),
            rv=jnp.zeros(max_n),
            rv_err=jnp.ones(max_n),
            inst_ids=jnp.zeros(max_n, dtype=int),
            valid=jnp.zeros(max_n, dtype=bool),
            instruments=_instrument_names(instruments),
        )


def _centered_rv_residual(rv_obs, rv_model, inst_ids, mask, n_inst):
    # Difference each series before subtracting them. Their zero points can
    # differ by more than either series' resolved velocity contrasts.
    rows = jnp.arange(mask.size)
    first = jax.ops.segment_min(
        jnp.where(mask, rows, mask.size), inst_ids, num_segments=n_inst
    )

    def centered(series):
        safe = jnp.where(mask, series, 0.0)
        anchor = jnp.concatenate((safe, jnp.zeros(1, dtype=safe.dtype)))[first]
        return jnp.where(mask, safe - anchor[inst_ids], 0.0)

    return centered(rv_obs) - centered(rv_model)


def loglike_rv_marginalized(rv_obs, rv_model, rv_err, inst_ids, n_inst, jitters, valid):
    """RV log-likelihood with analytically marginalized per-instrument offsets.

    Because the instrument design matrix is an indicator matrix, the analytical
    marginalization of zero-points gamma_j collapses to O(N) segment sums. This
    avoids sampling ``n_inst`` nuisance parameters. Each populated instrument
    removes one Gaussian dimension, so the normalization uses ``N_valid - J``.

    The result is a normalized density of the data modulo the offsets: over the
    valid velocities of each instrument it integrates to 1 on the hyperplane
    orthogonal to a common shift (Lebesgue measure in m/s there). The flat
    integral over an offset gamma_j shifts the ``N_j`` velocities of instrument
    ``j`` along ``(1, ..., 1)``, whose unit vector is that direction over
    ``sqrt(N_j)``, so the marginal carries ``1 / sqrt(N_j)`` per instrument and the
    returned value adds ``0.5 * sum_j log(N_j)`` over the populated instruments.

    Args:
        rv_obs: Observed RVs (m/s). Shape ``(N,)``.
        rv_model: Predicted RVs converted from forward model (m/s). Shape ``(N,)``.
        rv_err: RV measurement uncertainties (m/s). Shape ``(N,)``.
        inst_ids: Integer instrument index per obs, ``0..n_inst-1``.
            Shape ``(N,)``.
        n_inst: Number of instruments. Integer.
        jitters: Per-instrument jitter terms (m/s). Shape ``(n_inst,)``.
        valid: Boolean validity mask. Shape ``(N,)``.

    Returns:
        Scalar log-likelihood value.

    Raises:
        PrecisionError: If float64 is unavailable.
    """
    require_float64("Radial velocity log likelihood")
    # Mask invalid entries -- use jnp.where for gradient safety
    mask = valid  # boolean

    # Total variance: measurement + jitter
    inst_ids = jnp.where(mask, inst_ids, 0)
    err = jnp.where(mask, jnp.asarray(rv_err, jnp.float64), 1.0)
    jitter = jnp.where(mask, jnp.asarray(jitters, jnp.float64)[inst_ids], 0.0)
    var = err**2 + jitter**2
    w = jnp.where(mask, 1.0 / var, 0.0)
    rv_obs = jnp.asarray(rv_obs, jnp.float64)
    rv_model = jnp.asarray(rv_model, jnp.float64)
    shifted = _centered_rv_residual(rv_obs, rv_model, inst_ids, mask, n_inst)
    A = jax.ops.segment_sum(w, inst_ids, num_segments=n_inst)
    B = jax.ops.segment_sum(w * shifted, inst_ids, num_segments=n_inst)
    centre = B / jnp.where(A > 0, A, 1.0)
    spread = jnp.where(mask, shifted - centre[inst_ids], 0.0)
    C_centred = jax.ops.segment_sum(w * spread**2, inst_ids, num_segments=n_inst)

    counts = jax.ops.segment_sum(mask.astype(w.dtype), inst_ids, num_segments=n_inst)

    # Marginalized log-likelihood -- only count valid observations
    N_valid = jnp.sum(mask)
    J = jnp.sum(A > 0)
    A_safe = jnp.where(A > 0, A, 1.0)
    return -0.5 * (
        jnp.sum(C_centred)
        + jnp.sum(jnp.log(A_safe))
        + jnp.sum(jnp.where(mask, jnp.log(var), 0.0))
        + (N_valid - J) * jnp.log(2.0 * jnp.pi)
        - jnp.sum(jnp.log(jnp.where(counts > 0, counts, 1.0)))
    )


class RadialVelocityTerm(AbstractTerm):
    """Score stellar RVs in m/s, marginalizing each instrument's constant offset.

    ``system_path`` names the model path of a system with
    ``star_radial_velocity(time_d)`` in m/s, positive receding; the empty path is the
    model itself. Jitter is in m/s with shape ``(data.n_inst,)``: a fixed array in
    ``jitters``, a model leaf read at ``jitter_path`` (so a fit can free it), or zeros
    when neither is given. Predictions have shape ``(T,)`` with invalid rows zeroed.
    Data ``time_d`` is in days from the system's epoch.

    The offset of an instrument is one latent across the whole fit: the term names it
    in ``marginalizes`` as ``"rv_offset:<system_path>:<instrument>"``, so ``Fit``
    refuses a second term that integrates the same offset. Rows of an instrument that
    arrive later join the term through :meth:`merged` (``photomancy.infer.update``
    does it); a separate calibration of the same spectrograph takes its own name.

    Attributes:
        data: Radial velocity observations.
        system_path: Model path of the system whose ``star_radial_velocity(time_d)`` is
            read.
        jitters: Fixed per-instrument jitter (m/s), shape ``(data.n_inst,)``, or None.
        jitter_path: Model path of the jitter leaf (m/s), or None.
        forward_fields: Empty tuple because the term differentiates no stored fields.
        reads: Model paths a held build cached. Empty: the term reads the model
            directly.
    """

    data: RadialVelocityData
    system_path: str = eqx.field(static=True, default="")
    jitters: jnp.ndarray | None = None
    jitter_path: str | None = eqx.field(static=True, default=None)
    forward_fields = ()
    reads: ClassVar[tuple[str, ...]] = ()

    def __check_init__(self):
        """Require one jitter per instrument, from at most one source."""
        if self.jitters is not None and self.jitter_path is not None:
            raise ValueError("Choose fixed jitters or a jitter_path, not both")
        if self.jitters is not None and jnp.shape(self.jitters) != (self.data.n_inst,):
            raise ValueError("jitters must have shape (data.n_inst,)")

    @property
    def marginalizes(self) -> tuple[str, ...]:
        """One latent per instrument: ``"rv_offset:<system_path>:<instrument>"``."""
        return tuple(
            f"rv_offset:{self.system_path}:{name}" for name in self.data.instruments
        )

    def merged(self, other) -> Self:
        """Return the term scoring the rows of both under one offset per instrument.

        The instruments of ``other`` join by name: a name already held shares its
        offset, a new one is appended after the held names. Valid rows of ``self``
        come first, then those of ``other`` with ``inst_ids`` remapped. The row
        capacity of ``self`` is kept when the rows fit and grows to their count
        otherwise. Runs on the host, on concrete data.

        Args:
            other: A ``RadialVelocityTerm`` of the same ``system_path`` and jitter
                source.

        Returns:
            The merged term. Its score equals the one term built from all rows.

        Raises:
            TypeError: When ``other`` is not a ``RadialVelocityTerm``.
            ValueError: When a term is traced (both must be concrete); when the system
                path differs; when one term has a
                ``jitter_path`` and the other a different one or fixed jitters (a
                ``jitter_path`` also needs identical instruments, since its leaf has
                one entry per instrument); or when a shared name has two fixed jitters.
        """
        if not _concrete(self, other):
            raise ValueError(
                "merged reads the rows on the host, so both terms must be concrete; "
                "inside jit, replace the held term yourself with update(..., index=i)"
            )
        if not isinstance(other, RadialVelocityTerm):
            raise TypeError(
                f"RadialVelocityTerm.merged needs a RadialVelocityTerm, got "
                f"{type(other).__name__}"
            )
        if self.system_path != other.system_path:
            raise ValueError(
                f"cannot merge RV terms of system paths {self.system_path!r} and "
                f"{other.system_path!r}"
            )
        if self.jitter_path != other.jitter_path or (
            (self.jitters is None) != (other.jitters is None)
        ):
            raise ValueError("cannot merge RV terms with different jitter sources")
        mine, theirs = self.data.instruments, other.data.instruments
        if self.jitter_path is not None and mine != theirs:
            raise ValueError(
                "RV terms read at one jitter_path need identical instruments, got "
                f"{mine} and {theirs}"
            )
        instruments = mine + tuple(n for n in theirs if n not in mine)
        jitters = None
        if self.jitters is not None:
            fixed = dict(zip(mine, np.asarray(self.jitters).tolist(), strict=True))
            for name, value in zip(
                theirs, np.asarray(other.jitters).tolist(), strict=True
            ):
                if fixed.setdefault(name, value) != value:
                    raise ValueError(
                        f"instrument {name!r} has fixed jitter {fixed[name]} in one "
                        f"term and {value} in the other"
                    )
            jitters = jnp.asarray([fixed[n] for n in instruments])
        remap = np.array([instruments.index(n) for n in theirs], dtype=int)
        a, b = self.data, other.data
        keep_a, keep_b = np.asarray(a.valid), np.asarray(b.valid)
        ids_a = np.asarray(a.inst_ids)[keep_a]
        ids_b = np.asarray(b.inst_ids)[keep_b]
        offsets_a = np.asarray(a.rv_offset, dtype=np.float64)[keep_a]
        offsets_b = np.asarray(b.rv_offset, dtype=np.float64)[keep_b]
        origins = {}
        for data, ids, values in ((a, ids_a, offsets_a), (b, ids_b, offsets_b)):
            for i, name in enumerate(data.instruments):
                finite = (ids == i) & np.isfinite(values)
                if name not in origins and np.any(finite):
                    # Read one absolute observation to choose the common origin.
                    # All delivered contrasts below stay in their offset form.
                    origins[name] = float(
                        _shift_host(values[finite][0], -data.rv_reference[i])
                    )
        references = tuple(origins.get(name, 0.0) for name in instruments)

        def offsets(data, ids, values):
            # Shift each old origin before adding offsets so neither batch is
            # reconstructed at the scale of the absolute velocities.
            old = np.asarray(data.rv_reference)[ids]
            new = np.asarray([origins.get(name, 0.0) for name in data.instruments])[ids]
            return _shift_host(old, new, values)

        rv_offset = np.concatenate(
            [offsets(a, ids_a, offsets_a), offsets(b, ids_b, offsets_b)]
        )
        n = rv_offset.size
        capacity = max(a.valid.shape[0], n)
        data = RadialVelocityData(
            time_d=_pad_1d(
                np.concatenate(
                    [np.asarray(a.time_d)[keep_a], np.asarray(b.time_d)[keep_b]]
                ),
                capacity,
            ),
            rv_offset=canonical(
                np.pad(rv_offset, (0, capacity - n)),
                precision=np.result_type(a.rv_offset.dtype, b.rv_offset.dtype),
            ),
            rv_reference=references,
            rv_err=_pad_1d(
                np.concatenate(
                    [np.asarray(a.rv_err)[keep_a], np.asarray(b.rv_err)[keep_b]]
                ),
                capacity,
                fill=1.0,
            ),
            inst_ids=_pad_1d(np.concatenate([ids_a, remap[ids_b]]), capacity).astype(
                int
            ),
            valid=_valid_mask(n, capacity),
            instruments=instruments,
        )
        return type(self)(
            data=data,
            system_path=self.system_path,
            jitters=jitters,
            jitter_path=self.jitter_path,
        )

    def _jitters(self, model):
        if self.jitter_path is not None:
            jitters = resolve(model, self.jitter_path)
        elif self.jitters is not None:
            jitters = self.jitters
        else:
            jitters = jnp.zeros(self.data.n_inst)
        if jnp.shape(jitters) != (self.data.n_inst,):
            raise ValueError("jitters must have shape (data.n_inst,)")
        return jitters

    def __call__(self, model):
        """Score RVs with the jitter of ``jitters`` or ``jitter_path`` (m/s).

        Args:
            model: The model holding the system at ``system_path`` and any
                ``jitter_path``.

        Returns:
            Scalar log-likelihood with the instrument offsets marginalized.

        Raises:
            PrecisionError: If float64 is unavailable.
        """
        require_float64("Radial velocity log likelihood")
        data = self.data
        return loglike_rv_marginalized(
            data.rv_offset,
            self.predict(model),
            data.rv_err,
            data.inst_ids,
            data.n_inst,
            self._jitters(model),
            data.valid,
        )

    def predict(self, model):
        """Return the stellar radial velocity at each epoch (m/s), invalid rows zero.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            Stellar radial velocity (m/s) at each epoch, shape ``(T,)``.
        """
        valid = self.data.valid
        time_d = jnp.where(valid, self.data.time_d, 0.0)
        system = resolve(model, self.system_path)
        if isinstance(system, System):
            prediction = system.solve(time_d).radial_velocity_m_per_s(0, "barycenter")
        else:
            prediction = system.star_radial_velocity(time_d=time_d)
        return jnp.where(valid, prediction, 0.0)

    def draw(self, key, model) -> Self:
        """Draw Gaussian RV noise with zero instrument offsets.

        The score marginalizes offsets against a flat measure and is therefore
        improper on raw RVs. Zero offsets choose a representative; its residual
        contrasts have exactly the marginalized Gaussian law. Jitter is added
        in quadrature, as in the score.

        Args:
            key: JAX random key.
            model: The model holding the system at ``system_path`` and any
                ``jitter_path``.

        Returns:
            The term with velocities drawn about the prediction. Invalid rows are
            unchanged.
        """
        data = self.data
        valid = data.valid
        ids = jnp.where(valid, data.inst_ids, 0)
        jitter = self._jitters(model)
        sigma = jnp.sqrt(jnp.where(valid, data.rv_err, 1.0) ** 2 + jitter[ids] ** 2)
        draw = self.predict(model) + sigma * jax.random.normal(
            key, data.rv_offset.shape, dtype=sigma.dtype
        )
        sampled = RadialVelocityData(
            data.time_d,
            jnp.where(valid, draw, data.rv_offset),
            data.rv_err,
            data.inst_ids,
            valid,
            data.instruments,
        )
        return eqx.tree_at(lambda t: t.data, self, sampled)
