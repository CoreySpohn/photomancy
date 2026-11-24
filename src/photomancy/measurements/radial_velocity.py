"""Stellar radial velocities: data container, marginalized kernel and term.

Times are days from the system's epoch; velocities are m/s, positive receding.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.model_paths import resolve
from orbix.system import System

from photomancy.infer.fit import AbstractTerm
from photomancy.measurements._padding import _pad_1d, _valid_mask

__all__ = ["RadialVelocityData", "RadialVelocityTerm", "loglike_rv_marginalized"]


class RadialVelocityData(eqx.Module):
    """Radial velocity observations from one or more named instruments.

    An instrument is one calibration: its constant offset is a single latent across
    the whole fit. Two sets of rows that share a name share the offset; a separate
    calibration of the same spectrograph takes a name of its own.

    Attributes:
        time_d: Observation epochs (days). Shape ``(N,)``.
        rv: Measured radial velocities (m/s). Shape ``(N,)``.
        rv_err: RV measurement uncertainties (m/s). Shape ``(N,)``.
        inst_ids: Integer index into ``instruments`` per observation. Shape ``(N,)``.
            Every valid row lies in ``[0, n_inst)``; padded rows keep their sentinel.
        valid: Boolean validity mask. Shape ``(N,)``.
        instruments: Unique instrument names (static); ``n_inst`` is its length.
    """

    time_d: jnp.ndarray
    rv: jnp.ndarray
    rv_err: jnp.ndarray
    inst_ids: jnp.ndarray
    valid: jnp.ndarray
    instruments: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        """Require unique names and, on concrete values, in-range ids on valid rows."""
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError(f"instrument names must be unique: {self.instruments}")
        leaves = (self.inst_ids, self.valid)
        if not any(isinstance(x, jax.core.Tracer) for x in leaves):
            ids = np.asarray(self.inst_ids)[np.asarray(self.valid)]
            if np.any((ids < 0) | (ids >= self.n_inst)):
                raise ValueError(
                    f"inst_ids of valid rows must lie in [0, {self.n_inst}) for "
                    f"instruments {self.instruments}; got {np.unique(ids).tolist()}"
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
        n = jnp.asarray(time_d).shape[0]
        return cls(
            time_d=_pad_1d(time_d, max_n),
            rv=_pad_1d(rv, max_n),
            rv_err=_pad_1d(rv_err, max_n, fill=1.0),
            inst_ids=_pad_1d(inst_ids, max_n).astype(int),
            valid=_valid_mask(n, max_n),
            instruments=tuple(instruments),
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
            instruments=tuple(instruments),
        )


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
    """
    # Mask invalid entries -- use jnp.where for gradient safety
    mask = valid  # boolean

    # Total variance: measurement + jitter
    inst_ids = jnp.where(mask, inst_ids, 0)
    err = jnp.where(mask, rv_err, 1.0)
    jitter = jnp.where(mask, jitters[inst_ids], 0.0)
    var = err**2 + jitter**2
    w = jnp.where(mask, 1.0 / var, 0.0)
    r = jnp.where(mask, rv_obs, 0.0) - jnp.where(mask, rv_model, 0.0)

    # Segment sums collapse the matrix algebra to O(N)
    A = jax.ops.segment_sum(w, inst_ids, num_segments=n_inst)
    B = jax.ops.segment_sum(w * r, inst_ids, num_segments=n_inst)
    C = jax.ops.segment_sum(w * r**2, inst_ids, num_segments=n_inst)

    counts = jax.ops.segment_sum(mask.astype(float), inst_ids, num_segments=n_inst)

    # Marginalized log-likelihood -- only count valid observations
    N_valid = jnp.sum(mask)
    J = jnp.sum(A > 0)
    A_safe = jnp.where(A > 0, A, 1.0)
    return -0.5 * (
        jnp.sum(C - B**2 / A_safe)
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
        reads: Model paths a held build cached. Empty: the term reads the model
            directly.
    """

    data: RadialVelocityData
    system_path: str = eqx.field(static=True, default="")
    jitters: jnp.ndarray | None = None
    jitter_path: str | None = eqx.field(static=True, default=None)
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
            ValueError: When the system path differs; when one term has a
                ``jitter_path`` and the other a different one or fixed jitters (a
                ``jitter_path`` also needs identical instruments, since its leaf has
                one entry per instrument); or when a shared name has two fixed jitters.
        """
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
        rows = {
            "time_d": np.concatenate(
                [np.asarray(a.time_d)[keep_a], np.asarray(b.time_d)[keep_b]]
            ),
            "rv": np.concatenate([np.asarray(a.rv)[keep_a], np.asarray(b.rv)[keep_b]]),
            "rv_err": np.concatenate(
                [np.asarray(a.rv_err)[keep_a], np.asarray(b.rv_err)[keep_b]]
            ),
            "inst_ids": np.concatenate(
                [np.asarray(a.inst_ids)[keep_a], remap[np.asarray(b.inst_ids)[keep_b]]]
            ),
        }
        data = RadialVelocityData.pad(
            instruments=instruments,
            max_n=max(a.valid.shape[0], rows["rv"].shape[0]),
            **{k: jnp.asarray(v) for k, v in rows.items()},
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

    def _score(self, model):
        valid = self.data.valid
        data = RadialVelocityData(
            time_d=jnp.where(valid, self.data.time_d, 0.0),
            rv=jnp.where(valid, self.data.rv, 0.0),
            rv_err=jnp.where(valid, self.data.rv_err, 1.0),
            inst_ids=jnp.where(valid, self.data.inst_ids, 0),
            valid=valid,
            instruments=self.data.instruments,
        )
        system = resolve(model, self.system_path)
        if isinstance(system, System):
            prediction = system.solve(data.time_d).radial_velocity_m_per_s(
                0, "barycenter"
            )
        else:
            prediction = system.star_radial_velocity(time_d=data.time_d)
        prediction = jnp.where(valid, prediction, 0.0)
        value = loglike_rv_marginalized(
            data.rv,
            prediction,
            data.rv_err,
            data.inst_ids,
            data.n_inst,
            self._jitters(model),
            valid,
        )
        return value, prediction

    def __call__(self, model):
        """Score RVs with the jitter of ``jitters`` or ``jitter_path`` (m/s).

        Args:
            model: The model holding the system at ``system_path`` and any
                ``jitter_path``.

        Returns:
            Scalar log-likelihood with the instrument offsets marginalized.
        """
        return self._score(model)[0]

    def predict(self, model):
        """Return the stellar radial velocity at each epoch (m/s), invalid rows zero.

        Args:
            model: The model holding the system at ``system_path``.

        Returns:
            Stellar radial velocity (m/s) at each epoch, shape ``(T,)``.
        """
        return self._score(model)[1]

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
            key, data.rv.shape, dtype=jnp.float64
        )
        sampled = eqx.tree_at(lambda d: d.rv, data, jnp.where(valid, draw, data.rv))
        return eqx.tree_at(lambda t: t.data, self, sampled)
