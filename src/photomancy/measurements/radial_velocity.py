"""Stellar radial velocities: data container, marginalized kernel and term.

Times are days from the system's epoch; velocities are m/s, positive receding.
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.model_paths import resolve
from orbix.system import System

from photomancy.infer.fit import AbstractTerm
from photomancy.measurements._padding import _pad_1d, _valid_mask

__all__ = ["RadialVelocityData", "RadialVelocityTerm", "loglike_rv_marginalized"]


class RadialVelocityData(eqx.Module):
    """Radial velocity observations from one or more instruments.

    Args:
        time_d: Observation epochs (days). Shape ``(N,)``.
        rv: Measured radial velocities (m/s). Shape ``(N,)``.
        rv_err: RV measurement uncertainties (m/s). Shape ``(N,)``.
        inst_ids: Integer instrument index per observation, ``0..n_inst-1``.
            Shape ``(N,)``.
        is_valid: Boolean validity mask. Shape ``(N,)``.
        n_inst: Number of distinct instruments (static, for ``segment_sum``).
    """

    time_d: jnp.ndarray
    rv: jnp.ndarray
    rv_err: jnp.ndarray
    inst_ids: jnp.ndarray
    is_valid: jnp.ndarray
    n_inst: int = eqx.field(static=True)

    @classmethod
    def pad(cls, *, time_d, rv, rv_err, inst_ids, n_inst, max_n):
        """Create a padded RadialVelocityData with ``is_valid`` mask."""
        n = jnp.asarray(time_d).shape[0]
        return cls(
            time_d=_pad_1d(time_d, max_n),
            rv=_pad_1d(rv, max_n),
            rv_err=_pad_1d(rv_err, max_n, fill=1.0),
            inst_ids=_pad_1d(inst_ids, max_n).astype(int),
            is_valid=_valid_mask(n, max_n),
            n_inst=n_inst,
        )

    @classmethod
    def zeros(cls, max_n, n_inst=1):
        """Create an all-invalid placeholder (for model tracing)."""
        return cls(
            time_d=jnp.zeros(max_n),
            rv=jnp.zeros(max_n),
            rv_err=jnp.ones(max_n),
            inst_ids=jnp.zeros(max_n, dtype=int),
            is_valid=jnp.zeros(max_n, dtype=bool),
            n_inst=n_inst,
        )


def loglike_rv_marginalized(
    rv_obs, rv_model, rv_err, inst_ids, n_inst, jitters, is_valid
):
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
        n_inst: Number of distinct instruments. Integer.
        jitters: Per-instrument jitter terms (m/s). Shape ``(n_inst,)``.
        is_valid: Boolean validity mask. Shape ``(N,)``.

    Returns:
        Scalar log-likelihood value.
    """
    # Mask invalid entries -- use jnp.where for gradient safety
    mask = is_valid  # boolean

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
        valid = self.data.is_valid
        data = RadialVelocityData(
            time_d=jnp.where(valid, self.data.time_d, 0.0),
            rv=jnp.where(valid, self.data.rv, 0.0),
            rv_err=jnp.where(valid, self.data.rv_err, 1.0),
            inst_ids=jnp.where(valid, self.data.inst_ids, 0),
            is_valid=valid,
            n_inst=self.data.n_inst,
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
        """Score RVs with the jitter of ``jitters`` or ``jitter_path`` (m/s)."""
        return self._score(model)[0]

    def predict(self, model):
        """Return the stellar radial velocity at each epoch (m/s), invalid rows zero."""
        return self._score(model)[1]

    def draw(self, key, model) -> Self:
        """Draw Gaussian RV noise with zero instrument offsets.

        The score marginalizes offsets against a flat measure and is therefore
        improper on raw RVs. Zero offsets choose a representative; its residual
        contrasts have exactly the marginalized Gaussian law. Jitter is added
        in quadrature, as in the score.
        """
        data = self.data
        valid = data.is_valid
        ids = jnp.where(valid, data.inst_ids, 0)
        jitter = self._jitters(model)
        sigma = jnp.sqrt(jnp.where(valid, data.rv_err, 1.0) ** 2 + jitter[ids] ** 2)
        draw = self.predict(model) + sigma * jax.random.normal(
            key, data.rv.shape, dtype=jnp.float64
        )
        sampled = eqx.tree_at(lambda d: d.rv, data, jnp.where(valid, draw, data.rv))
        return eqx.tree_at(lambda t: t.data, self, sampled)
