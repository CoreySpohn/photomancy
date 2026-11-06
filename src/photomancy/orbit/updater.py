"""Fixed-capacity orbit updater: belief, targets, statuses and update contract.

An :class:`OrbitUpdater` assimilates one astrometric record at a time into a
weighted particle belief whose every array has a shape fixed by static
capacities: the number of particles, the tempering rungs and moves per rung of
its budget, and the capacity of the accepted-record buffer. Only array values
change from one update to the next, so a single compiled program serves every
update of every world, and a batch of worlds runs as one ``jax.vmap`` over
beliefs with some lanes active and some padded.

The model is supplied by an :class:`AbstractUpdateTarget`: a prior, a likelihood
over the record buffer, and the chart in which the kernel moves (with its
periodic coordinates). Three targets are provided:

- :class:`ConjugateGaussianTarget`, a two-dimensional Gaussian with a closed-form
  posterior, which isolates the updater's bookkeeping from orbit geometry;
- :class:`ThieleInnesTarget`, the independent Thiele-Innes basis of
  :class:`~photomancy.orbit.few_epoch.TIBasisModel` with its implied-mass window;
- :class:`FixedMassTarget`, physical elements at a fixed stellar mass with the
  priors of :func:`~photomancy.orbit.ofti.ofti`, in a chart that stays regular
  at zero eccentricity and carries that chart's Jacobian.

Update contract:

- ``update(belief, record, key) -> (belief, diagnostics)`` never raises on data.
  Every outcome is a :class:`UpdaterStatus` in the diagnostics.
- An invalid (padded) record, a duplicate record (its ``record_id`` is already
  in the buffer), a full buffer and a record no particle supports all return
  every leaf of the belief bitwise unchanged: particles, weights, buffer,
  accepted count, key and stored diagnostics.
- The belief stores the key of the call that produced it and that call's
  diagnostics, so an unchanged belief still describes its own provenance.
- Keys are typed threefry keys (``jax.random.key(seed, impl="threefry2x32")``)
  and the updater requires 64-bit mode.

The kernel in this module is a placeholder: one importance reweighting by the
new record's likelihood followed by systematic resampling. It does not use the
rung or move budget and never reports ``INCOMPLETE_LADDER``; a tempered kernel
with Metropolis moves replaces it behind the same interface.
"""

import enum
import math
from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractClassVar
from hwoutils.constants import G, Msun2kg, two_pi
from jax.scipy.special import betainc, logsumexp
from jax.scipy.stats import beta as beta_dist
from orbix.equations import period_to_sma
from orbix.equations.orbit import mean_motion

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.few_epoch import TI_PARAM_NAMES, TIBasisModel
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.posterior import SamplePosterior


def _as_float(x):
    return jnp.asarray(x, dtype=float)


def _as_int(x):
    return jnp.asarray(x, dtype=int)


def _as_bool(x):
    return jnp.asarray(x, dtype=bool)


class UpdaterStatus(enum.IntEnum):
    """Outcome of one update, stored as an integer in the diagnostics.

    Attributes:
        OK: The record was assimilated (or was a no-op: padded or duplicate).
        LOW_ESS: The record was assimilated, but the Kish effective sample size
            of the reweighted particles fell below the updater's threshold.
        NO_SUPPORT: No particle has a finite posterior weight under the record;
            the belief is returned unchanged.
        BUDGET_EXHAUSTED: A fixed capacity ran out (the record buffer is full);
            the belief is returned unchanged.
        INCOMPLETE_LADDER: The rung budget ended before the record's likelihood
            was fully tempered in. Reserved for a tempered kernel.
    """

    OK = 0
    LOW_ESS = 1
    NO_SUPPORT = 2
    BUDGET_EXHAUSTED = 3
    INCOMPLETE_LADDER = 4


class UpdaterRecord(eqx.Module):
    """One relative astrometric record, possibly a padding slot.

    Field names follow :class:`~photomancy.orbit.data.RelativeAstromData`, into
    whose buffer an accepted record is written. The conjugate Gaussian target
    reads ``(ra, dec)`` as its two-dimensional measurement and ``ra_err``,
    ``dec_err`` as its known noise scales.

    Args:
        time: Epoch (days).
        ra: East offset (arcsec).
        dec: North offset (arcsec).
        ra_err: East uncertainty (arcsec).
        dec_err: North uncertainty (arcsec).
        record_id: Caller-assigned identity. A record whose id is already in
            the belief's buffer is a duplicate and adds no likelihood.
        corr: East-north error correlation. Default 0.
        is_valid: False marks a padding slot, which is a no-op. Default True.
    """

    time: jax.Array = eqx.field(converter=_as_float)
    ra: jax.Array = eqx.field(converter=_as_float)
    dec: jax.Array = eqx.field(converter=_as_float)
    ra_err: jax.Array = eqx.field(converter=_as_float)
    dec_err: jax.Array = eqx.field(converter=_as_float)
    record_id: jax.Array = eqx.field(converter=_as_int)
    corr: jax.Array = eqx.field(converter=_as_float, default=0.0)
    is_valid: jax.Array = eqx.field(converter=_as_bool, default=True)

    def as_data(self):
        """The record as a one-entry ``RelativeAstromData``."""
        return RelativeAstromData(
            times=self.time[None],
            ra=self.ra[None],
            dec=self.dec[None],
            ra_err=self.ra_err[None],
            dec_err=self.dec_err[None],
            corr=self.corr[None],
            planet_id=jnp.zeros(1, dtype=int),
            is_valid=self.is_valid[None],
        )


class UpdaterDiagnostics(eqx.Module):
    """Scalar diagnostics of one update call (or of a prior belief).

    Args:
        status: A :class:`UpdaterStatus` value.
        applied: Whether the record changed the belief.
        duplicate: Whether the record's id was already in the buffer.
        ess: Kish effective sample size of the reweighted particles (of the
            current particles when nothing was applied).
        log_evidence_increment: Log of the record's marginal likelihood under
            the incoming belief (0 when nothing was applied).
        rungs_used: Tempering rungs used.
        acceptance_rate: Mean Metropolis acceptance rate over the moves (0 when
            no moves ran).
    """

    status: jax.Array = eqx.field(converter=_as_int)
    applied: jax.Array = eqx.field(converter=_as_bool)
    duplicate: jax.Array = eqx.field(converter=_as_bool)
    ess: jax.Array = eqx.field(converter=_as_float)
    log_evidence_increment: jax.Array = eqx.field(converter=_as_float)
    rungs_used: jax.Array = eqx.field(converter=_as_int)
    acceptance_rate: jax.Array = eqx.field(converter=_as_float)


class AbstractUpdateTarget(eqx.Module):
    """The model an :class:`OrbitUpdater` assimilates records into.

    Particles are points of the target's chart, the coordinates the kernel
    moves in. A subclass supplies:

    - ``param_names`` and ``periods`` (class constants): the chart's coordinate
      names and, per coordinate, its period (0 for a non-periodic coordinate);
    - :meth:`sample_prior`: draws from a law ``q`` with ``q = Z p`` on the
      support of the prior density ``p`` (``exp(log_prior)``); draws outside
      the support are allowed and receive zero weight, so a truncated prior is
      sampled by drawing its untruncated form;
    - :meth:`log_prior`: the prior density in the chart, including the chart's
      Jacobian, and ``-inf`` outside the support;
    - :meth:`predict`: the predicted ``(ra, dec)`` (arcsec) at given epochs.

    The likelihood is the astrometric likelihood
    :func:`~photomancy.orbit.likelihoods.loglike_relative_astrom` of the
    predictions against the valid entries of a record buffer.

    Array fields of a target are traced, not static, so a target with new
    values (a different truncation, say) reuses the compiled update.
    """

    param_names: AbstractClassVar[tuple[str, ...]]
    periods: AbstractClassVar[tuple[float, ...]]

    @abstractmethod
    def sample_prior(self, key, n):
        """Draw ``n`` chart points. Shape ``(n, d)``."""

    @abstractmethod
    def log_prior(self, theta):
        """Log prior density of one chart point ``(d,)``; ``-inf`` off support."""

    @abstractmethod
    def predict(self, theta, times):
        """Predicted ``(ra, dec)`` in arcsec at ``times`` (days), each ``(K,)``."""

    def log_likelihood(self, theta, records):
        """Astrometric log likelihood of one chart point over a record buffer.

        Args:
            theta: Chart point. Shape ``(d,)``.
            records: A ``RelativeAstromData``; invalid entries contribute zero.

        Returns:
            Scalar log likelihood.
        """
        ra, dec = self.predict(theta, records.times)
        return loglike_relative_astrom(ra, dec, records)

    def wrap(self, theta):
        """Map the periodic coordinates of ``theta`` into ``[0, period)``."""
        periods = jnp.asarray(self.periods, dtype=float)
        periodic = periods > 0.0
        wrapped = jnp.mod(theta, jnp.where(periodic, periods, 1.0))
        return jnp.where(periodic, wrapped, theta)


class ConjugateGaussianTarget(AbstractUpdateTarget):
    """Two-dimensional Gaussian prior with Gaussian records (closed form).

    The parameter ``x`` has prior ``N(prior_mean, prior_scale^2 I)``; a record
    measures ``(ra, dec) ~ N(x, diag(ra_err^2, dec_err^2))``. The record time is
    ignored.

    Args:
        prior_mean: Prior mean. Shape ``(2,)``. Default ``(0, 0)``.
        prior_scale: Prior standard deviation per axis. Default 1.
    """

    prior_mean: jax.Array = eqx.field(converter=_as_float, default=(0.0, 0.0))
    prior_scale: jax.Array = eqx.field(converter=_as_float, default=1.0)

    param_names: ClassVar[tuple[str, ...]] = ("x0", "x1")
    periods: ClassVar[tuple[float, ...]] = (0.0, 0.0)

    def sample_prior(self, key, n):
        """Draw ``n`` points from the Gaussian prior. Shape ``(n, 2)``."""
        return self.prior_mean + self.prior_scale * jax.random.normal(key, (n, 2))

    def log_prior(self, theta):
        """Normalized Gaussian log prior density."""
        z = (theta - self.prior_mean) / self.prior_scale
        return -0.5 * jnp.sum(z**2) - 2.0 * jnp.log(self.prior_scale) - jnp.log(two_pi)

    def predict(self, theta, times):
        """Predict the measurement ``x`` at every epoch."""
        ones = jnp.ones_like(times)
        return theta[0] * ones, theta[1] * ones

    def conjugate_posterior(self, records):
        """Exact posterior mean and per-axis variance given a record buffer.

        Assumes uncorrelated record errors (``corr == 0``).

        Args:
            records: A ``RelativeAstromData``; invalid entries are ignored.

        Returns:
            Tuple ``(mean, variance)``, each shape ``(2,)``.
        """
        valid = records.is_valid
        prec_ra = jnp.where(valid, records.ra_err**-2, 0.0)
        prec_dec = jnp.where(valid, records.dec_err**-2, 0.0)
        prec0 = self.prior_scale**-2
        prec = prec0 + jnp.array([jnp.sum(prec_ra), jnp.sum(prec_dec)])
        weighted = jnp.array(
            [
                jnp.sum(jnp.where(valid, prec_ra * records.ra, 0.0)),
                jnp.sum(jnp.where(valid, prec_dec * records.dec, 0.0)),
            ]
        )
        variance = 1.0 / prec
        return variance * (prec0 * self.prior_mean + weighted), variance


class ThieleInnesTarget(AbstractUpdateTarget):
    """Independent Thiele-Innes basis with an implied-mass window.

    The chart is ``theta = (e, lnP, tau, east_X, east_Y, north_X, north_Y)`` of
    :class:`~photomancy.orbit.few_epoch.TIBasisModel`, whose priors, support
    and forward model are used unchanged; ``tau`` is periodic with period 1.
    The model's bounds are stored as arrays, so lanes may carry different
    truncations.

    Args:
        model: The ``TIBasisModel``. Default: its field defaults.
        dist_pc: Distance (parsec). Default 10.
    """

    model: TIBasisModel
    dist_pc: jax.Array

    param_names: ClassVar[tuple[str, ...]] = TI_PARAM_NAMES
    periods: ClassVar[tuple[float, ...]] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(self, model=None, dist_pc=10.0):
        """Store the model with array-valued bounds."""
        model = TIBasisModel() if model is None else model
        self.model = jax.tree.map(_as_float, model)
        self.dist_pc = _as_float(dist_pc)

    def sample_prior(self, key, n):
        """Draw the untruncated product prior; the mass window is the support."""
        m = self.model
        k = jax.random.split(key, 4)
        return jnp.column_stack(
            [
                jax.random.uniform(k[0], (n,), maxval=m.e_max),
                jax.random.uniform(k[1], (n,), minval=m.lnP_lo, maxval=m.lnP_hi),
                jax.random.uniform(k[2], (n,)),
                m.s_ti * jax.random.normal(k[3], (n, 4)),
            ]
        )

    def log_prior(self, theta):
        """The model's Gaussian TI block inside its support, else ``-inf``."""
        return jnp.where(
            self.model.in_support(theta), self.model.log_prior_ti(theta), -jnp.inf
        )

    def predict(self, theta, times):
        """Model offsets in arcsec (AU divided by the distance in parsec)."""
        east_au, north_au = self.model.forward_au(theta, times)
        return east_au / self.dist_pc, north_au / self.dist_pc


class FixedMassTarget(AbstractUpdateTarget):
    """Physical elements at a fixed stellar mass, with OFTI's priors.

    Prior on the physical elements ``(a, e, cos_i, W, omega, M_ref)``:
    ``ln a`` uniform between the semimajor axes of ``period_lo_day`` and
    ``period_hi_day`` (equivalently ``ln P`` uniform), ``e ~ Beta(ecc_alpha,
    ecc_beta)`` restricted to ``e < e_max``, ``cos_i ~ U(-1, 1)`` and ``W``,
    ``omega``, ``M_ref`` uniform on ``[0, 2 pi)``, where ``M_ref`` is the mean
    anomaly at ``t_ref_day``. The defaults are the arguments of
    :func:`~photomancy.orbit.ofti.ofti` with ``ecc_prior="kipping13"``.

    The chart is ``(ln_a, sqrt(e) cos(omega), sqrt(e) sin(omega), cos_i, W,
    lambda_ref)`` with ``lambda_ref = omega + M_ref``, which stays regular as
    ``e -> 0``. The map ``(sqrt(e) cos(omega), sqrt(e) sin(omega)) -> (e,
    omega)`` has Jacobian determinant 2 and the shear to ``lambda_ref`` has 1,
    so :meth:`log_prior` is the physical prior density plus ``log 2``. ``W``
    and ``lambda_ref`` are periodic. The density is normalized, truncation
    included.

    Args:
        Ms_kg: Stellar mass (kg). Default one solar mass.
        dist_pc: Distance (parsec). Default 10.
        period_lo_day: Lower period bound (days). Default 50.
        period_hi_day: Upper period bound (days). Default 5000.
        e_max: Eccentricity truncation. Default 0.9.
        ecc_alpha: First Beta shape of the eccentricity prior. Default 0.867.
        ecc_beta: Second Beta shape of the eccentricity prior. Default 3.03.
        t_ref_day: Epoch of ``M_ref`` (days). Default 0.
    """

    Ms_kg: jax.Array = eqx.field(converter=_as_float, default=Msun2kg)
    dist_pc: jax.Array = eqx.field(converter=_as_float, default=10.0)
    period_lo_day: jax.Array = eqx.field(converter=_as_float, default=50.0)
    period_hi_day: jax.Array = eqx.field(converter=_as_float, default=5000.0)
    e_max: jax.Array = eqx.field(converter=_as_float, default=0.9)
    # Kipping (2013) Beta shapes, the "kipping13" prior of photomancy.orbit.priors.
    ecc_alpha: jax.Array = eqx.field(converter=_as_float, default=0.867)
    ecc_beta: jax.Array = eqx.field(converter=_as_float, default=3.03)
    t_ref_day: jax.Array = eqx.field(converter=_as_float, default=0.0)

    param_names: ClassVar[tuple[str, ...]] = (
        "ln_a",
        "sqrt_e_cos_w",
        "sqrt_e_sin_w",
        "cos_i",
        "W",
        "lambda_ref",
    )
    periods: ClassVar[tuple[float, ...]] = (0.0, 0.0, 0.0, 0.0, two_pi, two_pi)

    def ln_a_bounds(self):
        """Bounds of ``ln a`` (AU) implied by the period range at fixed mass."""
        return (
            jnp.log(period_to_sma(self.period_lo_day, self.Ms_kg)),
            jnp.log(period_to_sma(self.period_hi_day, self.Ms_kg)),
        )

    def to_physical(self, theta):
        """Chart point to ``(a, e, cos_i, W, omega, M_ref)``. Shape ``(6,)``."""
        ln_a, h, k, cos_i, W, lam = theta
        omega = jnp.mod(jnp.arctan2(k, h), two_pi)
        return jnp.array(
            [
                jnp.exp(ln_a),
                h**2 + k**2,
                cos_i,
                jnp.mod(W, two_pi),
                omega,
                jnp.mod(lam - omega, two_pi),
            ]
        )

    def from_physical(self, elements):
        """``(a, e, cos_i, W, omega, M_ref)`` to a chart point. Shape ``(6,)``."""
        a, e, cos_i, W, omega, m_ref = elements
        root_e = jnp.sqrt(e)
        return jnp.array(
            [
                jnp.log(a),
                root_e * jnp.cos(omega),
                root_e * jnp.sin(omega),
                cos_i,
                jnp.mod(W, two_pi),
                jnp.mod(omega + m_ref, two_pi),
            ]
        )

    def sample_prior(self, key, n):
        """Draw the prior with ``e`` untruncated; ``e >= e_max`` is off support."""
        ln_a_lo, ln_a_hi = self.ln_a_bounds()
        k = jax.random.split(key, 6)
        ln_a = jax.random.uniform(k[0], (n,), minval=ln_a_lo, maxval=ln_a_hi)
        e = jax.random.beta(k[1], self.ecc_alpha, self.ecc_beta, (n,))
        cos_i = jax.random.uniform(k[2], (n,), minval=-1.0, maxval=1.0)
        W = jax.random.uniform(k[3], (n,), maxval=two_pi)
        omega = jax.random.uniform(k[4], (n,), maxval=two_pi)
        m_ref = jax.random.uniform(k[5], (n,), maxval=two_pi)
        root_e = jnp.sqrt(e)
        return jnp.column_stack(
            [
                ln_a,
                root_e * jnp.cos(omega),
                root_e * jnp.sin(omega),
                cos_i,
                W,
                jnp.mod(omega + m_ref, two_pi),
            ]
        )

    def log_prior(self, theta):
        """Normalized prior density in the chart, ``-inf`` off support."""
        ln_a_lo, ln_a_hi = self.ln_a_bounds()
        ln_a, h, k, cos_i = theta[0], theta[1], theta[2], theta[3]
        e = h**2 + k**2
        in_support = (
            (ln_a >= ln_a_lo)
            & (ln_a <= ln_a_hi)
            & (e < self.e_max)
            & (jnp.abs(cos_i) <= 1.0)
        )
        # The Beta density diverges at e = 0 for ecc_alpha < 1; a floor keeps
        # the log finite at the single point h = k = 0.
        e_safe = jnp.clip(e, jnp.finfo(float).tiny, self.e_max)
        log_e = beta_dist.logpdf(e_safe, self.ecc_alpha, self.ecc_beta) - jnp.log(
            betainc(self.ecc_alpha, self.ecc_beta, self.e_max)
        )
        log_density = (
            -jnp.log(ln_a_hi - ln_a_lo)
            + log_e
            + math.log(2.0)  # Jacobian of (h, k) -> (e, omega)
            - 3.0 * math.log(two_pi)  # omega, W and M_ref
            - math.log(2.0)  # cos_i
        )
        return jnp.where(in_support, log_density, -jnp.inf)

    def predict(self, theta, times):
        """Relative astrometry of the chart point (arcsec)."""
        a, e, cos_i, W, omega, m_ref = self.to_physical(theta)
        mu = G * self.Ms_kg
        tp = self.t_ref_day - m_ref / mean_motion(a, mu)
        return predict_relative_astrometry(
            times,
            a,
            e,
            cos_i,
            W,
            jnp.cos(omega),
            jnp.sin(omega),
            tp,
            self.Ms_kg,
            self.dist_pc,
        )


class UpdaterBelief(eqx.Module):
    """A fixed-shape weighted particle belief with its record buffer.

    Args:
        target: The model (prior, likelihood, chart) the belief is over.
        particles: Chart points. Shape ``(n_particles, d)``.
        log_weights: Normalized log weights (``-inf`` for zero weight). Shape
            ``(n_particles,)``.
        records: Accepted records, the first ``n_records`` entries valid. Shape
            ``(record_capacity,)`` per field.
        record_ids: Ids of the accepted records. Shape ``(record_capacity,)``.
        n_records: Number of accepted records.
        key: Typed key of the call that produced this belief.
        diagnostics: Diagnostics of the call that produced this belief.
    """

    target: AbstractUpdateTarget
    particles: jax.Array
    log_weights: jax.Array
    records: RelativeAstromData
    record_ids: jax.Array
    n_records: jax.Array
    key: jax.Array
    diagnostics: UpdaterDiagnostics

    def posterior(self):
        """The belief as a ``SamplePosterior`` named by the target's chart."""
        return SamplePosterior(
            samples=self.particles,
            log_weights=self.log_weights,
            evidence=jnp.asarray(jnp.nan),
            param_names=self.target.param_names,
        )


def _kish_ess(log_weights):
    """Kish effective sample size of normalized log weights."""
    return 1.0 / jnp.sum(jnp.exp(2.0 * log_weights))


def _systematic_indices(key, log_weights, n):
    """Systematic resampling indices from normalized log weights."""
    cdf = jnp.cumsum(jnp.exp(log_weights))
    cdf = cdf / cdf[-1]
    u = (jax.random.uniform(key) + jnp.arange(n)) / n
    return jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, n - 1)


def _check_key(key):
    """Require a typed threefry key (trace-safe: reads only the dtype)."""
    if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError(
            "OrbitUpdater needs a typed key: jax.random.key(seed, "
            "impl='threefry2x32'), not a raw uint32 key."
        )
    impl = str(jax.random.key_impl(key))
    if impl != "threefry2x32":
        raise TypeError(f"OrbitUpdater needs a threefry2x32 key, got {impl}.")


def _select(pred, on_true, on_false):
    """Leafwise select of two pytrees of equal structure by a scalar bool."""
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), on_true, on_false)


class OrbitUpdater(eqx.Module):
    """Fixed-capacity sequential updater of a particle belief.

    Capacities and budget are static (they set array shapes and loop lengths);
    the ESS threshold is a traced array, so changing it does not recompile.
    Every method is ``jit``-compiled and composes with ``jax.vmap`` over
    beliefs, records and keys.

    Args:
        n_particles: Particles per belief.
        n_rungs: Tempering rungs per update (budget; unused by the placeholder
            kernel).
        n_moves: Metropolis moves per rung (budget; unused by the placeholder
            kernel).
        record_capacity: Capacity of the accepted-record buffer.
        low_ess_fraction: An applied update whose Kish ESS is below this
            fraction of ``n_particles`` reports ``LOW_ESS``. Default 0.1.

    Raises:
        ValueError: If a capacity is not positive.
        RuntimeError: If 64-bit mode is off at construction.
    """

    n_particles: int = eqx.field(static=True)
    n_rungs: int = eqx.field(static=True)
    n_moves: int = eqx.field(static=True)
    record_capacity: int = eqx.field(static=True)
    low_ess_fraction: jax.Array = eqx.field(converter=_as_float, default=0.1)

    def __check_init__(self):
        """Validate capacities and the floating-point mode."""
        for name in ("n_particles", "n_rungs", "n_moves", "record_capacity"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("OrbitUpdater requires jax_enable_x64.")

    def _diagnostics(self, status, applied, duplicate, ess, log_z=0.0, rungs=0):
        return UpdaterDiagnostics(
            status=status,
            applied=applied,
            duplicate=duplicate,
            ess=ess,
            log_evidence_increment=log_z,
            rungs_used=rungs,
            acceptance_rate=0.0,
        )

    @eqx.filter_jit
    def prior_belief(self, target, key):
        """A belief with no records: ``n_particles`` draws from the prior.

        Draws off the prior's support get zero weight. If no draw is on the
        support the diagnostics report ``NO_SUPPORT``.

        Args:
            target: An :class:`AbstractUpdateTarget`.
            key: Typed threefry key, stored in the belief.

        Returns:
            An :class:`UpdaterBelief`.
        """
        _check_key(key)
        particles = target.sample_prior(key, self.n_particles)
        log_prior = jax.vmap(target.log_prior)(particles)
        log_w = jnp.where(jnp.isfinite(log_prior), 0.0, -jnp.inf)
        lse = logsumexp(log_w)
        no_support = ~jnp.isfinite(lse)
        log_w = log_w - jnp.where(no_support, 0.0, lse)
        status = jnp.where(no_support, UpdaterStatus.NO_SUPPORT, UpdaterStatus.OK)
        return UpdaterBelief(
            target=target,
            particles=particles,
            log_weights=log_w,
            records=RelativeAstromData.zeros(self.record_capacity),
            record_ids=jnp.zeros(self.record_capacity, dtype=int),
            n_records=jnp.asarray(0, dtype=int),
            key=key,
            diagnostics=self._diagnostics(status, False, False, _kish_ess(log_w)),
        )

    @eqx.filter_jit
    def init(self, target, record, key):
        """One-epoch initializer: the prior conditioned on one record.

        Equivalent to :meth:`prior_belief` followed by :meth:`update`, with
        keys folded from ``key``; the returned belief stores ``key``. Bounded
        and fixed-shape; a padded record returns the prior belief.

        Args:
            target: An :class:`AbstractUpdateTarget`.
            record: An :class:`UpdaterRecord`.
            key: Typed threefry key.

        Returns:
            Tuple ``(belief, diagnostics)``.
        """
        _check_key(key)
        prior = self.prior_belief(target, jax.random.fold_in(key, 0))
        belief, diagnostics = self.update(prior, record, jax.random.fold_in(key, 1))
        return eqx.tree_at(lambda b: b.key, belief, key), diagnostics

    @eqx.filter_jit
    def update(self, belief, record, key):
        """Assimilate one record.

        Args:
            belief: An :class:`UpdaterBelief` from this updater.
            record: An :class:`UpdaterRecord`; a padded record is a no-op.
            key: Typed threefry key, stored in the belief when the record is
                applied.

        Returns:
            Tuple ``(belief, diagnostics)``. Unless ``diagnostics.applied``,
            every leaf of the returned belief equals the input bitwise.
        """
        _check_key(key)
        duplicate = record.is_valid & jnp.any(
            belief.records.is_valid & (belief.record_ids == record.record_id)
        )
        full = belief.n_records >= self.record_capacity
        attempt = record.is_valid & ~duplicate

        def skip(operands):
            belief, _, _ = operands
            status = jnp.where(
                attempt & full, UpdaterStatus.BUDGET_EXHAUSTED, UpdaterStatus.OK
            )
            ess = _kish_ess(belief.log_weights)
            return belief, self._diagnostics(status, False, duplicate, ess)

        def apply(operands):
            return self._placeholder_kernel(*operands)

        return jax.lax.cond(attempt & ~full, apply, skip, (belief, record, key))

    def _placeholder_kernel(self, belief, record, key):
        """Importance reweighting by the record, then systematic resampling.

        Placeholder for the tempered kernel: it ignores the rung and move
        budget and the buffered records (which a move kernel needs for its
        target density).
        """
        target = belief.target
        log_like = jax.vmap(target.log_likelihood, in_axes=(0, None))(
            belief.particles, record.as_data()
        )
        log_w = belief.log_weights + log_like
        log_w = jnp.where(jnp.isnan(log_w), -jnp.inf, log_w)
        lse = logsumexp(log_w)
        no_support = ~jnp.isfinite(lse)
        log_w = log_w - jnp.where(no_support, 0.0, lse)
        ess = _kish_ess(log_w)
        log_z = jnp.where(no_support, 0.0, lse - logsumexp(belief.log_weights))

        idx = _systematic_indices(key, log_w, self.n_particles)
        i = belief.n_records
        candidate = UpdaterBelief(
            target=target,
            particles=belief.particles[idx],
            log_weights=jnp.full(self.n_particles, -jnp.log(self.n_particles)),
            records=jax.tree.map(
                lambda buf, new: buf.at[i].set(new[0]), belief.records, record.as_data()
            ),
            record_ids=belief.record_ids.at[i].set(record.record_id),
            n_records=i + 1,
            key=key,
            diagnostics=belief.diagnostics,
        )
        low_ess = ess < self.low_ess_fraction * self.n_particles
        status = jnp.where(
            no_support,
            UpdaterStatus.NO_SUPPORT,
            jnp.where(low_ess, UpdaterStatus.LOW_ESS, UpdaterStatus.OK),
        )
        applied = ~no_support
        diagnostics = self._diagnostics(
            status,
            applied,
            False,
            jnp.where(no_support, _kish_ess(belief.log_weights), ess),
            log_z,
            jnp.where(applied, 1, 0),
        )
        candidate = eqx.tree_at(lambda b: b.diagnostics, candidate, diagnostics)
        return _select(applied, candidate, belief), diagnostics
