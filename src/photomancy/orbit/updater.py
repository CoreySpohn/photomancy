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

Kernel (fixed-particle tempered SMC with a fixed loop count):

- The new record's log likelihood ``l`` is tempered in from ``beta = 0`` to 1
  on top of the posterior given the buffered records, so rung ``r`` targets
  ``prior x L_buffer x L_new^beta_r``.
- Each rung picks its temperature increment by a fixed-count bisection so
  that the conditional effective sample size of the incremental weights equals
  ``target_cess_fraction`` of the particles, reweights, resamples
  systematically and applies exactly ``n_moves`` random-walk Metropolis moves
  that leave the rung's tempered target invariant.
- The ladder is a ``fori_loop`` over ``n_rungs``: once ``beta`` reaches 1 the
  remaining rungs are no-ops. If the budget ends below 1 the rest of the
  likelihood is applied as one importance reweighting without resampling or
  moves, so the returned belief is still a weighted approximation of the full
  posterior, and the status says the ladder was incomplete.
- Moves propose ``wrap(theta + scale L z)`` in the target's chart, with ``L``
  the Cholesky factor of the weighted particle covariance (periodic coordinates
  centered on their circular mean) and ``scale = proposal_scale / sqrt(d)``.
  The proposal is symmetric on the chart (a Gaussian step, wrapped on the
  circle for periodic coordinates), so the Hastings ratio is the ratio of the
  tempered target densities; any chart Jacobian already sits in
  :meth:`AbstractUpdateTarget.log_prior`.
- The sum over rungs of the log mean incremental weight is the log marginal
  likelihood of the record given the buffered records.
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

    When more than one condition holds, the first in the order ``NO_SUPPORT``,
    ``BUDGET_EXHAUSTED``, ``LOW_ESS``, ``INCOMPLETE_LADDER`` is reported; the
    diagnostics (``ess``, ``temperature``, ``rungs_used``) always carry what
    is needed to recover the others.

    Attributes:
        OK: The record was assimilated with the ladder reaching temperature 1
            inside the rung budget and every reweighting step keeping its
            conditional ESS at or above the threshold, or the record was a
            no-op (padded or duplicate).
        LOW_ESS: The record was assimilated, but the returned belief is
            effectively fewer than ``low_ess_fraction`` of the particles: the
            smallest conditional effective sample size of any reweighting step
            of the update (the ``ess`` diagnostic), or the number of distinct
            weighted particles (``n_unique``), is below that fraction. With
            the adaptive ladder every rung sits at ``target_cess_fraction`` by
            construction, so the first test catches a degenerate final
            correction of an incomplete ladder (or a rung whose bisection could
            not reach the target), and the second catches moves that failed to
            rejuvenate the duplicates resampling creates.
        NO_SUPPORT: No particle has a finite posterior weight under the record;
            the belief is returned unchanged.
        BUDGET_EXHAUSTED: A static capacity ran out: the accepted-record buffer
            is full, so the record cannot be stored. The belief is returned
            unchanged. A short rung budget is ``INCOMPLETE_LADDER``, not this.
        INCOMPLETE_LADDER: The record was assimilated, but the ``n_rungs``
            adaptive rungs ended with the temperature below 1. The remaining
            fraction of the record's log likelihood was applied as one
            importance reweighting without resampling or moves, so the belief
            is a weighted approximation of the full posterior whose particles
            were never moved under it; that correction kept its conditional
            ESS at or above the threshold (otherwise the status is
            ``LOW_ESS``).
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
        ess: When the record was applied, the smallest conditional effective
            sample size of any reweighting step of the update (each adaptive
            rung and, for an incomplete ladder, the final correction), in
            particles. The conditional ESS of incremental weights ``u`` on
            normalized weights ``W`` is ``N (sum W u)^2 / sum W u^2``. When
            nothing was applied, the Kish ESS of the current weights (0 when
            no particle has weight).
        log_evidence_increment: Log of the record's marginal likelihood under
            the incoming belief (0 when nothing was applied).
        rungs_used: Adaptive tempering rungs that ran.
        acceptance_rate: Fraction of Metropolis proposals accepted over all
            moves of the update (0 when no moves ran).
        temperature: Temperature reached by the adaptive rungs: 1 when the
            ladder completed, below 1 for an incomplete ladder, 0 when nothing
            was applied.
        n_unique: Number of distinct particles with positive weight in the
            belief this call returns (0 when no particle has weight). Far
            below the particle count after a completed ladder, it shows that
            the moves did not rejuvenate the resampled duplicates. A particle
            moved by any accepted step counts as distinct, so this bounds the
            diversity from above.
    """

    status: jax.Array = eqx.field(converter=_as_int)
    applied: jax.Array = eqx.field(converter=_as_bool)
    duplicate: jax.Array = eqx.field(converter=_as_bool)
    ess: jax.Array = eqx.field(converter=_as_float)
    log_evidence_increment: jax.Array = eqx.field(converter=_as_float)
    rungs_used: jax.Array = eqx.field(converter=_as_int)
    acceptance_rate: jax.Array = eqx.field(converter=_as_float)
    temperature: jax.Array = eqx.field(converter=_as_float)
    n_unique: jax.Array = eqx.field(converter=_as_int)


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

    def log_likelihood_split(self, theta, records, select):
        """Log likelihood over two disjoint parts of a buffer, one prediction.

        Args:
            theta: Chart point. Shape ``(d,)``.
            records: A ``RelativeAstromData``; invalid entries contribute zero.
            select: Boolean mask of the entries in the second part. Shape
                ``(K,)``.

        Returns:
            Tuple ``(rest, selected)`` of scalar log likelihoods over the valid
            entries outside and inside ``select``; they sum to
            :meth:`log_likelihood`.
        """
        ra, dec = self.predict(theta, records.times)

        def part(mask):
            masked = eqx.tree_at(lambda r: r.is_valid, records, records.is_valid & mask)
            return loglike_relative_astrom(ra, dec, masked)

        return part(~select), part(select)

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
            jnp.all(jnp.isfinite(theta))
            & (ln_a >= ln_a_lo)
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
    """Kish effective sample size of normalized log weights (0 if all -inf)."""
    total = jnp.sum(jnp.exp(2.0 * log_weights))
    return jnp.where(total > 0.0, 1.0 / jnp.where(total > 0.0, total, 1.0), 0.0)


def _normalize(log_weights):
    """Normalize log weights in max-shifted form.

    NaN counts as zero weight. Shifting by the largest weight before the
    log-sum-exp keeps the normalized weights exact when every weight carries
    a large common offset (a log likelihood of order -1e17 has a float spacing
    larger than ``log n``).

    Returns:
        Tuple ``(normalized, log_total, no_support)``: normalized log weights
        (all ``-inf`` when nothing is supported), the log of the unnormalized
        total, and whether no weight is finite.
    """
    log_weights = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
    peak = jnp.max(log_weights)
    no_support = ~jnp.isfinite(peak)
    shifted = log_weights - jnp.where(no_support, 0.0, peak)
    lse = logsumexp(shifted)
    normalized = jnp.where(no_support, -jnp.inf, shifted - lse)
    log_total = jnp.where(no_support, -jnp.inf, peak + lse)
    return normalized, log_total, no_support


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


def _finite_or_neg_inf(x):
    """Replace NaN (and +inf) by ``-inf``: an undefined density is zero."""
    return jnp.where(jnp.isnan(x) | (x == jnp.inf), -jnp.inf, x)


def _log_cess_fraction(log_weights, increment):
    """Log of the conditional ESS fraction ``(sum W u)^2 / sum W u^2``.

    Args:
        log_weights: Normalized log weights ``log W``. Shape ``(n,)``.
        increment: Incremental log weights ``log u``, shifted so that their
            largest finite value is near 0 (``-inf`` for zero). Shape ``(n,)``.

    Returns:
        Scalar in ``(-inf, 0]``; NaN when no weight is finite.
    """
    return 2.0 * logsumexp(log_weights + increment) - logsumexp(
        log_weights + 2.0 * increment
    )


def _weighted_cholesky(particles, log_weights, periods):
    """Cholesky factor of the weighted particle covariance in the chart.

    Periodic coordinates are centered on their weighted circular mean and their
    deviations wrapped into ``[-period / 2, period / 2)``, so a cloud that
    straddles the wrap point has its true (small) spread. A relative jitter of
    ``1e-10`` times the mean variance, with an absolute floor, keeps the factor
    finite for a degenerate cloud.

    Args:
        particles: Chart points. Shape ``(n, d)``.
        log_weights: Normalized log weights. Shape ``(n,)``.
        periods: Period of each coordinate (0 if non-periodic). Shape ``(d,)``.

    Returns:
        Lower-triangular factor. Shape ``(d, d)``.
    """
    w = jnp.exp(log_weights)
    periodic = periods > 0.0
    safe_period = jnp.where(periodic, periods, 1.0)
    angle = two_pi * particles / safe_period
    circular_mean = (
        safe_period / two_pi * jnp.arctan2(w @ jnp.sin(angle), w @ jnp.cos(angle))
    )
    center = jnp.where(periodic, circular_mean, w @ particles)
    dev = particles - center
    wrapped = jnp.mod(dev + 0.5 * safe_period, safe_period) - 0.5 * safe_period
    dev = jnp.where(periodic, wrapped, dev)
    dev = jnp.where((w > 0.0)[:, None], dev, 0.0)
    cov = (w[:, None] * dev).T @ dev
    d = particles.shape[1]
    jitter = 1e-10 * jnp.trace(cov) / d + jnp.finfo(float).tiny
    chol = jnp.linalg.cholesky(cov + jitter * jnp.eye(d))
    return jnp.where(jnp.isfinite(chol), chol, 0.0)


def _metropolis_step(key, particles, aux, step_chol, wrap, evaluate, log_target):
    """One random-walk Metropolis move of every particle.

    Each particle proposes ``wrap(theta + step_chol @ z)`` with ``z`` standard
    normal. The step is a fixed Gaussian, symmetric in the chart, and wrapping a
    symmetric step onto a circle keeps it symmetric (the wrapped density is a
    sum over images of a symmetric kernel), so the Hastings ratio reduces to
    ``exp(log_target(proposal) - log_target(current))``. A density defined in
    another parameterization must carry that chart's Jacobian.

    Args:
        key: Typed PRNG key.
        particles: Chart points. Shape ``(n, d)``.
        aux: Per-particle cached values ``evaluate(particles)`` (a pytree with
            leading axis ``n``).
        step_chol: Lower-triangular step factor, scale included. Shape
            ``(d, d)``.
        wrap: Map of a chart point onto the chart's fundamental domain.
        evaluate: Chart point ``(d,)`` to its cached values.
        log_target: Cached values of one point to its log target density; NaN
            counts as ``-inf``.

    Returns:
        Tuple ``(particles, aux, accepted)``, ``accepted`` boolean ``(n,)``.
    """
    k_prop, k_acc = jax.random.split(key)
    n, d = particles.shape
    proposal = jax.vmap(wrap)(
        particles + jax.random.normal(k_prop, (n, d)) @ step_chol.T
    )
    aux_prop = jax.vmap(evaluate)(proposal)
    log_alpha = _finite_or_neg_inf(
        jax.vmap(log_target)(aux_prop) - jax.vmap(log_target)(aux)
    )
    accepted = jnp.log(jax.random.uniform(k_acc, (n,))) < log_alpha

    def pick(new, old):
        mask = accepted.reshape((n,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, old)

    return (
        pick(proposal, particles),
        jax.tree.map(pick, aux_prop, aux),
        accepted,
    )


def _count_unique(particles, log_weights):
    """Number of distinct particles with positive weight (0 if none)."""
    positive = jnp.isfinite(log_weights)
    top = particles[jnp.argmax(log_weights)]
    rows = jnp.where(positive[:, None], particles, top)
    ordered = rows[jnp.lexsort(rows.T[::-1])]
    distinct = jnp.any(ordered[1:] != ordered[:-1], axis=1)
    return jnp.where(jnp.any(positive), 1 + jnp.sum(distinct), 0)


class _Ladder(eqx.Module):
    """Carry of the tempering ladder (internal)."""

    particles: jax.Array
    log_weights: jax.Array
    log_prior: jax.Array
    ll_rest: jax.Array
    ll_new: jax.Array
    beta: jax.Array
    log_z: jax.Array
    rungs: jax.Array
    accepted: jax.Array
    proposed: jax.Array
    min_cess: jax.Array


class OrbitUpdater(eqx.Module):
    """Fixed-capacity sequential updater of a particle belief.

    Capacities and budget are static (they set array shapes and loop lengths);
    the thresholds and the proposal scale are traced arrays, so changing them
    does not recompile. Every method is ``jit``-compiled and composes with
    ``jax.vmap`` over beliefs, records and keys. One update makes at most
    ``n_particles x (1 + n_rungs x n_moves)`` likelihood evaluations, each over
    the ``record_capacity`` buffer slots.

    Args:
        n_particles: Particles per belief.
        n_rungs: Adaptive tempering rungs per update (the ladder's cap).
        n_moves: Random-walk Metropolis moves per rung.
        record_capacity: Capacity of the accepted-record buffer.
        low_ess_fraction: An applied update whose ``ess`` diagnostic (the
            smallest conditional ESS of its reweighting steps) or ``n_unique``
            is below this fraction of ``n_particles`` reports ``LOW_ESS``.
            Default 0.1.
        target_cess_fraction: Conditional ESS, as a fraction of
            ``n_particles``, that each adaptive rung's temperature increment
            is bisected to. Must exceed ``low_ess_fraction``. Default 0.5.
        proposal_scale: Random-walk step in units of the particle covariance,
            divided by ``sqrt(d)``. Default 0.3, well below the 2.38 that is
            optimal for a Gaussian target: one or two astrometric epochs
            confine the orbit posteriors to thin curved regions where larger
            steps are almost always rejected.
        n_bisect: Bisection iterations per rung (static). The smallest
            temperature increment a rung can take is ``(1 - beta) / 2^n_bisect``.
            Default 40.

    Raises:
        ValueError: If a capacity is not positive.
        RuntimeError: If 64-bit mode is off at construction.
    """

    n_particles: int = eqx.field(static=True)
    n_rungs: int = eqx.field(static=True)
    n_moves: int = eqx.field(static=True)
    record_capacity: int = eqx.field(static=True)
    low_ess_fraction: jax.Array = eqx.field(converter=_as_float, default=0.1)
    target_cess_fraction: jax.Array = eqx.field(converter=_as_float, default=0.5)
    proposal_scale: jax.Array = eqx.field(converter=_as_float, default=0.3)
    n_bisect: int = eqx.field(static=True, default=40)

    def __check_init__(self):
        """Validate capacities and the floating-point mode."""
        for name in (
            "n_particles",
            "n_rungs",
            "n_moves",
            "record_capacity",
            "n_bisect",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("OrbitUpdater requires jax_enable_x64.")

    def _diagnostics(
        self,
        status,
        applied,
        duplicate,
        ess,
        n_unique,
        log_z=0.0,
        rungs=0,
        acceptance=0.0,
        temperature=0.0,
    ):
        return UpdaterDiagnostics(
            status=status,
            applied=applied,
            duplicate=duplicate,
            ess=ess,
            log_evidence_increment=log_z,
            rungs_used=rungs,
            acceptance_rate=acceptance,
            temperature=temperature,
            n_unique=n_unique,
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
        log_w, _, no_support = _normalize(
            jnp.where(jnp.isfinite(log_prior), 0.0, -jnp.inf)
        )
        status = jnp.where(no_support, UpdaterStatus.NO_SUPPORT, UpdaterStatus.OK)
        return UpdaterBelief(
            target=target,
            particles=particles,
            log_weights=log_w,
            records=RelativeAstromData.zeros(self.record_capacity),
            record_ids=jnp.zeros(self.record_capacity, dtype=int),
            n_records=jnp.asarray(0, dtype=int),
            key=key,
            diagnostics=self._diagnostics(
                status, False, False, _kish_ess(log_w), _count_unique(particles, log_w)
            ),
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
            return belief, self._unchanged_diagnostics(belief, status, duplicate)

        def apply(operands):
            return self._tempered_kernel(*operands)

        return jax.lax.cond(attempt & ~full, apply, skip, (belief, record, key))

    def _unchanged_diagnostics(self, belief, status, duplicate=False):
        """Diagnostics of a call that returns ``belief`` unchanged."""
        return self._diagnostics(
            status,
            False,
            duplicate,
            _kish_ess(belief.log_weights),
            _count_unique(belief.particles, belief.log_weights),
        )

    def _move_log_density(self, aux, beta):
        """Tempered log target ``log p + l_rest + beta l_new`` of one particle.

        Args:
            aux: Tuple ``(log_prior, ll_rest, ll_new)`` of scalars.
            beta: Temperature in ``(0, 1]``.

        Returns:
            Scalar log density (``-inf`` off support).
        """
        log_prior, ll_rest, ll_new = aux
        return log_prior + ll_rest + beta * ll_new

    def _increment(self, log_weights, shifted, beta):
        """Temperature increment of one rung by fixed-count bisection.

        Returns the largest increment ``delta`` in ``(0, 1 - beta]`` whose
        conditional ESS fraction is at least ``target_cess_fraction`` (to the
        bisection's resolution), or the smallest bracketed increment when no
        bisected value reaches the target, so every rung advances.

        Args:
            log_weights: Normalized log weights. Shape ``(n,)``.
            shifted: New-record log likelihoods minus their largest finite
                value. Shape ``(n,)``.
            beta: Current temperature, below 1.

        Returns:
            Tuple ``(delta, reaches_one)``.
        """
        remaining = 1.0 - beta
        log_target = jnp.log(self.target_cess_fraction)

        def good(delta):
            return _log_cess_fraction(log_weights, delta * shifted) >= log_target

        def halve(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            ok = good(mid)
            return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

        lo, hi = jax.lax.fori_loop(
            0, self.n_bisect, halve, (jnp.zeros_like(beta), remaining)
        )
        reaches_one = good(remaining)
        delta = jnp.where(reaches_one, remaining, jnp.where(lo > 0.0, lo, hi))
        return delta, reaches_one

    def _tempered_kernel(self, belief, record, key):
        """Temper the record's likelihood in over at most ``n_rungs`` rungs."""
        target = belief.target
        n = self.n_particles
        slot = belief.n_records
        records = jax.tree.map(
            lambda buf, new: buf.at[slot].set(new[0]), belief.records, record.as_data()
        )
        is_new = jnp.arange(self.record_capacity) == slot
        periods = jnp.asarray(target.periods, dtype=float)
        step_scale = self.proposal_scale / math.sqrt(len(target.periods))

        def evaluate(theta):
            ll_rest, ll_new = target.log_likelihood_split(theta, records, is_new)
            return (
                _finite_or_neg_inf(target.log_prior(theta)),
                _finite_or_neg_inf(ll_rest),
                _finite_or_neg_inf(ll_new),
            )

        log_prior, ll_rest, ll_new = jax.vmap(evaluate)(belief.particles)
        _, _, no_support = _normalize(belief.log_weights + ll_new)

        def shift(state):
            """New-record log likelihoods minus their weighted-support maximum."""
            live = jnp.isfinite(state.log_weights) & jnp.isfinite(state.ll_new)
            peak = jnp.max(jnp.where(live, state.ll_new, -jnp.inf))
            peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
            return peak, state.ll_new - peak

        def reweight(state, delta, peak, shifted):
            """Weights times ``u = exp(delta l)``; evidence and CESS updates."""
            increment = jnp.where(jnp.isfinite(shifted), delta * shifted, -jnp.inf)
            cess = n * jnp.exp(_log_cess_fraction(state.log_weights, increment))
            log_w, log_total, _ = _normalize(state.log_weights + increment)
            return eqx.tree_at(
                lambda s: (s.log_weights, s.log_z, s.min_cess),
                state,
                (
                    log_w,
                    state.log_z + delta * peak + log_total,
                    jnp.minimum(state.min_cess, cess),
                ),
            )

        def rung(r, state):
            def active(state):
                k_resample, k_move = jax.random.split(jax.random.fold_in(key, r))
                peak, shifted = shift(state)
                delta, reaches_one = self._increment(
                    state.log_weights, shifted, state.beta
                )
                state = reweight(state, delta, peak, shifted)
                beta = jnp.where(reaches_one, 1.0, jnp.minimum(state.beta + delta, 1.0))
                step_chol = step_scale * _weighted_cholesky(
                    state.particles, state.log_weights, periods
                )
                idx = _systematic_indices(k_resample, state.log_weights, n)
                particles = state.particles[idx]
                aux = (state.log_prior[idx], state.ll_rest[idx], state.ll_new[idx])

                def move(m, carry):
                    particles, aux, accepted = carry
                    particles, aux, acc = _metropolis_step(
                        jax.random.fold_in(k_move, m),
                        particles,
                        aux,
                        step_chol,
                        target.wrap,
                        evaluate,
                        lambda a: self._move_log_density(a, beta),
                    )
                    return particles, aux, accepted + jnp.sum(acc)

                particles, aux, accepted = jax.lax.fori_loop(
                    0, self.n_moves, move, (particles, aux, jnp.asarray(0, dtype=int))
                )
                return _Ladder(
                    particles=particles,
                    log_weights=jnp.full(n, -jnp.log(n)),
                    log_prior=aux[0],
                    ll_rest=aux[1],
                    ll_new=aux[2],
                    beta=beta,
                    log_z=state.log_z,
                    rungs=state.rungs + 1,
                    accepted=state.accepted + accepted,
                    proposed=state.proposed + n * self.n_moves,
                    min_cess=state.min_cess,
                )

            return jax.lax.cond(state.beta < 1.0, active, lambda s: s, state)

        def run_ladder(_):
            start = _Ladder(
                particles=belief.particles,
                log_weights=belief.log_weights,
                log_prior=log_prior,
                ll_rest=ll_rest,
                ll_new=ll_new,
                beta=jnp.asarray(0.0),
                log_z=jnp.asarray(0.0),
                rungs=jnp.asarray(0, dtype=int),
                accepted=jnp.asarray(0, dtype=int),
                proposed=jnp.asarray(0, dtype=int),
                min_cess=jnp.asarray(float(n)),
            )
            state = jax.lax.fori_loop(0, self.n_rungs, rung, start)

            def correct(state):
                peak, shifted = shift(state)
                return reweight(state, 1.0 - state.beta, peak, shifted)

            final = jax.lax.cond(state.beta < 1.0, correct, lambda s: s, state)
            n_unique = _count_unique(final.particles, final.log_weights)
            floor = self.low_ess_fraction * n
            low_ess = (final.min_cess < floor) | (n_unique < floor)
            status = jnp.where(
                low_ess,
                UpdaterStatus.LOW_ESS,
                jnp.where(
                    state.beta < 1.0, UpdaterStatus.INCOMPLETE_LADDER, UpdaterStatus.OK
                ),
            )
            diagnostics = self._diagnostics(
                status,
                True,
                False,
                final.min_cess,
                n_unique,
                final.log_z,
                final.rungs,
                final.accepted / jnp.maximum(final.proposed, 1),
                state.beta,
            )
            updated = UpdaterBelief(
                target=target,
                particles=final.particles,
                log_weights=final.log_weights,
                records=records,
                record_ids=belief.record_ids.at[slot].set(record.record_id),
                n_records=slot + 1,
                key=key,
                diagnostics=diagnostics,
            )
            return updated, diagnostics

        def refuse(_):
            return belief, self._unchanged_diagnostics(belief, UpdaterStatus.NO_SUPPORT)

        return jax.lax.cond(no_support, refuse, run_ladder, None)
