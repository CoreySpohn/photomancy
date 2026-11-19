"""Thiele-Innes orbit target, refit status and the linear-block conditional.

:class:`ThieleInnesTarget` is the independent Thiele-Innes basis of
:class:`~photomancy.orbit.few_epoch.TIBasisModel` on its chart, with the four
constants as a linear block. :func:`_linear_gaussian` gives the conditional
Gaussian of a linear block and its log marginal, :func:`_rest_log_prior` the prior
without that block, and :class:`UpdaterStatus` the outcome codes of a refit.
"""

import enum
from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi
from orbix.system import FrameOrbit, ThieleInnesChart, disk_to_plane

from photomancy.measurements import loglike_relative_astrom
from photomancy.orbit.few_epoch import TIBasisModel

_TI_CHART = ThieleInnesChart()


def _as_float(x):
    return jnp.asarray(x, dtype=float)


class UpdaterStatus(enum.IntEnum):
    """Outcome of a refit, stored as an integer.

    Attributes:
        OK: The importance sample reached its effective-size floor.
        LOW_ESS: The importance sample stayed under its effective-size floor.
        NO_SUPPORT: No draw has a finite posterior weight.
        BUDGET_EXHAUSTED: A static capacity ran out.
        INCOMPLETE_LADDER: A tempering ladder ended below temperature 1.
    """

    OK = 0
    LOW_ESS = 1
    NO_SUPPORT = 2
    BUDGET_EXHAUSTED = 3
    INCOMPLETE_LADDER = 4


class AbstractUpdateTarget(eqx.Module):
    """A one-planet orbit model on a chart: prior, predictions and a linear block.

    A subclass supplies :meth:`log_prior` (the prior density in the chart,
    including the chart's Jacobian, and ``-inf`` outside the support) and
    :meth:`predict` (the predicted ``(ra, dec)`` in arcsec at given epochs). The
    likelihood is :func:`~photomancy.measurements.loglike_relative_astrom` of the
    predictions against the valid entries of a record buffer.

    A target may declare a linear block, ``linear_index`` (class constant): chart
    coordinates ``c`` in which :meth:`predict` is linear and homogeneous and whose
    prior is an independent zero-mean Gaussian of scale :meth:`linear_prior_scale`,
    possibly times an indicator.

    Array fields of a target are traced, not static, so a target with new values
    (a different truncation, say) reuses the compiled program.
    """

    linear_index: ClassVar[tuple[int, ...]] = ()

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
        return _split_loglike(ra, dec, records, select)

    def linear_design(self, theta, times):
        """Design matrices of the linear block: ``predict = (H_ra c, H_dec c)``.

        Valid only for a target whose :meth:`predict` is linear and homogeneous
        in the coordinates ``linear_index`` (``c``). Column ``j`` is the
        prediction at the ``j``-th unit vector of ``c``, the other coordinates
        of ``theta`` held fixed. This costs one :meth:`predict` per linear
        coordinate; a target can override it with a cheaper exact form.

        Args:
            theta: Chart point (its linear coordinates are ignored). Shape
                ``(d,)``.
            times: Epochs (days). Shape ``(K,)``.

        Returns:
            Tuple ``(H_ra, H_dec)``, each shape ``(K, p)``.
        """
        idx = jnp.asarray(self.linear_index)
        units = jnp.eye(len(self.linear_index))
        ra, dec = jax.vmap(lambda u: self.predict(theta.at[idx].set(u), times))(units)
        return ra.T, dec.T

    def linear_prior_scale(self):
        """Standard deviation of the zero-mean Gaussian prior of the linear block.

        A target that declares ``linear_index`` must have :meth:`log_prior`
        equal to ``log p_rest(theta) - |c|^2 / (2 s^2)`` up to a constant, where
        ``s`` is this scale and ``p_rest`` (which may contain a support
        indicator depending on ``c``) is everything else.
        """
        raise NotImplementedError(f"{type(self).__name__} has no linear block.")


class ThieleInnesTarget(AbstractUpdateTarget):
    """Independent Thiele-Innes basis with an implied-mass window.

    The chart is ``theta = (e, lnP, tau, east_X, east_Y, north_X, north_Y)`` of
    :class:`~photomancy.orbit.few_epoch.TIBasisModel`, whose priors, support
    and forward model are used unchanged; ``tau`` is periodic with period 1.
    The four TI constants are the linear block: the predictions are linear in
    them and their prior is Gaussian times the implied-mass window.
    The model's bounds are stored as arrays, so lanes may carry different
    truncations.

    Args:
        model: The ``TIBasisModel``. Default: its field defaults.
        dist_pc: Distance (parsec). Default 10.
    """

    model: TIBasisModel
    dist_pc: jax.Array

    linear_index: ClassVar[tuple[int, ...]] = (3, 4, 5, 6)

    def __init__(self, model=None, dist_pc=10.0):
        """Store the model with array-valued bounds."""
        model = TIBasisModel() if model is None else model
        self.model = jax.tree.map(_as_float, model)
        self.dist_pc = _as_float(dist_pc)

    def log_prior(self, theta):
        """The model's Gaussian TI block inside its support, else ``-inf``."""
        return jnp.where(
            self.model.in_support(theta), self.model.log_prior_ti(theta), -jnp.inf
        )

    def predict(self, theta, times):
        """Model offsets in arcsec (AU divided by the distance in parsec)."""
        east_au, north_au = self.model.forward_au(theta, times)
        return east_au / self.dist_pc, north_au / self.dist_pc

    def chart_coordinates(self, theta):
        """The chart point as a ``ThieleInnesChart`` point, with the map's log Jacobian.

        The chart point ``(e, lnP, tau, east_X, east_Y, north_X, north_Y)`` has its
        constants on the periapsis axes; orbix's chart has them on the mean
        longitude axes at ``model.t_ref``, rotated from the periapsis axes by
        ``2 pi tau``. The coordinates ``(ln n, ux, uy, A, B, F, G)`` are on the
        ``+1`` line-of-sight sheet; the second sheet is the mirror orbit that
        shares the sky track.

        Args:
            theta: Chart point. Shape ``(7,)``.

        Returns:
            Tuple ``(x, log_jacobian)`` with ``x`` of shape ``(7,)`` and the log
            absolute determinant of ``dx / dtheta``,
            ``log(2 pi e) - 2 log(1 - e^2)``: the constants rotate (unit
            determinant), ``lnP -> -ln n`` has unit slope and
            ``(e, 2 pi tau)`` are polar coordinates of the eccentricity vector.
        """
        e, ln_period, tau = theta[0], theta[1], theta[2]
        angle = two_pi * tau
        c, s = jnp.cos(angle), jnp.sin(angle)
        block = jnp.array([[theta[3], theta[4]], [theta[5], theta[6]]])
        (b, g), (a, f) = block @ jnp.array([[c, s], [-s, c]])
        u = disk_to_plane(e * jnp.stack([c, s]))
        x = jnp.concatenate(
            [jnp.stack([jnp.log(two_pi) - ln_period]), u, jnp.stack([a, b, f, g])]
        )
        return x, jnp.log(two_pi * e) - 2.0 * jnp.log1p(-(e**2))

    def chart_orbit(self, theta):
        """The orbit of a chart point, on the ``+1`` line-of-sight sheet."""
        x, _ = self.chart_coordinates(theta)
        return _TI_CHART.to_orbit(x, mu_AU3_per_d2=1.0, epoch_d=self.model.t_ref)

    def linear_design(self, theta, times):
        """Design matrices from one Kepler solve.

        East is ``(X, Y) . (east_X, east_Y)`` and north ``(X, Y) . (north_X,
        north_Y)`` divided by the distance, with ``X = cos E - e`` and
        ``Y = sqrt(1 - e^2) sin E`` as in ``TIBasisModel.forward_au``. The
        generic implementation would solve Kepler's equation once per TI
        constant.

        Args:
            theta: Chart point (its TI constants are ignored). Shape ``(7,)``.
            times: Epochs (days). Shape ``(K,)``.

        Returns:
            Tuple ``(H_ra, H_dec)``, each shape ``(K, 4)``.
        """
        e, ln_period, tau = theta[0], theta[1], theta[2]
        orbit = FrameOrbit.from_unit_plane_phase(e, ln_period, tau, self.model.t_ref)
        xy, _ = orbit.perifocal_coordinates(times)
        x, y = xy[..., 0] / self.dist_pc, xy[..., 1] / self.dist_pc
        zero = jnp.zeros_like(x)
        return (
            jnp.stack([x, y, zero, zero], axis=1),
            jnp.stack([zero, zero, x, y], axis=1),
        )

    def linear_prior_scale(self):
        """The TI constants' prior scale ``s_ti`` (AU); the window is the indicator."""
        return self.model.s_ti


def _split_loglike(ra, dec, records, select):
    """Astrometric log likelihood outside and inside ``select`` (two scalars)."""

    def part(mask):
        masked = eqx.tree_at(lambda r: r.is_valid, records, records.is_valid & mask)
        return loglike_relative_astrom(ra, dec, masked)

    return part(~select), part(select)


def _linear_gaussian(target, theta, records, weights):
    """Conditional Gaussian of a target's linear block and its log marginal.

    With design matrices ``H_ra``, ``H_dec`` (``predict = H c``), per-record
    precision ``w_k Sigma_k^-1`` (``Sigma_k`` the record's 2 x 2 error
    covariance, ``w_k`` its tempering weight) and prior ``N(0, s^2 I)``:
    ``P = I / s^2 + sum_k w_k H_k^T Sigma_k^-1 H_k``,
    ``b = sum_k w_k H_k^T Sigma_k^-1 d_k``, conditional mean ``m = P^-1 b`` and
    covariance ``P^-1``, and the log of the tempered marginal likelihood with
    ``c`` integrated out, ``-log det(P) / 2 + b^T m / 2``, up to terms that do
    not depend on the nonlinear coordinates (they cancel in a Metropolis
    ratio at a fixed temperature).

    Args:
        target: Target with a linear block.
        theta: Chart point (its linear coordinates are ignored). Shape ``(d,)``.
        records: Record buffer, ``K`` slots.
        weights: Tempering weight of each slot (0 for an invalid slot). Shape
            ``(K,)``.

    Returns:
        Tuple ``(mean, chol, log_marginal, h_ra, h_dec)``: ``chol`` is the
        lower Cholesky factor of ``P``.
    """
    h_ra, h_dec = target.linear_design(theta, records.times)
    live = weights > 0.0
    sx = jnp.where(live, records.ra_err, 1.0)
    sy = jnp.where(live, records.dec_err, 1.0)
    rho = jnp.where(live, records.corr, 0.0)
    scale = weights / jnp.clip(1.0 - rho**2, 1e-30)
    pxx = scale / sx**2
    pyy = scale / sy**2
    pxy = -scale * rho / (sx * sy)
    ra = jnp.where(live, records.ra, 0.0)
    dec = jnp.where(live, records.dec, 0.0)
    a = (
        h_ra.T @ (pxx[:, None] * h_ra)
        + h_ra.T @ (pxy[:, None] * h_dec)
        + h_dec.T @ (pxy[:, None] * h_ra)
        + h_dec.T @ (pyy[:, None] * h_dec)
    )
    b = h_ra.T @ (pxx * ra + pxy * dec) + h_dec.T @ (pxy * ra + pyy * dec)
    p = len(target.linear_index)
    precision = a + jnp.eye(p) / target.linear_prior_scale() ** 2
    chol = jnp.linalg.cholesky(precision)
    mean = jax.scipy.linalg.cho_solve((chol, True), b)
    log_marginal = -jnp.sum(jnp.log(jnp.diag(chol))) + 0.5 * b @ mean
    return mean, chol, log_marginal, h_ra, h_dec


def _rest_log_prior(target, theta):
    """``log p_rest``: the target's log prior without its Gaussian linear block."""
    c = theta[jnp.asarray(target.linear_index)]
    return (
        target.log_prior(theta) + 0.5 * jnp.sum(c**2) / target.linear_prior_scale() ** 2
    )
