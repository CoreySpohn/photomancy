"""Exact few-epoch fitters: geometric reductions as measure-correct samplers.

Short astrometric arcs (1-3 epochs) leave the orbit posterior wide and
multi-modal, but they have exact structure: conditioning on the observed sky
positions of one or two epochs reduces the unknowns to line-of-sight depths
(plus a velocity for the single-epoch case), and per discrete solution family
the map from those latents to orbit parameters is a smooth change of
variables. Sampling the latents and weighting each particle by::

    prior(theta) * |det d(theta)/d(latents)| * like(other epochs) / proposal

is exact importance sampling of the declared-prior posterior: the only
approximation is Monte Carlo error, measured by the effective sample size.
Two ingredients make the weights well defined:

- the conditioned epochs' noise-free positions are themselves latent and are
  proposed from the Gaussian likelihood kernel around the measurements, which
  cancels those epochs' likelihood factors exactly;
- the Jacobian is computed by forward-mode autodiff through the transform --
  for two conditioned epochs, through the implicit-function gradients of the
  Lambert boundary-value solver (``orbix.equations.lambert_solve``), whose
  discrete (revolutions, way, branch) families enumerate every period alias.

The declared model is the independent Thiele-Innes basis (the four TI
constants are free Gaussians, so stellar mass is implied rather than fixed;
cf. Octofitter and O'Neil et al. 2019), restricted by an explicit
implied-mass window: without the mass restriction the TI basis has
unbounded-mass tails that no state-space sampler can cover. The basis is
self-consistent within this module (``TIBasisModel.forward_au`` defines it);
no element-space conversion is provided here. The coefficients
(east_X, east_Y) multiply the east axis, and (north_X, north_Y) the north
axis, with X = cos(E) - e and Y = sqrt(1-e^2) sin(E). Unlike the scaled
F and G returned by ``thiele_innes_fit``, the Y coefficients here exclude
sqrt(1-e^2).

The state -> theta map is two-to-one (reflection through the sky plane gives
the classic Thiele-Innes degeneracy); the multiplicity is constant almost
everywhere, so it cancels in the self-normalized weights.
"""

import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import G, Msun2kg, two_pi
from orbix.equations import lambert_solve
from orbix.system import (
    FrameOrbit,
    ThieleInnesChart,
    line_of_sight_branch,
    sky_block_geometry,
)

from photomancy.orbit._frame import plane_from_phase, sky_offsets_arcsec
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.thiele_innes import TI_CHART_PARAM_NAMES
from photomancy.posterior import SamplePosterior

MU_SUN = G * Msun2kg  # AU^3 / day^2

TI_PARAM_NAMES = ("e", "lnP", "tau", "east_X", "east_Y", "north_X", "north_Y")


class TIBasisModel(eqx.Module):
    """Independent Thiele-Innes orbit model with an implied-mass window.

    Parameters are ``theta = (e, lnP, tau, east_X, east_Y, north_X, north_Y)``
    with priors ``e ~ U(0, e_max)``, ``lnP ~ U(lnP_lo, lnP_hi)`` (P in days),
    ``tau ~ U(0, 1)`` (periastron phase relative to ``t_ref``), and each TI
    constant ``~ N(0, s_ti^2)`` (AU), restricted to
    ``mu_lo <= (2 pi / P)^2 a(TI)^3 <= mu_hi``.

    Args:
        s_ti: Gaussian prior scale of each TI constant (AU).
        e_max: Upper eccentricity bound.
        lnP_lo: Lower bound of ln(period / day).
        lnP_hi: Upper bound of ln(period / day).
        mu_lo: Lower implied gravitational parameter (AU^3/day^2).
        mu_hi: Upper implied gravitational parameter (AU^3/day^2).
        t_ref: Reference epoch (days) for the periastron phase ``tau``.
    """

    s_ti: float = 2.0
    e_max: float = 0.9
    lnP_lo: float = math.log(50.0)
    lnP_hi: float = math.log(5000.0)
    mu_lo: float = MU_SUN / 3.0
    mu_hi: float = 3.0 * MU_SUN
    t_ref: float = 0.0

    def forward_au(self, theta, times):
        """Return (east, north) in AU at ``times`` (days), each shape (N,).

        East = east_X * X + east_Y * Y; north = north_X * X + north_Y * Y,
        with X = cos(E) - e and Y = sqrt(1-e^2) sin(E).
        """
        e, lnP, tau = theta[0], theta[1], theta[2]
        xy, _ = plane_from_phase(e, lnP, tau, self.t_ref).perifocal_coordinates(times)
        X, Y = xy[..., 0], xy[..., 1]
        east_X, east_Y, north_X, north_Y = theta[3:]
        east = east_X * X + east_Y * Y
        north = north_X * X + north_Y * Y
        return east, north

    def log_prior_ti(self, theta):
        """Log prior density up to a constant (the Gaussian TI block)."""
        return -0.5 * jnp.sum(theta[3:] ** 2) / self.s_ti**2

    def in_support(self, theta):
        """Whether theta satisfies the bounds and the implied-mass window."""
        e, lnP = theta[0], theta[1]
        east_x, east_y, north_x, north_y = theta[3:]
        a, _ = sky_block_geometry(north_x, east_x, north_y, east_y)
        mu = (two_pi / jnp.exp(lnP)) ** 2 * a**3
        return (
            (e > 0.0)
            & (e < self.e_max)
            & (lnP > self.lnP_lo)
            & (lnP < self.lnP_hi)
            & (mu >= self.mu_lo)
            & (mu <= self.mu_hi)
            & jnp.all(jnp.isfinite(theta))
        )

    def sample_prior(self, key, n):
        """Draw ``n`` prior samples (rejection into the mass window)."""
        seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        out = []
        got = 0
        while got < n:
            m = 4 * (n - got) + 64
            cand = np.column_stack(
                [
                    rng.uniform(1e-6, self.e_max, m),
                    rng.uniform(self.lnP_lo, self.lnP_hi, m),
                    rng.uniform(0.0, 1.0, m),
                    self.s_ti * rng.standard_normal((m, 4)),
                ]
            )
            keep = np.asarray(jax.vmap(self.in_support)(jnp.asarray(cand)), dtype=bool)
            cand = cand[keep]
            out.append(cand)
            got += cand.shape[0]
        return jnp.asarray(np.concatenate(out)[:n])


def lambert_families(n_max_rev):
    """(N, long_way, high_branch) triples covering every elliptic family."""
    fams = [(0, False, False), (0, True, False)]
    for n in range(1, n_max_rev + 1):
        for way in (False, True):
            for high in (False, True):
                fams.append((n, way, high))
    arr = np.array(fams)
    return (
        jnp.asarray(arr[:, 0], dtype=jnp.int32),
        jnp.asarray(arr[:, 1], dtype=bool),
        jnp.asarray(arr[:, 2], dtype=bool),
    )


def _mask_epochs(data, drop_idx):
    """Copy of ``data`` with the given epoch indices marked invalid."""
    mask = np.asarray(data.is_valid, dtype=bool).copy()
    mask[list(drop_idx)] = False
    return eqx.tree_at(lambda d: d.is_valid, data, jnp.asarray(mask))


def _conditioned_pair(data, idx):
    """Resolve and validate the two conditioned epoch indices."""
    valid = np.flatnonzero(np.asarray(data.is_valid, dtype=bool))
    if idx is None:
        i1, i2 = int(valid[0]), int(valid[-1])
    else:
        i1, i2 = int(idx[0]), int(idx[1])
    if i1 == i2 or i1 not in valid or i2 not in valid:
        raise ValueError(f"need two distinct valid epochs, got {(i1, i2)}")
    for i in (i1, i2):
        if abs(float(data.corr[i])) > 1e-12:
            raise ValueError(
                "conditioned epochs must have zero RA/DEC correlation "
                "(the proposal cancellation assumes a diagonal kernel)"
            )
    return i1, i2


_TI_CHART = ThieleInnesChart()


def chart_in_support(model, x):
    """Whether chart point ``x`` satisfies the model's bounds and implied-mass window.

    ``x = (ln n, ux, uy, A, B, F, G)`` as in ``ThieleInnesChart``; the period is
    ``2 pi / n`` and the eccentricity ``|u| / sqrt(1 + |u|**2)``.
    """
    usq = jnp.sum(x[1:3] ** 2)
    e = jnp.sqrt(usq / (1.0 + usq))
    ln_period = jnp.log(two_pi) - x[0]
    mu = _TI_CHART.implied_mu(x)
    return (
        (e > 0.0)
        & (e < model.e_max)
        & (ln_period > model.lnP_lo)
        & (ln_period < model.lnP_hi)
        & (mu >= model.mu_lo)
        & (mu <= model.mu_hi)
        & jnp.all(jnp.isfinite(x))
    )


def chart_log_prior(model, x):
    """Model log density against the chart reference measure times ``d ln mu``.

    Flat in ``e``, ``ln P`` and the phase, Gaussian in the four constants, up to a
    constant, and ``-inf`` outside ``chart_in_support``. Against that measure the
    density is ``exp(-|C|**2 / (2 s**2)) J / e`` with ``J = a**4 (1 - cos(i)**2)``
    the determinant from the constants' angles and size. Add
    ``ThieleInnesChart.log_volume`` for the density against the chart's own
    differential.
    """
    usq = jnp.sum(x[1:3] ** 2)
    e = jnp.sqrt(usq / (1.0 + usq))
    a, gap = sky_block_geometry(x[3], x[4], x[5], x[6])
    log_density = (
        -0.5 * jnp.sum(x[3:] ** 2) / model.s_ti**2
        + 2.0 * jnp.log(a)
        + jnp.log(gap)
        - jnp.log(jnp.where(e > 0.0, e, 1.0))
    )
    return jnp.where(chart_in_support(model, x), log_density, -jnp.inf)


def _chart_parts(model, x, orbit, ok, ln_jac, ln_like):
    """Mask invalid lanes to a zero contribution and report the sheet."""
    ln_prior = chart_log_prior(model, x) + _TI_CHART.log_volume(x)
    return (
        x,
        jnp.where(ok, ln_prior, -jnp.inf),
        jnp.where(ok, ln_jac, 0.0),
        jnp.where(ok, ln_like, 0.0),
        line_of_sight_branch(orbit),
    )


@partial(jax.jit, static_argnames=("n_particles",))
def _chart_lambert_batch(
    key,
    obs4,
    sig4,
    t1,
    t2,
    data_other,
    dist_pc,
    model,
    z_max,
    fam_N,
    fam_way,
    fam_high,
    n_particles,
):
    """Chart-coordinate weighted particles for all Lambert families, (F, n, ...)."""
    tof = t2 - t1

    def orbit_of_u(u, N, way, high):
        v1, _, _ = lambert_solve(u[0:3], u[3:6], tof, jnp.exp(u[6]), N, way, high)
        return FrameOrbit.from_cartesian(
            u[0:3], v1, mu_AU3_per_d2=jnp.exp(u[6]), epoch_d=t1
        )

    def one(k, N, way, high):
        kz, km, kx = jax.random.split(k, 3)
        z12 = jax.random.uniform(kz, (2,), minval=-z_max, maxval=z_max)
        lnmu = jax.random.uniform(
            km, (), minval=jnp.log(model.mu_lo), maxval=jnp.log(model.mu_hi)
        )
        xy = obs4 + sig4 * jax.random.normal(kx, (4,))
        u = jnp.array([xy[0], xy[1], z12[0], xy[2], xy[3], z12[1], lnmu])

        def fn(uu):
            return _TI_CHART.from_orbit(orbit_of_u(uu, N, way, high))

        orbit = orbit_of_u(u, N, way, high)
        x = _TI_CHART.from_orbit(orbit)
        _, ln_jac = jnp.linalg.slogdet(jax.jacfwd(fn)(u))
        _, _, ok_lam = lambert_solve(u[0:3], u[3:6], tof, jnp.exp(u[6]), N, way, high)
        ra, dec = sky_offsets_arcsec(orbit, data_other.times, dist_pc)
        ln_like = loglike_relative_astrom(ra, dec, data_other)
        ok = (
            ok_lam
            & chart_in_support(model, x)
            & jnp.isfinite(ln_jac)
            & jnp.isfinite(ln_like)
        )
        return _chart_parts(model, x, orbit, ok, ln_jac, ln_like)

    keys = jax.random.split(key, fam_N.shape[0] * n_particles)
    keys = keys.reshape((fam_N.shape[0], n_particles, *keys.shape[1:]))
    per_family = jax.vmap(
        lambda ks, N, way, high: jax.vmap(lambda k: one(k, N, way, high))(ks)
    )
    return per_family(keys, fam_N, fam_way, fam_high)


@partial(jax.jit, static_argnames=("n_particles",))
def _chart_ar_batch(
    key, obs2, sig2, t1, data_other, dist_pc, model, z_max, v_max, n_particles
):
    """Chart-coordinate weighted particles of the single-epoch reduction."""

    def orbit_of_u(u):
        return FrameOrbit.from_cartesian(
            u[0:3], u[3:6], mu_AU3_per_d2=jnp.exp(u[6]), epoch_d=t1
        )

    def one(k):
        kz, kv, km, kx = jax.random.split(k, 4)
        z = jax.random.uniform(kz, (), minval=-z_max, maxval=z_max)
        vel = jax.random.uniform(kv, (3,), minval=-v_max, maxval=v_max)
        lnmu = jax.random.uniform(
            km, (), minval=jnp.log(model.mu_lo), maxval=jnp.log(model.mu_hi)
        )
        xy = obs2 + sig2 * jax.random.normal(kx, (2,))
        u = jnp.array([xy[0], xy[1], z, vel[0], vel[1], vel[2], lnmu])
        orbit = orbit_of_u(u)
        x = _TI_CHART.from_orbit(orbit)
        _, ln_jac = jnp.linalg.slogdet(
            jax.jacfwd(lambda uu: _TI_CHART.from_orbit(orbit_of_u(uu)))(u)
        )
        ra, dec = sky_offsets_arcsec(orbit, data_other.times, dist_pc)
        ln_like = loglike_relative_astrom(ra, dec, data_other)
        ok = chart_in_support(model, x) & jnp.isfinite(ln_jac) & jnp.isfinite(ln_like)
        return _chart_parts(model, x, orbit, ok, ln_jac, ln_like)

    return jax.vmap(one)(jax.random.split(key, n_particles))


def _assemble_chart(x, lp, lj, ll, branch, return_parts, epoch):
    """Flatten family axes and build the chart-coordinate SamplePosterior (+ parts)."""
    x = x.reshape(-1, 7)
    lp, lj, ll, branch = (v.reshape(-1) for v in (lp, lj, ll, branch))
    post = SamplePosterior(
        samples=x,
        log_weights=lp + lj + ll,
        evidence=jnp.nan,
        param_names=TI_CHART_PARAM_NAMES,
        chart=type(_TI_CHART).__name__,
        epoch=epoch,
    )
    if not return_parts:
        return post
    return post, {"ln_prior": lp, "ln_jac": lj, "ln_like": ll, "branch": branch}


def lambert_depth_fit(
    data,
    *,
    dist_pc,
    model,
    key,
    idx=None,
    n_per_family=8000,
    n_max_rev=6,
    z_max_au=8.0,
    return_parts=False,
    epoch=None,
):
    """Exact two-epoch Lambert fit weighted in ``ThieleInnesChart`` coordinates.

    Latents (a depth along the line of sight at each conditioned epoch, over the
    Lambert families) are mapped through ``FrameOrbit.from_cartesian`` to chart
    coordinates ``(ln n, ux, uy, A, B, F, G)`` (``TI_CHART_PARAM_NAMES``), the
    constants on the conditioned epoch's mean-longitude axes. Weights are
    ``chart_log_prior + chart log volume + |det dx/du| + likelihood``.

    Args:
        data: ``RelativeAstromData`` with at least two valid epochs.
        dist_pc: Distance (parsec).
        model: A ``TIBasisModel`` carrying the prior bounds.
        key: JAX PRNG key.
        idx: The two conditioned epoch indices (default: the first two valid).
        n_per_family: Latent draws per Lambert family.
        n_max_rev: Maximum number of revolutions covered.
        z_max_au: Depth bound (AU) of the line-of-sight proposal.
        return_parts: Also return the weight parts.
        epoch: The :class:`~hwoutils.epoch.Epoch` that the data times are measured
            from.

    Returns:
        A ``SamplePosterior`` over the chart coordinates, declaring the chart
        ``ThieleInnesChart`` and ``epoch``; with ``return_parts``, a
        ``(posterior, parts)`` tuple whose parts hold ``ln_prior`` (density against
        the chart differential), ``ln_jac``, ``ln_like`` and each particle's
        line-of-sight ``branch`` (+1 or -1).
    """
    i1, i2 = _conditioned_pair(data, idx)
    obs4 = jnp.array([data.ra[i1], data.dec[i1], data.ra[i2], data.dec[i2]]) * dist_pc
    sig4 = (
        jnp.array(
            [data.ra_err[i1], data.dec_err[i1], data.ra_err[i2], data.dec_err[i2]]
        )
        * dist_pc
    )
    fam_N, fam_way, fam_high = lambert_families(n_max_rev)
    parts = _chart_lambert_batch(
        key,
        obs4,
        sig4,
        data.times[i1],
        data.times[i2],
        _mask_epochs(data, (i1, i2)),
        dist_pc,
        model,
        z_max_au,
        fam_N,
        fam_way,
        fam_high,
        n_per_family,
    )
    return _assemble_chart(*parts, return_parts, epoch)


def admissible_region_fit(
    data,
    *,
    dist_pc,
    model,
    key,
    idx=None,
    n_particles=100_000,
    z_max_au=8.0,
    return_parts=False,
    epoch=None,
):
    """Exact one-epoch admissible-region fit in ``ThieleInnesChart`` coordinates.

    Args as ``lambert_depth_fit`` with one conditioned epoch ``idx`` (an int) and
    ``n_particles`` latent draws (velocity bound from the model's mass window);
    returns as ``lambert_depth_fit``.
    """
    valid = np.flatnonzero(np.asarray(data.is_valid, dtype=bool))
    i1 = int(valid[0]) if idx is None else int(idx)
    if i1 not in valid:
        raise ValueError(f"epoch {i1} is not valid")
    if abs(float(data.corr[i1])) > 1e-12:
        raise ValueError("conditioned epoch must have zero RA/DEC correlation")
    obs2 = jnp.array([data.ra[i1], data.dec[i1]]) * dist_pc
    sig2 = jnp.array([data.ra_err[i1], data.dec_err[i1]]) * dist_pc
    rho = float(jnp.linalg.norm(obs2))
    r_floor = max(rho - 5.0 * float(sig2.max()), 1e-3)
    v_max = float(jnp.sqrt(2.0 * model.mu_hi / r_floor))
    parts = _chart_ar_batch(
        key,
        obs2,
        sig2,
        data.times[i1],
        _mask_epochs(data, (i1,)),
        dist_pc,
        model,
        z_max_au,
        v_max,
        n_particles,
    )
    return _assemble_chart(*parts, return_parts, epoch)
