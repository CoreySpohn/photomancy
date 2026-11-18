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


def state_to_theta(r, v, mu, t, t_ref):
    """Convert a bound (east, north, away) state to TI-basis parameters.

    The perifocal unit vectors come from the Laplace (eccentricity) vector
    and the angular momentum, so no node/periapsis angle conventions enter;
    ``r[0]`` is east and ``r[1]`` north, matching ``TIBasisModel.forward_au``.
    Positions and velocities use the right-handed (east, north, away) frame;
    dz/dt is positive when receding. Unbound or circular states produce
    non-finite outputs, to be masked by the caller.

    Args:
        r: Position (AU), shape (3,).
        v: Velocity (AU/day), shape (3,).
        mu: Gravitational parameter (AU^3/day^2).
        t: Epoch of the state (days).
        t_ref: Reference epoch for the periastron phase.

    Returns:
        theta = (e, lnP, tau, east_X, east_Y, north_X, north_Y), shape (7,).
    """
    orbit = FrameOrbit.from_cartesian(r, v, mu_AU3_per_d2=mu, epoch_d=t)
    e, P = orbit.e, orbit.period_d
    basis = orbit.plane_basis()
    c, s = orbit.ex / e, orbit.ey / e
    p = basis[:, 0] * c + basis[:, 1] * s
    q = -basis[:, 0] * s + basis[:, 1] * c
    xy, _ = orbit.perifocal_coordinates(t)
    cosE = xy[0] + e
    sinE = xy[1] / jnp.sqrt(1.0 - e**2)
    E = jnp.arctan2(sinE, cosE)
    M = E - e * sinE
    tau = jnp.mod((t - t_ref) / P - M / two_pi, 1.0)
    return jnp.array(
        [
            e,
            jnp.log(P),
            tau,
            p[0],
            q[0],
            p[1],
            q[1],
        ]
    )


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


def _finalize_parts(theta, ok, ln_prior, ln_jac, ln_like):
    """Mask invalid lanes: prior -inf carries the mask, other parts zeroed."""
    neg = -jnp.inf
    return (
        theta,
        jnp.where(ok, ln_prior, neg),
        jnp.where(ok, ln_jac, 0.0),
        jnp.where(ok, ln_like, 0.0),
    )


@partial(jax.jit, static_argnames=("n_particles",))
def _lambert_batch(
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
    """Weighted particles for all Lambert families, shape (F, n, ...)."""
    tof = t2 - t1

    def theta_of_u(u, N, way, high):
        r1 = u[0:3]
        r2 = u[3:6]
        mu = jnp.exp(u[6])
        v1, _, _ = lambert_solve(r1, r2, tof, mu, N, way, high)
        return state_to_theta(r1, v1, mu, t1, model.t_ref)

    def one(k, N, way, high):
        kz, km, kx = jax.random.split(k, 3)
        z12 = jax.random.uniform(kz, (2,), minval=-z_max, maxval=z_max)
        lnmu = jax.random.uniform(
            km, (), minval=jnp.log(model.mu_lo), maxval=jnp.log(model.mu_hi)
        )
        xy = obs4 + sig4 * jax.random.normal(kx, (4,))
        u = jnp.array([xy[0], xy[1], z12[0], xy[2], xy[3], z12[1], lnmu])

        def fn(uu):
            return theta_of_u(uu, N, way, high)

        theta = fn(u)
        _, ln_jac = jnp.linalg.slogdet(jax.jacfwd(fn)(u))
        _, _, ok_lam = lambert_solve(u[0:3], u[3:6], tof, jnp.exp(u[6]), N, way, high)
        x_au, y_au = model.forward_au(theta, data_other.times)
        ln_like = loglike_relative_astrom(x_au / dist_pc, y_au / dist_pc, data_other)
        ok = (
            ok_lam
            & model.in_support(theta)
            & jnp.isfinite(ln_jac)
            & jnp.isfinite(ln_like)
        )
        return _finalize_parts(theta, ok, model.log_prior_ti(theta), ln_jac, ln_like)

    keys = jax.random.split(key, fam_N.shape[0] * n_particles)
    keys = keys.reshape((fam_N.shape[0], n_particles, *keys.shape[1:]))
    per_family = jax.vmap(
        lambda ks, N, way, high: jax.vmap(lambda k: one(k, N, way, high))(ks)
    )
    return per_family(keys, fam_N, fam_way, fam_high)


@partial(jax.jit, static_argnames=("n_particles",))
def _ar_batch(
    key, obs2, sig2, t1, data_other, dist_pc, model, z_max, v_max, n_particles
):
    """Weighted particles for the conditioned-single-epoch (AR) reduction."""

    def theta_of_u(u):
        r = u[0:3]
        v = u[3:6]
        return state_to_theta(r, v, jnp.exp(u[6]), t1, model.t_ref)

    def one(k):
        kz, kv, km, kx = jax.random.split(k, 4)
        z = jax.random.uniform(kz, (), minval=-z_max, maxval=z_max)
        vel = jax.random.uniform(kv, (3,), minval=-v_max, maxval=v_max)
        lnmu = jax.random.uniform(
            km, (), minval=jnp.log(model.mu_lo), maxval=jnp.log(model.mu_hi)
        )
        xy = obs2 + sig2 * jax.random.normal(kx, (2,))
        u = jnp.array([xy[0], xy[1], z, vel[0], vel[1], vel[2], lnmu])
        theta = theta_of_u(u)
        _, ln_jac = jnp.linalg.slogdet(jax.jacfwd(theta_of_u)(u))
        x_au, y_au = model.forward_au(theta, data_other.times)
        ln_like = loglike_relative_astrom(x_au / dist_pc, y_au / dist_pc, data_other)
        ok = model.in_support(theta) & jnp.isfinite(ln_jac) & jnp.isfinite(ln_like)
        return _finalize_parts(theta, ok, model.log_prior_ti(theta), ln_jac, ln_like)

    return jax.vmap(one)(jax.random.split(key, n_particles))


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


def _assemble(theta, lp, lj, ll, return_parts):
    """Flatten family axes and build the SamplePosterior (+ parts)."""
    theta = theta.reshape(-1, 7)
    lp, lj, ll = (x.reshape(-1) for x in (lp, lj, ll))
    post = SamplePosterior(
        samples=theta,
        log_weights=lp + lj + ll,
        evidence=jnp.nan,
        param_names=TI_PARAM_NAMES,
    )
    if not return_parts:
        return post
    return post, {"ln_prior": lp, "ln_jac": lj, "ln_like": ll}


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
):
    """Exact two-epoch-conditioned fit via Lambert depth search.

    Conditions on two epochs' sky positions, samples their line-of-sight
    depths and the implied gravitational parameter, solves the Lambert
    boundary-value problem per (revolutions, way, branch) family, and weights
    by prior x ``|det J|`` x likelihood of the remaining epochs. Exact for the
    ``TIBasisModel`` posterior up to Monte Carlo error.

    Args:
        data: ``RelativeAstromData`` (arcsec); conditioned epochs need
            ``corr = 0``.
        dist_pc: Distance (pc) converting arcsec to AU.
        model: The declared ``TIBasisModel``.
        key: PRNG key.
        idx: The two conditioned epoch indices (default: first and last
            valid).
        n_per_family: Particles per Lambert family.
        n_max_rev: Highest revolution count enumerated; families above the
            per-family minimum TOF are flagged invalid internally, so this
            only needs to be at least ``baseline / P_min``.
        z_max_au: Half width of the uniform depth proposal (AU).
        return_parts: Also return the ``{ln_prior, ln_jac, ln_like}`` weight
            components (e.g. for calibration ablations).

    Returns:
        A ``SamplePosterior`` over ``(e, lnP, tau, east_X, east_Y, north_X, north_Y)``;
        with ``return_parts``, a ``(posterior, parts)`` tuple.
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
    theta, lp, lj, ll = _lambert_batch(
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
    return _assemble(theta, lp, lj, ll, return_parts)


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
):
    """Exact single-epoch-conditioned fit (admissible-region reduction).

    Conditions on one epoch's sky position and samples the remaining state
    coordinates (depth, velocity, implied gravitational parameter); the
    velocity proposal box covers every mass-window-bound orbit through the
    conditioned point. Remaining epochs enter through the likelihood, so this
    also serves as a generic n-epoch fitter (at reduced efficiency compared
    to conditioning on two epochs).

    Args: as :func:`lambert_depth_fit`, with ``idx`` a single epoch index
        (default: first valid) and ``n_particles`` the total particle count.

    Returns:
        A ``SamplePosterior`` over ``(e, lnP, tau, east_X, east_Y, north_X, north_Y)``;
        with ``return_parts``, a ``(posterior, parts)`` tuple.
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
    theta, lp, lj, ll = _ar_batch(
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
    return _assemble(theta, lp, lj, ll, return_parts)


# ---------------------------------------------------------------------------
# Chart route: ThieleInnesChart coordinates, extended by d ln mu
# ---------------------------------------------------------------------------

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
    """Mask invalid lanes as ``_finalize_parts`` does, and report the sheet."""
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


def chart_lambert_depth_fit(
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
    """``lambert_depth_fit`` weighted in ``ThieleInnesChart`` coordinates.

    The latents, families and likelihood are those of ``lambert_depth_fit``; the
    state is mapped through ``FrameOrbit.from_cartesian`` to chart coordinates
    ``(ln n, ux, uy, A, B, F, G)`` (``TI_CHART_PARAM_NAMES``), the constants on the
    conditioned epoch's mean-longitude axes. Weights are
    ``chart_log_prior + chart log volume + |det dx/du| + likelihood``. At identical
    latents this equals the ``lambert_depth_fit`` weight plus one constant.

    Args: as ``lambert_depth_fit``, plus ``epoch``, the
        :class:`~hwoutils.epoch.Epoch` that the data times are measured from.

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


def chart_admissible_region_fit(
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
    """``admissible_region_fit`` weighted in ``ThieleInnesChart`` coordinates.

    Args: as ``admissible_region_fit``, plus ``epoch`` as
    ``chart_lambert_depth_fit``; returns as ``chart_lambert_depth_fit``.
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
