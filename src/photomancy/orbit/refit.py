"""Stateless importance refit of one planet's orbit from its record buffer.

A refit reads the records of a planet (the whole buffer, nothing carried
between calls) and returns an equal-weight cloud of physical orbit elements
with the Kish effective size of the importance sample behind it. The effective
size is the status: a refit that cannot reach ``ess_floor`` says so and the
caller decides what to do. Everything here is pure JAX with fixed shapes, so a
refit runs under ``jax.vmap`` over planets.

The declared model is :class:`PhysicalPriorTarget`: the physical prior of the
fixed-mass orbit model (``ln P`` uniform, ``e ~ Beta(0.867, 3.03)``, ``cos i``
uniform, three uniform angles) with the stellar mass uniform in its logarithm
inside a window around the catalog mass. In the Thiele-Innes chart of
:class:`~photomancy.orbit.updater.ThieleInnesTarget` that density is the
eccentricity density times ``1 / (a^2 sqrt(u^2 - 4 v^2))``: the map
``(ln a, cos i, omega, W) -> (four constants)`` has Jacobian
``a^4 (1 - cos^2 i) = a^2 sqrt(u^2 - 4 v^2)`` with ``u`` the sum of the
squared constants and ``v`` their determinant, and it is two to one (the
reflection through the sky plane, which keeps the sky track and flips the
phase angle).

Two engines share the one model:

- :func:`scale_rotate_refit`: exact importance sampling by scale-and-rotate
  with the mass drawn from its window. Used for one record, where the
  proposal carries the prior exactly.
- :func:`collapsed_refit`: the four linear constants are drawn from their
  conditional Gaussian (the marginal likelihood is the weight) and the other
  three coordinates from adaptive rounds, a defensive mixture of the prior box
  and Gaussian kernels on points resampled from the previous round. The weight
  carries the physical prior over the Gaussian reference prior of the constants.

:func:`importance_refit` picks between them by the number of valid records.
Under ``vmap`` the choice is a ``lax.cond`` that evaluates both engines, so a
batch whose entries share an engine should call that engine directly.

The helpers ``_linear_gaussian`` and ``_rest_log_prior`` come from
:mod:`photomancy.orbit.updater` unchanged; the resampler, the collapsed draw and the
kernel-fit machinery are written here and not shared with the updater.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, two_pi
from jax.scipy.special import betainc, logsumexp
from jax.scipy.stats import beta as beta_dist
from orbix.equations.orbit import mean_motion

from photomancy.orbit.few_epoch import MU_SUN, TIBasisModel, ti_semimajor_axis
from photomancy.orbit.ofti import _scale_rotate_score
from photomancy.orbit.updater import (
    ThieleInnesTarget,
    UpdaterStatus,
    _linear_gaussian,
    _rest_log_prior,
)

ECC_ALPHA = 0.867
ECC_BETA = 3.03
ELEMENT_NAMES = (
    "ln_period",
    "e",
    "cos_i",
    "W",
    "omega",
    "tau",
    "ln_mass",
    "ln_lam",
)

_TINY = 1e-300
_TAU_IMAGES = jnp.arange(-2.0, 3.0)
_MAP_BATCH = 2**14
_DENSITY_BATCH = 1024


class PhysicalPriorTarget(ThieleInnesTarget):
    """Thiele-Innes chart carrying the physical prior with a stellar-mass window.

    The chart, the forward model and the linear block are those of
    :class:`~photomancy.orbit.updater.ThieleInnesTarget`. ``model.mu_lo`` and
    ``model.mu_hi`` (AU^3/day^2) are the mass window, so the mass is uniform in
    its logarithm between them; ``model.lnP_lo`` and ``model.lnP_hi`` bound the
    period and ``model.e_max`` the eccentricity. ``model.s_ti`` is only the scale
    of the Gaussian the collapsed draw uses as its reference for the constants.

    :meth:`log_prior` is the physical density in the chart, not the Gaussian
    product of the parent: it is ``Beta(0.867, 3.03)`` on ``e`` truncated to
    ``e_max`` minus ``log(a^2 sqrt(u^2 - 4 v^2))``, and ``-inf`` outside the
    model's support. The density that remains constant over the support is
    :meth:`log_prior_constant`.
    """

    def log_prior(self, theta):
        """Log physical density of the chart point, up to a constant.

        Args:
            theta: Chart point. Shape ``(7,)``.

        Returns:
            Scalar; ``-inf`` outside ``model.in_support``.
        """
        c = theta[3:]
        u = jnp.sum(c**2)
        v = c[0] * c[3] - c[2] * c[1]
        disc = jnp.sqrt(jnp.maximum(u**2 - 4.0 * v**2, _TINY))
        log_a2 = jnp.log((u + disc) / 2.0)
        e = jnp.clip(theta[0], _TINY, 1.0 - 1e-12)
        log_e = beta_dist.logpdf(e, ECC_ALPHA, ECC_BETA) - jnp.log(
            betainc(ECC_ALPHA, ECC_BETA, self.model.e_max)
        )
        return jnp.where(
            self.model.in_support(theta), log_e - log_a2 - jnp.log(disc), -jnp.inf
        )

    def log_prior_constant(self):
        """Log of the density factor :meth:`log_prior` leaves out.

        The period density ``1 / (lnP_hi - lnP_lo)``, the density of ``ln a``
        at a given period ``3 / ln(mu_hi / mu_lo)``, the angle densities
        ``1 / (2 pi)^2`` and the factor 1/2 of ``cos i`` against the two-to-one
        factor 2 of the map. With it the evidence is the marginal likelihood.
        """
        m = self.model
        return (
            -jnp.log(m.lnP_hi - m.lnP_lo)
            - jnp.log(jnp.log(m.mu_hi / m.mu_lo) / 3.0)
            - 2.0 * jnp.log(two_pi)
        )

    def mass_window_msun(self):
        """The mass window ``(low, high)`` in solar masses."""
        return self.model.mu_lo / MU_SUN, self.model.mu_hi / MU_SUN


def physical_prior_target(mass_msun, dist_pc, half_width=0.10):
    """A :class:`PhysicalPriorTarget` with the mass inside ``mass_msun (1 +/- w)``.

    Args:
        mass_msun: Catalog stellar mass (solar masses).
        dist_pc: Distance (parsec).
        half_width: Fractional half width of the mass window. Default 0.10.

    Returns:
        The target. The period range and eccentricity bound are the model's
        defaults (50 to 5000 d, 0.9).
    """
    mu = MU_SUN * mass_msun
    model = TIBasisModel(mu_lo=(1.0 - half_width) * mu, mu_hi=(1.0 + half_width) * mu)
    return PhysicalPriorTarget(model=model, dist_pc=dist_pc)


class RefitCloud(eqx.Module):
    """Equal-weight orbit cloud of a refit.

    Args:
        elements: Physical elements ``(ln P, e, cos_i, W, omega, tau, ln_mass,
            ln_lam)``, shape ``(n, 8)``: period in days, node and argument of
            periastron in radians, ``tau`` the periastron phase with the mean
            anomaly ``2 pi ((t - t_ref) / P - tau)`` and ``t_ref`` the model's
            reference epoch, the mass in solar masses. ``ln_lam`` is NaN until
            a photometric term fills it.
        log_evidence: Log marginal likelihood of the records under the declared
            model (data normalization included).
        ess: Kish effective size of the last importance round.
        status: An :class:`~photomancy.orbit.updater.UpdaterStatus` value:
            ``OK``, ``LOW_ESS`` (``ess`` under the floor) or ``NO_SUPPORT``.
        rounds: Importance rounds run.
        max_weight: Largest normalized importance weight of the last round (the
            weight holds ``1 / (1 - cos_i^2)``, so face-on orbits can dominate).
    """

    elements: jax.Array
    log_evidence: jax.Array
    ess: jax.Array
    status: jax.Array
    rounds: jax.Array
    max_weight: jax.Array


def ti_to_elements(theta, branch):
    """Physical elements of a Thiele-Innes chart point.

    The constants fix ``(a, cos i)`` and the sums ``omega + W`` and ``W - omega``
    modulo ``2 pi``, so ``(omega, W)`` is known up to a joint shift by ``pi``:
    the two mirror orbits, which share their sky track.

    Args:
        theta: Chart point ``(e, lnP, tau, east_X, east_Y, north_X, north_Y)``.
        branch: 0 or 1, the mirror orbit (adds ``pi`` to ``omega`` and ``W``).

    Returns:
        Elements in the order of ``ELEMENT_NAMES``, with ``ln_lam`` NaN.
    """
    east_x, east_y, north_x, north_y = theta[3], theta[4], theta[5], theta[6]
    r_plus = jnp.hypot(north_x + east_y, east_x - north_y)
    r_minus = jnp.hypot(north_x - east_y, east_x + north_y)
    sigma = jnp.arctan2(east_x - north_y, north_x + east_y)
    phi = jnp.arctan2(east_x + north_y, north_x - east_y)
    shift = jnp.pi * branch
    node = jnp.mod((sigma + phi) / 2.0 + shift, two_pi)
    omega = jnp.mod((sigma - phi) / 2.0 + shift, two_pi)
    a = ti_semimajor_axis(east_x, east_y, north_x, north_y)
    cos_i = (r_plus - r_minus) / (r_plus + r_minus)
    ln_mu = 2.0 * jnp.log(two_pi) - 2.0 * theta[1] + 3.0 * jnp.log(a)
    return jnp.stack(
        [
            theta[1],
            theta[0],
            cos_i,
            node,
            omega,
            theta[2],
            ln_mu - jnp.log(MU_SUN),
            jnp.nan * jnp.ones(()),
        ]
    )


def _kish(log_w):
    """Kish effective size of log weights (0 when none is finite)."""
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    return jnp.where(total > 0.0, total**2 / jnp.sum(w**2), 0.0)


def _max_weight(log_w):
    """Largest normalized weight of log weights (0 when none is finite)."""
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    return jnp.where(total > 0.0, 1.0 / jnp.where(total > 0.0, total, 1.0), 0.0)


def _status(ess, log_w, ess_floor):
    """Refit status from the last round."""
    no_support = ~jnp.isfinite(jnp.max(log_w))
    return jnp.where(
        no_support,
        UpdaterStatus.NO_SUPPORT,
        jnp.where(ess < ess_floor, UpdaterStatus.LOW_ESS, UpdaterStatus.OK),
    ).astype(jnp.int32)


def _systematic_indices(key, log_weights, n):
    """Systematic resampling indices for ``n`` picks from normalized log weights.

    A copy of the updater's helper, which clips to ``n - 1`` and so is correct
    only when the picks are as many as the weights.
    """
    cdf = jnp.cumsum(jnp.exp(log_weights))
    cdf = cdf / cdf[-1]
    u = (jax.random.uniform(key) + jnp.arange(n)) / n
    return jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, cdf.shape[0] - 1)


def _resample_elements(key, log_w, n_particles, to_elements):
    """Systematic resample to equal weights, mirror branch drawn per particle."""
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    w = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 1.0 / w.size)
    k_idx, k_branch = jax.random.split(key)
    idx = _systematic_indices(k_idx, jnp.log(jnp.maximum(w, _TINY)), n_particles)
    branch = jax.random.bernoulli(k_branch, 0.5, (n_particles,)).astype(float)
    return to_elements(idx, branch)


def _prior_box(target):
    """Lower and upper bounds of ``(e, ln P, tau)`` under the prior."""
    m = target.model
    lo = jnp.stack([jnp.zeros(()), m.lnP_lo, jnp.zeros(())])
    hi = jnp.stack([m.e_max, m.lnP_hi, jnp.ones(())])
    return lo, hi


def _kernel_fit(key, u, log_w, lo, hi, n_centers):
    """Resampled centers and the Cholesky factor of the kernel covariance.

    The covariance is the weighted covariance of ``u`` (``tau`` unwrapped around
    its circular mean) times the squared Silverman bandwidth, plus a floor of
    1e-3 of the box width.
    """
    peak = jnp.max(log_w)
    safe = jnp.where(jnp.isfinite(peak), peak, 0.0)
    w = jnp.exp(log_w - safe)
    total = jnp.sum(w)
    w = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 1.0 / w.size)
    angle = two_pi * u[:, 2]
    mean_tau = jnp.arctan2(jnp.sum(w * jnp.sin(angle)), jnp.sum(w * jnp.cos(angle)))
    mean_tau = mean_tau / two_pi
    x = u.at[:, 2].set(mean_tau + jnp.mod(u[:, 2] - mean_tau + 0.5, 1.0) - 0.5)
    mean = w @ x
    centered = x - mean
    cov = (centered * w[:, None]).T @ centered / (1.0 - jnp.sum(w**2))
    h = 0.969 * jnp.maximum(1.0 / jnp.sum(w**2), 4.0) ** (-1.0 / 7.0)
    floor = 1e-3 * (hi - lo)
    cov = h**2 * cov + jnp.diag(floor**2)
    grid = (jax.random.uniform(key) + jnp.arange(n_centers)) / n_centers
    idx = jnp.minimum(jnp.searchsorted(jnp.cumsum(w), grid), w.size - 1)
    return u[idx], jnp.linalg.cholesky(cov)


def _log_kernel_mixture(u, centers, chol):
    """Log density at each row of ``u`` of the equal-weight Gaussian kernel mixture.

    ``tau`` is periodic: the kernel images shifted by -2 to 2 are summed.
    Distances are taken in the whitened frame of the kernel covariance.
    """
    solve = jax.scipy.linalg.solve_triangular
    centers_w = solve(chol, centers.T, lower=True).T
    shift = solve(chol, jnp.array([0.0, 0.0, 1.0]), lower=True)
    log_norm = -1.5 * jnp.log(two_pi) - jnp.sum(jnp.log(jnp.diag(chol)))
    log_n = jnp.log(centers.shape[0])

    def one(block):
        y = solve(chol, block.T, lower=True).T
        d = y[:, None, :] - centers_w[None]
        dt = d @ shift
        base = jnp.sum(d**2, axis=-1)
        sq = (
            base[..., None]
            + 2.0 * dt[..., None] * _TAU_IMAGES
            + jnp.sum(shift**2) * _TAU_IMAGES**2
        )
        return logsumexp(-0.5 * sq, axis=(1, 2)) - log_n + log_norm

    blocks = u.reshape(-1, min(_DENSITY_BATCH, u.shape[0]), 3)
    return jax.lax.map(one, blocks).reshape(-1)


def _propose(key, centers, chol, lo, hi, n, defensive):
    """Mixture draws of ``(e, ln P, tau)`` and their log proposal density."""
    k = jax.random.split(key, 4)
    log_box = -jnp.sum(jnp.log(hi - lo))
    prior = lo + (hi - lo) * jax.random.uniform(k[0], (n, 3))
    pick = jax.random.randint(k[1], (n,), 0, centers.shape[0])
    kern = centers[pick] + jax.random.normal(k[2], (n, 3)) @ chol.T
    kern = kern.at[:, 2].set(jnp.mod(kern[:, 2], 1.0))
    u = jnp.where(jax.random.uniform(k[3], (n, 1)) < defensive, prior, kern)
    inside = jnp.all((u > lo) & (u < hi), axis=1)
    log_q = jnp.logaddexp(
        jnp.log(defensive) + jnp.where(inside, log_box, -jnp.inf),
        jnp.log1p(-defensive) + _log_kernel_mixture(u, centers, chol),
    )
    return u, log_q


def _collapsed_weights(target, records, u, z):
    """Constants from their conditional Gaussian and the log weight before proposal.

    Returns ``(theta, log_w)`` with ``log_w`` the log marginal likelihood of the
    constants plus ``log p_rest`` (the physical density over the reference
    Gaussian); the caller subtracts the log proposal density of ``u``.
    """
    weights = records.is_valid.astype(float)

    def one(args):
        ui, zi = args
        theta = jnp.concatenate([ui, jnp.zeros(4)])
        mean, chol, log_marginal, _, _ = _linear_gaussian(
            target, theta, records, weights
        )
        c = mean + jax.scipy.linalg.solve_triangular(chol.T, zi, lower=False)
        theta = theta.at[3:].set(c)
        lw = log_marginal + _rest_log_prior(target, theta)
        return theta, jnp.where(jnp.isfinite(lw), lw, -jnp.inf)

    return jax.lax.map(one, (u, z), batch_size=min(_MAP_BATCH, u.shape[0]))


def _evidence_constant(target, records):
    """Constants of the collapsed weight that make its mean the marginal likelihood.

    The data normalization with the zero-constants prediction, the
    ``(2 pi)^(p/2)`` of the conditional Gaussian and the prior constant.
    """
    theta0 = jnp.array([0.1, 0.5 * (target.model.lnP_lo + target.model.lnP_hi), 0.5])
    theta0 = jnp.concatenate([theta0, jnp.zeros(4)])
    return (
        target.log_likelihood(theta0, records)
        + 2.0 * jnp.log(two_pi)
        + target.log_prior_constant()
    )


@eqx.filter_jit
def collapsed_refit(
    target,
    records,
    key,
    *,
    n_particles,
    n_round=2**16,
    n_rounds=3,
    n_last=None,
    n_centers=256,
    defensive=0.1,
    ess_floor=400.0,
):
    """Collapsed importance refit with adaptive rounds (any number of records).

    Round one draws ``(e, ln P, tau)`` from the prior box; each later round
    from a mixture of the box (fraction ``defensive``) and Gaussian kernels on
    ``n_centers`` points resampled from the previous round. The four constants
    always come from their conditional Gaussian. Every round is an exact
    importance sample, so a poor proposal costs effective size and never
    correctness; only the last round is returned.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer with at least one valid record.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        n_round: Proposals per round (static).
        n_rounds: Rounds (static); 1 draws from the prior box only.
        n_last: Proposals of the last round (static); default ``n_round``. The
            Kish size is linear in it for a fixed proposal.
        n_centers: Kernels of the adaptive proposal (static).
        defensive: Fraction of an adaptive round drawn from the prior box.
        ess_floor: Kish size under which the status is ``LOW_ESS``.

    Returns:
        A :class:`RefitCloud`.
    """
    n_last = n_round if n_last is None else n_last
    for n in (n_round, n_last):
        if n > _DENSITY_BATCH and n % _DENSITY_BATCH:
            raise ValueError(
                f"proposals per round ({n}) must be a multiple of {_DENSITY_BATCH}"
            )
    sizes = [n_round] * (n_rounds - 1) + [n_last]
    lo, hi = _prior_box(target)
    keys = jax.random.split(key, 3 * n_rounds + 1)
    u = lo + (hi - lo) * jax.random.uniform(keys[0], (sizes[0], 3))
    log_q = jnp.full(sizes[0], -jnp.sum(jnp.log(hi - lo)))
    log_w = None
    for r in range(n_rounds):
        if r > 0:
            centers, chol = _kernel_fit(keys[3 * r], u, log_w, lo, hi, n_centers)
            u, log_q = _propose(
                keys[3 * r + 1], centers, chol, lo, hi, sizes[r], defensive
            )
        z = jax.random.normal(keys[3 * r + 2], (sizes[r], 4))
        theta, log_marginal = _collapsed_weights(target, records, u, z)
        log_w = jnp.where(jnp.isfinite(log_marginal), log_marginal - log_q, -jnp.inf)
    ess = _kish(log_w)
    log_evidence = (
        logsumexp(log_w) - jnp.log(sizes[-1]) + _evidence_constant(target, records)
    )
    elements = _resample_elements(
        keys[-1],
        log_w,
        n_particles,
        lambda idx, branch: jax.vmap(ti_to_elements)(theta[idx], branch),
    )
    return RefitCloud(
        elements=elements,
        log_evidence=jnp.where(jnp.isfinite(ess) & (ess > 0), log_evidence, -jnp.inf),
        ess=ess,
        status=_status(ess, log_w, ess_floor),
        rounds=jnp.asarray(n_rounds),
        max_weight=_max_weight(log_w),
    )


@eqx.filter_jit
def scale_rotate_draw(target, records, key, *, n_proposals):
    """Weighted draws by scale-and-rotate, the mass drawn from its window.

    The first valid record is the anchor: orbit shapes ``(e, cos i, omega,
    M_ref)`` come from their priors, the mass log-uniformly from the window and
    the anchor position from its error kernel; ``(a, W)`` are solved so the orbit
    passes through it, which induces the ``ln P`` and node priors exactly. The
    weight is the likelihood of the other valid records over the squared anchor
    separation, so the draw is exact for the physical prior at any number of
    records but is efficient only for one.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer with at least one valid record.
        key: PRNG key.
        n_proposals: Proposals (static).

    Returns:
        ``(elements, log_w, log_offset)``: elements ``(n, 8)`` of every proposal,
        its log weights (``-inf`` for a proposal outside the support), and the
        constant so that ``logsumexp(log_w) - log(n) - log_offset`` is the log
        evidence.
    """
    m = target.model
    n = n_proposals
    anchor = jnp.argmax(records.is_valid)
    k = jax.random.split(key, 8)
    e = jax.random.beta(k[0], ECC_ALPHA, ECC_BETA, (n,))
    cos_i = jax.random.uniform(k[1], (n,), minval=-1.0, maxval=1.0)
    omega = jax.random.uniform(k[2], (n,), maxval=two_pi)
    m_ref = jax.random.uniform(k[3], (n,), maxval=two_pi)
    ln_mu = jax.random.uniform(
        k[4], (n,), minval=jnp.log(m.mu_lo), maxval=jnp.log(m.mu_hi)
    )
    mu = jnp.exp(ln_mu)
    sx, sy, rho = records.ra_err[anchor], records.dec_err[anchor], records.corr[anchor]
    z1 = jax.random.normal(k[5], (n,))
    z2 = jax.random.normal(k[6], (n,))
    tx = records.ra[anchor] + sx * z1
    ty = records.dec[anchor] + sy * (
        rho * z1 + jnp.sqrt(jnp.maximum(1.0 - rho**2, 0.0)) * z2
    )
    t_anchor = records.times[anchor]
    others = eqx.tree_at(
        lambda r: r.is_valid, records, records.is_valid.at[anchor].set(False)
    )
    log10_lo, log10_hi = m.lnP_lo / jnp.log(10.0), m.lnP_hi / jnp.log(10.0)

    def one(ei, ci, wi, mi, mui, txi, tyi):
        return _scale_rotate_score(
            ei,
            ci,
            jnp.cos(wi),
            jnp.sin(wi),
            mi,
            txi,
            tyi,
            others,
            t_anchor,
            mui,
            mui / G,
            target.dist_pc,
            log10_lo,
            log10_hi,
            m.e_max,
        )

    rows, ll = jax.lax.map(
        lambda args: one(*args),
        (e, cos_i, omega, m_ref, mu, tx, ty),
        batch_size=min(_MAP_BATCH, n),
    )
    log_w = ll - 2.0 * jnp.log(jnp.hypot(tx, ty))
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    log_offset = (
        jnp.log((2.0 / 3.0) * (m.lnP_hi - m.lnP_lo))
        + jnp.log(two_pi)
        + jnp.log(betainc(ECC_ALPHA, ECC_BETA, m.e_max))
    )
    a, ec, ci, w_node, cw, sw, tp = rows.T
    period = two_pi / mean_motion(a, mu)
    elements = jnp.stack(
        [
            jnp.log(period),
            ec,
            ci,
            w_node,
            jnp.mod(jnp.arctan2(sw, cw), two_pi),
            jnp.mod((tp - m.t_ref) / period, 1.0),
            ln_mu - jnp.log(MU_SUN),
            jnp.full(n, jnp.nan),
        ],
        axis=1,
    )
    return elements, log_w, log_offset


@eqx.filter_jit
def scale_rotate_refit(
    target, records, key, *, n_particles, n_proposals=2**17, ess_floor=400.0
):
    """Exact importance refit by scale-and-rotate, the mass drawn from its window.

    See :func:`scale_rotate_draw` for the draw and its weights; the returned
    cloud is ``n_particles`` equal-weight resamples.

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer with at least one valid record.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        n_proposals: Proposals (static).
        ess_floor: Kish size under which the status is ``LOW_ESS``.

    Returns:
        A :class:`RefitCloud` with ``rounds`` 1.
    """
    k_draw, k_pick = jax.random.split(key)
    elements, log_w, log_offset = scale_rotate_draw(
        target, records, k_draw, n_proposals=n_proposals
    )
    ess = _kish(log_w)
    log_evidence = logsumexp(log_w) - jnp.log(n_proposals) - log_offset
    picked = _resample_elements(
        k_pick, log_w, n_particles, lambda idx, branch: elements[idx]
    )
    return RefitCloud(
        elements=picked,
        log_evidence=jnp.where(ess > 0, log_evidence, -jnp.inf),
        ess=ess,
        status=_status(ess, log_w, ess_floor),
        rounds=jnp.asarray(1),
        max_weight=_max_weight(log_w),
    )


def importance_refit(
    target,
    records,
    report,
    key,
    *,
    n_particles,
    n_round=2**16,
    n_rounds=3,
    n_last=None,
    n_centers=256,
    defensive=0.1,
    ess_floor=400.0,
):
    """Stateless importance refit of an orbit from its record buffer.

    One valid record runs :func:`scale_rotate_refit` with ``n_round`` proposals;
    two or more run :func:`collapsed_refit`. With concrete records the engine is
    chosen in Python; under a transformation it is a ``lax.cond`` (both engines
    run under ``vmap``).

    Args:
        target: A :class:`PhysicalPriorTarget`.
        records: ``RelativeAstromData`` buffer.
        report: Reserved for the photometric report term; must be ``None``.
        key: PRNG key.
        n_particles: Size of the returned cloud.
        n_round: Proposals per round.
        n_rounds: Rounds of the collapsed engine.
        n_last: Proposals of the last collapsed round; default ``n_round``.
        n_centers: Kernels of the adaptive proposal.
        defensive: Prior fraction of an adaptive round.
        ess_floor: Kish size under which the status is ``LOW_ESS``.

    Returns:
        A :class:`RefitCloud`; with no valid record the status is ``NO_SUPPORT``
        and the elements are NaN.
    """
    if report is not None:
        raise NotImplementedError("the report term is not implemented")
    n_valid = jnp.sum(records.is_valid)

    def one(_):
        return scale_rotate_refit(
            target,
            records,
            key,
            n_particles=n_particles,
            n_proposals=n_round,
            ess_floor=ess_floor,
        )

    def many(_):
        return collapsed_refit(
            target,
            records,
            key,
            n_particles=n_particles,
            n_round=n_round,
            n_rounds=n_rounds,
            n_last=n_last,
            n_centers=n_centers,
            defensive=defensive,
            ess_floor=ess_floor,
        )

    if isinstance(n_valid, jax.core.Tracer):
        cloud = jax.lax.cond(n_valid >= 2, many, one, 0)
    else:
        cloud = many(0) if int(n_valid) >= 2 else one(0)
    supported = n_valid > 0
    cloud = eqx.tree_at(
        lambda c: c.elements, cloud, jnp.where(supported, cloud.elements, jnp.nan)
    )
    return eqx.tree_at(
        lambda c: c.status,
        cloud,
        jnp.where(supported, cloud.status, int(UpdaterStatus.NO_SUPPORT)).astype(
            jnp.int32
        ),
    )
