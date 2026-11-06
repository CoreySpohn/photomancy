"""Orbits For The Impatient (OFTI) -- discovery-tier rejection sampler.

OFTI (Blunt et al. 2017, AJ 153, 229) fits relative astrometry by drawing orbit
shapes from priors and a target from the reference-epoch Gaussian error.
Each orbit is scaled and rotated through that target; the other valid epochs
are applied by rejection. The prior on semimajor axis and ascending node is
induced by scale-and-rotate, with period and eccentricity support truncations.

This is a from-scratch JAX reimplementation, not a wrapper around orbitize!; the
scale-and-rotate mechanics were cross-checked against orbitize.sampler.OFTI.

The phase parameter is the mean anomaly at the reference epoch, ``m_ref`` (drawn
uniform). This decouples orbital phase from the semimajor axis, so the scale
step is exact and Kepler-consistent in orbix's derived-period parameterization
(period is always tied to ``a`` via Kepler's third law inside the forward model).
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G
from orbix.equations.orbit import mean_motion

from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.priors import ecc_distribution
from photomancy.posterior import SamplePosterior

# Output rows match grid_search column order for cross-tier compatibility.
PARAM_NAMES = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")

_TWO_PI = 2.0 * jnp.pi


def _default_ref_idx(data):
    """Index of the highest-SNR valid epoch (default scale-and-rotate anchor)."""
    sep = jnp.hypot(data.ra, data.dec)
    err = jnp.hypot(data.ra_err, data.dec_err)
    snr = jnp.where(data.is_valid, sep / jnp.maximum(err, 1e-30), -jnp.inf)
    return int(jnp.argmax(snr))


def _scale_rotate_score(
    e,
    cos_i,
    cos_w,
    sin_w,
    m_ref,
    tx,
    ty,
    data,
    t_ref,
    mu,
    Ms,
    dist_pc,
    log_P_lo,
    log_P_hi,
    e_max,
):
    """Scale-and-rotate one orbit onto (tx, ty) at t_ref; return (row, loglike).

    The orbit is generated with placeholder ``a=1, W=0`` at mean anomaly
    ``m_ref`` at the reference epoch, then scaled in ``a`` and rotated in ``W`` so
    its reference-epoch sky position lands exactly on the (noised) target point.
    ``tp`` is back-computed from the scaled period to preserve ``m_ref`` at the
    reference epoch.

    Args:
        e: Eccentricity. Scalar.
        cos_i: Cosine of inclination. Scalar.
        cos_w: Cosine of argument of periapsis. Scalar.
        sin_w: Sine of argument of periapsis. Scalar.
        m_ref: Mean anomaly at the reference epoch (radians). Scalar.
        tx: Target RA offset at the reference epoch (arcsec). Scalar.
        ty: Target DEC offset at the reference epoch (arcsec). Scalar.
        data: RelativeAstromData with the reference epoch masked out of is_valid.
        t_ref: Reference epoch time (days). Scalar.
        mu: Standard gravitational parameter G*Ms (AU^3/day^2). Scalar.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance (parsec). Scalar.
        log_P_lo: Lower log10-period truncation (days).
        log_P_hi: Upper log10-period truncation (days).
        e_max: Eccentricity truncation.

    Returns:
        Tuple ``(row, loglike)`` where ``row`` is the 7-vector
        ``(a, e, cos_i, W, cos_w, sin_w, tp)`` and ``loglike`` is a scalar
        (``-inf`` outside the truncated support).
    """
    # Reference-epoch model at a=1, W=0. With tp set so the mean anomaly at t_ref
    # equals m_ref, the eccentric anomaly there is independent of a.
    n1 = mean_motion(1.0, mu)
    tp1 = t_ref - m_ref / n1
    x0, y0 = predict_relative_astrometry(
        jnp.atleast_1d(t_ref), 1.0, e, cos_i, 0.0, cos_w, sin_w, tp1, Ms, dist_pc
    )
    x0 = x0[0]
    y0 = y0[0]

    # Scale a (separation ~ a) and rotate W (sky-plane rotation) to hit target.
    r0 = jnp.maximum(jnp.hypot(x0, y0), 1e-30)
    a_new = jnp.hypot(tx, ty) / r0
    # (x, y) = (east, north); position angle runs north through east.
    W_new = (jnp.arctan2(tx, ty) - jnp.arctan2(x0, y0)) % _TWO_PI

    # Kepler-consistency: new period from scaled a; tp preserves m_ref at t_ref.
    n_new = mean_motion(a_new, mu)
    tp_new = t_ref - m_ref / n_new
    log_P = jnp.log10(_TWO_PI / n_new)

    # The caller masks the reference epoch, already sampled in the proposal.
    ra_p, dec_p = predict_relative_astrometry(
        data.times, a_new, e, cos_i, W_new, cos_w, sin_w, tp_new, Ms, dist_pc
    )
    ll = loglike_relative_astrom(ra_p, dec_p, data)

    # Truncate the induced prior to the requested period range, and e.
    in_support = (log_P >= log_P_lo) & (log_P <= log_P_hi) & (e < e_max)
    ll = jnp.where(in_support, ll, -jnp.inf)

    row = jnp.array([a_new, e, cos_i, W_new, cos_w, sin_w, tp_new])
    return row, ll


class AbstractConditioner(eqx.Module):
    """Proposes orbits from priors and conditions them on the data.

    A conditioner encapsulates the OFTI proposal: draw raw shape parameters from
    priors, condition each onto the data, and return physical rows plus a
    log-likelihood for the rejection step. Subclasses differ only in how they
    condition (1-epoch scale-and-rotate vs all-epoch Thiele-Innes
    marginalization, the latter a planned twist).
    """

    @abstractmethod
    def sample_and_score(
        self, key, n, data, ref_idx, Ms, dist_pc, ecc_prior, log_P_range, e_max
    ):
        """Return ``(rows (n, 7), loglike (n,))`` for ``n`` proposed orbits."""


class ScaleAndRotate(AbstractConditioner):
    """Condition on a Gaussian reference target by scaling ``a`` and rotating ``W``.

    Draw shape parameters from their priors and a target from the correlated
    reference-epoch error, then map each orbit through that target. Rejection
    uses only the other valid epochs. The prior on ``(a, W)`` (``W`` is Omega)
    is induced by this transformation, subject to period and eccentricity
    truncations; no independent prior on ``a`` or ``W`` is imposed.

    See Blunt et al. (2017, AJ 153, 229) for OFTI.
    """

    def sample_and_score(
        self, key, n, data, ref_idx, Ms, dist_pc, ecc_prior, log_P_range, e_max
    ):
        """Draw n orbits from priors and scale-and-rotate onto the reference epoch.

        Args:
            key: JAX PRNG key.
            n: Number of orbits to propose.
            data: RelativeAstromData (with is_valid).
            ref_idx: Reference-epoch index for scale-and-rotate.
            Ms: Stellar mass (kg).
            dist_pc: Distance (parsec).
            ecc_prior: Eccentricity prior name (see priors.ECC_PRIOR_NAMES).
            log_P_range: (log10 P min, log10 P max) truncation on derived period.
            e_max: Eccentricity truncation.

        Returns:
            Tuple ``(rows, loglike)`` with shapes ``(n, 7)`` and ``(n,)``.
            Scores include only nonreference valid epochs and are ``-inf``
            outside the truncated support.
        """
        k_e, k_i, k_w, k_m, k_n1, k_n2 = jax.random.split(key, 6)
        e = ecc_distribution(ecc_prior).sample(k_e, (n,))
        cos_i = jax.random.uniform(k_i, (n,), minval=-1.0, maxval=1.0)
        omega = jax.random.uniform(k_w, (n,), minval=0.0, maxval=_TWO_PI)
        cos_w = jnp.cos(omega)
        sin_w = jnp.sin(omega)
        m_ref = jax.random.uniform(k_m, (n,), minval=0.0, maxval=_TWO_PI)

        # Reference-epoch target = observed point + correlated Gaussian noise.
        sx = data.ra_err[ref_idx]
        sy = data.dec_err[ref_idx]
        rho = data.corr[ref_idx]
        z1 = jax.random.normal(k_n1, (n,))
        z2 = jax.random.normal(k_n2, (n,))
        tx = data.ra[ref_idx] + sx * z1
        ty = data.dec[ref_idx] + sy * (
            rho * z1 + jnp.sqrt(jnp.maximum(1.0 - rho**2, 0.0)) * z2
        )

        mu = G * Ms
        t_ref = data.times[ref_idx]
        log_P_lo, log_P_hi = log_P_range
        score_data = eqx.tree_at(
            lambda d: d.is_valid, data, data.is_valid.at[ref_idx].set(False)
        )

        def cond_one(e_i, ci_i, cw_i, sw_i, m_i, tx_i, ty_i):
            return _scale_rotate_score(
                e_i,
                ci_i,
                cw_i,
                sw_i,
                m_i,
                tx_i,
                ty_i,
                score_data,
                t_ref,
                mu,
                Ms,
                dist_pc,
                log_P_lo,
                log_P_hi,
                e_max,
            )

        rows, ll = jax.vmap(cond_one)(e, cos_i, cos_w, sin_w, m_ref, tx, ty)
        return rows, ll


def ofti(
    data,
    *,
    Ms,
    dist_pc,
    key,
    n_accept=2000,
    ref_idx=None,
    ecc_prior="kipping13",
    log_P_range=(2.0, 5.0),
    e_max=0.99,
    batch=10**5,
    max_batches=1000,
    loglike_ref=None,
    conditioner=None,
):
    """Sample the orbit posterior for short-arc astrometry via OFTI.

    With ScaleAndRotate, draw shape parameters from their priors and a target
    from the reference-epoch Gaussian error, then scale and rotate each orbit
    through that target. Apply the other valid epochs by rejection until
    ``n_accept`` orbits are accepted. The prior on ``(a, W)`` (``W`` is Omega)
    is induced by scale-and-rotate, with the requested support truncations.
    With a valid rejection bound, the uniformly weighted particles sample the
    posterior under this induced measure. The default empirical bound is an
    approximation when later scores exceed it.

    See Blunt et al. (2017, AJ 153, 229) for OFTI.

    Args:
        data: RelativeAstromData (with is_valid) of relative astrometry.
        Ms: Stellar mass (kg).
        dist_pc: Distance (parsec).
        key: JAX PRNG key.
        n_accept: Target number of accepted orbits. Default 2000.
        ref_idx: Reference-epoch index; None selects the highest-SNR valid epoch.
        ecc_prior: Eccentricity prior name. Default "kipping13".
        log_P_range: (log10 P min, log10 P max) truncation on the derived period
            (days). Truncates the prior induced by scale-and-rotate.
        e_max: Eccentricity truncation. Default 0.99.
        batch: Orbits proposed per rejection batch. Default 100000.
        max_batches: Safety cap on the number of batches. Default 1000.
        loglike_ref: Log-likelihood bound subtracted before acceptance. None
            uses the first batch maximum; later scores above it are clipped to
            acceptance probability one, so this is not a guaranteed bound.
            Strict rejection requires an upper bound on the normalized score.
            For ScaleAndRotate, use the sum of
            ``-log(2*pi*ra_err*dec_err*sqrt(1-corr**2))`` over nonreference valid
            epochs. This is zero for a single valid epoch, for which both None
            and 0.0 accept every in-support proposal. In general 0.0 is not a
            strict bound because Gaussian log normalization can be positive.
        conditioner: An AbstractConditioner; None uses ScaleAndRotate().

    Returns:
        A SamplePosterior with ``n_accept`` particles, uniform log_weights, and
        param_names ``("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")``.

    Raises:
        RuntimeError: If ``max_batches`` is reached before ``n_accept`` orbits are
            accepted (acceptance too low; use Laplace/NUTS for well-constrained,
            long-arc orbits).
    """
    if conditioner is None:
        conditioner = ScaleAndRotate()
    if ref_idx is None:
        ref_idx = _default_ref_idx(data)

    @eqx.filter_jit
    def score_batch(k):
        return conditioner.sample_and_score(
            k, batch, data, ref_idx, Ms, dist_pc, ecc_prior, log_P_range, e_max
        )

    accepted = []
    n_have = 0
    n_tried = 0
    for _ in range(max_batches):
        key, k_b, k_u = jax.random.split(key, 3)
        rows, ll = score_batch(k_b)
        if loglike_ref is None:
            finite = ll[jnp.isfinite(ll)]
            loglike_ref = float(jnp.max(finite)) if finite.size > 0 else 0.0
        log_u = jnp.log(jax.random.uniform(k_u, (batch,)))
        accept = log_u < jnp.minimum(ll - loglike_ref, 0.0)
        accepted.append(rows[accept])
        n_have += int(accepted[-1].shape[0])
        n_tried += batch
        if n_have >= n_accept:
            break
    else:
        rate = n_have / max(n_tried, 1)
        raise RuntimeError(
            f"OFTI accepted {n_have} of {n_tried} orbits in {max_batches} "
            f"batches (rate {rate:.2e}); the short-arc assumption may be "
            f"violated -- use Laplace/NUTS for well-constrained orbits."
        )

    particles = jnp.concatenate(accepted, axis=0)[:n_accept]
    log_weights = jnp.full((n_accept,), -jnp.log(n_accept))
    return SamplePosterior(
        samples=particles,
        log_weights=log_weights,
        evidence=jnp.asarray(jnp.nan),
        param_names=PARAM_NAMES,
    )
