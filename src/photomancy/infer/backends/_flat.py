"""Private helpers the backends share: the flat unconstrained vector of a ``Fit``.

A gradient or sampling backend moves a flat vector ``z`` in the unconstrained
coordinates of ``fit.to_real``. The vector is the ``jax.flatten_util.ravel_pytree``
of one draw's entry dict (sorted keys), so ``unravel(z)`` is a valid argument to
``fit.real_log_prob`` and ``fit.from_real``. Only ``Fit`` methods are used.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.infer.evidence import importance_evidence
from photomancy.infer.posterior import Posterior


def draw_template(fit):
    """Zeros with the structure and shapes of one draw ``x`` of ``fit`` (float64)."""
    shapes = eqx.filter_eval_shape(lambda k: fit.sample_prior(k), jax.random.key(0))
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)


def flat_problem(fit):
    """Return ``(unravel, logdensity, dim)`` for the unconstrained flat vector.

    ``logdensity(z)`` is ``fit.real_log_prob(unravel(z))``, the density in ``u``
    including the log volume of the map from the stored coordinates.
    """
    template = fit.to_real(draw_template(fit))
    flat, unravel = ravel_pytree(template)

    def logdensity(z):
        return fit.real_log_prob(unravel(z))

    return unravel, logdensity, flat.shape[0]


def to_x(fit, unravel, z):
    """Stored-coordinate entries ``x`` of a batch of flat vectors ``z (N, d)``."""
    x, _ = fit.from_real(jax.vmap(unravel)(z))
    return x


def start_points(fit, init, key, n):
    """Return ``n`` flat starting vectors ``(n, d)`` in unconstrained coordinates.

    Args:
        fit: The fit.
        init: ``None`` for prior draws; one constrained draw ``x`` (an entry dict with
            leaf shapes, repeated ``n`` times); or a :class:`Posterior` (``n`` draws
            by its weights).
        key: PRNG key (unused for a single draw).
        n: Number of starts.
    """
    if init is None:
        u = fit.to_real(fit.sample_prior(key, (n,)))
    elif isinstance(init, Posterior):
        index = jax.random.categorical(key, init.log_weights, shape=(n,))
        u = fit.to_real(jax.tree.map(lambda a: a[index], init.samples))
    else:
        u = fit.to_real(
            jax.tree.map(lambda a: jnp.broadcast_to(a, (n, *a.shape)), init)
        )
    return jax.vmap(lambda one: ravel_pytree(one)[0])(u)


class FlatDensity(eqx.Module):
    """``z -> fit.real_log_prob(unravel(z))`` as a module, so the fit stays traced.

    The unravel function is rebuilt from ``template`` (zeros of one unconstrained draw)
    on every call, so the module has array leaves only and a ``filter_jit`` that
    receives it neither recompiles for a new instance nor bakes the fit's arrays in.
    """

    fit: object
    template: dict

    def __call__(self, z):
        """Density of the flat unconstrained vector ``z``."""
        unravel = ravel_pytree(self.template)[1]
        return self.fit.real_log_prob(unravel(z))


def flat_density(fit):
    """Return the :class:`FlatDensity` of ``fit``."""
    return FlatDensity(fit, fit.to_real(draw_template(fit)))


def mixture_importance(fit, density, means, covs, log_pi, n_draws, key):
    """Draw from a Gaussian mixture proposal and weight the draws to the fit.

    Args:
        fit: The fit.
        density: Its :class:`FlatDensity`.
        means: Component means ``(K, d)``.
        covs: Component covariances ``(K, d, d)``.
        log_pi: Unnormalized component log weights ``(K,)``; NaN counts as zero.
        n_draws: Number of draws.
        key: PRNG key.

    Returns:
        ``(x, log_weights, log_evidence, se, component, proposal_log_prob)``
        with ``x`` the stored-coordinate entries of the ``n_draws`` draws, unnormalized
        log weights ``density(z) - log q(z)``, the importance estimate of the evidence
        and its independent-draw error, the component of each draw and ``log q``.
    """
    log_pi = jnp.where(jnp.isnan(log_pi), -jnp.inf, log_pi)
    log_pi = log_pi - jax.scipy.special.logsumexp(log_pi)
    chol = jnp.linalg.cholesky(covs)
    k_comp, k_eps = jax.random.split(key)
    comp = jax.random.categorical(k_comp, log_pi, shape=(n_draws,))
    eps = jax.random.normal(k_eps, (n_draws, means.shape[-1]))
    z = means[comp] + jnp.einsum("nij,nj->ni", chol[comp], eps)

    def log_q(point):
        def one(mean, ch, lp):
            return lp + jax.scipy.stats.multivariate_normal.logpdf(
                point, mean, ch @ ch.T
            )

        return jax.scipy.special.logsumexp(jax.vmap(one)(means, chol, log_pi))

    proposal = jax.vmap(log_q)(z)
    log_w = jax.vmap(density)(z) - proposal
    log_z, se = importance_evidence(log_w)
    unravel = ravel_pytree(density.template)[1]
    x, _ = density.fit.from_real(jax.vmap(unravel)(z))
    return x, log_w, log_z, se, comp, proposal
