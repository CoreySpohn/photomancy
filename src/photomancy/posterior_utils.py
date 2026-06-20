"""General posterior utilities: manifold projection and variance-capped sampling.

Domain-agnostic operations over the unified posteriors. ``project_samples`` snaps
samples onto any logdensity's mode manifold via a short Adam descent;
``sample_capped`` draws from a Gaussian mixture with its covariance eigenvalues
capped (e.g. to keep visualization draws off prior boundaries). Both return
positions in the posterior's z-space; a domain layer maps them to physical
parameters if needed.
"""

import jax
import jax.numpy as jnp
import optax


def project_samples(samples, logdensity, *, n_steps=150, lr=0.005):
    """Project each sample onto ``logdensity``'s mode manifold (short Adam descent).

    Re-optimizes every row of ``samples`` against ``logdensity`` with a gentle Adam
    descent, snapping points from a Gaussian tangent plane onto the curved valley the
    data actually constrains. Domain-agnostic: it needs only a scalar ``logdensity(z)``.

    Args:
        samples: Flat positions to project. Shape ``(n, d)``.
        logdensity: ``z -> scalar`` log-density (maximized; the descent minimizes its
            negative).
        n_steps: Adam iterations per sample. Default 150.
        lr: Adam learning rate. Default 0.005 (gentle).

    Returns:
        The projected positions, shape ``(n, d)``, in the same space as ``samples``.
    """
    opt = optax.adam(lr)

    def _project_one(z0):
        def neg_logdensity(z):
            return -logdensity(z)

        def step(carry, _):
            z, state = carry
            grad = jax.grad(neg_logdensity)(z)
            updates, state = opt.update(grad, state, z)
            return (optax.apply_updates(z, updates), state), None

        (z_opt, _), _ = jax.lax.scan(step, (z0, opt.init(z0)), None, length=n_steps)
        return z_opt

    return jax.vmap(_project_one)(samples)


def sample_capped(posterior, key, n, *, max_variance=10.0):
    """Draw ``n`` mixture samples with each mode's covariance eigenvalues capped.

    Capping the eigenvalues at ``max_variance`` keeps draws from saturating prior
    boundaries through downstream constraint transforms (which yields pathological
    points in plots). Operates on any :class:`~photomancy.posterior.MixturePosterior`.

    Args:
        posterior: A ``MixturePosterior`` (``means (K, d)``, ``covs (K, d, d)``,
            ``log_evidences (K,)``).
        key: PRNG key.
        n: Number of samples.
        max_variance: Eigenvalue cap on each mode's covariance. Default 10.0.

    Returns:
        Samples in flat z-space, shape ``(n, d)``.
    """

    def _cap(cov):
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        return (eigvecs * jnp.minimum(eigvals, max_variance)) @ eigvecs.T

    chols = jax.vmap(jnp.linalg.cholesky)(jax.vmap(_cap)(posterior.covs))
    k_comp, k_draw = jax.random.split(key)
    comp = jax.random.categorical(k_comp, posterior.log_evidences, shape=(n,))
    noise = jax.random.normal(k_draw, (n, posterior.means.shape[1]))
    return posterior.means[comp] + jnp.einsum("nij,nj->ni", chols[comp], noise)
