"""Pathfinder backend: BlackJAX quasi-Newton variational inference -> Gaussian.

Pathfinder runs an L-BFGS optimization toward the mode and fits a Gaussian from the
quasi-Newton inverse-Hessian estimate along that trajectory, keeping the point on the
path with the best evidence lower bound. It is a more robust alternative to the
single-Hessian Laplace fit (it uses the whole trajectory rather than one Hessian at the
mode) and is purpose-built as a fast initializer for the sampling backends. The dense
covariance is reconstructed from the compact L-BFGS factors, and ``evidence`` carries
the Pathfinder ELBO, a lower bound on the log marginal likelihood, not the Laplace
``log Z``. The multi-start variant weights its modes by their ELBO and feeds the
analytic EIG layer the same way ``LaplaceMixtureBackend`` does.

``pathfinder_fit`` is ``filter_jit``-compiled, so when ``logdensity`` is a
``SceneLogDensity`` Module its forward's array leaves thread as traced inputs rather
than baking into the compiled kernel as constants.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.infer.backends._flat import flat_density, start_points
from photomancy.infer.backends.base import (
    AbstractBackend,
    LegacyBackend,
    import_extra,
)
from photomancy.infer.backends.importance import ImportanceInfo, modes_importance
from photomancy.infer.legacy_posterior import GaussianPosterior, MixturePosterior
from photomancy.infer.posterior import Posterior


@eqx.filter_jit
def pathfinder_fit(logdensity, init, key, maxiter, maxcor):
    """Single-path Pathfinder fit: mean, dense covariance, and the ELBO.

    Args:
        logdensity: ``z -> scalar`` log-density over the flat parameter position.
        init: Initial flat position. Shape ``(d,)``.
        key: PRNG key (Pathfinder draws Monte Carlo samples for the ELBO).
        maxiter: Maximum L-BFGS iterations along the optimization path.
        maxcor: L-BFGS history size (number of correction pairs).

    Returns:
        A tuple ``(mean, cov, elbo)``: the variational mean ``(d,)``, the dense
        covariance ``(d, d)`` reconstructed from the L-BFGS factors, and the scalar
        ELBO (a lower bound on ``log Z``).
    """
    pathfinder = import_extra("blackjax.vi.pathfinder", "blackjax")
    lbfgs = import_extra("blackjax.optimizers.lbfgs", "blackjax")
    state, _ = pathfinder.approximate(
        rng_key=key,
        logdensity_fn=logdensity,
        initial_position=init,
        maxiter=maxiter,
        maxcor=maxcor,
    )
    cov = lbfgs.lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
    return state.position, cov, state.elbo


class PathfinderBackend(LegacyBackend):
    """BlackJAX Pathfinder variational inference, returning a GaussianPosterior.

    A more robust alternative to ``LaplaceBackend`` and a fast initializer for the
    sampling backends. ``evidence`` is the Pathfinder ELBO, a lower bound on ``log Z``,
    rather than the Laplace evidence.

    Args:
        maxiter: Maximum L-BFGS iterations for the optimization path.
        maxcor: L-BFGS history size (number of correction pairs).
    """

    maxiter: int = 50
    maxcor: int = 10

    def run(self, logdensity, init, key=None):
        """Fit single-path Pathfinder and return a GaussianPosterior."""
        if key is None:
            raise ValueError("PathfinderBackend.run requires a PRNG key.")
        mean, cov, elbo = pathfinder_fit(
            logdensity, init, key, self.maxiter, self.maxcor
        )
        return GaussianPosterior(mean=mean, cov=cov, evidence=elbo)


class PathfinderMixtureBackend(LegacyBackend):
    """Multi-start Pathfinder: an ELBO-weighted mixture of Gaussians.

    Runs single-path Pathfinder from each of ``K`` starts (``init`` has shape
    ``(K, d)``, one row per start) and assembles a ``MixturePosterior`` whose modes are
    weighted by their ELBO. Like ``LaplaceMixtureBackend`` this is an EIG /
    value-of-information substrate, but it trades the single-Hessian Laplace covariance
    for Pathfinder's trajectory estimate, which is steadier when a mode is poorly
    conditioned.

    Args:
        maxiter: L-BFGS iterations per start.
        maxcor: L-BFGS history size per start.
    """

    maxiter: int = 50
    maxcor: int = 10

    def run(self, logdensity, init, key=None):
        """Fit one Pathfinder mode per start in ``init`` (shape ``(K, d)``)."""
        if key is None:
            raise ValueError("PathfinderMixtureBackend.run requires a PRNG key.")
        keys = jax.random.split(key, init.shape[0])

        def fit_one(z0, k):
            return pathfinder_fit(logdensity, z0, k, self.maxiter, self.maxcor)

        means, covs, elbos = jax.vmap(fit_one)(init, keys)
        return MixturePosterior(means=means, covs=covs, log_evidences=elbos)


class PathfinderInfo(eqx.Module):
    """The paths of a :class:`Pathfinder` run.

    Coordinates are the unconstrained flat vector of ``fit.to_real``.

    Attributes:
        means: Variational means ``(K, d)``.
        covs: Variational covariances ``(K, d, d)``.
        elbo: ELBO of each path, which sets its weight in the proposal ``(K,)``.
        kept: Whether each path entered the proposal (a finite ELBO) ``(K,)``.
        n_modes_kept: Number of paths kept; zero means the slice is NaN.
        importance: The :class:`~photomancy.infer.backends.importance.ImportanceInfo`.
    """

    means: jax.Array
    covs: jax.Array
    elbo: jax.Array
    kept: jax.Array
    n_modes_kept: jax.Array
    importance: ImportanceInfo


class Pathfinder(AbstractBackend):
    """Pathfinder proposals, wrapped on periodic coordinates, importance weighted.

    As :class:`~photomancy.infer.backends.laplace.Laplace`, with each component fitted
    by BlackJAX Pathfinder (a Gaussian from the L-BFGS trajectory, steadier than one
    Hessian when a mode is poorly conditioned) and weighted by its ELBO. The proposal
    is the mixture of the paths, wrapped on every periodic coordinate; the draws are
    weighted to the fit with the wrapped density, so the evidence is the importance mean
    and the proposal quality shows in the weights. The evidence and its standard error
    are NaN when the Pareto k-hat of the weights is above ``khat_threshold(n_draws)``
    (``info.importance.khat``); the draws and weights are not gated. A path whose ELBO
    is not finite leaves the mixture; with none left the slice is NaN. The cases where
    a Gaussian is a poor proposal, and the seed for ``AdaptiveImportance``, are those
    of ``Laplace``.

    Args:
        n_draws: Importance draws.
        n_modes: Number of starts and proposal components.
        maxiter: Maximum L-BFGS iterations along the path.
        maxcor: L-BFGS history size.
        init: ``None`` for prior draws as starts, one constrained draw, or a
            ``Posterior``.
    """

    n_draws: int = eqx.field(static=True, default=2000)
    n_modes: int = eqx.field(static=True, default=1)
    maxiter: int = eqx.field(static=True, default=50)
    maxcor: int = eqx.field(static=True, default=10)
    init: Posterior | dict | None = None

    @eqx.filter_jit
    def _run(self, fit, key, *, batch_size=None):
        """Fit the paths, then draw and weight from their mixture."""
        k_start, k_fit, k_draw = jax.random.split(key, 3)
        density = flat_density(fit)
        starts = start_points(fit, self.init, k_start, self.n_modes)
        keys = jax.random.split(k_fit, self.n_modes)
        means, covs, elbo = jax.vmap(
            lambda z0, k: pathfinder_fit(density, z0, k, self.maxiter, self.maxcor)
        )(starts, keys)
        kept = jnp.isfinite(elbo) & jnp.all(jnp.isfinite(means), axis=-1)
        draws = modes_importance(
            fit, k_draw, means, covs, elbo, kept, self.n_draws, batch_size
        )
        info = PathfinderInfo(means, covs, elbo, kept, jnp.sum(kept), draws.info)
        return eqx.tree_at(lambda d: d.info, draws, info)
