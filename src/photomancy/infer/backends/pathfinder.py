"""Pathfinder backend: BlackJAX quasi-Newton variational inference -> Gaussian.

Pathfinder runs an L-BFGS optimization toward the mode and fits a Gaussian from the
quasi-Newton inverse-Hessian estimate along that trajectory, keeping the point on the
path with the best evidence lower bound. It is a more robust alternative to the
single-Hessian Laplace fit (it uses the whole trajectory rather than one Hessian at the
mode) and is purpose-built as a fast initializer for the sampling backends. The dense
covariance is reconstructed from the compact L-BFGS factors. The ELBO of each path, a
lower bound on the log marginal likelihood, sets the path's weight in the proposal; the
reported evidence is the importance mean of the draws weighted to the fit, not the ELBO.
The multi-start variant mixes its paths by their ELBO, as ``Laplace`` mixes its modes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.infer.backends._flat import flat_density, start_points
from photomancy.infer.backends.base import (
    AbstractBackend,
    import_extra,
)
from photomancy.infer.backends.importance import ImportanceInfo, modes_importance
from photomancy.infer.posterior import Posterior


@eqx.filter_jit
def pathfinder_fit(logdensity, init, key, n_steps, history_size):
    """Single-path Pathfinder fit: mean, dense covariance, and the ELBO.

    Args:
        logdensity: ``z -> scalar`` log-density over the flat parameter position.
        init: Initial flat position. Shape ``(d,)``.
        key: PRNG key (Pathfinder draws Monte Carlo samples for the ELBO).
        n_steps: Maximum L-BFGS iterations along the optimization path.
        history_size: L-BFGS history size (number of correction pairs).

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
        maxiter=n_steps,
        maxcor=history_size,
    )
    cov = lbfgs.lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
    return state.position, cov, state.elbo


class PathfinderInfo(eqx.Module):
    """The paths of a :class:`Pathfinder` run.

    Coordinates are the unconstrained flat vector of ``fit.to_real``.

    Attributes:
        flat_means: Variational means ``(K, d)``.
        flat_covs: Variational covariances ``(K, d, d)``.
        elbo: ELBO of each path, which sets its weight in the proposal ``(K,)``.
        kept: Whether each path entered the proposal (a finite ELBO) ``(K,)``.
        n_modes_kept: Number of paths kept; zero means the slice is NaN.
        importance: The :class:`~photomancy.infer.backends.importance.ImportanceInfo`.
    """

    flat_means: jax.Array
    flat_covs: jax.Array
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
    and the proposal quality shows in the weights. The evidence follows the contract on
    :attr:`~photomancy.infer.posterior.Posterior.log_evidence` (the k-hat is
    ``info.importance.khat``); the draws and weights are not gated. A path whose ELBO
    is not finite leaves the mixture; with none left the slice is NaN. The cases where
    a Gaussian is a poor proposal, and the seed for ``AdaptiveImportance``, are those
    of ``Laplace``, walls (:attr:`Fit.walls`) included: they are not refused.

    Attributes:
        n_draws: Importance draws.
        n_modes: Number of starts and proposal components.
        n_steps: Maximum L-BFGS iterations along the path.
        history_size: L-BFGS history size.
        init: ``None`` for prior draws as starts, one constrained draw (repeated for
            every start), an entry dict whose every leaf has a leading axis of length
            ``n_modes`` (one explicit start each, so each alias keeps its own mode),
            or a ``Posterior`` whose weights pick the starts; the starts are placed in
            the flat vector by ``fit.to_flat``.
    """

    n_draws: int = eqx.field(static=True, default=2000)
    n_modes: int = eqx.field(static=True, default=1)
    n_steps: int = eqx.field(static=True, default=50)
    history_size: int = eqx.field(static=True, default=10)
    init: Posterior | dict | None = None

    @eqx.filter_jit
    def _run(self, fit, key, *, batch_size=None):
        """Fit the paths, then draw and weight from their mixture."""
        k_start, k_fit, k_draw = jax.random.split(key, 3)
        density = flat_density(fit)
        starts = start_points(fit, self.init, k_start, self.n_modes)
        keys = jax.random.split(k_fit, self.n_modes)
        means, covs, elbo = jax.vmap(
            lambda z0, k: pathfinder_fit(
                density, z0, k, self.n_steps, self.history_size
            )
        )(starts, keys)
        kept = jnp.isfinite(elbo) & jnp.all(jnp.isfinite(means), axis=-1)
        draws = modes_importance(
            fit, k_draw, means, covs, elbo, kept, self.n_draws, batch_size
        )
        info = PathfinderInfo(means, covs, elbo, kept, jnp.sum(kept), draws.info)
        return eqx.tree_at(lambda d: d.info, draws, info)
