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

from photomancy.infer.backends._flat import (
    flat_density,
    mixture_importance,
    start_points,
)
from photomancy.infer.backends.base import (
    AbstractBackend,
    BackendDraws,
    LegacyBackend,
    import_extra,
)
from photomancy.infer.backends.laplace import LaplaceInfo
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


class Pathfinder(AbstractBackend):
    """Pathfinder proposals and an importance-weighted posterior.

    As :class:`~photomancy.infer.backends.laplace.Laplace`, with each component fitted
    by BlackJAX Pathfinder (a Gaussian from the L-BFGS trajectory, steadier than one
    Hessian when a mode is poorly conditioned) and weighted by its ELBO. The draws are
    importance weighted to the fit, so the evidence is the importance mean and the
    proposal quality shows in the weights. Angle coordinates of an orbit block are
    plain reals whose density repeats every ``2 pi``; as for
    :class:`~photomancy.infer.backends.laplace.Laplace`, the importance evidence on such
    a block is correct only when the proposal lies well inside one cell, and the weight
    diagnostics do not catch the failure. Use ``SMC`` for such an evidence.

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

    def _run(self, fit, key):
        """Fit the paths, then draw and weight from their mixture."""
        k_start, k_fit, k_draw = jax.random.split(key, 3)
        density = flat_density(fit)
        starts = start_points(fit, self.init, k_start, self.n_modes)
        keys = jax.random.split(k_fit, self.n_modes)
        means, covs, elbo = jax.vmap(
            lambda z0, k: pathfinder_fit(density, z0, k, self.maxiter, self.maxcor)
        )(starts, keys)
        x, log_w, evidence, se, comp, proposal = mixture_importance(
            fit, density, means, covs, elbo, self.n_draws, k_draw
        )
        return BackendDraws(
            samples=x,
            log_weights=log_w,
            log_evidence=evidence,
            log_evidence_se=se,
            info=LaplaceInfo(means, covs, elbo, comp, proposal),
            kind="monte_carlo",
        )
