"""Nested-sampling backend (jaxns): posterior + Bayesian evidence (log Z).

Unlike the MAP / MCMC backends, nested sampling needs a proper, sampleable PRIOR (it
integrates over the prior to get the evidence). So this backend consumes a jaxns
``Model`` (a prior model that realizes its priors and returns the log-likelihood), not
a flat ``logdensity`` -- the same divergence ``SMCBackend`` has. Build the model for a
scene fit with ``build_scene_nested_model``. The headline output is the evidence
(``log Z``), which turns a posterior into model comparison / detection (a Bayes factor
between two scenes).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import SamplePosterior

# jaxns is imported lazily inside the functions below: importing it enables JAX x64
# globally as a side effect and pulls in tensorflow-probability transitively, so we keep
# ``import photomancy`` light, paying that only when nested sampling runs. The prior
# adapter below uses only the jaxctx ``AbstractPrior`` interface -- no tfp of our own.


def _to_jaxns_prior(prior, *, name):
    """Wrap a photomancy ``AbstractPrior`` as a jaxns-realizable prior named ``name``.

    jaxns 3 realizes priors through the jaxctx ``AbstractPrior`` interface, which asks
    for ``_forward`` (unit-cube -> X), ``_inverse``, ``_log_prob``, plus shape / dtype
    hooks -- exactly what a photomancy ``AbstractPrior`` provides over z. The class is
    defined here so jaxns (and jaxctx) stay lazy-imported.

    Args:
        prior: A :class:`~photomancy.priors.AbstractPrior` with ``ndim`` components.
        name: Variable name; the posterior samples are keyed by it.

    Returns:
        A jaxctx prior; call ``.realise()`` on it inside a jaxns prior model.
    """
    from jaxctx.priors import AbstractPrior as JaxctxPrior

    class _JaxnsPrior(JaxctxPrior):
        def __init__(self, dist, *, name):
            super().__init__(name=name)
            self.prior = dist

        def _dtype(self):
            return jnp.float64

        def _base_shape(self):
            return (self.prior.ndim,)

        def _shape(self):
            return (self.prior.ndim,)

        def _forward(self, U):
            return self.prior.forward(U)

        def _inverse(self, X):
            return self.prior.inverse(X)

        def _log_prob(self, X):
            return self.prior.log_prob(X)

    return _JaxnsPrior(prior, name=name)


def build_scene_nested_model(scene, forward, likelihood, *, fit_leaves, prior):
    """Build a jaxns ``Model`` for a scene fit from a photomancy ``AbstractPrior``.

    The nested-sampling parallel of ``build_scene_logdensity`` -- it produces a jaxns
    ``Model`` (a sampleable prior + a likelihood) rather than a flat logdensity, because
    nested sampling integrates over the prior. ``prior`` is an
    :class:`~photomancy.priors.AbstractPrior` over the raveled fitted leaves (the same
    z-space as ``build_scene_logdensity``), realized in jaxns through a thin adapter --
    so the prior path involves no tensorflow-probability of our own.

    Args:
        scene: The scene ``eqx.Module`` (its selected leaves are the parameters).
        forward: ``scene -> predicted`` (same plug-in the other scene fits use).
        likelihood: ``predicted -> scalar`` log-likelihood (the observed data is closed
            over by the caller). NOTE: likelihood only -- jaxns adds the prior.
        fit_leaves: ``scene -> list[leaf]`` selecting the fitted leaves.
        prior: An :class:`~photomancy.priors.AbstractPrior` over the raveled fitted
            leaves; ``prior.ndim`` must equal the number of raveled scalars.

    Returns:
        ``(model, unravel)``: the jaxns ``Model``, and ``unravel`` mapping a posterior
        sample row back to the fitted-leaf PyTree (the same z-space as
        ``build_scene_logdensity``).
    """
    from jaxns.model import Model

    n_fit = len(fit_leaves(scene))
    mask = jax.tree_util.tree_map(lambda _: False, scene)
    mask = eqx.tree_at(fit_leaves, mask, [True] * n_fit)
    params0, static = eqx.partition(scene, mask)
    _z0, unravel = ravel_pytree(params0)

    z_prior = _to_jaxns_prior(prior, name="z")

    def prior_model():
        z = z_prior.realise()
        return likelihood(forward(eqx.combine(unravel(z), static)))

    return Model(prior_model=prior_model), unravel


class JaxnsBackend(AbstractBackend):
    """jaxns nested sampling -> ``SamplePosterior`` carrying the evidence (``log Z``).

    ``run`` takes a jaxns ``Model`` (not a flat logdensity); ``init`` is ignored (live
    points are drawn from the prior). Returns weighted samples (``log_dp``) and the
    evidence (``log_Z_mean``, the classic expectation estimate of ``log Z``).

    Args:
        max_samples: Hard cap on stored samples. jaxns stops early (with a nonzero
            termination reason) if the buffer fills before its evidence-convergence
            condition is met.
        num_live_points: Number of independent root chains (jaxns
            ``root_allocation_degree``); ``None`` uses the jaxns default, ``30 * D``
            for ``D`` parameters. Cost scales with the parameter count, so jaxns suits
            low-dimensional model comparison.

    Note: like NUTS / SMC, jaxns jits the model internally, so a Module forward with a
    large array (e.g. a coronagraph PSF datacube) would bake it as a constant. Threading
    big-array forwards is the same ``filter_jit`` / partition TODO as the BlackJAX
    backends.
    """

    max_samples: int = 100_000
    num_live_points: int | None = None

    def run(self, model, init=None, key=None):
        """Run nested sampling on ``model``; returns weighted samples + ``log Z``."""
        from jaxns.core import NestedSampler

        if key is None:
            raise ValueError("JaxnsBackend.run requires a PRNG key.")
        ns = NestedSampler(
            model=model,
            max_samples=int(self.max_samples),
            root_allocation_degree=self.num_live_points,
        )
        results = ns.run(key=key).to_result().trim()
        # jaxns returns samples keyed by prior name {name: (N, *shape)}; stack to a flat
        # (N, d) in realization order. build_scene_nested_model realizes one "z" prior,
        # so the stacked columns are exactly build_scene_logdensity's z-space.
        cols = [
            jnp.asarray(v).reshape(jnp.asarray(v).shape[0], -1)
            for v in results.X_samples.values()
        ]
        samples = jnp.concatenate(cols, axis=1)
        return SamplePosterior(
            samples=samples,
            log_weights=results.log_dp,
            evidence=results.log_Z_mean,
        )
