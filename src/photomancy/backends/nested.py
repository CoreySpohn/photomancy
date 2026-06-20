"""Nested-sampling backend (jaxns): posterior + Bayesian evidence (log Z).

Unlike the MAP / MCMC backends, nested sampling needs a proper, sampleable PRIOR (it
integrates over the prior to get the evidence). So this backend consumes a jaxns
``Model`` (prior_model + log_likelihood), not a flat ``logdensity`` -- the same
divergence ``SMCBackend`` has. Build the model for a scene fit with
``build_scene_nested_model``. The headline output is the evidence (``log Z``), which
turns a posterior into model comparison / detection (a Bayes factor between two scenes).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import SamplePosterior

# jaxns is imported lazily inside the functions below: ``import jaxns`` enables JAX x64
# globally as a side effect and pulls in tensorflow-probability transitively, so we keep
# ``import photomancy`` light, paying that only when nested sampling runs. The prior
# adapter below uses only the jaxns ``SpecialPrior`` interface -- no tfp of our own.


def build_scene_nested_model(scene, forward, likelihood, *, fit_leaves, prior):
    """Build a jaxns ``Model`` for a scene fit from a photomancy ``AbstractPrior``.

    The nested-sampling parallel of ``build_scene_logdensity`` -- it produces a jaxns
    ``Model`` (a sampleable prior + a likelihood) rather than a flat logdensity, because
    nested sampling integrates over the prior. ``prior`` is an
    :class:`~photomancy.priors.AbstractPrior` over the raveled fitted leaves (the same
    z-space as ``build_scene_logdensity``), adapted to jaxns through a thin
    ``SpecialPrior`` wrapper -- so the prior path involves no tensorflow-probability.

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
    from jaxns.framework.model import Model
    from jaxns.framework.special_priors import SpecialPrior

    n_fit = len(fit_leaves(scene))
    mask = jax.tree_util.tree_map(lambda _: False, scene)
    mask = eqx.tree_at(fit_leaves, mask, [True] * n_fit)
    params0, static = eqx.partition(scene, mask)
    _z0, unravel = ravel_pytree(params0)

    class _JaxnsPrior(SpecialPrior):
        """Adapt a photomancy ``AbstractPrior`` to the jaxns ``SpecialPrior`` interface.

        The interfaces coincide: jaxns asks a prior for ``_forward`` (unit-cube -> X),
        ``_inverse``, ``_log_prob``, plus shape / dtype hooks -- exactly what an
        ``AbstractPrior`` provides over z. Defined locally so jaxns stays lazy-imported.
        """

        def __init__(self, dist, *, name=None):
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

    z_prior = _JaxnsPrior(prior, name="z")

    def prior_model():
        z = yield z_prior
        return eqx.combine(unravel(z), static)

    def log_likelihood(recombined_scene):
        return likelihood(forward(recombined_scene))

    return Model(prior_model=prior_model, log_likelihood=log_likelihood), unravel


class JaxnsBackend(AbstractBackend):
    """jaxns nested sampling -> ``SamplePosterior`` carrying the evidence (``log Z``).

    ``run`` takes a jaxns ``Model`` (not a flat logdensity); ``init`` is ignored (live
    points are drawn from the prior). Returns weighted samples (``log_dp_mean``) and the
    evidence (``log_Z_mean``).

    Args:
        max_samples: Termination cap on total samples.
        num_live_points: Approximate live points; ``None`` uses the jaxns default
            (``c * (k + 1)``, ``c = 20 * D``). Cost scales with the parameter count, so
            jaxns suits low-dimensional model comparison.
        gradient_guided: Gradient-guided slice sampling (forwards are differentiable).
        parameter_estimation: jaxns robustness preset tuned for parameter posteriors.

    Note: like NUTS / SMC, jaxns jits the model internally, so a Module forward with a
    large array (e.g. a coronagraph PSF datacube) would bake it as a constant. Threading
    big-array forwards is the same ``filter_jit`` / partition TODO as the BlackJAX
    backends.
    """

    max_samples: float = 1e5
    num_live_points: int | None = None
    gradient_guided: bool = False
    parameter_estimation: bool = False

    def run(self, model, init=None, key=None):
        """Run nested sampling on ``model``; returns weighted samples + ``log Z``."""
        from jaxns import NestedSampler

        if key is None:
            raise ValueError("JaxnsBackend.run requires a PRNG key.")
        kwargs = dict(
            model=model,
            max_samples=self.max_samples,
            gradient_guided=self.gradient_guided,
            parameter_estimation=self.parameter_estimation,
        )
        if self.num_live_points is not None:
            kwargs["num_live_points"] = self.num_live_points
        ns = NestedSampler(**kwargs)
        termination, state = ns(key)
        results = ns.to_results(termination_reason=termination, state=state)
        # jaxns returns samples as a dict {name: (N, *shape)}; stack to a flat (N, d) in
        # yield order. build_scene_nested_model yields one "z" prior, so the stacked
        # columns are exactly build_scene_logdensity's z-space.
        cols = [
            jnp.asarray(v).reshape(jnp.asarray(v).shape[0], -1)
            for v in results.samples.values()
        ]
        samples = jnp.concatenate(cols, axis=1)
        return SamplePosterior(
            samples=samples,
            log_weights=results.log_dp_mean,
            evidence=results.log_Z_mean,
        )
