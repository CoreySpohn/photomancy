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

# jaxns + tfp are imported lazily inside the functions below: ``import jaxns`` enables
# JAX x64 globally as an import-time side effect and pulls in tensorflow-probability, so
# we keep ``import photomancy`` light, paying that only when nested sampling runs.


def build_scene_nested_model(scene, forward, likelihood, *, fit_leaves, bounds):
    """Build a jaxns ``Model`` for a scene fit: Uniform priors on the fitted leaves.

    The nested-sampling parallel of ``build_scene_logdensity`` -- it produces a jaxns
    ``Model`` (a sampleable prior + a likelihood) rather than a flat logdensity, because
    nested sampling integrates over the prior. ``bounds`` is a list of ``(low, high)``
    arrays parallel to ``fit_leaves(scene)`` (each matching its leaf's shape); together
    they form an independent Uniform prior over the raveled fitted leaves. Heterogeneous
    (non-Uniform) per-leaf priors are a future extension.

    Args:
        scene: The scene ``eqx.Module`` (its selected leaves are the parameters).
        forward: ``scene -> predicted`` (same plug-in the other scene fits use).
        likelihood: ``predicted -> scalar`` log-likelihood (the observed data is closed
            over by the caller). NOTE: likelihood only -- jaxns adds the prior.
        fit_leaves: ``scene -> list[leaf]`` selecting the fitted leaves.
        bounds: list of ``(low, high)`` arrays, parallel to ``fit_leaves(scene)``.

    Returns:
        ``(model, unravel)``: the jaxns ``Model``, and ``unravel`` mapping a posterior
        sample row back to the fitted-leaf PyTree (the same z-space as
        ``build_scene_logdensity``).
    """
    import tensorflow_probability.substrates.jax as tfp
    from jaxns.framework.model import Model
    from jaxns.framework.prior import Prior

    tfpd = tfp.distributions

    n_fit = len(fit_leaves(scene))
    mask = jax.tree_util.tree_map(lambda _: False, scene)
    mask = eqx.tree_at(fit_leaves, mask, [True] * n_fit)
    params0, static = eqx.partition(scene, mask)
    _z0, unravel = ravel_pytree(params0)

    lows, highs = zip(*bounds, strict=True)
    low_z = ravel_pytree(
        eqx.partition(eqx.tree_at(fit_leaves, scene, list(lows)), mask)[0]
    )[0]
    high_z = ravel_pytree(
        eqx.partition(eqx.tree_at(fit_leaves, scene, list(highs)), mask)[0]
    )[0]
    # A batched Uniform (vector low/high) keeps a per-element quantile that jaxns needs
    # for the unit-cube transform; ``tfpd.Independent`` would hide it.
    z_prior = tfpd.Uniform(low=low_z, high=high_z)

    def prior_model():
        z = yield Prior(z_prior, name="z")
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
