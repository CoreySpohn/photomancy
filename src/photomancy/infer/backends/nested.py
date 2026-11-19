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

from photomancy.infer.backends._flat import draw_template
from photomancy.infer.backends.base import (
    AbstractBackend,
    BackendDraws,
    LegacyBackend,
    import_extra,
)
from photomancy.infer.legacy_posterior import SamplePosterior
from photomancy.infer.priors import flat_prior

# jaxns is an optional extra and is imported inside the functions below: importing it
# enables JAX x64 globally as a side effect and pulls in tensorflow-probability
# transitively, so ``import photomancy`` stays light and pays that only when nested
# sampling runs. The prior adapter uses only the jaxctx ``AbstractPrior`` interface.


def jaxns_prior(prior, *, name):
    """Wrap a prior over z as a jaxns-realizable prior named ``name``.

    jaxns 3 realizes priors through the jaxctx ``AbstractPrior`` interface, which asks
    for ``_forward`` (unit-cube -> X), ``_inverse``, ``_log_prob``, plus shape / dtype
    hooks: the ``icdf``, ``cdf`` and ``log_prob`` of a prior over z. The
    class is defined here so jaxns (and jaxctx) stay lazy-imported.

    Args:
        prior: A ``priorities`` distribution over z or an
            :class:`~photomancy.infer.priors.IndependentPrior`, with an inverse CDF.
        name: Variable name; the posterior samples are keyed by it.

    Returns:
        A jaxctx prior; call ``.realise()`` on it inside a jaxns prior model.

    Raises:
        ValueError: If the prior has no inverse CDF (for example a mixture).
    """
    prior = flat_prior(prior)
    if not prior.has_icdf:
        raise ValueError(
            "Nested sampling needs a prior with an inverse CDF; a mixture has "
            "none. Use a gradient or sample backend (Laplace, NUTS, SMC) instead."
        )
    JaxctxPrior = import_extra("jaxctx.priors", "jaxns").AbstractPrior

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
            return self.prior.icdf(U)

        def _inverse(self, X):
            return self.prior.cdf(X)

        def _log_prob(self, X):
            return self.prior.log_prob(X)

    return _JaxnsPrior(prior, name=name)


def build_scene_nested_model(scene, forward, likelihood, *, fit_leaves, prior):
    """Build a jaxns ``Model`` for a scene fit from a prior over z.

    The nested-sampling parallel of ``build_scene_logdensity`` -- it produces a jaxns
    ``Model`` (a sampleable prior + a likelihood) rather than a flat logdensity, because
    nested sampling integrates over the prior. ``prior`` is a ``priorities``
    distribution or an :class:`~photomancy.infer.priors.IndependentPrior` over the
    raveled fitted leaves (the same z-space as ``build_scene_logdensity``), realized
    in jaxns through a thin adapter, so the prior path involves no
    tensorflow-probability of our own.

    Args:
        scene: The scene ``eqx.Module`` (its selected leaves are the parameters).
        forward: ``scene -> predicted`` (same plug-in the other scene fits use).
        likelihood: ``predicted -> scalar`` log-likelihood (the observed data is closed
            over by the caller). NOTE: likelihood only -- jaxns adds the prior.
        fit_leaves: ``scene -> list[leaf]`` selecting the fitted leaves.
        prior: A distribution over the raveled fitted leaves with an inverse CDF; its
            length must equal the number of raveled scalars.

    Returns:
        ``(model, unravel)``: the jaxns ``Model``, and ``unravel`` mapping a posterior
        sample row back to the fitted-leaf PyTree (the same z-space as
        ``build_scene_logdensity``).
    """
    Model = import_extra("jaxns.model", "jaxns").Model

    n_fit = len(fit_leaves(scene))
    mask = jax.tree_util.tree_map(lambda _: False, scene)
    mask = eqx.tree_at(fit_leaves, mask, [True] * n_fit)
    params0, static = eqx.partition(scene, mask)
    _z0, unravel = ravel_pytree(params0)

    z_prior = jaxns_prior(prior, name="z")

    def prior_model():
        z = z_prior.realise()
        return likelihood(forward(eqx.combine(unravel(z), static)))

    return Model(prior_model=prior_model), unravel


class JaxnsBackend(LegacyBackend):
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
        NestedSampler = import_extra("jaxns.core", "jaxns").NestedSampler

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


class NestedInfo(eqx.Module):
    """Diagnostics of a :class:`NestedSampling` run.

    Attributes:
        log_likelihood: Log likelihood of each stored point ``(N,)``.
        ess: jaxns estimate of the Kish effective sample size.
        n_likelihood_evaluations: Total likelihood evaluations.
        termination_reason: Zero for convergence, else a hard-stop code (for example
            the sample buffer filled).
    """

    log_likelihood: jax.Array
    ess: jax.Array
    n_likelihood_evaluations: jax.Array
    termination_reason: jax.Array


class NestedSampling(AbstractBackend):
    """jaxns nested sampling of a ``Fit``: posterior draws and the evidence.

    Live points fill the unit cube and map to the fit's entries through the inverse
    CDF of each leaf law and the unit-cube map of each block (``Fit.from_unit``); a fit
    with a block that has no unit-cube map is refused. The draws are the dead points
    with weights ``L_i dX_i``; the evidence is jaxns's expectation estimate of
    ``log Z`` and its standard error is the standard deviation of ``log Z`` from the
    shrinkage distribution. That error is the shrinkage error only: on a concentrated
    seven-coordinate orbit fit the evidence scattered by about 1 nat over keys against
    a reported 0.3, so read it as good to about 1 nat at the defaults. The sampler is
    jitted internally and runs one fit at a time.

    Args:
        n_live: Number of independent root chains (jaxns ``root_allocation_degree``);
            ``None`` uses the jaxns default, ``30 * D``.
        max_samples: Hard cap on stored samples; jaxns stops early (nonzero
            ``termination_reason``) if the buffer fills first.
    """

    n_live: int | None = eqx.field(static=True, default=None)
    max_samples: int = eqx.field(static=True, default=100_000)

    def _run(self, fit, key, *, batch_size=None):
        """Run nested sampling on the unit cube of ``fit``."""
        NestedSampler = import_extra("jaxns.core", "jaxns").NestedSampler
        Model = import_extra("jaxns.model", "jaxns").Model
        JaxctxPrior = import_extra("jaxctx.priors", "jaxns").AbstractPrior
        unit = fit.unit_template()
        dim = ravel_pytree(unit)[0].shape[0]
        unravel_u = ravel_pytree(unit)[1]
        unravel_x = ravel_pytree(draw_template(fit))[1]

        def to_flat_x(u):
            return ravel_pytree(fit.from_unit(unravel_u(u)))[0]

        class _FitPrior(JaxctxPrior):
            def __init__(self):
                super().__init__(name="x")

            def _dtype(self):
                return jnp.float64

            def _base_shape(self):
                return (dim,)

            def _shape(self):
                return (dim,)

            def _forward(self, U):
                return to_flat_x(U)

            def _inverse(self, X):
                raise NotImplementedError("the unit-cube inverse is not needed")

            def _log_prob(self, X):
                return fit.log_prior(unravel_x(X))

        prior = _FitPrior()

        def prior_model():
            x = prior.realise()
            return fit.log_likelihood(unravel_x(x))

        ns = NestedSampler(
            model=Model(prior_model=prior_model),
            max_samples=int(self.max_samples),
            root_allocation_degree=self.n_live,
        )
        results = ns.run(key=key).to_result().trim()
        flat = jnp.asarray(results.X_samples["x"]).reshape(-1, dim)
        return BackendDraws(
            samples=jax.vmap(unravel_x)(flat),
            log_weights=jnp.asarray(results.log_dp),
            log_evidence=jnp.asarray(results.log_Z_mean),
            log_evidence_se=jnp.asarray(results.log_Z_uncert),
            info=NestedInfo(
                log_likelihood=jnp.asarray(results.log_L),
                ess=jnp.asarray(results.ess),
                n_likelihood_evaluations=jnp.asarray(
                    results.total_num_likelihood_evaluations
                ),
                termination_reason=jnp.asarray(results.termination_reason),
            ),
            kind="monte_carlo",
        )
