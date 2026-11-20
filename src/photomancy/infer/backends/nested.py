"""Nested-sampling backend (jaxns): posterior + Bayesian evidence (log Z).

Unlike the MAP / MCMC backends, nested sampling needs a proper, sampleable PRIOR (it
integrates over the prior to get the evidence). So this backend consumes a jaxns
``Model`` built from a ``Fit`` (a prior model that realizes its priors and returns the
log-likelihood). The headline output is the evidence
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
    import_extra,
)

# jaxns is an optional extra and is imported inside the functions below: importing it
# enables JAX x64 globally as a side effect and pulls in tensorflow-probability
# transitively, so ``import photomancy`` stays light and pays that only when nested
# sampling runs. The prior adapter uses only the jaxctx ``AbstractPrior`` interface.


class NestedInfo(eqx.Module):
    """Diagnostics of a :class:`NestedSampling` run.

    Attributes:
        log_likelihood: Log likelihood of each stored point ``(N,)``.
        n_eff: jaxns estimate of the Kish effective sample size.
        n_likelihood_evaluations: Total likelihood evaluations.
        termination_reason: Zero when jaxns met its own stopping rule, else a
            hard-stop code (1: the sample buffer filled). A nonzero code makes the
            evidence NaN.
    """

    log_likelihood: jax.Array
    n_eff: jax.Array
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
        max_samples: Hard cap on stored samples. If the buffer fills first, jaxns
            stops with a nonzero ``termination_reason`` (1 is "reached max samples")
            before the evidence has converged: the samples and weights are returned
            as they stand and the evidence and its error are NaN, a refusal under the
            contract on :attr:`~photomancy.infer.posterior.Posterior.log_evidence`.
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
        converged = jnp.asarray(results.termination_reason) == 0
        nan = jnp.asarray(jnp.nan, jnp.float64)
        return BackendDraws(
            samples=jax.vmap(unravel_x)(flat),
            log_weights=jnp.asarray(results.log_dp),
            log_evidence=jnp.where(converged, jnp.asarray(results.log_Z_mean), nan),
            log_evidence_se=jnp.where(
                converged, jnp.asarray(results.log_Z_uncert), nan
            ),
            info=NestedInfo(
                log_likelihood=jnp.asarray(results.log_L),
                n_eff=jnp.asarray(results.ess),
                n_likelihood_evaluations=jnp.asarray(
                    results.total_num_likelihood_evaluations
                ),
                termination_reason=jnp.asarray(results.termination_reason),
            ),
            kind="monte_carlo",
        )
