"""Adaptive-tempered SMC backend: posterior + evidence in one run.

Sequential Monte Carlo walks particles from the prior to the posterior along an
adaptively chosen temperature ladder (each rung is placed so the effective sample
size holds at ``target_ess``). The log marginal likelihood ``log Z`` accumulates
for free as the sum of the per-rung tempering increments -- the upgrade over the
Laplace-mixture evidence that model comparison and experimental design consume.

Unlike the single-logdensity backends, tempered SMC needs the prior and likelihood
separately: ``run(logprior, loglikelihood, init, key)``. The core builds the
combined logdensity from these same plug-ins, so a caller that has the pieces hands
them straight here (or recovers the split as ``loglikelihood = logjoint - logprior``).

TODO (traced-vs-baked forward arrays): ``logprior`` / ``loglikelihood`` are handed to
BlackJAX, which jits them internally, so a Module forward carrying a large array (e.g.
a coronagraph PSF datacube) BAKES that array as a constant here. Thread it as a traced
input the way ``LaplaceBackend`` does (filter_jit the compiled region with the
plug-ins as arguments). See ``AbstractBackend``.
"""

import blackjax
import blackjax.smc.resampling as resampling
import jax
import jax.numpy as jnp
from blackjax.smc import extend_params

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import SamplePosterior


class SMCBackend(AbstractBackend):
    """BlackJAX adaptive-tempered SMC, returning a SamplePosterior with log Z.

    Args:
        step_size: Inner-NUTS step size used to move particles each rung.
        n_mcmc_steps: Inner-NUTS steps per tempering rung.
        target_ess: Effective-sample-size fraction that sets the temperature ladder.
    """

    step_size: float = 0.5
    n_mcmc_steps: int = 10
    target_ess: float = 0.5

    def run(self, logprior, loglikelihood, init, key=None):
        """Temper the particles ``init`` (shape ``(n, d)``) from prior to posterior."""
        if key is None:
            raise ValueError("SMCBackend.run requires a PRNG key.")
        d = init.shape[-1]
        mcmc_parameters = extend_params(
            {"step_size": self.step_size, "inverse_mass_matrix": jnp.ones(d)}
        )
        smc = blackjax.adaptive_tempered_smc(
            logprior,
            loglikelihood,
            blackjax.nuts.build_kernel(),
            blackjax.nuts.init,
            mcmc_parameters,
            resampling.systematic,
            target_ess=self.target_ess,
            num_mcmc_steps=self.n_mcmc_steps,
        )
        state = smc.init(init)

        def cond(carry):
            smc_state, _, _ = carry
            return smc_state.tempering_param < 1.0

        def body(carry):
            smc_state, log_z, k = carry
            k, subk = jax.random.split(k)
            smc_state, info = smc.step(subk, smc_state)
            return smc_state, log_z + info.log_likelihood_increment, k

        final_state, log_z, _ = jax.lax.while_loop(
            cond, body, (state, jnp.asarray(0.0), key)
        )
        return SamplePosterior(
            samples=final_state.particles,
            log_weights=jnp.log(final_state.weights),
            evidence=log_z,
        )
