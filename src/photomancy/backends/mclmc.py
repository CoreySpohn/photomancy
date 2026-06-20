"""MCLMC backend: BlackJAX microcanonical Langevin Monte Carlo -> SamplePosterior.

Tuning (``mclmc_find_L_and_step_size``) sets the trajectory length ``L``, the step
size, and a diagonal mass matrix, then the chain runs through
``run_inference_algorithm`` (a single ``lax.scan``, O(1) compile regardless of chain
length). MCLMC is a gradient-based sampler that is often cheaper per effective sample
than NUTS in high dimension. Like NUTS, the draws are equally weighted and carry no
evidence estimate (``evidence`` is ``NaN``); use the SMC or nested backend when you
need ``log Z``.

``run`` is ``filter_jit``-wrapped, so when ``logdensity`` is a ``SceneLogDensity``
Module its forward's array leaves (e.g. a coronagraph PSF datacube) thread as traced
inputs to BlackJAX rather than being baked into the compiled kernel as constants.
"""

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.mcmc.mclmc import build_kernel
from blackjax.mcmc.mclmc import init as mclmc_init
from blackjax.util import run_inference_algorithm

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import SamplePosterior


class MCLMCBackend(AbstractBackend):
    """BlackJAX microcanonical Langevin Monte Carlo, returning a SamplePosterior.

    Args:
        n_tune: Tuning steps for the trajectory length, step size, and mass matrix.
        n_samples: Post-tuning samples drawn from the tuned kernel.
    """

    n_tune: int = 2000
    n_samples: int = 2000

    @eqx.filter_jit
    def run(self, logdensity, init, key=None):
        """Tune L / step size / mass matrix, then sample MCLMC from ``init``."""
        if key is None:
            raise ValueError("MCLMCBackend.run requires a PRNG key.")
        key_init, key_tune, key_sample = jax.random.split(key, 3)

        state = mclmc_init(position=init, logdensity_fn=logdensity, rng_key=key_init)

        def kernel(inverse_mass_matrix):
            return build_kernel(
                logdensity_fn=logdensity,
                integrator=isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        tuned_state, params, *_ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=self.n_tune,
            state=state,
            rng_key=key_tune,
        )

        sampler = blackjax.mclmc(
            logdensity,
            L=params.L,
            step_size=params.step_size,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )
        _, samples = run_inference_algorithm(
            rng_key=key_sample,
            initial_state=tuned_state,
            inference_algorithm=sampler,
            num_steps=self.n_samples,
            transform=lambda chain_state, info: chain_state.position,
        )
        return SamplePosterior(
            samples=samples,
            log_weights=jnp.zeros(self.n_samples),
            evidence=jnp.asarray(jnp.nan),
        )
