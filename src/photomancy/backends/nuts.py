"""NUTS backend: BlackJAX No-U-Turn sampler -> SamplePosterior.

Window adaptation tunes the step size and mass matrix, then the chain runs through
``run_inference_algorithm`` (a single ``lax.scan``, O(1) compile regardless of
chain length). The draws are equally weighted and carry no evidence estimate
(``evidence`` is ``NaN``); use the SMC backend when you need ``log Z``.

``run`` is ``filter_jit``-wrapped, so when ``logdensity`` is a ``SceneLogDensity``
Module its forward's array leaves (e.g. a coronagraph PSF datacube) thread as traced
inputs to BlackJAX rather than being baked into the compiled kernel as constants.
"""

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.util import run_inference_algorithm

from photomancy.backends.base import AbstractBackend
from photomancy.posterior import SamplePosterior


class NUTSBackend(AbstractBackend):
    """BlackJAX NUTS with window adaptation, returning a SamplePosterior.

    Args:
        n_warmup: Window-adaptation steps (tunes the step size + mass matrix).
        n_samples: Post-warmup samples drawn from the tuned kernel.
    """

    n_warmup: int = 500
    n_samples: int = 2000

    @eqx.filter_jit
    def run(self, logdensity, init, key=None):
        """Adapt, then sample NUTS from ``init``; returns equal-weight samples."""
        if key is None:
            raise ValueError("NUTSBackend.run requires a PRNG key.")
        key_warmup, key_sample = jax.random.split(key)

        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        (state, params), _ = warmup.run(key_warmup, init, num_steps=self.n_warmup)
        kernel = blackjax.nuts(logdensity, **params)

        _, samples = run_inference_algorithm(
            rng_key=key_sample,
            initial_state=state,
            inference_algorithm=kernel,
            num_steps=self.n_samples,
            transform=lambda chain_state, info: chain_state.position,
        )
        return SamplePosterior(
            samples=samples,
            log_weights=jnp.zeros(self.n_samples),
            evidence=jnp.asarray(jnp.nan),
        )
