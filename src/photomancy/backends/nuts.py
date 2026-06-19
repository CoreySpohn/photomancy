"""NUTS backend: BlackJAX No-U-Turn sampler -> SamplePosterior.

Window adaptation tunes the step size and mass matrix, then the chain runs through
``run_inference_algorithm`` (a single ``lax.scan``, O(1) compile regardless of
chain length). The draws are equally weighted and carry no evidence estimate
(``evidence`` is ``NaN``); use the SMC backend when you need ``log Z``.

TODO (traced-vs-baked forward arrays): ``logdensity`` is handed to BlackJAX, which
jits it internally, so a ``SceneLogDensity`` whose forward carries a large array
(e.g. a coronagraph PSF datacube) BAKES that array as a constant here. Thread it as
a traced input the way ``LaplaceBackend`` does (filter_jit the compiled region with
the logdensity as an argument). See ``AbstractBackend``.
"""

import blackjax
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
