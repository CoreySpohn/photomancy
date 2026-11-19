"""NUTS backend: BlackJAX No-U-Turn sampler -> SamplePosterior.

Window adaptation tunes the step size and mass matrix, then the chain runs through
``run_inference_algorithm`` (a single ``lax.scan``, O(1) compile regardless of
chain length). The draws are equally weighted and carry no evidence estimate
(``evidence`` is ``NaN``); use the SMC backend when you need ``log Z``.

``run`` is ``filter_jit``-wrapped, so when ``logdensity`` is a ``SceneLogDensity``
Module its forward's array leaves (e.g. a coronagraph PSF datacube) thread as traced
inputs to BlackJAX rather than being baked into the compiled kernel as constants.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.infer.backends._flat import flat_problem, start_points, to_x
from photomancy.infer.backends.base import (
    AbstractBackend,
    BackendDraws,
    LegacyBackend,
    import_extra,
)
from photomancy.infer.legacy_posterior import SamplePosterior
from photomancy.infer.posterior import Posterior


class NUTSBackend(LegacyBackend):
    """BlackJAX NUTS with window adaptation, returning a SamplePosterior.

    Args:
        n_warmup: Window-adaptation steps (tunes the step size + mass matrix).
        n_samples: Post-warmup samples drawn from the tuned kernel.
        dense_mass_matrix: If ``True`` (default), adapt a full (dense) mass matrix
            rather than a diagonal one. The orbit and disk posteriors are strongly
            correlated and low-dimensional, where a dense matrix converges much
            faster; set ``False`` for high-dimensional targets where the ``O(d^2)``
            mass matrix is too costly.
    """

    n_warmup: int = 500
    n_samples: int = 2000
    dense_mass_matrix: bool = True

    @eqx.filter_jit
    def run(self, logdensity, init, key=None):
        """Adapt, then sample NUTS from ``init``; returns equal-weight samples."""
        if key is None:
            raise ValueError("NUTSBackend.run requires a PRNG key.")
        blackjax = import_extra("blackjax", "blackjax")
        run_inference_algorithm = import_extra(
            "blackjax.util", "blackjax"
        ).run_inference_algorithm
        key_warmup, key_sample = jax.random.split(key)

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity,
            is_mass_matrix_diagonal=not self.dense_mass_matrix,
        )
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


class NUTSInfo(eqx.Module):
    """Chain structure and flags of a :class:`NUTS` run, one entry per stored draw.

    Attributes:
        chain: Chain index of each draw ``(N,)``; draws are stored chain by chain.
        divergent: Whether the transition that produced the draw diverged ``(N,)``.
        tree_depth: Number of subtrajectory expansions of that transition ``(N,)``.
        step_size: Adapted step size of each chain ``(n_chains,)``.
        thin: Thinning factor: every ``thin``-th transition is stored.
    """

    chain: jax.Array
    divergent: jax.Array
    tree_depth: jax.Array
    step_size: jax.Array
    thin: int = eqx.field(static=True)


@eqx.filter_jit
def _nuts_chains(fit, key, init, n_warmup, n_draws, dense, thin):
    """Window-adapt and sample one NUTS chain per start; return positions and flags."""
    blackjax = import_extra("blackjax", "blackjax")
    run_inference_algorithm = import_extra(
        "blackjax.util", "blackjax"
    ).run_inference_algorithm
    _, logdensity, _ = flat_problem(fit)
    keys = jax.random.split(key, init.shape[0])

    def one_chain(start, chain_key):
        key_warmup, key_sample = jax.random.split(chain_key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity, is_mass_matrix_diagonal=not dense
        )
        (state, params), _ = warmup.run(key_warmup, start, num_steps=n_warmup)
        kernel = blackjax.nuts(logdensity, **params)
        _, (positions, divergent, depth) = run_inference_algorithm(
            rng_key=key_sample,
            initial_state=state,
            inference_algorithm=kernel,
            num_steps=n_draws,
            transform=lambda s, info: (
                s.position,
                info.is_divergent,
                info.num_trajectory_expansions,
            ),
        )
        return positions[::thin], divergent[::thin], depth[::thin], params["step_size"]

    return jax.vmap(one_chain)(init, keys)


class NUTS(AbstractBackend):
    """BlackJAX NUTS with window adaptation, one chain per start.

    Chains start at prior draws (or at ``init``), adapt a step size and mass matrix
    during warmup, then sample. The draws are equally weighted, stored chain by chain
    on one flat draw axis, and carry no evidence: ``log_evidence`` is NaN (use
    ``Laplace(init=posterior)`` or ``SMC`` for an evidence). Angle coordinates of an
    orbit block are plain reals, so a chain may wander across copies of the same cell.

    Args:
        n_warmup: Window-adaptation steps per chain.
        n_draws: Transitions per chain after warmup.
        n_chains: Number of chains, vectorized.
        thin: Keep every ``thin``-th transition; ``NUTSInfo.thin`` records it.
        dense_mass_matrix: Adapt a dense (default) or diagonal mass matrix.
        init: ``None`` for prior draws, one constrained draw (a dict of the fit's
            entries), or a ``Posterior`` whose weights pick the starts.
    """

    n_warmup: int = eqx.field(static=True, default=500)
    n_draws: int = eqx.field(static=True, default=1000)
    n_chains: int = eqx.field(static=True, default=4)
    thin: int = eqx.field(static=True, default=1)
    dense_mass_matrix: bool = eqx.field(static=True, default=True)
    init: Posterior | dict | None = None

    def _run(self, fit, key, *, batch_size=None):
        """Adapt and sample every chain; return flat draws with uniform weights."""
        key_init, key_run = jax.random.split(key)
        starts = start_points(fit, self.init, key_init, self.n_chains)
        positions, divergent, depth, step = _nuts_chains(
            fit,
            key_run,
            starts,
            self.n_warmup,
            self.n_draws,
            self.dense_mass_matrix,
            self.thin,
        )
        unravel, _, dim = flat_problem(fit)
        n_kept = positions.shape[1]
        flat = positions.reshape(self.n_chains * n_kept, dim)
        info = NUTSInfo(
            chain=jnp.repeat(jnp.arange(self.n_chains), n_kept),
            divergent=divergent.reshape(-1),
            tree_depth=depth.reshape(-1),
            step_size=step,
            thin=self.thin,
        )
        return BackendDraws(
            samples=to_x(fit, unravel, flat),
            log_weights=jnp.zeros(flat.shape[0]),
            log_evidence=jnp.asarray(jnp.nan),
            log_evidence_se=jnp.asarray(jnp.nan),
            info=info,
            kind="monte_carlo",
        )
