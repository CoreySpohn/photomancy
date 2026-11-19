"""MCLMC backend: BlackJAX microcanonical Langevin Monte Carlo -> Posterior.

Tuning (``mclmc_find_L_and_step_size``) sets the trajectory length ``L``, the step
size, and a diagonal mass matrix, then the chain runs through
``run_inference_algorithm`` (a single ``lax.scan``, O(1) compile regardless of chain
length). MCLMC is a gradient-based sampler that is often cheaper per effective sample
than NUTS in high dimension. Like NUTS, the draws are equally weighted and carry no
evidence estimate (``log_evidence`` is ``NaN``); use the nested backend when you
need ``log Z``.

The chains run inside a ``filter_jit``, so the array leaves of the fit's forward (e.g. a
coronagraph PSF datacube) thread as traced inputs to BlackJAX rather than being baked
into the compiled kernel as constants.

BlackJAX 1.6 made ``logdensity_fn`` a required argument of the tuner (it raises when
left at its ``None`` default) and renamed ``Lfactor`` to ``l_factor``; the pinned floor
is 1.6 because 1.5 does not accept the keyword at all.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.infer.backends._flat import flat_problem, start_points, to_x
from photomancy.infer.backends.base import (
    AbstractBackend,
    BackendDraws,
    import_extra,
)
from photomancy.infer.posterior import Posterior


class MCLMCInfo(eqx.Module):
    """Chain structure and tuned settings of an :class:`MCLMC` run.

    Attributes:
        chain: Chain index of each draw ``(N,)``; draws are stored chain by chain.
        energy_change: Energy change of the transition that produced each draw ``(N,)``.
        trajectory_length: Tuned ``L`` of each chain ``(n_chains,)``.
        step_size: Tuned step size of each chain ``(n_chains,)``.
        thin: Thinning factor: every ``thin``-th transition is stored.
    """

    chain: jax.Array
    energy_change: jax.Array
    trajectory_length: jax.Array
    step_size: jax.Array
    thin: int = eqx.field(static=True)


@eqx.filter_jit
def _mclmc_chains(fit, key, init, n_tune, n_draws, thin):
    """Tune and sample one MCLMC chain per start; return positions and tuned values."""
    blackjax = import_extra("blackjax", "blackjax")
    mclmc = import_extra("blackjax.mcmc.mclmc", "blackjax")
    integrators = import_extra("blackjax.mcmc.integrators", "blackjax")
    run_inference_algorithm = import_extra(
        "blackjax.util", "blackjax"
    ).run_inference_algorithm
    _, logdensity, _ = flat_problem(fit)
    keys = jax.random.split(key, init.shape[0])

    def one_chain(start, chain_key):
        key_init, key_tune, key_sample = jax.random.split(chain_key, 3)
        state = mclmc.init(position=start, logdensity_fn=logdensity, rng_key=key_init)
        kernel = mclmc.build_kernel(integrator=integrators.isokinetic_mclachlan)
        tuned_state, params, *_ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=n_tune,
            state=state,
            rng_key=key_tune,
            logdensity_fn=logdensity,
        )
        sampler = blackjax.mclmc(
            logdensity,
            L=params.L,
            step_size=params.step_size,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )
        _, (positions, energy) = run_inference_algorithm(
            rng_key=key_sample,
            initial_state=tuned_state,
            inference_algorithm=sampler,
            num_steps=n_draws,
            transform=lambda s, info: (s.position, info.energy_change),
        )
        return positions[::thin], energy[::thin], params.L, params.step_size

    return jax.vmap(one_chain)(init, keys)


class MCLMC(AbstractBackend):
    """BlackJAX microcanonical Langevin Monte Carlo, one chain per start.

    Tunes the trajectory length, step size and a diagonal mass matrix, then samples.
    Draws are equally weighted, stored chain by chain, and carry no evidence
    (``log_evidence`` is NaN).

    Args:
        n_tune: Tuning steps per chain.
        n_draws: Transitions per chain after tuning.
        n_chains: Number of chains, vectorized.
        thin: Keep every ``thin``-th transition; ``MCLMCInfo.thin`` records it.
        init: ``None`` for prior draws, one constrained draw (repeated for every
            chain), an entry dict whose every leaf has a leading axis of length
            ``n_chains`` (one explicit start each), or a ``Posterior`` whose weights
            pick the starts; the starts are placed in the flat vector by
            ``fit.to_flat``.
    """

    n_tune: int = eqx.field(static=True, default=2000)
    n_draws: int = eqx.field(static=True, default=1000)
    n_chains: int = eqx.field(static=True, default=4)
    thin: int = eqx.field(static=True, default=1)
    init: Posterior | dict | None = None

    def _run(self, fit, key, *, batch_size=None):
        """Tune and sample every chain; return flat draws with uniform weights."""
        key_init, key_run = jax.random.split(key)
        starts = start_points(fit, self.init, key_init, self.n_chains)
        positions, energy, length, step = _mclmc_chains(
            fit, key_run, starts, self.n_tune, self.n_draws, self.thin
        )
        unravel, _, dim = flat_problem(fit)
        n_kept = positions.shape[1]
        flat = positions.reshape(self.n_chains * n_kept, dim)
        info = MCLMCInfo(
            chain=jnp.repeat(jnp.arange(self.n_chains), n_kept),
            energy_change=energy.reshape(-1),
            trajectory_length=length,
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
