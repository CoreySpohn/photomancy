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

``run`` is ``filter_jit``-wrapped, so when ``logprior`` / ``loglikelihood`` are
``SceneLogDensity``-style Modules their forward's array leaves (e.g. a coronagraph PSF
datacube) thread as traced inputs to BlackJAX rather than being baked into the
compiled kernel as constants -- which is what makes coronagraph/IFS-aware evidence
(``log Z``) tractable.
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


class SMCBackend(LegacyBackend):
    """BlackJAX adaptive-tempered SMC, returning a SamplePosterior with log Z.

    Args:
        step_size: Inner-NUTS step size used to move particles each rung.
        n_mcmc_steps: Inner-NUTS steps per tempering rung.
        target_ess: Effective-sample-size fraction that sets the temperature ladder.
    """

    step_size: float = 0.5
    n_mcmc_steps: int = 10
    target_ess: float = 0.5

    @eqx.filter_jit
    def run(self, logprior, loglikelihood, init, key=None):
        """Temper the particles ``init`` (shape ``(n, d)``) from prior to posterior."""
        if key is None:
            raise ValueError("SMCBackend.run requires a PRNG key.")
        blackjax = import_extra("blackjax", "blackjax")
        resampling = import_extra("blackjax.smc.resampling", "blackjax")
        d = init.shape[-1]
        mcmc_parameters = import_extra("blackjax.smc", "blackjax").extend_params(
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


class SMCInfo(eqx.Module):
    """The temperature ladder of an :class:`SMC` run.

    Attributes:
        ladder: Tempering parameter reached after each rung ``(max_rungs,)``, NaN past
            the last rung used.
        n_rungs: Number of rungs used (the loop stops at ``max_rungs``).
        reached: Whether the ladder reached temperature one.
    """

    ladder: jax.Array
    n_rungs: jax.Array
    reached: jax.Array


@eqx.filter_jit
def _smc_run(fit, key, init, step_size, n_mcmc_steps, target_ess, max_rungs):
    """Tempered SMC from prior particles; return particles, weights, log Z, ladder."""
    blackjax = import_extra("blackjax", "blackjax")
    resampling = import_extra("blackjax.smc.resampling", "blackjax")
    unravel, _, dim = flat_problem(fit)

    def logprior(z):
        x, volume = fit.from_real(unravel(z))
        return fit.log_prior(x) + volume

    def loglikelihood(z):
        x, _ = fit.from_real(unravel(z))
        return fit.log_likelihood(x)

    mcmc_parameters = import_extra("blackjax.smc", "blackjax").extend_params(
        {"step_size": step_size, "inverse_mass_matrix": jnp.ones(dim)}
    )
    smc = blackjax.adaptive_tempered_smc(
        logprior,
        loglikelihood,
        blackjax.nuts.build_kernel(),
        blackjax.nuts.init,
        mcmc_parameters,
        resampling.systematic,
        target_ess=target_ess,
        num_mcmc_steps=n_mcmc_steps,
    )
    state = smc.init(init)

    def cond(carry):
        smc_state, _, _, count, _ = carry
        return (smc_state.tempering_param < 1.0) & (count < max_rungs)

    def body(carry):
        smc_state, log_z, k, count, ladder = carry
        k, subk = jax.random.split(k)
        smc_state, info = smc.step(subk, smc_state)
        ladder = ladder.at[count].set(smc_state.tempering_param)
        return smc_state, log_z + info.log_likelihood_increment, k, count + 1, ladder

    ladder0 = jnp.full(max_rungs, jnp.nan)
    final, log_z, _, count, ladder = jax.lax.while_loop(
        cond, body, (state, jnp.asarray(0.0), key, jnp.asarray(0), ladder0)
    )
    return final.particles, jnp.log(final.weights), log_z, ladder, count


class SMC(AbstractBackend):
    """BlackJAX adaptive-tempered SMC with resample-move, in unconstrained space.

    Particles start at prior draws and move from the prior to the posterior along a
    temperature ladder placed so the effective sample size holds at ``target_ess``;
    inner NUTS steps rejuvenate them. The evidence is the sum of the per-rung
    increments. If the ladder does not reach one in ``max_rungs`` rungs the evidence is
    NaN and ``SMCInfo.reached`` is false. The standard error is NaN. Under
    ``eqx.filter_vmap`` the loop runs to the slowest slice.

    Args:
        n_particles: Number of particles.
        step_size: Inner NUTS step size.
        n_mcmc_steps: Inner NUTS steps per rung.
        target_ess: Effective-sample-size fraction that sets the ladder.
        max_rungs: Cap on the number of rungs (and the ladder buffer size).
    """

    n_particles: int = eqx.field(static=True, default=1000)
    step_size: float = 0.5
    n_mcmc_steps: int = eqx.field(static=True, default=10)
    target_ess: float = 0.5
    max_rungs: int = eqx.field(static=True, default=200)

    def _run(self, fit, key, *, batch_size=None):
        """Temper prior particles to the posterior; return them with the evidence."""
        key_init, key_run = jax.random.split(key)
        starts = start_points(fit, None, key_init, self.n_particles)
        particles, log_weights, log_z, ladder, count = _smc_run(
            fit,
            key_run,
            starts,
            self.step_size,
            self.n_mcmc_steps,
            self.target_ess,
            self.max_rungs,
        )
        unravel, _, _ = flat_problem(fit)
        reached = jnp.nanmax(ladder) >= 1.0
        return BackendDraws(
            samples=to_x(fit, unravel, particles),
            log_weights=log_weights,
            log_evidence=jnp.where(reached, log_z, jnp.nan),
            log_evidence_se=jnp.asarray(jnp.nan),
            info=SMCInfo(ladder=ladder, n_rungs=count, reached=reached),
            kind="monte_carlo",
        )
