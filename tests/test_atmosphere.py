"""photomancy.atmosphere: the retrieval inference glue (forward + targets + build).

Uses a toy scene mirroring skyscapes' species/log_mmr structure so the glue
is tested without a HITRAN download; the real ExoJAX retrieval lives in the demo script.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from photomancy.atmosphere import (
    abundance_fit_leaves,
    build_retrieval_logdensity,
    contrast_forward,
    default_abundance_prior,
)
from photomancy.backends import LaplaceBackend
from photomancy.priors import Uniform

jax.config.update("jax_enable_x64", True)

_WL = jnp.linspace(0.0, 1.0, 24)
_B0 = jnp.exp(-((_WL - 0.3) ** 2) / 0.01)
_B1 = jnp.exp(-((_WL - 0.7) ** 2) / 0.01)


class _Profile(eqx.Module):
    log_mmr: jnp.ndarray


class _Species(eqx.Module):
    profile: _Profile


class _Atmo(eqx.Module):
    species: tuple


def _atmo(lm0, lm1):
    return _Atmo(
        species=(
            _Species(profile=_Profile(log_mmr=jnp.array([lm0]))),
            _Species(profile=_Profile(log_mmr=jnp.array([lm1]))),
        )
    )


def _toy_forward(pm):
    a0 = jnp.exp(pm.species[0].profile.log_mmr[0])
    a1 = jnp.exp(pm.species[1].profile.log_mmr[0])
    return a0 * _B0 + a1 * _B1


def test_abundance_fit_leaves_selects_log_mmr():
    """abundance_fit_leaves returns each species' log_mmr leaf, in order."""
    leaves = abundance_fit_leaves(_atmo(-0.7, -1.6))
    assert len(leaves) == 2
    assert jnp.allclose(leaves[0], jnp.array([-0.7]))
    assert jnp.allclose(leaves[1], jnp.array([-1.6]))


def test_default_abundance_prior_is_uniform_over_range():
    """default_abundance_prior is a Uniform of the right ndim over the log-MMR box."""
    prior = default_abundance_prior(_atmo(-0.7, -1.6), log_mmr_range=(-12.0, 0.0))
    assert isinstance(prior, Uniform)
    assert prior.ndim == 2
    assert jnp.allclose(prior.low, -12.0)
    assert jnp.allclose(prior.high, 0.0)


def test_build_retrieval_logdensity_recovers_abundances():
    """A Laplace retrieval recovers the two abundances from a noisy toy spectrum."""
    truth = _atmo(float(jnp.log(0.5)), float(jnp.log(0.2)))
    clean = np.asarray(_toy_forward(truth))
    sigma = 0.02 * float(np.max(clean))
    rng = np.random.default_rng(0)
    data = jnp.asarray(clean + rng.normal(0.0, sigma, clean.shape))

    init = _atmo(float(jnp.log(0.5)) + 0.5, float(jnp.log(0.2)) - 0.5)
    prior = default_abundance_prior(init)
    logdensity, z0, unravel = build_retrieval_logdensity(
        init,
        data,
        fit_leaves=abundance_fit_leaves,
        noise_sigma=sigma,
        forward=_toy_forward,
        prior=prior,
    )
    assert jnp.isfinite(logdensity(z0))
    post = LaplaceBackend(n_steps=500, min_eigenvalue=1e-6).run(logdensity, z0)
    fit = unravel(post.mean)
    assert abs(float(fit.species[0].profile.log_mmr[0]) - float(jnp.log(0.5))) < 0.1
    assert abs(float(fit.species[1].profile.log_mmr[0]) - float(jnp.log(0.2))) < 0.1


def test_contrast_forward_extracts_single_planet_spectrum():
    """contrast_forward reads contrast_cube and returns the (n_wl,) planet spectrum."""

    class _CubeAtmo(eqx.Module):
        spec: jnp.ndarray

        def contrast_cube(self, phase, dist_pc, wavelengths_nm, Rp):
            return self.spec[:, None, None]  # (n_wl, 1 planet, 1 column)

    fwd = contrast_forward(
        phase=jnp.zeros((1, 1)),
        dist_pc=jnp.ones((1, 1)),
        wavelengths_nm=jnp.array([500.0, 600.0, 700.0]),
        Rp=jnp.ones((1,)),
    )
    out = fwd(_CubeAtmo(spec=jnp.array([1.0, 2.0, 3.0])))
    assert jnp.allclose(out, jnp.array([1.0, 2.0, 3.0]))
