"""photomancy.atmosphere: the retrieval inference glue (forward + targets + build).

Uses a toy scene mirroring skyscapes' species/log_mmr structure so the glue
is tested without a HITRAN download; the real ExoJAX retrieval lives in the demo script.
"""

import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.model_paths import resolve
from priorities import Uniform

jax.config.update("jax_enable_x64", True)

from photomancy.atmosphere import (  # noqa: E402
    abundance_paths,
    contrast_forward,
    default_abundance_prior,
)
from photomancy.infer import MAP, Fit, ForwardTerm, Laplace, run  # noqa: E402
from photomancy.infer.diagnostics import weighted_mean  # noqa: E402
from photomancy.infer.likelihoods import gaussian_loglike  # noqa: E402

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "forward_fit_golden.json").read_text()
)["atmosphere"]

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


class _ToyForward(eqx.Module):
    """``pm -> a0 * B0 + a1 * B1`` from the two species abundances."""

    def __call__(self, pm):
        a0 = jnp.exp(pm.species[0].profile.log_mmr[0])
        a1 = jnp.exp(pm.species[1].profile.log_mmr[0])
        return a0 * _B0 + a1 * _B1


_toy_forward = _ToyForward()


def _toy_data():
    """The noisy toy spectrum at the truth abundances, and its noise sigma."""
    clean = np.asarray(_toy_forward(_atmo(float(jnp.log(0.5)), float(jnp.log(0.2)))))
    sigma = 0.02 * float(np.max(clean))
    rng = np.random.default_rng(0)
    return jnp.asarray(clean + rng.normal(0.0, sigma, clean.shape)), sigma


def _fit(init, data, sigma):
    prior = default_abundance_prior(init)
    return Fit(init, prior, (ForwardTerm(_toy_forward, gaussian_loglike(data, sigma)),))


def test_abundance_paths_name_each_species_log_mmr():
    """abundance_paths names each species' log_mmr leaf, in order, and they resolve."""
    pm = _atmo(-0.7, -1.6)
    paths = abundance_paths(pm)
    assert paths == ("species[0].profile.log_mmr", "species[1].profile.log_mmr")
    assert jnp.allclose(resolve(pm, paths[0]), jnp.array([-0.7]))
    assert jnp.allclose(resolve(pm, paths[1]), jnp.array([-1.6]))


def test_default_abundance_prior_is_uniform_over_range():
    """default_abundance_prior is one Uniform per path, of the leaf's shape."""
    pm = _atmo(-0.7, -1.6)
    prior = default_abundance_prior(pm, log_mmr_range=(-12.0, 0.0))
    assert tuple(prior) == abundance_paths(pm)
    for law in prior.values():
        assert isinstance(law, Uniform)
        assert law.low.shape == (1,)
        assert jnp.allclose(law.low, -12.0)
        assert jnp.allclose(law.high, 0.0)
    only = default_abundance_prior(pm, paths=abundance_paths(pm)[:1])
    assert tuple(only) == abundance_paths(pm)[:1]


def test_fit_density_matches_the_legacy_density_up_to_the_gaussian_constant():
    """The fit density minus the Gaussian normalization is the stored legacy density.

    The legacy density was the unnormalized Gaussian plus the Uniform prior, captured at
    three points before the port (``data/forward_fit_golden.json``).
    Tolerance basis: floating-point (rtol 1e-12, float64).
    Fails if: a path, a law or the forward changed in the port.
    """
    data, sigma = _toy_data()
    init = _atmo(float(jnp.log(0.5)) + 0.5, float(jnp.log(0.2)) - 0.5)
    fit = _fit(init, data, sigma)
    n = GOLDEN["n_points"]
    assert data.size == n
    constant = -0.5 * n * math.log(2.0 * math.pi) - n * math.log(sigma)
    for name, (lm0, lm1) in GOLDEN["points"].items():
        x = {
            "species[0].profile.log_mmr": jnp.array([lm0]),
            "species[1].profile.log_mmr": jnp.array([lm1]),
        }
        np.testing.assert_allclose(
            float(fit.log_density(x)) - constant,
            GOLDEN["legacy_log_density"][name],
            rtol=1e-12,
        )


@pytest.mark.parametrize("backend", ["map", "laplace"])
def test_fit_recovers_abundances(backend):
    """MAP and Laplace recover the two abundances from a noisy toy spectrum."""
    data, sigma = _toy_data()
    init = _atmo(float(jnp.log(0.5)) + 0.5, float(jnp.log(0.2)) - 0.5)
    fit = _fit(init, data, sigma)
    x0 = {path: resolve(init, path) for path in fit.free}
    assert jnp.isfinite(fit.log_density(x0))
    if backend == "map":
        post = run(fit, MAP(n_steps=500, init=x0), jax.random.key(0))
        mean = {p: v[0] for p, v in post.samples.items()}
    else:
        post = run(fit, Laplace(n_draws=512, n_steps=500, init=x0), jax.random.key(0))
        mean = {p: weighted_mean(post.log_weights, v) for p, v in post.samples.items()}
    assert abs(float(mean["species[0].profile.log_mmr"][0]) - float(jnp.log(0.5))) < 0.1
    assert abs(float(mean["species[1].profile.log_mmr"][0]) - float(jnp.log(0.2))) < 0.1


def test_contrast_forward_extracts_single_planet_spectrum():
    """contrast_forward reads contrast_cube and returns the (n_wl,) planet spectrum."""

    class _CubeAtmo(eqx.Module):
        spec: jnp.ndarray

        def contrast_cube(
            self, *, phase_angle_rad, star_planet_distance_AU, wavelengths_nm, Rp_Rearth
        ):
            np.testing.assert_array_equal(star_planet_distance_AU, [[2.5]])
            return self.spec[:, None, None]  # (n_wl, 1 planet, 1 column)

    fwd = contrast_forward(
        phase_angle_rad=jnp.zeros((1, 1)),
        dist_AU=jnp.full((1, 1), 2.5),
        wavelengths_nm=jnp.array([500.0, 600.0, 700.0]),
        Rp_Rearth=jnp.ones((1,)),
    )
    out = fwd(_CubeAtmo(spec=jnp.array([1.0, 2.0, 3.0])))
    assert jnp.allclose(out, jnp.array([1.0, 2.0, 3.0]))
