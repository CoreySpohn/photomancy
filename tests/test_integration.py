"""End-to-end acceptance: a scene Module fit through a backend.

Exercises the full seam -- partition an ``eqx.Module`` scene into a flat
logdensity (``core.build_scene_logdensity``), run it through a generic
``Backend``, and read the unified ``Posterior`` -- on a toy Gaussian scene with a
known truth. The forward model, likelihood, and prior here stand in for the
orbix / skyscapes equations a real fit supplies.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.backends import LaplaceBackend
from photomancy.core import build_scene_logdensity
from photomancy.posterior import GaussianPosterior


class _ToyScene(eqx.Module):
    mu: jnp.ndarray
    label: str = eqx.field(static=True)


def test_scene_fit_through_laplace_brackets_truth():
    """build_scene_logdensity -> LaplaceBackend recovers a toy scene's truth."""
    truth = jnp.array([1.5, -0.5])
    sigma = 0.3
    obs = truth + sigma * jax.random.normal(jax.random.key(0), truth.shape)

    def forward(scene):
        return scene.mu

    def likelihood(pred):
        return -0.5 * jnp.sum(((pred - obs) / sigma) ** 2)

    def prior(scene):
        return -0.5 * jnp.sum((scene.mu / 10.0) ** 2)

    scene = _ToyScene(mu=jnp.zeros(2), label="toy")
    logdensity, z0, unravel = build_scene_logdensity(scene, forward, likelihood, prior)

    post = LaplaceBackend().run(logdensity, z0)

    assert isinstance(post, GaussianPosterior)

    std = jnp.sqrt(jnp.diag(post.cov))
    assert jnp.all(jnp.abs(post.mean - truth) < 3.0 * std)
    # broad prior -> the posterior scale tracks the likelihood sigma
    assert jnp.allclose(std, sigma, atol=0.05)
    # the MAP maps back onto the structured scene
    assert unravel(post.mean).mu.shape == (2,)
