"""Tests for the scene-PyTree boundary: partition + ravel + logdensity."""

import equinox as eqx
import jax.numpy as jnp

from photomancy.core import build_scene_logdensity


class _ToyScene(eqx.Module):
    mu: jnp.ndarray
    label: str = eqx.field(static=True)


def test_scene_logdensity_peaks_at_truth_over_flat_position():
    """build_scene_logdensity ravels a scene Module into a flat logdensity."""
    obs = 2.0

    def forward(scene):
        return scene.mu

    def likelihood(pred):
        return -0.5 * ((pred - obs) / 0.5) ** 2

    def prior(scene):
        return -0.5 * (scene.mu / 10.0) ** 2

    scene = _ToyScene(mu=jnp.asarray(0.0), label="toy")
    logdensity, z0, unravel = build_scene_logdensity(scene, forward, likelihood, prior)

    # Only the inexact-array leaf (mu) is sampled; the static label is excluded.
    assert z0.shape == (1,)
    assert logdensity(jnp.asarray([2.0])) > logdensity(jnp.asarray([-3.0]))
    assert jnp.allclose(unravel(jnp.asarray([2.0])).mu, 2.0)
