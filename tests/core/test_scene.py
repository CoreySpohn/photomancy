"""Tests for the scene-PyTree boundary: partition + ravel + logdensity."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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


def test_forward_module_arrays_are_traced_not_baked():
    """Module-forward arrays stay traced inputs, not baked constants.

    A forward written as an eqx.Module exposes its arrays as PyTree leaves of the
    logdensity, so a filter_jit'd backend threads them as traced inputs instead of
    constant-folding them into the kernel (e.g. a coronagraph PSF datacube).
    """
    kernel = jnp.ones((32, 32))

    class _KernelForward(eqx.Module):
        kernel: jnp.ndarray

        def __call__(self, scene):
            return scene.mu + jnp.sum(self.kernel)

    def likelihood(pred):
        return -0.5 * (pred - 1.0) ** 2

    def prior(scene):
        return 0.0

    scene = _ToyScene(mu=jnp.asarray(0.0), label="toy")
    logdensity, z0, _ = build_scene_logdensity(
        scene, _KernelForward(kernel), likelihood, prior
    )

    # The forward's kernel is a (traced) array leaf of the logdensity Module.
    leaves = jax.tree_util.tree_leaves(eqx.filter(logdensity, eqx.is_array))
    assert any(leaf.shape == (32, 32) for leaf in leaves), "kernel not a leaf"

    # When the dynamic partition is the jit input, the kernel is an INVAR
    # (traced input buffer), not a CONSTVAR (baked into the kernel).
    dynamic, static = eqx.partition(logdensity, eqx.is_array)

    def call(dyn, z):
        return eqx.combine(dyn, static)(z)

    jaxpr = jax.make_jaxpr(call)(dynamic, z0)
    invar_sizes = {int(np.prod(v.aval.shape)) for v in jaxpr.jaxpr.invars}
    constvar_sizes = [int(np.prod(v.aval.shape)) for v in jaxpr.jaxpr.constvars]
    assert 32 * 32 in invar_sizes, "kernel is not a traced input"
    assert all(s < 32 * 32 for s in constvar_sizes), "kernel was baked as a constant"
