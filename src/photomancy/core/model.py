"""Assemble a logdensity over a partitioned scene PyTree.

The domain-agnostic core: a fit is a ``logdensity(params)`` built from three
plug-ins -- a forward model (``params -> predicted``), a likelihood
(``predicted -> scalar``), and a prior (``params -> scalar``). The core imports
nothing domain-specific; orbix / skyscapes supply the forward models.
"""

from collections.abc import Callable

import equinox as eqx
from jax.flatten_util import ravel_pytree


def build_logdensity(
    forward_model: Callable,
    likelihood: Callable,
    prior: Callable,
) -> Callable:
    """Compose a logdensity from a forward model, a likelihood, and a prior.

    Args:
        forward_model: Maps a parameter PyTree to predicted data.
        likelihood: Maps predicted data to a scalar log-likelihood (the observed
            data is closed over by the caller).
        prior: Maps a parameter PyTree to a scalar log-prior.

    Returns:
        A function ``logdensity(params) -> scalar`` equal to
        ``prior(params) + likelihood(forward_model(params))``.
    """

    def logdensity(params):
        return prior(params) + likelihood(forward_model(params))

    return logdensity


def build_scene_logdensity(
    scene,
    forward_model: Callable,
    likelihood: Callable,
    prior: Callable,
    filter_spec: Callable = eqx.is_inexact_array,
):
    """Build a flat-position logdensity over a scene Module's differentiable leaves.

    Partitions ``scene`` into differentiable params + a static remainder, ravels
    the params to a flat array (the sampler position), and wraps a logdensity that
    recombines params + static before calling the plug-ins on the full scene. This
    is the scene-as-PyTree hinge: the forward model, likelihood, and prior operate
    on the structured scene, while the sampler sees a single flat array.

    Args:
        scene: An ``eqx.Module`` whose inexact-array leaves are the parameters.
        forward_model: Maps the (recombined) scene to predicted data.
        likelihood: Maps predicted data to a scalar log-likelihood.
        prior: Maps the (recombined) scene to a scalar log-prior.
        filter_spec: Partition filter selecting the differentiable leaves
            (default: ``eqx.is_inexact_array``).

    Returns:
        A tuple ``(logdensity, z0, unravel)`` where ``logdensity(z)`` scores a flat
        position, ``z0`` is the scene's initial flat position, and ``unravel(z)``
        reconstructs the params PyTree.
    """
    params0, static = eqx.partition(scene, filter_spec)
    z0, unravel = ravel_pytree(params0)

    def scene_forward(params):
        return forward_model(eqx.combine(params, static))

    def scene_prior(params):
        return prior(eqx.combine(params, static))

    pytree_logdensity = build_logdensity(scene_forward, likelihood, scene_prior)

    def logdensity(z):
        return pytree_logdensity(unravel(z))

    return logdensity, z0, unravel
