"""Assemble a logdensity over a partitioned scene PyTree.

The domain-agnostic core: a fit is a ``logdensity(params)`` built from three
plug-ins -- a forward model (``params -> predicted``), a likelihood
(``predicted -> scalar``), and a prior (``params -> scalar``). The core imports
nothing domain-specific; orbix / skyscapes supply the forward models.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.priors import AbstractPrior


class SceneLogDensity(eqx.Module):
    """A callable logdensity over a partitioned scene; array deps stay PyTree leaves.

    Holding ``forward_model`` / ``likelihood`` / ``prior`` / ``static`` as fields --
    rather than closing over them in a bare function -- keeps any arrays they carry as
    leaves of this Module. In particular, a forward written as an ``eqx.Module`` (e.g.
    a coronagraph forward holding a PSF datacube) exposes its arrays here. A backend
    that ``filter_jit``s with this logdensity as an argument then threads those arrays
    as traced inputs instead of constant-folding them into the compiled kernel.

    Arrays captured inside a *closure* forward/likelihood (rather than a Module) remain
    hidden and are baked as before; keep those small (e.g. the observed image) and make
    any large-array forward a Module.
    """

    forward_model: Callable
    likelihood: Callable
    prior: Callable | AbstractPrior
    static: Any
    unravel: Callable = eqx.field(static=True)

    def __call__(self, z):
        """Score a flat position ``z``: ``prior + likelihood(forward(scene))``.

        An :class:`~photomancy.priors.AbstractPrior` is scored in z-space
        (``prior.log_prob(z)``); a plain callable is scored on the recombined scene.
        """
        scene = eqx.combine(self.unravel(z), self.static)
        if isinstance(self.prior, AbstractPrior):
            log_prior = self.prior.log_prob(z)
        else:
            log_prior = self.prior(scene)
        return log_prior + self.likelihood(self.forward_model(scene))


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
    prior: Callable | AbstractPrior,
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
        prior: Either an :class:`~photomancy.priors.AbstractPrior` (scored in z-space
            as ``prior.log_prob(z)``) or a callable mapping the scene to a scalar
            log-prior.
        filter_spec: Partition filter selecting the differentiable leaves
            (default: ``eqx.is_inexact_array``).

    Returns:
        A tuple ``(logdensity, z0, unravel)`` where ``logdensity(z)`` scores a flat
        position, ``z0`` is the scene's initial flat position, and ``unravel(z)``
        reconstructs the params PyTree.
    """
    params0, static = eqx.partition(scene, filter_spec)
    z0, unravel = ravel_pytree(params0)

    logdensity = SceneLogDensity(
        forward_model=forward_model,
        likelihood=likelihood,
        prior=prior,
        static=static,
        unravel=unravel,
    )
    return logdensity, z0, unravel


def build_gaussian_fit(
    scene,
    data,
    *,
    fit_leaves: Callable,
    noise_sigma,
    forward: Callable,
    prior: Callable | AbstractPrior | None = None,
):
    """Fit selected scene leaves to ``data`` under iid Gaussian noise.

    The shared shape of the disk / atmosphere (and any image- or spectrum-) fit: select
    the leaves ``fit_leaves(scene)``, form a Gaussian likelihood
    ``-0.5 * sum((forward(scene) - data)**2 / noise_sigma**2)``, and delegate to
    :func:`build_scene_logdensity`. The forward is injected, so the physics (a skyscapes
    render) stays out of photomancy and a coronagraph forward drops in unchanged.

    Args:
        scene: The scene ``eqx.Module`` (its selected leaves are the parameters).
        data: Observed data array (image, spectrum, ...), broadcastable to the forward.
        fit_leaves: ``scene -> list[leaf]`` selecting the leaves to fit.
        noise_sigma: Per-element Gaussian noise sigma (scalar or broadcastable).
        forward: ``scene -> predicted`` (the injected, swappable physics).
        prior: Optional prior forwarded to :func:`build_scene_logdensity` -- an
            ``AbstractPrior`` (scored in z-space) or a ``scene -> scalar`` callable.
            ``None`` is flat (improper).

    Returns:
        ``(logdensity, z0, unravel)`` from :func:`build_scene_logdensity`.
    """
    n_fit = len(fit_leaves(scene))
    mask = jax.tree_util.tree_map(lambda _: False, scene)
    mask = eqx.tree_at(fit_leaves, mask, [True] * n_fit)

    data = jnp.asarray(data)
    inv_var = 1.0 / jnp.asarray(noise_sigma) ** 2

    def likelihood(predicted):
        return -0.5 * jnp.sum((predicted - data) ** 2 * inv_var)

    if prior is None:

        def prior(_scene):
            return 0.0

    return build_scene_logdensity(scene, forward, likelihood, prior, filter_spec=mask)
