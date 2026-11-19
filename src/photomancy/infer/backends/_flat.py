"""Private helpers the backends share: the flat unconstrained vector of a ``Fit``.

A gradient or sampling backend moves a flat vector ``z`` in the unconstrained
coordinates of ``fit.to_real``. The vector is the ``jax.flatten_util.ravel_pytree``
of one draw's entry dict (sorted keys), so ``unravel(z)`` is a valid argument to
``fit.real_log_prob`` and ``fit.from_real``. Only ``Fit`` methods are used.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.infer.posterior import Posterior


def draw_template(fit):
    """Zeros with the structure and shapes of one draw ``x`` of ``fit`` (float64)."""
    shapes = eqx.filter_eval_shape(lambda k: fit.sample_prior(k), jax.random.key(0))
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)


def flat_problem(fit):
    """Return ``(unravel, logdensity, dim)`` for the unconstrained flat vector.

    ``logdensity(z)`` is ``fit.real_log_prob(unravel(z))``, the density in ``u``
    including the log volume of the map from the stored coordinates.
    """
    template = fit.to_real(draw_template(fit))
    flat, unravel = ravel_pytree(template)

    def logdensity(z):
        return fit.real_log_prob(unravel(z))

    return unravel, logdensity, flat.shape[0]


def to_x(fit, unravel, z):
    """Stored-coordinate entries ``x`` of a batch of flat vectors ``z (N, d)``."""
    x, _ = fit.from_real(jax.vmap(unravel)(z))
    return x


def start_points(fit, init, key, n):
    """Return ``n`` flat starting vectors ``(n, d)`` in unconstrained coordinates.

    Args:
        fit: The fit.
        init: ``None`` for prior draws; one constrained draw ``x`` (an entry dict with
            leaf shapes, repeated ``n`` times); or a :class:`Posterior` (``n`` draws
            by its weights).
        key: PRNG key (unused for a single draw).
        n: Number of starts.
    """
    if init is None:
        x = fit.sample_prior(key, (n,))
    elif isinstance(init, Posterior):
        index = jax.random.categorical(key, init.log_weights, shape=(n,))
        x = jax.tree.map(lambda a: a[index], init.samples)
    else:
        x = jax.tree.map(lambda a: jnp.broadcast_to(a, (n, *a.shape)), init)
    return fit.to_flat(x)


class FlatDensity(eqx.Module):
    """``z -> fit.real_log_prob(unravel(z))`` as a module, so the fit stays traced.

    The unravel function is rebuilt from ``template`` (zeros of one unconstrained draw)
    on every call, so the module has array leaves only and a ``filter_jit`` that
    receives it neither recompiles for a new instance nor bakes the fit's arrays in.
    """

    fit: object
    template: dict

    def __call__(self, z):
        """Density of the flat unconstrained vector ``z``."""
        unravel = ravel_pytree(self.template)[1]
        return self.fit.real_log_prob(unravel(z))


def flat_density(fit):
    """Return the :class:`FlatDensity` of ``fit``."""
    return FlatDensity(fit, fit.to_real(draw_template(fit)))
