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
    """Zeros with the structure and shapes of one draw ``x`` of ``fit`` (float64).

    Args:
        fit: The fit whose prior draw sets the structure and shapes.

    Returns:
        Entry dict of zeros, one leaf (or block of leaves) per entry of a draw of
        ``fit``, in the stored coordinates.
    """
    shapes = eqx.filter_eval_shape(lambda k: fit.sample_prior(k), jax.random.key(0))
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)


def require_no_walls(fit, name, key=None):
    """Refuse a fit whose density has walls, for a backend that needs a gradient.

    Against a wall every rejected step is flagged as a divergence, so the sampler's
    own diagnostic carries no information.

    Args:
        fit: The fit to check.
        name: Name of the backend, for the message.
        key: Sampler key to carry runtime support checks for traced blocks.

    Returns:
        The key, with any traced wall checks attached.

    Raises:
        ValueError: Naming each entry and its walls, and the ways out.
    """

    def message(walls):
        entries = "; ".join(
            f"free[{entry!r}] does not map the bounds of {', '.join(names)} onto its "
            "coordinates"
            for entry, names in walls.items()
        )
        return (
            f"{name} needs a density with no wall. {entries}: use a block whose "
            "coordinates carry them, a law without those bounds, or Importance or "
            "NestedSampling."
        )

    walls = fit.walls
    for entry, block in fit.blocks.items():
        flags = getattr(block, "wall_flags", None)
        if flags is None or not any(
            isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(block)
        ):
            continue
        walls.pop(entry, None)
        current = flags()
        if not current:
            continue
        names = tuple(current)
        masks = range(1 << len(names))
        messages = [
            message({entry: tuple(n for i, n in enumerate(names) if mask & (1 << i))})
            for mask in masks
        ]
        index = sum(
            flag.astype(jnp.int32) << i for i, flag in enumerate(current.values())
        )
        # Equinox's runtime callback accepts raw key data, not typed keys.
        typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        data = jax.random.key_data(key)
        data = eqx.branched_error_if(data, index != 0, index, messages)
        key = (
            jax.random.wrap_key_data(data, impl=jax.random.key_impl(key))
            if typed
            else data
        )
    if walls:
        raise ValueError(message(walls))
    return key


def flat_problem(fit):
    """Return ``(unravel, logdensity, dim)`` for the unconstrained flat vector.

    ``logdensity(z)`` is ``fit.real_log_prob(unravel(z))``, the density in ``u``
    including the log volume of the map from the stored coordinates.

    Args:
        fit: The fit whose unconstrained density is flattened.

    Returns:
        ``unravel`` maps a flat vector ``(d,)`` to the unconstrained entry dict,
        ``logdensity`` maps a flat vector to a scalar log density, and ``dim`` is the
        length ``d`` of the flat vector.
    """
    template = fit.to_real(draw_template(fit))
    flat, unravel = ravel_pytree(template)

    def logdensity(z):
        return fit.real_log_prob(unravel(z))

    return unravel, logdensity, flat.shape[0]


def to_x(fit, unravel, z):
    """Stored-coordinate entries ``x`` of a batch of flat vectors ``z (N, d)``.

    Args:
        fit: The fit that defines the coordinate map.
        unravel: Function mapping one flat vector ``(d,)`` to the unconstrained entry
            dict, as returned by :func:`flat_problem`.
        z: Batch of flat unconstrained vectors, shape ``(N, d)``.

    Returns:
        Entry dict of the draws in the stored coordinates, with a leading axis of
        length ``N``.
    """
    x, _ = fit.from_real(jax.vmap(unravel)(z))
    return x


def start_points(fit, init, key, n):
    """Return ``n`` flat starting vectors ``(n, d)`` in unconstrained coordinates.

    Args:
        fit: The fit.
        init: ``None`` for prior draws; one constrained draw ``x`` (an entry dict with
            leaf shapes, repeated ``n`` times); an entry dict whose every leaf has one
            extra leading axis of length ``n`` (one explicit start each, in order);
            or a :class:`Posterior` (``n`` draws by its weights).
        key: PRNG key (unused for a single draw).
        n: Number of starts.

    Returns:
        Flat starting vectors in unconstrained coordinates, shape ``(n, d)``.
    """
    if init is None:
        return fit.to_flat(fit.sample_prior(key, (n,)))
    if isinstance(init, Posterior):
        index = jax.random.categorical(key, init.log_weights, shape=(n,))
        x = jax.tree.map(lambda a: a[index], init.samples)
    else:
        x = init
    leading = fit.check_draws(x)
    if not leading and not isinstance(init, Posterior):
        x = jax.tree.map(lambda a: jnp.broadcast_to(a, (n, *jnp.shape(a))), x)
    elif leading != (n,):
        raise ValueError(
            f"init has leading shape {leading}, expected {(n,)}, one start each"
        )
    return fit.to_flat(x)


class FlatDensity(eqx.Module):
    """``z -> fit.real_log_prob(unravel(z))`` as a module, so the fit stays traced.

    The unravel function is rebuilt from ``template`` (zeros of one unconstrained draw)
    on every call, so the module has array leaves only and a ``filter_jit`` that
    receives it neither recompiles for a new instance nor bakes the fit's arrays in.

    Attributes:
        fit: The fit whose ``real_log_prob`` is evaluated.
        template: Zeros of one unconstrained draw, from which the unravel function is
            rebuilt.
    """

    fit: object
    template: dict

    def __call__(self, z):
        """Density of the flat unconstrained vector ``z``.

        Args:
            z: Flat unconstrained vector, shape ``(d,)``.

        Returns:
            Scalar log density of ``z`` in the unconstrained coordinates.
        """
        unravel = ravel_pytree(self.template)[1]
        return self.fit.real_log_prob(unravel(z))


def flat_density(fit):
    """Return the :class:`FlatDensity` of ``fit``.

    Args:
        fit: The fit to wrap.

    Returns:
        The flat density module of ``fit``.
    """
    return FlatDensity(fit, fit.to_real(draw_template(fit)))
