"""Dense Hessian built one column at a time."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def dense_hessian(f: Callable, x: jax.Array) -> jax.Array:
    """Symmetrized Hessian of a scalar ``f`` at the flat vector ``x``.

    Each column is one Hessian-vector product (forward over reverse), and the columns
    are taken in sequence with :func:`jax.lax.map`, so peak memory is that of one
    gradient whatever the dimension. :func:`jax.hessian` vectorizes all columns at
    once and holds every intermediate of ``f`` once per parameter.

    Args:
        f: ``x -> scalar``.
        x: Point of shape ``(n,)``.

    Returns:
        The ``(n, n)`` matrix ``0.5 (H + H^T)``.
    """
    grad_fn = jax.grad(f)

    def column(v):
        return jax.jvp(grad_fn, (x,), (v,))[1]

    hess = jax.lax.map(column, jnp.eye(x.shape[0], dtype=x.dtype))
    return 0.5 * (hess + hess.T)
