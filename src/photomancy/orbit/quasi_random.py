"""Roberts low-discrepancy quasi-random sequence (pure JAX).

Vendored from the implementation proposed in JAX PR #23808
(jax.random.roberts_sequence). Pure-JAX so orbix needs no scipy. Replace with
the upstream import if/when that PR merges.
"""

import jax
import jax.numpy as jnp


def _phi(dim, n_iters=50):
    """Generalized plastic constant: real root of x**(dim+1) = x + 1."""
    x = 2.0
    for _ in range(n_iters):
        x = (1.0 + x) ** (1.0 / (dim + 1.0))
    return x


def roberts_sequence(n, dim, key=None):
    """Return ``(n, dim)`` quasi-random points in the half-open unit cube.

    Args:
        n: Number of points.
        dim: Dimension.
        key: Optional PRNG key for a Cranley-Patterson rotation (random shift).

    Returns:
        Array of shape ``(n, dim)`` with values in ``[0, 1)``.
    """
    g = _phi(dim)
    alpha = (1.0 / g) ** jnp.arange(1, dim + 1)
    idx = jnp.arange(1, n + 1)[:, None]
    pts = (0.5 + alpha[None, :] * idx) % 1.0
    if key is not None:
        shift = jax.random.uniform(key, (dim,))
        pts = (pts + shift[None, :]) % 1.0
    return pts
