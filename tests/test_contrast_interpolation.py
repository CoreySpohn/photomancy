"""Shared interpolation of padded detection and noise curves."""

import jax
import jax.numpy as jnp
import numpy as np

from photomancy.orbit._contrast import interpolate_contrast_curve


def test_padded_contrast_values_and_gradients():
    """Exact-algebra basis: linear slopes and constant edge values, 8 eps.

    The two real knots (1,10) and (3,14) imply dmag=10+2*(alpha-1),
    and noise knots (1,2), (3,6) imply noise=2+2*(alpha-1). Include
    an entirely padded row and queries just beyond the last real knot.
    """
    alpha = jnp.array([0.5, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0 + 1e-6])
    sep = jnp.tile(jnp.array([1.0, 3.0, 0.0, 0.0]), (7, 1))
    mag = jnp.tile(jnp.array([10.0, 14.0, -jnp.inf, -jnp.inf]), (7, 1))
    mag = mag.at[5].set(-jnp.inf)
    noise = jnp.tile(jnp.array([2.0, 6.0, 999.0, 999.0]), (7, 1))
    noise = noise.at[5].set(7.0)
    got, limit = jax.jit(interpolate_contrast_curve)(alpha, sep, mag, noise)
    expected = [-1e8, 10.0, 12.0, 14.0, -1e8, -1e8, -1e8]
    np.testing.assert_allclose(got, expected, rtol=8 * np.finfo(float).eps)
    np.testing.assert_array_equal(limit, [2, 2, 4, 6, 6, 7, 6])
    np.testing.assert_array_equal(got, interpolate_contrast_curve(alpha, sep, mag))
    grad = jax.grad(lambda x: interpolate_contrast_curve(x, sep, mag).sum())(alpha)
    np.testing.assert_array_equal(grad[jnp.array([0, 2, 4, 5, 6])], [0, 2, 0, 0, 0])
    assert np.isfinite(grad).all()
