"""Tests for template providers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.postproc.templates import (
    AbstractTemplateProvider,
    ArrayTemplateProvider,
)


class TestAbstractTemplateProvider:
    """AbstractTemplateProvider is abstract and cannot be instantiated."""

    def test_abc_rejects_instantiation(self):
        """Instantiating AbstractTemplateProvider directly raises TypeError."""
        with pytest.raises(TypeError):
            AbstractTemplateProvider()


class TestArrayTemplateProvider:
    """ArrayTemplateProvider returns a fixed stamp independent of position."""

    def test_returns_same_stamp_for_any_position(self):
        """The provider returns the same stamp for different positions."""
        stamp = jnp.ones((7, 7))
        provider = ArrayTemplateProvider(stamp)
        pos1 = jnp.array([10.0, 10.0])
        pos2 = jnp.array([50.3, 25.7])
        result1 = provider(pos1)
        result2 = provider(pos2)
        np.testing.assert_array_equal(np.asarray(result1), np.asarray(result2))
        np.testing.assert_array_equal(np.asarray(result1), np.asarray(stamp))

    def test_non_square_input_raises_valueerror(self):
        """Non-square stamps raise ValueError."""
        with pytest.raises(ValueError, match="square"):
            ArrayTemplateProvider(jnp.ones((5, 7)))

    def test_vmap_over_positions_returns_batched_stamps(self):
        """Vmapping over positions returns (n, K, K) batched stamps."""
        stamp = jax.random.normal(jax.random.PRNGKey(0), (7, 7))
        provider = ArrayTemplateProvider(stamp)
        positions = jnp.array(
            [
                [10.0, 10.0],
                [20.5, 15.3],
                [30.0, 40.0],
            ]
        )

        @jax.jit
        def vmapped_call(positions):
            """Call provider for all positions via vmap."""
            return jax.vmap(provider)(positions)

        result = vmapped_call(positions)
        assert result.shape == (3, 7, 7)
        expected = jnp.stack([stamp, stamp, stamp])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
