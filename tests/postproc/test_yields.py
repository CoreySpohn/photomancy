"""test_yields.py - Tests for yield analysis tools.

Run with: pytest tests/test_yields.py -v
"""

import jax.numpy as jnp

from photomancy.postproc.analysis.yields import (
    get_perfect_residuals,
)


class TestGetPerfectResiduals:
    """Tests for perfect subtraction (Faux-RDI)."""

    def test_exact_subtraction(self):
        """Should return exact difference."""
        observation = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        model = jnp.array([[5.0, 10.0], [15.0, 20.0]])
        residual = get_perfect_residuals(observation, model)
        expected = jnp.array([[5.0, 10.0], [15.0, 20.0]])
        assert jnp.allclose(residual, expected)

    def test_preserves_planet_signal(self):
        """Planet signal not in model should remain in residual."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        # Background model (stellar halo)
        background = 100 * jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 200)

        # Planet signal
        planet_pos = (40, 50)
        planet = 10 * jnp.exp(
            -((y - planet_pos[0]) ** 2 + (x - planet_pos[1]) ** 2) / 10
        )

        observation = background + planet
        residual = get_perfect_residuals(observation, background)

        # Residual should contain only planet
        assert jnp.allclose(residual, planet, atol=1e-5)
