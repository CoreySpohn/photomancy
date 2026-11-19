"""Tests for the traceable reference-aperture count."""

import jax
import jax.numpy as jnp

from photomancy.postproc.core.geometry import (
    calculate_n_apertures,
    n_reference_apertures,
)


class TestNReferenceApertures:
    """Traceable aperture-count formula."""

    def test_matches_calculate_n_apertures(self):
        """Traceable count equals the public Python-int helper everywhere."""
        for radius in (5.0, 10.0, 17.3, 30.0, 50.0):
            for fwhm in (3.0, 4.5, 5.0, 8.0):
                traced = int(n_reference_apertures(radius, fwhm))
                assert traced == calculate_n_apertures(radius, fwhm)

    def test_matches_fused_core_inline_formula(self):
        """Reproduce the v1.1.1 _snr_batch_core inline ops bit-for-bit."""
        radius, fwhm, buffer = 12.0, 5.0, 0.5
        half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(radius, 0.1), 1.0))
        d_theta = 2.0 * half_angle
        n_theoretical = jnp.floor(2 * jnp.pi / jnp.maximum(d_theta, 0.01))
        expected = jnp.maximum((n_theoretical - 1 - 2 * buffer).astype(int), 1)
        assert int(n_reference_apertures(radius, fwhm, buffer)) == int(expected)

    def test_small_radius_floors_at_one(self):
        """Radii inside the first resolution element still report one aperture."""
        assert int(n_reference_apertures(0.5, 5.0)) == 1

    def test_traceable_under_jit_and_vmap(self):
        """The count works with traced radius (the fused core requirement)."""
        counts = jax.jit(jax.vmap(lambda r: n_reference_apertures(r, 5.0)))(
            jnp.array([8.0, 16.0, 24.0])
        )
        assert counts.shape == (3,)
        assert int(counts[1]) == calculate_n_apertures(16.0, 5.0)
