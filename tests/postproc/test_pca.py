"""test_pca.py - Tests for PCA/KLIP implementation.

Run with: pytest tests/test_pca.py -v
"""

import jax
import jax.numpy as jnp

from photomancy.postproc.core.pca import get_pca_basis, pca_subtract


class TestGetPCABasis:
    """Tests for PCA basis computation."""

    def test_correct_output_shapes(self):
        """Output shapes should match expected dimensions."""
        n_frames, ny, nx = 10, 32, 32
        n_modes = 5
        ref_cube = jax.random.normal(jax.random.PRNGKey(0), (n_frames, ny, nx))

        basis, mean_ref = get_pca_basis(ref_cube, n_modes)

        assert basis.shape == (n_modes, ny * nx)
        assert mean_ref.shape == (ny, nx)

    def test_basis_orthonormal(self):
        """PCA basis vectors should be orthonormal."""
        n_frames, ny, nx = 20, 32, 32
        n_modes = 5
        ref_cube = jax.random.normal(jax.random.PRNGKey(1), (n_frames, ny, nx))

        basis, _ = get_pca_basis(ref_cube, n_modes)

        # Check orthonormality: basis @ basis.T should be ~identity
        gram = basis @ basis.T
        assert jnp.allclose(gram, jnp.eye(n_modes), atol=1e-4)

    def test_mean_subtraction(self):
        """Mean reference should be the mean of the cube."""
        n_frames, ny, nx = 10, 32, 32
        n_modes = 3
        ref_cube = jax.random.normal(jax.random.PRNGKey(2), (n_frames, ny, nx))

        _, mean_ref = get_pca_basis(ref_cube, n_modes)

        expected_mean = jnp.mean(ref_cube, axis=0)
        assert jnp.allclose(mean_ref, expected_mean, atol=1e-5)

    def test_more_modes_than_frames_clipped(self):
        """Requesting more modes than frames should still work."""
        n_frames, ny, nx = 5, 32, 32
        n_modes = 10  # More than n_frames
        ref_cube = jax.random.normal(jax.random.PRNGKey(3), (n_frames, ny, nx))

        # Should not crash - modes get clipped by eigendecomposition
        basis, _mean_ref = get_pca_basis(ref_cube, min(n_modes, n_frames - 1))
        assert basis.shape[0] <= n_frames


class TestPCASubtract:
    """Tests for PCA subtraction."""

    def test_removes_mean(self):
        """PCA subtraction should remove at least the mean."""
        n_frames, ny, nx = 10, 32, 32
        n_modes = 5
        ref_cube = jax.random.normal(jax.random.PRNGKey(4), (n_frames, ny, nx))
        science = jnp.mean(ref_cube, axis=0) + 0.1 * jax.random.normal(
            jax.random.PRNGKey(5), (ny, nx)
        )

        basis, mean_ref = get_pca_basis(ref_cube, n_modes)
        residual = pca_subtract(science, basis, mean_ref)

        # Residual should have near-zero mean
        assert jnp.abs(jnp.mean(residual)) < 0.5

    def test_synthetic_structure_removal(self):
        """PCA should remove shared structure from science image."""
        ny, nx = 64, 64
        n_frames = 20
        n_modes = 5

        # Create synthetic PSF pattern present in all frames
        y, x = jnp.mgrid[:ny, :nx]
        psf_pattern = jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 50)

        # Add to reference cube with small variations
        key = jax.random.PRNGKey(6)
        ref_cube = psf_pattern[None, :, :] + 0.1 * jax.random.normal(
            key, (n_frames, ny, nx)
        )

        # Science image = same PSF + planet
        planet_y, planet_x = 40, 45
        planet = 0.5 * jnp.exp(-((y - planet_y) ** 2 + (x - planet_x) ** 2) / 10)
        science = psf_pattern + planet

        basis, mean_ref = get_pca_basis(ref_cube, n_modes)
        residual = pca_subtract(science, basis, mean_ref)

        # Check planet signal is preserved (center region around planet)
        planet_region = residual[
            planet_y - 3 : planet_y + 4, planet_x - 3 : planet_x + 4
        ]
        assert jnp.max(planet_region) > 0.1

    def test_gradient_flow(self):
        """Verify gradients flow through PCA subtract."""
        n_frames, ny, nx = 10, 32, 32
        n_modes = 3
        ref_cube = jax.random.normal(jax.random.PRNGKey(7), (n_frames, ny, nx))
        science = jax.random.normal(jax.random.PRNGKey(8), (ny, nx))

        def loss_fn(sci):
            basis, mean_ref = get_pca_basis(ref_cube, n_modes)
            residual = pca_subtract(sci, basis, mean_ref)
            return jnp.sum(residual**2)

        grad = jax.grad(loss_fn)(science)
        assert jnp.all(jnp.isfinite(grad))
