"""test_verification.py - Numerical, Geometric, and Physical Verification.

Validates:
1. Numerical accuracy of Snapshot PCA (vs Standard SVD).
2. Cross-module coordinate consistency (Modeling vs Transforms).
3. End-to-end signal preservation and differentiability.

Run with: pytest tests/test_verification.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.transforms import resample_flux

from photomancy.postproc.core.modeling import inject_planet, make_simple_disk
from photomancy.postproc.core.pca import get_pca_basis, pca_subtract


@pytest.fixture(autouse=True)
def _enable_x64():
    """Run each test in this module in float64, restoring the flag afterwards.

    The math-identity checks here need double precision. Scoping the flag to
    this module keeps the rest of the suite in the default float32 regime:
    previously the module enabled x64 globally at import, so full-suite runs
    were float64 everywhere while single-file runs stayed float32.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


class TestPCANumericalAccuracy:
    """Verify the custom Snapshot PCA implementation matches standard Linear Algebra."""

    def test_snapshot_matches_numpy_svd(self):
        """The computed basis should match standard SVD basis vectors (NumPy).

        This is critical verification: it proves your optimized 'Snapshot Method'
        yields the exact same mathematical result as the slow, standard O(N^3) SVD.
        """
        # Create random reference cube (N_frames < N_pixels)
        n_frames, ny, nx = 20, 10, 10
        ref_cube_np = np.random.default_rng(0).standard_normal((n_frames, ny, nx))
        n_modes = 5

        # 1. Run the snapshot PCA (JAX)
        basis_jax, _ = get_pca_basis(jnp.array(ref_cube_np), n_modes)

        # 2. Run Standard SVD (NumPy) for ground truth
        # Flatten -> Center -> SVD
        flat_ref = ref_cube_np.reshape(n_frames, -1)
        flat_ref -= np.mean(flat_ref, axis=0)
        # SVD: X = U S Vt. Basis vectors are rows of Vt.
        _, _, vt = np.linalg.svd(flat_ref, full_matrices=False)
        basis_svd = vt[:n_modes]

        # 3. Compare Basis Vectors
        # Note: Eigenvectors are unique up to a sign flip (+v or -v are both valid).
        # We check that the absolute correlation is ~1.0.
        for i in range(n_modes):
            dot_prod = jnp.dot(basis_jax[i], basis_svd[i])
            assert jnp.abs(jnp.abs(dot_prod) - 1.0) < 1e-5, (
                f"Mode {i} mismatch. Correlation: {dot_prod}"
            )


class TestCoordinateConsistency:
    """Ensure Modeling and Transform modules share the same coordinate reality."""

    def test_disk_rotation_linkage(self):
        """Verify rotating a 'North' disk (PA=0) by 90deg equals an 'East' disk (PA=90).

        A symmetric ring verifies axis alignment, but cannot distinguish the
        rotation sign. Oblique sky PAs are checked in test_validation.py.
        """
        shape = (101, 101)

        # 1. Generate Vertical Disk (PA=0) and rotate 90 deg
        disk_north = make_simple_disk(
            shape, radius=30, inclination_deg=80, width=5, pa_deg=0
        )
        # With origin="lower", +90 rotates CCW (north -> east, -column).
        disk_rotated = resample_flux(disk_north, 1.0, 1.0, shape, rotation_deg=90.0)

        # 2. Generate Horizontal Disk (PA=90) directly
        disk_east = make_simple_disk(
            shape, radius=30, inclination_deg=80, width=5, pa_deg=90
        )

        # 3. They should match. A 90 deg turn about the center of an odd grid
        # permutes pixels, so the residual is roundoff; an unrotated or empty
        # image gives a relative residual of order 1.
        residual = jnp.linalg.norm(disk_rotated - disk_east) / jnp.linalg.norm(
            disk_east
        )
        assert residual < 1e-6, f"relative residual {float(residual):.2e}"

    def test_astrometry_recovery(self):
        """Inject a planet and verify the centroid is recovered at the correct position.

        This validates the chain: (Coordinates) -> inject_planet -> (Image).
        """
        ny, nx = 51, 51
        image = jnp.zeros((ny, nx))

        # Inject at a sub-pixel position
        true_pos = (20.3, 30.7)  # (y, x)

        # Gaussian PSF
        y, x = jnp.mgrid[:ny, :nx]
        psf = jnp.exp(-((y - 25) ** 2 + (x - 25) ** 2) / 2.0)
        psf /= jnp.sum(psf)

        injected = inject_planet(image, psf, flux=100.0, pos=true_pos)

        # Measure Center of Mass
        y_idx, x_idx = jnp.indices((ny, nx))
        total = jnp.sum(injected)
        meas_y = jnp.sum(y_idx * injected) / total
        meas_x = jnp.sum(x_idx * injected) / total

        # Should match within < 0.05 pixel
        assert jnp.abs(meas_y - true_pos[0]) < 0.05
        assert jnp.abs(meas_x - true_pos[1]) < 0.05


class TestEndToEndPhysics:
    """Verify scientific fidelity (RDI Throughput) and Differentiability."""

    def test_rdi_signal_preservation(self):
        """Verify PCA preserves a planet absent from the reference library (RDI)."""
        ny, nx = 31, 31
        n_frames = 50

        # Reference Cube (Noise only, no planet)
        key = jax.random.PRNGKey(0)
        ref_cube = jax.random.normal(key, (n_frames, ny, nx))

        # Science Image (Planet + Noise)
        psf = jnp.exp(
            -((jnp.mgrid[:ny, :nx][0] - 15) ** 2 + (jnp.mgrid[:ny, :nx][1] - 15) ** 2)
            / 2.0
        )
        psf /= jnp.sum(psf)
        science_image = inject_planet(
            jnp.zeros((ny, nx)), psf, flux=100.0, pos=(15.0, 20.0)
        )

        # PCA Subtraction
        basis, mean_ref = get_pca_basis(ref_cube, n_modes=5)
        residual = pca_subtract(science_image, basis, mean_ref)

        # Measure Flux in Residual (Aperture Photometry)
        # In RDI, the planet is orthogonal to the reference noise, so it is preserved.
        y, x = jnp.indices((ny, nx))
        dist = jnp.sqrt((y - 15.0) ** 2 + (x - 20.0) ** 2)
        aperture_flux = jnp.sum(residual * (dist < 2.0))

        # Should recover >75% of flux (some loss to chance correlations is normal)
        assert aperture_flux > 75.0

    def test_differentiability(self):
        """Verify gradients flow from Residuals back to Injection parameters."""
        # This confirms the library is ready for HMC / Forward Modeling
        ref_cube = jax.random.normal(jax.random.PRNGKey(1), (5, 20, 20))
        basis, mean_ref = get_pca_basis(ref_cube, 2)
        psf = jnp.ones((20, 20))

        def loss_fn(flux, pos_y, pos_x):
            img = inject_planet(jnp.zeros((20, 20)), psf, flux, (pos_y, pos_x))
            res = pca_subtract(img, basis, mean_ref)
            return jnp.sum(res**2)

        # Calculate gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(10.0, 10.0, 10.0)

        # Gradients must be finite and non-zero
        assert all(jnp.isfinite(g) for g in grads)
        assert all(abs(g) > 0.0 for g in grads)
