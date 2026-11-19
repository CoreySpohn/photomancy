"""Tests for differentiable fixed-size patch extraction."""

import jax
import jax.numpy as jnp
import numpy as np

from photomancy.postproc.core.detection.patches import extract_patch


class TestExtractPatch:
    """Patch extraction is exact on-grid, differentiable, and vmap/jit safe."""

    def test_integer_center_matches_sliced_subarray(self):
        """At an integer center, order=3 exactly reproduces the sub-array.

        The Keys cubic kernel is a true interpolant (weight 1 at zero offset,
        0 at nonzero integer offsets), so sampling a ramp image on an
        integer-centered grid reproduces the sliced sub-array to within
        floating-point round-off.
        """
        image = jnp.arange(400.0).reshape(20, 20)
        patch = extract_patch(image, jnp.array([10.0, 10.0]), 5)
        expected = image[8:13, 8:13]
        np.testing.assert_allclose(np.asarray(patch), np.asarray(expected), atol=1e-5)

    def test_odd_size_centers_on_the_middle_pixel(self):
        """An odd patch size centers exactly on the requested pixel."""
        image = jax.random.normal(jax.random.PRNGKey(0), (20, 20))
        patch = extract_patch(image, jnp.array([5.0, 5.0]), 5)
        np.testing.assert_allclose(
            np.asarray(patch[2, 2]), np.asarray(image[5, 5]), atol=1e-5
        )

    def test_off_chip_samples_fill_with_cval(self):
        """A center near the corner fills the off-chip band with cval.

        With an integer center, samples whose row or column index falls
        outside the image resolve to cval regardless of the other
        coordinate; samples with both coordinates on-chip return the finite
        image value.
        """
        image = jnp.full((10, 10), 3.0)
        patch = extract_patch(image, jnp.array([1.0, 1.0]), 5)
        np.testing.assert_array_equal(np.asarray(patch[0, :]), np.zeros(5))
        np.testing.assert_array_equal(np.asarray(patch[:, 0]), np.zeros(5))
        interior = patch[1:, 1:]
        assert bool(jnp.all(jnp.isfinite(interior)))
        np.testing.assert_array_equal(np.asarray(interior), np.full((4, 4), 3.0))

    def test_grad_through_center_is_finite_and_nonzero(self):
        """The patch sum is differentiable with respect to the center."""
        image = jax.random.normal(jax.random.PRNGKey(1), (20, 20))

        def loss(center):
            """Sum of the patch extracted at center."""
            return extract_patch(image, center, 5).sum()

        grad = jax.grad(loss)(jnp.array([9.3, 10.7]))
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad != 0.0))

    def test_vmap_under_jit_matches_the_loop_of_single_calls(self):
        """Vmapping a batch of centers under jit matches per-center calls."""
        image = jax.random.normal(jax.random.PRNGKey(2), (30, 30))
        centers = jnp.array([[10.0, 10.0], [15.3, 12.7], [5.0, 20.0]])

        @jax.jit
        def batched(centers):
            """Extract one patch per center via vmap."""
            return jax.vmap(lambda c: extract_patch(image, c, 5))(centers)

        patches = batched(centers)
        assert patches.shape == (3, 5, 5)
        expected = jnp.stack([extract_patch(image, c, 5) for c in centers])
        np.testing.assert_array_equal(np.asarray(patches), np.asarray(expected))
