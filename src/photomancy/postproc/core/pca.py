"""Principal Component Analysis (PCA/KLIP) for PSF subtraction.

Implements the efficient 'Snapshot Method' for SVD on large image cubes,
particularly suited for coronagraphy (where pixels >> frames).
All functions are JIT-compilable and fully differentiable.
"""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=["n_modes"])
def get_pca_basis(
    ref_cube: jnp.ndarray,
    n_modes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute PCA basis vectors from a reference cube (Snapshot Method).

    Uses eigendecomposition of the NxN covariance matrix instead of full SVD,
    which is O(N**2 x P) vs O(P**2 x N) for pixels P >> frames N.

    Args:
        ref_cube: Reference image cube of shape (N_frames, Height, Width).
        n_modes: Number of principal components (modes) to keep.

    Returns:
        basis: The top n_modes eigen-images, shape (n_modes, Height*Width).
        mean_ref: The mean of the reference cube, shape (Height, Width).
    """
    n_frames, ny, nx = ref_cube.shape
    flat_refs = ref_cube.reshape(n_frames, -1)

    # Center the data
    mean_ref = jnp.mean(flat_refs, axis=0)
    centered = flat_refs - mean_ref

    # Covariance Matrix (N_frames x N_frames)
    cov = jnp.dot(centered, centered.T)

    # Eigen Decomposition (Symmetric/Hermitian)
    vals, vecs = jnp.linalg.eigh(cov)

    # Sort and Truncate (eigh returns ascending order)
    vals = vals[: -n_modes - 1 : -1]
    vecs = vecs[:, : -n_modes - 1 : -1]

    # Project back to Image Space (KL transform)
    # Basis = (Vectors^T . Centered_Data) / sqrt(Eigenvalues)
    normalization = 1.0 / jnp.sqrt(jnp.maximum(vals, 1e-12))
    basis = jnp.dot(vecs.T, centered) * normalization[:, None]

    return basis, mean_ref.reshape(ny, nx)


@jax.jit
def pca_subtract(
    science_image: jnp.ndarray,
    basis: jnp.ndarray,
    mean_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Project science image onto PCA basis and subtract the model.

    Args:
        science_image: 2D science image, shape (Height, Width).
        basis: PCA basis from get_pca_basis, shape (n_modes, Height*Width).
        mean_ref: Mean reference from get_pca_basis, shape (Height, Width).

    Returns:
        Residual image with PSF model subtracted, shape (Height, Width).
    """
    ny, nx = science_image.shape
    flat_sci = science_image.reshape(-1)
    flat_mean = mean_ref.reshape(-1)

    # Center target
    centered_sci = flat_sci - flat_mean

    # Coefficients = Basis . Image
    coeffs = jnp.dot(basis, centered_sci)

    # Model = Coefficients . Basis
    model = jnp.dot(coeffs, basis)

    # Residual
    residual = centered_sci - model
    return residual.reshape(ny, nx)
