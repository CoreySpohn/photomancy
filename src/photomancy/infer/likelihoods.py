"""Masked, normalized log-likelihood builders for gridded data.

Each builder takes the observed ``data``, a per-element ``sigma`` and an optional
boolean ``mask`` (True means the element is used) and returns a callable
``predicted -> scalar log-likelihood``. That callable is the ``likelihood`` plug-in
of :class:`photomancy.infer.terms.ForwardTerm`.

In every builder ``sigma`` is the *scale parameter* of the density of the
standardized residual ``z = (predicted - data) / sigma`` (the same convention as
``scipy.stats`` ``scale=``). For the Gaussian it is the standard deviation; for the
Laplace and Student-t densities it is the scale, not the standard deviation (the
Laplace standard deviation is ``sqrt(2) * sigma``). The returned values are proper
normalized log densities, including the ``-log(sigma)`` and shape-dependent constants,
so evidences and model comparisons across likelihoods are meaningful.

Elements with ``mask`` False contribute exactly zero to the value and zero to the
gradient, even where ``data``, ``sigma`` or ``predicted`` are NaN there. ``sigma`` must
be positive at the elements that are used.

Available densities:

``gaussian_loglike``
    ``log N(z; 0, 1)``.
``low_rank_gaussian_loglike``
    Gaussian with diagonal noise plus a marginalized low-rank covariance.
``laplace_loglike``
    ``log (exp(-|z|) / 2)``: an L1 loss.
``huber_loglike``
    A Gaussian core for ``|z| <= delta`` and Laplace tails beyond, joined with a
    continuous first derivative. Normalized exactly (see its docstring).
``student_t_loglike``
    Student-t with ``nu`` degrees of freedom, whose power-law tails absorb unmodeled
    outliers such as cosmic rays and hot pixels.
"""

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _gaussian_kernel(z, params):
    return 0.5 * z**2


def _laplace_kernel(z, params):
    return jnp.abs(z)


def _huber_kernel(z, params):
    delta = params
    a = jnp.abs(z)
    return jnp.where(a <= delta, 0.5 * z**2, delta * a - 0.5 * delta**2)


def _student_t_kernel(z, params):
    nu = params
    return 0.5 * (nu + 1.0) * jnp.log1p(z**2 / nu)


_KERNELS = {
    "gaussian": _gaussian_kernel,
    "laplace": _laplace_kernel,
    "huber": _huber_kernel,
    "student_t": _student_t_kernel,
}


class MaskedLogLikelihood(eqx.Module):
    """Normalized log-likelihood of a standardized-residual density over used elements.

    Built by :func:`gaussian_loglike`, :func:`laplace_loglike`,
    :func:`huber_loglike` and :func:`student_t_loglike`. The data, sigma and mask are
    Module leaves, so a backend that passes this as a ``filter_jit`` argument threads
    them as traced inputs rather than baking them into the compiled program.

    Attributes:
        data: Observed values with excluded elements set to zero.
        sigma: Scale at every element, with excluded elements set to one.
        mask: Boolean array, True where the element is used.
        log_norm: Total normalization over the used elements,
            ``n_used * log_const - sum(log sigma)``.
        kind: Density name, a key of the kernel table.
        shape_param: ``delta`` for Huber, ``nu`` for Student-t, unused otherwise.
    """

    data: jax.Array
    sigma: jax.Array
    mask: jax.Array
    log_norm: jax.Array
    kind: str = eqx.field(static=True)
    shape_param: float = eqx.field(static=True)

    def __call__(self, predicted):
        """Return the log-likelihood of ``predicted`` (broadcastable to ``data``)."""
        z = (jnp.asarray(predicted) - self.data) / self.sigma
        z = jnp.where(self.mask, z, 0.0)
        kernel = _KERNELS[self.kind](z, self.shape_param)
        return self.log_norm - jnp.sum(jnp.where(self.mask, kernel, 0.0))


class LowRankGaussianLogLikelihood(eqx.Module):
    """Masked Gaussian with covariance ``diag(sigma**2) + U @ U.T``.

    The model is ``data = predicted + U @ a + noise``, with independent diagonal
    Gaussian noise of standard deviation ``sigma``. The low-rank basis ``U`` is
    shared between images; its coefficients have a unit Gaussian prior
    ``a ~ N(0, I)`` and are marginalized for each image. Reusing the basis does
    not couple the coefficients of separate likelihood instances.

    The Woodbury identity and matrix determinant lemma reduce the calculation to
    a rank-sized system. See Rasmussen and Williams (2006), *Gaussian Processes
    for Machine Learning*, appendix A.3, for the matrix inversion and determinant
    lemmas. The quadratic is evaluated as a sum of squares to avoid cancellation.

    Construct outside JAX tracing: the fixed covariance is factored on the host
    in double precision, then stored in the input dtype. For large, strongly
    dependent basis columns, use float64 inputs with JAX x64 enabled. All arrays
    are Module leaves, so passing this object as
    a ``filter_jit`` argument keeps them as traced inputs. Excluded pixels
    contribute neither density nor gradient, even when their inputs are NaN.

    Attributes:
        data: Observations, zero outside the mask.
        sigma: Diagonal noise standard deviations, one outside the mask.
        mask: Boolean array, True where the pixel is used.
        U: Basis of shape ``(data.size, rank)``, zero outside the mask.
        whitened: Basis divided by the flattened diagonal noise scale.
        cholesky: Lower Cholesky factor of ``I + whitened.T @ whitened``.
        log_norm: Gaussian log normalization over the used pixels.
    """

    data: jax.Array
    sigma: jax.Array
    mask: jax.Array
    U: jax.Array
    whitened: jax.Array
    cholesky: jax.Array
    log_norm: jax.Array

    def __init__(self, data, sigma, basis, mask=None):
        """Prepare a fixed covariance from observations and a flattened basis.

        Args:
            data: Observed image values.
            sigma: Diagonal noise standard deviation, broadcastable to ``data``;
                finite and positive on the mask.
            basis: Array of shape ``(data.size, rank)`` in flattened image order,
                in the same units as ``data``. Use zero columns for diagonal noise.
            mask: Optional boolean array broadcastable to ``data``; True uses a
                pixel. Data and basis must be finite on used pixels.

        Raises:
            ValueError: If the basis shape or used data, sigma, or basis is invalid.
        """
        dtype = jnp.result_type(
            jnp.asarray(data), jnp.asarray(sigma), jnp.asarray(basis), float
        )
        data, sigma, basis = np.asarray(data), np.asarray(sigma), np.asarray(basis)
        mask = np.asarray(True if mask is None else mask, dtype=bool)
        data, sigma, mask = np.broadcast_arrays(data, sigma, mask)
        if basis.ndim != 2 or basis.shape[0] != data.size:
            raise ValueError("basis must have shape (data.size, rank)")
        if not np.all(np.isfinite(sigma[mask]) & (sigma[mask] > 0)):
            raise ValueError("sigma must be finite and positive on the mask")
        if not np.all(np.isfinite(data[mask])) or not np.all(
            np.isfinite(basis[mask.ravel()])
        ):
            raise ValueError("data and basis must be finite on the mask")
        self.mask = jnp.asarray(mask)
        self.data = jnp.asarray(np.where(mask, data, 0), dtype)
        self.sigma = jnp.asarray(np.where(mask, sigma, 1), dtype)
        self.U = jnp.asarray(np.where(mask.ravel()[:, None], basis, 0), dtype)
        self.whitened = self.U / self.sigma.ravel()[:, None]
        # Factor on the host in double precision, including for float32 inputs.
        w = np.asarray(self.whitened, dtype=np.float64)
        chol = np.linalg.cholesky(np.eye(basis.shape[1]) + w.T @ w)
        self.cholesky = jnp.asarray(chol, dtype)
        self.log_norm = (
            jnp.sum(self.mask) * -_LOG_SQRT_2PI
            - jnp.sum(jnp.log(self.sigma))
            - jnp.asarray(np.log(np.diag(chol)).sum(), dtype)
        )

    def _whiten(self, image):
        return (jnp.where(self.mask, image, 0) / self.sigma).ravel()

    def _coefficients(self, z):
        return jsp.linalg.cho_solve((self.cholesky, True), self.whitened.T @ z)

    def solve(self, image):
        """Apply the masked inverse covariance; return an image shaped like data.

        ``image`` is broadcastable to ``data``. Excluded pixels return zero,
        including where the input is NaN.
        """
        z = self._whiten(image)
        if self.U.shape[1]:
            z = z - self.whitened @ self._coefficients(z)
        return z.reshape(self.data.shape) / self.sigma

    def posterior_term(self, predicted):
        """Return ``U E[a | data, predicted]`` for the unit Gaussian coefficients.

        ``predicted`` is broadcastable to ``data``. The result has the data shape
        and is zero outside the mask; rank zero returns an all-zero image.
        """
        z = self._whiten(self.data - jnp.where(self.mask, predicted, 0))
        if not self.U.shape[1]:
            return jnp.zeros_like(self.data)
        return (self.U @ self._coefficients(z)).reshape(self.data.shape)

    def __call__(self, predicted):
        """Return the normalized log density; ``predicted`` broadcasts to data."""
        z = self._whiten(jnp.where(self.mask, predicted, 0) - self.data)
        if not self.U.shape[1]:
            return self.log_norm - jnp.sum(0.5 * z**2)
        a = self._coefficients(z)
        # Avoid subtracting two large quadratic forms for strongly aligned images.
        return self.log_norm - 0.5 * (
            jnp.sum((z - self.whitened @ a) ** 2) + jnp.sum(a**2)
        )


def residual_basis(residuals, rank, mask=None):
    """Build an uncentered low-rank covariance basis from normalized residuals.

    Each residual image must already be normalized to the desired common scale.
    The images are neither centered nor spatially registered. A shared basis can
    model residuals between images as ``basis @ a``, with unit Gaussian
    coefficients marginalized by :func:`low_rank_gaussian_loglike` alongside
    diagonal noise. Scale the basis to the data units before building a likelihood.
    This preparation runs on the host in double precision, outside JAX tracing.

    Args:
        residuals: Stack of shape ``(n, *image_shape)`` with at least one image.
        rank: Integer number of retained modes, between zero and ``n`` and no
            larger than the number of used pixels.
        mask: Optional boolean array broadcastable to ``image_shape``; True uses
            a pixel. Non-finite residuals outside the mask are ignored.

    Returns:
        ``(basis, singular_values)`` as NumPy arrays. The basis has shape
        ``(image_size, rank)`` in flattened image order and is zero off the mask.
        Its columns are the left singular vectors of the pixel-by-image residual
        matrix times their singular values divided by ``sqrt(n)``. All singular
        values are returned, before division, even when ``rank`` is zero.

    Raises:
        ValueError: If the stack is empty or lacks an image axis, rank is invalid,
            or any residual on the mask is non-finite.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim < 2 or residuals.shape[0] == 0:
        raise ValueError("residuals must contain at least one residual image")
    n = residuals.shape[0]
    if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= n:
        raise ValueError("residual rank must be between zero and n residual images")
    used = np.broadcast_to(
        np.asarray(True if mask is None else mask, dtype=bool), residuals.shape[1:]
    )
    if rank > used.sum():
        raise ValueError("residual rank exceeds the number of used pixels")
    matrix = residuals.reshape(n, used.size)[:, used.ravel()].T
    if not np.all(np.isfinite(matrix)):
        raise ValueError("residuals must be finite on the mask")
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    basis = np.zeros((used.size, rank))
    basis[used.ravel()] = left[:, :rank] * singular[:rank] / np.sqrt(n)
    return basis, singular


def _build(data, sigma, mask, kind, log_const, shape_param=0.0):
    data = jnp.asarray(data)
    sigma = jnp.asarray(sigma)
    dtype = jnp.result_type(data, sigma, float)
    data = data.astype(dtype)
    sigma = sigma.astype(dtype)
    mask = jnp.ones((), bool) if mask is None else jnp.asarray(mask, dtype=bool)
    shape = jnp.broadcast_shapes(data.shape, sigma.shape, mask.shape)
    data = jnp.broadcast_to(data, shape)
    sigma = jnp.broadcast_to(sigma, shape)
    mask = jnp.broadcast_to(mask, shape)

    data = jnp.where(mask, data, 0.0)
    sigma = jnp.where(mask, sigma, 1.0)
    n_used = jnp.sum(mask)
    log_norm = n_used * log_const - jnp.sum(jnp.log(sigma))
    return MaskedLogLikelihood(
        data=data,
        sigma=sigma,
        mask=mask,
        log_norm=log_norm,
        kind=kind,
        shape_param=float(shape_param),
    )


def gaussian_loglike(data, sigma, mask=None) -> Callable:
    """Build the Gaussian log-likelihood ``predicted -> scalar``.

    Args:
        data: Observed values.
        sigma: Standard deviation, scalar or broadcastable to ``data``.
        mask: Optional boolean array broadcastable to ``data``; True uses the element.

    Returns:
        A callable summing ``log N(predicted; data, sigma)`` over used elements.
    """
    return _build(data, sigma, mask, "gaussian", -_LOG_SQRT_2PI)


def low_rank_gaussian_loglike(
    data, sigma, basis, mask=None
) -> LowRankGaussianLogLikelihood:
    """Build a Gaussian with diagonal noise plus a marginalized low-rank term.

    The basis is shared between images, with unit Gaussian coefficients
    marginalized independently for each image. The resulting covariance is
    ``diag(sigma**2) + basis @ basis.T`` on used pixels. The Woodbury identity
    and determinant lemma avoid a dense pixel covariance; see Rasmussen and
    Williams (2006), *Gaussian Processes for Machine Learning*, appendix A.3.
    Construct outside JAX tracing to factor the covariance on the host.

    Args:
        data: Observed values.
        sigma: Diagonal noise standard deviation, scalar or broadcastable to
            ``data``; finite and positive on used pixels.
        basis: Array of shape ``(data.size, rank)`` in flattened image order and
            data units. Both data and basis must be finite on used pixels.
        mask: Optional boolean array broadcastable to ``data``; True uses a pixel.

    Returns:
        A :class:`LowRankGaussianLogLikelihood`, callable on predicted images,
        with inverse-covariance and conditional residual-mean methods. Rank zero
        reproduces :func:`gaussian_loglike`.
    """
    return LowRankGaussianLogLikelihood(data, sigma, basis, mask)


def laplace_loglike(data, sigma, mask=None) -> Callable:
    """Build the Laplace log-likelihood ``predicted -> scalar`` (an L1 loss).

    Args:
        data: Observed values.
        sigma: Laplace scale ``b`` (standard deviation ``sqrt(2) * b``), scalar or
            broadcastable to ``data``.
        mask: Optional boolean array broadcastable to ``data``; True uses the element.

    Returns:
        A callable summing ``-|predicted - data| / sigma - log(2 sigma)`` over used
        elements.
    """
    return _build(data, sigma, mask, "laplace", -math.log(2.0))


def huber_loglike(data, sigma, mask=None, *, delta: float = 1.345) -> Callable:
    """Build the Huber log-likelihood ``predicted -> scalar``.

    The density of ``z = (predicted - data) / sigma`` is ``exp(-rho(z)) / Z`` with
    ``rho(z) = z**2 / 2`` for ``|z| <= delta`` and ``delta |z| - delta**2 / 2``
    beyond: Gaussian core, Laplace tails of rate ``delta``, with ``rho`` and ``rho'``
    continuous at ``|z| = delta``. The normalizer is exact,

        Z = sqrt(2 pi) erf(delta / sqrt(2)) + 2 exp(-delta**2 / 2) / delta,

    the Gaussian mass inside the core plus two exponential tails. The log-likelihood
    is therefore a proper density in the data. As ``delta`` grows it approaches the
    Gaussian, and the default 1.345 is the conventional 95 percent efficiency point.

    Args:
        data: Observed values.
        sigma: Scale of the Gaussian core, scalar or broadcastable to ``data``.
        mask: Optional boolean array broadcastable to ``data``; True uses the element.
        delta: Core threshold in units of ``sigma``; must be positive.

    Returns:
        A callable summing ``-rho(z) - log(sigma) - log(Z)`` over used elements.
    """
    delta = float(delta)
    if not delta > 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    z_norm = (
        math.sqrt(2.0 * math.pi) * math.erf(delta / math.sqrt(2.0))
        + 2.0 * math.exp(-0.5 * delta**2) / delta
    )
    return _build(data, sigma, mask, "huber", -math.log(z_norm), delta)


def student_t_loglike(data, sigma, mask=None, *, nu: float = 4.0) -> Callable:
    """Build the Student-t log-likelihood ``predicted -> scalar``.

    Args:
        data: Observed values.
        sigma: Scale (not the standard deviation, which is ``sigma sqrt(nu / (nu - 2))``
            for ``nu > 2``), scalar or broadcastable to ``data``.
        mask: Optional boolean array broadcastable to ``data``; True uses the element.
        nu: Degrees of freedom, a fixed positive float. Small values give heavier
            tails; large values approach the Gaussian.

    Returns:
        A callable summing the Student-t log density over used elements.
    """
    nu = float(nu)
    if not nu > 0.0:
        raise ValueError(f"nu must be positive, got {nu}")
    log_const = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
    )
    return _build(data, sigma, mask, "student_t", log_const, nu)
