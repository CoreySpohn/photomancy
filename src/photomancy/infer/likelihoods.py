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
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from hwoutils.precision import float_dtype, like, require_float64

__all__ = [
    "LowRankGaussianLogLikelihood",
    "MaskedLogLikelihood",
    "gaussian_loglike",
    "huber_loglike",
    "laplace_loglike",
    "low_rank_gaussian_loglike",
    "residual_basis",
    "student_t_loglike",
]

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


def _information_constant(kind, shape_param):
    """Expected information of a location in units of ``1 / sigma**2``.

    ``E[psi(z)**2]`` of the kernel's score ``psi = d kernel / dz`` under the
    kernel's own normalized density: 1 for the Gaussian and the Laplace,
    ``sqrt(2 pi) erf(delta / sqrt(2)) / Z`` for the Huber density (``Z`` its
    normalization), ``(nu + 1) / (nu + 3)`` for the Student-t.
    """
    if kind in ("gaussian", "laplace"):
        return 1.0
    if kind == "student_t":
        return (shape_param + 1.0) / (shape_param + 3.0)
    delta = shape_param
    core = math.sqrt(2.0 * math.pi) * math.erf(delta / math.sqrt(2.0))
    return core / (core + 2.0 * math.exp(-0.5 * delta**2) / delta)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Covariance:
    """Fixed noise and its inference and detection factors, all dynamic leaves.

    JAX reconstructs registered dataclasses after a tree map. When a precision
    conversion narrows the inference normalizer, refactor from the converted
    noise here. Ordinary flattening, partitioning and scoring reuse the cache.

    Attributes:
        sigma: Diagonal noise scales, one outside the mask.
        mask: Boolean array indicating which elements are used.
        basis: Optional masked low-rank covariance basis in flattened element order.
        log_const: Per-element log normalization constant, or None for a fixed total.
        inference: Cached inference normalization and any low-rank factors.
        detection: Cached normalization and any low-rank factors in detection precision.
    """

    sigma: jax.Array
    mask: jax.Array
    basis: jax.Array | None
    log_const: float | None = field(metadata={"static": True})
    inference: tuple = ()
    detection: tuple = ()

    def __post_init__(self):
        if self.log_const is None:
            return  # A directly supplied total normalization.
        if self.inference:
            norm = self.inference[-1]
            if not eqx.is_array(norm) or norm.dtype != np.float32:
                return
            if isinstance(norm, jax.core.Tracer) and float_dtype() == np.float32:
                return  # Detection in a float32-only trace cannot use inference.

        # Concrete low-rank preparation also works in a float32-only process.
        # A traced diagonal builder can still be used in a detection program.
        traced = any(
            isinstance(x, jax.core.Tracer) for x in (self.sigma, self.mask, self.basis)
        )
        device_dtype = float_dtype()
        xp = jnp if traced or device_dtype == np.float64 else np
        dtype = device_dtype if traced else np.float64
        sigma = xp.asarray(self.sigma, dtype)
        norm = xp.sum(xp.asarray(self.mask), dtype=dtype) * self.log_const - xp.sum(
            xp.log(sigma)
        )
        factors = ()
        if self.basis is not None:
            # Factor concrete matrices on the host, even with x64 disabled.
            linalg = jnp if traced else np
            w = (
                linalg.asarray(self.basis, dtype)
                / linalg.asarray(self.sigma, dtype).ravel()[:, None]
            )
            chol = linalg.linalg.cholesky(linalg.eye(w.shape[1], dtype=dtype) + w.T @ w)
            norm = norm - linalg.log(linalg.diag(chol)).sum()
            factors = (w, chol)
        factors = (*factors, norm)
        if device_dtype == np.float64:
            factors = tuple(jnp.asarray(f) for f in factors)
        object.__setattr__(self, "inference", factors)
        object.__setattr__(
            self, "detection", tuple(jnp.asarray(f, self.sigma.dtype) for f in factors)
        )

    @property
    def working(self):
        """Select the matching factors even after widening a stored likelihood."""
        return self.inference if self.sigma.dtype == jnp.float64 else self.detection


class _FixedNoiseLikelihood(eqx.Module):
    data: jax.Array
    _noise: _Covariance

    @property
    def sigma(self):
        """Diagonal noise scales, one outside the mask."""
        return self._noise.sigma

    @property
    def mask(self):
        """Pixels used in the score."""
        return self._noise.mask

    @property
    def log_norm(self):
        """Total normalization in the detection precision."""
        return self._noise.working[-1]


class MaskedLogLikelihood(_FixedNoiseLikelihood):
    """Normalized log-likelihood of a standardized-residual density over used elements.

    Built by :func:`gaussian_loglike`, :func:`laplace_loglike`,
    :func:`huber_loglike` and :func:`student_t_loglike`. The data, sigma and mask are
    Module leaves, so a backend that passes this as a ``filter_jit`` argument threads
    them as traced inputs rather than baking them into the compiled program.

    Attributes:
        kind: Density name, a key of the kernel table.
        shape_param: ``delta`` for Huber, ``nu`` for Student-t, unused otherwise.
    """

    kind: str = eqx.field(static=True)
    shape_param: float = eqx.field(static=True)

    def __init__(self, data, sigma, mask, log_norm, kind, shape_param):
        """Store a supplied total normalization or prepare it from the masked noise.

        Args:
            data: Observed values with excluded elements set to zero.
            sigma: Scale at every element, with excluded elements set to one.
            mask: Boolean array, True where the element is used.
            log_norm: Total log normalization, or None to compute it from the noise.
            kind: Density name, a key of the kernel table.
            shape_param: ``delta`` for Huber, ``nu`` for Student-t, unused otherwise.
        """
        self.data = data
        self.kind = kind
        self.shape_param = shape_param
        if log_norm is None:
            self._noise = _Covariance(
                sigma, mask, None, _kernel_log_const(kind, shape_param)
            )
        else:
            self._noise = _Covariance(sigma, mask, None, None, (log_norm,), (log_norm,))

    def __call__(self, predicted):
        """Return the log-likelihood of ``predicted`` (broadcastable to ``data``).

        Args:
            predicted: Model prediction, broadcastable to ``data``.

        Returns:
            Scalar normalized log likelihood over the unmasked elements.

        Raises:
            PrecisionError: When float64 inference is unavailable.
        """
        require_float64("masked log density")
        data = self.data.astype(jnp.float64)
        sigma = self.sigma.astype(jnp.float64)
        norm = self._noise.inference[-1]
        return self._score(jnp.asarray(predicted, jnp.float64), data, sigma, norm)

    def detection_score(self, predicted):
        """Evaluate the profile-detection objective in the data precision.

        Args:
            predicted: Model prediction, broadcastable to the observations.

        Returns:
            Scalar objective over the unmasked elements.
        """
        return self._score(
            like(predicted, self.data), self.data, self.sigma, self.log_norm
        )

    def _score(self, predicted, data, sigma, log_norm):
        z = (jnp.broadcast_to(predicted, data.shape) - data) / sigma
        z = jnp.where(self.mask, z, 0)
        kernel = _KERNELS[self.kind](z, self.shape_param)
        return log_norm - jnp.sum(jnp.where(self.mask, kernel, 0))

    def residual(self, predicted):
        """Return ``(data - predicted) / sigma``, NaN at excluded elements.

        Defined for the Gaussian density only, where it has mean zero and unit
        variance at the right prediction. A robust density (Laplace, Huber,
        Student-t) has no residual with that null and raises.

        Args:
            predicted: Model prediction, broadcastable to ``data``.

        Returns:
            Whitened residual with the shape of ``data``, NaN where ``mask`` is false.

        Raises:
            NotImplementedError: When the density is not the Gaussian.
        """
        if self.kind != "gaussian":
            raise NotImplementedError(
                f"the {self.kind} likelihood has no residual with a unit-variance null"
            )
        predicted = jnp.broadcast_to(
            jnp.asarray(predicted, float_dtype()), self.data.shape
        )
        return jnp.where(self.mask, (self.data - predicted) / self.sigma, jnp.nan)

    def fisher_root(self, predicted):
        """Return the flat vector whose Jacobian gives the expected information.

        The scale ``sigma`` is fixed, so the information of the prediction is
        ``c / sigma**2`` per used element, with ``c`` the expected squared score of
        the kernel under its own density: 1 for the Gaussian and the Laplace
        density, ``(nu + 1) / (nu + 3)`` for the Student-t, and the Huber constant of
        the normalization (see ``_information_constant``). The root is
        ``sqrt(c) * predicted / sigma``, zero at excluded elements, shape
        ``(data.size,)``.

        Args:
            predicted: Model prediction, broadcastable to ``data``.

        Returns:
            Flat Fisher root, shape ``(data.size,)``, zero at excluded elements.
        """
        predicted = jnp.broadcast_to(
            jnp.asarray(predicted, float_dtype()), self.data.shape
        )
        scale = math.sqrt(_information_constant(self.kind, self.shape_param))
        return jnp.ravel(jnp.where(self.mask, scale * predicted / self.sigma, 0.0))


class LowRankGaussianLogLikelihood(_FixedNoiseLikelihood):
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
    in double precision. Scoring reuses the double-precision factors. Detection
    and inverse-covariance operations use factors in the input dtype. All arrays
    are Module leaves, so passing this object as
    a ``filter_jit`` argument keeps them as traced inputs. Excluded pixels
    contribute neither density nor gradient, even when their inputs are NaN.
    """

    @property
    def U(self):
        """Basis in the storage precision, zero outside the mask."""
        return self._noise.basis

    @property
    def whitened(self):
        """Whitened basis for detection and inverse-covariance operations."""
        return self._noise.working[0]

    @property
    def cholesky(self):
        """Cholesky factor for detection and inverse-covariance operations."""
        return self._noise.working[1]

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
        self.data = jnp.asarray(np.where(mask, data, 0), dtype)
        self._noise = _Covariance(
            jnp.asarray(np.where(mask, sigma, 1), dtype),
            jnp.asarray(mask),
            jnp.asarray(np.where(mask.ravel()[:, None], basis, 0), dtype),
            -_LOG_SQRT_2PI,
        )

    def _whiten(self, image):
        return (jnp.where(self.mask, image, 0) / self.sigma).ravel()

    def _coefficients(self, z):
        return jsp.linalg.cho_solve((self.cholesky, True), self.whitened.T @ z)

    def solve(self, image):
        """Apply the masked inverse covariance; return an image shaped like data.

        ``image`` is broadcastable to ``data``. Excluded pixels return zero,
        including where the input is NaN.

        Args:
            image: Image to apply the inverse covariance to, broadcastable to ``data``.

        Returns:
            Image with the shape of ``data``, zero at excluded pixels.
        """
        z = self._whiten(image)
        if self.U.shape[1]:
            z = z - self.whitened @ self._coefficients(z)
        return z.reshape(self.data.shape) / self.sigma

    def posterior_term(self, predicted):
        """Return ``U E[a | data, predicted]`` for the unit Gaussian coefficients.

        ``predicted`` is broadcastable to ``data``. The result has the data shape
        and is zero outside the mask; rank zero returns an all-zero image.

        Args:
            predicted: Model prediction, broadcastable to ``data``.

        Returns:
            Image with the shape of ``data``, zero outside the mask.
        """
        z = self._whiten(self.data - jnp.where(self.mask, predicted, 0))
        if not self.U.shape[1]:
            return jnp.zeros_like(self.data)
        return (self.U @ self._coefficients(z)).reshape(self.data.shape)

    def __call__(self, predicted):
        """Return the normalized log density; ``predicted`` broadcasts to data.

        Args:
            predicted: Model prediction, broadcastable to ``data``.

        Returns:
            Scalar normalized log density of ``data`` given ``predicted``.

        Raises:
            PrecisionError: When float64 inference is unavailable.
        """
        require_float64("low-rank Gaussian log density")
        predicted = jnp.asarray(jnp.where(self.mask, predicted, 0), jnp.float64)
        data = self.data.astype(jnp.float64)
        sigma = self.sigma.astype(jnp.float64)
        z = (predicted - data) / sigma
        return self._score(z.ravel(), *self._noise.inference)

    def detection_score(self, predicted):
        """Evaluate the profile-detection objective in the data precision.

        Args:
            predicted: Model prediction, broadcastable to the observations.

        Returns:
            Scalar objective over the unmasked elements.
        """
        predicted = like(predicted, self.data)
        z = self._whiten(jnp.where(self.mask, predicted, 0) - self.data)
        return self._score(z, self.whitened, self.cholesky, self.log_norm)

    @staticmethod
    def _score(z, whitened, chol, log_norm):
        if not whitened.shape[1]:
            return log_norm - jnp.sum(0.5 * z**2)
        a = jsp.linalg.cho_solve((chol, True), whitened.T @ z)
        # Avoid subtracting two large quadratic forms for strongly aligned images.
        return log_norm - 0.5 * (jnp.sum((z - whitened @ a) ** 2) + jnp.sum(a**2))


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


def _kernel_log_const(kind, shape):
    if kind == "gaussian":
        return -_LOG_SQRT_2PI
    if kind == "laplace":
        return -math.log(2.0)
    if kind == "huber":
        core = math.sqrt(2.0 * math.pi) * math.erf(shape / math.sqrt(2.0))
        return -math.log(core + 2.0 * math.exp(-0.5 * shape**2) / shape)
    return (
        math.lgamma(0.5 * (shape + 1.0))
        - math.lgamma(0.5 * shape)
        - 0.5 * math.log(shape * math.pi)
    )


def _build(data, sigma, mask, kind, shape_param=0.0):
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

    data = jnp.where(mask, data, 0)
    sigma = jnp.where(mask, sigma, 1)
    return MaskedLogLikelihood(
        data=data,
        sigma=sigma,
        mask=mask,
        log_norm=None,
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
    return _build(data, sigma, mask, "gaussian")


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
    return _build(data, sigma, mask, "laplace")


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
    return _build(data, sigma, mask, "huber", delta)


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
    return _build(data, sigma, mask, "student_t", nu)
