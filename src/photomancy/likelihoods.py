"""Masked, normalized log-likelihood builders for gridded data.

Each builder takes the observed ``data``, a per-element ``sigma`` and an optional
boolean ``mask`` (True means the element is used) and returns a callable
``predicted -> scalar log-likelihood``. That callable is the ``likelihood`` plug-in
of :func:`photomancy.core.build_scene_logdensity`.

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
