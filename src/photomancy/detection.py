"""Matched-filter detection: profile a source amplitude, map it over positions.

A detection asks whether adding a source of unknown amplitude ``a`` at a trial position
improves the fit of a model that does not contain it. With a base prediction ``b`` (the
best fit of the null model) and the image ``t(x)`` of a unit-amplitude source at
position ``x``, the likelihood ratio statistic is

    Lambda(x) = 2 [max_a log L(b + a t(x)) - log L(b)],

and its signed root ``z = sign(a_hat) sqrt(Lambda)`` is the detection significance at
``x``. For a Gaussian likelihood this is the whitened matched filter exactly: with
residual ``r = data - b`` and weights ``W = 1 / sigma**2``,
``a_hat = t' W r / t' W t`` and ``z = t' W r / sqrt(t' W t)``.

The functions here never render an image themselves. The caller supplies the base
prediction and a template function, an ``eqx.Module``, so the model that produces
``t(x)`` (optics, detector, spectral weights) stays outside this module.

- :func:`profile_amplitude` maximizes the log-likelihood over one amplitude with a
  fixed number of Newton steps.
- :func:`likelihood_ratio_map` does that at every trial position, for one or several
  exposures that share the amplitude.
- :func:`curvature_block` gives the negative Hessian of a log-density over one small
  group of parameters, the check that a group is identifiable before it is freed.
"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp

__all__ = [
    "DetectionMap",
    "curvature_block",
    "likelihood_ratio_map",
    "profile_amplitude",
]


class DetectionMap(eqx.Module):
    """Result of :func:`likelihood_ratio_map`, one entry per trial position.

    Attributes:
        a_hat: Best-fit amplitude at each position.
        z: Signed root of the likelihood ratio statistic,
            ``sign(a_hat) * sqrt(2 [max_a logL(a) - logL(0)])``.
        sigma: Formal standard error of ``a_hat``, ``1 / sqrt(curvature)``. It is
            infinite where the template is zero.
    """

    a_hat: jax.Array
    z: jax.Array
    sigma: jax.Array


def _check_curvature(likelihood) -> None:
    """Reject a likelihood that is piecewise linear in its prediction."""
    if getattr(likelihood, "kind", None) == "laplace":
        raise ValueError(
            "the Laplace likelihood is piecewise linear in the prediction, so its "
            "curvature in the amplitude is zero and Newton steps have nothing to "
            "follow; use a Gaussian or Huber likelihood"
        )


def _profile(objective: Callable, n_newton: int, dtype):
    """Newton-maximize a scalar ``objective(a)`` from ``a = 0``.

    The derivatives are taken in forward mode, so a template that is NaN where the
    likelihood is masked adds nothing: the mask selects the tangent before it is used.
    A step is taken only where the curvature is negative.
    """
    d1 = jax.jacfwd(objective)
    d2 = jax.jacfwd(d1)

    def step(a, _):
        g, h = d1(a), d2(a)
        ok = h < 0.0
        return a - jnp.where(ok, g / jnp.where(ok, h, 1.0), 0.0), None

    a0 = jnp.zeros((), dtype)
    a_hat, _ = jax.lax.scan(step, a0, None, length=n_newton)
    two_delta = 2.0 * (objective(a_hat) - objective(a0))
    return a_hat, two_delta, -d2(a_hat)


def profile_amplitude(loglike_of_prediction, base, template, n_newton: int = 4):
    """Maximize ``a -> loglike(base + a * template)`` over the amplitude ``a``.

    Runs ``n_newton`` Newton steps from ``a = 0``, a fixed count with no data-dependent
    loop, so the function jits and vmaps. A Gaussian log-likelihood is quadratic in
    ``a`` and is exact after one step; a Huber log-likelihood, which is piecewise
    quadratic, needs a few. A step is skipped where the curvature is not negative, so
    a zero template returns ``a_hat = 0`` with no NaN.

    A likelihood must be curved in the prediction. The Laplace builder is rejected with
    a ``ValueError`` because its curvature is zero everywhere. Newton steps assume the
    log-likelihood is concave in ``a`` near the optimum, as it is for the Gaussian and
    Huber builders; a heavy-tailed density such as Student-t is not guaranteed to
    converge from ``a = 0``.

    Elements the likelihood masks out may be NaN in ``base`` and ``template``; they
    contribute nothing to the value or the derivatives.

    Args:
        loglike_of_prediction: ``prediction -> scalar log-likelihood``, normally a
            builder from :mod:`photomancy.likelihoods`.
        base: Prediction of the model without the source.
        template: Image of the source at unit amplitude, broadcastable with ``base``.
        n_newton: Number of Newton steps.

    Returns:
        ``(a_hat, two_delta_loglike, curvature)``: the amplitude estimate, the
        statistic ``2 [logL(a_hat) - logL(0)]`` (zero or positive), and the negative
        second derivative of the log-likelihood in ``a`` at ``a_hat``. The formal
        standard error of ``a_hat`` is ``1 / sqrt(curvature)``.

    Raises:
        ValueError: If the likelihood is the Laplace density.
    """
    _check_curvature(loglike_of_prediction)
    base = jnp.asarray(base)
    template = jnp.asarray(template)
    dtype = jnp.result_type(base, template, float)

    def objective(a):
        return loglike_of_prediction(base + a * template)

    return _profile(objective, n_newton, dtype)


def _map_terms(terms, n_newton, position):
    images = tuple(template_fn(position) for _, _, template_fn in terms)

    def objective(a):
        return sum(
            likelihood(base + a * image)
            for (likelihood, base, _), image in zip(terms, images, strict=True)
        )

    dtype = jnp.result_type(*(b for _, b, _ in terms), float)
    a_hat, two_delta, curvature = _profile(objective, n_newton, dtype)
    z = jnp.sign(a_hat) * jnp.sqrt(jnp.maximum(two_delta, 0.0))
    sigma = jnp.where(curvature > 0.0, jax.lax.rsqrt(jnp.abs(curvature)), jnp.inf)
    return a_hat, z, sigma


@eqx.filter_jit
def _run_map(terms, positions, n_newton, batch_size):
    a_hat, z, sigma = jax.lax.map(
        lambda x: _map_terms(terms, n_newton, x), positions, batch_size=batch_size
    )
    return DetectionMap(a_hat=a_hat, z=z, sigma=sigma)


def likelihood_ratio_map(
    terms: Sequence[tuple[Callable, Any, Callable]],
    positions,
    batch_size: int | None = None,
    n_newton: int = 4,
) -> DetectionMap:
    """Profile a shared source amplitude at every trial position.

    At each position ``x`` the amplitude maximizes the summed log-likelihood
    ``sum_k loglike_k(base_k + a * template_k(x))`` over all terms, so several
    exposures constrain one amplitude. The statistic is
    ``Lambda = 2 [max_a logL(a) - logL(0)]``; the map reports ``a_hat``, the signed
    root ``z = sign(a_hat) sqrt(Lambda)`` and the formal standard error of ``a_hat``.
    With Gaussian likelihoods this is the whitened matched filter.

    Positions are processed with :func:`jax.lax.map`, ``batch_size`` of them at a time
    (all in sequence when ``None``), which sets the peak memory: each position renders
    one template per term. Masked or NaN pixels do not leak, as in
    :func:`profile_amplitude`.

    Args:
        terms: Sequence of ``(likelihood, base_prediction, template_fn)``. The
            likelihood is ``prediction -> scalar``; ``template_fn(position)`` returns
            the unit-amplitude source image for that term at one position. Make
            ``template_fn`` an ``eqx.Module`` so the arrays it holds are traced inputs
            instead of compiled constants.
        positions: Trial positions, an array (or PyTree of arrays) whose leading axis
            indexes positions; ``template_fn`` receives one slice.
        batch_size: Positions per ``lax.map`` batch, or ``None`` for one at a time.
        n_newton: Newton steps per position, as in :func:`profile_amplitude`.

    Returns:
        A :class:`DetectionMap` with ``a_hat``, ``z`` and ``sigma`` of shape
        ``(n_positions,)``.

    Raises:
        ValueError: If a likelihood is the Laplace density, or ``terms`` is empty.
    """
    terms = tuple((lik, base, fn) for lik, base, fn in terms)
    if not terms:
        raise ValueError("terms must contain at least one (likelihood, base, template)")
    for likelihood, _, _ in terms:
        _check_curvature(likelihood)
    terms = tuple((lik, jnp.asarray(base), fn) for lik, base, fn in terms)
    return _run_map(terms, positions, n_newton, batch_size)


def curvature_block(logdensity: Callable, params, where: Callable) -> jax.Array:
    """Negative Hessian of ``logdensity`` over one selected group of leaves.

    The group is chosen with a selector ``where(params) -> leaf | sequence of leaves``,
    the form ``equinox.tree_at`` takes (and that
    :func:`photomancy.backends.staged.group_labels` uses). Every other leaf is held at
    its value in ``params``. The Hessian is taken over the flattened group, in the
    order the leaves are selected, and returned as a dense symmetric matrix.

    The block is the observed information of the group at ``params``. Its eigenvalues
    show whether the group is identifiable before it is freed (a near-zero eigenvalue
    is a direction the data do not constrain), and at a maximum of the log-density its
    inverse is the formal covariance of the group.

    Args:
        logdensity: ``params -> scalar``.
        params: The PyTree to differentiate at.
        where: Selector of the group.

    Returns:
        A ``(n, n)`` matrix, ``n`` the number of scalars in the group.
    """
    selected = where(params)
    flat, unravel = jax.flatten_util.ravel_pytree(selected)

    def negative(vector):
        return -logdensity(eqx.tree_at(where, params, unravel(vector)))

    hessian = jax.hessian(negative)(flat)
    return 0.5 * (hessian + hessian.T)
