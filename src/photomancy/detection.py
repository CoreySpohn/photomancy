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
- :func:`detect` is the same map over a :class:`~photomancy.infer.fit.Fit`: one free
  path is the amplitude and is profiled at each trial of the others, so the noise and
  the response come from the fit's terms (a :class:`photomancy.image.ImageTerm` or any
  ``AbstractTerm``) and the coordinates reach the model as ``Fit.model_at`` writes them.
- :func:`measure` maximizes the fit's likelihood from a start (a detection peak) and
  returns a :class:`LikelihoodEstimate`: the value as an entry dictionary, the
  covariance from the observed information (``Fit.information(kind="observed")``) and
  whether that information is positive definite there.
- :func:`curvature_block` gives the negative Hessian of a log-density over one small
  group of parameters, the check that a group is identifiable before it is freed.
"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
from hwoutils.memory import DEFAULT_MAX_BYTES
from hwoutils.model_paths import resolve
from priorities import AbstractDistribution

from photomancy.infer.fit import Fit, observed_hessian

__all__ = [
    "DetectionMap",
    "LikelihoodEstimate",
    "curvature_block",
    "detect",
    "likelihood_ratio_map",
    "measure",
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
        assumptions: The fit's ``Fit.assumptions`` for a map from :func:`detect`;
            empty from :func:`likelihood_ratio_map`, which holds no fit. A map is
            comparable with another only under the same assumptions.
    """

    a_hat: jax.Array
    z: jax.Array
    sigma: jax.Array
    assumptions: tuple[str, ...] = eqx.field(static=True, default=())


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
            builder from :mod:`photomancy.infer.likelihoods`.
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


def _summarize(a_hat, two_delta, curvature):
    z = jnp.sign(a_hat) * jnp.sqrt(jnp.maximum(two_delta, 0.0))
    sigma = jnp.where(curvature > 0.0, jax.lax.rsqrt(jnp.abs(curvature)), jnp.inf)
    return a_hat, z, sigma


def _map_terms(terms, n_newton, position):
    images = tuple(template_fn(position) for _, _, template_fn in terms)

    def objective(a):
        return sum(
            likelihood(base + a * image)
            for (likelihood, base, _), image in zip(terms, images, strict=True)
        )

    dtype = jnp.result_type(*(b for _, b, _ in terms), float)
    return _summarize(*_profile(objective, n_newton, dtype))


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


def _check_entries(entries, who):
    if set(entries) != set(who):
        raise ValueError(
            f"entries must have exactly the keys {sorted(who)}, got {sorted(entries)}"
        )


@eqx.filter_jit
def _detect(fit, trials, amplitude, n_newton, batch_size):
    shape = jnp.shape(resolve(fit.model, amplitude))

    def at(x):
        def objective(a):
            full = {**x, amplitude: jnp.reshape(a, shape)}
            return jnp.asarray(fit.log_likelihood(full), jnp.float64)

        return _summarize(*_profile(objective, n_newton, jnp.float64))

    a_hat, z, sigma = jax.lax.map(at, trials, batch_size=batch_size)
    return DetectionMap(a_hat=a_hat, z=z, sigma=sigma, assumptions=fit.assumptions)


def detect(
    fit: Fit,
    trials: dict,
    *,
    amplitude: str,
    n_newton: int = 4,
    batch_size: int | None = None,
) -> DetectionMap:
    """Matched-filter map of a source over trial values, from the fit's likelihood.

    At each trial the free path ``amplitude`` is profiled: ``a`` maximizes
    ``fit.log_likelihood`` by Newton steps from ``a = 0`` while every other free entry
    is held at the trial value, and the map reports ``a_hat``, the signed root
    ``z = sign(a_hat) sqrt(2 [ell(a_hat) - ell(0)])`` and the formal standard error
    ``1 / sqrt(curvature)``. The noise, the response and the detector come from the
    fit's terms, so there is no second likelihood, and the way coordinates reach the
    model is the fit's own (``Fit.model_at``: leaf laws, then blocks). Priors play no
    part. When the model is linear in ``a`` and the likelihood is Gaussian with fixed
    variance, this is the classical whitened matched filter ``z = t'Wr / sqrt(t'Wt)``.
    Several terms (exposures) constrain one shared amplitude.

    Args:
        fit: A :class:`~photomancy.infer.fit.Fit` whose model holds the source. At
            ``amplitude = 0`` the source must be absent (zero flux), so the statistic
            compares against the model without it.
        trials: Entry dictionary in the format of the fit's ``x``, one leading trial
            axis on every entry, with exactly the keys of ``fit.free`` except
            ``amplitude``.
        amplitude: The free leaf-law path that is profiled; its leaf has one element.
        n_newton: Newton steps per trial. A Gaussian likelihood linear in the
            amplitude is exact after one; a Poisson likelihood needs a few.
        batch_size: Trials per ``jax.lax.map`` batch (which sets peak memory), or
            ``None`` for one at a time.

    Returns:
        A :class:`DetectionMap` with leaves of shape ``(n_trials,)`` and the fit's
        ``assumptions``.

    Raises:
        ValueError: If ``amplitude`` is not a free leaf law of one element, or the
            keys of ``trials`` are not the other free paths.
    """
    entry = fit.free.get(amplitude)
    if not isinstance(entry, AbstractDistribution):
        raise ValueError(f"amplitude {amplitude!r} must be a free leaf-law path")
    if jnp.size(resolve(fit.model, amplitude)) != 1:
        raise ValueError(f"the leaf at {amplitude!r} must have one element")
    _check_entries(trials, set(fit.free) - {amplitude})
    return _detect(fit, trials, amplitude, n_newton, batch_size)


class LikelihoodEstimate(eqx.Module):
    """The maximum of a fit's likelihood and the covariance from its curvature.

    Attributes:
        value: The estimate, an entry dictionary keyed by ``fit.free`` (the format of
            the fit's ``x``).
        cov: The formal covariance, the inverse of the observed information
            (``Fit.information(kind="observed", include_prior=False)`` at the estimate),
            shape ``(D, D)`` with rows in the order of
            :func:`photomancy.infer.posterior.ravel_samples` (the entries' leaves in
            ``jax.tree.leaves`` order, each row-major). It is a likelihood-only
            statement and says nothing about the prior. Where ``regular`` is false the
            diagonal can be negative and the widths in :attr:`sigma` are NaN there: the
            point is not a maximum and the NaN says so.
        corr: The correlation matrix ``cov_ij / sqrt(cov_ii cov_jj)``; meaningful where
            ``regular`` is true.
        regular: True when the observed information is positive definite, so the
            estimate is a regular maximum and ``cov`` is a covariance. False at a
            saddle, on a ridge or off the peak (an undetected source).
        min_eigenvalue: The smallest eigenvalue of the observed information, in the
            units of the stored coordinates (negative when ``regular`` is false).
        assumptions: The fit's ``Fit.assumptions``, copied because the estimate does
            not hold the fit. An estimate is comparable with another only under the
            same assumptions.
    """

    value: dict
    cov: jax.Array
    corr: jax.Array
    regular: jax.Array
    min_eigenvalue: jax.Array
    assumptions: tuple[str, ...] = eqx.field(static=True, default=())

    @property
    def sigma(self) -> jax.Array:
        """Formal standard errors, shape ``(D,)``; NaN where ``cov`` is negative."""
        return jnp.sqrt(jnp.diag(self.cov))


def _require_float64(flat):
    if not jax.config.jax_enable_x64 or flat.dtype != jnp.float64:
        raise RuntimeError(
            "curvature needs float64 (jax_enable_x64 on and float64 parameters); "
            "a float32 Hessian is too noisy to invert"
        )


def _correlation_inverse(info):
    """Invert a symmetric matrix in correlation form and classify it.

    Scaling by the square roots of the absolute diagonal, ``C = D^-1 A D^-1``, takes a
    matrix whose entries span many orders of magnitude (condition number 4e12 on a
    full-scene fit) to one near unity (3.7 there); the inverse is
    ``D^-1 C^-1 D^-1``. A congruence by a positive diagonal keeps the signature, so
    the sign of the eigenvalues of ``C`` is that of ``A`` without the scale's rounding.

    Returns:
        ``(cov, corr, regular, min_eigenvalue)``: the inverse, its correlation matrix,
        whether ``A`` is positive definite, and the smallest eigenvalue of ``A``.
    """
    scale = jnp.sqrt(jnp.abs(jnp.diag(info)))
    scale = jnp.where(scale > 0.0, scale, 1.0)
    scaled = info / jnp.outer(scale, scale)
    inverse = jnp.linalg.inv(scaled)
    cov = inverse / jnp.outer(scale, scale)
    cov = 0.5 * (cov + cov.T)
    root = jnp.sqrt(jnp.abs(jnp.diag(inverse)))
    root = jnp.where(root > 0.0, root, 1.0)
    corr = inverse / jnp.outer(root, root)
    regular = jnp.min(jnp.linalg.eigvalsh(scaled)) > 0.0
    return cov, 0.5 * (corr + corr.T), regular, jnp.min(jnp.linalg.eigvalsh(info))


@eqx.filter_jit
def _measure(fit, start, n_newton, batch_size, max_bytes):
    start = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), start)
    flat, unravel = jax.flatten_util.ravel_pytree(start)
    _require_float64(flat)

    def loglike(theta):
        return jnp.asarray(fit.log_likelihood(unravel(theta)), jnp.float64)

    def information(theta):
        return fit.information(
            unravel(theta),
            kind="observed",
            include_prior=False,
            batch_size=batch_size,
            max_bytes=max_bytes,
        )

    grad = jax.grad(loglike)

    def step(theta, _):
        g, h = grad(theta), -information(theta)
        ok = jnp.all(jnp.linalg.eigvalsh(h) < 0.0)
        delta = jnp.linalg.solve(jnp.where(ok, h, -jnp.eye(theta.size)), g)
        return theta - jnp.where(ok, delta, 0.0), None

    theta, _ = jax.lax.scan(step, flat, None, length=n_newton)
    return (unravel(theta), *_correlation_inverse(information(theta)))


def measure(
    fit: Fit,
    start: dict,
    *,
    n_newton: int = 4,
    batch_size: int | None = 1,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
) -> LikelihoodEstimate:
    """Maximize a fit's likelihood from ``start`` and take the covariance there.

    Newton steps on ``fit.log_likelihood`` over every free scalar jointly, from
    ``start`` (a detection peak from :func:`detect`, say). A step is skipped while the
    Hessian is not negative definite. The covariance is the inverse of the observed
    information (:meth:`photomancy.infer.fit.Fit.information` with ``kind="observed"``
    and ``include_prior=False``) at the estimate, inverted in correlation form, so it
    is the formal covariance of the maximum-likelihood estimate and is not a statement
    about the prior. The estimate is only as good as the start: a start far from any
    peak can leave the Newton iteration on a saddle.

    The result says whether it is a maximum. ``regular`` is false, and the widths of
    the affected parameters are NaN, where the information is not positive definite: an
    undetected source (its amplitude is consistent with zero, so its position is not
    constrained), a saddle, or a point the iteration did not reach the peak from. The
    Fisher matrix (``kind="fisher"``) cannot make that statement.

    Needs ``jax_enable_x64``.

    Args:
        fit: A :class:`~photomancy.infer.fit.Fit`.
        start: Entry dictionary, the format of the fit's ``x`` for one draw, with
            exactly the keys of ``fit.free``.
        n_newton: Newton steps.
        batch_size: Hessian columns per batch (``None`` for all at once); the default
            holds one gradient's intermediates.
        max_bytes: Memory limit of the closed-form estimate, as in ``Fit.information``.

    Returns:
        A :class:`LikelihoodEstimate`.

    Raises:
        ValueError: If the keys of ``start`` are not the free paths.
        RuntimeError: If ``jax_enable_x64`` is off.
    """
    _check_entries(start, set(fit.free))
    value, cov, corr, regular, min_eig = _measure(
        fit, start, n_newton, batch_size, max_bytes
    )
    return LikelihoodEstimate(
        value=value,
        cov=cov,
        corr=corr,
        regular=regular,
        min_eigenvalue=min_eig,
        assumptions=fit.assumptions,
    )


def curvature_block(logdensity: Callable, params, where: Callable) -> jax.Array:
    """Negative Hessian of ``logdensity`` over one selected group of leaves.

    The group is chosen with a selector ``where(params) -> leaf | sequence of leaves``,
    the form ``equinox.tree_at`` takes (and that
    :func:`photomancy.infer.backends.staged.group_labels` uses). Every other leaf is
    held at its value in ``params``. The Hessian is taken over the flattened group, in
    the order the leaves are selected, and returned as a dense symmetric matrix.

    The block is the observed information of the group at ``params``, built by the same
    column-by-column Hessian that :meth:`photomancy.infer.fit.Fit.information` uses for
    ``kind="observed"`` (this function takes any scalar log-density, not only a fit,
    and needs float64). Its eigenvalues
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

    _require_float64(flat)
    return observed_hessian(negative, flat, batch_size=1)
