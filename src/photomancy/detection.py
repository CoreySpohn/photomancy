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
  fixed number of safeguarded iterations and reports whether the peak was resolved.
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
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from hwoutils.memory import DEFAULT_MAX_BYTES
from hwoutils.model_paths import resolve
from priorities import AbstractDistribution

from photomancy.infer._scale import information_scale
from photomancy.infer.backends.staged import (
    STOP_MIN_DELTA,
    default_stage,
    map_fit,
)
from photomancy.infer.fit import Fit, _admit_information, _mapped, observed_hessian

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


def _profile(objective: Callable, n_newton: int, dtype, n_elements: int = 1):
    """Maximize a scalar objective from zero with forward-mode derivatives.

    Allow 8 eps times max(1, abs(current log likelihood)) per reduction level:
    one for a scalar score plus ceil(log2(n_elements)) for a pairwise image sum.
    This allows for rounding in the pixel scores and their accumulation. An
    opaque fit uses its term array sizes to estimate the reduction depth.
    Try the full Newton step and up to eight halvings. A shortened uphill
    step and the preceding rejected trial bracket a peak with the starting point.
    With nonnegative second derivative there is no local curvature scale. Start
    one amplitude unit along the slope and double up to 40 times while ascending.
    Refine a bracket with 48 golden-section steps, retaining its best finite point.

    Resolution requires a bracket of width at most 1e-8 * max(1, abs(a)), or a
    stationary point. With positive curvature, abs(slope) / sqrt(curvature)
    estimates the amplitude error in formal standard errors. A point is
    stationary if this error is at most 1e-6 or its square (the Newton
    decrement) is within the objective's rounding floor. A stationary flat
    null also resolves. Forward differentiation lets the likelihood mask NaN
    template tangents.
    """
    d1 = jax.jacfwd(objective)
    d2 = jax.jacfwd(d1)
    zero = jnp.zeros((), dtype)
    null = jnp.asarray(objective(zero))
    reduction_depth = 1 + (max(1, n_elements) - 1).bit_length()
    relative_roundoff = 8 * jnp.finfo(null.dtype).eps * reduction_depth

    def update_if(active, update, state):
        # A per-item cond under vmap evaluates both branches and batches their
        # operands, including the renderer. Branch once for all mapped items
        # and select only the updated scalar results for each active item.
        def selected():
            return jax.tree.map(
                lambda new, old: jnp.where(active, new, old), update(), state
            )

        return jax.lax.cond(eqx.internal.unvmap_any(active), selected, lambda: state)

    def roundoff(value):
        return relative_roundoff * jnp.maximum(1, jnp.abs(value))

    def small_slope(g, curvature, tolerance):
        return (curvature > 0) & (
            jnp.abs(g) <= tolerance * jnp.sqrt(jnp.abs(curvature))
        )

    def advance(a, value, g, h, move):
        newton = h < 0
        step = jnp.where(newton, -g / jnp.where(newton, h, 1), jnp.sign(g)).astype(
            dtype
        )
        floor = value - roundoff(value)

        def acceptable(offset, score):
            return jnp.isfinite(a + offset) & jnp.isfinite(score) & (score >= floor)

        def halve(_, trial):
            offset, score = trial
            return update_if(
                ~acceptable(offset, score),
                lambda: (offset / 2, objective(a + offset / 2)),
                trial,
            )

        full = objective(a + step)
        trial = (step, full)
        offset, score = update_if(
            move & ~acceptable(step, full),
            lambda: jax.lax.fori_loop(0, 8, halve, trial),
            trial,
        )
        accepted = acceptable(offset, score)
        best = jnp.where(accepted, a + offset, a)
        score = jnp.where(accepted, score, value)
        shortened = accepted & (offset != step)
        # A shortened uphill step has a rejected trial just beyond it.
        # A non-rising unit step can straddle a peak with the starting point.
        turned = move & ~newton & accepted & (score <= value)
        right = jnp.where(turned, best, a + 2 * offset)
        best = jnp.where(turned, a, best)
        score = jnp.where(turned, value, score)
        # A rejected trial (downhill or outside the domain) is the far end of a
        # bracket around the starting point, whatever the amplitude scale.
        failed = move & ~accepted
        right = jnp.where(failed, a + step, right)
        bracketed = turned | (shortened & (score > value)) | failed

        def expand(_, state):
            left, peak, right, peak_value, bracketed = state
            trial = a + 2 * (peak - a)
            trial_value = objective(trial)
            rising = ~bracketed & jnp.isfinite(trial) & jnp.isfinite(trial_value)
            rising &= trial_value > peak_value
            return (
                jnp.where(rising, peak, left),
                jnp.where(rising, trial, peak),
                jnp.where(bracketed, right, trial),
                jnp.where(rising, trial_value, peak_value),
                ~rising,
            )

        bounds = (a, best, right, score, bracketed)
        left, best, right, score, bracketed = update_if(
            move & ~newton & accepted & ~shortened & (score > value),
            lambda: jax.lax.fori_loop(0, 40, expand, bounds),
            bounds,
        )

        def refine(_, bounds):
            left, peak, right, peak_value = bounds
            go_right = jnp.abs(right - peak) > jnp.abs(left - peak)
            far = jnp.where(go_right, right, left)
            near = jnp.where(go_right, left, right)
            trial = peak + 0.3819660112501051 * (far - peak)
            trial_value = objective(trial)
            rising = (
                jnp.isfinite(trial)
                & jnp.isfinite(trial_value)
                & (trial_value > peak_value)
            )
            return (
                jnp.where(rising, peak, near),
                jnp.where(rising, trial, peak),
                jnp.where(rising, far, trial),
                jnp.where(rising, trial_value, peak_value),
            )

        bounds = (left, best, right, score)
        left, best, right, score = update_if(
            move & bracketed, lambda: jax.lax.fori_loop(0, 48, refine, bounds), bounds
        )
        closed = bracketed & (jnp.abs(right - left) <= 1e-8 * jnp.maximum(1, abs(best)))
        return best, score, closed

    def iteration(i, state):
        a, value, closed = state
        g, h = d1(a), d2(a)
        move = ~closed & (g != 0) & jnp.isfinite(g) & jnp.isfinite(h)
        # Always offer the first Newton step, including for a very faint Gaussian.
        move &= (i == 0) | ~small_slope(g, -h, 1e-6)
        return update_if(move, lambda: advance(a, value, g, h, move), state)

    a_hat, value, closed = jax.lax.fori_loop(
        0, n_newton, iteration, (zero, null, jnp.asarray(False))
    )
    g, curvature = d1(a_hat), -d2(a_hat)
    flat_null = (a_hat == 0) & (g == 0) & (curvature == 0)
    two_delta = 2 * (value - null)
    finite = jnp.isfinite(a_hat) & jnp.isfinite(two_delta)
    resolved = finite & jnp.isfinite(g) & jnp.isfinite(curvature)
    # The score can resolve to roundoff while further steps still refine a_hat.
    tolerance = jnp.maximum(1e-6, jnp.sqrt(roundoff(value)))
    resolved &= small_slope(g, curvature, tolerance) | closed | flat_null
    # Apply the subtraction tolerance while both objective values are available.
    rounding = 2 * roundoff(jnp.maximum(jnp.abs(null), jnp.abs(value)))
    rounded_loss = resolved & (two_delta < 0) & (two_delta >= -rounding)
    two_delta = jnp.where(rounded_loss, 0, two_delta)
    return a_hat, two_delta, curvature, resolved


def profile_amplitude(loglike_of_prediction, base, template, n_newton: int = 8):
    """Maximize ``a -> loglike(base + a * template)`` over the amplitude ``a``.

    A fixed number of safeguarded iterations from zero supports jit and vmap.
    A Gaussian likelihood with fixed variance is quadratic in amplitude and one
    Newton step is exact. Huber and domain-limited likelihoods such as Poisson
    can require backtracking or bracketing. The Laplace builder is unsupported.
    A heavy-tailed likelihood such as Student-t can have several amplitude peaks.
    The bracket may miss the global maximum and the profile may report an
    unresolved estimate or a local maximum.

    Masked elements of the base and template may be NaN. They contribute
    nothing to the objective or its forward-mode derivatives.

    Args:
        loglike_of_prediction: ``prediction -> scalar log-likelihood``, normally a
            builder from :mod:`photomancy.infer.likelihoods`.
        base: Prediction of the model without the source.
        template: Image of the source at unit amplitude, broadcastable with ``base``.
        n_newton: Number of safeguarded amplitude iterations.

    Returns:
        ``(a_hat, two_delta_loglike, curvature, resolved)``. The statistic is
        twice the gain over the null and curvature is the negative second
        derivative at the estimate. Its inverse square root is the formal
        standard error. A finite profile is resolved when the estimated error
        from the slope is within 1e-6 standard errors, the Newton decrement is
        within the objective's rounding floor at positive curvature, the bracket
        width is at most 1e-8 times max(1, abs(a_hat)), or the null is flat and
        stationary. A zero template resolves at zero amplitude with zero gain.

    Raises:
        ValueError: If the likelihood is the Laplace density.
    """
    _check_curvature(loglike_of_prediction)
    base = jnp.asarray(base)
    template = jnp.asarray(template)
    dtype = jnp.result_type(base, template, float)

    def objective(a):
        return loglike_of_prediction(base + a * template)

    return _profile(
        objective,
        n_newton,
        dtype,
        prod(jnp.broadcast_shapes(base.shape, template.shape)),
    )


def _summarize(a_hat, two_delta, curvature, resolved):
    """Return signed significance for a resolved profile and its curvature error.

    The subtraction in ``_profile`` absorbs losses of up to 16 eps per reduction
    level times the larger absolute log likelihood, with a unit scale floor.
    The factor of two comes from doubling the gain. Larger losses give NaN,
    as does an unresolved profile.
    """
    rounding = 16 * jnp.finfo(jnp.asarray(two_delta).dtype).eps
    gain = jnp.where(
        resolved & (two_delta >= -rounding), jnp.maximum(two_delta, 0), jnp.nan
    )
    z = jnp.sign(a_hat) * jnp.sqrt(gain)
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
    n_elements = sum(
        prod(jnp.broadcast_shapes(b.shape, jnp.shape(t)))
        for (_, b, _), t in zip(terms, images, strict=True)
    )
    return _summarize(*_profile(objective, n_newton, dtype, n_elements))


@eqx.filter_jit
def _run_map(terms, positions, n_newton, batch_size):
    a_hat, z, sigma = _mapped(
        lambda x: _map_terms(terms, n_newton, x), positions, batch_size
    )
    return DetectionMap(a_hat=a_hat, z=z, sigma=sigma)


def likelihood_ratio_map(
    terms: Sequence[tuple[Callable, Any, Callable]],
    positions,
    batch_size: int | None = 1,
    n_newton: int = 8,
) -> DetectionMap:
    """Profile a shared source amplitude at every trial position.

    At each position ``x`` the amplitude maximizes the summed log-likelihood
    ``sum_k loglike_k(base_k + a * template_k(x))`` over all terms, so several
    exposures constrain one amplitude. The statistic is
    ``Lambda = 2 [max_a logL(a) - logL(0)]``; the map reports ``a_hat``, the signed
    root ``z = sign(a_hat) sqrt(Lambda)`` and the formal standard error of ``a_hat``.
    With Gaussian likelihoods this is the whitened matched filter.

    Positions are processed ``batch_size`` of them at a time (one at a time by default,
    every position at once for ``None``), which sets the peak memory: each position
    renders one template per term. Masked or NaN pixels do not leak, as in
    :func:`profile_amplitude`.

    Args:
        terms: Sequence of ``(likelihood, base_prediction, template_fn)``. The
            likelihood is ``prediction -> scalar``; ``template_fn(position)`` returns
            the unit-amplitude source image for that term at one position. Make
            ``template_fn`` an ``eqx.Module`` so the arrays it holds are traced inputs
            instead of compiled constants.
        positions: Trial positions, an array (or PyTree of arrays) whose leading axis
            indexes positions; ``template_fn`` receives one slice.
        batch_size: Positions per batch: 1 (the default) runs them in sequence, ``None``
            runs every position at once (the fast choice on a GPU with the memory for
            it).
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


def _splits(fit) -> bool:
    """Whether every term of ``fit`` offers ``AbstractTerm.split``."""
    try:
        for term in fit.terms:
            eqx.filter_eval_shape(lambda m, t=term: t.split(m)[0], fit.model)
    except NotImplementedError:
        return False
    return True


def _linear_in(a, base, tangent):
    """``base + a * tangent`` on the inexact leaves; every other leaf is the base."""
    return jax.tree.map(
        lambda p, t: p + a * t if eqx.is_inexact_array(p) else p, base, tangent
    )


LINEARITY_RTOL = 1e-3
"""Largest departure from linearity ``detect`` accepts, relative to the signal."""


def _departure(actual, linear, signal):
    """Whether ``actual`` is within the linearity tolerance of ``linear``.

    The bound is ``LINEARITY_RTOL`` times the norm of the signal (the amplitude times
    the unit render) plus 100 machine epsilons of the prediction's norm, the rounding
    of a render that carries a bright base beside a faint signal. Elements where the
    linear prediction is not finite (masked pixels) do not count; a non-finite render
    where the linear prediction is finite is a departure.
    """
    err = norm = floor = 0.0
    for a, ell, sig in zip(actual, linear, signal, strict=True):
        for pa, pl, ps in zip(
            jax.tree.leaves(a), jax.tree.leaves(ell), jax.tree.leaves(sig), strict=True
        ):
            if not eqx.is_inexact_array(pa):
                continue
            d = jnp.where(jnp.isfinite(pl), pa - pl, 0.0)
            err = err + jnp.sum(d**2)
            norm = norm + jnp.sum(jnp.where(jnp.isfinite(ps), ps, 0.0) ** 2)
            floor = (
                floor
                + jnp.sum(jnp.where(jnp.isfinite(pa), pa, 0.0) ** 2)
                * (100.0 * jnp.finfo(pa.dtype).eps) ** 2
            )
    return err <= LINEARITY_RTOL**2 * norm + floor


def _profile_split(fit, x, amplitude, shape, n_newton):
    """Profile the amplitude from one render and its derivative.

    The model is rendered once, at ``a = 0`` with the tangent of the render along the
    amplitude, and every Newton step scores ``prediction + a * tangent`` pointwise, so
    the nested derivatives never reach the renderer. This is exact when the
    prediction is linear in the amplitude, so one more render at the amplitude found
    is compared with ``base + a_hat * tangent`` (:data:`LINEARITY_RTOL`). Returns the
    amplitude, root and error of :func:`_summarize` and whether the check held.
    """
    holder = []

    def render(a):
        model = fit.model_at({**x, amplitude: jnp.reshape(a, shape)})
        parts = [term.split(model) for term in fit.terms]
        dyn, stat = zip(
            *(eqx.partition(score, eqx.is_array) for _, score in parts), strict=True
        )
        holder.append(stat)
        return tuple(p for p, _ in parts), dyn

    zero = jnp.zeros((), jnp.float64)
    base, tangent, dyn = jax.jvp(
        render, (zero,), (jnp.ones((), jnp.float64),), has_aux=True
    )
    scores = tuple(eqx.combine(d, s) for d, s in zip(dyn, holder[-1], strict=True))

    def objective(a):
        total = sum(
            score(_linear_in(a, p, t))
            for score, p, t in zip(scores, base, tangent, strict=True)
        )
        return jnp.asarray(total, jnp.float64)

    n_elements = sum(p.size for p in jax.tree.leaves(base) if eqx.is_array(p))
    a_hat, z, sigma = _summarize(
        *_profile(objective, n_newton, jnp.float64, n_elements)
    )
    model = fit.model_at({**x, amplitude: jnp.reshape(a_hat, shape)})
    actual = tuple(term.split(model)[0] for term in fit.terms)
    linear = tuple(_linear_in(a_hat, p, t) for p, t in zip(base, tangent, strict=True))
    signal = tuple(
        jax.tree.map(lambda t: a_hat * t if eqx.is_inexact_array(t) else t, t)
        for t in tangent
    )
    return a_hat, z, sigma, _departure(actual, linear, signal)


def _profile_render(fit, x, amplitude, shape, n_newton):
    """Profile the amplitude by Newton steps through the fit's own likelihood."""

    def objective(a):
        full = {**x, amplitude: jnp.reshape(a, shape)}
        return jnp.asarray(fit.log_likelihood(full), jnp.float64)

    n_elements = sum(p.size for p in jax.tree.leaves(fit.terms) if eqx.is_array(p))
    return _summarize(*_profile(objective, n_newton, jnp.float64, n_elements))


@eqx.filter_jit
def _detect(fit, trials, amplitude, n_newton, batch_size, once):
    """The map and, per trial, whether the render-once path was linear enough."""
    shape = jnp.shape(resolve(fit.model, amplitude))

    def at(x):
        if once:
            return _profile_split(fit, x, amplitude, shape, n_newton)
        return (*_profile_render(fit, x, amplitude, shape, n_newton), jnp.asarray(True))

    a_hat, z, sigma, linear = _mapped(at, trials, batch_size)
    found = DetectionMap(a_hat=a_hat, z=z, sigma=sigma, assumptions=fit.assumptions)
    return found, linear


def detect(
    fit: Fit,
    trials: dict,
    *,
    amplitude: str,
    n_newton: int = 8,
    batch_size: int | None = 1,
) -> DetectionMap:
    """Matched-filter map of a source over trial values, from the fit's likelihood.

    At each trial the free path ``amplitude`` is profiled: ``a`` maximizes
    ``fit.log_likelihood`` by safeguarded steps from ``a = 0`` with the other entries
    held at their trial values. The map reports ``a_hat``, the signed root
    ``z = sign(a_hat) sqrt(2 [ell(a_hat) - ell(0)])`` and the formal standard error
    ``1 / sqrt(curvature)``. The noise, the response and the detector come from the
    fit's terms, so there is no second likelihood, and the way coordinates reach the
    model is the fit's own (``Fit.model_at``: leaf laws, then blocks). Priors play no
    part. When the model is linear in ``a`` and the likelihood is Gaussian with fixed
    variance, this is the classical whitened matched filter ``z = t'Wr / sqrt(t'Wt)``.
    Several terms (exposures) constrain one shared amplitude.

    The profiler supports Gaussian, Huber and Poisson likelihoods. Resolution
    requires a finite profile with a small curvature-scaled slope, a closed
    bracket or a stationary flat null, as in :func:`profile_amplitude`. An
    unresolved trial has NaN significance. A heavy-tailed likelihood such as
    Student-t can have multiple amplitude peaks. Its bracket can miss the
    global maximum and report an unresolved trial or a local maximum.

    When every term offers :meth:`~photomancy.infer.terms.AbstractTerm.split`, each
    trial renders the model once, at ``a = 0`` with its tangent along the amplitude,
    and profiles ``prediction + a * tangent``; this needs the prediction to be linear
    in the amplitude. The result is checked: one more render at the amplitude found
    must agree with ``base + a_hat * tangent`` to within 1e-3 of the signal's norm
    (plus 100 machine epsilons of the prediction's; a non-finite render where the
    linear prediction is finite fails), else the trials are profiled again through
    ``fit.log_likelihood`` by Newton steps, which holds for any prediction at the cost
    of nested derivatives through the renderer, and the failing trials take that
    answer (one more compile and a Newton pass over every trial, only when a trial
    fails). A term without ``split`` always takes the
    Newton path, and so does a call inside a ``jax`` transformation, where the
    check cannot be read on the host.

    Args:
        fit: A :class:`~photomancy.infer.fit.Fit` whose model holds the source. At
            ``amplitude = 0`` the source must be absent (zero flux), so the statistic
            compares against the model without it.
        trials: Entry dictionary in the format of the fit's ``x``, one leading trial
            axis on every entry, with exactly the keys of ``fit.free`` except
            ``amplitude``.
        amplitude: The free leaf-law path that is profiled; its leaf has one element.
        n_newton: Safeguarded iterations per trial. One step is exact for a
            linear prediction with fixed-variance Gaussian noise. Poisson
            and Huber likelihoods can require several iterations.
        batch_size: Trials per batch, which sets peak memory: 1 (the default) runs them
            in sequence, ``None`` runs every trial at once.

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
    found, linear = _detect(fit, trials, amplitude, n_newton, batch_size, _splits(fit))
    if isinstance(linear, jax.core.Tracer):
        # inside a transformation the check cannot choose a path on the host
        return _detect(fit, trials, amplitude, n_newton, batch_size, False)[0]
    if not bool(jnp.all(linear)):
        # every trial goes through Newton, so the program has one shape whatever the
        # number of failing trials; only the failing ones are taken from it
        redo, _ = _detect(fit, trials, amplitude, n_newton, batch_size, False)
        found = eqx.tree_at(
            lambda f: (f.a_hat, f.z, f.sigma),
            found,
            tuple(
                jnp.where(linear, kept, new)
                for kept, new in (
                    (found.a_hat, redo.a_hat),
                    (found.z, redo.z),
                    (found.sigma, redo.sigma),
                )
            ),
        )
    return found


class LikelihoodEstimate(eqx.Module):
    """The maximum of a fit's likelihood and the covariance from its curvature.

    Three statements make an estimate regular, and each is a field:
    ``regular = finite & converged & positive_definite``.

    Attributes:
        value: The estimate, an entry dictionary keyed by ``fit.free`` (the format of
            the fit's ``x``).
        cov: The formal covariance, the inverse of the observed information
            (``Fit.information(kind="observed", include_prior=False)`` at the estimate),
            shape ``(D, D)`` with rows in the order of
            :func:`photomancy.infer.posterior.ravel_samples` (the entries' leaves in
            ``jax.tree.leaves`` order, each row-major). It is a likelihood-only
            statement and says nothing about the prior. Where ``positive_definite`` is
            false the diagonal can be negative and the widths in :attr:`sigma` are NaN
            there: the point is not a maximum and the NaN says so. NaN throughout when
            ``finite`` is false.
        corr: The correlation matrix ``cov_ij / sqrt(cov_ii cov_jj)``; meaningful where
            ``positive_definite`` is true.
        regular: True when the estimate is a regular maximum: finite, converged and
            positive definite. False at a saddle, on a ridge (a direction the data do
            not constrain), off the peak, against the edge of a leaf's support (a
            bounded law whose maximum lies outside it), or after a run that did not
            converge. It does not require ``stopped_early``: the Newton polish can
            finish a run that hit its step cap.
        converged: True when the estimate is stationary where the data constrain it:
            ``newton_decrement <= max_decrement``, the scaled gradient has no
            component on a direction below the eigenvalue floor larger than 1e-3, and
            the final log likelihood is not below the start's or below the best the
            optimizer's trace reached by more than the stall tolerance (1e-7 nat, or
            1e-10 of the value when larger: one sum in two orders differs by rounding).
        positive_definite: True when the smallest eigenvalue of the correlation-form
            information :math:`C = D^{-1} A D^{-1}` exceeds 1e-8, with ``A`` the
            observed information and ``D = sqrt(|diag A|)``. The test is unit free:
            rescaling a coordinate (arcsec to radians, a flux by 1e-4) changes ``A``
            but not ``C``.
        finite: True when the log likelihood, the gradient, the information and the
            value are all finite.
        min_eigenvalue: The smallest eigenvalue of ``C`` (unit free; at most 1 minus
            the largest correlation, so below the floor for a ridge, negative at a
            saddle).
        grad_norm: The Euclidean norm of the scaled gradient ``D^-1 g`` of the log
            likelihood at the estimate (unit free; zero at a stationary point).
        newton_decrement: ``0.5 g' cov g`` over the directions of ``C`` above the
            floor, the nats the estimate still lies below the peak of the Gaussian
            that ``cov`` describes. Unit free, and finite beside a flat direction,
            which it leaves out (the projection of the gradient on those directions
            is part of ``converged``).
        log_likelihood: The log likelihood at the estimate.
        n_steps: L-BFGS steps run.
        stopped_early: Whether the optimizer's stop rule ended the run before its step
            cap (loss stalled and scaled gradient small); false for a run without a
            scale (a term with no Fisher root) and after a non-finite loss.
        n_newton: Newton polishing steps that moved the estimate.
        assumptions: The fit's ``Fit.assumptions``, copied because the estimate does
            not hold the fit. An estimate is comparable with another only under the
            same assumptions.
    """

    value: dict
    cov: jax.Array
    corr: jax.Array
    regular: jax.Array
    converged: jax.Array
    positive_definite: jax.Array
    finite: jax.Array
    min_eigenvalue: jax.Array
    grad_norm: jax.Array
    newton_decrement: jax.Array
    log_likelihood: jax.Array
    n_steps: jax.Array
    stopped_early: jax.Array
    n_newton: jax.Array
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


_EIGENVALUE_FLOOR = 1e-8
"""Smallest correlation-form eigenvalue of a direction the data constrain."""

_FLAT_GRADIENT = 1e-3
"""Largest scaled gradient along the directions below the floor of a converged point."""


def _scaling(info):
    """``D = sqrt(|diag A|)`` with a zero entry replaced by one."""
    scale = jnp.sqrt(jnp.abs(jnp.diag(info)))
    return jnp.where(scale > 0.0, scale, 1.0)


def _correlation_inverse(info):
    """Invert a symmetric matrix in correlation form and classify it.

    Scaling by the square roots of the absolute diagonal, ``C = D^-1 A D^-1``, takes a
    matrix whose entries span many orders of magnitude (condition number 4e12 on a
    full-scene fit) to one near unity (3.7 there); the inverse is
    ``D^-1 C^-1 D^-1``. A congruence by a positive diagonal keeps the signature, so
    the sign of the eigenvalues of ``C`` is that of ``A`` without the scale's rounding,
    and the eigenvalues of ``C`` do not depend on the units of the coordinates.

    Returns:
        ``(cov, corr, positive_definite, min_eigenvalue)``: the inverse, its
        correlation matrix, whether the smallest eigenvalue of ``C`` is above 1e-8, and
        that eigenvalue.
    """
    scale = _scaling(info)
    scaled = info / jnp.outer(scale, scale)
    inverse = jnp.linalg.inv(scaled)
    cov = inverse / jnp.outer(scale, scale)
    cov = 0.5 * (cov + cov.T)
    root = jnp.sqrt(jnp.abs(jnp.diag(inverse)))
    root = jnp.where(root > 0.0, root, 1.0)
    corr = inverse / jnp.outer(root, root)
    smallest = jnp.linalg.eigvalsh(scaled)[0]
    return cov, 0.5 * (corr + corr.T), smallest > _EIGENVALUE_FLOOR, smallest


def _stationarity(info, grad):
    """Scaled gradient, Newton decrement and the gradient along the flat directions.

    With ``C = D^-1 A D^-1`` and ``g_s = D^-1 g``, the decrement is
    ``0.5 sum (v_i . g_s)**2 / lambda_i`` over the eigenpairs of ``C`` above the floor
    (the pseudo-inverse form: finite beside a flat direction), and the projection of
    ``g_s`` on the eigenvectors below the floor is returned as a norm, so a gradient
    along a direction the decrement leaves out is still judged.

    Returns:
        ``(norm of g_s, decrement, norm of the flat projection)``.
    """
    scale = _scaling(info)
    values, vectors = jnp.linalg.eigh(info / jnp.outer(scale, scale))
    scaled = grad / scale
    along = vectors.T @ scaled
    above = values > _EIGENVALUE_FLOOR
    safe = jnp.where(above, values, 1.0)
    decrement = 0.5 * jnp.sum(jnp.where(above, along**2 / safe, 0.0))
    flat = jnp.linalg.norm(jnp.where(above, 0.0, along))
    return jnp.linalg.norm(scaled), decrement, flat


class _LikelihoodInReal(eqx.Module):
    """``u -> fit.log_likelihood`` at the stored values of the unconstrained ``u``.

    No prior and no volume term: the likelihood does not depend on the coordinates,
    so its maximum is the same in the support coordinates ``u`` as in the stored ones.

    Attributes:
        fit: The fit whose likelihood is maximized.
    """

    fit: Fit

    def __call__(self, u):
        x, _ = self.fit.from_real(u)
        return jnp.asarray(self.fit.log_likelihood(x), jnp.float64)


_BACKTRACKS = 8
"""Halvings of a Newton step tried before the step is refused."""


@eqx.filter_jit
def _newton(fit, u, scale, n_newton, batch_size):
    """Newton steps on the likelihood in the support coordinates, where they help.

    A step is taken only where the (scaled) information is positive definite and the
    step raises the likelihood, so it can neither follow a saddle nor leave the
    support (the coordinates are unconstrained). A full step that does not raise the
    likelihood (curvature at the start says little about a point far from the peak) is
    halved up to ``_BACKTRACKS`` times. A trial that rounds to the same bit pattern as
    the current vector ends backtracking because smaller updates round there too.
    A refused step ends polishing because another step at the same point would make
    the same proposal.

    Args:
        fit: The fit whose likelihood is maximized.
        u: Initial entries in the unconstrained coordinates.
        scale: Per-entry widths used to scale the Newton solve.
        n_newton: Maximum number of Newton steps.
        batch_size: Hessian columns per batch, or ``None`` for all at once.

    Returns:
        The final entries, the number of accepted steps, and the last Hessian in
        unconstrained coordinates. After a refusal the Hessian is at the final point.
    """
    flat, unravel = jax.flatten_util.ravel_pytree(u)
    width = jax.flatten_util.ravel_pytree(scale)[0]
    objective = _LikelihoodInReal(fit)

    def negative(theta):
        return -objective(unravel(theta))

    grad = jax.grad(negative)

    def attempt(theta):
        info = observed_hessian(negative, theta, batch_size=batch_size)
        scaled = info * jnp.outer(width, width)
        ok = jnp.all(jnp.linalg.eigvalsh(scaled) > 0.0)
        safe = jnp.where(ok, scaled, jnp.eye(theta.size))
        direction = width * jnp.linalg.solve(safe, width * grad(theta))
        before = negative(theta)
        theta_bits = jax.lax.bitcast_convert_type(theta, jnp.uint8)

        def trial(fraction, count):
            candidate = theta - fraction * direction
            candidate_bits = jax.lax.bitcast_convert_type(candidate, jnp.uint8)
            distinct = jnp.any(candidate_bits != theta_bits)
            value = jax.lax.cond(distinct, negative, lambda _: before, candidate)
            return fraction, candidate, value, distinct, count

        def halve(carry):
            fraction, _, _, _, count = carry
            return trial(0.5 * fraction, count + 1)

        def pending(carry):
            _, _, value, distinct, count = carry
            return distinct & (count < _BACKTRACKS) & ~(value < before)

        _, candidate, value, _, _ = jax.lax.while_loop(pending, halve, trial(1.0, 0))
        moved = ok & (value < before)
        return jnp.where(moved, candidate, theta), info, ~moved

    def step(carry, _):
        theta, _, refused = carry
        carry = jax.lax.cond(refused, lambda _: carry, attempt, theta)
        return carry, ~carry[2]

    empty = jnp.zeros((flat.size, flat.size), flat.dtype)
    (theta, info, _), moved = jax.lax.scan(
        step, (flat, empty, False), None, length=n_newton
    )
    return unravel(theta), jnp.sum(moved, dtype=jnp.int32), info


def _gradient(fit, x):
    """Gradient of the log likelihood at ``x``, flat in the stored coordinates."""
    flat, unravel = jax.flatten_util.ravel_pytree(x)
    return jax.grad(lambda theta: fit.log_likelihood(unravel(theta)))(flat)


def measure(
    fit: Fit,
    start: dict,
    *,
    n_steps: int = 100,
    n_newton: int = 4,
    batch_size: int | None = 1,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
    max_decrement: float = 1e-3,
) -> LikelihoodEstimate:
    """Maximize a fit's likelihood from ``start`` and take the covariance there.

    The likelihood alone is maximized, with no prior density and no volume term: the
    maximum does not depend on the coordinates, so the run moves in the support
    coordinates of each leaf law (``fit.to_real``: a bounded law is mapped onto the
    real line, so no step can leave the support). It uses the optimizer ``MAP`` uses
    (:func:`photomancy.infer.backends.staged.map_fit` with its default stage, L-BFGS),
    in coordinates scaled by the Fisher information at the start without the prior, and
    stops by the same rule: the loss has stalled and the scaled gradient norm is below
    its tolerance (a fit with a term that has no Fisher root runs unscaled and takes
    all ``n_steps``). It then polishes with up to ``n_newton`` Newton steps where the
    information is positive definite and the step (halved up to eight times) raises the
    likelihood, and returns the last iterate, not the best one. It is for a likelihood
    summary (a flux and a position per source), not a posterior: a posterior is ``MAP``
    or a sampler, and neither takes a flag that turns it into this.

    The covariance is the inverse of the observed information
    (:meth:`photomancy.infer.fit.Fit.information` with ``kind="observed"`` and
    ``include_prior=False``) at the estimate, in the stored coordinates, inverted in
    correlation form (scaled by the square roots of its diagonal first, so the scaled
    matrix decides without rounding whether it is positive definite). It is the formal
    covariance of the maximum-likelihood estimate and says nothing about the prior.

    The result says whether it is a maximum, in three parts that
    :class:`LikelihoodEstimate` reports separately: ``finite``, ``converged`` and
    ``positive_definite``; ``regular`` is their conjunction. All three are unit free.
    Positive definiteness is the smallest eigenvalue of the correlation-form
    information above 1e-8, a floor on numerical rank that does not move when a
    coordinate is rescaled. It certifies that the matrix is invertible and has no
    negative curvature, not that the data constrain every combination of parameters:
    a source whose amplitude runs to a value far above its width can sit at a
    correlation-form eigenvalue of 1e-3 and pass. Judge that with the source's own
    block (:func:`curvature_block`, scaled by its diagonal). Convergence is a Newton
    decrement at most ``max_decrement`` (1e-3 nat, about 0.045 formal widths) together
    with a scaled gradient under 1e-3 along any flat direction and a final log
    likelihood that is neither below the start's nor below the best the optimizer's
    trace reached by more than its stall tolerance (1e-7 nat, or 1e-10 of the value).
    A run that ends non-finite has ``finite`` false and NaN widths. A run that walked
    along a ridge and was ended by its step cap is positive definite and not
    converged; ``regular`` is false.

    The amplitude of a source that may be faint needs a law with signed support at this
    stage: ``Fit.with_priors`` with an unbounded law (a wide ``Normal``; its width is
    irrelevant because no density is read) lets a source of amplitude 0.3 plus or
    minus 1 return a finite summary. A positive law puts the maximum of such a source
    on the edge, where the stored-coordinate gradient is not zero: the decrement
    exposes it (``converged=False``) once the data pull past the edge by more than
    ``max_decrement`` nats (a measured amplitude below minus 0.045 formal errors at the
    default); a smaller pull is within the noise and is reported as measured. The
    Fisher matrix (``kind="fisher"``) cannot make these statements.

    Size. The observed information over every free scalar is the cost: about 20 s and
    7 GB for ten parameters on a full-scene fit. ``measure`` refuses with
    :class:`hwoutils.memory.MemoryLimit` beyond ``max_bytes`` before it optimizes,
    and is for tens of free scalars, not free pixels. The optimization loop runs on
    the host, so ``measure`` is not itself compiled or vmapped.

    Needs ``jax_enable_x64``.

    Args:
        fit: A :class:`~photomancy.infer.fit.Fit`.
        start: Entry dictionary, the format of the fit's ``x`` for one draw, with
            exactly the keys of ``fit.free``, inside every leaf's support.
        n_steps: L-BFGS iterations at the most (the default stage's stop rule can end
            the run earlier).
        n_newton: Newton polishing steps.
        batch_size: Hessian columns per batch (``None`` for all at once); the default
            holds one gradient's intermediates.
        max_bytes: Memory limit of the closed-form estimate, as in ``Fit.information``.
        max_decrement: Largest Newton decrement, in nats, of a converged estimate.

    Returns:
        A :class:`LikelihoodEstimate`.

    Raises:
        ValueError: If the keys of ``start`` are not the free paths.
        RuntimeError: If ``jax_enable_x64`` is off.
        MemoryLimit: If the information matrix does not fit ``max_bytes``.
    """
    _check_entries(start, set(fit.free))
    start = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), start)
    flat, _ = jax.flatten_util.ravel_pytree(start)
    _require_float64(flat)
    _admit_information(
        flat.size, flat.size * 8, 2 * flat.size * flat.size * 8, batch_size, max_bytes
    )
    u = fit.to_real(start)
    scale = information_scale(fit, u, include_prior=False)
    stage = default_stage(n_steps, optax.lbfgs(), scale)
    labels = jax.tree.map(lambda _: "all", u)
    result = map_fit(_LikelihoodInReal(fit), u, labels, {}, (stage,))
    trace = result.stages[0]
    u = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), result.params)
    width = flat.size
    nan = jnp.full((width, width), jnp.nan)
    if not bool(jnp.all(jnp.isfinite(jax.flatten_util.ravel_pytree(u)[0]))):
        # a run that ended off the real line: nothing to differentiate there
        x = jax.tree.map(lambda a: jnp.full_like(a, jnp.nan), start)
        return LikelihoodEstimate(
            value=x,
            cov=nan,
            corr=nan,
            regular=jnp.asarray(False),
            converged=jnp.asarray(False),
            positive_definite=jnp.asarray(False),
            finite=jnp.asarray(False),
            min_eigenvalue=jnp.asarray(jnp.nan),
            grad_norm=jnp.asarray(jnp.nan),
            newton_decrement=jnp.asarray(jnp.nan),
            log_likelihood=jnp.asarray(jnp.nan),
            n_steps=jnp.asarray(trace.n_steps, jnp.int32),
            stopped_early=jnp.asarray(False),
            n_newton=jnp.asarray(0, jnp.int32),
            assumptions=fit.assumptions,
        )
    moved = jnp.asarray(0, jnp.int32)
    info = None
    if n_newton:
        unit = jax.tree.map(jnp.ones_like, u)
        u, moved, last_info = _newton(
            fit, u, unit if scale is None else scale, n_newton, batch_size
        )
        # Real-support leaves and nonperiodic blocks pass through from_real unchanged.
        identity = type(fit) is Fit and all(
            entry.bounds == "real"
            if isinstance(entry, AbstractDistribution)
            else all(period is None for period in entry.periods)
            for entry in fit.free.values()
        )
        if identity and int(moved) < n_newton:
            info = last_info
    x, _ = fit.from_real(u)
    if info is None:
        info = fit.information(
            x,
            kind="observed",
            include_prior=False,
            batch_size=batch_size,
            max_bytes=max_bytes,
        )
    cov, corr, positive, min_eig = _correlation_inverse(info)
    grad = _gradient(fit, x)
    grad_norm, decrement, flat_gradient = _stationarity(info, grad)
    likelihood = jnp.asarray(fit.log_likelihood(x), jnp.float64)
    finite = (
        jnp.all(jnp.isfinite(jax.flatten_util.ravel_pytree(x)[0]))
        & jnp.isfinite(likelihood)
        & jnp.all(jnp.isfinite(grad))
        & jnp.all(jnp.isfinite(info))
    )
    # the trace holds the loss before each step, so its first entry is the start
    reached = -jnp.min(jnp.asarray(trace.loss)) if trace.n_steps else -jnp.inf
    # the loss inside the optimizer and this evaluation sum the pixels in different
    # orders, so they agree to about 1e-12 of the value (1.3e-7 nat at 6.5e4)
    below = likelihood < reached - jnp.maximum(STOP_MIN_DELTA, 1e-10 * jnp.abs(reached))
    converged = (
        (decrement <= max_decrement)
        & (flat_gradient < _FLAT_GRADIENT)
        & ~below
        & (not trace.failed)
    )
    return LikelihoodEstimate(
        value=x,
        cov=jnp.where(finite, cov, jnp.nan),
        corr=jnp.where(finite, corr, jnp.nan),
        regular=finite & converged & positive,
        converged=converged,
        positive_definite=positive,
        finite=finite,
        min_eigenvalue=min_eig,
        grad_norm=grad_norm,
        newton_decrement=decrement,
        log_likelihood=likelihood,
        n_steps=jnp.asarray(trace.n_steps, jnp.int32),
        stopped_early=jnp.asarray(trace.stopped_early),
        n_newton=moved,
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
