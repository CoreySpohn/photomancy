"""The coordinate scale of a fit, one private helper for the optimizers.

An imaging fit frees parameters whose information differs by many orders of magnitude
(a star flux scale known to a few parts per million beside epochs known to days), and a
quasi-Newton step in such coordinates crawls along the stiff ones. ``information_scale``
returns the width the data (and, optionally, the prior) give each unconstrained
coordinate, so an optimizer can move in coordinates of unit scale. It is a helper and
not a stage: ``MAP`` and ``measure`` call it, and a later sampler can take it as the
start of its mass matrix.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from photomancy.infer.fit import Fit, _Float32OwnerError

__all__ = ["SCALE_SPAN", "information_scale"]

SCALE_SPAN = 1e12
"""Largest ratio of one coordinate's scale to the smallest scale of the fit.

A coordinate the data and the prior leave with no curvature has an infinite scale; it
takes ``SCALE_SPAN`` times the smallest scale instead of stretching the optimizer's
coordinates without bound. The real entries of a full-scene fit span ten orders of
magnitude (the star flux scale against the epochs), so the span sits above that.
"""


@eqx.filter_jit
def _entries(fit, u, include_prior):
    """Squared tangent norm of the Fisher root along each coordinate, plus the prior."""
    flat, unravel = ravel_pytree(u)

    def root(theta):
        x, _ = fit.from_real(unravel(theta))
        return fit._root_vector(x, flat.dtype)

    def tangent_norm(direction):
        return jnp.sum(jax.jvp(root, (flat,), (direction,))[1] ** 2)

    eye = jnp.eye(flat.size, dtype=flat.dtype)
    entries = jax.lax.map(tangent_norm, eye)
    if include_prior:

        def prior(theta):
            x, volume = fit.from_real(unravel(theta))
            return fit.log_prior(x) + volume

        def curvature(direction):
            def slope(theta):
                return jax.jvp(prior, (theta,), (direction,))[1]

            return jax.jvp(slope, (flat,), (direction,))[1]

        entries = entries + jnp.abs(jax.lax.map(curvature, eye))
    return jnp.where(jnp.isfinite(entries), entries, 0.0)


def information_scale(fit: Fit, u: dict, *, include_prior: bool, span=SCALE_SPAN):
    """The width of each unconstrained coordinate of ``fit`` at ``u``.

    Entry ``i`` of the information is the squared norm of the derivative of the terms'
    Fisher roots (:meth:`photomancy.infer.fit.AbstractTerm.fisher_root`) along the
    unconstrained coordinate ``u_i``, through ``from_real`` and ``model_at``, one
    forward-mode pass per coordinate. It is the diagonal of the Fisher information
    written in the coordinates an optimizer moves in, and it is not a diagonal read of
    :meth:`~photomancy.infer.fit.Fit.information`, which is written in the stored
    coordinates. With ``include_prior`` the absolute value of the second derivative of
    the log prior plus the log volume of the map along ``u_i`` is added, so a
    coordinate the data do not constrain takes the prior's scale; the absolute value is
    needed because that curvature is indefinite in general.

    The scale is ``1 / sqrt(entry)``, bounded above by ``span`` times the smallest
    scale (a coordinate with no curvature at all takes that bound) and set to one
    everywhere when no coordinate has any.

    Args:
        fit: The fit.
        u: One draw in the unconstrained entries (the format of ``fit.to_real``), at
            which the information is taken. Its dtype sets the dtype of the result.
        include_prior: Add the prior's curvature to the data's.
        span: Largest ratio of a scale to the smallest scale.

    Returns:
        The scale as an entry dict with the structure and shapes of ``u``, or ``None``
        when a term has no Fisher root. The optimizer then uses its unscaled stage.

    Raises:
        PrecisionError: If a forward owner is narrow, float64 is unavailable,
            or a term's Fisher root cannot be evaluated at the working precision.
    """
    model = fit._require_float64_forward("Fisher scale", fit.from_real(u)[0])
    return _scale_from_roots(fit, model, u, include_prior, span)


def _map_scale(fit, start):
    """Return the default MAP scale and any float32 forward-owner diagnostic."""
    try:
        x, _ = fit.from_real(start)
        model = fit._require_float64_forward("MAP Fisher scale", x)
    except _Float32OwnerError as err:
        return None, str(err)
    return _scale_from_roots(fit, model, start, True, SCALE_SPAN), None


def _scale_from_roots(fit, model, u, include_prior, span):
    try:
        fit._root_length(model)
    except NotImplementedError:
        return None
    entries = _entries(fit, u, include_prior)
    scale = jnp.where(
        entries > 0.0, jax.lax.rsqrt(jnp.where(entries > 0.0, entries, 1.0)), jnp.inf
    )
    scale = jnp.minimum(scale, span * jnp.min(scale))
    scale = jnp.where(jnp.isfinite(scale), scale, 1.0)
    return ravel_pytree(u)[1](scale)
