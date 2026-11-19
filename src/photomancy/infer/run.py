"""``run``: the one constructor of a :class:`~photomancy.infer.posterior.Posterior`."""

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from photomancy.infer.backends.base import AbstractBackend
from photomancy.infer.fit import Fit
from photomancy.infer.posterior import Posterior

__all__ = ["run"]


def _normalized(log_weights):
    """Normalized float64 log weights of one slice; NaN where there is no support."""
    log_weights = jnp.asarray(log_weights, jnp.float64)
    shifted = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
    peak = jnp.max(shifted, axis=-1, keepdims=True)
    no_support = ~jnp.isfinite(peak)
    safe = jnp.where(no_support, 0.0, shifted - peak)
    norm = safe - logsumexp(safe, axis=-1, keepdims=True)
    return jnp.where(no_support, jnp.nan, norm)


def run(fit: Fit, backend: AbstractBackend, key) -> Posterior:
    """Run ``backend`` on ``fit`` and return the :class:`Posterior`.

    The backend returns its native draws and unnormalized log weights; this function
    alone normalizes them, evaluates ``fit.log_density`` at the draws for
    ``log_target``, fixes the float64 fields and attaches the fit. It performs no hidden
    second pass: a backend with no evidence gives NaN. ``Fit`` already refuses to be
    built with ``jax_enable_x64`` off, so the float64 fields are guaranteed.

    The function is traceable. A stack of worlds is a stacked ``Fit`` mapped with
    ``eqx.filter_vmap``; the stack of posteriors is the same type with a leading batch
    axis. Backends built on ``lax`` loops (``NUTS``, ``MCLMC``, ``SMC``, ``Laplace``,
    ``Pathfinder``, ``Grid``) map; a ``while_loop`` runs to the slowest slice. ``MAP``
    (host-driven stages) and ``NestedSampling`` (jaxns) run one fit at a time.

    Args:
        fit: The fit to run.
        backend: An :class:`~photomancy.infer.backends.base.AbstractBackend`.
        key: PRNG key.

    Returns:
        A :class:`Posterior` over ``fit``.
    """
    draws = backend._run(fit, key)
    log_target = jnp.asarray(fit.log_density(draws.samples), jnp.float64)
    return Posterior(
        samples=draws.samples,
        log_weights=_normalized(draws.log_weights),
        log_target=log_target,
        log_evidence=jnp.asarray(draws.log_evidence, jnp.float64),
        log_evidence_se=jnp.asarray(draws.log_evidence_se, jnp.float64),
        fit=fit,
        info=draws.info,
        kind=draws.kind,
    )
