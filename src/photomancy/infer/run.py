"""``run``: the one constructor of a :class:`~photomancy.infer.posterior.Posterior`."""

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from photomancy.infer.backends.base import AbstractBackend
from photomancy.infer.fit import Fit
from photomancy.infer.posterior import Posterior

__all__ = ["run"]


def _normalized(log_weights):
    """Normalized float64 log weights of one slice.

    A slice with no support (every weight ``-inf``) or with any NaN weight is all NaN:
    a NaN target is a refusal, never a zero weight.
    """
    log_weights = jnp.asarray(log_weights, jnp.float64)
    peak = jnp.max(log_weights, axis=-1, keepdims=True)
    refused = jnp.any(jnp.isnan(log_weights), axis=-1, keepdims=True) | ~jnp.isfinite(
        peak
    )
    safe = jnp.where(refused, 0.0, log_weights - peak)
    norm = safe - logsumexp(safe, axis=-1, keepdims=True)
    return jnp.where(refused, jnp.nan, norm)


def run(
    fit: Fit, backend: AbstractBackend, key, *, batch_size: int | None = None
) -> Posterior:
    """Run ``backend`` on ``fit`` and return the :class:`Posterior`.

    The backend returns its native draws and unnormalized log weights; this function
    alone normalizes them, evaluates ``fit.log_density`` at the draws for
    ``log_target``, fixes the float64 fields and attaches the fit. It performs no hidden
    second pass: a backend with no evidence gives NaN. A finite ``log_evidence`` means
    every validity condition of its estimator that the backend can compute from its
    own run has held (for the importance backends, a Pareto k-hat of the weights at or
    below ``khat_threshold(n_draws)`` and a proposal that reports its weights valid,
    ``info.proposal_valid``; for ``any_epoch``, at least half the pool effective);
    otherwise the evidence and its standard error
    are NaN while the draws and weights are still returned. ``Fit`` already refuses to
    be built with ``jax_enable_x64`` off, so the float64 fields are guaranteed.

    The function is traceable. A stack of worlds is a stacked ``Fit`` mapped with
    ``eqx.filter_vmap``; the stack of posteriors is the same type with a leading batch
    axis. Backends built on ``lax`` loops (``NUTS``, ``MCLMC``, ``Laplace``,
    ``Pathfinder``, ``Grid``) map; a ``while_loop`` runs to the slowest slice. ``MAP``
    (host-driven stages) and ``NestedSampling`` (jaxns, a host loop with the fit closed
    over) cannot be mapped: they run one fit at a time.

    Args:
        fit: The fit to run.
        backend: An :class:`~photomancy.infer.backends.base.AbstractBackend`.
        key: PRNG key.
        batch_size: Chunk of draws evaluated at once when the density of a cloud is
            taken (a ``jax.lax.map`` batch, the same values to rounding), for the
            importance backends' weights and for ``log_target``. ``None`` evaluates
            every draw at once; give it when a term (an ``ImageTerm``) makes memory
            grow with the draw count. A sampler's own per-step density is not batched.

    Returns:
        A :class:`Posterior` over ``fit``.
    """
    draws = backend._run(fit, key, batch_size=batch_size)
    log_target = (
        fit.log_density(draws.samples, batch_size=batch_size)
        if draws.log_target is None
        else draws.log_target
    )
    log_target = jnp.asarray(log_target, jnp.float64)
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
