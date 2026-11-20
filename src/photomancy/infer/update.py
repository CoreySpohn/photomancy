"""``update``: add or replace one term, and refit where the reweighting is refused."""

import equinox as eqx
import jax
import jax.numpy as jnp

from photomancy.infer.posterior import Posterior, UpdateInfo
from photomancy.infer.run import run

__all__ = ["update"]


def _dynamic(posterior):
    """The array fields of a posterior (everything but the fit, info and kind)."""
    return (
        posterior.samples,
        posterior.log_weights,
        posterior.log_target,
        posterior.log_evidence,
        posterior.log_evidence_se,
    )


def _signature(tree):
    return jax.tree.structure(tree), [
        (jnp.shape(leaf), jnp.result_type(leaf)) for leaf in jax.tree.leaves(tree)
    ]


def _filled(shape):
    """A placeholder for an info leaf: NaN floats, zero integers, false booleans."""
    if not isinstance(shape, jax.ShapeDtypeStruct):
        return shape
    nan = jnp.issubdtype(shape.dtype, jnp.floating)
    return jnp.full(shape.shape, jnp.nan if nan else 0, shape.dtype)


def update(
    posterior,
    term,
    backend,
    key,
    *,
    min_n_eff,
    max_khat,
    index=None,
    batch_size=None,
):
    """Move a posterior to one more (or one replaced) term, refitting where it must.

    The cheap move is :meth:`Posterior.append_term` (``index=None``) or
    :meth:`Posterior.replace_term` (``index`` given) with the gate arguments. The
    function reads only the result of that gate: a slice whose weights come back NaN
    is refitted by ``run(new_fit, backend, key)`` and every other slice keeps the
    reweighted cloud. There is no second criterion.

    The result carries ``info``, an :class:`~photomancy.infer.posterior.UpdateInfo`.
    ``info.reweight`` is the :class:`~photomancy.infer.posterior.ReweightInfo` of the
    gate, read before any refit; ``info.reweight.refused`` is true where the refit ran,
    which is also how a fresh NaN evidence is told from one inherited from an earlier
    refusal. ``info.refit`` is the backend's own info, equal to
    ``run(new_fit, backend, key).info`` where the refit ran. Elsewhere it has the same
    structure, filled (floats NaN, integers zero, booleans false), so both
    ``jax.lax.cond`` branches agree. Read ``info.refit.proposal_valid`` and the
    backend's k-hat there. The refit must return the cloud's own layout: the
    same draw count, sample shapes and dtypes and the same ``kind``; a backend that
    does not raises ``ValueError`` on the host when traced. Its draw count is the
    backend's, so build it with the posterior's size. The evidence of a refit follows
    the contract on :attr:`Posterior.log_evidence`.

    The choice is one ``jax.lax.cond``. For one slice it is lazy: the backend runs
    only on a refusal. Under ``eqx.filter_vmap`` over worlds the predicate is batched
    and both branches run for every world, the refit included, then each world takes
    its own; a stack of worlds pays one refit per call whether or not any world needs
    it. Call ``update`` once per world (``lax.map`` or a loop) when refits are rare.
    A backend that cannot be mapped (``MAP``, ``NestedSampling``) cannot be used under
    ``filter_vmap``. Takes one slice.

    Args:
        posterior: A :class:`~photomancy.infer.posterior.Posterior` of one slice.
        term: The new term, or the replacement when ``index`` is given.
        backend: The backend that refits the new fit on a refusal.
        key: PRNG key for the refit.
        min_n_eff: As in :meth:`Posterior.with_priors`: an absolute Kish count that
            the caller states, not a fraction of the draw count. The refit is not
            gated, so read ``kish_ess(new.log_weights)`` on the result, and size the
            backend so that its cloud clears the same number. For
            :func:`~photomancy.orbit.any_epoch` that is about 100 draws per unit of
            ``min_n_eff`` (``n_draws=2**16`` suits ``min_n_eff`` near 650). The
            expected Kish size of a refit is not known before it runs, so ``update``
            keeps one criterion and no retry.
        max_khat: As in :meth:`Posterior.with_priors`.
        index: Position of the term to replace in ``posterior.fit.terms``, or
            ``None`` to append.
        batch_size: Draws evaluated at once for the term and for the refit (a
            ``jax.lax.map`` chunk).

    Returns:
        A :class:`~photomancy.infer.posterior.Posterior` over the new fit, whose
        ``info`` is an :class:`~photomancy.infer.posterior.UpdateInfo`.

    Raises:
        ValueError: On the host, as :meth:`Posterior.append_term`, or for a refit whose
            layout differs from the cloud's.
    """
    if index is None:
        moved = posterior.append_term(
            term, min_n_eff=min_n_eff, max_khat=max_khat, batch_size=batch_size
        )
    else:
        moved = posterior.replace_term(
            index, term, min_n_eff=min_n_eff, max_khat=max_khat, batch_size=batch_size
        )
    refused = jnp.any(jnp.isnan(moved.log_weights))
    kept = _dynamic(moved)

    def refit():
        new = run(moved.fit, backend, key, batch_size=batch_size)
        if new.kind != moved.kind:
            raise ValueError(
                f"the refit backend returned kind {new.kind!r}, the posterior is "
                f"{moved.kind!r}"
            )
        fresh = _dynamic(new)
        if _signature(fresh) != _signature(kept):
            raise ValueError(
                "the refit backend must return the posterior's layout (draw count, "
                "sample shapes and dtypes); got "
                f"{_signature(fresh)[1]} against {_signature(kept)[1]}"
            )
        return fresh, new.info

    placeholder = jax.tree.map(_filled, eqx.filter_eval_shape(refit)[1])
    (samples, lw, target, evidence, evidence_se), refit_info = jax.lax.cond(
        refused, refit, lambda: (kept, placeholder)
    )
    return Posterior(
        samples=samples,
        log_weights=lw,
        log_target=target,
        log_evidence=evidence,
        log_evidence_se=evidence_se,
        fit=moved.fit,
        info=UpdateInfo(reweight=moved.info, refit=refit_info),
        kind=moved.kind,
    )
