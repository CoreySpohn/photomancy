"""Retrieval targets for atmosphere fitting: fit-leaf selectors + default priors.

A retrieval "target" is a ``(fit-leaf selector, default prior)`` pair. The fixed-T
abundance target (``abundance_fit_leaves`` + ``default_abundance_prior``) is the first;
temperature / TP-structure targets slot in alongside as skyscapes grows its T-varying
forward (the ``for_retrieval`` model is inert in temperature, so abundance is the only
fittable target there).
"""

import jax.numpy as jnp

from photomancy.priors import Uniform


def abundance_fit_leaves(pm):
    """The per-species ``log_mmr`` leaves (in species order) -- the abundance target."""
    return [s.profile.log_mmr for s in pm.species]


def default_abundance_prior(
    pm, *, fit_leaves=abundance_fit_leaves, log_mmr_range=(-12.0, 0.0)
):
    """An uninformative ``Uniform`` over the fitted log10 mass-mixing-ratio leaves.

    Every abundance scalar gets ``Uniform(log_mmr_range)``, so the prior aligns to
    the fit's z automatically (no per-leaf ordering needed). This is the honest
    blind-retrieval default -- it does not assume you know the abundance.

    Args:
        pm: The atmosphere model whose fitted leaves set the prior dimension.
        fit_leaves: ``pm -> list[leaf]`` selecting the abundance leaves
            (default :func:`abundance_fit_leaves`).
        log_mmr_range: ``(low, high)`` bounds on each ``log10`` mass-mixing-ratio.

    Returns:
        A :class:`~photomancy.priors.Uniform` over the raveled abundance leaves.
    """
    ndim = sum(int(jnp.size(leaf)) for leaf in fit_leaves(pm))
    low, high = log_mmr_range
    return Uniform(low=jnp.full((ndim,), low), high=jnp.full((ndim,), high))
