"""Retrieval targets for atmosphere fitting: free paths and default priors.

A retrieval "target" is a set of model paths and a law for each. The fixed-T
abundance target (``abundance_paths`` + ``default_abundance_prior``) is the first;
temperature / TP-structure targets slot in alongside as skyscapes grows its T-varying
forward (the ``for_retrieval`` model is inert in temperature, so abundance is the only
fittable target there).
"""

import jax.numpy as jnp
from hwoutils.model_paths import resolve
from priorities import Uniform

__all__ = ["abundance_paths", "default_abundance_prior"]


def abundance_paths(pm):
    """The per-species ``log_mmr`` model paths, in species order (abundance target).

    Args:
        pm: The atmosphere model; its ``species`` sequence sets the number of paths.

    Returns:
        One model path string ``"species[k].profile.log_mmr"`` per species.
    """
    return tuple(f"species[{k}].profile.log_mmr" for k in range(len(pm.species)))


def default_abundance_prior(pm, *, paths=None, log_mmr_range=(-12.0, 0.0)):
    """An uninformative uniform law for each fitted log10 mass-mixing-ratio leaf.

    Every abundance scalar gets ``Uniform(log_mmr_range)`` over the shape of its leaf.
    This is the honest blind-retrieval default -- it does not assume you know the
    abundance.

    Args:
        pm: The atmosphere model whose leaves set the shapes of the laws.
        paths: Model paths of the abundance leaves (default :func:`abundance_paths`).
        log_mmr_range: ``(low, high)`` bounds on each ``log10`` mass-mixing-ratio.

    Returns:
        ``{path: Uniform}`` for the ``free`` field of a ``Fit`` over ``pm``.
    """
    paths = abundance_paths(pm) if paths is None else paths
    low, high = log_mmr_range
    free = {}
    for path in paths:
        shape = jnp.shape(resolve(pm, path))
        free[path] = Uniform(jnp.full(shape, low), jnp.full(shape, high))
    return free
