"""The one posterior type, built only by :func:`photomancy.infer.run`.

A :class:`Posterior` is a weighted cloud of draws of a :class:`Fit`
with the fit it targets: ``samples`` keyed by the fit's free paths, normalized log
weights, the fit's own log density at each draw, and the evidence and its standard error
where the backend has them. Every backend fills the same fields, so a consumer reads one
layout whichever method ran.

Layout. The draw axis is flat and in stored order (chains are concatenated, never an
axis). Leading axes are batch axes: a stack of worlds is the same type with a longer
prefix, produced by mapping ``run`` over a stacked fit with ``eqx.filter_vmap``. Shape
checks read only the trailing axes of ``log_weights`` and the batch prefix they define.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from priorities import Mixture, MultivariateNormal

from photomancy.infer.fit import Fit

__all__ = ["Posterior", "cluster_to_mixture", "ravel_samples"]

_KINDS = ("monte_carlo", "quadrature")


class Posterior(eqx.Module):
    """A weighted cloud of draws of a ``Fit`` and the density it targets.

    Attributes:
        samples: Entry dict with exactly the keys of ``fit.free``. A leaf-law entry is
            ``(*batch, N, *leaf_shape)``; a block entry is a dict of coordinate arrays
            ``(*batch, N, R)``. The draw axis is flat, in the backend's stored order.
        log_weights: Normalized log weights ``(*batch, N)``, float64
            (``logsumexp`` over ``N`` is zero). A slice that has no support is NaN.
        log_target: ``fit.log_density`` at the draws ``(*batch, N)``, float64: the
            normalized prior plus terms against Lebesgue measure on the stored
            coordinates, exactly ``-inf`` off the prior's support. It is never a
            proposal density.
        log_evidence: Log marginal likelihood ``(*batch,)``, float64, NaN when absent
            (the MCMC backends have none, and no estimate is invented).
        log_evidence_se: Standard error of ``log_evidence`` ``(*batch,)``, float64, NaN
            when the backend has none.
        fit: The ``Fit`` whose density the weights target.
        info: The backend's diagnostics module (chain index, divergence flags, thin
            factor, ladder, stage traces), or ``None``. Chain structure lives here and
            never in ``samples``.
        kind: ``"monte_carlo"`` (a random estimator) or ``"quadrature"`` (a
            positive-weight rule whose error is judged by refinement).
    """

    samples: dict
    log_weights: jax.Array
    log_target: jax.Array
    log_evidence: jax.Array
    log_evidence_se: jax.Array
    fit: Fit
    info: eqx.Module | None
    kind: str = eqx.field(static=True)

    def __check_init__(self):
        """Refuse a malformed posterior on the host (trailing axes only).

        Raises:
            ValueError: For a wrong ``kind``, key set, dtype or shape.
        """
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if set(self.samples) != set(self.fit.free):
            raise ValueError(
                "samples must have exactly the keys of fit.free: "
                f"{sorted(self.samples)} against {sorted(self.fit.free)}"
            )
        for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
            dtype = jnp.asarray(getattr(self, name)).dtype
            if dtype != jnp.float64:
                raise ValueError(f"{name} must be float64, got {dtype}")
        shape = jnp.shape(self.log_weights)
        if not shape:
            raise ValueError("log_weights needs a draw axis")
        batch, n = shape[:-1], shape[-1]
        if jnp.shape(self.log_target) != shape:
            raise ValueError(
                f"log_target has shape {jnp.shape(self.log_target)}, "
                f"log_weights {shape}"
            )
        for name in ("log_evidence", "log_evidence_se"):
            if jnp.shape(getattr(self, name)) != batch:
                raise ValueError(
                    f"{name} has shape {jnp.shape(getattr(self, name))}, "
                    f"expected the batch shape {batch}"
                )
        for key, entry in self.samples.items():
            for leaf in jax.tree.leaves(entry):
                if jnp.ndim(leaf) < len(batch) + 1 or jnp.shape(leaf)[
                    : len(batch) + 1
                ] != (*batch, n):
                    raise ValueError(
                        f"samples[{key!r}] has a leaf of shape {jnp.shape(leaf)}; "
                        f"expected leading axes {(*batch, n)} (batch, then draws)"
                    )


def ravel_samples(posterior, keys=None):
    """Flatten each draw to a vector: ``(*batch, N, D)`` and an unravel function.

    The columns are the entries' leaves in the deterministic order of
    ``jax.tree.leaves`` (sorted keys, then each leaf row-major). ``unravel`` maps an
    array ``(*lead, D)`` back to the entry dict of those keys, leaves shaped
    ``(*lead, *leaf_shape)``.

    Args:
        posterior: A :class:`Posterior`.
        keys: Entries to flatten, default all of ``posterior.samples``.
    """
    samples = (
        posterior.samples
        if keys is None
        else {key: posterior.samples[key] for key in keys}
    )
    batch = jnp.ndim(posterior.log_weights) - 1
    leaves, treedef = jax.tree.flatten(samples)
    trailing = [jnp.shape(leaf)[batch + 1 :] for leaf in leaves]
    sizes = [math.prod(t) for t in trailing]
    lead = jnp.shape(posterior.log_weights)
    flat = jnp.concatenate(
        [
            jnp.reshape(leaf, (*lead, size))
            for leaf, size in zip(leaves, sizes, strict=True)
        ],
        axis=-1,
    )
    offsets = [sum(sizes[:i]) for i in range(len(sizes) + 1)]

    def unravel(vector):
        parts = [
            jnp.reshape(vector[..., a:b], (*jnp.shape(vector)[:-1], *t))
            for a, b, t in zip(offsets[:-1], offsets[1:], trailing, strict=True)
        ]
        return jax.tree.unflatten(treedef, parts)

    return flat, unravel


def cluster_to_mixture(posterior, k, key, *, keys=None, iters=25, cov_floor=1e-6):
    """Cluster a posterior into a ``k``-mode Gaussian ``priorities.Mixture``.

    Weighted k-means (fixed iterations) assigns the draws, flattened by
    :func:`ravel_samples`, to ``k`` clusters. Each mode is the cluster's weighted mean
    and covariance and its weight is the cluster's mass, so the mixture follows the
    posterior's modes. The result is a distribution over the flat vector: use it as the
    law of one free leaf of shape ``(D,)`` in the next ``Fit``. It has no inverse CDF,
    so it serves the gradient and sample backends and not nested sampling.

    Args:
        posterior: An unbatched :class:`Posterior` (one slice; map over worlds).
        k: Number of modes.
        key: PRNG key for the k-means start.
        keys: Entries to cluster, default all.
        iters: k-means iterations.
        cov_floor: Diagonal added to every covariance.

    Returns:
        A :class:`priorities.Mixture` of :class:`priorities.MultivariateNormal`.
    """
    if jnp.ndim(posterior.log_weights) != 1:
        raise ValueError("cluster_to_mixture takes one slice; map it over a batch")
    x, _ = ravel_samples(posterior, keys)
    n, d = x.shape
    w = jax.nn.softmax(posterior.log_weights)
    centers = x[jax.random.choice(key, n, (k,), replace=False)]

    def assign(centers):
        sq = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        return jax.nn.one_hot(jnp.argmin(sq, axis=1), k) * w[:, None]

    def step(centers, _):
        mass_w = assign(centers)
        mass = jnp.sum(mass_w, axis=0)
        new = (mass_w.T @ x) / jnp.maximum(mass[:, None], 1e-30)
        return jnp.where(mass[:, None] > 0, new, centers), None

    centers, _ = jax.lax.scan(step, centers, None, length=iters)
    mass_w = assign(centers)
    mass = jnp.sum(mass_w, axis=0)
    means = (mass_w.T @ x) / jnp.maximum(mass[:, None], 1e-30)

    def cov(mu, wcol):
        diff = x - mu
        return (diff * wcol[:, None]).T @ diff / jnp.maximum(jnp.sum(wcol), 1e-30)

    covs = jax.vmap(cov)(means, mass_w.T) + cov_floor * jnp.eye(d)[None]
    log_mass = jnp.log(jnp.maximum(mass, 1e-30))
    return Mixture(log_mass, MultivariateNormal(means, jnp.linalg.cholesky(covs)))
