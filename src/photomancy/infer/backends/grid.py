"""Grid backend: positive-weight quadrature of the posterior on a product grid.

Every free degree of freedom gets an axis of nodes and positive weights. The draws are
the nodes of the product grid and the weight of a node is the product of its axis
weights times the fit's density there, so the sum of the weights is the evidence by
the quadrature rule. A rule with a negative weight (a Newton-Cotes rule of high order)
is excluded by construction: ``Grid`` refuses non-positive weights, so the weights stay
a valid probability vector and every operation on them keeps its meaning. The error is
judged by refining the axes, not by a standard error.
"""

import jax
import jax.numpy as jnp
import numpy as np

from photomancy.infer._precision import _as_wide
from photomancy.infer.backends._flat import draw_template
from photomancy.infer.backends.base import AbstractBackend, BackendDraws

__all__ = ["Grid", "gauss_legendre_rule", "midpoint_rule"]


def midpoint_rule(low, high, n):
    """Midpoint rule on ``[low, high]``: ``n`` equal cells, nodes at their centers.

    Returns:
        ``(nodes, weights)``, both ``(n,)``; the weights are the cell width.

    Args:
        low: Lower end of the interval.
        high: Upper end of the interval.
        n: Number of cells and nodes.
    """
    width = (high - low) / n
    nodes = low + width * (jnp.arange(n) + 0.5)
    return nodes, jnp.full(n, width)


def gauss_legendre_rule(low, high, n):
    """Gauss-Legendre rule on ``[low, high]`` with ``n`` nodes (positive weights).

    Returns:
        ``(nodes, weights)``, both ``(n,)``.

    Args:
        low: Lower end of the interval.
        high: Upper end of the interval.
        n: Number of nodes.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (high - low)
    return jnp.asarray(low + half * (x + 1.0)), jnp.asarray(half * w)


class Grid(AbstractBackend):
    """Quadrature on the product of per-axis rules, in the stored coordinates.

    Each free degree of freedom must be one scalar: a leaf law over a leaf of one
    element is keyed by its free path, a coordinate of a one-row block by
    ``"<key>/<coordinate>"``. ``axes`` maps each such name to ``(nodes, weights)`` (see
    :func:`midpoint_rule` and :func:`gauss_legendre_rule`) with every weight positive.
    The result has ``prod(len(nodes))`` draws in C order of the sorted axis names,
    weights ``quadrature weight * density`` and ``kind = "quadrature"``. The evidence is
    the sum of the weights; its standard error is NaN. The contract on
    :attr:`~photomancy.infer.posterior.Posterior.log_evidence` applies without its
    tail condition, and the error of the sum is judged by refining the grid. Coordinates
    are those of ``fit.sample_prior`` (a block's own, unconstrained, coordinates), so
    choose the nodes in them and cover the posterior mass.

    Attributes:
        axes: ``{name: (nodes, weights)}``, one entry per scalar degree of freedom.
    """

    axes: dict

    def _run(self, fit, key, *, batch_size=None):
        """Evaluate the density on the product grid."""
        template = draw_template(fit)
        slots = {}
        for key_, entry in template.items():
            if isinstance(entry, dict):
                slots.update(
                    {f"{key_}/{name}": (key_, name, a) for name, a in entry.items()}
                )
            else:
                slots[key_] = (key_, None, entry)
        names = sorted(slots)
        if set(names) != set(self.axes):
            raise ValueError(
                f"axes must name exactly the degrees of freedom {names}, "
                f"got {sorted(self.axes)}"
            )
        for name in names:
            if slots[name][2].size != 1:
                raise ValueError(
                    f"Grid needs scalar degrees of freedom; {name!r} has shape "
                    f"{slots[name][2].shape}"
                )
            weights = np.asarray(self.axes[name][1])
            if np.any(~(weights > 0)):
                raise ValueError(f"axis {name!r} has a non-positive weight")
        nodes = [_as_wide(self.axes[n][0]) for n in names]
        wts = [_as_wide(self.axes[n][1]) for n in names]
        mesh = jnp.meshgrid(*nodes, indexing="ij")
        log_w = sum(jnp.meshgrid(*[jnp.log(w) for w in wts], indexing="ij")).reshape(-1)
        flat = {n: m.reshape(-1) for n, m in zip(names, mesh, strict=True)}
        n_nodes = log_w.shape[0]
        samples = {}
        for name in names:
            key_, coord, proto = slots[name]
            value = flat[name].reshape((n_nodes, *proto.shape))
            if coord is None:
                samples[key_] = value
            else:
                samples.setdefault(key_, {})[coord] = value
        log_target = fit.log_density(samples)
        log_weights = log_w + log_target
        log_z = jax.scipy.special.logsumexp(log_weights)
        return BackendDraws(
            samples=samples,
            log_weights=log_weights,
            log_evidence=log_z,
            log_evidence_se=jnp.asarray(jnp.nan),
            info=None,
            kind="quadrature",
        )
