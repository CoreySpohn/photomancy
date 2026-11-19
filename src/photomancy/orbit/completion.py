"""Completion of the four sky constants of a Thiele-Innes orbit fit.

Relative astrometry is exactly Gaussian in the four sky constants ``c = (A, B, F, G)``
of one orbit row at fixed nonlinear shape ``theta`` (``ln n`` and the eccentricity
plane, here every entry of the flat vector that is not a constant). A completed
proposal draws ``theta`` from a base proposal and the constants from their Gaussian
conditional, which removes four of the seven orbit dimensions from the search.

``Completion`` holds the split of the flat vector, the fit without the linear terms and
the reference scale, and answers :meth:`Completion.conditional`. ``CompletedProposal``
turns a proposal over ``theta`` into a properly weighted proposal over the whole vector
(its ``log_q`` is not a density; see the class). ``ThetaPrior`` is the proposal over the
three block coordinates that follows the block's own period and eccentricity laws.

The weight of one ``theta`` uses ``n_constants`` draws of the constants from
``N(m, P^-1)``. With ``g = N(0, scale_AU^2 I)`` the reference prior of the constants and
``rest = fit_rest.log_density + volume - log g``, the fit density at ``(theta, c)`` is
``rest + log Z(theta) + log N(c | m, P^-1)``, so the weight is
``log Z + log mean_j rest_j``: an unbiased estimate of the marginal density of
``theta``. One draw is kept with probability proportional to its ``rest``, which makes
``(theta, c_kept)`` with that weight properly weighted.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import two_pi
from hwoutils.model_paths import resolve
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from orbix.system import ThieleInnesBlock, linear_gaussian_conditional
from orbix.system.charts import disk_to_plane

from photomancy.infer import (
    AbstractLinearTerm,
    AbstractProposal,
    DefensiveProposal,
    Fit,
    LinearForm,
    MixtureProposal,
)
from photomancy.infer.backends._flat import draw_template
from photomancy.measurements.astrometry import _dynamics

__all__ = ["CompletedProposal", "Completion", "ThetaPrior"]

CONSTANTS = ("A_AU", "B_AU", "F_AU", "G_AU")
# A well-conditioned sky block: only the orbit shape of the model it builds is read.
_PLACEHOLDER = (0.7, 0.3, -0.3, 0.7)
_FIELDS = ("design", "target", "precision")
_TABLE = 2**13
_BATCH = 2**14


def _flat_layout(fit):
    """Names and sizes of the flat vector of ``fit``, ``(key, name or None)`` per leaf.

    The order is that of ``ravel_pytree`` over ``fit.to_real`` (sorted keys, sorted
    coordinate names inside a block), taken from the leaf shapes on the host.
    """
    shapes = eqx.filter_eval_shape(
        lambda k: fit.to_real(fit.sample_prior(k)), jax.random.key(0)
    )
    leaves = jax.tree_util.tree_flatten_with_path(shapes)[0]
    names, sizes = [], []
    for path, leaf in leaves:
        keys = tuple(p.key for p in path)
        names.append((keys[0], keys[1] if len(keys) > 1 else None))
        sizes.append(int(np.prod(leaf.shape, dtype=int)))
    return names, sizes


def _block_of(fit):
    """The one Thiele-Innes block of ``fit`` as ``(key, block)``."""
    blocks = [
        (key, entry)
        for key, entry in fit.free.items()
        if isinstance(entry, ThieleInnesBlock)
    ]
    if len(blocks) != 1:
        raise ValueError(
            f"a completion needs exactly one ThieleInnesBlock, found {len(blocks)}"
        )
    return blocks[0]


def _support(block):
    """Chart-coordinate bounds of the block's law (private to orbix; see report)."""
    return block._support()


class _ThetaView(eqx.Module):
    """What a proposal over ``theta`` reads of the fit: its periods and the block."""

    block: ThieleInnesBlock
    theta_periods: tuple = eqx.field(static=True)

    def periods(self):
        """Periods of the ``theta`` entries."""
        return self.theta_periods


class Completion(eqx.Module):
    """The split of a Thiele-Innes fit into ``theta`` and the four sky constants.

    Build it with :meth:`from_fit`. ``theta`` is every entry of the flat unconstrained
    vector except the block's sky constants.

    Attributes:
        fit_rest: The fit without its linear terms.
        terms: The linear terms (``AbstractLinearTerm``) whose forms are concatenated.
        scale_AU: Standard deviation of the reference Gaussian on the constants.
        block_key: Key of the block in ``fit.free``.
        theta_idx: Flat indices of ``theta``.
        c_idx: Flat indices of the constants, in the order ``A, B, F, G``.
        theta_periods: Period of each ``theta`` entry (``None`` if none).
        dim: Length of the flat vector.
    """

    fit_rest: Fit
    terms: tuple[AbstractLinearTerm, ...]
    scale_AU: jax.Array
    block_key: str = eqx.field(static=True)
    theta_idx: tuple = eqx.field(static=True)
    c_idx: tuple = eqx.field(static=True)
    theta_periods: tuple = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    @classmethod
    def from_fit(cls, fit, scale_AU=None):
        """Split ``fit`` and build its completion.

        Args:
            fit: A fit with one ``ThieleInnesBlock`` and at least one linear term.
            scale_AU: Scale of the reference Gaussian on the constants, in AU. By
                default the largest projected separation of the valid relative
                astrometry rows, in AU at the system distance; there is no fixed
                default.

        Raises:
            ValueError: With no linear term, or when a term's system is not the
                block's system (named in the message).
        """
        key, block = _block_of(fit)
        linear = tuple(t for t in fit.terms if isinstance(t, AbstractLinearTerm))
        if not linear:
            raise ValueError("a completion needs a linear term (AbstractLinearTerm)")
        target = resolve(fit.model, block.system)
        for term in linear:
            if _dynamics(resolve(fit.model, term.system_path)) is not target:
                raise ValueError(
                    f"term system_path {term.system_path!r} does not resolve to the "
                    f"system {block.system!r} of the orbit block {key!r}"
                )
        rest = Fit(
            fit.model,
            fit.free,
            tuple(t for t in fit.terms if not isinstance(t, AbstractLinearTerm)),
        )
        names, sizes = _flat_layout(fit)
        edges = np.concatenate([[0], np.cumsum(sizes)])
        missing = [c for c in CONSTANTS if (key, c) not in names]
        if missing:
            raise ValueError(f"the block does not expose the constants {missing}")
        c_idx = tuple(int(edges[names.index((key, c))]) for c in CONSTANTS)
        theta_idx = tuple(i for i in range(int(edges[-1])) if i not in c_idx)
        periods = fit.periods()
        scale = _data_scale_AU(linear, target) if scale_AU is None else scale_AU
        return cls(
            fit_rest=rest,
            terms=linear,
            scale_AU=jnp.asarray(scale),
            block_key=key,
            theta_idx=theta_idx,
            c_idx=c_idx,
            theta_periods=tuple(periods[i] for i in theta_idx),
            dim=int(edges[-1]),
        )

    @property
    def block(self):
        """The Thiele-Innes block."""
        return self.fit_rest.free[self.block_key]

    @property
    def view(self):
        """The object a proposal over ``theta`` is given in place of the fit."""
        return _ThetaView(self.block, self.theta_periods)

    def split(self, z):
        """Return ``(theta, constants)`` of flat vectors ``z (..., d)``."""
        return (
            z[..., jnp.asarray(self.theta_idx)],
            z[..., jnp.asarray(self.c_idx)],
        )

    def join(self, theta, constants):
        """Flat vectors from ``theta (..., dt)`` and ``constants (..., 4)``."""
        batch = jnp.broadcast_shapes(theta.shape[:-1], constants.shape[:-1])
        z = jnp.zeros((*batch, self.dim))
        z = z.at[..., jnp.asarray(self.theta_idx)].set(theta)
        return z.at[..., jnp.asarray(self.c_idx)].set(constants)

    def _stored(self, z):
        """Stored entries and log volume of flat vectors ``z (n, d)``."""
        _, unravel = ravel_pytree(self.fit_rest.to_real(draw_template(self.fit_rest)))
        return self.fit_rest.from_real(jax.vmap(unravel)(z))

    def _form(self, theta):
        """The concatenated linear form at the orbit shape of ``theta``."""
        z0 = self.join(theta, jnp.asarray(_PLACEHOLDER))
        x, _ = self._stored(z0[None])
        model = self.fit_rest.model_at(jax.tree.map(lambda a: a[0], x))
        row = self.block.rows[0]
        forms = [t.linear_form(model, row) for t in self.terms]
        return LinearForm(
            *(jnp.concatenate([getattr(f, n) for f in forms]) for n in _FIELDS)
        )

    def conditional(self, theta):
        """Gaussian conditional of the constants given one ``theta (dt,)``.

        Returns:
            ``(mean (4,), chol (4, 4), log_marginal)``: ``chol`` is the lower Cholesky
            factor of the conditional precision (draw ``c = mean + solve(chol.T, e)``)
            and ``log_marginal`` the density of the data with the constants integrated
            out against ``N(0, scale_AU^2 I)``, in the units of the data.
        """
        form = self._form(theta)
        return linear_gaussian_conditional(
            form.design, form.target, form.precision, 1.0 / self.scale_AU**2
        )

    def _one(self, theta, key, n_constants):
        """Weight and kept draw of one ``theta``: ``(z_kept, lw, log_density_kept)``."""
        mean, chol, log_z = self.conditional(theta)
        k_draw, k_keep = jax.random.split(key)
        eps = jax.random.normal(k_draw, (n_constants, 4))
        c = mean + solve_triangular(chol.T, eps.T, lower=False).T
        z = self.join(theta, c)
        x, volume = self._stored(z)
        rest = self.fit_rest.log_density(x) + volume - _log_reference(c, self.scale_AU)
        keep = jnp.argmax(rest + jax.random.gumbel(k_keep, (n_constants,)))
        lw = log_z + logsumexp(rest) - jnp.log(n_constants)
        diff = chol.T @ (c[keep] - mean)
        log_cond = -0.5 * jnp.sum(diff**2) + jnp.sum(jnp.log(jnp.diag(chol)))
        log_cond = log_cond - 2.0 * jnp.log(two_pi)
        return z[keep], lw, log_z + log_cond + rest[keep]

    def complete(self, key, theta, n_constants):
        """Complete ``theta (n, dt)``: return ``(z (n, d), lw (n,), log_density (n,))``.

        ``lw`` is the log marginal weight of each ``theta``; ``log_density`` is the
        fit's log density (plus volume) at the kept draw, from the form.
        """
        keys = jax.random.split(key, theta.shape[0])
        return jax.lax.map(
            lambda a: self._one(*a, n_constants),
            (theta, keys),
            batch_size=min(_BATCH, theta.shape[0]),
        )


def _log_reference(c, scale_AU):
    """Log density of ``N(0, scale_AU^2 I)`` at ``c (..., p)``."""
    p = c.shape[-1]
    return -0.5 * jnp.sum(c**2, -1) / scale_AU**2 - p * (
        jnp.log(scale_AU) + 0.5 * jnp.log(two_pi)
    )


def _data_scale_AU(terms, system):
    """Largest projected separation of the valid rows of ``terms``, in AU."""
    dist_pc = _dynamics(system).frame.dist_pc
    sep = jnp.zeros(())
    for term in terms:
        data = term.data
        sep = jnp.maximum(
            sep, jnp.max(jnp.where(data.is_valid, jnp.hypot(data.ra, data.dec), 0.0))
        )
    # arcsec times parsec is AU
    return eqx.error_if(
        sep * dist_pc, sep <= 0, "completion: no valid astrometric row sets a scale"
    )


def eccentricity_table(block):
    """``(grid, cdf)``: the block law's eccentricity distribution function on a grid.

    The inverse of this table (``jnp.interp(u, cdf, grid)``) is the eccentricity sampler
    of the orbit proposals; its density is piecewise constant, :func:`log_density_e`.
    The law's own inverse distribution function iterates a root find that costs seconds
    per ``2**17`` draws, the distribution function costs a millisecond.
    """
    low, high = block.law.support()["e"]
    grid = jnp.linspace(low, high, _TABLE + 1)
    return grid, block.law.eccentricity.cdf(grid)


def draw_eccentricity(key, table, n):
    """``n`` eccentricities from the table of :func:`eccentricity_table`."""
    grid, cdf = table
    return jnp.interp(jax.random.uniform(key, (n,)), cdf, grid)


def log_density_e(e, table):
    """Log density of :func:`draw_eccentricity` at ``e`` (zero outside the table)."""
    grid, cdf = table
    k = jnp.clip(jnp.searchsorted(grid, e) - 1, 0, _TABLE - 1)
    mass = cdf[k + 1] - cdf[k]
    return jnp.log(jnp.maximum(mass, 0.0)) - jnp.log(grid[k + 1] - grid[k])


def block_coordinates(block, y):
    """Chart coordinates ``x (n, 7)`` and log Jacobian of stored ``y (n, 7)``."""
    return block.chart.constrain(y, _support(block))


class ThetaPrior(AbstractProposal):
    """The block's own law of ``(ln n, ux, uy)`` as a proposal over those coordinates.

    ``ln n`` is uniform on the period bounds, the eccentricity has the law's marginal
    (a table of its distribution function, so ``log_q`` is the density of the draws
    exactly) and the eccentricity angle is uniform, all in the stored coordinates of the
    Thiele-Innes block. Valid when ``theta`` is exactly the three block coordinates.

    Fails if the draws and ``log_q`` disagree (the weights of a prior-only fit are not
    one).
    """

    def _check(self, fit):
        if len(fit.periods()) != 3:
            raise ValueError("ThetaPrior needs theta to be the three block coordinates")

    def log_prob(self, z, fit):
        """Log density of ``z (n, 3)`` in the stored coordinates."""
        self._check(fit)
        block = fit.block
        x, log_jac = block_coordinates(block, jnp.concatenate([z, _pad(z)], -1))
        lo, hi = _support(block)["ln_size"]
        r2 = x[:, 1] ** 2 + x[:, 2] ** 2
        e = jnp.sqrt(r2 / (1.0 + r2))
        log_pe = log_density_e(e, eccentricity_table(block))
        log_x = -jnp.log(hi - lo) + log_pe - jnp.log(two_pi * e) - 2.0 * jnp.log1p(r2)
        return log_x + log_jac

    def sample(self, key, fit, n):
        """Draw ``n`` points; return ``(z, log_q)``."""
        self._check(fit)
        block = fit.block
        k_n, k_e, k_a = jax.random.split(key, 3)
        lo, hi = _support(block)["ln_size"]
        ln_n = lo + (hi - lo) * jax.random.uniform(k_n, (n,))
        e = draw_eccentricity(k_e, eccentricity_table(block), n)
        ang = two_pi * jax.random.uniform(k_a, (n,))
        evec = jnp.stack([e * jnp.cos(ang), e * jnp.sin(ang)], -1)
        x = jnp.concatenate([ln_n[:, None], disk_to_plane(evec)], -1)
        y = block.chart.unconstrain(jnp.concatenate([x, _pad(x)], -1), _support(block))
        z = y[:, :3]
        return z, self.log_prob(z, fit)

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive):
        """Refit a mixture on the cloud, with this prior as the defensive component."""
        mixture = MixtureProposal.from_cloud(
            key, z, log_weights, n_centers=n_centers, periods=fit.periods()
        )
        return DefensiveProposal(mixture, defensive, prior=self)


def _pad(a):
    """Zeros for the four sky constants, matching the batch of ``a``."""
    return jnp.zeros((*a.shape[:-1], 4))


class CompletedProposal(AbstractProposal):
    """Draw ``theta`` from ``base`` and complete the four sky constants.

    ``sample`` returns the flat vector of the kept constants draw and the ``log_q``
    that makes ``fit.log_density + volume - log_q`` equal the completed weight
    ``log Z(theta) + log mean_j rest_j - log q(theta)``. That ``log_q`` is not the
    density of the draw (the constants have a mixture density over the kept draw), so
    ``log_prob`` is undefined and ``temperable`` is false: ``AdaptiveImportance`` refits
    on the weights of the fit's own density and the refit is over ``theta`` only.

    Attributes:
        base: Proposal over ``theta`` (it sees a view with ``periods()`` and
            ``block``); :class:`ThetaPrior` or a refit of it.
        n_constants: Draws of the constants per ``theta``.
        scale_AU: Scale of the reference Gaussian on the constants; ``None`` takes the
            largest projected separation of the valid rows (see
            :meth:`Completion.from_fit`).
    """

    temperable: ClassVar[bool] = False

    base: AbstractProposal
    n_constants: int = eqx.field(static=True, default=8)
    scale_AU: jax.Array | None = None

    def __init__(self, base, *, n_constants=8, scale_AU=None):
        """Wrap ``base``; ``scale_AU`` None takes the data scale of the fit."""
        self.base = base
        self.n_constants = n_constants
        self.scale_AU = None if scale_AU is None else jnp.asarray(scale_AU)

    def sample(self, key, fit, n):
        """Draw ``n`` completed points; return ``(z, log_q)``."""
        completion = Completion.from_fit(fit, self.scale_AU)
        k_base, k_c = jax.random.split(key)
        theta, log_q = self.base.sample(k_base, completion.view, n)
        z, lw, log_density = completion.complete(k_c, theta, self.n_constants)
        return z, jnp.where(jnp.isneginf(lw), log_q, log_q + log_density - lw)

    def log_prob(self, z, fit):
        """Undefined: a completed proposal is properly weighted, not a density."""
        raise NotImplementedError("a completed proposal has no density; use sample")

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive):
        """Refit the base over ``theta`` of the weighted cloud."""
        completion = Completion.from_fit(fit, self.scale_AU)
        theta, _ = completion.split(z)
        base = self.base.adapt(
            key,
            theta,
            log_weights,
            completion.view,
            n_centers=n_centers,
            defensive=defensive,
        )
        return eqx.tree_at(lambda p: p.base, self, base)
