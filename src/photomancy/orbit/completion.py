"""Completion of the four sky constants of a Thiele-Innes orbit fit.

Relative astrometry is exactly Gaussian in the four sky constants ``c = (A, B, F, G)``
of one orbit row at fixed nonlinear shape ``theta`` (``ln n`` and the eccentricity
plane of the block). A completed proposal draws ``theta`` from a base proposal and the
constants from their Gaussian conditional, which removes four of the seven orbit
dimensions from the search.

``Completion`` splits the flat vector three ways: ``theta``, the constants, and the
conditional leaves (a free leaf that a term proposes, an
:class:`~photomancy.infer.AbstractConditionalTerm` such as the report term's planet
radius). It holds the fit without the linear terms and the reference scale, and answers
:meth:`Completion.conditional`. ``CompletedProposal`` turns a proposal over ``theta``
into a properly weighted proposal over the whole vector (its ``log_q`` is not a
density; the class states the estimator). ``ThetaPrior`` is the proposal over the three
block coordinates that follows the block's own period and eccentricity laws.

With ``g = N(0, scale_AU^2 I)`` the reference prior of the constants and
``f = fit_rest.density * volume / g``, the fit density at ``(theta, c, t)`` is
``f * Z(theta) * N(c | m, P^-1)``, so the mean of ``f`` over constants drawn from
``N(m, P^-1)`` (and leaves drawn from their proposal, over its density) estimates the
marginal density of ``theta`` without bias. The constants are drawn as a pool screened
by the prior of the fit alone, which costs no term evaluation, and only a few of them,
resampled by the screen, are scored by the terms.

A proposed leaf comes from ``q(t | theta, c)``, the term's record (a Gaussian mixture,
each component widened by ``leaf_widen`` and truncated to the leaf law's support, mixed
with a uniform share ``leaf_uniform`` of that support, all of it when the term has no
peak). The record is approximate, so the score carries the exact fit density over the
closed-form density of the leaf draw, ``f / q``: the record changes the variance of the
weights, never their mean. It is never a target.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import two_pi
from hwoutils.model_paths import resolve
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import log_ndtr, logsumexp
from orbix.system import ThieleInnesBlock, linear_gaussian_conditional
from orbix.system.charts import disk_to_plane
from priorities import from_real, to_real

from photomancy.infer import (
    AbstractConditionalTerm,
    AbstractDensityProposal,
    AbstractLinearTerm,
    AbstractProposal,
    DefensiveProposal,
    Fit,
    LinearForm,
    MixtureProposal,
)
from photomancy.infer.backends._flat import draw_template
from photomancy.infer.particles import systematic_indices
from photomancy.measurements.astrometry import _dynamics

__all__ = ["CompletedProposal", "Completion", "ThetaPrior"]

CONSTANTS = ("A_AU", "B_AU", "F_AU", "G_AU")
# A well-conditioned sky block: only the orbit shape of the model it builds is read.
_PLACEHOLDER = (0.7, 0.3, -0.3, 0.7)
_FIELDS = ("design", "target", "precision")
_TABLE = 2**13
_BATCH = 2**14
# a leaf component reaches this many widened sd, so its tail draws stay accurate
_WINDOW = 6.0


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


class Completion(eqx.Module):
    """The three-way split of a Thiele-Innes fit and the proposal stages it completes.

    Build it with :meth:`from_fit`. Every entry of the flat unconstrained vector is in
    exactly one of three groups: ``theta``, the orbit coordinates of the block that a
    proposal over ``theta`` covers (``ln n`` and the eccentricity plane); the four sky
    constants, completed from the linear form; and the conditional leaves, each drawn
    from the record of the term that proposes it. A completion is also what a proposal
    over ``theta`` is given in place of the fit: it answers ``periods()``, ``block``
    and ``model``.

    Attributes:
        fit_rest: The fit without its linear terms.
        terms: The linear terms (``AbstractLinearTerm``) whose forms are concatenated.
        scale_AU: Standard deviation of the reference Gaussian on the constants.
        block_key: Key of the block in ``fit.free``.
        theta_idx: Flat indices of ``theta``.
        c_idx: Flat indices of the constants, in the order ``A, B, F, G``.
        theta_periods: Period of each ``theta`` entry (``None`` if none).
        dim: Length of the flat vector.
        leaf_terms: Indices in ``fit_rest.terms`` of the terms that propose a free
            leaf (an :class:`AbstractConditionalTerm` whose ``leaf_path`` is free).
        leaf_paths: The leaf each of those terms proposes.
        leaf_idx: Flat index of each of those leaves (a leaf holds one value).
        leaf_widen: Factor on the standard deviation of the proposal components.
        leaf_uniform: Share of the leaf's support drawn uniformly.
    """

    fit_rest: Fit
    terms: tuple[AbstractLinearTerm, ...]
    scale_AU: jax.Array
    block_key: str = eqx.field(static=True)
    theta_idx: tuple = eqx.field(static=True)
    c_idx: tuple = eqx.field(static=True)
    theta_periods: tuple = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    leaf_terms: tuple = eqx.field(static=True, default=())
    leaf_paths: tuple = eqx.field(static=True, default=())
    leaf_idx: tuple = eqx.field(static=True, default=())
    leaf_widen: float = eqx.field(static=True, default=1.5)
    leaf_uniform: float = eqx.field(static=True, default=0.1)

    @classmethod
    def from_fit(cls, fit, scale_AU=None, *, leaf_widen=1.5, leaf_uniform=0.1):
        """Split ``fit`` and build its completion.

        Args:
            fit: A fit with one ``ThieleInnesBlock`` and at least one linear term.
            scale_AU: Scale of the reference Gaussian on the constants, in AU. By
                default the largest projected separation of the valid relative
                astrometry rows, in AU at the system distance; there is no fixed
                default.
            leaf_widen: Factor on the standard deviation of a term's leaf proposal.
            leaf_uniform: Share of a free leaf's support drawn uniformly.

        Raises:
            ValueError: With no linear term, when a term's system is not the
                block's system (named in the message), when two terms propose one
                leaf, when a proposed leaf holds more than one value, or when a free
                entry is in none of the three groups (a free leaf no term offers a
                conditional for, or a second block), named in the message.
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
        leaf_terms, leaf_paths, leaf_idx = [], [], []
        for i, term in enumerate(rest.terms):
            path = term.leaf_path if isinstance(term, AbstractConditionalTerm) else None
            if path is None or path not in fit.leaf_laws:
                continue
            if path in leaf_paths:
                raise ValueError(f"two terms propose the leaf {path!r}; keep one")
            at = names.index((path, None))
            if sizes[at] != 1:
                raise ValueError(f"the proposed leaf {path!r} must hold a single value")
            leaf_terms.append(i)
            leaf_paths.append(path)
            leaf_idx.append(int(edges[at]))
        theta_idx, orphans = [], []
        for (owner, name), at, size in zip(names, edges[:-1], sizes, strict=True):
            if owner == key and name not in CONSTANTS:
                theta_idx.extend(range(int(at), int(at) + size))
            elif owner != key and owner not in leaf_paths:
                orphans.append(owner)
        if orphans:
            raise ValueError(
                f"free entries {sorted(set(orphans))} are neither orbit coordinates "
                f"of the block {key!r} nor leaves a term proposes "
                "(AbstractConditionalTerm.leaf_path); a completion cannot draw them"
            )
        theta_idx = tuple(theta_idx)
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
            leaf_terms=tuple(leaf_terms),
            leaf_paths=tuple(leaf_paths),
            leaf_idx=tuple(leaf_idx),
            leaf_widen=leaf_widen,
            leaf_uniform=leaf_uniform,
        )

    @property
    def block(self):
        """The Thiele-Innes block."""
        return self.fit_rest.free[self.block_key]

    @property
    def model(self):
        """The model of the fit."""
        return self.fit_rest.model

    def periods(self):
        """Periods of the ``theta`` entries (what a proposal over ``theta`` reads)."""
        return self.theta_periods

    def split(self, z):
        """Return ``(theta, constants)`` of flat vectors ``z (..., d)``.

        The proposed leaves belong to neither.
        """
        return (
            z[..., jnp.asarray(self.theta_idx)],
            z[..., jnp.asarray(self.c_idx)],
        )

    def join(self, theta, constants):
        """Flat vectors from ``theta (..., dt)`` and ``constants (..., 4)``.

        The proposed leaves are zero (a valid real value of any leaf law).
        """
        batch = jnp.broadcast_shapes(theta.shape[:-1], constants.shape[:-1])
        z = jnp.zeros((*batch, self.dim))
        z = z.at[..., jnp.asarray(self.theta_idx)].set(theta)
        return z.at[..., jnp.asarray(self.c_idx)].set(constants)

    def _draw_leaves(self, key, z):
        """Draw the proposed leaves of ``z (n, d)``; return ``(z, log_q)``.

        ``log_q`` ``(n,)`` is the log density of the draws in the real coordinates of
        the flat vector: the density of the leaf value times the volume of the map
        from real to leaf value.
        """
        x, _ = self._stored(z)
        models = jax.vmap(self.fit_rest.model_at)(x)
        log_q = jnp.zeros(z.shape[0])
        for i, (path, index, term_index) in enumerate(
            zip(self.leaf_paths, self.leaf_idx, self.leaf_terms, strict=True)
        ):
            dist = self.fit_rest.free[path]
            record = jax.vmap(self.fit_rest.terms[term_index].leaf_conditional)(models)
            low, high = (jnp.log(jnp.maximum(b, 1e-300)) for b in dist.support())
            keys = jax.random.split(jax.random.fold_in(key, i), z.shape[0])
            draw = partial(
                _draw_leaf,
                low=low,
                high=high,
                widen=self.leaf_widen,
                uniform=self.leaf_uniform,
            )
            ln_leaf, log_q_ln = jax.vmap(draw)(keys, record)
            leaf = jnp.exp(ln_leaf)[:, None]
            real = to_real(dist, leaf)
            volume = jnp.sum(from_real(dist, real)[1], axis=-1)
            log_q = log_q + log_q_ln - ln_leaf + volume
            z = z.at[:, index].set(real[:, 0])
        return z, log_q

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

    def _constants(self, theta, key, n_constants):
        """Pool of constants at one ``theta``: ``(c, z, log_s, log_z, mean, chol)``.

        ``log_s`` is the screen: the prior of the fit alone (every proposed leaf at its
        placeholder, no term evaluated) plus the volume over the reference Gaussian,
        ``-inf`` wherever it is not finite.
        """
        mean, chol, log_z = self.conditional(theta)
        eps = jax.random.normal(key, (n_constants, 4))
        c = mean + solve_triangular(chol.T, eps.T, lower=False).T
        z = self.join(theta, c)
        x, volume = self._stored(z)
        log_s = self.fit_rest.log_prior(x) + volume - _log_reference(c, self.scale_AU)
        return c, z, jnp.where(jnp.isfinite(log_s), log_s, -jnp.inf), log_z, mean, chol

    def _one(self, theta, key, n_constants, n_scored):
        """Weight and kept draw of one ``theta``: ``(z_kept, lw, log_density_kept)``.

        The pool of ``n_constants`` constants is screened by the prior; ``n_scored`` of
        them, resampled systematically by the screen, get their leaves drawn and the
        terms evaluated (see :class:`CompletedProposal`). With no term left to score
        the screen is the whole density and nothing is resampled.
        """
        k_draw, k_keep = jax.random.split(key)
        c, z, log_s, log_z, mean, chol = self._constants(theta, k_draw, n_constants)
        total = logsumexp(log_s)
        lw = log_z + total - jnp.log(n_constants)
        if not self.fit_rest.terms:
            keep = jnp.argmax(log_s + jax.random.gumbel(k_keep, (n_constants,)))
            return (
                z[keep],
                lw,
                log_z + _log_conditional(c[keep], mean, chol) + log_s[keep],
            )
        alive = jnp.isfinite(total)
        normalized = jnp.where(alive, log_s - total, -jnp.log(n_constants))
        idx = systematic_indices(jax.random.fold_in(key, 2), normalized, n_scored)
        c, z = c[idx], z[idx]
        log_q_leaf = jnp.zeros(n_scored)
        if self.leaf_paths:
            z, log_q_leaf = self._draw_leaves(jax.random.fold_in(key, 1), z)
        x, volume = self._stored(z)
        target = (
            self.fit_rest.log_density(x) + volume - _log_reference(c, self.scale_AU)
        )
        r = target - log_q_leaf - log_s[idx]
        r = jnp.where(alive & ~jnp.isnan(r), r, -jnp.inf)
        keep = jnp.argmax(r + jax.random.gumbel(k_keep, (n_scored,)))
        lw = jnp.where(alive, lw + logsumexp(r) - jnp.log(n_scored), -jnp.inf)
        return z[keep], lw, log_z + _log_conditional(c[keep], mean, chol) + target[keep]

    def complete(self, key, theta, n_constants, n_scored):
        """Complete ``theta (n, dt)``: return ``(z (n, d), lw (n,), log_density (n,))``.

        ``lw`` is the log marginal weight of each ``theta``; ``log_density`` is the
        fit's log density (plus volume) at the kept draw, from the form. ``n_scored``
        is not read when the fit has no term left to score.
        """
        keys = jax.random.split(key, theta.shape[0])
        return jax.lax.map(
            lambda a: self._one(*a, n_constants, n_scored),
            (theta, keys),
            batch_size=min(_BATCH, theta.shape[0]),
        )


def _log_conditional(c, mean, chol):
    """Log density of ``N(mean, P^-1)`` at ``c (4,)``, ``chol`` the factor of ``P``."""
    diff = chol.T @ (c - mean)
    return (
        -0.5 * jnp.sum(diff**2)
        + jnp.sum(jnp.log(jnp.diag(chol)))
        - 2.0 * jnp.log(two_pi)
    )


def _log_normal_mass(a, b):
    """``log(Phi(b) - Phi(a))`` for ``a <= b``, from the tail with the smaller mass."""
    upper = a > 0
    lo, hi = jnp.where(upper, -b, a), jnp.where(upper, -a, b)
    log_hi = log_ndtr(hi)
    return log_hi + jnp.log1p(-jnp.exp(log_ndtr(lo) - log_hi))


def _draw_leaf(key, record, low, high, widen, uniform):
    """One draw of ``ln(leaf)`` from a record's mixture; return ``(ln_leaf, log_q)``.

    Component ``k`` is ``Normal(mean_k, (widen * sd_k)^2)`` truncated to ``[low, high]``
    (the record's own bounds intersected with the law's) and to ``_WINDOW`` widened
    standard deviations about its mean (a component with no overlap is dropped), with
    weight proportional to
    ``exp(log_pi_k)`` times the mass the unwidened Gaussian has inside. The uniform
    share ``uniform`` of ``[low, high]`` is the rest, and all of it when no component
    has weight. ``log_q`` is the exact log density of the draw in ``ln(leaf)``; it is at
    least ``log(uniform) - log(high - low)`` for every draw inside the range.
    """
    low = jnp.maximum(low, record.low)
    high = jnp.minimum(high, record.high)
    scale = widen * record.sd
    win_lo = jnp.maximum(low, record.mean - _WINDOW * scale)
    win_hi = jnp.minimum(high, record.mean + _WINDOW * scale)
    reach = win_hi > win_lo
    win_lo, win_hi = jnp.where(reach, win_lo, 0.0), jnp.where(reach, win_hi, 1.0)
    inside = _log_normal_mass(
        (low - record.mean) / record.sd, (high - record.mean) / record.sd
    )
    log_w = jnp.where(reach, record.log_pi + inside, -jnp.inf)
    top = jnp.max(log_w)
    peak = jnp.isfinite(top)
    share = jnp.where(peak, uniform, 1.0)
    log_w = jnp.where(
        peak,
        log_w - logsumexp(jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)),
        -jnp.inf,
    )
    weight = jnp.exp(log_w) * (1.0 - share)
    lo_std, hi_std = (win_lo - record.mean) / scale, (win_hi - record.mean) / scale
    k_pick, k_norm, k_flat = jax.random.split(key, 3)
    cdf = jnp.cumsum(jnp.concatenate([weight, jnp.asarray([share])]))
    pick = jnp.searchsorted(cdf, jax.random.uniform(k_pick) * cdf[-1], side="right")
    pick = jnp.minimum(pick, cdf.shape[0] - 1)
    flat = pick == cdf.shape[0] - 1
    comp = jnp.minimum(pick, scale.shape[0] - 1)
    # an interval in the upper tail is drawn as its mirror image in the lower tail
    mirror = lo_std[comp] > 0
    a = jnp.where(mirror, -hi_std[comp], lo_std[comp])
    b = jnp.where(mirror, -lo_std[comp], hi_std[comp])
    z = jax.random.truncated_normal(k_norm, a, b)
    z = jnp.where(mirror, -z, z)
    draw_n = record.mean[comp] + scale[comp] * z
    draw_u = low + (high - low) * jax.random.uniform(k_flat)
    ln_leaf = jnp.where(flat, draw_u, draw_n)
    ln_leaf = jnp.clip(ln_leaf, low, high)
    log_mass = _log_normal_mass(lo_std, hi_std)
    z_at = (ln_leaf - record.mean) / scale
    log_comp = (
        jnp.log(jnp.where(weight > 0, weight, 1.0))
        - 0.5 * z_at**2
        - 0.5 * jnp.log(two_pi)
        - jnp.log(scale)
        - log_mass
    )
    in_window = (ln_leaf >= win_lo) & (ln_leaf <= win_hi)
    log_comp = jnp.where((weight > 0) & in_window, log_comp, -jnp.inf)
    log_flat = jnp.log(share) - jnp.log(high - low)
    return ln_leaf, logsumexp(jnp.concatenate([log_comp, log_flat[None]]))


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


def block_coordinates(block, model, y):
    """Chart coordinates ``x (n, 7)`` and log Jacobian of stored ``y (n, 7)``."""
    return block.chart.constrain(y, block.chart_support(model))


class ThetaPrior(AbstractDensityProposal):
    """The block's own law of ``(ln n, ux, uy)`` as a proposal over those coordinates.

    ``ln n`` is uniform on the period bounds, the eccentricity has the law's marginal
    (a table of its distribution function, so ``log_q`` is the density of the draws
    exactly) and the eccentricity angle is uniform, all in the stored coordinates of the
    Thiele-Innes block. It is given a :class:`Completion` in place of the fit, whose
    ``theta`` is exactly the three block coordinates (``Completion.from_fit`` refuses
    any other free entry).

    Fails if the draws and ``log_q`` disagree (the weights of a prior-only fit are not
    one).
    """

    def log_prob(self, z, fit):
        """Log density of ``z (n, 3)`` in the stored coordinates."""
        block = fit.block
        x, log_jac = block_coordinates(
            block, fit.model, jnp.concatenate([z, _pad(z)], -1)
        )
        lo, hi = block.chart_support(fit.model)["ln_size"]
        r2 = x[:, 1] ** 2 + x[:, 2] ** 2
        e = jnp.sqrt(r2 / (1.0 + r2))
        log_pe = log_density_e(e, eccentricity_table(block))
        log_x = -jnp.log(hi - lo) + log_pe - jnp.log(two_pi * e) - 2.0 * jnp.log1p(r2)
        return log_x + log_jac

    def sample(self, key, fit, n):
        """Draw ``n`` points; return ``(z, log_q)``."""
        block = fit.block
        k_n, k_e, k_a = jax.random.split(key, 3)
        lo, hi = block.chart_support(fit.model)["ln_size"]
        ln_n = lo + (hi - lo) * jax.random.uniform(k_n, (n,))
        e = draw_eccentricity(k_e, eccentricity_table(block), n)
        ang = two_pi * jax.random.uniform(k_a, (n,))
        evec = jnp.stack([e * jnp.cos(ang), e * jnp.sin(ang)], -1)
        x = jnp.concatenate([ln_n[:, None], disk_to_plane(evec)], -1)
        y = block.chart.unconstrain(
            jnp.concatenate([x, _pad(x)], -1), block.chart_support(fit.model)
        )
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
    """Draw ``theta`` from ``base`` and complete the constants and the proposed leaves.

    At one ``theta``, with ``Z``, ``m`` and ``P`` the marginal and the conditional of
    the linear form and ``g`` the reference Gaussian of the constants:

    1. a pool ``c_1 .. c_M`` (``M = n_constants``) is drawn from ``N(m, P^-1)``;
    2. the screen ``s_j = prior(theta, c_j) * volume / g(c_j)`` is the prior of the fit
       alone, every proposed leaf at its placeholder, and no term is evaluated;
       ``S = sum_j s_j``;
    3. ``K = n_scored`` indices ``I_k`` are resampled systematically on ``s``;
    4. each gets its leaves ``t_k`` from the proposing terms' records
       ``q(t | theta, c_{I_k})`` and the score
       ``r_k = f(theta, c_{I_k}, t_k) / (q(t_k) * s_{I_k})``, with ``f`` the density of
       the fit without its linear terms, times the volume, over ``g``;
    5. the weight is ``W = Z * (S / M) * mean_k r_k`` and one ``k`` is kept with
       probability proportional to ``r_k``.

    Systematic resampling gives pool member ``j`` an expected count ``K s_j / S``, so
    ``E[W | pool] = Z * mean_j F(c_j)`` with ``F(c)`` the integral of ``f`` over the
    leaves: ``W`` is an unbiased estimate of the marginal density of ``theta`` and the
    kept draw with weight ``W`` is properly weighted. The screen and the leaf records
    are proposals, never targets: they change the variance of the weights, never their
    mean, and need only be positive wherever ``f`` is. A ``theta`` whose pool has no
    draw inside the prior gets ``W = 0`` (``-inf``). With no term left to score,
    ``r_k = 1``: ``W = Z * S / M`` and the kept draw is picked on ``s``.

    ``sample`` returns the flat vector of the kept draw and the ``log_q`` that makes
    ``fit.log_density + volume - log_q`` equal ``log W - log q(theta)``. That ``log_q``
    is not the density of the draw, so no density exists at ``z`` (it is not an
    ``AbstractDensityProposal``): ``AdaptiveImportance`` refits on the weights of the
    fit's own density and the refit is over ``theta`` only.

    Attributes:
        base: Proposal over ``theta`` (it is given the :class:`Completion`, which
            answers ``periods()``, ``block`` and ``model``); :class:`ThetaPrior` or a
            refit of it.
        n_constants: Size of the pool of constants per ``theta``; ``None`` takes 8
            when the fit has no term left to score and 32 otherwise.
        n_scored: Pool members, resampled by the screen, that get leaf draws and term
            evaluations per ``theta``; not read when no term is left to score.
        scale_AU: Scale of the reference Gaussian on the constants; ``None`` takes the
            largest projected separation of the valid rows (see
            :meth:`Completion.from_fit`).
        leaf_widen: Factor on the standard deviation of the leaf proposal components
            (see the module docstring); only used when a term proposes a free leaf.
        leaf_uniform: Share of the leaf's support drawn uniformly.
    """

    base: AbstractDensityProposal
    n_constants: int | None = eqx.field(static=True, default=None)
    n_scored: int = eqx.field(static=True, default=4)
    scale_AU: jax.Array | None = None
    leaf_widen: float = eqx.field(static=True, default=1.5)
    leaf_uniform: float = eqx.field(static=True, default=0.1)

    def __init__(
        self,
        base,
        *,
        n_constants=None,
        n_scored=4,
        scale_AU=None,
        leaf_widen=1.5,
        leaf_uniform=0.1,
    ):
        """Wrap ``base``; ``scale_AU`` None takes the data scale of the fit."""
        self.base = base
        self.n_constants = n_constants
        self.n_scored = n_scored
        self.scale_AU = None if scale_AU is None else jnp.asarray(scale_AU)
        self.leaf_widen = leaf_widen
        self.leaf_uniform = leaf_uniform

    def _completion(self, fit):
        return Completion.from_fit(
            fit,
            self.scale_AU,
            leaf_widen=self.leaf_widen,
            leaf_uniform=self.leaf_uniform,
        )

    def sample(self, key, fit, n):
        """Draw ``n`` completed points; return ``(z, log_q)``."""
        completion = self._completion(fit)
        n_constants = self.n_constants
        if n_constants is None:
            n_constants = 32 if completion.fit_rest.terms else 8
        k_base, k_c = jax.random.split(key)
        theta, log_q = self.base.sample(k_base, completion, n)
        z, lw, log_density = completion.complete(k_c, theta, n_constants, self.n_scored)
        return z, jnp.where(jnp.isneginf(lw), log_q, log_q + log_density - lw)

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive):
        """Refit the base over ``theta`` of the weighted cloud."""
        completion = self._completion(fit)
        theta, _ = completion.split(z)
        base = self.base.adapt(
            key,
            theta,
            log_weights,
            completion,
            n_centers=n_centers,
            defensive=defensive,
        )
        return eqx.tree_at(lambda p: p.base, self, base)
