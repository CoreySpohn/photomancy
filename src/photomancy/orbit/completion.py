"""Completion of the four sky constants of a Thiele-Innes orbit fit.

Relative astrometry is exactly Gaussian in the four sky constants ``c = (A, B, F, G)``
of one orbit row at a fixed nonlinear orbit shape (``ln n`` and the eccentricity plane
of the block). A completed proposal draws ``shape`` from a base proposal and the
constants from their Gaussian conditional, which removes four of the seven orbit
dimensions from the search.

``Completion`` splits the flat vector three ways: ``shape``, the constants, and the
conditional leaves (a free leaf that a term proposes, an
:class:`~photomancy.infer.AbstractConditionalTerm` such as the report term's planet
radius). It holds the fit without the linear terms and the reference scale, and answers
:meth:`Completion.conditional`. ``CompletedProposal`` turns a proposal over ``shape``
into a properly weighted proposal over the whole vector (it has no density; the class
states the estimator). The base proposal over ``shape`` is the block's own law,
:class:`~photomancy.orbit.proposals.ShapePriorProposal`, or a refit of it.

With ``g = N(0, scale_AU^2 I)`` the reference prior of the constants and
``f = fit_rest.density * volume / g``, the fit density at ``(shape, c, t)`` is
``f * Z(shape) * N(c | m, P^-1)``, so the mean of ``f`` over constants drawn from
``N(m, P^-1)`` (and leaves drawn from their proposal, over its density) estimates the
marginal density of ``shape`` without bias. The constants are drawn as a pool screened
by the prior of the fit alone, which costs no term evaluation, and only a few of them,
resampled by the screen, are scored by the terms.

A proposed leaf comes from ``q(t | shape, c)``, the term's record (a Gaussian mixture,
each component widened by ``leaf_widen`` and truncated to the leaf law's support, mixed
with a uniform share ``leaf_uniform`` of that support, all of it when the term has no
peak). The record is approximate, so the score carries the exact fit density over the
closed-form density of the leaf draw, ``f / q``: the record changes the variance of the
weights, never their mean. It is never a target.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import two_pi
from hwoutils.model_paths import resolve
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from orbix.system import System, ThieleInnesBlock, linear_gaussian_conditional
from priorities import from_real, to_real

from photomancy.infer import (
    AbstractConditionalTerm,
    AbstractDensityProposal,
    AbstractLinearTerm,
    AbstractProposal,
    Fit,
    LinearForm,
)
from photomancy.infer.particles import systematic_indices

__all__ = ["CompletedProposal", "Completion"]

CONSTANTS = ("A_AU", "B_AU", "F_AU", "G_AU")
# A well-conditioned sky block: only the orbit shape of the model it builds is read.
_PLACEHOLDER = (0.7, 0.3, -0.3, 0.7)
_FIELDS = ("design", "target", "precision")
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
    blocks = [(k, e) for k, e in fit.free.items() if isinstance(e, ThieleInnesBlock)]
    if len(blocks) != 1:
        raise ValueError(
            f"a completion needs exactly one ThieleInnesBlock, found {len(blocks)}"
        )
    return blocks[0]


class Completion(eqx.Module):
    """The three-way split of a Thiele-Innes fit and the proposal stages it completes.

    Build it with :meth:`from_fit`. Every entry of the flat unconstrained vector is in
    exactly one of three groups: ``shape``, the orbit coordinates of the block that a
    proposal over ``shape`` covers (``ln n`` and the eccentricity plane); the four sky
    constants, completed from the linear form; and the conditional leaves, each drawn
    from the record of the term that proposes it. A completion is also what a proposal
    over ``shape`` is given in place of the fit: it answers ``periods()``, ``block``
    and ``model``.

    Attributes:
        fit_rest: The fit without its linear terms.
        terms: The linear terms (``AbstractLinearTerm``) whose forms are concatenated.
        scale_AU: Standard deviation of the reference Gaussian on the constants.
        block_key: Key of the block in ``fit.free``.
        shape_idx: Flat indices of ``shape``.
        constants_idx: Flat indices of the constants, in the order ``A, B, F, G``.
        shape_periods: Period of each ``shape`` entry (``None`` if none).
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
    shape_idx: tuple = eqx.field(static=True)
    constants_idx: tuple = eqx.field(static=True)
    shape_periods: tuple = eqx.field(static=True)
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
                astrometry rows, in AU at the system distance (NaN with no valid
                row, which refuses the slice); there is no fixed default.
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
            system = resolve(fit.model, term.system_path)
            dynamics = system if isinstance(system, System) else system.dynamics
            if dynamics is not target:
                raise ValueError(
                    f"term system_path {term.system_path!r} does not resolve to the "
                    f"system {block.system!r} of the orbit block {key!r}"
                )
        rest = tuple(t for t in fit.terms if not isinstance(t, AbstractLinearTerm))
        rest = Fit(fit.model, fit.free, rest)
        names, sizes = _flat_layout(fit)
        edges = np.concatenate([[0], np.cumsum(sizes)])
        missing = [c for c in CONSTANTS if (key, c) not in names]
        if missing:
            raise ValueError(f"the block does not expose the constants {missing}")
        constants_idx = tuple(int(edges[names.index((key, c))]) for c in CONSTANTS)
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
        shape_idx, orphans = [], []
        for (owner, name), at, size in zip(names, edges[:-1], sizes, strict=True):
            if owner == key and name not in CONSTANTS:
                shape_idx.extend(range(int(at), int(at) + size))
            elif owner != key and owner not in leaf_paths:
                orphans.append(owner)
        if orphans:
            raise ValueError(
                f"free entries {sorted(set(orphans))} are neither orbit coordinates "
                f"of the block {key!r} nor leaves a term proposes "
                "(AbstractConditionalTerm.leaf_path); a completion cannot draw them"
            )
        shape_idx, periods = tuple(shape_idx), fit.periods()
        scale = _data_scale_AU(linear, target) if scale_AU is None else scale_AU
        return cls(
            fit_rest=rest,
            terms=linear,
            scale_AU=jnp.asarray(scale),
            block_key=key,
            shape_idx=shape_idx,
            constants_idx=constants_idx,
            shape_periods=tuple(periods[i] for i in shape_idx),
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
        """Periods of the ``shape`` entries (what a proposal over ``shape`` reads)."""
        return self.shape_periods

    def split(self, z):
        """Return ``(shape, constants)`` of flat vectors ``z (..., d)``.

        The proposed leaves belong to neither.
        """
        idx = jnp.asarray(self.shape_idx), jnp.asarray(self.constants_idx)
        return z[..., idx[0]], z[..., idx[1]]

    def join(self, shape, constants):
        """Flat vectors from ``shape (..., dt)`` and ``constants (..., 4)``.

        The proposed leaves are zero (a valid real value of any leaf law).
        """
        batch = jnp.broadcast_shapes(shape.shape[:-1], constants.shape[:-1])
        z = jnp.zeros((*batch, self.dim))
        z = z.at[..., jnp.asarray(self.shape_idx)].set(shape)
        return z.at[..., jnp.asarray(self.constants_idx)].set(constants)

    def _draw_leaves(self, key, z):
        """Draw the proposed leaves of ``z (n, d)``; return ``(z, log_q)``.

        ``log_q`` ``(n,)`` is the log density of the draws in the real coordinates of
        the flat vector: the density of the leaf value times the volume of the map
        from real to leaf value.
        """
        x, _ = self.fit_rest.from_flat(z)
        models = jax.vmap(self.fit_rest.model_at)(x)
        log_q = jnp.zeros(z.shape[0])
        for i, (path, index, term_index) in enumerate(
            zip(self.leaf_paths, self.leaf_idx, self.leaf_terms, strict=True)
        ):
            dist = self.fit_rest.free[path]
            record = jax.vmap(self.fit_rest.terms[term_index].leaf_conditional)(models)
            low, high = (jnp.log(jnp.maximum(b, 1e-300)) for b in dist.support())
            keys = jax.random.split(jax.random.fold_in(key, i), z.shape[0])
            ln_leaf, log_q_ln = jax.vmap(
                lambda k, r, lo, hi: r.sample(
                    k, lo, hi, widen=self.leaf_widen, uniform=self.leaf_uniform
                ),
                in_axes=(0, 0, None, None),
            )(keys, record, low, high)
            leaf = jnp.exp(ln_leaf)[:, None]
            real = to_real(dist, leaf)
            volume = jnp.sum(from_real(dist, real)[1], axis=-1)
            log_q = log_q + log_q_ln - ln_leaf + volume
            z = z.at[:, index].set(real[:, 0])
        return z, log_q

    def _form(self, shape):
        """The concatenated linear form at the orbit shape ``shape``."""
        z0 = self.join(shape, jnp.asarray(_PLACEHOLDER))
        x, _ = self.fit_rest.from_flat(z0[None])
        model = self.fit_rest.model_at(jax.tree.map(lambda a: a[0], x))
        row = self.block.rows[0]
        forms = [t.linear_form(model, row) for t in self.terms]
        return LinearForm(
            *(jnp.concatenate([getattr(f, n) for f in forms]) for n in _FIELDS)
        )

    def conditional(self, shape):
        """Gaussian conditional of the constants given one ``shape (dt,)``.

        Returns:
            ``(mean (4,), chol (4, 4), log_marginal)``: ``chol`` is the lower Cholesky
            factor of the conditional precision (draw ``c = mean + solve(chol.T, e)``)
            and ``log_marginal`` the density of the data with the constants integrated
            out against ``N(0, scale_AU^2 I)``, in the units of the data.
        """
        form = self._form(shape)
        return linear_gaussian_conditional(
            form.design, form.target, form.precision, 1.0 / self.scale_AU**2
        )

    def _constants(self, shape, key, n_pool):
        """Pool of constants at one ``shape``.

        Returns ``(c, z, log_s, log_z, mean, chol, volume)``. ``log_s`` is the screen:
        the prior of the fit alone (every proposed leaf at its placeholder, no term
        evaluated) plus the volume over the reference Gaussian, ``-inf`` wherever it is
        not finite; ``volume`` is the log volume of each pool member's map.
        """
        mean, chol, log_z = self.conditional(shape)
        eps = jax.random.normal(key, (n_pool, 4))
        c = mean + solve_triangular(chol.T, eps.T, lower=False).T
        z = self.join(shape, c)
        x, volume = self.fit_rest.from_flat(z)
        log_s = self.fit_rest.log_prior(x) + volume - _log_reference(c, self.scale_AU)
        log_s = jnp.where(jnp.isfinite(log_s), log_s, -jnp.inf)
        return c, z, log_s, log_z, mean, chol, volume

    def _one(self, shape, key, n_pool, n_scored):
        """Weight and kept draw of one ``shape``.

        Returns ``(z_kept, lw, log_target_kept, pool_kish)``. ``pool_kish`` is the Kish
        size of the screen over the pool, ``(sum_j s_j)**2 / sum_j s_j**2``, and zero
        when no pool member is inside the prior.

        The pool of ``n_pool`` constants is screened by the prior; ``n_scored`` of
        them, resampled systematically by the screen, get their leaves drawn and the
        terms evaluated (see :class:`CompletedProposal`). With no term left to score
        the screen is the whole density and nothing is resampled.
        """
        k_draw, k_keep = jax.random.split(key)
        c, z, log_s, log_z, mean, chol, volume = self._constants(shape, k_draw, n_pool)
        total = logsumexp(log_s)
        lw = log_z + total - jnp.log(n_pool)
        alive = jnp.isfinite(total)
        pool_kish = jnp.where(
            alive,
            jnp.exp(2.0 * jnp.where(alive, total, 0.0) - logsumexp(2.0 * log_s)),
            0.0,
        )
        if not self.fit_rest.terms:
            keep = jnp.argmax(log_s + jax.random.gumbel(k_keep, (n_pool,)))
            log_form = log_z + _log_conditional(c[keep], mean, chol)
            return z[keep], lw, log_form + log_s[keep] - volume[keep], pool_kish
        normalized = jnp.where(alive, log_s - total, -jnp.log(n_pool))
        idx = systematic_indices(jax.random.fold_in(key, 2), normalized, n_scored)
        c, z = c[idx], z[idx]
        log_q_leaf = jnp.zeros(n_scored)
        if self.leaf_paths:
            z, log_q_leaf = self._draw_leaves(jax.random.fold_in(key, 1), z)
        x, volume = self.fit_rest.from_flat(z)
        target = (
            self.fit_rest.log_density(x) + volume - _log_reference(c, self.scale_AU)
        )
        r = target - log_q_leaf - log_s[idx]
        r = jnp.where(alive, r, -jnp.inf)
        keep = jnp.argmax(r + jax.random.gumbel(k_keep, (n_scored,)))
        lw = jnp.where(alive, lw + logsumexp(r) - jnp.log(n_scored), -jnp.inf)
        log_form = log_z + _log_conditional(c[keep], mean, chol)
        log_target = jnp.where(alive, log_form + target[keep] - volume[keep], -jnp.inf)
        return z[keep], lw, log_target, pool_kish

    def complete(self, key, shapes, n_pool, n_scored):
        """Complete ``shapes (n, dt)``.

        Returns ``(z (n, d), lw (n,), log_target (n,), pool_kish (n,))``. ``lw`` is the
        log marginal weight of each shape; ``log_target`` is ``fit.log_density`` at the
        kept draw, from the form, and ``-inf`` where ``lw`` is; ``pool_kish`` is the
        Kish size of the screen over the pool of each shape (see :meth:`_one`).
        ``n_scored`` is not read when the fit has no term left to score.
        """
        keys = jax.random.split(key, shapes.shape[0])
        return jax.lax.map(
            lambda a: self._one(*a, n_pool, n_scored),
            (shapes, keys),
            batch_size=min(_BATCH, shapes.shape[0]),
        )


def _log_conditional(c, mean, chol):
    """Log density of ``N(mean, P^-1)`` at ``c (4,)``, ``chol`` the factor of ``P``."""
    diff = chol.T @ (c - mean)
    return (
        -0.5 * jnp.sum(diff**2)
        + jnp.sum(jnp.log(jnp.diag(chol)))
        - 2.0 * jnp.log(two_pi)
    )


def _log_reference(c, scale_AU):
    """Log density of ``N(0, scale_AU^2 I)`` at ``c (..., p)``."""
    p = c.shape[-1]
    return -0.5 * jnp.sum(c**2, -1) / scale_AU**2 - p * (
        jnp.log(scale_AU) + 0.5 * jnp.log(two_pi)
    )


def _data_scale_AU(terms, system):
    """Largest projected separation of the valid rows of ``terms`` in AU, NaN if none.

    ``system`` is the block's orbix system. A world with no valid row has no scale, and
    the NaN refuses its slice (under ``vmap`` the other worlds are untouched).
    """
    sep = jnp.zeros(())
    for term in terms:
        data = term.data
        sep = jnp.maximum(
            sep, jnp.max(jnp.where(data.is_valid, jnp.hypot(data.ra, data.dec), 0.0))
        )
    # arcsec times parsec is AU
    return jnp.where(sep > 0, sep * system.frame.dist_pc, jnp.nan)


class CompletedProposal(AbstractProposal):
    """Draw ``shape`` from ``base`` and complete the constants and the proposed leaves.

    A collapsed, or Rao-Blackwellized, proposal: the linear constants are drawn from
    their exact conditional, so the search is over the orbit shape alone.
    At one ``shape``, with ``Z``, ``m`` and ``P`` the marginal and the conditional of
    the linear form and ``g`` the reference Gaussian of the constants:

    1. a pool ``c_1 .. c_M`` (``M = n_pool``) is drawn from ``N(m, P^-1)``;
    2. the screen ``s_j = prior(shape, c_j) * volume / g(c_j)`` is the prior of the fit
       alone, every proposed leaf at its placeholder, and no term is evaluated;
       ``S = sum_j s_j``;
    3. ``K = n_scored`` indices ``I_k`` are resampled systematically on ``s``;
    4. each gets its leaves ``t_k`` from the proposing terms' records
       ``q(t | shape, c_{I_k})`` and the score
       ``r_k = f(shape, c_{I_k}, t_k) / (q(t_k) * s_{I_k})``, with ``f`` the density of
       the fit without its linear terms, times the volume, over ``g``;
    5. the weight is ``W = Z * (S / M) * mean_k r_k`` and one ``k`` is kept with
       probability proportional to ``r_k``.

    Systematic resampling gives pool member ``j`` an expected count ``K s_j / S``, so
    ``E[W | pool] = Z * mean_j F(c_j)`` with ``F(c)`` the integral of ``f`` over the
    leaves: ``W`` is an unbiased estimate of the marginal density of ``shape`` and the
    kept draw with weight ``W`` is properly weighted. The screen and the leaf records
    are proposals, never targets: they change the variance of the weights, never their
    mean, and need only be positive wherever ``f`` is. A ``shape`` whose pool has no
    draw inside the prior gets ``W = 0`` (``-inf``). A NaN score (a NaN term or record
    at a live draw) is an error, not a zero: it reaches ``W``, and the importance
    backend returns a NaN slice. With no term left to score,
    ``r_k = 1``: ``W = Z * S / M`` and the kept draw is picked on ``s``.

    ``weigh`` returns the flat vector of the kept draw, the log weight
    ``log W - log q(shape)``, the fit's density at the kept draw from the form, and
    ``valid``. No density exists at ``z`` (it is not an ``AbstractDensityProposal``):
    ``AdaptiveImportance`` refits on the weights of the fit's own density and the refit
    is over ``shape`` only.

    Pool condition. The screen resamples the pool, so the estimate is an average over
    the pool only while the screen is spread over it. The pool fraction of a draw is
    the Kish size of its screen, ``(sum_j s_j)**2 / sum_j s_j**2``, over ``n_pool``;
    the reading of the cloud is the mean of that fraction under the normalized weights.
    Below one half the screen is a lottery over a few live members, the weights of the
    shapes that carry the posterior are dominated by it, and the importance evidence
    reads low with a standard error that does not show it. ``valid`` is true when the
    reading is at least ``min_pool_fraction``; the backend then publishes an evidence,
    and otherwise returns the draws and weights with a NaN evidence. The cause is a
    pool drawn at the reference scale that does not know a tight constraint on the
    orbit size (a mass law over a few epochs); a larger pool does not remove it.

    Attributes:
        base: Proposal over ``shape`` (it is given the :class:`Completion`, which
            answers ``periods()``, ``block`` and ``model``); a
            :class:`ShapePriorProposal` or a refit of it.
        n_pool: Size of the pool of constants per ``shape``; ``None`` takes 8
            when the fit has no term left to score and 32 otherwise.
        n_scored: Pool members, resampled by the screen, that get leaf draws and term
            evaluations per ``shape``; not read when no term is left to score.
        scale_AU: Scale of the reference Gaussian on the constants; ``None`` takes the
            largest projected separation of the valid rows (see
            :meth:`Completion.from_fit`).
        leaf_widen: Factor on the standard deviation of the leaf proposal components
            (see the module docstring); only used when a term proposes a free leaf.
        leaf_uniform: Share of the leaf's support drawn uniformly.
        min_pool_fraction: Smallest weighted mean pool fraction (see the pool condition
            above) for which ``weigh`` reports its weights valid.
    """

    base: AbstractDensityProposal
    n_pool: int | None = eqx.field(static=True, default=None, kw_only=True)
    n_scored: int = eqx.field(static=True, default=4, kw_only=True)
    scale_AU: jax.Array | None = eqx.field(
        default=None,
        kw_only=True,
        converter=lambda x: None if x is None else jnp.asarray(x),
    )
    leaf_widen: float = eqx.field(static=True, default=1.5, kw_only=True)
    leaf_uniform: float = eqx.field(static=True, default=0.1, kw_only=True)
    min_pool_fraction: float = eqx.field(static=True, default=0.5, kw_only=True)

    def _completion(self, fit):
        return Completion.from_fit(
            fit,
            self.scale_AU,
            leaf_widen=self.leaf_widen,
            leaf_uniform=self.leaf_uniform,
        )

    def weigh(self, key, fit, n, *, batch_size=None):
        """Draw ``n`` completed points; return ``(z, log_weights, log_target, valid)``.

        The completion evaluates in its own batches, so ``batch_size`` is not read.
        ``valid`` is the pool condition of the class docstring; a cloud with no live
        draw has nothing to read and is valid (its evidence is ``-inf``).
        """
        completion = self._completion(fit)
        n_pool = self.n_pool
        if n_pool is None:
            n_pool = 32 if completion.fit_rest.terms else 8
        k_base, k_c = jax.random.split(key)
        shapes, log_q = self.base.sample(k_base, completion, n)
        z, lw, log_target, pool_kish = completion.complete(
            k_c, shapes, n_pool, self.n_scored
        )
        # a world with no valid astrometric row has no scale: refuse its slice
        lw = jnp.where(jnp.isnan(completion.scale_AU), jnp.nan, lw)
        log_w = lw - log_q
        live = jnp.isfinite(log_w)
        weights = jnp.where(
            live, jnp.exp(log_w - jnp.max(log_w, where=live, initial=-jnp.inf)), 0.0
        )
        reading = jnp.sum(weights * pool_kish) / jnp.sum(weights) / n_pool
        valid = jnp.where(jnp.any(live), reading >= self.min_pool_fraction, True)
        return z, log_w, log_target, valid

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive_fraction):
        """Refit the base over the orbit shapes of the weighted cloud."""
        completion = self._completion(fit)
        shapes, _ = completion.split(z)
        base = self.base.adapt(
            key,
            shapes,
            log_weights,
            completion,
            n_centers=n_centers,
            defensive_fraction=defensive_fraction,
        )
        return eqx.tree_at(lambda p: p.base, self, base)
