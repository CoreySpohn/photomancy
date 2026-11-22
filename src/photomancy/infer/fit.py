"""A fit: a base model, the paths that are free, and the data terms.

``Fit(model, free, terms)`` is one normalized log density over named free entries. A
path is a dotted string relative to ``model`` (:mod:`hwoutils.model_paths`); a path that
is absent from ``free`` is held at the value of ``model``. A free entry is one of two
kinds, and ``isinstance`` is the only dispatch:

* a ``priorities`` distribution over one whole array leaf of the model (a *leaf law*),
  written with ``assign`` unchanged;
* an :class:`hwoutils.blocks.AbstractBlock`, which owns a node of the model, carries its
  own unconstrained coordinates and normalized density, and writes the node.

Leaf laws are applied first, then blocks in ``block_order``. Every term is normalized
(the constants that depend on the free values are inside), so
``log_density = log_prior + sum of terms`` is a normalized log density against Lebesgue
measure on the stored coordinates: leaf values for a leaf law, chart coordinates for a
block.

Methods take one draw (leaf shapes) or a batch of draws: a leaf-law entry may carry any
leading batch axes, and a block entry is a dict of arrays whose leading axes are the
batch. ``model_at`` is per draw; ``jax.vmap(fit.model_at)(samples)`` builds models for a
cloud.

A free star mass with no orbit block on a model whose orbits store a mean motion leaves
the orbits at the old mass (a held orbit does not follow the mass). That fit is
inconsistent until the orbits carry no stored rate; free the mass through the block that
owns the orbits.
"""

import hashlib
from abc import abstractmethod
from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.blocks import AbstractBlock, block_order, check_writers
from hwoutils.constants import two_pi
from hwoutils.memory import DEFAULT_MAX_BYTES, MemoryLimit, largest_batch
from hwoutils.model_paths import assign, paths_overlap, resolve
from jax.flatten_util import ravel_pytree
from jax.scipy.special import log_ndtr, logsumexp
from priorities import AbstractDistribution, from_real, to_real

__all__ = [
    "AbstractLinearGaussianTerm",
    "AbstractProposingTerm",
    "AbstractTerm",
    "Fit",
    "LeafProposal",
    "LinearGaussian",
    "observed_hessian",
]


class AbstractTerm(eqx.Module):
    """One data contribution: a normalized log likelihood of held data at a model.

    A term closes over its data (arrays held as fields, so they stay in the pytree) and
    reads the model it is given; it never builds or caches from the model. A term that
    holds a build made from model values (``coronagraphoto.ApertureRecord``) declares
    the model paths the build read in ``reads``, prefixed with the scene's path in
    ``fit.model``, as a property that returns the build's own tuple. A term with no
    build declares ``reads: ClassVar[tuple[str, ...]] = ()``.

    A term that sums over a latent inside itself (a discrete label of the model that
    is not a free entry, such as the line-of-sight sheet of an orbit row) names it in
    ``marginalizes``. Two terms that each sum over one latent would be a product of two
    sums, not the sum of the product, so ``Fit`` refuses them.

    Attributes:
        reads: Model paths a held build cached. ``Fit`` refuses a free path or block
            write that meets one, because the likelihood would no longer see the change.
        marginalizes: Names of the latents the term sums over inside itself; empty by
            default.
    """

    reads: eqx.AbstractVar[tuple[str, ...]]
    marginalizes: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def __call__(self, model) -> jax.Array:
        """Return the scalar log likelihood of the data, constants included."""

    @abstractmethod
    def draw(self, key, model) -> Self:
        """Return a copy of this term holding a data realization drawn at ``model``."""

    def predict(self, model):
        """Return the prediction of ``model`` for this term's data, where it has one."""
        raise NotImplementedError(f"{type(self).__name__} has no prediction")

    def fisher_root(self, model) -> jax.Array:
        """Return a flat vector ``r(model)`` whose Jacobian gives the information.

        With ``J = d r / d theta`` for any parameters ``theta`` of the model, ``J^T J``
        is the expected information (the Fisher matrix) of the term's log likelihood,
        summed over its independent data. The vector is flat, masked entries
        contribute zero, and it does not depend on the observed data.

        Optional. The default raises ``NotImplementedError``: a term whose likelihood is
        an arbitrary callable, one that marginalizes a latent, or a correlated Gaussian
        has no closed-form root here. The rule is per fit, never per term:
        :meth:`Fit.information` with ``kind="fisher"`` raises and names the term when
        any term of the fit lacks a root, because a sum of expected and observed pieces
        is neither matrix.
        """
        raise NotImplementedError(f"{type(self).__name__} has no Fisher root")


class LinearGaussian(eqx.Module):
    """A measurement as a Gaussian in the four sky constants of one orbit row.

    At fixed nonlinear orbit shape the predicted offset of a record is linear in the
    sky constants ``c = (A, B, F, G)`` in AU, ``target_t = design_t @ c + noise_t``,
    in the units of the data: the design is in arcsec per AU, the target in arcsec and
    the precision in arcsec^-2. A density computed from the form is therefore the
    density of the data as the term reports it. Padded records have zero precision.
    Forms concatenate along the record axis.

    Attributes:
        design: Shape ``(T, 2, 4)``.
        target: Observed east/north offsets, shape ``(T, 2)``.
        precision: Inverse noise covariance per record, shape ``(T, 2, 2)``.
    """

    design: jax.Array
    target: jax.Array
    precision: jax.Array


class AbstractLinearGaussianTerm(AbstractTerm):
    """A term whose likelihood is exactly Gaussian in the sky constants of one planet.

    The capability is the type: a consumer that completes or marginalizes the constants
    selects terms with ``isinstance``. ``linear_gaussian`` is exact only where the
    term's records depend on the constants through the planet's own orbit; a term
    refuses the models for which that fails.
    """

    @abstractmethod
    def linear_gaussian(self, model, planet: int = 0) -> LinearGaussian:
        """The term as a Gaussian in the sky constants of ``planet`` of ``model``."""


class LeafProposal(eqx.Module):
    """A proposal density for one free model leaf, as a mixture of Gaussians in its log.

    A record of what a term can say about a leaf given the rest of the model: component
    ``k`` is a Gaussian of ``ln(leaf)`` with mean ``mean[k]`` and standard deviation
    ``sd[k]``, and the components carry log masses ``log_pi`` (not normalized; all
    ``-inf`` means the term has no peak to offer). ``low`` and ``high`` bound
    ``ln(leaf)`` where the term itself knows a bound (``-inf`` and ``inf`` otherwise);
    the proposal also stays inside the support of the leaf's own law.

    **This record is a proposal density and never a target.** It describes an
    approximation of the conditional of the leaf, built from fixed variances or a frozen
    nuisance. Only the exact term enters a fit density, and a consumer corrects a draw
    from this record by the exact density over the density of the draw.

    Attributes:
        path: Model path of the positive leaf this record is about.
        mean: Component means of ``ln(leaf)``, shape ``(K,)``.
        sd: Component standard deviations, shape ``(K,)``.
        log_pi: Log mass of each component, shape ``(K,)``.
        low: Lower bound of ``ln(leaf)`` the term knows, scalar.
        high: Upper bound of ``ln(leaf)`` the term knows, scalar.
    """

    path: str = eqx.field(static=True)
    mean: jax.Array
    sd: jax.Array
    log_pi: jax.Array
    low: jax.Array
    high: jax.Array

    def sample(self, key, low, high, *, widen, uniform):
        """One draw of ``ln(leaf)`` from the widened record: ``(ln_leaf, log_q)``.

        Component ``k`` is ``Normal(mean_k, (widen * sd_k)^2)`` truncated to
        ``[low, high]`` (the given bounds, those of the leaf's law, intersected with
        the record's own) and to six widened standard deviations about its mean, so its
        tail draws stay accurate (a component with no overlap is dropped), with weight
        proportional to ``exp(log_pi_k)`` times the mass the unwidened Gaussian has
        inside. The uniform share ``uniform`` of ``[low, high]`` is the rest, and all
        of it when no component has weight. ``log_q`` is the exact log density of the
        draw in ``ln(leaf)``; it is at least ``log(uniform) - log(high - low)`` for
        every draw inside the range. One record, unbatched; ``jax.vmap`` it for many.
        """
        low = jnp.maximum(low, self.low)
        high = jnp.minimum(high, self.high)
        scale = widen * self.sd
        win_lo = jnp.maximum(low, self.mean - _WINDOW * scale)
        win_hi = jnp.minimum(high, self.mean + _WINDOW * scale)
        reach = win_hi > win_lo
        win_lo, win_hi = jnp.where(reach, win_lo, 0.0), jnp.where(reach, win_hi, 1.0)
        inside = _log_normal_mass(
            (low - self.mean) / self.sd, (high - self.mean) / self.sd
        )
        log_w = jnp.where(reach, self.log_pi + inside, -jnp.inf)
        peak = jnp.isfinite(jnp.max(log_w))
        share = jnp.where(peak, uniform, 1.0)
        log_w = jnp.where(
            peak,
            log_w - logsumexp(jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)),
            -jnp.inf,
        )
        weight = jnp.exp(log_w) * (1.0 - share)
        lo_std, hi_std = (win_lo - self.mean) / scale, (win_hi - self.mean) / scale
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
        draw_n = self.mean[comp] + scale[comp] * z
        draw_u = low + (high - low) * jax.random.uniform(k_flat)
        ln_leaf = jnp.clip(jnp.where(flat, draw_u, draw_n), low, high)
        log_mass = _log_normal_mass(lo_std, hi_std)
        z_at = (ln_leaf - self.mean) / scale
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


# A leaf component reaches this many widened standard deviations.
_WINDOW = 6.0


def _log_normal_mass(a, b):
    """``log(Phi(b) - Phi(a))`` for ``a <= b``, from the tail with the smaller mass."""
    upper = a > 0
    lo, hi = jnp.where(upper, -b, a), jnp.where(upper, -a, b)
    log_hi = log_ndtr(hi)
    return log_hi + jnp.log1p(-jnp.exp(log_ndtr(lo) - log_hi))


class AbstractProposingTerm(AbstractTerm):
    """A term that can propose one of the leaves it reads from its own data.

    The capability is the type: a proposal that completes leaves selects terms with
    ``isinstance``. ``leaf_proposal`` is approximate by contract; see
    :class:`LeafProposal`. It never enters a target.

    Attributes:
        leaf_path: Path of the positive model leaf the term offers a conditional for,
            or ``None`` when it offers none.
    """

    leaf_path: eqx.AbstractVar[str | None]

    @abstractmethod
    def leaf_proposal(self, model) -> LeafProposal:
        """The proposal record of ``leaf_path`` given everything else in ``model``.

        The value of the leaf in ``model`` is read only to scale the prediction; the
        record does not depend on it.
        """


def _sum_leaf(value, leaf_ndim, dist):
    """Sum an elementwise log density over the leaf axes it keeps."""
    n = leaf_ndim - dist.event_dim
    return jnp.sum(value, axis=tuple(range(-n, 0))) if n > 0 else value


def _admit_information(n_items, per_item, fixed, batch_size, max_bytes):
    """Refuse an information matrix whose closed-form estimate exceeds ``max_bytes``."""
    n_items = max(n_items, 1)
    try:
        fits = largest_batch(n_items, per_item, max_bytes, fixed_bytes=fixed)
    except MemoryLimit as err:
        raise MemoryLimit(
            f"{err}. A fit this large needs the product v -> J^T (J v) built with "
            "jax.linearize and jax.linear_transpose in place of the dense matrix"
        ) from err
    batch = n_items if batch_size is None else min(batch_size, n_items)
    if batch > fits:
        raise MemoryLimit(
            f"information with batch_size={batch_size} needs an estimated "
            f"{fixed + batch * per_item} bytes, above max_bytes={max_bytes}; "
            f"batch_size={fits} fits (or raise max_bytes, or pass max_bytes=None)"
        )


def _to_cell(block, coordinates):
    """Reduce every periodic coordinate of a block entry to its cell ``[0, period)``."""
    period = dict(zip(block.coordinate_names, block.periods, strict=True))
    return {
        name: value if period[name] is None else jnp.mod(value, period[name])
        for name, value in coordinates.items()
    }


def _mapped(fn, items, batch_size):
    """Map ``fn`` over the leading axis of ``items``.

    ``batch_size=None`` runs every item at once (``jax.vmap``), ``1`` runs them in
    sequence, and any other size runs that many at a time (``jax.lax.map``).
    """
    if batch_size is None:
        return jax.vmap(fn)(items)
    if batch_size == 1:
        return jax.lax.map(fn, items)
    return jax.lax.map(fn, items, batch_size=batch_size)


def observed_hessian(f, x, *, batch_size: int | None = None) -> jax.Array:
    """Symmetrized Hessian of a scalar ``f`` at the flat vector ``x``.

    Column ``j`` is one Hessian-vector product (forward over reverse) along ``e_j``,
    so only the tangent columns in flight are held: ``batch_size=1`` runs the columns
    in sequence and holds one gradient's intermediates (on a CPU a wider batch buys at
    most 20 to 25 percent and costs memory in proportion), ``None`` runs all at once
    (what :func:`jax.hessian` does, the fast choice on a GPU with the memory for it).

    Args:
        f: ``x -> scalar``.
        x: Point of shape ``(n,)``.
        batch_size: Columns per batch, or ``None`` for all at once.

    Returns:
        The ``(n, n)`` matrix ``0.5 (H + H^T)``.
    """
    grad_fn = jax.grad(f)

    def column(v):
        return jax.jvp(grad_fn, (x,), (v,))[1]

    hess = _mapped(column, jnp.eye(x.shape[0], dtype=x.dtype), batch_size)
    return 0.5 * (hess + hess.T)


@eqx.filter_jit
def _observed_information(fit, x, include_prior, batch_size):
    flat, unravel = ravel_pytree(x)

    def negative(theta):
        entries = unravel(theta)
        total = jnp.asarray(fit.log_likelihood(entries), jnp.float64)
        if include_prior:
            total = total + fit.log_prior(entries)
        return -total

    return observed_hessian(negative, flat, batch_size=batch_size)


@eqx.filter_jit
def _fisher_information(fit, x, include_prior, batch_size, forward, n_root):
    flat, unravel = ravel_pytree(x)

    def root(theta):
        return fit._root_vector(unravel(theta))

    if n_root == 0:
        info = jnp.zeros((flat.size, flat.size), flat.dtype)
    elif forward:
        columns = _mapped(
            lambda v: jax.jvp(root, (flat,), (v,))[1],
            jnp.eye(flat.size, dtype=flat.dtype),
            batch_size,
        )
        info = columns @ columns.T
    else:
        _, pullback = jax.vjp(root, flat)
        rows = _mapped(
            lambda c: pullback(c)[0], jnp.eye(n_root, dtype=flat.dtype), batch_size
        )
        info = rows.T @ rows
    if include_prior:

        def negative_prior(theta):
            return -fit.log_prior(unravel(theta))

        info = info + observed_hessian(negative_prior, flat, batch_size=batch_size)
    return 0.5 * (info + info.T)


class Fit(eqx.Module):
    """A normalized log density over the free entries of a model.

    Attributes:
        model: The base model, every leaf an array. A held path keeps its value here.
        free: Path to leaf law (a ``priorities`` distribution over the whole leaf) or
            to a block (keyed by the path it owns). Keys are static.
        terms: Data terms, each a normalized log likelihood.
    """

    model: eqx.Module
    free: dict[str, AbstractDistribution | AbstractBlock]
    terms: tuple[AbstractTerm, ...] = ()

    def __check_init__(self):
        """Refuse an inconsistent fit on the host.

        Raises:
            RuntimeError: When ``jax_enable_x64`` is off.
            TypeError: For an entry that is neither a distribution nor a block, or a
                term that is not an :class:`AbstractTerm`.
            ValueError: For an unknown path, a leaf law that does not fit its leaf, two
                writers of one path, a block cycle, a free path or block write that
                meets a term's ``reads``, or two terms that marginalize the same latent
                (``AbstractTerm.marginalizes``), named in the message.
        """
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "Fit needs jax_enable_x64 (float64 log densities); "
                "call jax.config.update('jax_enable_x64', True) first"
            )
        leaf_paths, blocks = [], {}
        for path, entry in self.free.items():
            if isinstance(entry, AbstractDistribution):
                self._check_leaf(path, entry)
                leaf_paths.append(path)
            elif isinstance(entry, AbstractBlock):
                blocks[path] = entry
            else:
                raise TypeError(
                    f"free[{path!r}] is a {type(entry).__name__}; "
                    "expected a priorities distribution or a block"
                )
        check_writers(blocks, leaf_paths)
        block_order(blocks)
        for key, block in blocks.items():
            for path in (*block.reads, *block.writes):
                self._require_path(path, f"block {key!r}")
        written = [*leaf_paths, *(w for b in blocks.values() for w in b.writes)]
        summed = {}
        for i, term in enumerate(self.terms):
            if not isinstance(term, AbstractTerm):
                raise TypeError(
                    f"terms[{i}] is a {type(term).__name__}; expected an AbstractTerm"
                )
            for latent in term.marginalizes:
                if latent in summed:
                    raise ValueError(
                        f"terms[{summed[latent]}] and terms[{i}] both marginalize "
                        f"{latent!r}; hold the terms that see it in one sum"
                    )
                summed[latent] = i
            for read in term.reads:
                self._require_path(read, f"{type(term).__name__}.reads")
                for path in written:
                    if paths_overlap(path, read):
                        raise ValueError(
                            f"free path {path!r} meets {type(term).__name__}.reads "
                            f"entry {read!r}: the term holds a build that cached it"
                        )

    def _require_path(self, path, who):
        try:
            return resolve(self.model, path)
        except (AttributeError, IndexError, TypeError, ValueError) as err:
            raise ValueError(f"unknown model path {path!r} (from {who})") from err

    def _check_leaf(self, path, dist):
        leaf = self._require_path(path, "free")
        if not eqx.is_array(leaf):
            raise ValueError(
                f"free path {path!r} names a {type(leaf).__name__}, not one array "
                "leaf; free the node through a block"
            )
        support = dist.support()[0].shape
        shape = jnp.shape(leaf)
        if support != shape and not (support == () and dist.event_dim == 0):
            raise ValueError(
                f"law at {path!r} has parameter shape {support}, "
                f"but the leaf has shape {shape}"
            )

    # ----- entries ---------------------------------------------------------------

    @property
    def leaf_laws(self) -> dict[str, AbstractDistribution]:
        """The free entries that are leaf laws."""
        return {
            k: v for k, v in self.free.items() if isinstance(v, AbstractDistribution)
        }

    @property
    def blocks(self) -> dict[str, AbstractBlock]:
        """The free entries that are blocks."""
        return {k: v for k, v in self.free.items() if isinstance(v, AbstractBlock)}

    @property
    def block_order(self) -> tuple[str, ...]:
        """Block keys in the order they are applied."""
        return block_order(self.blocks)

    def _batch_ndim(self, x):
        """Number of leading batch axes of an entry dict (all entries agree)."""
        for key, entry in self.free.items():
            if isinstance(entry, AbstractBlock):
                return jnp.ndim(next(iter(x[key].values()))) - 1
            return jnp.ndim(x[key]) - jnp.ndim(resolve(self.model, key))
        return 0

    def _batch_shape(self, x):
        """Shape of the leading batch axes of an entry dict."""
        for key, entry in self.free.items():
            if isinstance(entry, AbstractBlock):
                return jnp.shape(next(iter(x[key].values())))[:-1]
            return jnp.shape(x[key])[
                : jnp.ndim(x[key]) - jnp.ndim(resolve(self.model, key))
            ]
        return ()

    def _over_batch(self, fn, x, batch_size=None):
        """Map ``fn`` over the batch axes of ``x``.

        With ``batch_size`` the outermost axis runs as a ``jax.lax.map`` of that chunk
        size (memory bounded by the chunk), the inner axes as ``vmap``.
        """
        ndim = self._batch_ndim(x)
        if batch_size is None or ndim == 0:
            for _ in range(ndim):
                fn = jax.vmap(fn)
            return fn(x)
        for _ in range(ndim - 1):
            fn = jax.vmap(fn)
        return jax.lax.map(fn, x, batch_size=batch_size)

    # ----- the model at a draw ---------------------------------------------------

    def _assign_leaf_laws(self, x, model):
        for path in self.leaf_laws:
            model = assign(model, path, x[path])
        return model

    def _walk(self, x):
        """Walk the block chain once: return the model of ``x`` and its log prior.

        Leaf laws are assigned first and scored, then each block in order scores its
        ``log_prob`` on the running model before its writes are applied.
        """
        model = self._assign_leaf_laws(x, self.model)
        total = jnp.zeros(())
        for path, dist in self.leaf_laws.items():
            ndim = jnp.ndim(resolve(self.model, path))
            total = total + _sum_leaf(dist.log_prob(x[path]), ndim, dist)
        blocks = self.blocks
        for key in self.block_order:
            block = blocks[key]
            total = total + block.log_prob(x[key], model)
            for path, value in block.forward(x[key], model).items():
                model = assign(model, path, value)
        return model, total

    def model_at(self, x: dict) -> eqx.Module:
        """Return the model of one draw ``x``: leaf laws first, then blocks in order."""
        return self._walk(x)[0]

    # ----- density ---------------------------------------------------------------

    def _log_prior_one(self, x):
        return self._walk(x)[1]

    def log_prior(self, x: dict, *, batch_size: int | None = None) -> jax.Array:
        """Return the normalized log prior of ``x`` against Lebesgue ``dx``.

        Leaf laws are summed over the axes of their leaf. Blocks follow on the running
        model, each ``log_prob`` read before that block's writes, so the blocks form a
        chain of normalized conditionals. ``batch_size`` evaluates that many draws at
        once (a ``jax.lax.map`` chunk over the leading axis), the same values to
        rounding; the default maps every draw at once with ``jax.vmap``.
        """
        return self._over_batch(self._log_prior_one, x, batch_size)

    @staticmethod
    def _terms_at(terms, model):
        total = jnp.zeros(())
        for term in terms:
            total = total + term(model)
        return total

    def _log_likelihood_one(self, x):
        return self._terms_at(self.terms, self.model_at(x))

    def log_likelihood(self, x: dict, *, batch_size: int | None = None) -> jax.Array:
        """Return the sum of the terms at ``x`` (``batch_size`` as in ``log_prior``)."""
        return self._over_batch(self._log_likelihood_one, x, batch_size)

    def _log_density_one(self, x):
        model, prior = self._walk(x)
        total = prior + self._terms_at(self.terms, model)
        return jnp.where(jnp.isneginf(prior), -jnp.inf, total)

    def log_density(self, x: dict, *, batch_size: int | None = None) -> jax.Array:
        """Return ``log_prior + log_likelihood`` (``-inf`` off the prior's support).

        ``batch_size`` is as in :meth:`log_prior`.
        """
        return self._over_batch(self._log_density_one, x, batch_size)

    # ----- curvature -------------------------------------------------------------

    def _root_vector(self, x):
        """Concatenated float64 Fisher roots of every term at the draw ``x``."""
        model = self.model_at(x)
        parts = [
            jnp.ravel(term.fisher_root(model)).astype(jnp.float64)
            for term in self.terms
        ]
        return jnp.concatenate(parts) if parts else jnp.zeros((0,))

    def _root_length(self):
        """Length of :meth:`_root_vector`, from shapes only.

        Raises:
            NotImplementedError: Naming the first term that has no Fisher root.
        """
        total = 0
        for i, term in enumerate(self.terms):
            try:
                shape = eqx.filter_eval_shape(term.fisher_root, self.model)
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"terms[{i}] ({type(term).__name__}) has no Fisher root, so "
                    "kind='fisher' is unavailable for this fit (all terms or none); "
                    f"use kind='observed'. Reason: {err}"
                ) from err
            total += int(np.prod(shape.shape, dtype=int))
        return total

    def information(
        self,
        x: dict,
        *,
        kind: str,
        include_prior: bool = True,
        batch_size: int | None = None,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> jax.Array:
        """Curvature of the negative log density at ``x``, in the stored coordinates.

        The matrix is ``(P, P)`` over the flat vector of ``x`` in the order of
        :func:`jax.flatten_util.ravel_pytree` (sorted entry keys; each entry row-major),
        in the coordinates ``x`` is written in (leaf values for a leaf law, chart
        coordinates for a block), not the unconstrained ``to_real`` ones. Two matrices,
        each with its own job:

        * ``kind="observed"``: the exact Hessian of the negative log density, built
          column by column. It is the reported width at a converged maximum, the matrix
          of the Laplace evidence, and the check that a point is a regular maximum: it
          is not positive definite away from one, and that is information.
        * ``kind="fisher"``: ``J^T J`` of the terms' :meth:`AbstractTerm.fisher_root`,
          the expected curvature of the likelihood at ``x`` (not a function of the
          data). It is positive semidefinite, so it cannot flag a bad point; use it to
          scale an optimizer, for a forecast, or for a width before data exist. Every
          term must have a root, else ``NotImplementedError`` names the first that has
          none. The Jacobian is built in forward mode when the root is longer than the
          parameter vector (an image) and in reverse mode when it is shorter (a few
          astrometric points).

        ``include_prior=True`` adds the Hessian of the negative log prior in the same
        coordinates. The sum is then not positive semidefinite in general (a chart
        volume or a logit map has indefinite curvature); pass ``False`` for the
        likelihood alone.

        Memory. The resident part is estimated in closed form before anything is
        built: the ``P`` by ``P`` result and, for ``"fisher"``, the tangent matrix
        (``P`` times the root length times 8 bytes); a batch adds one tangent of the
        root (or one gradient-sized column of length ``P``) per item. That is a lower
        bound on the intermediates of the model's own render, which the estimate does
        not see. A request above ``max_bytes`` raises
        :class:`hwoutils.memory.MemoryLimit` and names the largest batch that fits. A
        fit with too many parameters for even one item (tens of thousands of disk
        pixel fluxes) is refused, and the way forward is the product ``v -> J^T (J v)``
        built with :func:`jax.linearize` and :func:`jax.linear_transpose`, as
        ``tiptilt.MatrixFreeEFCController`` does.

        Args:
            x: One draw of entries (``Fit.log_density`` format, no batch axes).
            kind: ``"observed"`` or ``"fisher"``.
            include_prior: Add the Hessian of the negative log prior.
            batch_size: Columns (or rows, in reverse mode) per batch. ``None`` runs all
                at once (the fast choice on a GPU), ``1`` one at a time (the choice
                on a CPU with a large image).
            max_bytes: Memory limit of the estimate above, or ``None`` for none.

        Returns:
            The symmetric ``(P, P)`` float64 matrix.

        Raises:
            RuntimeError: When ``jax_enable_x64`` is off (a float32 curvature is too
                ill conditioned to invert).
            ValueError: For an unknown ``kind``.
            NotImplementedError: For ``kind="fisher"`` when a term has no root.
            MemoryLimit: When the estimate is above ``max_bytes``.
        """
        if kind not in ("observed", "fisher"):
            raise ValueError(f"kind must be 'observed' or 'fisher', got {kind!r}")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("information needs jax_enable_x64 (float64 curvature)")
        x = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), x)
        n_param = sum(int(np.size(a)) for a in jax.tree.leaves(x))
        if kind == "observed":
            _admit_information(
                n_param, n_param * 8, 2 * n_param * n_param * 8, batch_size, max_bytes
            )
            return _observed_information(self, x, include_prior, batch_size)
        n_root = self._root_length()
        forward = n_root > n_param
        _admit_information(
            n_param if forward else n_root,
            (n_root if forward else n_param) * 8,
            (n_param * n_param + n_param * n_root) * 8,
            batch_size,
            max_bytes,
        )
        return _fisher_information(self, x, include_prior, batch_size, forward, n_root)

    # ----- unconstrained coordinates ---------------------------------------------

    def to_real(self, x: dict) -> dict:
        """Map ``x`` to unconstrained coordinates.

        A leaf law uses :func:`priorities.to_real`; a block entry is already
        unconstrained and passes through unchanged.
        """
        return {
            key: to_real(entry, x[key])
            if isinstance(entry, AbstractDistribution)
            else x[key]
            for key, entry in self.free.items()
        }

    def from_real(self, u: dict) -> tuple[dict, jax.Array]:
        """Map unconstrained coordinates to ``x`` and the log volume of the map.

        The volume sums the leaf laws' ``priorities.from_real`` volumes over the leaf
        axes (zero for a block entry) and has the batch shape of ``u``. A block
        coordinate with a period is reduced to its cell ``[0, period)``, so stored
        angles never leave one cell; the block density is periodic, so the reduction
        changes no density.
        """
        x, volume = {}, jnp.zeros(())
        for key, entry in self.free.items():
            if isinstance(entry, AbstractDistribution):
                x[key], vol = from_real(entry, u[key])
                ndim = jnp.ndim(resolve(self.model, key))
                volume = volume + _sum_leaf(vol, ndim, entry)
            else:
                x[key] = _to_cell(entry, u[key])
        return x, jnp.broadcast_to(volume, self._batch_shape(u))

    def flat_layout(self) -> tuple[tuple[tuple[str, str | None], ...], tuple[int, ...]]:
        """Names and sizes of the segments of the flat unconstrained vector, in order.

        The flat vector is the ``ravel_pytree`` of one draw of :meth:`to_real` (sorted
        entry keys; inside a block, sorted coordinate names). Segment ``i`` is the
        entry ``names[i] = (key, coordinate name)`` (the name is ``None`` for a leaf-law
        entry) and has ``sizes[i]`` elements. Both tuples are static (host values).

        Returns:
            ``(names, sizes)``.
        """
        shapes = eqx.filter_eval_shape(
            lambda k: self.to_real(self.sample_prior(k)), jax.random.key(0)
        )
        names, sizes = [], []
        for key in sorted(self.free):
            if isinstance(self.free[key], AbstractDistribution):
                names.append((key, None))
                sizes.append(int(np.prod(shapes[key].shape, dtype=int)))
                continue
            for name in sorted(shapes[key]):
                names.append((key, name))
                sizes.append(int(np.prod(shapes[key][name].shape, dtype=int)))
        return tuple(names), tuple(sizes)

    def periods(self) -> tuple[float | None, ...]:
        """Period of every entry of the flat unconstrained vector, ``None`` if none.

        The order is that of :meth:`flat_layout`, so the tuple is assembled per block
        through its ``coordinate_names`` and never by position. A leaf-law entry is
        never periodic. The tuple is static (host values) and has the length of the
        flat vector.
        """
        names, sizes = self.flat_layout()
        out: list[float | None] = []
        for (key, name), size in zip(names, sizes, strict=True):
            if name is None:
                out += [None] * size
            else:
                entry = self.free[key]
                period = dict(zip(entry.coordinate_names, entry.periods, strict=True))
                out += [period[name]] * size
        return tuple(out)

    def unravel(self, z) -> dict:
        """Unconstrained entries of flat vectors ``z (..., d)``.

        The flat vector is the ``ravel_pytree`` of one draw of :meth:`to_real` (see
        :meth:`flat_layout`); the result is an argument of :meth:`from_real` with the
        batch shape of ``z``.
        """
        shapes = eqx.filter_eval_shape(
            lambda k: self.to_real(self.sample_prior(k)), jax.random.key(0)
        )
        template = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)
        unravel = ravel_pytree(template)[1]
        z = jnp.asarray(z)
        u = jax.vmap(unravel)(z.reshape(-1, z.shape[-1]))
        return jax.tree.map(lambda a: a.reshape(*z.shape[:-1], *a.shape[1:]), u)

    def to_flat(self, x: dict) -> jax.Array:
        """Flat unconstrained vectors ``z (..., d)`` of stored entries ``x``.

        The inverse of :meth:`from_flat`: :meth:`to_real` of every draw, with each
        periodic block coordinate reduced to its cell ``[0, period)``, raveled in the
        order :meth:`unravel` reads (see :meth:`flat_layout`). ``x`` may carry leading
        batch axes, which come before the flat axis of the result. It places a cloud
        (``posterior.samples``) in the coordinates of a proposal's ``adapt`` or of
        ``MixtureProposal.from_cloud``.
        """
        batch = self._batch_shape(x)
        n = int(np.prod(batch, dtype=int))
        one_axis = jax.tree.map(lambda a: a.reshape(n, *a.shape[len(batch) :]), x)

        def flat_one(one):
            u = self.to_real(one)
            for key, entry in self.free.items():
                if not isinstance(entry, AbstractDistribution):
                    u[key] = _to_cell(entry, u[key])
            return ravel_pytree(u)[0]

        z = jax.vmap(flat_one)(one_axis)
        return z.reshape(*batch, z.shape[-1])

    def from_flat(self, z) -> tuple[dict, jax.Array]:
        """Stored entries and the log volume of the map from flat vectors ``z``.

        ``from_real(unravel(z))``: the volume is that of the map from the flat
        unconstrained vector, so ``log_density(x) + volume`` is the density in ``z``.
        """
        return self.from_real(self.unravel(z))

    def real_log_prob(self, u: dict) -> jax.Array:
        """Return the density in ``u``: the density of ``from_real(u)`` plus volume."""
        x, volume = self.from_real(u)
        return self.log_density(x) + volume

    # ----- prior draws -----------------------------------------------------------

    def _sample_prior_one(self, key):
        keys = jax.random.split(key, len(self.free))
        subkeys = dict(zip(sorted(self.free), keys, strict=True))
        x = {}
        for path, dist in self.leaf_laws.items():
            shape = jnp.shape(resolve(self.model, path))
            own = dist.support()[0].shape
            x[path] = dist.sample(subkeys[path], shape if own == () else ())
        model = self._assign_leaf_laws(x, self.model)
        blocks = self.blocks
        for bkey in self.block_order:
            block = blocks[bkey]
            x[bkey] = block.sample(subkeys[bkey], model)
            for path, value in block.forward(x[bkey], model).items():
                model = assign(model, path, value)
        return x

    def sample_prior(self, key, sample_shape: tuple[int, ...] = ()) -> dict:
        """Draw ``x`` from the prior, leaf laws then blocks on the running model.

        Returns entries shaped ``(*sample_shape, *leaf_shape)`` (a block entry is a
        dict of arrays shaped ``(*sample_shape, R)``).
        """
        n = int(np.prod(sample_shape, dtype=int))
        flat = jax.vmap(self._sample_prior_one)(jax.random.split(key, n))
        return jax.tree.map(lambda a: a.reshape(*sample_shape, *a.shape[1:]), flat)

    # ----- unit cube -------------------------------------------------------------

    def unit_template(self) -> dict:
        """Zeros of one unit-cube draw.

        A leaf-law entry has the shape of its model leaf; a block entry has shape
        ``(unit_size,)``, the length of the cube vector its ``from_unit`` consumes.

        Raises:
            ValueError: For a block with no unit-cube map (``unit_size`` is ``None``) or
                a leaf law with no inverse CDF (a mixture).
        """
        out = {}
        for key, entry in self.free.items():
            if isinstance(entry, AbstractBlock):
                if entry.unit_size is None:
                    raise ValueError(
                        f"block {key!r} ({type(entry).__name__}) has no unit-cube map"
                    )
                out[key] = jnp.zeros((entry.unit_size,))
                continue
            if not entry.has_icdf:
                raise ValueError(
                    f"law at {key!r} ({type(entry).__name__}) has no inverse CDF"
                )
            out[key] = jnp.zeros(jnp.shape(resolve(self.model, key)))
        return out

    def from_unit(self, u: dict) -> dict:
        """Map unit-cube coordinates to entries.

        Leaf laws map through their inverse CDF and are written first; each block then
        maps its cube vector through ``from_unit`` on the running model, in block order,
        so a block that reads what an earlier block writes (a star mass) sees it.

        Args:
            u: One draw, each entry shaped like :meth:`unit_template`, values in
                ``(0, 1)``.
        """
        self.unit_template()
        x = {key: entry.icdf(u[key]) for key, entry in self.leaf_laws.items()}
        model = self._assign_leaf_laws(x, self.model)
        blocks = self.blocks
        for key in self.block_order:
            x[key] = blocks[key].from_unit(u[key], model)
            for path, value in blocks[key].forward(x[key], model).items():
                model = assign(model, path, value)
        return {key: x[key] for key in self.free}

    # ----- edits -----------------------------------------------------------------

    def with_priors(self, priors: dict) -> "Fit":
        """Return a fit with some entries replaced, the free set unchanged.

        A leaf law is replaced by a leaf law, and a block by a block of the same layout
        (``old.same_layout(new)``: the same type, ownership, chart and rows, only the
        law differs). A new law may move the bounds that the block's map carries, so a
        coordinate vector made for the old fit may mean something else under the new
        one; whether stored draws survive is for the caller to decide
        (:meth:`Posterior.with_priors` does, by ``same_map``).

        Raises:
            ValueError: For a key outside the free set, a replacement of the other
                kind, or a block of another layout.
        """
        extra = set(priors) - set(self.free)
        if extra:
            raise ValueError(
                f"with_priors cannot change the free set; not free: {sorted(extra)}"
            )
        for key, new in priors.items():
            old = self.free[key]
            if isinstance(old, AbstractBlock):
                if not isinstance(new, AbstractBlock) or not old.same_layout(new):
                    raise ValueError(
                        f"block {key!r} can only be replaced by a block of the same "
                        "layout (type, ownership, chart and rows)"
                    )
            elif not isinstance(new, AbstractDistribution):
                raise ValueError(f"leaf law {key!r} needs a priorities distribution")
        return Fit(self.model, {**self.free, **priors}, self.terms)

    def append_term(self, term: AbstractTerm) -> "Fit":
        """Return a fit with one more term; the free set is unchanged."""
        return Fit(self.model, self.free, (*self.terms, term))

    def replace_term(self, index: int, term: AbstractTerm) -> "Fit":
        """Return a fit with ``terms[index]`` replaced; the free set is unchanged.

        The replacement may be any term whose reads pass the checks of ``Fit``. A stack
        of worlds that must not recompile needs a replacement with the same pytree
        structure as the old term (a padded row written in place).

        Raises:
            IndexError: When ``index`` is outside ``range(len(terms))`` (negative
                indices are refused, so a stacked fit cannot shift a slot).
        """
        if not 0 <= index < len(self.terms):
            raise IndexError(
                f"replace_term index {index} outside range({len(self.terms)})"
            )
        terms = (*self.terms[:index], term, *self.terms[index + 1 :])
        return Fit(self.model, self.free, terms)

    def digest(self) -> str:
        """Hash of the free keys, prior records, term kinds and model structure.

        Evaluated on the host from concrete arrays. Equal for fits that share a
        density, so a stored result can be checked against the fit it targets.
        """
        h = hashlib.sha256()

        def feed(tag, tree):
            h.update(f"{tag}:{jax.tree.structure(tree)}".encode())
            for leaf in jax.tree.leaves(tree):
                arr = np.asarray(leaf)
                h.update(f"{arr.dtype}{arr.shape}".encode())
                h.update(arr.tobytes())

        for key in sorted(self.free):
            entry = self.free[key]
            h.update(
                f"{key}={type(entry).__module__}.{type(entry).__qualname__}".encode()
            )
            feed("entry", entry)
        for term in self.terms:
            h.update(f"term={type(term).__module__}.{type(term).__qualname__}".encode())
        h.update(str(jax.tree.structure(self.model)).encode())
        return h.hexdigest()
