"""A fit: a base model, the paths that are free, and the data terms.

``Fit(model, free, terms)`` is one normalized log density over named free entries. A
path is a dotted string relative to ``model`` (:mod:`hwoutils.model_paths`); a path that
is absent from ``free`` is held at the value of ``model``. A free entry is one of two
kinds, and ``isinstance`` is the only dispatch:

* a ``priorities`` distribution over one whole array leaf of the model (a *leaf law*),
  written at the dtype of the destination leaf;
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

A model node may declare fields derived from others (``tied_fields``, see
:func:`hwoutils.blocks.check_ties`), as the orbix ``System`` does for its stored orbits,
which its star and planet masses rate. ``Fit`` refuses an entry inside a derived field
(a leaf law on one field of a stored orbit) and an entry that writes a source of a
derived field while no entry that reads or writes that source also writes the field (a
free mass with no orbit block that reads it).

A block may have walls (:attr:`hwoutils.blocks.AbstractBlock.walls`): bounds of its law
that its coordinates do not map, so the density is zero on part of the coordinates.
:attr:`Fit.walls` reads them from the blocks when the laws are concrete; a gradient
sampler refuses such a fit. Under a transformation the law parameters are traced and no
edge can be read, so ``Fit`` also stores the walls it finds when it is built on the
host, as static data, and a stacked or jitted fit reports those. The edits
(``with_priors``, ``append_term``, ``replace_term``) store the walls of the entries they
keep; a fit built anew inside a transformation from traced laws counts an unmapped edge
whose value is traced as a wall. Gradient sampler admission uses the current
predicates of blocks that expose ``wall_flags()``, including traced edits. A leaf law
is assumed to have
none: every ``bounds`` kind of ``priorities.to_real`` maps onto the real line, and a law
whose support has an interior gap (a mixture of disjoint bounded components) is not
detected.
"""

import hashlib
from abc import abstractmethod
from dataclasses import fields
from typing import ClassVar, NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.blocks import AbstractBlock, block_order, check_ties, check_writers
from hwoutils.constants import two_pi
from hwoutils.leaves import (
    as_float_leaf,
    canonical,
    is_foreign,
    is_param,
    require_tree,
)
from hwoutils.memory import DEFAULT_MAX_BYTES, MemoryLimit, largest_batch
from hwoutils.model_paths import assign, canonical_path, paths_overlap, resolve
from hwoutils.precision import PrecisionError, require_float64
from jax.flatten_util import ravel_pytree
from jax.scipy.special import log_ndtr, logsumexp
from priorities import AbstractDistribution, from_real, to_real

from photomancy.infer._precision import _as_wide

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

    A term whose likelihood rests on a modeling assumption the data cannot check
    (``ImageTerm(static_beyond_speckle=True)``, no speckle drift beyond the field)
    names it in :attr:`assumptions`; ``Fit.assumptions`` collects them, and the
    detection results copy them.

    Attributes:
        reads: Model paths a held build cached. ``Fit`` refuses a free path or block
            write that meets one, because the likelihood would no longer see the change.
        marginalizes: Names of the latents the term sums over inside itself; empty by
            default.
        forward_fields: Field names holding the objects differentiated by the
            forward calculation. By default, Equinox modules held directly or in
            containers are checked for narrow floating leaves. Declare names when
            the forward uses a raw array or when a module holds only observations.
            A held term applies its own declaration recursively.
    """

    reads: eqx.AbstractVar[tuple[str, ...]]
    marginalizes: ClassVar[tuple[str, ...]] = ()

    forward_fields: ClassVar[tuple[str, ...] | None] = None

    def _check_forward_precision(self, operation, path):
        declared = self.forward_fields
        names = (field.name for field in fields(self)) if declared is None else declared
        for name in names:
            value = getattr(self, name)
            leaves = jax.tree_util.tree_leaves_with_path(
                value, is_leaf=lambda node: isinstance(node, eqx.Module)
            )
            for keys, owner in leaves:
                location = f"{path}.{name}" + jax.tree_util.keystr(keys)
                if isinstance(owner, AbstractTerm):
                    owner._check_forward_precision(operation, location)
                elif declared is not None or isinstance(owner, eqx.Module):
                    _require_wide_owner(operation, owner, location)

    @abstractmethod
    def __call__(self, model) -> jax.Array:
        """Return the scalar log likelihood of the data, constants included.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            Scalar log likelihood of the held data, constants included.
        """

    @abstractmethod
    def draw(self, key, model) -> Self:
        """Return a copy of this term holding a data realization drawn at ``model``.

        Args:
            key: PRNG key for the data realization.
            model: The model at one draw, every leaf an array.

        Returns:
            A copy of the term holding the drawn data.
        """

    def predict(self, model):
        """Return the prediction of ``model`` for this term's data, where it has one.

        Args:
            model: The model at one draw, every leaf an array.
        """
        raise NotImplementedError(f"{type(self).__name__} has no prediction")

    def merged(self, other) -> Self:
        """Return the term scoring the union of both terms' data under shared latents.

        ``photomancy.infer.update`` calls it, on the host, when a new term names a
        latent (:attr:`marginalizes`) that a held term of the same type already sums
        over: the two cannot stay separate terms, so the held one is replaced by this
        merge. The default raises; a term type adds it when it gains a shared latent.

        Args:
            other: The term to join, holding more data of the same latents.

        Returns:
            A term whose log likelihood equals the one term built from all the data.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be merged with another term"
        )

    def split(self, model) -> tuple:
        """Return the prediction at ``model`` and the function that scores it.

        The pair ``(prediction, score)`` has ``score(prediction) == self(model)``:
        ``prediction`` is what the model renders (a PyTree whose inexact leaves are
        the numbers a source amplitude scales) and ``score`` is a PyTree of arrays
        that is callable, holding the data and everything else the likelihood reads.
        It lets :func:`photomancy.detection.detect` render once at an amplitude of
        zero and its derivative, and then score ``prediction + a * tangent`` at any
        ``a`` without rendering again; that is exact only for a prediction linear in
        the amplitude, which ``detect`` checks with one more render and answers by
        Newton steps through the term when it fails. A term that does not offer it
        keeps the default, which raises ``NotImplementedError``, and ``detect`` renders
        at every Newton step for a fit that holds such a term.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            ``(prediction, score)``.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot split its likelihood")

    def residual(self, model) -> jax.Array:
        """Return the whitened residual of the data at ``model``, shaped like the data.

        Each scored element is ``(observed - mean) / standard deviation`` under the
        term's own noise law at ``model``, so where the model is right it has mean
        zero and unit variance; unscored elements are NaN. Positive means the data
        are above the model.

        Optional, like :meth:`fisher_root`. The default raises
        ``NotImplementedError``: a robust law (Laplace, Huber, Student-t) has no
        residual with a unit-variance null, and a correlated law has no unique
        element-wise one.

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            Whitened residual with the shape of the term's data, NaN where an element
            is not scored.
        """
        raise NotImplementedError(f"{type(self).__name__} has no whitened residual")

    def _fit_residual(self, model):
        """Evaluate the term's residual at Fit precision."""
        return self.residual(model)

    @property
    def footprint_paths(self) -> tuple[str, ...]:
        """Model paths whose value changes what the term can read (the field of view).

        ``Fit`` runs :meth:`check` at both ends of the prior of a free leaf under one
        of these paths. Empty by default.
        """
        return ()

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Names of the modeling assumptions the term's likelihood makes; empty here.

        Static Python strings, never traced, so they survive a transformation.
        """
        return ()

    def check(self, model) -> None:
        """Raise when the term cannot be scored at ``model``; the default accepts all.

        ``Fit`` calls it once at setup on the concrete base model, and at both ends of
        the prior of every free leaf under :attr:`footprint_paths`.

        Args:
            model: The model to test, with concrete values.
        """

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

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            Flat vector ``r(model)`` of the term, with zero at masked entries.
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
        """The term as a Gaussian in the sky constants of ``planet`` of ``model``.

        Args:
            model: The model whose planet sets the nonlinear orbit shape.
            planet: Index of the planet whose sky constants the form is written in.

        Returns:
            The term's Gaussian in the four sky constants of that planet.
        """


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

        Args:
            key: PRNG key for one draw.
            low: Lower bound of ``ln(leaf)`` from the leaf's law, intersected with the
                record's own.
            high: Upper bound of ``ln(leaf)`` from the leaf's law, intersected with the
                record's own.
            widen: Factor that multiplies every component standard deviation.
            uniform: Share of the draw mass given to the uniform component on
                ``[low, high]``, in ``[0, 1]``.

        Returns:
            ``ln_leaf`` is the drawn natural log of the leaf and ``log_q`` is the log
            density of the draw in ``ln(leaf)``, both scalars.
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

        Args:
            model: The model at one draw, every leaf an array.

        Returns:
            The proposal record for the leaf at ``leaf_path``.
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
        total = _as_wide(fit.log_likelihood(entries))
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


class _Float32OwnerError(PrecisionError):
    """A forward owner has a float32 or complex64 leaf in an x64 process."""


def _require_wide_owner(operation, owner, name):
    """Check the owner, including narrow bulk and weak leaves, with their path."""
    require_float64(operation)
    leaves = jax.tree_util.tree_leaves_with_path(
        owner, is_leaf=lambda node: isinstance(node, AbstractTerm)
    )
    for path, leaf in leaves:
        if isinstance(leaf, AbstractTerm):
            leaf._check_forward_precision(operation, name + jax.tree_util.keystr(path))
            continue
        if is_param(leaf) and jnp.finfo(leaf.dtype).dtype != np.float64:
            location = name + jax.tree_util.keystr(path)
            error = (
                _Float32OwnerError
                if jnp.finfo(leaf.dtype).dtype == np.float32
                else PrecisionError
            )
            raise error(
                f"{operation} requires float64. {location} has dtype {leaf.dtype}. "
                "Use canonical(tree, precision='float64')."
            )
    require_float64(operation, [leaf for _, leaf in leaves if is_param(leaf)])


def _keep_model_leaf(x):
    """Keep Python shape/flag constants and explicitly narrow working arrays."""
    return isinstance(x, int) or (
        isinstance(x, jax.Array)
        and x.dtype in (jnp.float32, jnp.complex64)
        and not x.weak_type
    )


def _block_node(previous, value):
    """Cast matching leaves by path, using the replacement's tree and metadata."""
    destinations = dict(jax.tree.leaves_with_path(previous))
    leaves, tree = jax.tree.flatten_with_path(value)
    converted = None
    result = []
    for key, leaf in leaves:
        destination = destinations.get(key)
        if eqx.is_array(destination) and eqx.is_array_like(leaf):
            leaf = jnp.asarray(leaf, destination.dtype)
        else:
            # Canonicalize the node in context, respecting bulk and foreign fields.
            if converted is None:
                converted = dict(jax.tree.leaves_with_path(canonical(value)))
            leaf = converted[key]
        result.append(leaf)
    return jax.tree.unflatten(tree, result)


class _BlockPaths(NamedTuple):
    """One snapshot of a block's path declarations for the structural checks.

    Attributes:
        owns: Path of the owned node.
        reads: Paths whose values the block uses.
        writes: Paths the block may replace.
    """

    owns: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]


def _check_path_aliases(paths, what):
    """Refuse two original spellings of one path before either is discarded."""
    seen = set()
    for key, path in paths:
        if path in seen:
            first = next(other for other, target in paths if target == path)
            raise ValueError(f"{what} {first!r} and {key!r} both name {path!r}")
        seen.add(path)


def _write_block(model, writes, *, declared_paths=None):
    """Check returned paths and write nodes at their destination leaf dtypes.

    The support-swap Jacobian reapplies writes from a block already checked by Fit.
    """
    paths = [(key, _require_path(model, key, "block forward")) for key in writes]
    _check_path_aliases(paths, "block returned aliases")
    if declared_paths is not None:
        declared = tuple(canonical_path(model, path) for path in declared_paths)
        for key, path in paths:
            if not any(
                len(target) <= len(path) and paths_overlap(target, path)
                for target in declared
            ):
                raise ValueError(
                    f"block returned path {key!r} outside its declared writes "
                    f"{declared_paths}"
                )
    for key, path in paths:
        value = writes[key]
        node = _block_node(resolve(model, path), value)
        model = assign(model, path, node)
    return model


def _require_path(model, path, who):
    try:
        return canonical_path(model, path)
    except (AttributeError, IndexError, TypeError, ValueError) as err:
        raise ValueError(f"unknown model path {path!r} (from {who})") from err


def _wide_coordinates(x):
    return jax.tree.map(
        lambda a: _as_wide(as_float_leaf(a)),
        x,
        is_leaf=lambda a: isinstance(a, list | tuple),
    )


def _check_keys(x, schema, path):
    if not isinstance(x, dict):
        raise ValueError(f"{path} must be a dict with keys {sorted(schema)}")
    missing = sorted(schema.keys() - x.keys())
    extra = sorted(x.keys() - schema.keys(), key=repr)
    if missing or extra:
        raise ValueError(f"{path} has missing keys {missing}, extra keys {extra}")


def _check_draws(x, schema):
    """Check the keys and array shapes against a full or selected draw schema."""
    _check_keys(x, schema, "draw (keys of fit.free)")
    leading = None
    for key, entry in schema.items():
        if isinstance(entry, dict):
            _check_keys(x[key], entry, key)
            fields = (
                (f"{key}.{name}", x[key][name], spec) for name, spec in entry.items()
            )
        else:
            fields = [(key, x[key], entry)]
        for path, value, spec in fields:
            if not eqx.is_array_like(value):
                raise ValueError(
                    f"{path} needs an array with trailing shape {spec.shape}"
                )
            shape = jnp.shape(value)
            n = len(shape) - len(spec.shape)
            if n < 0 or shape[n:] != spec.shape:
                raise ValueError(
                    f"{path} has shape {shape}, expected trailing shape {spec.shape}"
                )
            batch = shape[:n]
            if leading is not None and batch != leading:
                raise ValueError(
                    f"{path} has leading shape {batch}, expected {leading}"
                )
            leading = batch
    return leading or ()


def _leaf_map_difference(old, new, shape):
    """Describe the first difference between two priorities coordinate maps.

    Bounds are compared after broadcasting to the leaf ``shape``. A bound that
    depends on traced values cannot be shown equal, so it counts as a difference.
    """
    if old.bounds != new.bounds:
        return f"transform families differ ({old.bounds} and {new.bounds})"
    with jax.ensure_compile_time_eval():
        for a, b in zip(old.support(), new.support(), strict=True):
            if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
                return "traced support bounds cannot be compared"
            try:
                same = np.array_equal(
                    np.broadcast_to(a, shape), np.broadcast_to(b, shape)
                )
            except ValueError:
                return "support bounds have different shapes"
            if not same:
                return "support bounds differ"
    return None


class Fit(eqx.Module):
    """A normalized log density over the free entries of a model.

    Attributes:
        model: The base model as converted by :func:`hwoutils.leaves.canonical`: every
            float and array leaf is a JAX array; Python ``int`` and ``bool`` scalars
            are kept as they are. A held path keeps its value here.
        free: Path to leaf law (a ``priorities`` distribution over the whole leaf) or
            to a block (keyed by the path it owns). Keys are static.
        terms: Data terms, each a normalized log likelihood.
    """

    model: eqx.Module
    free: dict[str, AbstractDistribution | AbstractBlock]
    terms: tuple[AbstractTerm, ...] = ()
    _walls: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    _canonical: tuple[tuple[str, str], ...] = eqx.field(static=True, init=False)

    def __init__(
        self,
        model: eqx.Module,
        free: dict[str, AbstractDistribution | AbstractBlock],
        terms: tuple[AbstractTerm, ...] = (),
        _walls: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ):
        """Convert the model and store its entries and the walls of the blocks.

        Args:
            model: The base model. Its leaves are converted with
                :func:`hwoutils.leaves.canonical` (Python floats and complex numbers,
                NumPy arrays and weakly typed arrays become JAX arrays in the active
                precision; explicitly typed float32 and complex64 JAX leaves keep
                their precision, as do bulk tables and third-party Modules).
                Python ``int`` and ``bool`` scalars are kept: under ``eqx.filter_jit``
                they are compile-time constants, which a model may use as a shape or
                in a Python ``if``, and an array would break that. ``fit.model`` is
                the converted tree, not the object passed in.
            free: The free entries by path.
            terms: The data terms.
            _walls: Walls to keep for use under a transformation, as
                ``((key, walls), ...)``; evaluated from the blocks when omitted. The
                edits pass the walls of the entries they keep, and
                ``dataclasses.replace`` passes the stored value.

        Raises:
            PrecisionError: When float64 is unavailable.
        """
        require_float64("Fit")
        kept, rest = eqx.partition(model, _keep_model_leaf, is_leaf=is_foreign)
        rest = canonical(rest)
        require_tree(rest)
        self.model = eqx.combine(kept, rest, is_leaf=is_foreign)
        self.free = free
        self.terms = terms
        paths = []
        for key in sorted(free):
            who = (
                "a block must be keyed by the path it owns"
                if isinstance(free[key], AbstractBlock)
                else "free"
            )
            paths.append((key, self._require_path(key, who)))
        self._canonical = tuple(paths)

        self._check_fit()
        self._walls = _block_walls(free) if _walls is None else _walls

    def _check_fit(self):
        """Refuse an inconsistent fit on the host.

        Raises:
            TypeError: For an entry that is neither a distribution nor a block, or a
                term that is not an :class:`AbstractTerm`.
            ValueError: For an unknown path, a leaf law that does not fit its leaf, an
                entry that breaks a declared tie (``check_ties``), two writers of one
                path, a block cycle, a free path or block write that meets a term's
                ``reads``, or two terms that marginalize the same latent
                (``AbstractTerm.marginalizes``), named in the message.
        """
        _check_path_aliases(self._canonical, "two writers:")
        canonical = self.canonical_keys
        leaf_paths, blocks = [], {}
        for path, entry in self.free.items():
            if isinstance(entry, AbstractDistribution):
                self._check_leaf(path, entry)
                leaf_paths.append(path)
            elif isinstance(entry, AbstractBlock):
                blocks[path] = _BlockPaths(entry.owns, entry.reads, entry.writes)
            else:
                raise TypeError(
                    f"free[{path!r}] is a {type(entry).__name__}; "
                    "expected a priorities distribution or a block"
                )
        written = [(path, canonical[path]) for path in leaf_paths]
        for key, paths in blocks.items():
            who = f"block {key!r}"
            self._require_path(paths.owns, who)
            for path in paths.reads:
                self._require_path(path, who)
            written.extend(
                (path, self._require_path(path, who)) for path in paths.writes
            )
        check_ties(
            model=self.model,
            writes={
                **{path: (path,) for path in leaf_paths},
                **{key: block.writes for key, block in blocks.items()},
            },
            reads={key: block.reads for key, block in blocks.items()},
        )
        check_writers(blocks, leaf_paths, model=self.model)
        block_order(blocks, model=self.model)
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
                target = self._require_path(read, f"{type(term).__name__}.reads")
                for path, source in written:
                    if paths_overlap(source, target):
                        raise ValueError(
                            f"free path {path!r} meets {type(term).__name__}.reads "
                            f"entry {read!r}: the term holds a build that cached it"
                        )
        self._check_terms([(path, canonical[path]) for path in leaf_paths])

    def _check_terms(self, leaf_paths):
        """Run each term's :meth:`AbstractTerm.check` on the concrete model.

        The base model is checked once; every free leaf under a term's
        ``footprint_paths`` is also set to each end of its prior and checked there,
        because the field of view moves with it. A model with traced leaves (a fit
        built inside a transformation) is not checked.
        """
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(self.model)):
            return
        for i, term in enumerate(self.terms):
            term.check(self.model)
            footprints = tuple(
                self._require_path(path, f"terms[{i}].footprint_paths")
                for path in term.footprint_paths
            )
            for path, target in leaf_paths:
                if not any(paths_overlap(target, f) for f in footprints):
                    continue
                leaf = resolve(self.model, path)
                for end in self.free[path].support():
                    if not bool(jnp.all(jnp.isfinite(end))):
                        raise ValueError(
                            f"free path {path!r} moves the field of view of "
                            f"terms[{i}] and its prior has an unbounded end: bound "
                            "the prior or hold the path"
                        )
                    moved = assign(
                        self.model,
                        path,
                        jnp.broadcast_to(end, leaf.shape).astype(leaf.dtype),
                    )
                    term.check(moved)

    def _require_path(self, path, who):
        return _require_path(self.model, path, who)

    def _check_leaf(self, path, dist):
        leaf = resolve(self.model, path)
        if not is_param(leaf):
            if eqx.is_array(leaf):
                raise ValueError(
                    f"free path {path!r} names an array of dtype {leaf.dtype}, not a "
                    "floating leaf; an integer or boolean array fixes structure and "
                    "cannot be free"
                )
            if isinstance(leaf, bool | int | float | complex | str | tuple):
                raise ValueError(
                    f"free path {path!r} names a {type(leaf).__name__}, a static "
                    "field that fixes structure and cannot be free"
                )
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
    def canonical_keys(self) -> dict[str, str]:
        """Map each free coordinate key to its model path with numeric indices."""
        return dict(self._canonical)

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
    def walls(self) -> dict[str, tuple[str, ...]]:
        """The non-empty ``walls`` of the blocks, by entry key.

        A wall is a bound of a block's law that its coordinates do not map, so the
        density is zero on a set of positive measure at finite coordinates. Read
        from the blocks when their laws are concrete, so a fit edited with
        ``eqx.tree_at`` reports the walls of its new laws; under a transformation
        (traced laws, whose edges cannot be read) the walls stored when the fit was
        built are returned. Leaf laws contribute none (see the module docstring for
        the assumption).
        """
        cached = dict(self._walls)
        walls = {}
        for key, block in self.blocks.items():
            names = (
                cached.get(key, ())
                if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(block))
                else block.walls
            )
            if names:
                walls[key] = tuple(names)
        return walls

    @property
    def block_order(self) -> tuple[str, ...]:
        """Block keys in the order they are applied."""
        return block_order(self.blocks, model=self.model)

    @property
    def assumptions(self) -> tuple[str, ...]:
        """The sorted union of the terms' :attr:`AbstractTerm.assumptions`.

        A likelihood, an information matrix or an evidence is comparable with another
        only when the two fits make the same assumptions.
        """
        return tuple(sorted({name for term in self.terms for name in term.assumptions}))

    def _schema_of(self, key):
        """Stored coordinate shape from the current model, at the draw precision."""
        entry = self.free[key]
        if isinstance(entry, AbstractBlock):
            spec = jax.eval_shape(lambda: entry.inverse(self.model))
            return jax.tree.map(
                lambda s: jax.ShapeDtypeStruct(s.shape, jnp.float64), spec
            )
        leaf = resolve(self.model, key)
        return jax.ShapeDtypeStruct(leaf.shape, jnp.float64)

    @property
    def _schema(self) -> dict:
        return {key: self._schema_of(key) for key in sorted(self.free)}

    def _coordinate_difference(self, other):
        """Return the first differing key and reason, or None for the same map."""

        def shapes(spec):
            return jax.tree.map(lambda s: s.shape, spec)

        for key in sorted(self.free.keys() | other.free.keys()):
            if key not in self.free or key not in other.free:
                return key, "free keys differ"
            old, new = self.free[key], other.free[key]
            if isinstance(old, AbstractBlock) != isinstance(new, AbstractBlock):
                return key, "entry kinds differ"
            if isinstance(old, AbstractBlock):
                with jax.ensure_compile_time_eval():
                    if not old.same_map(new):
                        return key, "block maps differ"
            else:
                reason = _leaf_map_difference(old, new, self._schema_of(key).shape)
                if reason:
                    return key, reason
            if shapes(self._schema_of(key)) != shapes(other._schema_of(key)):
                return key, "trailing shapes differ"
        return None

    def same_coordinates(self, other: "Fit") -> bool:
        """Whether draws in both fits have the same keys, shapes and coordinate maps.

        Blocks decide map equality with their same_map method. Leaf laws compare
        their real transform family and support endpoints, broadcast to the leaf
        shape. An endpoint that depends on traced values cannot be compared and
        counts as a difference.

        Args:
            other: Fit whose coordinates are compared with this fit.

        Returns:
            True when every free entry agrees.
        """
        return self._coordinate_difference(other) is None

    def draw_schema(self) -> dict:
        """Return the shape and dtype of each coordinate of one stored draw.

        Returns:
            Free keys mapped to shape and dtype records. Each block holds a dict
            of these records keyed by coordinate name.
        """
        return self._schema

    def check_draws(self, x: dict) -> tuple[int, ...]:
        """Validate stored draws against the free entries of this fit.

        Args:
            x: Entry dict with the same leading batch axes on every array.

        Returns:
            Common leading shape, empty for a single draw or an empty free set.

        Raises:
            ValueError: For missing or extra keys or incompatible array shapes.
        """
        return _check_draws(x, self._schema)

    def _batch_ndim(self, x):
        """Number of leading batch axes of an entry dict (all entries agree)."""
        return len(self._batch_shape(x))

    def _batch_shape(self, x):
        """Shape of the leading batch axes of an entry dict."""
        for key in self.free:
            spec = self._schema_of(key)
            value = x[key]
            if isinstance(spec, dict):
                name = next(iter(spec))
                value, spec = value[name], spec[name]
            shape = jnp.shape(value)
            return shape[: len(shape) - len(spec.shape)]
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
            value = x[path]
            leaf = resolve(model, path)
            expected = leaf.shape
            if value.shape != expected:
                raise ValueError(f"{path} has shape {value.shape}, expected {expected}")
            model = assign(model, path, value.astype(leaf.dtype))
        return model

    def _walk(self, x):
        """Walk the block chain once: return the model of ``x`` and its log prior.

        Leaf laws are assigned first and scored, then each block in order scores its
        ``log_prob`` on the running model before its writes are applied.
        """
        x = _wide_coordinates(x)
        model = self._assign_leaf_laws(x, self.model)
        total = jnp.zeros(())
        for path, dist in self.leaf_laws.items():
            ndim = jnp.ndim(resolve(self.model, path))
            total = total + _sum_leaf(dist.log_prob(x[path]), ndim, dist)
        blocks = self.blocks
        for key in self.block_order:
            block = blocks[key]
            total = total + block.log_prob(x[key], model)
            model = _write_block(
                model, block.forward(x[key], model), declared_paths=block.writes
            )
        return model, total

    def model_at(self, x: dict) -> eqx.Module:
        """Return the model of one draw ``x``: leaf laws first, then blocks in order.

        Args:
            x: Entry dict of one draw, in the stored coordinates.

        Returns:
            The base model with every free entry written.
        """
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

        Args:
            x: Entry dict of draws in the stored coordinates, each leaf with leading
                batch axes and then its leaf shape.
            batch_size: Chunk size of the ``jax.lax.map`` over the outermost batch axis,
                or ``None`` to map every draw at once.

        Returns:
            Normalized log prior, with the batch shape of ``x``.
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
        """Return the sum of the terms at ``x`` (``batch_size`` as in ``log_prior``).

        Args:
            x: Entry dict of draws in the stored coordinates, each leaf with leading
                batch axes and then its leaf shape.
            batch_size: Chunk size of the ``jax.lax.map`` over the outermost batch axis,
                or ``None`` to map every draw at once.

        Returns:
            Summed log likelihood of the terms, with the batch shape of ``x``.
        """
        return self._over_batch(self._log_likelihood_one, x, batch_size)

    def residuals(self, x: dict) -> tuple[jax.Array, ...]:
        """Return each term's whitened residual at one draw ``x``.

        One entry per term, in the order of ``terms``, each
        :meth:`AbstractTerm.residual` at ``model_at(x)``: shaped like that term's
        data, NaN where an element is not scored. A term is one exposure or one data
        set, so the sum of squares of an entry is the per-frame statistic that shows
        a frame the model does not describe.

        Args:
            x: Entry dict of one draw, in the stored coordinates.

        Returns:
            Tuple of whitened residuals, one per term.

        Raises:
            PrecisionError: When float64 is unavailable.
            NotImplementedError: When a term has no whitened residual (a robust or
                correlated law), naming the term.
        """
        require_float64("Fit.residuals")
        model = self.model_at(x)
        out = []
        for i, term in enumerate(self.terms):
            try:
                out.append(term._fit_residual(model))
            except NotImplementedError as err:
                raise NotImplementedError(f"terms[{i}]: {err}") from err
        return tuple(out)

    def _log_density_one(self, x):
        model, prior = self._walk(x)
        total = prior + self._terms_at(self.terms, model)
        return jnp.where(jnp.isneginf(prior), -jnp.inf, total)

    def log_density(self, x: dict, *, batch_size: int | None = None) -> jax.Array:
        """Return ``log_prior + log_likelihood`` (``-inf`` off the prior's support).

        ``batch_size`` is as in :meth:`log_prior`.

        Args:
            x: Entry dict of draws in the stored coordinates, each leaf with leading
                batch axes and then its leaf shape.
            batch_size: Chunk size of the ``jax.lax.map`` over the outermost batch axis,
                or ``None`` to map every draw at once.

        Returns:
            Log density with the batch shape of ``x``.
        """
        return self._over_batch(self._log_density_one, x, batch_size)

    # ----- curvature -------------------------------------------------------------

    def _root_vector(self, x, dtype=jnp.float64):
        """Concatenated Fisher roots of every term at the draw ``x``, in ``dtype``."""
        model = self.model_at(x)
        parts = [
            jnp.ravel(term.fisher_root(model)).astype(dtype) for term in self.terms
        ]
        return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype)

    def _root_length(self, model):
        """Length of :meth:`_root_vector`, from shapes only.

        Raises:
            NotImplementedError: Naming the first term that has no Fisher root.
        """
        total = 0
        for i, term in enumerate(self.terms):
            try:
                shape = eqx.filter_eval_shape(term.fisher_root, model)
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"terms[{i}] ({type(term).__name__}) has no Fisher root, so "
                    "kind='fisher' is unavailable for this fit (all terms or none); "
                    f"use kind='observed'. Reason: {err}"
                ) from err
            total += int(np.prod(shape.shape, dtype=int))
        return total

    def _require_float64_forward(self, operation, x):
        """Return the checked forward model after applying all writes at one draw."""
        require_float64(operation)
        model = self.model_at(x)
        _require_wide_owner(operation, model, "model")
        for i, term in enumerate(self.terms):
            term._check_forward_precision(operation, f"terms[{i}]")
        return model

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
            PrecisionError: When float64 is unavailable or a forward owner is narrow.
            ValueError: For an unknown ``kind``.
            NotImplementedError: For ``kind="fisher"`` when a term has no root.
            MemoryLimit: When the estimate is above ``max_bytes``.
        """
        if kind not in ("observed", "fisher"):
            raise ValueError(f"kind must be 'observed' or 'fisher', got {kind!r}")
        model = self._require_float64_forward("Fit.information", x)
        x = jax.tree.map(_as_wide, x)
        n_param = sum(int(np.size(a)) for a in jax.tree.leaves(x))
        if kind == "observed":
            _admit_information(
                n_param, n_param * 8, 2 * n_param * n_param * 8, batch_size, max_bytes
            )
            return _observed_information(self, x, include_prior, batch_size)
        n_root = self._root_length(model)
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

        Args:
            x: Entry dict of draws in the stored coordinates.

        Returns:
            Entry dict in unconstrained coordinates, with the shapes of ``x``.
        """
        x = _wide_coordinates(x)
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

        Args:
            u: Entry dict in unconstrained coordinates, with leading batch axes.

        Returns:
            ``x`` is the entry dict in the stored coordinates and ``volume`` is the log
            volume of the map, with the batch shape of ``u``.
        """
        u = _wide_coordinates(u)
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

        Returns:
            Tuple with one period per element of the flat vector (in the units of the
            block coordinate), or ``None`` where the element is not periodic.
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

        Args:
            z: Flat unconstrained vectors, shape ``(..., d)``.

        Returns:
            Entry dict in unconstrained coordinates, with the batch shape of ``z``.
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

        Args:
            x: Entry dict of draws in the stored coordinates, with optional leading
                batch axes.

        Returns:
            Flat unconstrained vectors, shape ``(*batch, d)``.
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

        Args:
            z: Flat unconstrained vectors, shape ``(..., d)``.

        Returns:
            ``x`` is the entry dict in the stored coordinates and ``volume`` is the log
            volume of the map, with the batch shape of ``z``.
        """
        return self.from_real(self.unravel(z))

    def real_log_prob(self, u: dict) -> jax.Array:
        """Return the density in ``u``: the density of ``from_real(u)`` plus volume.

        Args:
            u: Entry dict in unconstrained coordinates, with optional leading batch
                axes.

        Returns:
            Log density in ``u``, with the batch shape of ``u``.
        """
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
            x[path] = _wide_coordinates(
                dist.sample(subkeys[path], shape if own == () else ())
            )
        model = self._assign_leaf_laws(x, self.model)
        blocks = self.blocks
        for bkey in self.block_order:
            block = blocks[bkey]
            x[bkey] = _wide_coordinates(block.sample(subkeys[bkey], model))
            model = _write_block(
                model, block.forward(x[bkey], model), declared_paths=block.writes
            )
        return x

    def sample_prior(self, key, sample_shape: tuple[int, ...] = ()) -> dict:
        """Draw ``x`` from the prior, leaf laws then blocks on the running model.

        Returns entries shaped ``(*sample_shape, *leaf_shape)`` (a block entry is a
        dict of arrays shaped ``(*sample_shape, R)``).

        Args:
            key: PRNG key.
            sample_shape: Leading batch shape of the draws.

        Returns:
            Entry dict of prior draws in the stored coordinates.
        """
        n = int(np.prod(sample_shape, dtype=int))
        flat = jax.vmap(self._sample_prior_one)(jax.random.split(key, n))
        return jax.tree.map(lambda a: a.reshape(*sample_shape, *a.shape[1:]), flat)

    # ----- unit cube -------------------------------------------------------------

    def unit_template(self) -> dict:
        """Zeros of one unit-cube draw.

        A leaf-law entry has the shape of its model leaf; a block entry has shape
        ``(unit_size,)``, the length of the cube vector its ``from_unit`` consumes.

        Returns:
            Entry dict of zeros, one per free entry.

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

        Returns:
            Entry dict of one draw in the stored coordinates, keyed like ``free``.
        """
        self.unit_template()
        u = _wide_coordinates(u)
        x = {
            key: _wide_coordinates(entry.icdf(u[key]))
            for key, entry in self.leaf_laws.items()
        }
        model = self._assign_leaf_laws(x, self.model)
        blocks = self.blocks
        for key in self.block_order:
            block = blocks[key]
            x[key] = _wide_coordinates(block.from_unit(u[key], model))
            model = _write_block(
                model, block.forward(x[key], model), declared_paths=block.writes
            )
        return {key: x[key] for key in self.free}

    # ----- edits -----------------------------------------------------------------

    def _priors_by_key(self, priors: dict) -> dict:
        """Map each replacement path to its free key, refusing the rest."""
        canonical = self.canonical_keys
        by_path = {path: key for key, path in self._canonical}
        paths, extra = [], []
        for path in priors:
            if path in canonical:
                paths.append((path, path))
                continue
            try:
                key = by_path[self._require_path(path, "with_priors")]
            except (KeyError, ValueError):
                extra.append(path)
            else:
                paths.append((path, key))
        if extra:
            raise ValueError(
                f"with_priors cannot change the free set; not free: {sorted(extra)}"
            )
        _check_path_aliases(paths, "prior replacements")
        return {key: priors[path] for path, key in paths}

    def with_priors(self, priors: dict) -> "Fit":
        """Return a fit with some entries replaced, the free set unchanged.

        A leaf law is replaced by a leaf law, and a block by a block of the same layout
        (``old.same_layout(new)``: the same type, ownership, chart and rows, only the
        law differs). A new law may move the bounds that the block's map carries, so a
        coordinate vector made for the old fit may mean something else under the new
        one; whether stored draws survive is for the caller to decide
        (:meth:`Posterior.with_priors` does, by ``same_map``).

        Args:
            priors: ``{path: law or block}`` of the replacements. Each path can be a
                free key or another spelling of its model path.

        Returns:
            A new fit with the replacements and the same terms.

        Raises:
            ValueError: For a key outside the free set, two paths for one entry,
                a replacement of the other kind, or a block of another layout.
        """
        priors = self._priors_by_key(priors)
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
        free = {**self.free, **priors}
        kept = dict(self._walls)
        fresh = dict(_block_walls(priors))
        walls = tuple(
            (k, fresh[k] if k in priors else kept[k])
            for k in free
            if (fresh if k in priors else kept).get(k)
        )
        return Fit(self.model, free, self.terms, walls)

    def append_term(self, term: AbstractTerm) -> "Fit":
        """Return a fit with one more term; the free set is unchanged.

        Args:
            term: The term to add at the end of ``terms``.

        Returns:
            A new fit with the extra term.
        """
        return Fit(self.model, self.free, (*self.terms, term), self._walls)

    def replace_term(self, index: int, term: AbstractTerm) -> "Fit":
        """Return a fit with ``terms[index]`` replaced; the free set is unchanged.

        The replacement may be any term whose reads pass the checks of ``Fit``. A stack
        of worlds that must not recompile needs a replacement with the same pytree
        structure as the old term (a padded row written in place).

        Args:
            index: Position in ``terms`` to replace.
            term: The replacement term.

        Returns:
            A new fit with the replacement.

        Raises:
            IndexError: When ``index`` is outside ``range(len(terms))`` (negative
                indices are refused, so a stacked fit cannot shift a slot).
        """
        if not 0 <= index < len(self.terms):
            raise IndexError(
                f"replace_term index {index} outside range({len(self.terms)})"
            )
        terms = (*self.terms[:index], term, *self.terms[index + 1 :])
        return Fit(self.model, self.free, terms, self._walls)

    def digest(self) -> str:
        """Hash of the free keys, prior records, term kinds and model structure.

        Evaluated on the host from concrete arrays. Equal for fits that share a
        density, so a stored result can be checked against the fit it targets.

        Returns:
            Hexadecimal SHA-256 digest.
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


def _block_walls(free) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The non-empty walls of the block entries, as ``((key, walls), ...)``."""
    return tuple(
        (key, tuple(walls))
        for key, entry in free.items()
        if isinstance(entry, AbstractBlock) and (walls := entry.walls)
    )
