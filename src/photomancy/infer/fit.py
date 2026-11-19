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
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.blocks import AbstractBlock, block_order, check_writers
from hwoutils.model_paths import assign, paths_overlap, resolve
from priorities import AbstractDistribution, from_real, to_real

__all__ = ["AbstractTerm", "Fit"]


class AbstractTerm(eqx.Module):
    """One data contribution: a normalized log likelihood of held data at a model.

    A term closes over its data (arrays held as fields, so they stay in the pytree) and
    reads the model it is given; it never builds or caches from the model. A term that
    holds a build made from model values (``coronagraphoto.ApertureRecord``) declares
    the model paths the build read in ``reads``, prefixed with the scene's path in
    ``fit.model``, as a property that returns the build's own tuple. A term with no
    build declares ``reads: ClassVar[tuple[str, ...]] = ()``.

    Attributes:
        reads: Model paths a held build cached. ``Fit`` refuses a free path or block
            write that meets one, because the likelihood would no longer see the change.
    """

    reads: eqx.AbstractVar[tuple[str, ...]]

    @abstractmethod
    def __call__(self, model) -> jax.Array:
        """Return the scalar log likelihood of the data, constants included."""

    @abstractmethod
    def draw(self, key, model) -> Self:
        """Return a copy of this term holding a data realization drawn at ``model``."""

    def predict(self, model):
        """Return the prediction of ``model`` for this term's data, where it has one."""
        raise NotImplementedError(f"{type(self).__name__} has no prediction")

    def value_and_prediction(self, model):
        """Return ``(log likelihood, prediction)``; override to share one render."""
        return self(model), self.predict(model)


def _sum_leaf(value, leaf_ndim, dist):
    """Sum an elementwise log density over the leaf axes it keeps."""
    n = leaf_ndim - dist.event_dim
    return jnp.sum(value, axis=tuple(range(-n, 0))) if n > 0 else value


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
                writers of one path, a block cycle, or a free path or block write that
                meets a term's ``reads``.
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
        for i, term in enumerate(self.terms):
            if not isinstance(term, AbstractTerm):
                raise TypeError(
                    f"terms[{i}] is a {type(term).__name__}; expected an AbstractTerm"
                )
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

    def _over_batch(self, fn, x):
        for _ in range(self._batch_ndim(x)):
            fn = jax.vmap(fn)
        return fn(x)

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

    def log_prior(self, x: dict) -> jax.Array:
        """Return the normalized log prior of ``x`` against Lebesgue ``dx``.

        Leaf laws are summed over the axes of their leaf. Blocks follow on the running
        model, each ``log_prob`` read before that block's writes, so the blocks form a
        chain of normalized conditionals.
        """
        return self._over_batch(self._log_prior_one, x)

    @staticmethod
    def _terms_at(terms, model):
        total = jnp.zeros(())
        for term in terms:
            total = total + term(model)
        return total

    def _log_likelihood_one(self, x):
        return self._terms_at(self.terms, self.model_at(x))

    def log_likelihood(self, x: dict) -> jax.Array:
        """Return the sum of the terms at ``x``."""
        return self._over_batch(self._log_likelihood_one, x)

    def _log_density_one(self, x):
        model, prior = self._walk(x)
        total = prior + self._terms_at(self.terms, model)
        return jnp.where(jnp.isneginf(prior), -jnp.inf, total)

    def log_density(self, x: dict) -> jax.Array:
        """Return ``log_prior + log_likelihood`` (``-inf`` off the prior's support)."""
        return self._over_batch(self._log_density_one, x)

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
        axes (zero for a block entry) and has the batch shape of ``u``.
        """
        x, volume = {}, jnp.zeros(())
        for key, entry in self.free.items():
            if isinstance(entry, AbstractDistribution):
                x[key], vol = from_real(entry, u[key])
                ndim = jnp.ndim(resolve(self.model, key))
                volume = volume + _sum_leaf(vol, ndim, entry)
            else:
                x[key] = u[key]
        return x, jnp.broadcast_to(volume, self._batch_shape(u))

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
        """Zeros of one unit-cube draw: each entry has the shape of its model leaf.

        Raises:
            ValueError: For a block entry (a block has no unit-cube map yet) or a leaf
                law with no inverse CDF (a mixture).
        """
        out = {}
        for key, entry in self.free.items():
            if not isinstance(entry, AbstractDistribution):
                raise ValueError(
                    f"block {key!r} has no unit-cube map; nested sampling needs "
                    "leaf laws with an inverse CDF"
                )
            if not entry.has_icdf:
                raise ValueError(
                    f"law at {key!r} ({type(entry).__name__}) has no inverse CDF"
                )
            out[key] = jnp.zeros(jnp.shape(resolve(self.model, key)))
        return out

    def from_unit(self, u: dict) -> dict:
        """Map unit-cube coordinates to entries through each leaf law's inverse CDF.

        Args:
            u: One draw, each entry shaped like :meth:`unit_template`, values in
                ``(0, 1)``.
        """
        self.unit_template()
        return {key: entry.icdf(u[key]) for key, entry in self.free.items()}

    # ----- edits -----------------------------------------------------------------

    def with_priors(self, priors: dict) -> "Fit":
        """Return a fit with some entries replaced, the free set unchanged.

        A leaf law is replaced by a leaf law and a block by a block that has the same
        map (``old.same_map(new)``), so the swap changes the density and nothing else.

        Raises:
            ValueError: For a key outside the free set, a replacement of the other
                kind, or a block with a different map.
        """
        extra = set(priors) - set(self.free)
        if extra:
            raise ValueError(
                f"with_priors cannot change the free set; not free: {sorted(extra)}"
            )
        for key, new in priors.items():
            old = self.free[key]
            if isinstance(old, AbstractBlock):
                if not isinstance(new, AbstractBlock) or not old.same_map(new):
                    raise ValueError(
                        f"block {key!r} can only be replaced by a block with the same "
                        "map"
                    )
            elif not isinstance(new, AbstractDistribution):
                raise ValueError(f"leaf law {key!r} needs a priorities distribution")
        return Fit(self.model, {**self.free, **priors}, self.terms)

    def append(self, term: AbstractTerm) -> "Fit":
        """Return a fit with one more term; the free set is unchanged."""
        return Fit(self.model, self.free, (*self.terms, term))

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
