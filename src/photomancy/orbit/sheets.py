"""The sum over the two line-of-sight sheets of one planet orbit, as a term combinator.

A Thiele-Innes block holds the orbit of its planet on one sheet; the reflection of that
orbit through the sky plane has the same sky track and the opposite line-of-sight sign.
Relative astrometry cannot tell the two apart, but radial velocity and phase-dependent
photometry can. ``SheetSum`` scores the terms it holds on both sheets and averages the
likelihoods, which is the likelihood of the block's sheet-summed density.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.model_paths import assign, resolve
from orbix.system import reflect_line_of_sight

from photomancy.infer import AbstractProposingTerm, AbstractTerm, LeafProposal

__all__ = ["SheetSum"]


class SheetSum(AbstractProposingTerm):
    """The terms that tell the sheets of one planet apart, summed over both sheets.

    The value is ``logaddexp(sum_t t(model), sum_t t(reflected)) - log 2``, with
    ``reflected`` the model whose orbit of ``planet`` at ``orbit_path`` is reflected
    through the sky plane (:func:`orbix.system.reflect_line_of_sight`). Every term that
    tells the sheets of that planet apart belongs inside one ``SheetSum``: a product of
    two sheet sums is not the sheet sum of the product. ``marginalizes`` names the
    sheet, so a ``Fit`` refuses a second ``SheetSum`` over the same planet, and a term
    added later enters through ``replace_term`` of the ``SheetSum``. Terms that do not
    see the sheet (relative astrometry) stay outside it. For an orbit law that is
    symmetric under the reflection the sum is exact.

    ``check`` and ``footprint_paths`` are those of the held terms, so a ``Fit`` tests a
    held ``ImageTerm`` against its support as it would the bare term.

    The leaf record is that of the one held term that proposes a leaf, on both sheets:
    its components on the model, then those on the reflected model.

    Attributes:
        terms: The terms summed over the two sheets.
        orbit_path: Model path of the stacked orbits.
        planet: Index of the planet whose orbit sheet is summed.
    """

    terms: tuple[AbstractTerm, ...]
    orbit_path: str = eqx.field(static=True)
    planet: int = eqx.field(static=True, default=0)

    def __check_init__(self):
        """Refuse an empty sum, a non-term, or two held terms that propose a leaf."""
        if not self.terms:
            raise ValueError("SheetSum needs at least one term")
        for i, term in enumerate(self.terms):
            if not isinstance(term, AbstractTerm):
                raise TypeError(
                    f"terms[{i}] is a {type(term).__name__}; expected an AbstractTerm"
                )
        if len(self._proposers()) > 1:
            raise ValueError("SheetSum holds at most one term that proposes a leaf")

    def _proposers(self):
        return [
            t
            for t in self.terms
            if isinstance(t, AbstractProposingTerm) and t.leaf_path is not None
        ]

    @property
    def reads(self) -> tuple[str, ...]:
        """The reads of every held term."""
        return tuple(r for t in self.terms for r in t.reads)

    @property
    def marginalizes(self) -> tuple[str, ...]:
        """The sheet of the planet orbit, and what the held terms marginalize."""
        own = f"{self.orbit_path}[{self.planet}].sheet"
        return (own, *(m for t in self.terms for m in t.marginalizes))

    @property
    def assumptions(self) -> tuple[str, ...]:
        """The sorted union of the held terms' assumptions."""
        return tuple(sorted({a for t in self.terms for a in t.assumptions}))

    @property
    def footprint_paths(self) -> tuple[str, ...]:
        """The union of the held terms' footprint paths, in first-seen order."""
        return tuple(dict.fromkeys(p for t in self.terms for p in t.footprint_paths))

    def check(self, model) -> None:
        """Run every held term's check on ``model``.

        The two sheets share one sky track, so the support a term reads is the same on
        both and one check covers the sum.

        Args:
            model: The model to test, with concrete values.

        Raises:
            ValueError: When ``planet`` is outside the orbit rows at ``orbit_path``.
        """
        count = jax.tree.leaves(resolve(model, self.orbit_path))[0].shape[0]
        if not 0 <= self.planet < count:
            raise ValueError(
                f"SheetSum planet {self.planet} is outside the {count} orbit row(s) "
                f"at {self.orbit_path!r}; indices lie in [0, {count})"
            )
        for term in self.terms:
            term.check(model)

    @property
    def leaf_path(self) -> str | None:
        """The leaf the held proposing term offers, or ``None``."""
        proposers = self._proposers()
        return proposers[0].leaf_path if proposers else None

    def reflected(self, model):
        """``model`` with the planet orbit reflected through the sky plane.

        Args:
            model: The model PyTree holding the stacked orbits at ``orbit_path``.

        Returns:
            A copy of ``model`` with the orbit of ``planet`` reflected.
        """
        orbits = resolve(model, self.orbit_path)
        return assign(
            model, self.orbit_path, reflect_line_of_sight(orbits, self.planet)
        )

    def _total(self, model):
        return sum(t(model) for t in self.terms)

    def __call__(self, model) -> jax.Array:
        """The log of the mean of the two sheets' likelihoods.

        Args:
            model: The model PyTree to score.

        Returns:
            The scalar log of the mean likelihood over the two sheets.
        """
        return jnp.logaddexp(
            self._total(model), self._total(self.reflected(model))
        ) - jnp.log(2.0)

    def leaf_proposal(self, model) -> LeafProposal:
        """The proposing term's record on both sheets, components concatenated.

        Args:
            model: The model PyTree at which the record is built.

        Returns:
            The proposing term's :class:`~photomancy.infer.LeafProposal` with the
            components of both sheets concatenated.

        Raises:
            ValueError: When no held term proposes a leaf.
        """
        proposers = self._proposers()
        if not proposers:
            raise ValueError("no term in this SheetSum proposes a leaf")
        term = proposers[0]
        plus = term.leaf_proposal(model)
        minus = term.leaf_proposal(self.reflected(model))
        return LeafProposal(
            plus.path,
            jnp.concatenate([plus.mean, minus.mean]),
            jnp.concatenate([plus.sd, minus.sd]),
            jnp.concatenate([plus.log_pi, minus.log_pi]),
            jnp.minimum(plus.low, minus.low),
            jnp.maximum(plus.high, minus.high),
        )

    def draw(self, key, model) -> Self:
        """Draw each held term's data at ``model``, on the sheet the model holds.

        Args:
            key: PRNG key; one subkey is used per held term.
            model: The model PyTree at which the data are drawn.

        Returns:
            A copy of this sum whose held terms hold the drawn data.
        """
        keys = jax.random.split(key, len(self.terms))
        terms = tuple(t.draw(k, model) for t, k in zip(self.terms, keys, strict=True))
        return eqx.tree_at(lambda s: s.terms, self, terms)
