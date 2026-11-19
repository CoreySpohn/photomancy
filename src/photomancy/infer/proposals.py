"""Proposal distributions over the flat unconstrained vector of a ``Fit``.

A proposal gives properly weighted draws ``z`` in the flat real vector of
``fit.to_real`` (the ``ravel_pytree`` of one draw: sorted entry keys, sorted coordinate
names inside a block; ``fit.from_flat`` reads it). ``weigh`` returns the draws, their
log weights, whose mean estimates the evidence, and the fit's log density at them. A
density proposal draws ``z`` with its own log density ``log_q`` and weighs a draw by
``fit.log_density + volume - log_q``, the target density in the flat vector over the
proposal's.

Periodic coordinates. A block coordinate with a period (``fit.periods()``) is stored
reduced to the cell ``[0, period)``, and the block density repeats outside it. A
proposal therefore returns draws reduced to the cell and the density of the *wrapped*
distribution on the cell: the sum of the density over a static window of images
(``n_images`` on each side of every periodic coordinate). Without the wrap a proposal
wider than a cell integrates the same mass several times and inflates the evidence while
the weight diagnostics still pass.

Zero-density regions. A block may leave regions of zero density inside its coordinates:
a disk chart whose eccentricity law has a lower bound above zero leaves a hole around
the origin of the eccentricity vector, and a Thiele-Innes chart with a bounded size law
leaves both walls. The fit reports ``-inf`` there. A proposal may land in such a region
(a Gaussian does); the draw then has weight zero and counts in ``N``. A proposal must
have positive density wherever the target does: a region of the target that the
proposal cannot reach is silently missed by an importance estimate. A NaN target is an
error, never a zero weight: the slice of weights becomes NaN. The prior and the unit
cube maps never land in a zero-density region, so a proposal that starts from them
starts outside.
"""

import math
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

from photomancy.infer.particles import systematic_indices

__all__ = [
    "AbstractDensityProposal",
    "AbstractProposal",
    "DefensiveProposal",
    "MixtureProposal",
    "PriorProposal",
]

# Elements of the largest intermediate array of a density evaluation (float64).
_BLOCK_ELEMENTS = 2**22


def _period_array(periods):
    """Periods of the flat vector as floats, zero where a coordinate has none."""
    return jnp.asarray([0.0 if p is None else p for p in periods])


def _wrap_cell(z, periods):
    """Reduce every periodic entry of ``z (..., d)`` to its cell ``[0, period)``."""
    per = _period_array(periods)
    return jnp.where(per > 0, jnp.mod(z, jnp.where(per > 0, per, 1.0)), z)


def _image_shifts(periods, n_images):
    """Host array ``(S, d)``: the shifts of the image window of the periodic entries."""
    shifts = np.zeros((1, len(periods)))
    for i, period in enumerate(periods):
        if period is None:
            continue
        offsets = np.arange(-n_images, n_images + 1) * period
        grid = np.zeros((len(offsets), len(periods)))
        grid[:, i] = offsets
        shifts = (shifts[:, None, :] + grid[None]).reshape(-1, len(periods))
    return jnp.asarray(shifts)


def _blockwise(fn, z, per_row):
    """Apply ``fn`` to blocks of rows of ``z`` with ``jax.lax.map``, join the results.

    The block holds ``_BLOCK_ELEMENTS // per_row`` rows (at least one); the last block
    is padded with copies of the first row and the padding is dropped.
    """
    n = z.shape[0]
    size = int(min(n, max(1, _BLOCK_ELEMENTS // per_row)))
    n_blocks = -(-n // size)
    padded = jnp.concatenate(
        [z, jnp.broadcast_to(z[:1], (n_blocks * size - n, z.shape[1]))]
    )
    out = jax.lax.map(fn, padded.reshape(n_blocks, size, z.shape[1]))
    return out.reshape(-1)[:n]


class AbstractProposal(eqx.Module):
    """Properly weighted draws over the flat unconstrained vector of a fit.

    ``weigh`` is what a proposal is for: ``n`` draws of the fit with log weights whose
    mean estimates the evidence. A proposal with a density at any point derives from
    :class:`AbstractDensityProposal`, which implements ``weigh`` once; a properly
    weighted proposal with no density at ``z`` (a completed orbit proposal) derives
    from this class directly and implements ``weigh`` itself. Subclasses are
    ``eqx.Module`` configurations.
    """

    @abstractmethod
    def weigh(
        self, key, fit, n: int, *, batch_size: int | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Draw ``n`` properly weighted points of ``fit``.

        A dead draw has log weight ``-inf`` and its coordinates are not read; this is
        the only death signal of a proposal. A NaN log weight is an error, which the
        backend turns into a refused slice.

        Args:
            key: PRNG key.
            fit: The fit.
            n: Number of draws.
            batch_size: Draws per batch of the fit's density, where it is evaluated.

        Returns:
            ``(z, log_weights, log_target)``: the draws in the flat vector, ``(n, d)``;
            their log weights, ``(n,)``; and ``fit.log_density`` at the draws, ``(n,)``
            (``-inf`` at a dead draw).
        """

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive_fraction):
        """Return the next round's proposal, fitted to a weighted cloud ``z``.

        The default is a kernel mixture on the cloud with a share ``defensive_fraction``
        of the fit's prior (:meth:`MixtureProposal.from_cloud`,
        :class:`DefensiveProposal`).

        Args:
            key: PRNG key.
            z: Draws of this proposal in the flat vector, ``(N, d)``.
            log_weights: Their log weights, ``(N,)``.
            fit: The fit.
            n_centers: Kernels of the mixture.
            defensive_fraction: Share of the defensive prior component.
        """
        mixture = MixtureProposal.from_cloud(
            key,
            z,
            log_weights,
            n_centers=n_centers,
            periods=fit.periods(),
        )
        return DefensiveProposal(mixture, defensive_fraction)


class AbstractDensityProposal(AbstractProposal):
    """A proposal with a density, evaluable at any point.

    ``sample`` returns ``(z, log_q)`` with ``log_q`` the density of the draws, wrapped
    over the cell for periodic coordinates (see the module docstring); it agrees with
    ``log_prob``. The tempering of
    :class:`~photomancy.infer.backends.importance.AdaptiveImportance` and the
    defensive mixture need the density at points the proposal did not draw.
    """

    @abstractmethod
    def sample(self, key, fit, n: int) -> tuple[jax.Array, jax.Array]:
        """Draw ``n`` points; return ``(z, log_q)`` with ``z`` of shape ``(n, d)``."""

    @abstractmethod
    def log_prob(self, z, fit) -> jax.Array:
        """Return the proposal log density of ``z (n, d)`` as ``(n,)``."""

    def weigh(self, key, fit, n, *, batch_size=None):
        """Draw ``n`` points; weigh each by ``fit.log_density + volume - log_q``.

        A draw in a zero-density region of the fit has weight zero.
        """
        z, log_q = self.sample(key, fit, n)
        x, volume = fit.from_flat(z)
        log_target = fit.log_density(x, batch_size=batch_size)
        return z, log_target + volume - log_q, log_target


class PriorProposal(AbstractDensityProposal):
    """The fit's prior, as a proposal in the flat unconstrained vector.

    ``log_q`` is the prior density in the unconstrained coordinates (``log_prior`` of
    the stored draw plus the log volume of the map), so the importance weight of a
    ``PriorProposal`` draw is the likelihood.

    Attributes:
        sequence: ``"random"`` draws independently from the prior
            (``fit.sample_prior``).
            ``"roberts"`` maps a randomly shifted Roberts low-discrepancy sequence of
            the unit cube through ``fit.from_unit``: every draw is marginally a prior
            draw and the set covers the prior more evenly, but the draws are not
            independent, so the standard error of an evidence from them is only a
            conservative guide.
    """

    sequence: str = eqx.field(static=True, default="random")

    def __check_init__(self):
        """Refuse an unknown sequence name."""
        if self.sequence not in ("random", "roberts"):
            raise ValueError(
                f"sequence must be 'random' or 'roberts', got {self.sequence!r}"
            )

    def sample(self, key, fit, n):
        """Draw ``n`` prior points; return ``(z, log_q)``."""
        if self.sequence == "random":
            x = fit.sample_prior(key, (n,))
        else:
            template = fit.unit_template()
            flat, unravel = ravel_pytree(template)
            cube = _roberts(key, n, flat.shape[0])
            x = jax.vmap(lambda c: fit.from_unit(unravel(c)))(cube)
        z = jax.vmap(lambda one: ravel_pytree(one)[0])(fit.to_real(x))
        z = _wrap_cell(z, fit.periods())
        return z, self.log_prob(z, fit)

    def log_prob(self, z, fit):
        """Prior log density in the unconstrained coordinates."""
        x, volume = fit.from_flat(z)
        return fit.log_prior(x) + volume


def _roberts(key, n, dim):
    """``n`` points of the ``dim``-dimensional Roberts sequence with a random shift."""
    phi = 2.0
    for _ in range(64):
        phi = (1.0 + phi) ** (1.0 / (dim + 1.0))
    alpha = jnp.asarray(phi ** -(np.arange(dim) + 1.0))
    shift = jax.random.uniform(key, (dim,))
    cube = jnp.mod(shift + (jnp.arange(n)[:, None] + 1.0) * alpha, 1.0)
    return jnp.clip(cube, 1e-12, 1.0 - 1e-12)


class MixtureProposal(AbstractDensityProposal):
    """A Gaussian mixture; a single Gaussian is one component.

    Draws are reduced to the cell of every periodic coordinate and ``log_q`` is the
    wrapped density (module docstring). Component means are reduced to the cell before
    the density is taken, which leaves the wrapped density unchanged and keeps the image
    window centered on the draws. The window is exact while the standard deviation of a
    component along a periodic coordinate is below about half a period (the window then
    holds four standard deviations); a wider component loses the mass beyond it.

    With one covariance shared by every component (``from_cloud``) the density takes a
    fast path: the draws and centers are whitened once and the squared distances come
    from a matrix product. The cost is ``n * K * S * d`` for ``n`` draws, ``K``
    components and ``S = (2 n_images + 1) ** p`` images of ``p`` periodic coordinates,
    evaluated in blocks of draws so memory stays bounded.

    Attributes:
        means: Component means ``(K, d)``.
        chols: Lower Cholesky factors of the covariances, ``(K, d, d)``, or
            ``(1, d, d)`` for one covariance shared by every component.
        log_pi: Normalized component log weights ``(K,)``; a component with ``-inf`` is
            never drawn.
        n_images: Number of images on each side of every periodic coordinate.
    """

    means: jax.Array
    chols: jax.Array
    log_pi: jax.Array
    n_images: int = eqx.field(static=True, default=2)

    def __init__(self, means, covs, log_pi=None, *, n_images: int = 2):
        """Build the mixture.

        Args:
            means: Component means ``(K, d)``.
            covs: Component covariances ``(K, d, d)``, or one covariance ``(d, d)``
                shared by every component.
            log_pi: Unnormalized component log weights ``(K,)`` (equal when ``None``);
                NaN counts as zero weight.
            n_images: Images on each side of every periodic coordinate.
        """
        means = jnp.asarray(means, jnp.float64)
        covs = jnp.asarray(covs, jnp.float64)
        if covs.ndim == 2:
            covs = covs[None]
        log_pi = jnp.zeros(means.shape[0]) if log_pi is None else jnp.asarray(log_pi)
        log_pi = jnp.where(jnp.isnan(log_pi), -jnp.inf, log_pi)
        self.means = means
        self.chols = jnp.linalg.cholesky(covs)
        self.log_pi = log_pi - logsumexp(log_pi)
        self.n_images = n_images

    @property
    def dim(self) -> int:
        """Length of the flat vector."""
        return self.means.shape[-1]

    @classmethod
    def from_cloud(
        cls,
        key,
        x,
        log_weights,
        *,
        n_centers: int,
        bandwidth: str | float = "silverman",
        periods=None,
        n_images: int = 2,
    ):
        """A kernel mixture on a weighted cloud, with one shared covariance.

        ``n_centers`` centers are drawn from the cloud by systematic resampling and
        each carries the weighted covariance of the cloud scaled by the squared
        bandwidth. The Silverman bandwidth uses ``min(ess, n_centers)`` points: the
        mixture has ``n_centers`` kernels however many draws made the cloud. A small
        share of the variance is added on the diagonal so a cloud that is flat in some
        direction keeps a positive-definite covariance. Periodic coordinates are
        unwrapped around their circular mean before the moments are taken, so a cloud
        that straddles a cell edge has the covariance of its true spread.

        Args:
            key: PRNG key (the resampling of the centers).
            x: Cloud in the flat vector, ``(N, d)``.
            log_weights: Log weights ``(N,)``, normalized here.
            n_centers: Number of kernels.
            bandwidth: ``"silverman"``, or a positive float that multiplies the
                Silverman bandwidth.
            periods: Periods of the flat vector (``fit.periods()``): ``None``, or a
                tuple with ``None`` for a non-periodic coordinate.
            n_images: Images of the resulting mixture.
        """
        d = x.shape[1]
        factor = 1.0 if bandwidth == "silverman" else float(bandwidth)
        log_w = log_weights - logsumexp(log_weights)
        w = jnp.exp(log_w)
        ess = 1.0 / jnp.sum(w**2)
        xs = x
        for i, period in enumerate(periods or ()):
            if period is None:
                continue
            ang = 2.0 * jnp.pi * x[:, i] / period
            center = jnp.arctan2(jnp.sum(w * jnp.sin(ang)), jnp.sum(w * jnp.cos(ang)))
            center = center * period / (2.0 * jnp.pi)
            half = 0.5 * period
            xs = xs.at[:, i].set(
                center + jnp.mod(x[:, i] - center + half, period) - half
            )
        mean = w @ xs
        centered = xs - mean
        cov = (centered * w[:, None]).T @ centered / (1.0 - jnp.sum(w**2))
        n_eff = jnp.clip(ess, 4.0, n_centers)
        h = (n_eff * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        cov = (factor * h) ** 2 * cov
        cov = cov + 1e-6 * jnp.diag(jnp.diag(cov)) + 1e-14 * jnp.eye(d)
        idx = systematic_indices(key, log_w, n_centers)
        return cls(x[idx], cov, None, n_images=n_images)

    def _reduced_means(self, periods):
        return _wrap_cell(self.means, periods)

    def log_prob(self, z, fit):
        """Wrapped mixture log density of ``z (n, d)``, in blocks of draws."""
        periods = fit.periods()
        z = _wrap_cell(z, periods)
        means = self._reduced_means(periods)
        shifts = _image_shifts(periods, self.n_images)
        log_pi = self.log_pi - logsumexp(self.log_pi)
        k, d = means.shape
        if self.chols.shape[0] == 1:
            chol = self.chols[0]
            inv = solve_triangular(chol, jnp.eye(d), lower=True)
            const = (
                log_pi
                - jnp.sum(jnp.log(jnp.diag(chol)))
                - 0.5 * d * math.log(2.0 * math.pi)
            )
            centers = means @ inv.T
            c2 = jnp.sum(centers**2, axis=-1)
            white_shifts = shifts @ inv.T

            def block(rows):
                y = (rows @ inv.T)[:, None, :] + white_shifts[None]
                d2 = (
                    jnp.sum(y**2, axis=-1)[..., None]
                    + c2
                    - 2.0 * jnp.einsum("bsd,kd->bsk", y, centers)
                )
                return logsumexp(-0.5 * d2 + const, axis=(1, 2))

        else:
            inv = jax.vmap(lambda c: solve_triangular(c, jnp.eye(d), lower=True))(
                self.chols
            )
            log_det = jnp.sum(jnp.log(jnp.diagonal(self.chols, axis1=1, axis2=2)), -1)
            const = log_pi - log_det - 0.5 * d * math.log(2.0 * math.pi)

            def block(rows):
                shifted = rows[:, None, :] + shifts[None]
                diff = shifted[:, :, None, :] - means[None, None]
                y = jnp.einsum("kij,bskj->bski", inv, diff)
                return logsumexp(-0.5 * jnp.sum(y**2, axis=-1) + const, axis=(1, 2))

        return _blockwise(block, z, shifts.shape[0] * k)

    def sample(self, key, fit, n):
        """Draw ``n`` points, reduce them to the cell and return the wrapped density.

        Components are picked by the inverse CDF of the weights, one uniform per draw
        (``jax.random.categorical`` draws ``n * K`` Gumbel variates, which at
        ``K = 256`` costs more than the rest of a round); a component of zero weight is
        never picked.
        """
        periods = fit.periods()
        k_comp, k_eps = jax.random.split(key)
        cdf = jnp.cumsum(jnp.exp(self.log_pi))
        u = jax.random.uniform(k_comp, (n,), dtype=cdf.dtype)
        comp = jnp.minimum(jnp.searchsorted(cdf, u, side="right"), cdf.shape[0] - 1)
        eps = jax.random.normal(k_eps, (n, self.dim), jnp.float64)
        chol = self.chols[jnp.where(self.chols.shape[0] == 1, 0, comp)]
        z = self._reduced_means(periods)[comp] + jnp.einsum("nij,nj->ni", chol, eps)
        z = _wrap_cell(z, periods)
        return z, self.log_prob(z, fit)


class DefensiveProposal(AbstractDensityProposal):
    """The mixture ``fraction * prior + (1 - fraction) * base``.

    The prior component keeps the proposal density positive wherever the prior is, so
    the weights of a base that missed a region stay bounded there.

    Attributes:
        base: The proposal that carries ``1 - fraction`` of the draws.
        fraction: Share of draws from the prior component.
        prior: The prior component, the fit's prior (:class:`PriorProposal`) unless a
            proposal over a sub-vector of the fit supplies its own.
    """

    base: AbstractDensityProposal
    fraction: float = eqx.field(static=True, default=0.1)
    prior: AbstractDensityProposal = eqx.field(default_factory=PriorProposal)

    def __check_init__(self):
        """Refuse a fraction outside the open unit interval."""
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {self.fraction}")

    def log_prob(self, z, fit):
        """Log density of the two-component mixture."""
        return jnp.logaddexp(
            math.log(self.fraction) + self.prior.log_prob(z, fit),
            math.log1p(-self.fraction) + self.base.log_prob(z, fit),
        )

    def sample(self, key, fit, n):
        """Draw each point from the prior with probability ``fraction``."""
        k_base, k_prior, k_pick = jax.random.split(key, 3)
        z_base, _ = self.base.sample(k_base, fit, n)
        z_prior, _ = self.prior.sample(k_prior, fit, n)
        pick = jax.random.uniform(k_pick, (n, 1)) < self.fraction
        z = jnp.where(pick, z_prior, z_base)
        return z, self.log_prob(z, fit)

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive_fraction):
        """The default refit, keeping this proposal's own prior component."""
        refit = super().adapt(
            key,
            z,
            log_weights,
            fit,
            n_centers=n_centers,
            defensive_fraction=defensive_fraction,
        )
        return eqx.tree_at(lambda d: d.prior, refit, self.prior)
