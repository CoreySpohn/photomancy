"""Orbit proposals for a Thiele-Innes fit: the block's own law, and scale and rotate.

``ShapePriorProposal`` draws the orbit shape, the three nonlinear block coordinates
``(ln n, ux, uy)``, from the block's own period and eccentricity laws (the base of a
completed proposal). The eccentricity is drawn by inverting a table of the law's
distribution function.

``ScaleRotateProposal`` draws an orbit shape from the block's own law and fixes the
size and the node so that the orbit passes through a sky position drawn from the error
kernel of one astrometric row (the anchor), the construction of Blunt et al. (2017)
("Orbits for the Impatient"), which is exact in the law and efficient when one or two
rows constrain the orbit. Its density in the flat vector is closed form, so it is a
proposal like any other: ``Importance(ScaleRotateProposal())`` weights its draws to the
fit's own density.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi
from hwoutils.model_paths import resolve
from orbix.system import ClassicalChart
from orbix.system.charts import disk_to_plane
from orbix.system.core import GM_SUN_AU3_PER_D2

from photomancy.infer import (
    AbstractDensityProposal,
    DefensiveProposal,
    MixtureProposal,
)
from photomancy.measurements import RelativeAstrometryTerm
from photomancy.orbit.completion import _block_of, _flat_layout

__all__ = ["ScaleRotateProposal", "ShapePriorProposal"]

_TABLE = 2**13

_CLASSICAL = ClassicalChart()


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


class ShapePriorProposal(AbstractDensityProposal):
    """The block's own law of ``(ln n, ux, uy)`` as a proposal over those coordinates.

    ``ln n`` is uniform on the period bounds, the eccentricity has the law's marginal
    (a table of its distribution function, so ``log_q`` is the density of the draws
    exactly) and the eccentricity angle is uniform, all in the stored coordinates of the
    Thiele-Innes block. It is given a :class:`Completion` in place of the fit, whose
    ``shape`` is exactly the three block coordinates (``Completion.from_fit`` refuses
    any other free entry).

    Attributes:
        bandwidth: Factor on the Silverman bandwidth of the refit kernels, applied in
            every adaptation round (``MixtureProposal.from_cloud``). The default 2 makes
            the kernels over-dispersed: the Silverman width fits the bulk of the cloud
            and leaves the proposal lighter-tailed than the posterior where the centers
            thin out (the ends of the period-eccentricity ridge, the edge of a mass
            peak), which gives a heavy importance-weight tail.

    Fails if the draws and ``log_q`` disagree (the weights of a prior-only fit are not
    one).
    """

    bandwidth: float = eqx.field(static=True, default=2.0)

    def log_prob(self, z, fit):
        """Log density of ``z (n, 3)`` in the stored coordinates."""
        block = fit.block
        y = jnp.concatenate([z, _pad(z)], -1)
        x, log_jac = block.chart.constrain(y, block.chart_support(fit.model))
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
        support = block.chart_support(fit.model)
        y = block.chart.unconstrain(jnp.concatenate([x, _pad(x)], -1), support)
        z = y[:, :3]
        return z, self.log_prob(z, fit)

    def adapt(self, key, z, log_weights, fit, *, n_centers, defensive_fraction):
        """Refit a mixture on the cloud, with this prior as the defensive component."""
        mixture = MixtureProposal.from_cloud(
            key,
            z,
            log_weights,
            n_centers=n_centers,
            bandwidth=self.bandwidth,
            periods=fit.periods(),
        )
        return DefensiveProposal(
            mixture, defensive_fraction, prior=self, bandwidth=self.bandwidth
        )


def _pad(a):
    """Zeros for the four sky constants, matching the batch of ``a``."""
    return jnp.zeros((*a.shape[:-1], 4))


class ScaleRotateProposal(AbstractDensityProposal):
    """Orbits through a sky position drawn near one astrometric row.

    A draw takes the eccentricity, ``cos i``, the argument of periapsis and the mean
    anomaly at the anchor time from the block's isotropic law, the star mass from its
    law, and the anchor position ``t`` from the Gaussian of the row. The size ``a`` and
    the node ``W`` that put the orbit through ``t`` are solved, the mean motion follows
    from the enclosed mass, and the orbit is moved to the model epoch and written in the
    Thiele-Innes chart. At fixed shape the map ``t -> (ln a, W)`` has Jacobian
    ``1 / rho^2`` with ``rho = |t|``, so the density in the stored coordinates is

    ``log 2 + log p_e - log e + log N(t) + 2 log rho + log p_M + log M_enclosed
    + log_volume + log |dx/dy|``

    (the factor 2 is the two line-of-sight sheets that map to one chart point). A draw
    with a period outside the law's bounds has zero target density; it is a dead draw,
    NaN with ``log_q = -inf`` from ``sample`` (``log_prob`` is ``-inf`` there) and log
    weight ``-inf`` from ``weigh``, so the proposal mass is below one and the weights
    stay unbiased.

    Valid for a fit whose only free entry is one ``ThieleInnesBlock`` and whose terms
    include one ``RelativeAstrometryTerm``. A fit with any other free entry (the planet
    radius of a report fit, say) is refused by name: draw it with
    :func:`~photomancy.orbit.backends.any_epoch`, which completes a proposed leaf.
    Reference: Blunt et al. (2017), AJ 153, 229.

    Fails if the Jacobian is wrong: the evidence of a fit with no terms is not zero.

    Attributes:
        anchor: Row of the astrometry term to anchor on; ``None`` takes the first
            valid row.
    """

    anchor: int | None = eqx.field(static=True, default=None)

    def _setup(self, fit):
        """Block, anchor data and row, flat positions, system, size, mass, epoch.

        ``inner`` is the mass interior to the block's row and ``epoch`` the row's epoch.
        """
        key, block = _block_of(fit)
        others = sorted(k for k in fit.free if k != key)
        if others:
            raise ValueError(
                f"ScaleRotateProposal draws only the orbit block {key!r}; the free "
                f"entries {others} need a proposal that completes them (any_epoch)"
            )
        terms = [t for t in fit.terms if isinstance(t, RelativeAstrometryTerm)]
        if len(terms) != 1:
            raise ValueError("ScaleRotateProposal needs one RelativeAstrometryTerm")
        data = terms[0].data
        row = jnp.argmax(data.is_valid) if self.anchor is None else self.anchor
        names, sizes = _flat_layout(fit)
        position = [names.index((key, n)) for n in block.coordinate_names]
        planets = resolve(fit.model, block.reads[0])
        inner = jnp.sum(planets[..., : block.rows[0] + 1], axis=-1)
        epoch = resolve(fit.model, block.owns).epoch_d[..., block.rows[0]]
        system = resolve(fit.model, block.system)
        return block, data, row, position, system, len(sizes), inner, epoch

    def _log_gaussian(self, data, row, east, north):
        """Log density of the anchor row's error kernel at ``(east, north)``."""
        sx, sy, rho = data.ra_err[row], data.dec_err[row], data.corr[row]
        dx, dy = (east - data.ra[row]) / sx, (north - data.dec[row]) / sy
        quad = (dx**2 - 2 * rho * dx * dy + dy**2) / (1.0 - rho**2)
        return -0.5 * quad - jnp.log(two_pi * sx * sy * jnp.sqrt(1.0 - rho**2))

    def _density(self, block, data, row, e, t, mass, enclosed, x, log_jac, table):
        """Closed-form log density in the stored coordinates (see the class)."""
        rho = jnp.hypot(t[:, 0], t[:, 1])
        return (
            jnp.log(2.0)
            + log_density_e(e, table)
            - jnp.log(e)
            + self._log_gaussian(data, row, t[:, 0], t[:, 1])
            + 2.0 * jnp.log(rho)
            + block.star_mass_Msun.log_prob(mass)
            + jnp.log(enclosed)
            + block.chart.log_volume(x)
            + log_jac
        )

    def _draw(self, key, fit, n):
        """Draw ``n`` orbits; return ``(z, log_q, inside)``, ``z`` NaN outside."""
        block, data, row, position, system, dim, inner, epoch = self._setup(fit)
        t_anchor = data.time_d[row]
        dist_pc = system.frame.dist_pc
        k = jax.random.split(key, 7)
        table = eccentricity_table(block)
        e = draw_eccentricity(k[0], table, n)
        cos_i = jax.random.uniform(k[1], (n,), minval=-1.0, maxval=1.0)
        omega = jax.random.uniform(k[2], (n,), maxval=two_pi)
        m_ref = jax.random.uniform(k[3], (n,), maxval=two_pi)
        mass = block.star_mass_Msun.sample(k[4], (n,))
        enclosed = mass + inner
        mu = GM_SUN_AU3_PER_D2 * enclosed
        sx, sy, corr = data.ra_err[row], data.dec_err[row], data.corr[row]
        z1 = jax.random.normal(k[5], (n,))
        z2 = jax.random.normal(k[6], (n,))
        east = data.ra[row] + sx * z1
        north = data.dec[row] + sy * (
            corr * z1 + jnp.sqrt(jnp.maximum(1.0 - corr**2, 0.0)) * z2
        )
        t = jnp.stack([east, north], -1)
        zero = jnp.zeros_like(e)
        logit_e = jnp.log(e) - jnp.log1p(-e)
        shape = jnp.stack(
            [zero, logit_e, jnp.arctanh(cos_i), zero, omega, m_ref], axis=-1
        )
        unit = _CLASSICAL.to_orbit(shape, mu_AU3_per_d2=mu, epoch_d=t_anchor)
        position_AU, _ = unit.to_cartesian()
        x0, y0 = position_AU[..., 0] / dist_pc, position_AU[..., 1] / dist_pc
        r0 = jnp.maximum(jnp.hypot(x0, y0), 1e-30)
        log_a = jnp.log(jnp.hypot(east, north) / r0)
        node = (jnp.arctan2(east, north) - jnp.arctan2(x0, y0)) % two_pi
        n_rad = jnp.sqrt(mu / jnp.exp(3.0 * log_a))
        phase = (m_ref + n_rad * (epoch - t_anchor)) % two_pi
        final = jnp.stack([log_a, logit_e, jnp.arctanh(cos_i), node, omega, phase], -1)
        orbit = _CLASSICAL.to_orbit(final, mu_AU3_per_d2=mu, epoch_d=epoch)
        x = block.chart.from_orbit(orbit)
        support = block.chart_support(fit.model)
        lo, hi = support["ln_size"]
        inside = (x[:, 0] > lo) & (x[:, 0] < hi)
        y = jnp.where(inside[:, None], block.chart.unconstrain(x, support), jnp.nan)
        x, log_jac = block.chart.constrain(y, support)
        log_q = self._density(block, data, row, e, t, mass, enclosed, x, log_jac, table)
        z = jnp.zeros((n, dim)).at[:, jnp.asarray(position)].set(y)
        return z, log_q, inside

    def sample(self, key, fit, n):
        """Draw ``n`` orbits; return ``(z, log_q)`` (NaN and ``-inf`` when dead)."""
        z, log_q, inside = self._draw(key, fit, n)
        return z, jnp.where(inside, log_q, -jnp.inf)

    def weigh(self, key, fit, n, *, batch_size=None):
        """Draw and weigh ``n`` orbits; a draw outside the period bounds is dead.

        The proposal has a density, so ``valid`` (the fourth value) is true.
        """
        z, log_q, inside = self._draw(key, fit, n)
        x, volume = fit.from_flat(z)
        log_target = fit.log_density(x, batch_size=batch_size)
        log_target = jnp.where(inside, log_target, -jnp.inf)
        log_w = jnp.where(inside, log_target + volume - log_q, -jnp.inf)
        return z, log_w, log_target, jnp.asarray(True)

    def log_prob(self, z, fit):
        """Log density of flat vectors ``z (n, d)`` (zero outside the period bounds)."""
        block, data, row, position, system, _, inner, epoch = self._setup(fit)
        support = block.chart_support(fit.model)
        x, log_jac = block.chart.constrain(z[:, jnp.asarray(position)], support)
        r2 = x[:, 1] ** 2 + x[:, 2] ** 2
        e = jnp.sqrt(r2 / (1.0 + r2))
        enclosed = block.chart.implied_mu(x) / GM_SUN_AU3_PER_D2
        orbit = block.chart.to_orbit(x, mu_AU3_per_d2=1.0, epoch_d=epoch)
        east, north = orbit.offsets_arcsec(data.time_d[row], system.frame.dist_pc)
        t = jnp.stack([east, north], -1)
        table = eccentricity_table(block)
        mass = enclosed - inner
        log_q = self._density(block, data, row, e, t, mass, enclosed, x, log_jac, table)
        lo, hi = support["ln_size"]
        return jnp.where((x[:, 0] > lo) & (x[:, 0] < hi), log_q, -jnp.inf)
