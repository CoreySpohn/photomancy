"""Orbit proposals for a Thiele-Innes fit: scale and rotate onto one astrometric row.

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
from orbix.system.core import GM_SUN_AU3_PER_D2

from photomancy.infer import AbstractDensityProposal
from photomancy.measurements import RelativeAstrometryTerm
from photomancy.orbit.completion import (
    _block_of,
    _flat_layout,
    block_coordinates,
    draw_eccentricity,
    eccentricity_table,
    log_density_e,
)

__all__ = ["ScaleRotateProposal"]

_CLASSICAL = ClassicalChart()


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
    with a period outside the law's bounds has zero target density; it is returned as a
    NaN draw with ``log_q = +inf``, which an importance backend weights zero, so the
    proposal mass is below one and the weights stay unbiased.

    Valid for a fit whose only free entry is one ``ThieleInnesBlock`` and whose terms
    include one ``RelativeAstrometryTerm``. A fit with any other free entry (the planet
    radius of a report fit, say) is refused by name: draw it with
    :func:`~photomancy.orbit.backends.collapsed`, which completes a proposed leaf.
    Reference: Blunt et al. (2017), AJ 153, 229.

    Fails if the Jacobian is wrong: the evidence of a fit with no terms is not zero.

    Attributes:
        anchor: Row of the astrometry term to anchor on; ``None`` takes the first
            valid row.
    """

    anchor: int | None = eqx.field(static=True, default=None)

    def _setup(self, fit):
        """Block, term row data, flat positions of the block coordinates and sizes."""
        key, block = _block_of(fit)
        others = sorted(k for k in fit.free if k != key)
        if others:
            raise ValueError(
                f"ScaleRotateProposal draws only the orbit block {key!r}; the free "
                f"entries {others} need a proposal that completes them (collapsed)"
            )
        terms = [t for t in fit.terms if isinstance(t, RelativeAstrometryTerm)]
        if len(terms) != 1:
            raise ValueError("ScaleRotateProposal needs one RelativeAstrometryTerm")
        data = terms[0].data
        row = jnp.argmax(data.is_valid) if self.anchor is None else self.anchor
        names, sizes = _flat_layout(fit)
        position = [names.index((key, n)) for n in block.coordinate_names]
        model = fit.model
        system = resolve(model, block.system)
        return block, data, row, position, system, len(sizes)

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

    def _place(self, z7, position, dim):
        """Flat vectors from the block coordinates ``z7 (n, 7)`` in chart order."""
        z = jnp.zeros((z7.shape[0], dim))
        return z.at[:, jnp.asarray(position)].set(z7)

    def sample(self, key, fit, n):
        """Draw ``n`` orbits; return ``(z, log_q)``."""
        block, data, row, position, system, dim = self._setup(fit)
        held = resolve(fit.model, block.owns)
        planets = resolve(fit.model, block.reads[0])
        inner = jnp.sum(planets[..., : block.rows[0] + 1], axis=-1)
        epoch = held.epoch_d[..., block.rows[0]]
        t_anchor = data.times[row]
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
        xc, log_jac = block_coordinates(block, fit.model, y)
        log_q = self._density(
            block, data, row, e, t, mass, enclosed, xc, log_jac, table
        )
        log_q = jnp.where(inside, log_q, jnp.inf)
        return self._place(y, position, dim), log_q

    def log_prob(self, z, fit):
        """Log density of flat vectors ``z (n, d)`` (zero outside the period bounds)."""
        block, data, row, position, system, _ = self._setup(fit)
        held = resolve(fit.model, block.owns)
        planets = resolve(fit.model, block.reads[0])
        inner = jnp.sum(planets[..., : block.rows[0] + 1], axis=-1)
        epoch = held.epoch_d[..., block.rows[0]]
        y = z[:, jnp.asarray(position)]
        x, log_jac = block_coordinates(block, fit.model, y)
        r2 = x[:, 1] ** 2 + x[:, 2] ** 2
        e = jnp.sqrt(r2 / (1.0 + r2))
        enclosed = block.chart.implied_mu(x) / GM_SUN_AU3_PER_D2
        orbit = block.chart.to_orbit(x, mu_AU3_per_d2=1.0, epoch_d=epoch)
        east, north = orbit.offsets_arcsec(data.times[row], system.frame.dist_pc)
        t = jnp.stack([east, north], -1)
        log_q = self._density(
            block,
            data,
            row,
            e,
            t,
            enclosed - inner,
            enclosed,
            x,
            log_jac,
            eccentricity_table(block),
        )
        lo, hi = block.chart_support(fit.model)["ln_size"]
        return jnp.where((x[:, 0] > lo) & (x[:, 0] < hi), log_q, -jnp.inf)
