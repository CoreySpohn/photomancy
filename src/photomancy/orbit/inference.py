"""One-planet chart densities and the orbit problem every backend consumes."""

import dataclasses
import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, Mearth2kg, two_pi
from hwoutils.epoch import Epoch
from jax.flatten_util import ravel_pytree
from orbix.system import (
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
)

from photomancy.priors import LOG_HAAR
from photomancy.system_density import SystemLogDensity

# Chart name that posteriors over the periapsis elements of
# ``FrameOrbit.to_periapsis_elements`` (stacked as columns) declare.
ORBIT_ROWS_CHART = "OrbitRows"


def orbit_chart_log_volume(chart, coords):
    """Chart pullback of e de d(cos i) dW dw dM dln(a/AU) with unnormalized Haar.

    The chart volumes are exact against normalized Haar orientation, so adding
    the log total Haar measure 8*pi^2 gives the density in one angular cell.
    The classical and eccentricity-vector charts are restricted to that cell.
    """
    volume = chart.log_volume(coords) + LOG_HAAR
    if isinstance(chart, CartesianChart):
        nonradial = jnp.any(jnp.cross(coords[..., :3], coords[..., 3:]) != 0, axis=-1)
        return jnp.where(nonradial, volume, -jnp.inf)
    if isinstance(chart, FrameChart):
        return volume
    if isinstance(chart, ClassicalChart):
        angles, low = coords[..., 3:], 0.0
    elif isinstance(chart, EccentricityVectorChart):
        # from_orbit recovers node and longitude with atan2.
        angles, low = coords[..., 4:], -math.pi
    else:
        raise ValueError(f"Unsupported fixed-mass orbit chart {type(chart).__name__}")
    in_cell = jnp.all((angles >= low) & (angles < low + two_pi), axis=-1)
    return jnp.where(in_cell, volume, -jnp.inf)


class _ChartSystemBuilder(eqx.Module):
    chart: ClassicalChart | EccentricityVectorChart | FrameChart | CartesianChart
    build: Callable
    star_mass_kg: float | jax.Array

    def __call__(self, params, **body):
        mass_kg = self.star_mass_kg + Mearth2kg * body.get("mass_Mearth", 0.0)
        orbit = self.chart.to_orbit(params["orbit"], mu_AU3_per_d2=G * mass_kg)
        return self.build(orbit, **body)


def _orbit_volume(chart, params):
    return orbit_chart_log_volume(chart, params["orbit"])


def build_chart_logdensity(
    chart,
    build,
    prior,
    *,
    star_mass_kg,
    terms=(),
    scalars=None,
    body_inputs=(),
    term_inputs=None,
):
    """Construct a one-planet density with a chart and named scalar coordinates.

    Supply a position dict with an ``orbit`` vector and a leaf per scalar.
    ``build(orbit, **body)`` constructs the system scored by ``prior(system)``.
    A sampled ``mass_Mearth`` goes into both the two-body orbit and the builder.
    List other physical properties (e.g. radius_AU and albedo) in body_inputs.
    Bind instrument coordinates to likelihood keywords with term_inputs, e.g.
    ``{0: ("jitter_m_per_s",)}``. An empty terms tuple gives a prior-only fit.
    """
    scalars = {} if scalars is None else scalars
    if "mass_Mearth" in scalars and "mass_Mearth" not in body_inputs:
        body_inputs = (*body_inputs, "mass_Mearth")
    return SystemLogDensity(
        _ChartSystemBuilder(chart, build, star_mass_kg),
        tuple(terms),
        prior,
        eqx.Partial(_orbit_volume, chart),
        scalars,
        body_inputs,
        {} if term_inputs is None else term_inputs,
    )


class ChartOrbitProblem(eqx.Module):
    """Ravel a one-planet chart density for fitting, sampling and prediction.

    Pass the output of ``build_chart_logdensity`` and one position specifying
    the shape of each coordinate. The flat vector follows JAX's sorted dict
    order. Use ``logdensity`` for gradient backends, and ``logprior`` and
    ``loglikelihood`` for tempered SMC with particles drawn from the prior.

    The orbits are built at day zero of the optional ``epoch`` (an
    :class:`~hwoutils.epoch.Epoch`): every time handed to the density, and every
    time a posterior of this problem refers to, is days from it. :meth:`run`
    and :meth:`declare` stamp the chart and epoch on a posterior, and
    :meth:`require` checks them before a consumer reads one.
    """

    density: SystemLogDensity
    position_template: dict
    unflatten: Callable
    param_names: tuple[str, ...]
    epoch: Epoch | None = eqx.field(static=True)

    def __init__(self, density, position_template, epoch=None):
        """Bind the coordinate layout used for all flat backend positions."""
        if not isinstance(density.build, _ChartSystemBuilder):
            raise ValueError("Construct the density with build_chart_logdensity")
        if set(position_template) != {"orbit", *density.scalars}:
            raise ValueError("Position must contain orbit and the declared scalars")
        position = jax.tree.map(jnp.asarray, position_template)
        if position["orbit"].shape != (density.build.chart.n_coords,):
            raise ValueError("Expected one orbit vector for the selected chart")
        if any(not jnp.issubdtype(v.dtype, jnp.floating) for v in position.values()):
            raise ValueError("Sampler coordinates must have floating-point dtype")
        self.density = density
        self.epoch = epoch
        self.position_template = position
        _, self.unflatten = ravel_pytree(position)
        self.param_names = tuple(
            label
            for name in sorted(position)
            for label in (
                (name,)
                if position[name].ndim == 0
                else tuple(f"{name}[{i}]" for i in range(position[name].size))
            )
        )

    @property
    def chart(self):
        """The chart used to build and score the orbit."""
        return self.density.build.chart

    @property
    def chart_name(self):
        """Name of the chart class, as a posterior of this problem declares it."""
        return type(self.chart).__name__

    @property
    def star_mass_kg(self):
        """Stellar mass used in the two-body orbit."""
        return self.density.build.star_mass_kg

    def declare(self, posterior):
        """Return ``posterior`` with this problem's chart and epoch declared."""
        return dataclasses.replace(posterior, chart=self.chart_name, epoch=self.epoch)

    def run(self, backend, init, key=None):
        """Run ``backend`` on :meth:`logdensity` and declare the chart and epoch."""
        return self.declare(backend.run(self.logdensity, init, key))

    def require(self, posterior):
        """Raise ``ValueError`` unless ``posterior`` matches this chart and epoch.

        A posterior from a bare backend call (``backend.run(problem.logdensity,
        ...)``) is undeclared and refused: pass it through :meth:`declare`, or run
        the backend with :meth:`run`. The epoch is checked only when the problem
        has one.
        """
        if posterior.chart is None:
            raise ValueError(
                "posterior declares no chart: run the backend with problem.run(...) "
                "or pass the posterior through problem.declare(...)"
            )
        posterior.require(chart=self.chart_name, epoch=self.epoch)

    def flatten(self, params):
        """Pack one position with the same coordinate shapes as the template."""
        if params.keys() != self.position_template.keys() or any(
            jnp.shape(params[k]) != v.shape for k, v in self.position_template.items()
        ):
            raise ValueError("Position keys and shapes must match the chart template")
        return ravel_pytree(params)[0]

    def logdensity(self, z):
        """Joint density against the flat chart differential."""
        return self.density(self.unflatten(z))

    def logprior(self, z):
        """Physical and scalar priors with their chart volumes."""
        return self.density.logprior(self.unflatten(z))

    def loglikelihood(self, z):
        """Sum of the fitted system's data terms."""
        return self.density.loglikelihood(self.unflatten(z))

    def to_physical(self, z):
        """Decode one flat chart draw into physical orbit and nuisance values."""
        return _chart_physical(self.density, self.unflatten(z))


def _chart_physical(density, params):
    """Decode a one-planet system and all fitted scalars at one chart position.

    Report angles in radians, mass in kg, semimajor axis and radius in AU, and
    period and periapsis time in days relative to the system's reference epoch.
    Instrument nuisance coordinates retain their declared names and shapes.
    """
    from hwoutils.constants import Msun2kg
    from orbix.system import System

    system, scalars = density._system_and_scalars(params)
    dynamics = system if isinstance(system, System) else system.dynamics
    if dynamics.orbits.a_AU.shape != (1,):
        raise ValueError("Chart physical decoding requires a one-planet system")
    orbit = jax.tree.map(lambda leaf: leaf[0], dynamics.orbits)
    elements = orbit.to_classical()
    inclination = elements["i_rad"]
    mass_kg = dynamics.planets.mass_Msun[0] * Msun2kg
    physical = dict(
        T=orbit.period_d,
        a=orbit.a_AU,
        e=orbit.e,
        cos_i=jnp.cos(inclination),
        W=elements["W_rad"] % two_pi,
        cos_w=jnp.cos(elements["w_rad"]),
        sin_w=jnp.sin(elements["w_rad"]),
        tp=orbit.epoch_d - (elements["M0_rad"] % two_pi) * orbit.period_d / two_pi,
        Mp=mass_kg,
        Mp_sini=mass_kg * jnp.sin(inclination),
    )
    if "radius_AU" in scalars:
        physical["Rp"] = scalars["radius_AU"]
    if "albedo" in scalars:
        physical["Ag"] = scalars["albedo"]
    if "Rp" in physical and "Ag" in physical:
        physical["Lambda"] = physical["Ag"] * physical["Rp"] ** 2
    physical.update(
        (name, value)
        for name, value in scalars.items()
        if name not in {"mass_Mearth", "radius_AU", "albedo"}
    )
    return physical
