"""Fixed-mass one-planet chart problems and fits shared by the orbit tests."""

import jax
import jax.numpy as jnp
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from orbix.system import Body, ClassicalChart, SkyFrame, System

from photomancy import RelativeAstrometryTerm
from photomancy.orbit.inference import ChartOrbitProblem, build_chart_logdensity
from photomancy.orbit.init import find_init
from photomancy.orbit.laplace import map_laplace_fit
from photomancy.priors import FactorizedOrbitPrior, LogUniformPrior, orbit_log_prior
from photomancy.system_density import ScalarCoordinate


def _first_orbit(system):
    return jax.tree.map(lambda leaf: leaf[0], system.orbits)


def classical_coordinates(a, e, cos_i, W, w, tp, period_d):
    """``ClassicalChart`` coordinates at day zero of classical elements.

    ``a`` in AU, angles in radians, periapsis time ``tp`` and ``period_d`` in days:
    ``(ln a, logit e, atanh cos i, W, w, M0)`` with ``M0 = n (0 - tp)``.
    """
    mean_anomaly = (2 * jnp.pi / period_d) * (0.0 - tp)
    return jnp.stack(
        [
            jnp.log(a),
            jnp.log(e) - jnp.log1p(-e),
            jnp.arctanh(cos_i),
            jnp.asarray(W) + 0.0,
            jnp.asarray(w) + 0.0,
            mean_anomaly % (2 * jnp.pi),
        ]
    )


def chart_problem(
    astrom,
    params,
    *,
    star_mass_kg,
    dist_pc,
    chart=None,
    period_range_d=(600.0, 1600.0),
    terms=None,
):
    """One-planet fixed-mass chart problem for relative astrometry.

    ``params`` is the dict of chart coordinates ``{"orbit": ...}`` that fixes the
    pytree structure and the problem's reference position.
    """
    chart = ClassicalChart() if chart is None else chart
    frame = SkyFrame(dist_pc, Epoch())

    def build(orbit):
        return System(
            Body(1.0), Body(jnp.zeros(1)), jax.tree.map(lambda x: x[None], orbit), frame
        )

    def prior(system):
        orbit = jax.tree.map(lambda leaf: leaf[0], system.orbits)
        return orbit_log_prior(orbit, period_range_d=period_range_d)

    density = build_chart_logdensity(
        chart,
        build,
        prior,
        star_mass_kg=star_mass_kg,
        terms=(RelativeAstrometryTerm(astrom),) if terms is None else terms,
    )
    return ChartOrbitProblem(density, params)


def blind_fit(
    astrom, *, star_mass_kg, dist_pc, log_T_range, k=1, n_steps=100, period_range_d=None
):
    """MAP and Laplace fit from the blind Thiele-Innes initializer.

    Returns ``(posterior, problem)`` with the chart posterior declared.
    """
    chart = ClassicalChart()
    starts, _ = find_init(
        astrom, star_mass_kg, dist_pc, k=k, log_T_range=log_T_range, chart=chart
    )
    if period_range_d is None:
        period_range_d = tuple(float(10.0**x) for x in log_T_range)
    problem = chart_problem(
        astrom,
        {"orbit": starts[0]},
        star_mass_kg=star_mass_kg,
        dist_pc=dist_pc,
        chart=chart,
        period_range_d=period_range_d,
    )
    init = problem.flatten({"orbit": starts[0]})
    return map_laplace_fit(problem, init, n_steps=n_steps), problem


def mass_problem(
    terms,
    orbit_coordinates,
    mass_Mearth,
    *,
    star_mass_kg,
    dist_pc,
    chart=None,
    period_range_d=(600.0, 1600.0),
    mass_bounds=(0.01, 1.0e4),
    ecc_prior="kipping13",
):
    """One-planet chart problem whose planet mass is sampled in ``ln(Mearth)``.

    Radial velocity, stellar astrometry and proper-motion anomaly terms read the
    mass; ``orbit_coordinates`` and ``mass_Mearth`` give the reference position.
    """
    chart = ClassicalChart() if chart is None else chart
    frame = SkyFrame(dist_pc, Epoch())

    def build(orbit, mass_Mearth):
        return System(
            Body(star_mass_kg / Msun2kg),
            Body(jnp.atleast_1d(mass_Mearth * Mearth2kg / Msun2kg)),
            jax.tree.map(lambda x: x[None], orbit),
            frame,
        )

    density = build_chart_logdensity(
        chart,
        build,
        FactorizedOrbitPrior(_first_orbit, period_range_d, ecc_prior),
        star_mass_kg=star_mass_kg,
        terms=tuple(terms),
        scalars={"mass_Mearth": ScalarCoordinate(LogUniformPrior(mass_bounds))},
    )
    position = {"orbit": orbit_coordinates, "mass_Mearth": jnp.log(mass_Mearth)}
    return ChartOrbitProblem(density, position)


def fitted_orbit(problem, flat_position):
    """Orbit and mass (Mearth) decoded from one flat position of a mass problem."""
    position = problem.unflatten(flat_position)
    system = problem.density.build_system(position)
    return jax.tree.map(lambda leaf: leaf[0], system.orbits), jnp.exp(
        position["mass_Mearth"]
    )


def scale_problem(
    terms,
    orbit_coordinates,
    *,
    star_mass_kg,
    dist_pc,
    chart=None,
    period_range_d=(600.0, 1600.0),
    term_inputs=None,
    radius_bounds=(1e-5, 10**-2.5),
    albedo_bounds=(0.01, 1.0),
):
    """One-planet scene problem with log-uniform radius (AU) and albedo scalars.

    The planet is massless; the scene supplies its Lambertian photometry. The
    position is ``{"orbit": ..., "radius_AU": ln radius, "albedo": ln albedo}``
    (with the reference radius and albedo at the geometric means of the bounds).
    """
    from functools import partial

    from hwoutils.constants import Rearth2AU
    from skyscapes.physical_model import LambertianPhysicalModel
    from skyscapes.scene import FlatStar, Planet, Scene

    from photomancy.priors import log_uniform_prior

    chart = ClassicalChart() if chart is None else chart
    frame = SkyFrame(dist_pc, Epoch())

    def build(orbit, *, radius_AU, albedo):
        dynamics = System(
            Body(star_mass_kg / Msun2kg),
            Body(jnp.zeros(1)),
            jax.tree.map(lambda x: x[None], orbit),
            frame,
        )
        planet = Planet(
            Rp_Rearth=jnp.atleast_1d(radius_AU / Rearth2AU),
            physical_model=LambertianPhysicalModel(Ag=jnp.atleast_1d(albedo)),
        )
        return Scene(FlatStar(1.0), (planet,), dynamics)

    def prior(scene):
        orbit = jax.tree.map(lambda leaf: leaf[0], scene.dynamics.orbits)
        return orbit_log_prior(orbit, period_range_d=period_range_d)

    density = build_chart_logdensity(
        chart,
        build,
        prior,
        star_mass_kg=star_mass_kg,
        terms=tuple(terms),
        scalars={
            "radius_AU": ScalarCoordinate(
                partial(log_uniform_prior, bounds=radius_bounds)
            ),
            "albedo": ScalarCoordinate(
                partial(log_uniform_prior, bounds=albedo_bounds)
            ),
        },
        body_inputs=("radius_AU", "albedo"),
        term_inputs=term_inputs or {},
    )
    position = {
        "orbit": orbit_coordinates,
        "radius_AU": jnp.log(jnp.sqrt(radius_bounds[0] * radius_bounds[1])),
        "albedo": jnp.log(jnp.sqrt(albedo_bounds[0] * albedo_bounds[1])),
    }
    return ChartOrbitProblem(density, position)
