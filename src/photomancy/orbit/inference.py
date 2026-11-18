"""One-planet chart densities and the NumPyro orbit inference route."""

import dataclasses
import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, Mearth2kg, two_pi
from hwoutils.epoch import Epoch
from jax.flatten_util import ravel_pytree
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import initialize_model
from orbix.system import (
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
)

from photomancy.orbit._numpyro_bridge import (
    _build_orbit_model,
    _init_dict_to_z_flat,
    make_constrain,
)
from photomancy.orbit.init import elements_to_sites
from photomancy.posterior import SamplePosterior
from photomancy.system_density import SystemLogDensity


def orbit_chart_log_volume(chart, coords):
    """Normalized chart pullback of e de d(cos i) dW dw dM dln(a/AU).

    Use one angular cell for the classical and eccentricity-vector charts.
    Both Frame lifts integrate to the unnormalized Haar volume 8*pi^2.
    The Cartesian determinant restores the constant omitted by orbix.
    """
    volume = chart.log_volume(coords)
    if isinstance(chart, CartesianChart):
        nonradial = jnp.any(jnp.cross(coords[..., :3], coords[..., 3:]) != 0, axis=-1)
        return jnp.where(nonradial, volume + math.log(16), -jnp.inf)
    if isinstance(chart, FrameChart):
        sd = chart.gauge_sd
        if sd is not None:
            if sd <= 0 or not math.isfinite(sd):
                raise ValueError("Frame gauge_sd must be positive and finite")
            # Integral over S^3 and the positive truncated Gaussian radius.
            positive_fraction = 0.5 * math.erfc(-1 / (math.sqrt(2) * sd))
            radial_integral = sd * math.sqrt(two_pi) * positive_fraction
            return volume + math.log(4 / radial_integral)
        return volume + math.log(8 * math.pi**2)
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


class OrbitProblem(eqx.Module):
    """An orbit fit reduced to a flat logdensity for the generic backends.

    Attributes:
        logdensity: ``z -> scalar`` log-density over the flat unconstrained position.
        to_physical: ``z -> dict`` mapping a flat position to named physical
            parameters (``T``, ``e``, ``a``, ...).
        init_to_z: ``init_dict -> z`` converting a blind-initializer dict (e.g. from
            ``photomancy.orbit.init.find_init``) to a flat unconstrained position.
        unflatten: ``z -> dict`` of raw (unconstrained) sample sites.
        constrain: ``raw-site dict -> physical-site dict`` applying each site's forward
            constraint bijector (the seam the EIG forward differentiates through).
        param_names: The unconstrained sample-site names, in flat order.
    """

    logdensity: Callable = eqx.field(static=True)
    to_physical: Callable = eqx.field(static=True)
    init_to_z: Callable = eqx.field(static=True)
    unflatten: Callable = eqx.field(static=True)
    constrain: Callable = eqx.field(static=True)
    param_names: tuple[str, ...] = eqx.field(static=True)


def build_numpyro_logdensity(
    Ms,
    dist_pc,
    *,
    rv_data=None,
    relative_astrom_data=None,
    stellar_astrom_data=None,
    pm_anomaly_data=None,
    null_data=None,
    imaging_data=None,
    custom_loglike: Callable[[dict], jax.Array | float] | None = None,
    custom_data=None,
    sample_planet_scale: bool = False,
    allow_overlap: bool = False,
    log_P_range=(1.0, 4.0),
    log_Mp_range=(-2.0, 4.0),
    log_Rp_range=(-5.0, -2.5),
    log_Ag_range=(-2.0, 0.0),
    ecc_prior="kipping13",
    jitter_scale=0.00017314568368055556,  # 1e-10 AU/day in m/s
    n_planets=1,
    seed=0,
):
    """Build a flat orbit ``logdensity`` (and its z<->physical maps) for a backend.

    Traces the NumPyro orbit model once (cached), pads the data to the model's
    ``MAX_*`` shapes, and binds the data into ``logdensity = -potential``. Feed the
    returned ``OrbitProblem.logdensity`` to any generic backend; seed multi-start /
    MCMC from ``init_to_z(find_init...(...))`` and map results back with
    ``to_physical``.

    Args:
        Ms: Stellar mass (kg).
        dist_pc: Distance to the system (parsec).
        rv_data: An :class:`~photomancy.orbit.data.RVData`, or ``None``.
        relative_astrom_data: A
            :class:`~photomancy.orbit.data.RelativeAstromData`, or ``None``.
        stellar_astrom_data: A
            :class:`~photomancy.orbit.data.StellarAstromData`, or ``None``.
        pm_anomaly_data: A
            :class:`~photomancy.orbit.data.PMAnomalyData`, or ``None``.
        null_data: A :class:`~photomancy.orbit.data.NullData`, or ``None``.
        imaging_data: An :class:`~photomancy.orbit.data.ImagingData`, or ``None``.
        custom_loglike: JAX-compatible ``params -> scalar`` data log-likelihood.
            Default None. Return a scalar without prior terms, or a ValueError
            is raised for a vector result.
            Receives physical orbit and planet-scale parameters, plus ``Ms``,
            ``dist_pc`` and ``custom_data``; see
            :func:`~photomancy.orbit.model.build_model` for keys, units and
            shapes. Absent built-in channels add no terms.
        custom_data: Pytree bound into the callback's parameter dictionary under
            ``"custom_data"`` (default None). Array leaves must support the
            callback's JAX transformations. Each custom fit owns its model trace.
        sample_planet_scale: Activate ``log_Rp``, ``log_Ag`` and ``log_Mp``
            without built-in channels, allowing prior-only fits. A custom
            likelihood also activates these sites. Default False.
        allow_overlap: Allow a custom likelihood together with any built-in
            channel. Default False; combining them otherwise raises ValueError.
            With True, all supplied likelihood terms are added.
        log_P_range: ``(min, max)`` for the ``log10(period/days)`` prior.
        log_Mp_range: ``(min, max)`` for the ``log10(mass/M_earth)`` prior.
        log_Rp_range: ``(min, max)`` for the ``log10(Rp/AU)`` prior.
        log_Ag_range: ``(min, max)`` for the ``log10(geometric albedo)`` prior.
        ecc_prior: Eccentricity prior name (see
            :func:`~photomancy.orbit.model.build_model`).
        jitter_scale: Scale for the HalfNormal RV-jitter prior (m/s).
        n_planets: Number of planets to fit (the model vmaps over planets and
            indexes astrometry by ``planet_id``).
        seed: PRNG seed for the one-time model trace.

    Returns:
        An :class:`OrbitProblem`.
    """
    model, model_args, cached = _build_orbit_model(
        Ms,
        dist_pc,
        rv_data=rv_data,
        relative_astrom_data=relative_astrom_data,
        stellar_astrom_data=stellar_astrom_data,
        pm_anomaly_data=pm_anomaly_data,
        null_data=null_data,
        imaging_data=imaging_data,
        custom_loglike=custom_loglike,
        custom_data=custom_data,
        sample_planet_scale=sample_planet_scale,
        allow_overlap=allow_overlap,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
        n_planets=n_planets,
        seed=seed,
    )
    if cached is None:
        info = initialize_model(
            jax.random.PRNGKey(seed), model, dynamic_args=True, model_args=model_args
        )
        z_template = info.param_info.z
        _, unflatten = ravel_pytree(z_template)
        cached = dict(
            potential_fn_factory=info.potential_fn,
            postprocess_fn_factory=info.postprocess_fn,
            inv_transforms={
                name: biject_to(site["fn"].support).inv
                for name, site in info.model_trace.items()
                if site["type"] == "sample" and not site.get("is_observed", False)
            },
            z_template=z_template,
            unflatten=unflatten,
            param_names=tuple(sorted(z_template)),
        )

    unflatten = cached["unflatten"]
    potential_fn = cached["potential_fn_factory"](*model_args)

    def logdensity(z):
        return -potential_fn(unflatten(z))

    try:
        raw_postprocess = cached["postprocess_fn_factory"](*model_args)
    except TypeError:
        raw_postprocess = cached["postprocess_fn_factory"]

    def to_physical(z):
        phys = raw_postprocess(unflatten(z))
        return {k: jnp.squeeze(v) for k, v in phys.items()}

    def init_to_z(init_dict):
        return _init_dict_to_z_flat(
            init_dict, cached["z_template"], cached["inv_transforms"]
        )

    # Forward (unconstrained -> physical) bijectors are the inverse of the cached
    # inverse transforms; the closure keeps them off the module as static array leaves.
    constrain = make_constrain(
        {name: t.inv for name, t in cached["inv_transforms"].items()}
    )

    return OrbitProblem(
        logdensity=logdensity,
        to_physical=to_physical,
        init_to_z=init_to_z,
        unflatten=unflatten,
        constrain=constrain,
        param_names=cached["param_names"],
    )


def to_unconstrained(posterior, problem, Ms):
    """Map a physical orbit SamplePosterior to the model's unconstrained z-space.

    OFTI / grid_search emit physical rows ``(a, e, cos_i, W, cos_w, sin_w, tp)``; the
    EIG needs the model's unconstrained ``z``. Converts ``a -> period`` (Kepler) then
    each row -> raw sites -> ``z`` via ``problem.init_to_z`` (vmapped, no Python loop).
    A true-mass ``Mp`` column (kg) is included in the Kepler inversion and carried
    into the model's mass site. Mass-free rows use stellar mass, matching the
    OFTI / grid_search orbit construction. Retained ``Rp`` (AU) and ``Ag``
    columns are carried into their logarithmic model sites as well.

    Args:
        posterior: A physical :class:`~photomancy.posterior.SamplePosterior` whose
            ``param_names`` include ``a, e, cos_i, W, cos_w, sin_w, tp``.
        problem: The :class:`OrbitProblem` defining the target ``z`` space.
        Ms: Stellar mass (kg), for the ``a -> period`` Kepler inversion.

    Returns:
        A :class:`~photomancy.posterior.SamplePosterior` in ``z`` space (``param_names``
        are ``problem.param_names``), preserving the input weights and evidence.
    """
    cols = {
        name: posterior.samples[:, i] for i, name in enumerate(posterior.param_names)
    }
    period = 2.0 * jnp.pi * jnp.sqrt(cols["a"] ** 3 / (G * (Ms + cols.get("Mp", 0.0))))

    def one(t, row):
        sites = elements_to_sites(
            t,
            *(row[k] for k in ("e", "cos_i", "W", "cos_w", "sin_w", "tp")),
            n_planets=1,
        )
        for name, unit in (("Mp", Mearth2kg), ("Rp", 1.0), ("Ag", 1.0)):
            if name in row:
                sites[f"log_{name}"] = jnp.atleast_1d(jnp.log10(row[name] / unit))
        return problem.init_to_z(sites)

    z = jax.vmap(one)(period, cols)
    return SamplePosterior(
        samples=z,
        log_weights=posterior.log_weights,
        evidence=posterior.evidence,
        param_names=problem.param_names,
    )


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
