"""Chart-coordinate nested sampling and tempered SMC.

Both integrate likelihood terms over independent physical priors and return
draws in the flat coordinates of the supplied orbit problem.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, Mearth2kg, two_pi
from jax.scipy.special import ndtr, ndtri
from orbix.equations import period_to_sma
from orbix.system import FrameChart, FrameOrbit

from photomancy.infer.backends.nested import jaxns_prior
from photomancy.infer.backends.smc import SMCBackend
from photomancy.infer.posterior import SamplePosterior
from photomancy.orbit.inference import ChartOrbitProblem
from photomancy.priors import (
    FactorizedOrbitPrior,
    HalfNormalPrior,
    LogUniformPrior,
    Uniform,
)


class OrbitNestedPosterior(SamplePosterior):
    """Nested samples with evidence and sampling diagnostics.

    ``diagnostics`` contains ``log_evidence``, ``log_evidence_error`` (standard
    deviation of log Z), ``converged`` (remaining evidence fraction below
    ``evidence_tolerance``), ``weight_ess`` (Kish ESS before resampling),
    ``num_iterations`` (jaxns goal iterations), ``num_distinct`` (distinct
    rows after resampling), and the jaxns ``termination_reason``.
    ``converged`` evaluates the evidence criterion independently of the stopping
    reason; it can be true even when storage is exhausted at the same boundary.

    References:
        Skilling (2006), Bayesian Analysis 1(4), 833-859 (nested evidence).
        Kish (1965), Survey Sampling (effective sample size of weighted draws).
    """

    diagnostics: dict = eqx.field(kw_only=True)


class OrbitPriorTransform(eqx.Module):
    """Normalized prior draw in a one-planet chart problem's flat coordinates.

    Cube columns are log period, eccentricity, argument of periapsis, cos i,
    ascending node and mean anomaly. Each scalar follows in sorted name order
    and row-major order within an array (e.g. one jitter per instrument).
    A Frame chart has one final coordinate for quaternion sign and radius.
    """

    problem: ChartOrbitProblem

    def __check_init__(self):
        """Reject priors without a matching independent cube transform."""
        density = self.problem.density
        if not isinstance(density.prior, FactorizedOrbitPrior):
            raise ValueError("OrbitPriorTransform requires a FactorizedOrbitPrior.")
        for name, coordinate in density.scalars.items():
            if not isinstance(coordinate.prior, (LogUniformPrior, HalfNormalPrior)):
                raise ValueError(
                    f"Scalar {name!r} requires a prior with supported quantiles."
                )

    @property
    def ndim(self):
        """Number of independent continuous unit-cube coordinates."""
        return self.problem.chart.n_coords + sum(
            math.prod(jnp.shape(self.problem.position_template[name]))
            for name in self.problem.density.scalars
        )

    def forward(self, unit):
        """Push one unit-cube point through physical elements to the fit chart."""
        problem = self.problem
        law = problem.density.prior
        # Angular poles, e=0 and infinite-tail quantiles are singular chart
        # coordinates. Extend the map to the measure-zero cube boundary.
        unit = jnp.asarray(unit)
        eps = jnp.finfo(unit.dtype).eps
        unit = jnp.clip(unit, eps, 1.0 - eps)
        params = {}
        total_mass_kg = problem.star_mass_kg
        start = 6
        for name, coord in sorted(problem.density.scalars.items()):
            shape = jnp.shape(problem.position_template[name])
            end = start + math.prod(shape)
            value = coord.prior.icdf(unit[start:end].reshape(shape))
            params[name] = jnp.log(value) if coord.logarithmic else value
            if name == "mass_Mearth":
                total_mass_kg += value * Mearth2kg
            start = end

        period_d = law.period_icdf(unit[0])
        orbit = FrameOrbit.from_classical(
            period_to_sma(period_d, total_mass_kg),
            law.eccentricity_icdf(unit[1]),
            jnp.arccos(2 * unit[3] - 1),
            two_pi * unit[4],
            two_pi * unit[2],
            two_pi * unit[5],
            mu_AU3_per_d2=G * total_mass_kg,
        )
        x = problem.chart.from_orbit(orbit)
        if isinstance(problem.chart, FrameChart):
            # Halves of the cube choose the two equivalent quaternion signs.
            # Within each half draw the independent radial gauge.
            u = unit[start]
            q = (2 * u) % 1.0
            q = jnp.where(q == 0.0, 0.5, q)
            sd = problem.chart.gauge_sd
            if sd is None:
                from tensorflow_probability.substrates.jax.math import igammainv

                radius = jnp.sqrt(2 * igammainv(2.0, q))  # chi distribution, 4 d.o.f.
            else:
                # A normal centered at one conditioned on a positive radius.
                radius = 1 - sd * ndtri((1 - q) * ndtr(1 / sd))
            x = x.at[3:].multiply(jnp.where(u < 0.5, -radius, radius))
        params["orbit"] = x
        return problem.flatten(params)

    def sample(self, key, n):
        """Draw n independent flat chart positions for prior initialization."""
        unit = jax.random.uniform(key, (n, self.ndim))
        return jax.vmap(self.forward)(unit)


def build_chart_nested_model(problem):
    """Build a jaxns model with the fit's chart columns and a terms-only likelihood."""
    from jaxctx.context import set_state
    from jaxns.model import Model

    transform = OrbitPriorTransform(problem)
    unit_prior = jaxns_prior(
        Uniform(jnp.zeros(transform.ndim), jnp.ones(transform.ndim)), name="unit"
    )

    def prior_model():
        unit = unit_prior.realise(X_collection="cube")
        z = transform.forward(unit)
        set_state(name="z", collection="X", value=z)
        return problem.loglikelihood(z)

    return Model(prior_model=prior_model)


def orbit_tempered_smc(
    problem: ChartOrbitProblem, backend: SMCBackend, *, n_particles, key
):
    """Temper normalized chart-prior particles onto the data terms.

    Args:
        problem: Chart problem whose density carries a cube transform.
        backend: The tempered SMC configuration to run.
        n_particles: Number of particles drawn from the prior, at least two.
        key: PRNG key.

    Returns:
        A ``SamplePosterior`` in the problem's chart with the log evidence.
    """
    if n_particles < 2:
        raise ValueError("Chart SMC requires at least two particles")
    draw_key, run_key = jax.random.split(key)
    init = OrbitPriorTransform(problem).sample(draw_key, n_particles)
    posterior = backend.run(problem.logprior, problem.loglikelihood, init, run_key)
    return problem.declare(
        SamplePosterior(
            posterior.samples,
            posterior.log_weights,
            posterior.evidence,
            param_names=problem.param_names,
        )
    )


def orbit_nested_sampling(
    problem: ChartOrbitProblem,
    *,
    num_live_points=None,
    max_samples=50000,
    num_samples=2000,
    evidence_tolerance=1e-3,
    key=None,
):
    """Integrate a chart problem and return equal-weight chart draws and log Z.

    Supply a FactorizedOrbitPrior and sampleable independent scalar priors to
    the chart density. The cube transform carries every prior and Jacobian.
    Only the measurement terms enter the nested likelihood.
    """
    if key is None:
        raise ValueError("orbit_nested_sampling requires a PRNG key.")
    if not 0.0 < evidence_tolerance < 1.0:
        raise ValueError("evidence_tolerance must lie in (0, 1).")
    run_key, sample_key = jax.random.split(key)
    results, state, converged = _run_nested(
        build_chart_nested_model(problem),
        (),
        run_key,
        num_live_points,
        max_samples,
        evidence_tolerance,
    )
    samples = results.resample(num_samples, key=sample_key).X_samples["z"]
    return problem.declare(
        _nested_posterior(samples, problem.param_names, results, state, converged)
    )


def _run_nested(
    model, model_args, run_key, num_live_points, max_samples, evidence_tolerance
):
    from jaxns.algorithm.depth import _depth_condition_reached
    from jaxns.core import NestedSampler
    from jaxns.depth_condition import DepthCondition

    ns = NestedSampler(
        model=model,
        max_samples=int(max_samples),
        root_allocation_degree=num_live_points,
        depth_condition=DepthCondition(dlogZ=evidence_tolerance),
    )
    state = ns.run(key=run_key, args=model_args)
    results = state.to_result().trim()
    converged = _depth_condition_reached(state, ns.depth_condition)
    return results, state, converged


def _nested_posterior(samples, param_names, results, state, converged):
    return OrbitNestedPosterior(
        samples=samples,
        log_weights=jnp.zeros(samples.shape[0]),
        evidence=jnp.asarray(results.log_Z_mean),
        param_names=param_names,
        diagnostics=dict(
            log_evidence=jnp.asarray(results.log_Z_mean),
            log_evidence_error=jnp.asarray(results.log_Z_uncert),
            converged=jnp.asarray(converged),
            weight_ess=jnp.asarray(results.ess),
            num_iterations=jnp.asarray(state.goal_loop_iter),
            num_distinct=jnp.asarray(jnp.unique(samples, axis=0).shape[0]),
            termination_reason=jnp.asarray(results.termination_reason),
        ),
    )
