"""Nested sampling for the orbit model via jaxns.

The orbit fit is already a NumPyro model, so it gets nested-sampling evidence without a
model rewrite: every latent site is reparametrized onto the unit cube (its inverse CDF
applied to a unit-cube variable), the unit-cube variables become jaxns priors, and the
NumPyro log-density of the reparametrized model is the jaxns log-likelihood. The
headline output is the Bayesian evidence (``log Z``), which turns an orbit fit into
model comparison (planet count, period-alias resolution). The result is returned as a
photomancy :class:`~photomancy.posterior.SamplePosterior` in the same physical columns
OFTI / grid_search emit, plus true mass when sampled, so it drops into the EIG /
clustering machinery (``cluster_to_mixture``, ``to_unconstrained``).

jaxns is imported lazily inside the function: importing it enables JAX x64 globally as
a side effect, so ``import photomancy`` stays light.
"""

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import Reparam
from numpyro.infer.util import log_density

from photomancy.backends.nested import _to_jaxns_prior
from photomancy.orbit._numpyro_bridge import _build_orbit_model
from photomancy.posterior import SamplePosterior
from photomancy.priors import Uniform

# Physical columns, matching the OFTI / grid_search SamplePosterior convention so an
# orbit nested-sampling posterior is interchangeable with them downstream.
_ORBIT_PHYS_NAMES = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")


class OrbitNestedPosterior(SamplePosterior):
    """Physical nested samples with evidence and sampling diagnostics.

    ``diagnostics`` contains ``log_evidence``, ``log_evidence_error`` (standard
    deviation of log Z), ``converged`` (remaining evidence fraction below
    ``evidence_tolerance``), ``weight_ess`` (Kish ESS before resampling),
    ``num_iterations`` (jaxns goal iterations), ``num_distinct`` (distinct
    physical rows after resampling), and the jaxns ``termination_reason``.
    ``converged`` evaluates the evidence criterion independently of the stopping
    reason; it can be true even when storage is exhausted at the same boundary.

    References:
        Skilling (2006), Bayesian Analysis 1(4), 833-859 (nested evidence).
        Kish (1965), Survey Sampling (effective sample size of weighted draws).
    """

    diagnostics: dict = eqx.field(kw_only=True)


class _UnitCubeReparam(Reparam):
    """Reparametrize a latent site as its inverse CDF of a unit-cube variable.

    The site ``name`` becomes a deterministic function of a new ``{name}_base`` site
    whose (masked) Uniform(0, 1) density contributes nothing to the log-density, so
    the reparametrized model's log-density is the likelihood alone and the prior lives
    entirely in the unit-cube push-forward that nested sampling integrates over. Every
    latent distribution in the orbit model (Uniform, Beta, Weibull, HalfNormal) has
    an ``icdf``.
    """

    def __call__(self, name, fn, obs):
        """Replace site ``name`` by ``fn.icdf`` of a unit-cube ``{name}_base`` site."""
        if obs is not None:
            raise ValueError(f"Cannot reparametrize the observed site {name!r}.")
        shape = fn.shape()
        base_fn, _expand_shape, event_dim = self._unwrap(fn)
        u = numpyro.sample(
            f"{name}_base",
            dist.Uniform(0.0, 1.0).expand(shape).to_event(event_dim).mask(False),
        )
        return None, base_fn.icdf(u)


def orbit_nested_sampling(
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
    num_live_points=None,
    max_samples=50000,
    num_samples=2000,
    evidence_tolerance=1e-3,
    key=None,
):
    """Nested-sample the orbit model -> ``SamplePosterior`` + evidence (``log Z``).

    Builds the NumPyro orbit model for the supplied data channels, reparametrizes its
    latent sites onto the unit cube, and runs it through the jaxns ``NestedSampler``
    with a remaining-evidence fraction stopping condition. The returned posterior
    carries equally-weighted physical samples (``a, e, cos_i, W, cos_w, sin_w, tp``),
    with an additional ``Mp`` column
    in kg when true mass is sampled, and the evidence; the evidence is
    the point of nested sampling -- a Bayes factor between two fits is orbit model
    comparison (e.g. planet present vs absent, alias A vs B).

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
        custom_loglike: JAX-compatible ``params -> scalar`` data log-likelihood
            with no prior terms (default None). Vector returns raise ValueError.
            Uses the same :func:`~photomancy.orbit.model.build_model` route as
            :func:`~photomancy.orbit.inference.build_orbit_logdensity`. Receives
            physical orbit and planet-scale parameters, ``Ms``, ``dist_pc`` and
            ``custom_data``. No term is added for an absent built-in channel.
        custom_data: Pytree bound into ``params["custom_data"]`` in the callback
            (default None). Array leaves must support JAX transformations.
        sample_planet_scale: Sample ``log_Rp``, ``log_Ag`` and ``log_Mp`` even
            without built-in data, allowing prior-only fits. Custom likelihoods
            also activate these sites. Default False.
        allow_overlap: Explicitly permit built-in channels with a custom
            likelihood. Default False; combining them otherwise raises ValueError.
            With True, all supplied likelihood terms are added.
        log_P_range: ``(min, max)`` for the ``log10(period/days)`` prior.
        log_Mp_range: ``(min, max)`` for the ``log10(mass/M_earth)`` prior.
        log_Rp_range: ``(min, max)`` for the ``log10(Rp/AU)`` prior.
        log_Ag_range: ``(min, max)`` for the ``log10(geometric albedo)`` prior.
        ecc_prior: Eccentricity prior name (``"kipping13"``, ``"rayleigh"``,
            ``"vaneylen19"``, ``"disk"``).
        jitter_scale: Scale for the HalfNormal RV-jitter prior (m/s).
        num_live_points: Number of independent root chains (jaxns
            ``root_allocation_degree``); ``None`` uses the jaxns default, ``30 * D``
            for ``D`` latent parameters.
        max_samples: Hard cap on stored nested-sampling samples; jaxns stops early if
            the buffer fills before its evidence-convergence condition is met.
        num_samples: Number of equally-weighted posterior samples to return.
        evidence_tolerance: Remaining-evidence fraction threshold, in (0, 1).
            Default 1e-3. This is not the uncertainty on log evidence.
        key: PRNG key (required).

    Returns:
        An :class:`OrbitNestedPosterior` (a ``SamplePosterior``) over
        ``_ORBIT_PHYS_NAMES``, plus ``Mp`` when sampled and ``Rp``, ``Ag`` and
        ``Lambda`` whenever their sites are active. Includes ``diagnostics``,
        uniform ``log_weights``, and
        ``evidence`` set to the nested-sampling ``log Z`` (the jaxns classic
        expectation estimate, ``log_Z_mean``).

    Raises:
        ValueError: If ``key`` is ``None``, ``evidence_tolerance`` is outside
            (0, 1), or custom and built-in terms overlap without explicit opt-in.
    """
    if key is None:
        raise ValueError("orbit_nested_sampling requires a PRNG key.")
    if not 0.0 < evidence_tolerance < 1.0:
        raise ValueError("evidence_tolerance must lie in (0, 1).")
    model, model_args, _ = _build_orbit_model(
        Ms,
        dist_pc,
        n_planets=1,
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
    )
    from jaxns.algorithm.depth import _depth_condition_reached
    from jaxns.core import NestedSampler
    from jaxns.depth_condition import DepthCondition
    from jaxns.model import Model

    trace_key, run_key, sample_key, predict_key = jax.random.split(key, 4)
    prototype = trace(seed(model, trace_key)).get_trace(*model_args)
    phys_names = (*_ORBIT_PHYS_NAMES, "Mp") if "Mp" in prototype else _ORBIT_PHYS_NAMES
    if "Rp" in prototype:
        phys_names = (*phys_names, "Rp", "Ag", "Lambda")
    latent_shapes = {
        name: site["fn"].shape()
        for name, site in prototype.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    unit_model = reparam(
        model, config={name: _UnitCubeReparam() for name in latent_shapes}
    )
    unit_priors = {
        name: _to_jaxns_prior(
            Uniform(low=jnp.zeros(math.prod(shape)), high=jnp.ones(math.prod(shape))),
            name=f"{name}_base",
        )
        for name, shape in latent_shapes.items()
    }

    def prior_model(Ms, dist_pc, data):
        unit = {
            f"{name}_base": unit_priors[name].realise().reshape(shape)
            for name, shape in latent_shapes.items()
        }
        return log_density(unit_model, (Ms, dist_pc, data), {}, unit)[0]

    ns = NestedSampler(
        model=Model(prior_model=prior_model),
        max_samples=int(max_samples),
        root_allocation_degree=num_live_points,
        depth_condition=DepthCondition(dlogZ=evidence_tolerance),
    )
    state = ns.run(key=run_key, args=model_args)
    results = state.to_result().trim()
    converged = _depth_condition_reached(state, ns.depth_condition)
    unit_draws = results.resample(num_samples, key=sample_key).X_samples
    unit_draws = {
        f"{name}_base": jnp.reshape(unit_draws[f"{name}_base"], (num_samples, *shape))
        for name, shape in latent_shapes.items()
    }
    # The model exposes the physical orbit parameters as deterministic sites, so the
    # equally-weighted unit-cube draws map to physical columns through the model
    # (squeeze the planet axis).
    draws = Predictive(
        unit_model, posterior_samples=unit_draws, return_sites=list(phys_names)
    )(predict_key, *model_args)
    samples = jnp.stack(
        [jnp.reshape(draws[name], (num_samples,)) for name in phys_names],
        axis=1,
    )

    return OrbitNestedPosterior(
        samples=samples,
        log_weights=jnp.zeros(num_samples),
        evidence=jnp.asarray(results.log_Z_mean),
        param_names=phys_names,
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
