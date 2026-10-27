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

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import Reparam
from numpyro.infer.util import log_density

from photomancy.backends.nested import _to_jaxns_prior
from photomancy.orbit.data import OrbitData
from photomancy.orbit.model import build_model
from photomancy.posterior import SamplePosterior
from photomancy.priors import Uniform

# Physical columns, matching the OFTI / grid_search SamplePosterior convention so an
# orbit nested-sampling posterior is interchangeable with them downstream.
_ORBIT_PHYS_NAMES = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")


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
    log_P_range=(1.0, 4.0),
    log_Mp_range=(-2.0, 4.0),
    log_Rp_range=(-5.0, -2.5),
    log_Ag_range=(-2.0, 0.0),
    ecc_prior="kipping13",
    jitter_scale=0.00017314568368055556,  # 1e-10 AU/day in m/s
    num_live_points=None,
    max_samples=50000,
    num_samples=2000,
    key=None,
):
    """Nested-sample the orbit model -> ``SamplePosterior`` + evidence (``log Z``).

    Builds the NumPyro orbit model for the supplied data channels, reparametrizes its
    latent sites onto the unit cube, and runs it through the jaxns ``NestedSampler``
    with the default jaxns stopping condition (remaining evidence below
    ``log(1 + 1e-3)``). The returned posterior carries equally-weighted physical
    samples (``a, e, cos_i, W, cos_w, sin_w, tp``), with an additional ``Mp`` column
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
        key: PRNG key (required).

    Returns:
        A :class:`~photomancy.posterior.SamplePosterior` with ``samples`` of shape
        ``(num_samples, 7)`` over ``_ORBIT_PHYS_NAMES`` (8 columns including ``Mp``
        when true mass is sampled), uniform ``log_weights``, and
        ``evidence`` set to the nested-sampling ``log Z`` (the jaxns classic
        expectation estimate, ``log_Z_mean``).

    Raises:
        ValueError: If ``key`` is ``None``.
    """
    if key is None:
        raise ValueError("orbit_nested_sampling requires a PRNG key.")
    from jaxns.core import NestedSampler
    from jaxns.model import Model

    model = build_model(
        n_planets=1,
        has_rv=rv_data is not None,
        has_relative_astrom=relative_astrom_data is not None,
        has_stellar_astrom=stellar_astrom_data is not None,
        has_pm_anomaly=pm_anomaly_data is not None,
        has_null=null_data is not None,
        has_imaging=imaging_data is not None,
        log_P_range=log_P_range,
        log_Mp_range=log_Mp_range,
        log_Rp_range=log_Rp_range,
        log_Ag_range=log_Ag_range,
        ecc_prior=ecc_prior,
        jitter_scale=jitter_scale,
    )
    data = OrbitData(
        rv=rv_data,
        relative_astrom=relative_astrom_data,
        stellar_astrom=stellar_astrom_data,
        pm_anomaly=pm_anomaly_data,
        null=null_data,
        imaging=imaging_data,
    )
    model_args = (Ms, dist_pc, data)

    trace_key, run_key, sample_key, predict_key = jax.random.split(key, 4)
    prototype = trace(seed(model, trace_key)).get_trace(*model_args)
    phys_names = (*_ORBIT_PHYS_NAMES, "Mp") if "Mp" in prototype else _ORBIT_PHYS_NAMES
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
    )
    results = ns.run(key=run_key, args=model_args).to_result().trim()
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

    return SamplePosterior(
        samples=samples,
        log_weights=jnp.zeros(num_samples),
        evidence=jnp.asarray(results.log_Z_mean),
        param_names=phys_names,
    )
