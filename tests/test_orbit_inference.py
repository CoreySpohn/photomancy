"""Tests for the orbit -> generic-backend bridge (build_orbit_logdensity)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hwoutils.constants import G  # noqa: E402
from orbix.equations import period_a, period_to_sma  # noqa: E402

from photomancy.backends import LaplaceMixtureBackend  # noqa: E402
from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.forward import predict_relative_astrometry  # noqa: E402
from photomancy.orbit.inference import build_orbit_logdensity  # noqa: E402
from photomancy.orbit.init import find_init, find_init_top_k  # noqa: E402
from photomancy.posterior import MixturePosterior  # noqa: E402

MSUN_KG = 1.989e30
DIST_PC = 10.0
TRUE_T = 1096.0
LOG_P_RANGE = (float(jnp.log10(600.0)), float(jnp.log10(1600.0)))
ASTROM_ERR = 5.0e-3


def _make_astrom(n_obs, seed=42):
    """Synthetic relative astrometry for a single planet at TRUE_T."""
    a_true = float(period_to_sma(TRUE_T, MSUN_KG))
    tp_true = -1.5 / (2.0 * jnp.pi / TRUE_T)
    np.random.seed(seed)  # noqa: NPY002 - Preserve seeded fixture stream.
    times = np.sort(
        np.random.uniform(0, 3 * TRUE_T, 20)  # noqa: NPY002 - Preserve seeded times.
    )[:n_obs]
    ra_true, dec_true = predict_relative_astrometry(
        jnp.array(times),
        a_true,
        0.15,
        0.5,
        2.3,
        jnp.cos(0.8),
        jnp.sin(0.8),
        tp_true,
        MSUN_KG,
        DIST_PC,
    )
    ra = jnp.array(
        np.array(ra_true) + np.random.randn(n_obs) * ASTROM_ERR  # noqa: NPY002 - Preserve seeded noise.
    )
    dec = jnp.array(
        np.array(dec_true) + np.random.randn(n_obs) * ASTROM_ERR  # noqa: NPY002 - Preserve seeded noise.
    )
    return RelativeAstromData(
        times=jnp.array(times),
        ra=ra,
        dec=dec,
        ra_err=jnp.full(n_obs, ASTROM_ERR),
        dec_err=jnp.full(n_obs, ASTROM_ERR),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


def test_build_orbit_logdensity_maps_between_z_and_physical():
    """The bridge yields a finite flat logdensity and round-trips the period."""
    astrom = _make_astrom(6)
    init_dict = find_init(astrom, MSUN_KG, DIST_PC, log_T_range=LOG_P_RANGE)

    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, relative_astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    z_init = problem.init_to_z(init_dict)

    assert jnp.isfinite(problem.logdensity(z_init))

    phys = problem.to_physical(z_init)
    assert "T" in phys
    init_T = 10.0 ** float(jnp.squeeze(init_dict["log_P"]))
    assert abs(float(phys["T"]) - init_T) / init_T < 0.05

    assert len(problem.param_names) > 0


def test_orbit_fit_through_generic_backend_recovers_period():
    """Orbit fitting run through the generic LaplaceMixtureBackend recovers T.

    Proves the orbit NumPyro model fits through the domain-agnostic engine: the
    bridge logdensity + TI-seeded multi-start Laplace mixture recover the true
    period within 10% on sparse (n=5) astrometry, matching the orbit-specific path.
    """
    astrom = _make_astrom(5)
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, relative_astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    init_dicts = find_init_top_k(astrom, MSUN_KG, DIST_PC, k=5, log_T_range=LOG_P_RANGE)
    inits = jnp.stack([problem.init_to_z(d) for d in init_dicts])

    post = LaplaceMixtureBackend(min_eigenvalue=1.0, n_steps=300).run(
        problem.logdensity, inits
    )

    assert isinstance(post, MixturePosterior)
    best_z = post.means[jnp.argmax(post.log_evidences)]
    t_fit = float(problem.to_physical(best_z)["T"])
    assert abs(t_fit - TRUE_T) / TRUE_T < 0.10, f"t_fit={t_fit:.1f} vs {TRUE_T}"


def test_orbit_problem_exposes_unflatten_and_trace():
    """The problem exposes the model unflatten + trace for the EIG forward."""
    astrom = _make_astrom(6)
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, relative_astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    init_dict = find_init(astrom, MSUN_KG, DIST_PC, log_T_range=LOG_P_RANGE)
    z = problem.init_to_z(init_dict)

    assert callable(problem.unflatten)
    assert callable(problem.constrain)
    assert set(problem.unflatten(z)) == set(problem.param_names)
    phys = problem.constrain(problem.unflatten(z))
    assert -1.0 < float(phys["cos_i"]) < 1.0  # constrained into Uniform(-1, 1)


def test_to_unconstrained_round_trips_known_orbit():
    """Physical orbit rows -> z -> physical recovers the orbit (OFTI bridge)."""
    from photomancy.orbit.inference import to_unconstrained
    from photomancy.posterior import SamplePosterior

    T = 700.0
    a = float(period_to_sma(T, MSUN_KG))
    cw, sw = float(jnp.cos(0.6)), float(jnp.sin(0.6))
    row = jnp.array([[a, 0.2, 0.4, 1.1, cw, sw, 30.0]])  # a,e,cos_i,W,cos_w,sin_w,tp
    phys_post = SamplePosterior(
        samples=row,
        log_weights=jnp.zeros(1),
        evidence=jnp.asarray(jnp.nan),
        param_names=("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp"),
    )
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, relative_astrom_data=_make_astrom(6), log_P_range=LOG_P_RANGE
    )
    zpost = to_unconstrained(phys_post, problem, MSUN_KG)
    assert zpost.samples.shape == (1, len(problem.param_names))

    phys = jax.vmap(problem.to_physical)(zpost.samples)
    assert abs(float(phys["T"][0]) - T) / T < 0.02
    assert abs(float(phys["e"][0]) - 0.2) < 0.02
    assert abs(float(phys["cos_i"][0]) - 0.4) < 0.02


@pytest.mark.slow
def test_orbit_nested_sampling_recovers_period_and_evidence():
    """Nested sampling (jaxns) on the NumPyro orbit model -> SamplePosterior + log Z."""
    from photomancy.orbit.nested import orbit_nested_sampling
    from photomancy.posterior import SamplePosterior

    astrom = _make_astrom(6)
    post = orbit_nested_sampling(
        MSUN_KG,
        DIST_PC,
        relative_astrom_data=astrom,
        log_P_range=LOG_P_RANGE,
        max_samples=30000,
        num_samples=1500,
        key=jax.random.key(0),
    )
    assert isinstance(post, SamplePosterior)
    assert post.param_names == ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")
    assert post.samples.shape == (1500, 7)
    assert bool(jnp.isfinite(post.evidence))
    # With normalized priors, Z cannot exceed the zero-residual Gaussian peak.
    # There is no analytic orbit log Z target to compare within sampling error.
    loglike_upper = -2 * astrom.times.size * jnp.log(jnp.sqrt(2 * jnp.pi) * ASTROM_ERR)
    assert float(post.evidence) <= float(loglike_upper)

    periods = period_a(post.samples[:, 0], G * MSUN_KG)
    period_median = float(jnp.median(periods))
    period_std = float(jnp.std(periods))
    period_error = abs(period_median - TRUE_T)
    # Recover within 3 posterior standard deviations (seed 0: z = 1.82).
    # Six epochs cover less than half an orbit; also keep a loose absolute guard.
    assert 0 < period_std < float("inf")
    assert period_error < 3 * period_std, (
        f"period={period_median:.3f}, truth={TRUE_T}, std={period_std:.3f}, "
        f"z={period_error / period_std:.3f}"
    )
    assert period_error / TRUE_T < 0.25


@pytest.mark.parametrize("use_x64", [False, True])
def test_orbit_model_respects_precision(use_x64):
    """Standalone model tracing follows JAX precision without dtype warnings."""
    from numpyro.distributions.transforms import biject_to
    from numpyro.handlers import seed, trace

    from photomancy.orbit._numpyro_bridge import _init_dict_to_z_flat
    from photomancy.orbit.data import OrbitData
    from photomancy.orbit.model import build_model

    with jax.enable_x64(use_x64):
        astrom = _make_astrom(3)
        model = build_model(has_relative_astrom=True, log_P_range=LOG_P_RANGE)
        sites = trace(seed(model, rng_seed=0)).get_trace(
            MSUN_KG, DIST_PC, OrbitData(relative_astrom=astrom)
        )
        ll = sites["ll_total"]["value"]
        assert ll.dtype == (jnp.float64 if use_x64 else jnp.float32)
        assert jnp.isfinite(ll)

        # The same precision must survive physical -> unconstrained conversion.
        latent = {
            name: site
            for name, site in sites.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        z = _init_dict_to_z_flat(
            {name: site["value"] for name, site in latent.items()},
            {name: jnp.zeros_like(site["value"]) for name, site in latent.items()},
            {name: biject_to(site["fn"].support).inv for name, site in latent.items()},
        )
        assert z.dtype == ll.dtype
        assert jnp.all(jnp.isfinite(z))
