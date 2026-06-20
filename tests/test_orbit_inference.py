"""Tests for the orbit -> generic-backend bridge (build_orbit_logdensity)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from orbix.equations import period_to_sma  # noqa: E402

from photomancy.backends import LaplaceMixtureBackend  # noqa: E402
from photomancy.orbit.data import AstromData  # noqa: E402
from photomancy.orbit.forward import predict_astrometry  # noqa: E402
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
    np.random.seed(seed)
    times = np.sort(np.random.uniform(0, 3 * TRUE_T, 20))[:n_obs]
    ra_true, dec_true = predict_astrometry(
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
    ra = jnp.array(np.array(ra_true) + np.random.randn(n_obs) * ASTROM_ERR)
    dec = jnp.array(np.array(dec_true) + np.random.randn(n_obs) * ASTROM_ERR)
    return AstromData(
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_dict = find_init(astrom, MSUN_KG, DIST_PC, log_T_range=LOG_P_RANGE)

    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, astrom_data=astrom, log_P_range=LOG_P_RANGE
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
        MSUN_KG, DIST_PC, astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_dicts = find_init_top_k(
            astrom, MSUN_KG, DIST_PC, k=5, log_T_range=LOG_P_RANGE
        )
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
        MSUN_KG, DIST_PC, astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        MSUN_KG, DIST_PC, astrom_data=_make_astrom(6), log_P_range=LOG_P_RANGE
    )
    zpost = to_unconstrained(phys_post, problem, MSUN_KG)
    assert zpost.samples.shape == (1, len(problem.param_names))

    phys = jax.vmap(problem.to_physical)(zpost.samples)
    assert abs(float(phys["T"][0]) - T) / T < 0.02
    assert abs(float(phys["e"][0]) - 0.2) < 0.02
    assert abs(float(phys["cos_i"][0]) - 0.4) < 0.02


def test_orbit_nested_sampling_recovers_period_and_evidence():
    """NumPyro NestedSampler on the orbit model -> SamplePosterior + finite log Z."""
    from photomancy.orbit.nested import orbit_nested_sampling
    from photomancy.posterior import SamplePosterior

    astrom = _make_astrom(6)
    post = orbit_nested_sampling(
        MSUN_KG,
        DIST_PC,
        astrom_data=astrom,
        log_P_range=LOG_P_RANGE,
        max_samples=30000,
        num_samples=1500,
        key=jax.random.key(0),
    )
    assert isinstance(post, SamplePosterior)
    assert post.param_names == ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")
    assert post.samples.shape == (1500, 7)
    assert bool(jnp.isfinite(post.evidence))
    # the period (via a) is recovered: median a near the truth.
    a_true = float(period_to_sma(TRUE_T, MSUN_KG))
    assert abs(float(jnp.median(post.samples[:, 0])) - a_true) / a_true < 0.1
