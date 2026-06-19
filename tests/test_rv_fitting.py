"""RV-only and joint RV + astrometry orbit fitting through the generic engine."""

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402

from photomancy.backends import LaplaceBackend, LaplaceMixtureBackend  # noqa: E402
from photomancy.orbit.data import AstromData, RVData  # noqa: E402
from photomancy.orbit.forward import predict_astrometry, predict_rv  # noqa: E402
from photomancy.orbit.inference import build_orbit_logdensity  # noqa: E402
from photomancy.orbit.init import find_init_top_k  # noqa: E402
from photomancy.posterior import MixturePosterior  # noqa: E402

MSUN_KG = 1.989e30
MEARTH_KG = 5.972e24
DIST_PC = 10.0
TRUE_T = 800.0
LOG_P_RANGE = (float(np.log10(500.0)), float(np.log10(1200.0)))

E, COS_I, BIG_OMEGA, LITTLE_OMEGA = 0.1, 0.5, 2.0, 0.5
TP = -1.2 / (2.0 * jnp.pi / TRUE_T)
COS_W, SIN_W = float(jnp.cos(LITTLE_OMEGA)), float(jnp.sin(LITTLE_OMEGA))
MP_KG = 300.0 * MEARTH_KG
MP_SINI = MP_KG * float(jnp.sqrt(1.0 - COS_I**2))


def _make_rv(n_obs=15, seed=3):
    rng = np.random.default_rng(seed)
    times = jnp.asarray(np.sort(rng.uniform(0.0, 2.5 * TRUE_T, n_obs)))
    rv_clean = predict_rv(times, TRUE_T, MSUN_KG, MP_SINI, E, COS_W, SIN_W, TP)
    err = 0.05 * float(jnp.std(rv_clean))
    rv = rv_clean + jnp.asarray(rng.normal(0.0, err, n_obs))
    return RVData(
        times=times,
        rv=rv,
        rv_err=jnp.full(n_obs, err),
        inst_ids=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
        n_inst=1,
    )


def _make_astrom(n_obs=6, seed=4):
    a = float(period_to_sma(TRUE_T, MSUN_KG))
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 2.5 * TRUE_T, n_obs))
    err = 3.0e-3
    ra, dec = predict_astrometry(
        jnp.asarray(times), a, E, COS_I, BIG_OMEGA, COS_W, SIN_W, TP, MSUN_KG, DIST_PC
    )
    return AstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(np.asarray(ra) + rng.normal(0.0, err, n_obs)),
        dec=jnp.asarray(np.asarray(dec) + rng.normal(0.0, err, n_obs)),
        ra_err=jnp.full(n_obs, err),
        dec_err=jnp.full(n_obs, err),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


def test_rv_only_fit_recovers_period_from_seed():
    """An RV-only fit recovers the period when seeded near the true period.

    RV alone has no Thiele-Innes blind initializer (that is astrometry-only), so the
    fit is seeded from a period guess; it then recovers the true period.
    """
    rv = _make_rv()
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, rv_data=rv, log_P_range=LOG_P_RANGE
    )
    z0 = problem.init_to_z({"log_P": jnp.array([float(np.log10(TRUE_T))])})

    post = LaplaceBackend(min_eigenvalue=1.0, n_steps=400).run(problem.logdensity, z0)
    t_fit = float(problem.to_physical(post.mean)["T"])
    assert abs(t_fit - TRUE_T) / TRUE_T < 0.10


def test_joint_rv_astrometry_recovers_period_and_mass():
    """A joint RV + astrometry fit recovers the period (astrometry) and mass (RV)."""
    rv = _make_rv()
    astrom = _make_astrom()
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, rv_data=rv, astrom_data=astrom, log_P_range=LOG_P_RANGE
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

    phys = problem.to_physical(post.means[jnp.argmax(post.log_evidences)])
    assert abs(float(phys["T"]) - TRUE_T) / TRUE_T < 0.10
    assert abs(float(phys["Mp_sini"]) - MP_SINI) / MP_SINI < 0.5
