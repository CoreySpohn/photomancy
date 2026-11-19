"""RV-only and joint RV + astrometry orbit fitting through the generic engine."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hwoutils.constants import AU2m, Mearth2kg, Msun2kg, d2s  # noqa: E402
from kinematics import (  # noqa: E402
    predict_relative_astrometry,
    predict_rv,
)
from orbit_fit import classical_coordinates, mass_problem  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402

from photomancy import RadialVelocityTerm, RelativeAstrometryTerm  # noqa: E402
from photomancy.infer.legacy_posterior import MixturePosterior  # noqa: E402
from photomancy.measurements import RelativeAstromData, RVData  # noqa: E402
from photomancy.orbit.init import find_init  # noqa: E402
from photomancy.orbit.laplace import (  # noqa: E402
    map_laplace_fit,
    map_laplace_mixture_fit,
)

MSUN_KG = Msun2kg
MEARTH_KG = Mearth2kg
DIST_PC = 10.0
TRUE_T = 800.0
PERIOD_RANGE_D = (500.0, 1200.0)
LOG_P_RANGE = tuple(float(np.log10(x)) for x in PERIOD_RANGE_D)

E, COS_I, BIG_OMEGA, LITTLE_OMEGA = 0.1, 0.5, 2.0, 0.5
TP = -1.2 / (2.0 * jnp.pi / TRUE_T)
COS_W, SIN_W = float(jnp.cos(LITTLE_OMEGA)), float(jnp.sin(LITTLE_OMEGA))
MP_KG = 300.0 * MEARTH_KG
MP_SINI = MP_KG * float(jnp.sqrt(1.0 - COS_I**2))


def _make_rv(n_obs=15, seed=3):
    rng = np.random.default_rng(seed)
    times = jnp.asarray(np.sort(rng.uniform(0.0, 2.5 * TRUE_T, n_obs)))
    rv_clean = predict_rv(times, TRUE_T, MSUN_KG, MP_SINI, E, COS_W, SIN_W, TP)
    rv_clean = rv_clean * (AU2m / d2s)
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
    ra, dec = predict_relative_astrometry(
        jnp.asarray(times), a, E, COS_I, BIG_OMEGA, COS_W, SIN_W, TP, MSUN_KG, DIST_PC
    )
    return RelativeAstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(np.asarray(ra) + rng.normal(0.0, err, n_obs)),
        dec=jnp.asarray(np.asarray(dec) + rng.normal(0.0, err, n_obs)),
        ra_err=jnp.full(n_obs, err),
        dec_err=jnp.full(n_obs, err),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


# Medium: The RV model, optimizer and Hessian require a cold compile.
@pytest.mark.medium
def test_rv_only_fit_recovers_period_from_seed():
    """An RV-only fit recovers the period when seeded near the true period.

    RV alone has no Thiele-Innes blind initializer (that is astrometry-only), so the
    fit is seeded at the true semi-major axis (period) with the other elements off;
    it then recovers the true period.
    """
    rv = _make_rv()
    seed = classical_coordinates(
        float(period_to_sma(TRUE_T, MSUN_KG)), 0.3, 0.2, 1.0, 1.0, 0.0, TRUE_T
    )
    problem = mass_problem(
        (RadialVelocityTerm(rv),),
        seed,
        100.0,
        star_mass_kg=MSUN_KG,
        dist_pc=DIST_PC,
        period_range_d=PERIOD_RANGE_D,
    )
    post = map_laplace_fit(
        problem, problem.flatten(problem.position_template), n_steps=400
    )
    t_fit = float(problem.to_physical(post.mean)["T"])
    assert abs(t_fit - TRUE_T) / TRUE_T < 0.10


# Medium: The joint RV/astrometry multi-start fit exceeds the unit compile budget.
@pytest.mark.medium
def test_joint_rv_astrometry_recovers_period_and_mass():
    """A joint RV + astrometry fit recovers the period (astrometry) and mass (RV)."""
    rv = _make_rv()
    astrom = _make_astrom()
    starts, _ = find_init(astrom, MSUN_KG, DIST_PC, k=2, log_T_range=LOG_P_RANGE)
    problem = mass_problem(
        (RadialVelocityTerm(rv), RelativeAstrometryTerm(astrom)),
        starts[0],
        100.0,
        star_mass_kg=MSUN_KG,
        dist_pc=DIST_PC,
        period_range_d=PERIOD_RANGE_D,
    )
    inits = jnp.stack(
        [
            problem.flatten({"orbit": start, "mass_Mearth": jnp.log(100.0)})
            for start in starts
        ]
    )
    post = map_laplace_mixture_fit(problem, inits, n_steps=100)
    assert isinstance(post, MixturePosterior)

    phys = problem.to_physical(post.means[jnp.argmax(post.log_evidences)])
    assert abs(float(phys["T"]) - TRUE_T) / TRUE_T < 0.10
    assert abs(float(phys["Mp_sini"]) - MP_SINI) / MP_SINI < 0.5
