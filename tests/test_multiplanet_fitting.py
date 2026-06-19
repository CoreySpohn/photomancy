"""Multi-planet (n_planets > 1) orbit fitting through the high-level API.

The NumPyro model has an ``n_planets > 1`` branch (vmap over planets, ``planet_id``
indexing of astrometry), but the high-level fit builders previously hardcoded
``n_planets = 1``. These tests reach the multi-planet path through
``build_orbit_logdensity`` and check it builds, evaluates, and recovers two periods
when seeded near truth (blind multi-planet initialization is separate, future work).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402

from photomancy.backends import LaplaceBackend  # noqa: E402
from photomancy.orbit.data import AstromData  # noqa: E402
from photomancy.orbit.forward import predict_astrometry  # noqa: E402
from photomancy.orbit.inference import build_orbit_logdensity  # noqa: E402

MSUN_KG = 1.989e30
DIST_PC = 10.0
T0, T1 = 600.0, 1200.0
LOG_P_RANGE = (float(np.log10(450.0)), float(np.log10(1500.0)))


def _make_two_planet_astrom(seed=11):
    rng = np.random.default_rng(seed)
    err = 3.0e-3
    times, ra, dec, pid = [], [], [], []
    for planet, (t_days, cos_i, big_omega) in enumerate(
        [(T0, 0.6, 1.5), (T1, 0.3, 3.0)]
    ):
        a = float(period_to_sma(t_days, MSUN_KG))
        tp = -1.0 / (2.0 * jnp.pi / t_days)
        t = np.sort(rng.uniform(0.0, 2.0 * t_days, 6))
        r, d = predict_astrometry(
            jnp.asarray(t),
            a,
            0.1,
            cos_i,
            big_omega,
            jnp.cos(0.5),
            jnp.sin(0.5),
            tp,
            MSUN_KG,
            DIST_PC,
        )
        times.append(t)
        ra.append(np.asarray(r) + rng.normal(0.0, err, 6))
        dec.append(np.asarray(d) + rng.normal(0.0, err, 6))
        pid.append(np.full(6, planet))
    n = 12
    return AstromData(
        times=jnp.asarray(np.concatenate(times)),
        ra=jnp.asarray(np.concatenate(ra)),
        dec=jnp.asarray(np.concatenate(dec)),
        ra_err=jnp.full(n, err),
        dec_err=jnp.full(n, err),
        corr=jnp.zeros(n),
        planet_id=jnp.asarray(np.concatenate(pid), dtype=int),
        is_valid=jnp.ones(n, dtype=bool),
    )


def test_build_orbit_logdensity_two_planets_builds_and_evaluates():
    """The API exposes n_planets and a 2-planet astrometry model evaluates finitely."""
    astrom = _make_two_planet_astrom()
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, astrom_data=astrom, n_planets=2, log_P_range=LOG_P_RANGE
    )
    z0 = problem.init_to_z({})
    assert jnp.isfinite(problem.logdensity(z0))
    assert z0.shape[0] > 6  # per-planet (plate-shaped) position, larger than 1-planet


def test_two_planet_fit_recovers_both_periods_from_seed():
    """A 2-planet fit seeded near truth recovers both periods.

    Blind multi-planet initialization is future work, so both planets are seeded
    near their true elements; the fit then refines and holds both periods.
    """
    astrom = _make_two_planet_astrom()
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, astrom_data=astrom, n_planets=2, log_P_range=LOG_P_RANGE
    )
    z0 = problem.init_to_z(
        {
            "log_P": jnp.array([float(np.log10(T0)), float(np.log10(T1))]),
            "e_raw": jnp.array([0.1, 0.1]),
            "w_raw": jnp.array([0.5, 0.5]),
            "cos_i": jnp.array([0.6, 0.3]),
            "W": jnp.array([1.5, 3.0]),
            "M0": jnp.array([1.0, 1.0]),
        }
    )
    post = LaplaceBackend(min_eigenvalue=1.0, n_steps=400).run(problem.logdensity, z0)
    periods = np.sort(np.asarray(problem.to_physical(post.mean)["T"]))
    assert abs(periods[0] - T0) / T0 < 0.10
    assert abs(periods[1] - T1) / T1 < 0.10
