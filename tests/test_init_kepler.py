"""find_init must be Kepler-consistent.

Regression for the bug where find_init ranked periods by the a-free Thiele-Innes
chi2 (relative astrometry fits the apparent ellipse at many periods), and so
returned a period whose Kepler-III semimajor axis (a = period_to_sma(T, Ms), the
quantity the full model actually uses) does NOT fit the data. The returned
period's Kepler-implied orbit must thread the detections.
"""

import jax
import jax.numpy as jnp
from orbix.equations import period_to_sma

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.init import find_init

MSUN = 1.98892e30
DIST_PC = 10.0
T_TRUE = 365.25


def _seed3001_three_epoch():
    """Reproduce the documented n=3, seed-3001 alias-prone case."""
    a_true = period_to_sma(T_TRUE, MSUN)
    times = jnp.linspace(0.0, 1.5 * T_TRUE, 3)
    ra0, dec0 = predict_relative_astrometry(
        times, a_true, 0.1, 0.7, 0.8, jnp.cos(0.5), jnp.sin(0.5), 40.0, MSUN, DIST_PC
    )
    sigma = 1e-3  # 1 mas, in arcsec
    kd, _ = jax.random.split(jax.random.PRNGKey(3001))
    kra, kdec = jax.random.split(kd)
    ra = ra0 + sigma * jax.random.normal(kra, ra0.shape)
    dec = dec0 + sigma * jax.random.normal(kdec, dec0.shape)
    err = jnp.full(3, sigma)
    data = RelativeAstromData(
        times=times,
        ra=ra,
        dec=dec,
        ra_err=err,
        dec_err=err,
        corr=jnp.zeros(3),
        planet_id=jnp.zeros(3, dtype=int),
        is_valid=jnp.ones(3, dtype=bool),
    )
    return data, float(a_true)


def test_find_init_period_is_kepler_consistent():
    """find_init's period must imply (via Kepler III) the true orbit size."""
    data, a_true = _seed3001_three_epoch()
    iv = find_init(data, MSUN, DIST_PC, log_T_range=(2.0, 3.0))
    T = 10.0 ** float(iv["log_P"][0])
    a_kepler = float(period_to_sma(T, MSUN))
    # the Kepler-implied semimajor axis must match the true orbit, not an a-free alias
    assert abs(a_kepler - a_true) / a_true < 0.15, (
        f"got T={T:.0f}d -> a={a_kepler:.3f}, truth {a_true:.3f}"
    )


def test_find_init_kepler_residual_is_small():
    """The Kepler-implied orbit from find_init should thread the data to ~noise."""
    data, _ = _seed3001_three_epoch()
    iv = find_init(data, MSUN, DIST_PC, log_T_range=(2.0, 3.0))
    T = 10.0 ** float(iv["log_P"][0])
    a = period_to_sma(T, MSUN)
    e = float(iv["e_raw"][0])
    w = float(iv["w_raw"][0])
    ra, dec = predict_relative_astrometry(
        data.times,
        a,
        e,
        float(iv["cos_i"][0]),
        float(iv["W"][0]),
        jnp.cos(w),
        jnp.sin(w),
        -float(iv["M0"][0]) * T / (2.0 * jnp.pi),
        MSUN,
        DIST_PC,
    )
    sq = (ra - data.ra) ** 2 + (dec - data.dec) ** 2
    rms_mas = float(jnp.sqrt(jnp.mean(sq)) * 1000.0)
    assert rms_mas < 5.0, f"find_init Kepler orbit rms = {rms_mas:.1f} mas"
