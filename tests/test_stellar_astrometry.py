"""Tests for stellar-reflex astrometry: the forward, the likelihood, and the fit.

Stellar astrometry measures the star's reflex wobble (scaled by the mass ratio),
so unlike relative astrometry it constrains the planet mass. These tests cover the
forward (a scaled, sign-flipped relative orbit), the shared bivariate-Gaussian
likelihood, mass identifiability through the generic engine, and the wiring of the
stellar channel into the NumPyro orbit model.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hwoutils.constants import Mearth2kg  # noqa: E402

from photomancy import LaplaceBackend, build_scene_logdensity  # noqa: E402
from photomancy.orbit import (  # noqa: E402
    StellarAstromData,
    build_orbit_logdensity,
    loglike_relative_astrom,
    loglike_stellar_astrom,
    predict_relative_astrometry,
    predict_stellar_astrometry,
)
from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.priors import Uniform  # noqa: E402

MSUN_KG = 1.989e30
DIST_PC = 10.0


def _truth_args(times):
    return dict(
        times=times,
        a=jnp.array(5.2),
        e=jnp.array(0.2),
        cos_i=jnp.array(0.5),
        W=jnp.array(0.8),
        cos_w=jnp.array(np.cos(1.0)),
        sin_w=jnp.array(np.sin(1.0)),
        tp=jnp.array(1500.0),
    )


def test_stellar_forward_is_scaled_relative():
    """Stellar reflex = -Mp/(Ms+Mp) times the relative astrometry (orvara Eq 22/23)."""
    times = jnp.linspace(0.0, 4000.0, 12)
    args = _truth_args(times)
    Mp = 317.8 * Mearth2kg  # ~1 Jupiter mass

    ra_rel, dec_rel = predict_relative_astrometry(Ms=MSUN_KG, dist_pc=DIST_PC, **args)
    ra_star, dec_star = predict_stellar_astrometry(
        Ms=MSUN_KG, Mp=Mp, dist_pc=DIST_PC, **args
    )

    frac = Mp / (MSUN_KG + Mp)
    assert jnp.allclose(ra_star, -frac * ra_rel)
    assert jnp.allclose(dec_star, -frac * dec_rel)
    # the reflex is much smaller than the relative orbit (mass ratio ~1e-3)
    assert jnp.max(jnp.abs(ra_star)) < jnp.max(jnp.abs(ra_rel))


def test_stellar_and_relative_likelihoods_share_form():
    """The two astrometry likelihoods are the same bivariate Gaussian on (ra, dec)."""
    n = 5
    ra_pred = jnp.linspace(-0.1, 0.1, n)
    dec_pred = jnp.linspace(0.05, -0.05, n)
    common = dict(
        times=jnp.arange(n, dtype=float),
        ra=ra_pred + 1e-3,
        dec=dec_pred - 1e-3,
        ra_err=jnp.full(n, 2e-3),
        dec_err=jnp.full(n, 2e-3),
        corr=jnp.zeros(n),
        is_valid=jnp.ones(n, dtype=bool),
    )
    stellar = StellarAstromData(**common)
    relative = RelativeAstromData(planet_id=jnp.zeros(n, dtype=int), **common)

    ll_stellar = loglike_stellar_astrom(ra_pred, dec_pred, stellar)
    ll_relative = loglike_relative_astrom(ra_pred, dec_pred, relative)
    assert jnp.allclose(ll_stellar, ll_relative)


class _Orbit(eqx.Module):
    a: jnp.ndarray
    e: jnp.ndarray
    cos_i: jnp.ndarray
    W: jnp.ndarray
    omega: jnp.ndarray
    tp: jnp.ndarray
    log_Mp: jnp.ndarray


def test_mass_is_identifiable_from_reflex():
    """A reflex-astrometry fit recovers the planet mass (period -> a -> amplitude)."""
    P_days = 4331.0  # a = 5.2 AU around 1 Msun
    times = jnp.asarray(np.linspace(0.0, 1.4 * P_days, 30))
    log_Mp_true = float(np.log10(317.8))
    truth = _Orbit(
        a=jnp.array(5.2),
        e=jnp.array(0.2),
        cos_i=jnp.array(0.5),
        W=jnp.array(0.8),
        omega=jnp.array(1.0),
        tp=jnp.array(1500.0),
        log_Mp=jnp.array(log_Mp_true),
    )
    err = 5e-5  # 0.05 mas

    def forward(o):
        Mp = (10.0**o.log_Mp) * Mearth2kg
        ra, dec = predict_stellar_astrometry(
            times,
            o.a,
            o.e,
            o.cos_i,
            o.W,
            jnp.cos(o.omega),
            jnp.sin(o.omega),
            o.tp,
            MSUN_KG,
            Mp,
            DIST_PC,
        )
        return jnp.concatenate([ra, dec])

    ra_t, dec_t = predict_stellar_astrometry(
        times,
        truth.a,
        truth.e,
        truth.cos_i,
        truth.W,
        jnp.cos(truth.omega),
        jnp.sin(truth.omega),
        truth.tp,
        MSUN_KG,
        (10.0**log_Mp_true) * Mearth2kg,
        DIST_PC,
    )
    key = jax.random.key(0)
    obs = jnp.concatenate([ra_t, dec_t]) + err * jax.random.normal(
        key, (2 * len(times),)
    )

    def likelihood(pred):
        return -0.5 * jnp.sum(((pred - obs) / err) ** 2)

    prior = Uniform(
        low=jnp.array([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0]),
        high=jnp.array([10.0, 0.9, 1.0, 2 * np.pi, 2 * np.pi, P_days, 4.0]),
    )
    init = _Orbit(
        a=jnp.array(5.5),
        e=jnp.array(0.25),
        cos_i=jnp.array(0.45),
        W=jnp.array(0.7),
        omega=jnp.array(1.15),
        tp=jnp.array(1580.0),
        log_Mp=jnp.array(log_Mp_true + 0.2),
    )
    ld, z0, unravel = build_scene_logdensity(init, forward, likelihood, prior)
    post = LaplaceBackend(n_steps=800, min_eigenvalue=1e-12).run(ld, z0)
    fit = unravel(post.mean)
    sigma = jnp.sqrt(jnp.diag(post.cov))

    assert jnp.isfinite(post.evidence)
    assert abs(float(fit.log_Mp) - log_Mp_true) < 0.05  # mass to ~12%
    assert float(sigma[-1]) < 0.1  # log_Mp is well constrained, not degenerate


def test_numpyro_model_wires_stellar_channel():
    """build_orbit_logdensity wires the stellar channel and fits the mass."""
    times = jnp.linspace(0.0, 4000.0, 10)
    Mp = 317.8 * Mearth2kg
    ra, dec = predict_stellar_astrometry(
        Ms=MSUN_KG, Mp=Mp, dist_pc=DIST_PC, **_truth_args(times)
    )
    sd = StellarAstromData(
        times=times,
        ra=ra,
        dec=dec,
        ra_err=jnp.full(times.shape, 5e-5),
        dec_err=jnp.full(times.shape, 5e-5),
        corr=jnp.zeros(times.shape),
        is_valid=jnp.ones(times.shape, dtype=bool),
    )
    problem = build_orbit_logdensity(
        MSUN_KG, DIST_PC, stellar_astrom_data=sd, log_P_range=(3.0, 4.0)
    )
    # the planet mass is a fitted parameter once stellar astrometry is present
    assert "log_Mp" in problem.param_names
    z0 = jnp.zeros(len(problem.param_names))
    assert jnp.isfinite(problem.logdensity(z0))
