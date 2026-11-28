"""Tests for stellar-reflex astrometry: the forward, the likelihood, and the fit.

Stellar astrometry measures the star's reflex wobble (scaled by the mass ratio),
so unlike relative astrometry it constrains the planet mass. These tests cover the
forward (a scaled, sign-flipped relative orbit), the shared bivariate-Gaussian
likelihood, mass identifiability through the generic engine, and the wiring of the
stellar channel into the chart density.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg
from kinematics import (
    predict_relative_astrometry,
    predict_stellar_astrometry,
)
from priorities import Uniform

from photomancy.infer import (
    Fit,
    ForwardTerm,
    Laplace,
    run,
    weighted_cov,
    weighted_mean,
)
from photomancy.infer.likelihoods import gaussian_loglike
from photomancy.measurements import (
    AstrometryData,
    loglike_relative_astrom,
    loglike_stellar_astrom,
)

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

    ra_rel, dec_rel = predict_relative_astrometry(
        Ms=MSUN_KG, Mp=Mp, dist_pc=DIST_PC, **args
    )
    ra_star, dec_star = predict_stellar_astrometry(
        Ms=MSUN_KG, Mp=Mp, dist_pc=DIST_PC, **args
    )

    frac = Mp / (MSUN_KG + Mp)
    assert jnp.allclose(ra_star, -frac * ra_rel)
    assert jnp.allclose(dec_star, -frac * dec_rel)
    # the reflex is much smaller than the relative orbit (mass ratio ~1e-3)
    assert jnp.max(jnp.abs(ra_star)) < jnp.max(jnp.abs(ra_rel))


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_stellar_and_relative_likelihoods_share_form():
    """The two astrometry likelihoods are the same bivariate Gaussian on (ra, dec)."""
    n = 5
    ra_pred = jnp.linspace(-0.1, 0.1, n)
    dec_pred = jnp.linspace(0.05, -0.05, n)
    data = AstrometryData(
        time_d=jnp.arange(n, dtype=float),
        east_arcsec=ra_pred + 1e-3,
        north_arcsec=dec_pred - 1e-3,
        east_err_arcsec=jnp.full(n, 2e-3),
        north_err_arcsec=jnp.full(n, 2e-3),
        corr=jnp.zeros(n),
        planet=jnp.zeros(n, dtype=int),
        valid=jnp.ones(n, dtype=bool),
    )

    ll_stellar = loglike_stellar_astrom(ra_pred, dec_pred, data)
    ll_relative = loglike_relative_astrom(ra_pred, dec_pred, data)
    assert jnp.allclose(ll_stellar, ll_relative)


class _Orbit(eqx.Module):
    a: jnp.ndarray
    e: jnp.ndarray
    cos_i: jnp.ndarray
    W: jnp.ndarray
    omega: jnp.ndarray
    tp: jnp.ndarray
    log_Mp: jnp.ndarray


class _Reflex(eqx.Module):
    """``orbit -> (ra, dec)`` stellar reflex, concatenated."""

    times: jax.Array

    def __call__(self, o):
        Mp = (10.0**o.log_Mp) * Mearth2kg
        ra, dec = predict_stellar_astrometry(
            self.times,
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(
    reason="compiles gradient and curvature for the joint orbit and stellar-reflex "
    "fit, 19 s projected on CI"
)
def test_mass_is_identifiable_from_reflex():
    """A reflex-astrometry fit recovers the planet mass (period -> a -> amplitude).

    Fails if the reflex amplitude loses its mass dependence (the mass is then
    unconstrained and the width test fails) or the fit does not reach the mode.
    """
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
    forward = _Reflex(times)
    obs = forward(truth) + err * jax.random.normal(jax.random.key(0), (2 * len(times),))

    low = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0]
    high = [10.0, 0.9, 1.0, 2 * np.pi, 2 * np.pi, P_days, 4.0]
    paths = ("a", "e", "cos_i", "W", "omega", "tp", "log_Mp")
    prior = {p: Uniform(lo, hi) for p, lo, hi in zip(paths, low, high, strict=True)}
    x0 = dict(
        a=jnp.array(5.5),
        e=jnp.array(0.25),
        cos_i=jnp.array(0.45),
        W=jnp.array(0.7),
        omega=jnp.array(1.15),
        tp=jnp.array(1580.0),
        log_Mp=jnp.array(log_Mp_true + 0.2),
    )
    fit = Fit(truth, prior, (ForwardTerm(forward, gaussian_loglike(obs, err)),))
    post = run(fit, Laplace(n_draws=512, n_steps=800, init=x0), jax.random.key(1))

    assert int(post.info.n_modes_kept) == 1
    lw, mass = post.log_weights, post.samples["log_Mp"]
    assert abs(float(weighted_mean(lw, mass)) - log_Mp_true) < 0.05  # mass to ~12%
    # log_Mp is well constrained, not degenerate
    assert float(jnp.sqrt(weighted_cov(lw, mass))) < 0.1
