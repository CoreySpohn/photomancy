"""Tests for the Hipparcos-Gaia proper-motion-anomaly (PMa) observable.

The PMa is the difference between the Gaia instantaneous proper motion (a path-fit
slope over the mission window) and the long-baseline Hipparcos-Gaia mean proper
motion. The barycenter proper motion cancels, so the anomaly is a pure reflex signal
that scales with the planet mass and pins the mass when the orbit shape is known.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Mearth2kg
from kinematics import (
    predict_pm_anomaly,
    predict_relative_astrometry,
)
from orbix.equations.orbit import period_a, period_to_sma
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
    ProperMotionAnomalyData,
    loglike_pm_anomaly,
)

MSUN_KG = 1.989e30
DIST_PC = 10.0
MU = G * MSUN_KG
TWO_PI = 2.0 * np.pi
YR = 365.25
T_HIP, T_GAIA = 0.0, 24.75 * YR
GAIA_WIN = 2.8 * YR

# truth: a Jupiter analog
A_TRUE, E_TRUE, COSI_TRUE, W_TRUE, OM_TRUE, M0_TRUE = 5.2, 0.15, 0.4, 0.8, 1.0, 0.6
LOG_MP_TRUE = float(np.log10(317.8))
P_TRUE = float(period_a(A_TRUE, MU))
TP_TRUE = -M0_TRUE * P_TRUE / TWO_PI
CW_TRUE, SW_TRUE = float(np.cos(OM_TRUE)), float(np.sin(OM_TRUE))


def _anom(Mp):
    return predict_pm_anomaly(
        T_HIP,
        T_GAIA,
        GAIA_WIN,
        25,
        A_TRUE,
        E_TRUE,
        COSI_TRUE,
        W_TRUE,
        CW_TRUE,
        SW_TRUE,
        TP_TRUE,
        MSUN_KG,
        Mp,
        DIST_PC,
    )


def test_pm_anomaly_scales_linearly_with_mass():
    """The reflex anomaly is linear in the planet mass."""
    Mp = 317.8 * Mearth2kg
    a1 = _anom(Mp)
    a2 = _anom(2.0 * Mp)
    assert a1.shape == (2,)
    assert jnp.all(jnp.isfinite(a1))
    assert jnp.allclose(a2, 2.0 * a1, rtol=1e-6)
    # realistic magnitude: ~0.1-1 mas/yr for a Jupiter at 10 pc
    mas_per_yr = float(jnp.hypot(*a1)) * 1e3 * YR
    assert 0.02 < mas_per_yr < 2.0


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_pm_anomaly_likelihood_masks_invalid():
    """The placeholder channel (valid False) contributes zero log-likelihood."""
    placeholder = ProperMotionAnomalyData.zeros()
    ll = loglike_pm_anomaly(jnp.array([1.0, -1.0]), placeholder)
    assert float(ll) == 0.0


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_pm_anomaly_likelihood_peaks_at_truth():
    """The likelihood is maximized at the true anomaly."""
    anom_true = _anom(317.8 * Mearth2kg)
    cov = (0.02e-3 / YR) ** 2 * jnp.eye(2)
    data = ProperMotionAnomalyData(
        t_hip=jnp.array(T_HIP),
        t_gaia=jnp.array(T_GAIA),
        gaia_window=jnp.array(GAIA_WIN),
        pm_anomaly=anom_true,
        pm_anomaly_cov=cov,
        valid=jnp.array(True),
    )
    ll_true = loglike_pm_anomaly(anom_true, data)
    ll_off = loglike_pm_anomaly(anom_true + 0.1e-3 / YR, data)
    assert float(ll_true) > float(ll_off)


class _Sys(eqx.Module):
    logP: jnp.ndarray
    e: jnp.ndarray
    cos_i: jnp.ndarray
    W: jnp.ndarray
    omega: jnp.ndarray
    M0: jnp.ndarray
    log_Mp: jnp.ndarray


class _Forward(eqx.Module):
    """``system -> (ra, dec, pma)`` concatenated: relative orbit plus the anomaly."""

    t_ast: jax.Array

    def __call__(self, s):
        P = 10.0**s.logP
        a = period_to_sma(P, MSUN_KG)
        tp = -s.M0 * P / TWO_PI
        cw, sw = jnp.cos(s.omega), jnp.sin(s.omega)
        Mp = (10.0**s.log_Mp) * Mearth2kg
        ra, dec = predict_relative_astrometry(
            self.t_ast, a, s.e, s.cos_i, s.W, cw, sw, tp, MSUN_KG, DIST_PC
        )
        anom = predict_pm_anomaly(
            T_HIP, T_GAIA, GAIA_WIN, 25, a, s.e, s.cos_i, s.W, cw, sw, tp, MSUN_KG, Mp,
            DIST_PC,
        )  # fmt: skip
        return jnp.concatenate([ra, dec, anom])


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.slow(
    reason="compiles joint seven-parameter PMa gradient and Hessian, 32.93 s on CI"
)
def test_joint_relative_astrom_and_pma_recovers_mass():
    """Relative astrometry pins the orbit shape; the PMa then pins the mass.

    Fails if the anomaly channel is dropped from the forward or mis-scaled (the mass
    is then unconstrained and the width test fails), or if the Laplace stage does not
    reach the joint mode.
    """
    t_ast = jnp.asarray(np.linspace(0.0, T_GAIA, 12))
    forward = _Forward(t_ast)
    truth = _Sys(
        logP=jnp.array(np.log10(P_TRUE)),
        e=jnp.array(E_TRUE),
        cos_i=jnp.array(COSI_TRUE),
        W=jnp.array(W_TRUE),
        omega=jnp.array(OM_TRUE),
        M0=jnp.array(M0_TRUE),
        log_Mp=jnp.array(LOG_MP_TRUE),
    )
    aerr, pmerr = 1e-3, 0.02e-3 / YR
    sigma = jnp.concatenate([jnp.full(24, aerr), jnp.full(2, pmerr)])
    noise = jnp.concatenate(
        [
            jax.random.normal(jax.random.key(0), (12,)),
            jax.random.normal(jax.random.key(1), (12,)),
            jax.random.normal(jax.random.key(2), (2,)),
        ]
    )
    obs = forward(truth) + sigma * noise

    low = [3.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
    high = [4.0, 0.9, 1.0, TWO_PI, TWO_PI, TWO_PI, 4.0]
    paths = ("logP", "e", "cos_i", "W", "omega", "M0", "log_Mp")
    prior = {p: Uniform(lo, hi) for p, lo, hi in zip(paths, low, high, strict=True)}
    x0 = {p: getattr(truth, p) for p in paths}
    x0["log_Mp"] = x0["log_Mp"] + 0.3
    fit = Fit(truth, prior, (ForwardTerm(forward, gaussian_loglike(obs, sigma)),))
    post = run(fit, Laplace(n_draws=512, n_steps=100, init=x0), jax.random.key(3))
    lw, mass = post.log_weights, post.samples["log_Mp"]
    mean = float(weighted_mean(lw, mass))
    width = float(jnp.sqrt(weighted_cov(lw, mass)))
    assert abs(mean - LOG_MP_TRUE) < 0.1  # mass recovered to ~25%
    assert width < 0.2  # mass is constrained
