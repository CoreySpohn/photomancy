"""Tests for photomancy.orbit.ofti (OFTI rejection sampler).

Covers the core scale-and-rotate geometry (exact reference-epoch hit,
Kepler-consistent period), prior recovery, truth recovery, the output
container, and a posterior cross-check against NUTS.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from hwoutils.constants import G
from orbix.equations import period_to_sma
from orbix.equations.orbit import mean_anomaly_tp, mean_motion

from photomancy.orbit.data import AstromData
from photomancy.orbit.forward import predict_astrometry
from photomancy.orbit.grid_search import ParticlePosterior
from photomancy.orbit.ofti import (
    PARAM_NAMES,
    _default_ref_idx,
    _scale_rotate_score,
    ofti,
)
from photomancy.orbit.priors import ECC_PRIOR_NAMES, ecc_distribution

MSUN = 1.98892e30
DIST_PC = 10.0
MU = G * MSUN
T_TRUE = 365.25

TRUTH = {
    "a": float(period_to_sma(T_TRUE, MSUN)),
    "e": 0.1,
    "cos_i": 0.7,
    "W": 0.8,
    "cos_w": float(jnp.cos(0.5)),
    "sin_w": float(jnp.sin(0.5)),
    "tp": 40.0,
}


def make_data(key, n_obs, noise_mas=1.0, span=1.5 * T_TRUE):
    """Synthetic AstromData: n_obs epochs over `span` days at TRUTH."""
    times = jnp.linspace(0.0, span, n_obs)
    ra, dec = predict_astrometry(
        times,
        TRUTH["a"],
        TRUTH["e"],
        TRUTH["cos_i"],
        TRUTH["W"],
        TRUTH["cos_w"],
        TRUTH["sin_w"],
        TRUTH["tp"],
        MSUN,
        DIST_PC,
    )
    sigma = noise_mas / 1000.0  # arcsec
    kra, kdec = jax.random.split(key)
    ra = ra + sigma * jax.random.normal(kra, ra.shape)
    dec = dec + sigma * jax.random.normal(kdec, dec.shape)
    err = jnp.full(n_obs, sigma)
    return AstromData(
        times=times,
        ra=ra,
        dec=dec,
        ra_err=err,
        dec_err=err,
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


def _logP_from_a(a):
    """log10(period in days) from semimajor axis (AU) via Kepler III."""
    return np.log10(2.0 * np.pi / np.sqrt(MU / np.asarray(a) ** 3))


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


def test_scale_and_rotate_hits_ref_point():
    """Zero injected noise: the conditioned orbit passes through the ref point."""
    data = make_data(jax.random.PRNGKey(0), 4)
    ref = _default_ref_idx(data)
    t_ref = data.times[ref]
    tx = data.ra[ref]
    ty = data.dec[ref]

    # Arbitrary drawn shape; wide truncation so nothing is masked.
    row, ll = _scale_rotate_score(
        0.2,
        0.5,
        jnp.cos(1.0),
        jnp.sin(1.0),
        1.3,
        tx,
        ty,
        data,
        t_ref,
        MU,
        MSUN,
        DIST_PC,
        -30.0,
        30.0,
        0.99,
    )
    a, e, cos_i, W, cos_w, sin_w, tp = (float(row[i]) for i in range(7))
    ra_p, dec_p = predict_astrometry(
        jnp.atleast_1d(t_ref), a, e, cos_i, W, cos_w, sin_w, tp, MSUN, DIST_PC
    )
    assert jnp.isfinite(ll)
    assert float(ra_p[0]) == pytest.approx(float(tx), abs=1e-5)
    assert float(dec_p[0]) == pytest.approx(float(ty), abs=1e-5)


def test_period_kepler_consistent():
    """Derived period matches Kepler III for the scaled a; tp preserves m_ref."""
    data = make_data(jax.random.PRNGKey(0), 4)
    ref = _default_ref_idx(data)
    t_ref = data.times[ref]
    m_ref = 1.3
    row, _ = _scale_rotate_score(
        0.2,
        0.5,
        jnp.cos(1.0),
        jnp.sin(1.0),
        m_ref,
        data.ra[ref],
        data.dec[ref],
        data,
        t_ref,
        MU,
        MSUN,
        DIST_PC,
        -30.0,
        30.0,
        0.99,
    )
    a = float(row[0])
    tp = float(row[6])
    n_new = mean_motion(a, MU)
    m_recovered = float(mean_anomaly_tp(t_ref, n_new, tp) % (2.0 * jnp.pi))
    assert m_recovered == pytest.approx(m_ref % (2.0 * np.pi), abs=1e-4)


# ---------------------------------------------------------------------------
# Priors / output container
# ---------------------------------------------------------------------------


def test_returns_particle_posterior():
    """Output is a normalized ParticlePosterior with the expected contract."""
    data = make_data(jax.random.PRNGKey(0), 2)
    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(1),
        n_accept=500,
        log_P_range=(2.0, 3.0),
        batch=20000,
    )
    assert isinstance(pp, ParticlePosterior)
    assert pp.param_names == PARAM_NAMES
    assert pp.particles.shape == (500, 7)
    assert float(jax.scipy.special.logsumexp(pp.log_weights)) == pytest.approx(
        0.0, abs=1e-5
    )


def test_prior_recovery():
    """Single epoch: accepted shape parameters reproduce the input priors."""
    data = make_data(jax.random.PRNGKey(0), 1)
    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(1),
        n_accept=6000,
        ecc_prior="disk",
        log_P_range=(-30.0, 30.0),
        e_max=0.999,
        batch=20000,
    )
    s = pp.sample(jax.random.PRNGKey(2), 6000)
    cos_i = np.asarray(s["cos_i"])
    e = np.asarray(s["e"])
    assert cos_i.mean() == pytest.approx(0.0, abs=0.05)  # Uniform(-1, 1)
    assert e.mean() == pytest.approx(0.5, abs=0.05)  # Uniform(0, 1)


def test_single_epoch_broad_posterior():
    """Single epoch yields a broad (prior-like) period posterior, not pinned."""
    data = make_data(jax.random.PRNGKey(0), 1)
    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(1),
        n_accept=4000,
        ecc_prior="disk",
        log_P_range=(2.0, 3.0),
        batch=20000,
    )
    s = pp.sample(jax.random.PRNGKey(2), 4000)
    logP = _logP_from_a(s["a"])
    # Broad: fills much of the unit-wide log-P range (std of U[2,3] ~ 0.29).
    assert logP.std() > 0.15
    assert logP.min() >= 2.0 - 1e-6
    assert logP.max() <= 3.0 + 1e-6


# ---------------------------------------------------------------------------
# Truth recovery
# ---------------------------------------------------------------------------


def test_truth_recovery():
    """Short arc (OFTI's sweet spot): posterior brackets and constrains truth a."""
    data = make_data(jax.random.PRNGKey(3), 3, noise_mas=1.5, span=0.08 * T_TRUE)
    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(4),
        n_accept=800,
        ecc_prior="disk",
        log_P_range=(1.5, 4.0),
        batch=100000,
        max_batches=200,
    )
    a = np.asarray(pp.sample(jax.random.PRNGKey(5), 3000)["a"])
    lo, hi = np.percentile(a, [2.5, 97.5])
    assert lo <= TRUTH["a"] <= hi
    assert np.median(a) == pytest.approx(TRUTH["a"], rel=0.4)


# ---------------------------------------------------------------------------
# Eccentricity distribution helper
# ---------------------------------------------------------------------------


def test_ecc_distribution():
    """ecc_distribution returns the right distribution per registry name."""
    assert isinstance(ecc_distribution("kipping13"), dist.Beta)
    assert isinstance(ecc_distribution("disk"), dist.Uniform)
    assert isinstance(ecc_distribution("rayleigh"), dist.Weibull)
    assert isinstance(ecc_distribution("vaneylen19"), dist.Weibull)
    for name in ECC_PRIOR_NAMES:
        draws = ecc_distribution(name).sample(jax.random.PRNGKey(0), (2000,))
        assert bool(jnp.all(draws >= 0.0))
    with pytest.raises(ValueError):
        ecc_distribution("not-a-prior")


# ---------------------------------------------------------------------------
# Posterior cross-check against NUTS (the decisive correctness test)
# ---------------------------------------------------------------------------


def test_ofti_matches_nuts():
    """OFTI and NUTS marginals agree on a short arc with matched priors."""
    from functools import partial

    from numpyro.infer import MCMC, NUTS, init_to_value

    from photomancy.orbit.data import ImagingData, NullData, RVData
    from photomancy.orbit.init import find_init
    from photomancy.orbit.model import build_model

    log_P_range = (1.5, 4.0)
    data = make_data(jax.random.PRNGKey(7), 3, noise_mas=2.0, span=0.15 * T_TRUE)

    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(8),
        n_accept=2000,
        ecc_prior="kipping13",
        log_P_range=log_P_range,
        batch=100000,
        max_batches=400,
    )
    a_ofti = np.asarray(pp.sample(jax.random.PRNGKey(9), 2000)["a"])

    padded = AstromData.pad(
        times=data.times,
        ra=data.ra,
        dec=data.dec,
        ra_err=data.ra_err,
        dec_err=data.dec_err,
        corr=data.corr,
        planet_id=data.planet_id,
    )
    model = partial(
        build_model(has_astrom=True, log_P_range=log_P_range),
        MSUN,
        DIST_PC,
        RVData.zeros(),
        padded,
        NullData.zeros(),
        ImagingData.zeros(),
    )
    iv = find_init(data, MSUN, DIST_PC, log_T_range=log_P_range)
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9, init_strategy=init_to_value(values=iv)),
        num_warmup=500,
        num_samples=2000,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(10))
    a_nuts = np.asarray(mcmc.get_samples()["a"])

    # Marginal a quantiles agree within a loose tolerance (MC + prior-shape noise).
    for q in (10, 50, 90):
        assert (
            abs(np.percentile(a_ofti, q) - np.percentile(a_nuts, q)) / TRUTH["a"] < 0.25
        )
