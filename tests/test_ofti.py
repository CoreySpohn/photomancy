"""Tests for photomancy.orbit.ofti (OFTI rejection sampler).

Covers the core scale-and-rotate geometry (exact reference-epoch hit,
Kepler-consistent period), prior recovery, truth recovery, the output
container, and a posterior cross-check against NUTS.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from hwoutils.constants import G
from hwoutils.epoch import Epoch
from orbix.equations import period_to_sma
from orbix.equations.orbit import mean_anomaly_tp, mean_motion

from photomancy.orbit._frame import ORBIT_ROWS_CHART
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.ofti import (
    PARAM_NAMES,
    ChartScaleAndRotate,
    ScaleAndRotate,
    _default_ref_idx,
    _scale_rotate_score,
    chart_scale_rotate,
    ofti,
)
from photomancy.orbit.priors import ECC_PRIOR_NAMES, ecc_distribution
from photomancy.posterior import SamplePosterior

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
    """Synthetic RelativeAstromData: n_obs epochs over `span` days at TRUTH."""
    times = jnp.linspace(0.0, span, n_obs)
    ra, dec = predict_relative_astrometry(
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
    return RelativeAstromData(
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


_sample_and_score = eqx.filter_jit(ScaleAndRotate().sample_and_score)
_chart_sample_and_score = eqx.filter_jit(ChartScaleAndRotate().sample_and_score)
_predict_rows = eqx.filter_jit(
    jax.vmap(
        predict_relative_astrometry, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None, None)
    )
)


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
    ra_p, dec_p = predict_relative_astrometry(
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


def test_chart_scale_rotate_matches_scale_rotate():
    """Fails if the chart route's orbit rows or scores drift from the legacy target.

    Identical draws (one key) through both conditioners: 512 shapes spanning the
    eccentricity law, cos_i in (-1, 1), uniform phase and the periods the scaling
    implies, with a period window that masks about half of them. The support
    masks agree exactly; rows match to rtol 1e-9 (tp modulo nothing: both
    preserve the reference mean anomaly) and scores to the 64 eps floor.
    """
    data = make_data(jax.random.PRNGKey(0), 4, noise_mas=5.0, span=0.6 * T_TRUE)
    args = (jax.random.PRNGKey(3), 512, data, 0, MSUN, DIST_PC, "kipping13")
    window = ((2.0, 3.0), 0.9)
    rows_old, ll_old = _sample_and_score(*args, *window)
    rows_new, ll_new = _chart_sample_and_score(*args, *window)
    finite = np.isfinite(np.asarray(ll_old))
    assert 0.2 < finite.mean() < 0.9
    np.testing.assert_array_equal(finite, np.isfinite(np.asarray(ll_new)))
    np.testing.assert_allclose(rows_new, rows_old, rtol=1e-9, atol=1e-9)
    error = np.abs(np.asarray(ll_new - ll_old))[finite]
    floor = max(1e-10, 64 * np.finfo(float).eps * np.sum(1 + np.abs(ll_old[finite])))
    assert error.max() <= floor
    ra, dec = _predict_rows(data.times, *rows_new.T, MSUN, DIST_PC)
    ra_old, dec_old = _predict_rows(data.times, *rows_old.T, MSUN, DIST_PC)
    np.testing.assert_allclose(ra, ra_old, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dec, dec_old, rtol=1e-9, atol=1e-12)
    print(f"chart OFTI: score error {error.max():.3g}, floor {floor:.3g}")


def test_anchor_map_jacobian_is_inverse_rho_squared():
    """Fails if the induced anchor prior is not f(e) rho**2 / e.

    At six shapes (e from 0.05 to 0.8, cos_i of both signs, several phases and
    scales), autodiff of (ln a, W) with respect to the target (tx, ty) has
    absolute determinant 1 / rho**2 to 1e-9: a Gaussian target therefore induces
    rho**2 on the chart measure, which is not the flat-ln-a prior.
    """
    shapes = np.array(
        [
            [0.05, 0.7, 0.4, 0.3],
            [0.2, -0.5, 1.9, 2.2],
            [0.45, 0.1, 3.3, 4.0],
            [0.6, -0.9, 5.2, 1.1],
            [0.8, 0.3, 0.9, 5.5],
            [0.3, 0.95, 2.6, 3.1],
        ]
    )
    targets = np.array(
        [[0.3, 0.2], [-0.1, 0.5], [0.8, -0.4], [0.05, -0.02], [-0.6, -0.7], [1.4, 0.9]]
    )

    def coordinates(t, shape):
        orbit = chart_scale_rotate(
            *shape, *t, epoch_d=3.0, mu_AU3_per_d2=MU, dist_pc=DIST_PC
        )
        return jnp.stack([jnp.log(orbit.a_AU), orbit.to_classical()["W_rad"]])

    for shape, target in zip(shapes, targets, strict=True):
        jac = jax.jacfwd(coordinates)(jnp.asarray(target), jnp.asarray(shape))
        rho2 = float(target @ target)
        assert abs(abs(float(jnp.linalg.det(jac))) * rho2 - 1.0) < 1e-9


def test_corrected_anchor_prior_is_distinguishable_from_the_induced_prior():
    """Fails if OFTI's induced prior and the corrected f(e)/e prior are not told apart.

    One epoch at (1, 0) arcsec, sigma 0.1 arcsec, the period window wide: the
    accepted target separation rho has density proportional to N(rho) for the
    default (induced) route and to N(rho) / rho**2 with correct_induced_prior.
    Mean ln rho of 4000 iid accepted orbits (SE = std / sqrt(4000), N = (4 sigma
    / delta)**2 with delta 0.02 and sigma 0.1 is 400) matches its own quadrature
    reference to 4 SE and misses the other's by more than 4 SE.
    """
    one = RelativeAstromData(
        times=jnp.zeros(1),
        ra=jnp.ones(1),
        dec=jnp.zeros(1),
        ra_err=jnp.full(1, 0.1),
        dec_err=jnp.full(1, 0.1),
        corr=jnp.zeros(1),
        planet_id=jnp.zeros(1, dtype=int),
        is_valid=jnp.ones(1, dtype=bool),
    )
    common = dict(
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(0),
        n_accept=4000,
        log_P_range=(-30.0, 30.0),
        e_max=0.99,
        batch=20000,
        max_batches=1,
    )
    rho = np.linspace(0.2, 2.0, 1800)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)[None, :]
    gauss = np.exp(-((rho * np.cos(phi) - 1.0) ** 2 + (rho * np.sin(phi)) ** 2) / 0.02)
    reference = {
        "induced": np.sum(np.log(rho) * gauss * rho) / np.sum(gauss * rho),
        "corrected": np.sum(np.log(rho) * gauss / rho) / np.sum(gauss / rho),
    }
    # Accept-all bound for the induced score; rho >= 0.5 (5 sigma) bounds the other.
    runs = {
        "induced": dict(conditioner=ChartScaleAndRotate(), loglike_ref=0.0),
        "corrected": dict(
            conditioner=ChartScaleAndRotate(correct_induced_prior=True),
            loglike_ref=-2.0 * np.log(0.5),
        ),
    }
    stats = {}
    for name, options in runs.items():
        post = ofti(one, **common, **options)
        ra, dec = _predict_rows(one.times, *post.samples.T, MSUN, DIST_PC)
        ln_rho = np.log(np.hypot(np.asarray(ra[:, 0]), np.asarray(dec[:, 0])))
        stats[name] = (ln_rho.mean(), ln_rho.std() / np.sqrt(len(ln_rho)))
    for name, (mean, se) in stats.items():
        other = "corrected" if name == "induced" else "induced"
        assert abs(mean - reference[name]) < 4 * se
        assert abs(mean - reference[other]) > 4 * se
    print({k: (round(v[0], 4), round(v[1], 4)) for k, v in stats.items()}, reference)


# ---------------------------------------------------------------------------
# Priors / output container
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loglike_ref", [None, 0.0])
def test_single_epoch_reference_scatter(loglike_ref):
    """Recover the reference error from 1024 accepted orbits.

    Fails if rejection weights the reference epoch twice or rejects any
    in-support proposal when the remaining likelihood is constant.
    """
    data = make_data(jax.random.PRNGKey(0), 1)
    key = jax.random.PRNGKey(0)
    n = 1024
    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=key,
        n_accept=n,
        ecc_prior="disk",
        log_P_range=(-30.0, 30.0),
        e_max=1.0,
        batch=4096,
        max_batches=1,
        loglike_ref=loglike_ref,
        epoch=Epoch(2460000.0),
    )
    assert (pp.chart, pp.epoch) == (ORBIT_ROWS_CHART, Epoch(2460000.0))
    ra, dec = _predict_rows(data.times, *pp.samples.T, MSUN, DIST_PC)
    residuals = np.column_stack(
        (
            (ra[:, 0] - data.ra[0]) / data.ra_err[0],
            (dec[:, 0] - data.dec[0]) / data.dec_err[0],
        )
    )
    # For Gaussian draws, SE(s / sigma) ~= 1 / sqrt(2 * (n - 1)).
    # Four SE at n=1024 is 0.0884, well below the double-count deficit 0.293.
    tolerance = 4.0 / np.sqrt(2.0 * (n - 1))
    np.testing.assert_allclose(residuals.std(axis=0, ddof=1), 1.0, atol=tolerance)

    _, proposal_key, _ = jax.random.split(key, 3)
    rows, ll = _sample_and_score(
        proposal_key, 4096, data, 0, MSUN, DIST_PC, "disk", (-30.0, 30.0), 1.0
    )
    np.testing.assert_array_equal(ll, 0.0)
    np.testing.assert_allclose(pp.samples, rows[:n], rtol=1e-5, atol=1e-5)


def test_score_uses_only_nonreference_epoch():
    """Fails if the reference epoch or an invalid epoch enters the score."""
    data = make_data(jax.random.PRNGKey(0), 3, noise_mas=10.0, span=0.08 * T_TRUE)
    data = eqx.tree_at(
        lambda d: (d.is_valid, d.corr),
        data,
        (jnp.array([True, True, False]), jnp.full(3, 0.4)),
    )
    rows, ll = _sample_and_score(
        jax.random.PRNGKey(0), 5, data, 0, MSUN, DIST_PC, "disk", (-30.0, 30.0), 1.0
    )
    ra, dec = _predict_rows(data.times, *rows.T, MSUN, DIST_PC)
    dx = np.asarray((data.ra[1] - ra[:, 1]) / data.ra_err[1])
    dy = np.asarray((data.dec[1] - dec[:, 1]) / data.dec_err[1])
    rho = float(data.corr[1])
    second_epoch_ll = -0.5 * (dx**2 + dy**2 - 2.0 * rho * dx * dy) / (1.0 - rho**2)
    second_epoch_ll -= np.log(
        2.0 * np.pi * float(data.ra_err[1] * data.dec_err[1]) * np.sqrt(1.0 - rho**2)
    )
    np.testing.assert_allclose(ll, second_epoch_ll, rtol=1e-4, atol=1e-4)


# Medium: Two-epoch rejection sampling includes the cold OFTI program.
@pytest.mark.medium
def test_returns_particle_posterior():
    """Fails if two-epoch rejection loses its normalized SamplePosterior contract."""
    data = make_data(jax.random.PRNGKey(0), 2)
    pp = ofti(
        data,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(1),
        n_accept=32,
        log_P_range=(2.0, 3.0),
        batch=4096,
    )
    assert isinstance(pp, SamplePosterior)
    assert pp.param_names == PARAM_NAMES
    assert pp.samples.shape == (32, 7)
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
    s = pp.sample_dict(jax.random.PRNGKey(2), 6000)
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
    s = pp.sample_dict(jax.random.PRNGKey(2), 4000)
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
    a = np.asarray(pp.sample_dict(jax.random.PRNGKey(5), 3000)["a"])
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
    assert isinstance(ecc_distribution("rayleigh"), dist.Distribution)
    assert isinstance(ecc_distribution("vaneylen19"), dist.Distribution)
    for name in ECC_PRIOR_NAMES:
        draws = ecc_distribution(name).sample(jax.random.PRNGKey(0), (2000,))
        assert bool(jnp.all(draws >= 0.0))
    with pytest.raises(ValueError):
        ecc_distribution("not-a-prior")


# ---------------------------------------------------------------------------
# Posterior cross-check against NUTS (the decisive correctness test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ofti_matches_nuts():
    """OFTI and NUTS marginals agree on a short arc with matched priors."""
    from functools import partial

    from numpyro.infer import MCMC, NUTS, init_to_value

    from photomancy.orbit.data import OrbitData
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
    a_ofti = np.asarray(pp.sample_dict(jax.random.PRNGKey(9), 2000)["a"])

    padded = RelativeAstromData.pad(
        times=data.times,
        ra=data.ra,
        dec=data.dec,
        ra_err=data.ra_err,
        dec_err=data.dec_err,
        corr=data.corr,
        planet_id=data.planet_id,
    )
    model = partial(
        build_model(has_relative_astrom=True, log_P_range=log_P_range),
        MSUN,
        DIST_PC,
        OrbitData(relative_astrom=padded),
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
