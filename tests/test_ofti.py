"""Tests for photomancy.orbit.ofti (OFTI rejection sampler).

Covers the core scale-and-rotate geometry (exact reference-epoch hit,
Kepler-consistent period), prior recovery, truth recovery, the output
container, and a posterior cross-check against NUTS.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G
from hwoutils.epoch import Epoch
from kinematics import predict_relative_astrometry
from orbix.equations import period_to_sma
from orbix.equations.orbit import mean_anomaly_tp, mean_motion

from photomancy.infer.posterior import SamplePosterior
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.inference import ORBIT_ROWS_CHART
from photomancy.orbit.ofti import (
    PARAM_NAMES,
    ScaleAndRotate,
    _default_ref_idx,
    _draw_anchor_proposals,
    chart_scale_rotate,
    ofti,
    scored_orbit,
)
from photomancy.priors import ECC_PRIOR_NAMES, eccentricity_draws

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


def _anchor_correction(key, n, data, ecc_prior="disk"):
    """``2 ln rho`` of the reference targets that ``sample_and_score`` draws."""
    *_, tx, ty = _draw_anchor_proposals(key, n, data, 0, ecc_prior)
    return 2.0 * jnp.log(jnp.hypot(tx, ty))


_sample_and_score = eqx.filter_jit(ScaleAndRotate().sample_and_score)
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
    row, ll = scored_orbit(
        0.2,
        0.5,
        1.0,
        1.3,
        tx,
        ty,
        data,
        t_ref,
        MU,
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
    row, _ = scored_orbit(
        0.2,
        0.5,
        1.0,
        m_ref,
        data.ra[ref],
        data.dec[ref],
        data,
        t_ref,
        MU,
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


# Proposals (e, cos_i, omega, m_ref, tx, ty) of 512 draws (key 3) at the first six
# supported lanes of the removed perifocal scale-and-rotate route, with its rows and
# uncorrected scores there, frozen (window (2, 3), e_max 0.9).
_FROZEN_INPUTS = [
    [
        0.299536018807035,
        -0.19542685878630683,
        1.1315200830810292,
        1.388375811943064,
        0.06586971839905724,
        0.07543176677741478,
    ],
    [
        0.39857716718560593,
        -0.18542189994528657,
        4.052317488345221,
        3.4203994146925227,
        0.05940571255048556,
        0.06444331421983666,
    ],
    [
        0.5139847800789855,
        -0.458542104696944,
        1.19805282848515,
        6.016222746849979,
        0.06046510644350518,
        0.06833937873009548,
    ],
    [
        0.033766095416027424,
        -0.43369750779179483,
        0.10165529204016559,
        5.0845810476564415,
        0.052769150508682396,
        0.06707403347786732,
    ],
    [
        0.5378346030277037,
        -0.7085595964335591,
        2.6767050794878013,
        3.603185014744072,
        0.05324808475581141,
        0.0759643497478638,
    ],
    [
        0.08506949958504499,
        0.1326732187684061,
        3.481116131683367,
        0.16401795677348716,
        0.05622684451134807,
        0.065968818703686,
    ],
]
_FROZEN_LL = [
    -319.78038027811203,
    -289.2987651160807,
    -699.2889371800504,
    -1810.3336105398866,
    -1092.7519483388032,
    -106.98625355782531,
]
_FROZEN_ROWS = [
    [
        0.9684655540751232,
        0.299536018807035,
        -0.19542685878630683,
        3.8541304556760396,
        0.4252845338610497,
        0.9050597025934751,
        -76.912364500207,
    ],
    [
        1.1906705794589163,
        0.39857716718560593,
        -0.18542189994528657,
        1.051998252246594,
        -0.6131733285623259,
        -0.7899483965043527,
        -258.3016103675115,
    ],
    [
        1.7094711022489266,
        0.5139847800789855,
        -0.458542104696944,
        0.8719448650941238,
        0.3641719063534632,
        0.931331746813607,
        -781.5907467985706,
    ],
    [
        1.5328392382008815,
        0.033766095416027424,
        -0.43369750779179483,
        6.165462608275634,
        0.9948375487422884,
        0.10148030159806941,
        -560.8709470933067,
    ],
    [
        0.6265723988858183,
        0.5378346030277037,
        -0.7085595964335591,
        0.3966493025534922,
        -0.8938719742939885,
        0.44832230991973526,
        -103.8738691540535,
    ],
    [
        1.095969523846897,
        0.08506949958504499,
        0.1326732187684061,
        3.7690460326552095,
        -0.9429134723901245,
        -0.33303781104432867,
        -10.938353895306681,
    ],
]


def test_scale_rotate_matches_frozen_perifocal_route():
    """Fails if the orbix-built orbit rows or scores drift from the frozen target.

    Six frozen proposals spanning the eccentricity law, cos_i in (-1, 1) and uniform
    phase, scored on four epochs of 5 mas data: the rows match the frozen rows to
    rtol 1e-9 (both routes preserve the reference mean anomaly) and the scores to
    the 64 eps floor. Through ``sample_and_score`` the support mask is the window
    and e_max exactly, and the importance correction ``-2 ln rho`` is the only
    difference from the uncorrected score.
    """
    data = make_data(jax.random.PRNGKey(0), 4, noise_mas=5.0, span=0.6 * T_TRUE)
    inputs = jnp.asarray(_FROZEN_INPUTS)
    t_ref = data.times[0]
    score_data = eqx.tree_at(lambda d: d.is_valid, data, data.is_valid.at[0].set(False))
    rows, ll = jax.vmap(
        lambda r: scored_orbit(*r, score_data, t_ref, MU, DIST_PC, 2.0, 3.0, 0.9)
    )(inputs)
    np.testing.assert_allclose(rows, np.asarray(_FROZEN_ROWS), rtol=1e-9, atol=1e-9)
    error = np.abs(np.asarray(ll) - np.asarray(_FROZEN_LL))
    floor = max(1e-10, 64 * np.finfo(float).eps * np.sum(1 + np.abs(_FROZEN_LL)))
    assert error.max() <= floor
    print(f"OFTI: score error {error.max():.3g}, floor {floor:.3g}")

    key = jax.random.PRNGKey(3)
    args = (key, 512, data, 0, MSUN, DIST_PC, "kipping13")
    _, scored = _sample_and_score(*args, (2.0, 3.0), 0.9)
    _, uncut = _sample_and_score(*args, (-30.0, 30.0), 1.0)
    inside = np.isfinite(np.asarray(scored))
    assert 0.2 < inside.mean() < 0.95
    np.testing.assert_array_equal(np.isfinite(np.asarray(uncut)), True)
    np.testing.assert_allclose(
        np.asarray(scored)[inside], np.asarray(uncut)[inside], rtol=1e-12
    )


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


def test_corrected_anchor_prior_is_the_default_and_is_distinguishable():
    """Fails if OFTI samples the induced prior instead of the corrected f(e)/e prior.

    One epoch at (1, 0) arcsec, sigma 0.1 arcsec, the period window wide: the
    accepted target separation rho has density proportional to N(rho) / rho**2
    (the corrected prior, the default) and not N(rho) (the induced prior). Mean
    ln rho of 4000 iid accepted orbits (SE = std / sqrt(4000), N = (4 sigma /
    delta)**2 with delta 0.02 and sigma 0.1 is 400) matches the corrected
    quadrature reference to 4 SE and misses the induced one by more than 4 SE.
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
    rho = np.linspace(0.2, 2.0, 1800)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)[None, :]
    gauss = np.exp(-((rho * np.cos(phi) - 1.0) ** 2 + (rho * np.sin(phi)) ** 2) / 0.02)
    corrected = np.sum(np.log(rho) * gauss / rho) / np.sum(gauss / rho)
    induced = np.sum(np.log(rho) * gauss * rho) / np.sum(gauss * rho)
    # rho >= 0.5 (5 sigma) bounds the corrected score at -2 ln 0.5.
    post = ofti(
        one,
        Ms=MSUN,
        dist_pc=DIST_PC,
        key=jax.random.PRNGKey(0),
        n_accept=4000,
        log_P_range=(-30.0, 30.0),
        e_max=0.99,
        batch=20000,
        max_batches=1,
        loglike_ref=-2.0 * np.log(0.5),
    )
    ra, dec = _predict_rows(one.times, *post.samples.T, MSUN, DIST_PC)
    ln_rho = np.log(np.hypot(np.asarray(ra[:, 0]), np.asarray(dec[:, 0])))
    mean, se = ln_rho.mean(), ln_rho.std() / np.sqrt(len(ln_rho))
    assert abs(mean - corrected) < 4 * se
    assert abs(mean - induced) > 4 * se
    print(round(mean, 4), round(se, 4), corrected, induced)


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
    # The only score is the importance correction -2 ln rho of the reference target
    # (rho well under 1 arcsec here, so it is positive and below the bound 0.0).
    correction = _anchor_correction(proposal_key, 4096, data)
    np.testing.assert_allclose(ll + correction, 0.0, atol=1e-12)
    assert bool(jnp.all(ll > 0.0))
    if loglike_ref is not None:
        # A bound of 0.0 accepts every in-support proposal in order.
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
    ll = ll + _anchor_correction(jax.random.PRNGKey(0), 5, data, "disk")
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


def test_eccentricity_draws():
    """Every named law draws eccentricities in [0, 1); unknown names raise.

    Fails if a registry name is dropped or a law leaves the unit interval.
    """
    for name in ECC_PRIOR_NAMES:
        draws = eccentricity_draws(jax.random.PRNGKey(0), (2000,), name)
        assert draws.shape == (2000,)
        assert bool(jnp.all((draws >= 0.0) & (draws < 1.0)))
    with pytest.raises(ValueError):
        eccentricity_draws(jax.random.PRNGKey(0), (2,), "not-a-prior")


# ---------------------------------------------------------------------------
# Posterior cross-check against NUTS (the decisive correctness test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ofti_matches_nuts():
    """OFTI and NUTS marginals agree on a short arc with matched priors."""
    from orbit_fit import chart_problem
    from orbix.system import ClassicalChart

    from photomancy.infer.backends import NUTSBackend
    from photomancy.orbit.init import find_init

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
        e_max=1.0,
        batch=100000,
        max_batches=400,
    )
    a_ofti = np.asarray(pp.sample_dict(jax.random.PRNGKey(9), 2000)["a"])

    chart = ClassicalChart()
    starts, _ = find_init(data, MSUN, DIST_PC, log_T_range=log_P_range, chart=chart)
    problem = chart_problem(
        data,
        {"orbit": starts[0]},
        star_mass_kg=MSUN,
        dist_pc=DIST_PC,
        chart=chart,
        period_range_d=tuple(float(10.0**x) for x in log_P_range),
    )
    nuts = problem.run(
        NUTSBackend(n_warmup=500, n_samples=2000),
        problem.flatten({"orbit": starts[0]}),
        jax.random.PRNGKey(10),
    )
    a_nuts = np.asarray(jax.vmap(problem.to_physical)(nuts.samples)["a"])

    # Marginal a quantiles agree within a loose tolerance (MC + prior-shape noise).
    for q in (10, 50, 90):
        assert (
            abs(np.percentile(a_ofti, q) - np.percentile(a_nuts, q)) / TRUTH["a"] < 0.25
        )
