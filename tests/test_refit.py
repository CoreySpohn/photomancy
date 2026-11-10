"""Windowed physical-prior refit: prior density, elements, and the acceptance test.

The unit and medium tests each protect one stated property. The slow test is the
acceptance test of the refit: on the fixture of ``tools/refit_golden.py`` it grades
eight keys per record count against the golden reference.
"""

import functools
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hwoutils.constants import G, Msun2kg, two_pi  # noqa: E402
from jax.scipy.special import betainc  # noqa: E402
from jax.scipy.stats import beta as beta_dist  # noqa: E402

from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.few_epoch import MU_SUN  # noqa: E402
from photomancy.orbit.forward import predict_relative_astrometry  # noqa: E402
from photomancy.orbit.refit import (  # noqa: E402
    ECC_ALPHA,
    ECC_BETA,
    _systematic_indices,
    collapsed_refit,
    importance_refit,
    physical_prior_target,
    ti_to_elements,
)
from photomancy.orbit.updater import UpdaterStatus  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "refit_golden.json"
DIST_PC = 10.0
CHART = jnp.array([0.2, np.log(400.0), 0.3, 0.5, -0.7, 0.9, 0.4])


def test_prior_density_carries_the_constants_jacobian():
    """The log prior is minus the log Jacobian of the map to the constants.

    Fails if the Jacobian ``a^4 (1 - cos^2 i)`` of ``(ln a, cos i, omega, W)`` to the
    four Thiele-Innes constants is missing or wrong, checked against autodiff of
    the inverse map.
    """
    target = physical_prior_target(1.0, DIST_PC)
    ln_p = CHART[1]

    def inverse(c):
        theta = jnp.concatenate([CHART[:3], c])
        el = ti_to_elements(theta, 0)
        ln_a = (el[6] + jnp.log(MU_SUN) - 2.0 * jnp.log(two_pi) + 2.0 * ln_p) / 3.0
        return jnp.stack([ln_a, el[2], el[3], el[4]])

    jac = jax.jacfwd(inverse)(CHART[3:])
    log_det = jnp.linalg.slogdet(jac)[1]
    e = CHART[0]
    log_e = beta_dist.logpdf(e, ECC_ALPHA, ECC_BETA) - jnp.log(
        betainc(ECC_ALPHA, ECC_BETA, target.model.e_max)
    )
    assert float(target.log_prior(CHART) - log_e) == pytest.approx(
        float(log_det), abs=1e-9
    )


@pytest.mark.parametrize("branch", [0, 1])
def test_elements_reproduce_the_chart_sky_track(branch):
    """Both mirror orbits of a chart point share its predicted sky positions."""
    target = physical_prior_target(1.0, DIST_PC)
    el = ti_to_elements(CHART, branch)
    ln_p, e, cos_i, node, omega, tau, ln_mass, _ = el
    period, mass = jnp.exp(ln_p), jnp.exp(ln_mass) * Msun2kg
    a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
    times = jnp.array([0.0, 100.0, 300.0])
    ra, dec = predict_relative_astrometry(
        times,
        a,
        e,
        cos_i,
        node,
        jnp.cos(omega),
        jnp.sin(omega),
        tau * period,
        mass,
        DIST_PC,
    )
    ra0, dec0 = target.predict(CHART, times)
    assert jnp.allclose(ra, ra0, atol=1e-12) and jnp.allclose(dec, dec0, atol=1e-12)


def test_the_mass_window_is_a_declared_width():
    """The window is the catalog mass times one plus or minus the declared width."""
    lo, hi = physical_prior_target(0.8, DIST_PC, 0.10).mass_window_msun()
    assert (lo, hi) == pytest.approx((0.72, 0.88))
    lo, hi = physical_prior_target(0.8, DIST_PC, 0.02).mass_window_msun()
    assert (lo, hi) == pytest.approx((0.784, 0.816))


def test_a_refit_without_records_has_no_support():
    """No valid record is `NO_SUPPORT`; a report term is refused until it exists."""
    target = physical_prior_target(1.0, DIST_PC)
    records = RelativeAstromData.zeros(3)
    cloud = importance_refit(
        target, records, None, jax.random.key(0), n_particles=8, n_round=2**8
    )
    assert int(cloud.status) == int(UpdaterStatus.NO_SUPPORT)
    assert bool(jnp.all(jnp.isnan(cloud.elements)))
    with pytest.raises(NotImplementedError):
        importance_refit(
            target, records, object(), jax.random.key(0), n_particles=8, n_round=2**8
        )


def test_proposals_above_the_density_block_must_divide_into_blocks():
    """A round of 1500 proposals is refused: the density blocks need whole 1024s."""
    target = physical_prior_target(1.0, DIST_PC)
    with pytest.raises(ValueError, match="multiple of 1024"):
        collapsed_refit(
            target,
            RelativeAstromData.zeros(3),
            jax.random.key(0),
            n_particles=8,
            n_round=1500,
        )


def test_a_small_cloud_resamples_from_every_proposal():
    """Eight picks from 64 equal weights span the proposals, not the first eight.

    Fails if the resampler clips its indices to the number of picks, which makes a
    1024-draw cloud a copy of the first 1024 proposals.
    """
    idx = _systematic_indices(jax.random.key(0), jnp.full(64, -jnp.log(64.0)), 8)
    assert int(idx.max()) >= 56 and len(set(np.asarray(idx).tolist())) == 8


# -- acceptance test ---------------------------------------------------------------

KEYS = 8
N_CLOUD = 4096
# Keys under the floor, at most, of the eight: the measured low-size rate of the
# collapsed refit is 0.10 at two and three records (200 worlds), so about one key in
# ten falls under it; a larger last round and a physical-prior component for the
# constants (which cut the largest weight by three) did not lower it.
MAX_LOW = {1: 0, 2: 2, 3: 2}
ESS_FLOOR = 1600.0
PDET_TOL = 0.05
FUSED = {
    1: dict(n_round=2**17),
    2: dict(n_round=2**16, n_last=2**18),
    3: dict(n_round=2**16, n_last=2**18),
}


def _replicate_ess(runs, cap):
    """Effective multinomial size from independent runs of bin masses, in 1 / n^2.

    With ``p`` the mean bin masses, ``M = diag(p) - p p^T`` and ``T`` the trace of
    the sample covariance on ``nu = K - 1`` degrees of freedom,
    ``n = sqrt((tr M)^2 + 2 tr(M^2) / nu) / T``.
    """
    x = np.asarray(runs, float)
    nu = x.shape[0] - 1
    p = x.mean(axis=0)
    t = float(np.sum(x.var(axis=0, ddof=1)))
    if t <= 0.0:
        return float(cap)
    m = np.diag(p) - np.outer(p, p)
    scale = np.sqrt(np.trace(m) ** 2 + 2.0 * np.sum(m * m) / nu)
    return float(min(max(scale / t, 1.0), cap))


def _positions(elements, future_d):
    from photomancy.orbit.forward import predict_relative_astrometry as predict

    def one(el):
        period, mass = jnp.exp(el[0]), jnp.exp(el[6]) * Msun2kg
        a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
        ra, dec = predict(
            jnp.asarray([future_d]),
            a,
            el[1],
            el[2],
            el[3],
            jnp.cos(el[4]),
            jnp.sin(el[4]),
            el[5] * period,
            mass,
            DIST_PC,
        )
        return jnp.stack([ra[0], dec[0]])

    return np.asarray(jax.vmap(one)(elements))


@functools.cache
def _graded(k):
    """Eight keys of the refit at record count ``k`` and the golden reference."""
    golden = json.loads(GOLDEN.read_text())
    ref = golden["reference"][str(k)]
    raw = golden["records"][str(k)]
    names = ("times", "ra", "dec", "ra_err", "dec_err", "corr")
    records = RelativeAstromData(
        **{n: jnp.asarray(raw[n]) for n in names},
        planet_id=jnp.zeros(len(raw["times"]), int),
        is_valid=jnp.asarray(raw["is_valid"], bool),
    )
    fx = golden["fixture"]
    target = physical_prior_target(
        fx["mass_msun"], fx["dist_pc"], fx["mass_half_width"]
    )
    edges = np.asarray(ref["bin_edges"])
    out = {key: [] for key in ("ess", "vis", "mean", "spread", "ratio", "bins")}
    for i in range(KEYS):
        cloud = importance_refit(
            target, records, None, jax.random.key(i), n_particles=N_CLOUD, **FUSED[k]
        )
        assert int(cloud.status) in (int(UpdaterStatus.OK), int(UpdaterStatus.LOW_ESS))
        xy = _positions(cloud.elements, fx["future_d"])
        out["ess"].append(float(cloud.ess))
        out["vis"].append(
            float(np.mean(np.hypot(xy[:, 0], xy[:, 1]) > fx["iwa_arcsec"]))
        )
        out["mean"].append(xy.mean(axis=0))
        out["spread"].append(xy.std(axis=0))
        out["ratio"].append(np.exp(float(cloud.log_evidence) - ref["log_evidence"]))
        hist, _ = np.histogram(np.asarray(cloud.elements[:, 0]), bins=edges)
        out["bins"].append(hist / hist.sum())
    sd = np.sqrt(np.diag(np.asarray(ref["position_cov"])))
    return ref, sd, {key: np.asarray(v) for key, v in out.items()}


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
def test_refit_matches_the_golden_reference(k):
    """Acceptance items 2 to 5: eight keys agree with the exact reference.

    The visibility probability is within 0.05 of the golden value; the mean predicted
    position is within a quarter of a predictive standard deviation and its spread
    within 0.8 to 1.25; the mean evidence ratio is within three standard errors of
    one; and the size implied by the scatter between keys is at least half the
    reported size (capped at the cloud). Fails if the cloud is narrower than it
    reports, or a prior or Jacobian term is missing.
    """
    ref, sd, g = _graded(k)
    assert np.abs(g["vis"] - ref["p_visible"]).max() <= PDET_TOL
    assert (np.abs(g["mean"] - ref["position_mean"]) / sd).max() <= 0.25
    assert (g["spread"] / sd).min() >= 0.8 and (g["spread"] / sd).max() <= 1.25
    ratio = g["ratio"]
    assert abs(ratio.mean() - 1.0) <= 3.0 * ratio.std(ddof=1) / np.sqrt(KEYS)
    reported = min(np.median(g["ess"]), N_CLOUD)
    assert _replicate_ess(g["bins"], N_CLOUD) >= 0.5 * reported


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
def test_keys_under_the_size_floor_stay_within_the_measured_rate(k):
    """Acceptance item 1: at most `MAX_LOW[k]` of eight keys report a size under 1600.

    One record always reaches the floor. Two and three records reach it on about
    nine keys in ten, the rate measured over 200 prior-drawn worlds, so up to two of
    eight keys may fall under it. Fails if more keys fall short, which is a loss of
    effective size and not a statistical fluctuation of that rate.
    """
    _, _, g = _graded(k)
    assert int(np.sum(g["ess"] < ESS_FLOOR)) <= MAX_LOW[k]


def test_the_one_record_golden_agrees_with_plain_prior_sampling():
    """The stored scale-and-rotate golden matches a draw-from-the-prior reference.

    The refit and the golden builder share ``scale_rotate_draw``, so agreement with it
    proves nothing about the draw. Plain importance sampling from the prior through
    the forward model (``tools/refit_golden.py``, 8.4e6 draws, Kish 7.7e3) gives the
    visibility probability within four combined standard errors and every ``ln P``
    bin mass within 0.01.
    """
    golden = json.loads(GOLDEN.read_text())
    ref, indep = golden["reference"]["1"], golden["independent"]["1"]
    se = np.hypot(ref["p_visible_se"], indep["p_visible_se"])
    assert abs(ref["p_visible"] - indep["p_visible"]) <= 4.0 * se
    diff = np.abs(np.asarray(ref["bin_masses"]) - np.asarray(indep["bin_masses"]))
    assert diff.max() <= 0.01
