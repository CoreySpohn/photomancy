"""Old-versus-new parity of the chart targets, report scores and scale integrals.

Every reference here is written in this file from the formulas of the code the chart
densities replaced (the fixed-mass and Thiele-Innes targets, the element and chart
report scores); nothing is imported from the retired forward model. Kinematics are an
independent Newton solve of Kepler's equation with the Thiele-Innes rotation, and the
photometry is the Lambert phase law with the contrast ``Lambda phi / r^2`` and the
``1e-300`` floor. Constants (Beta shapes, the 24 Gauss-Legendre nodes) are typed in.

Every comparison runs in x64 at prediction rtol 1e-12 and the log-density floor
``max(1e-10, 64 eps sum(1 + |term|))`` over the terms of the compared sum. Each test
prints its largest error next to the floor it was held to.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hwoutils.constants import G, Msun2kg, Rearth2AU, two_pi  # noqa: E402
from scipy import integrate, special, stats  # noqa: E402

from photomancy.measurements import RelativeAstromData  # noqa: E402
from photomancy.orbit.few_epoch import MU_SUN  # noqa: E402
from photomancy.orbit.refit import (  # noqa: E402
    ReportData,
    physical_prior_target,
    report_log_weights,
)

EPS = np.finfo(float).eps
DIST_PC = 10.0
THRESH = 5.0
LIMIT = 100.0
BETA_A, BETA_B = 0.867, 3.03
LN_LAM_REF = np.log(0.3 * Rearth2AU**2)
LN_LAM_RANGE = (LN_LAM_REF - 1.125, LN_LAM_REF + 1.125)
# The 24 nodes and weights of the refit's scale quadrature.
GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def _floor(*terms):
    """``max(1e-10, 64 eps sum(1 + |term|))`` over every number in ``terms``."""
    values = np.concatenate([np.ravel(np.asarray(t, dtype=float)) for t in terms])
    return max(1e-10, 64.0 * EPS * float(np.sum(1.0 + np.abs(values))))


def _check(label, got, want, floors):
    """Compare finite values row by row against per-row floors; print the worst."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    np.testing.assert_array_equal(np.isneginf(got), np.isneginf(want))
    fin = np.isfinite(want)
    error = np.abs(got[fin] - want[fin])
    floors = np.asarray(floors, float)[fin]
    print(f"{label}: max_error={error.max():.3g} floor={floors.max():.3g}")
    assert np.all(error <= floors), (error.max(), floors[np.argmax(error - floors)])


# ---------------------------------------------------------------------------
# Independent kinematics and photometry
# ---------------------------------------------------------------------------


def _kepler(mean_anomaly, e):
    """Eccentric anomaly by Newton iteration."""
    m = np.mod(mean_anomaly + np.pi, two_pi) - np.pi
    big_e = m + e * np.sin(m)
    for _ in range(80):
        big_e = big_e - (big_e - e * np.sin(big_e) - m) / (1.0 - e * np.cos(big_e))
    return big_e


def _plane_xy(times, e, n, tp):
    """Plane coordinates ``(cos E - e, sqrt(1 - e^2) sin E)`` at mean motion ``n``."""
    big_e = _kepler(n * (np.asarray(times, float) - tp), e)
    return np.cos(big_e) - e, np.sqrt(1.0 - e * e) * np.sin(big_e)


def _ti_constants(a, cos_i, node, omega):
    """Thiele-Innes constants ``(A, B, F, G)`` in AU."""
    cw, sw, cn, sn = np.cos(omega), np.sin(omega), np.cos(node), np.sin(node)
    return (
        a * (cw * cn - sw * sn * cos_i),
        a * (cw * sn + sw * cn * cos_i),
        a * (-sw * cn - cw * sn * cos_i),
        a * (-sw * sn + cw * cn * cos_i),
    )


def _sky(times, a, e, cos_i, node, omega, n, tp):
    """East, north and away positions (AU) of a classical orbit."""
    x, y = _plane_xy(times, e, n, tp)
    big_a, big_b, big_f, big_g = _ti_constants(a, cos_i, node, omega)
    away = a * np.sqrt(1.0 - cos_i**2) * (np.sin(omega) * x + np.cos(omega) * y)
    return big_b * x + big_g * y, big_a * x + big_f * y, away


def _lambert_dmag(east, north, away, area_AU2):
    """``-2.5 log10(Lambda phi / r^2)`` with the contrast floored at 1e-300."""
    r = np.sqrt(east**2 + north**2 + away**2)
    beta = np.arccos(away / r)
    phase = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    return -2.5 * np.log10(np.maximum(area_AU2 * phase / r**2, 1e-300))


# ---------------------------------------------------------------------------
# Chart log target
# ---------------------------------------------------------------------------


def _ti_theta(elements, t_ref=0.0):
    """Chart point ``(e, lnP, tau, B, G, A, F)`` of the element rows."""
    period, e, cos_i, node, omega, tau, mass = np.moveaxis(np.asarray(elements), -1, 0)
    mu = MU_SUN * mass
    a = (mu * (period / two_pi) ** 2) ** (1.0 / 3.0)
    big_a, big_b, big_f, big_g = _ti_constants(a, cos_i, node, omega)
    return np.stack([e, np.log(period), tau, big_b, big_g, big_a, big_f], axis=-1)


def _ti_elements():
    """``(P, e, cos_i, W, w, tau, mass)`` rows over e, i, P, phase and mass."""
    rows = []
    for ie, e in enumerate((0.05, 0.3, 0.6, 0.85)):
        for ip, period in enumerate((80.0, 300.0, 1100.0, 4200.0)):
            for ic, cos_i in enumerate((-0.8, 0.15, 0.9)):
                k = ie + 2 * ip + 3 * ic
                rows.append(
                    (
                        period,
                        e,
                        cos_i,
                        np.mod(0.3 + 0.7 * k, two_pi),
                        np.mod(1.1 + 0.9 * k, two_pi),
                        np.mod(0.13 + 0.37 * k, 1.0),
                        (0.93, 1.0, 1.07)[k % 3],
                    )
                )
    rows.append((300.0, 0.3, 0.2, 1.0, 2.0, 0.5, 1.3))  # above the mass window
    rows.append((300.0, 0.95, 0.2, 1.0, 2.0, 0.5, 1.0))  # above e_max
    rows.append((30.0, 0.3, 0.2, 1.0, 2.0, 0.5, 1.0))  # under the period range
    return np.array(rows)


def test_extended_thiele_innes_chart_density_equals_the_old_density():
    """Fails if the new build, shared prior and chart volume change the old density.

    The old density of the extended target in its chart ``(e, lnP, tau, east, north)``
    is ``Beta(e) / (a^2 gap)`` times the closed-form period, mass, angle and cos-i
    factors, with ``a^2 = (u + gap) / 2`` and ``gap = sqrt(u^2 - 4 v^2)`` from the
    four constants. The new density is built from orbix's chart through the shared
    one-planet prior, and it equals the old one at the floor at 48 regular points
    (e, i, P, phase and mass) and is minus infinity where the old support fails.
    """
    target = physical_prior_target(1.0, DIST_PC)
    m = target.model
    elements = _ti_elements()
    theta = _ti_theta(elements)
    e, ln_p = theta[:, 0], theta[:, 1]
    consts = theta[:, 3:]
    u = np.sum(consts**2, axis=1)
    v = consts[:, 0] * consts[:, 3] - consts[:, 1] * consts[:, 2]
    gap = np.sqrt(np.maximum(u**2 - 4.0 * v**2, 1e-300))
    a2 = 0.5 * (u + gap)
    mu_implied = (two_pi / np.exp(ln_p)) ** 2 * a2**1.5
    log_e = stats.beta.logpdf(e, BETA_A, BETA_B) - np.log(
        special.betainc(BETA_A, BETA_B, m.e_max)
    )
    constants = np.array(
        [
            -np.log(m.lnP_hi - m.lnP_lo),
            -np.log(np.log(m.mu_hi / m.mu_lo) / 3.0),
            -2.0 * np.log(two_pi),
        ]
    )
    support = (
        (e > 0.0)
        & (e < m.e_max)
        & (ln_p > m.lnP_lo)
        & (ln_p < m.lnP_hi)
        & (mu_implied >= m.mu_lo)
        & (mu_implied <= m.mu_hi)
    )
    want = np.where(
        support, log_e - np.log(a2) - np.log(gap) + constants.sum(), -np.inf
    )
    floors = [
        _floor(constants, log_e[i], np.log(a2[i]), np.log(gap[i]))
        for i in range(len(e))
    ]
    assert support.sum() == 48 and not support[-3:].any()
    got = jax.vmap(target.log_prior)(jnp.asarray(theta))
    _check("extended Thiele-Innes chart log prior", got, want, floors)


# ---------------------------------------------------------------------------
# Likelihood increments
# ---------------------------------------------------------------------------

TIMES = np.array([0.0, 120.0, 340.0, 700.0, 1500.0])
SIGMA = np.array([0.004, 0.006, 0.005, 0.004, 0.006])
RHO = np.array([0.3, -0.2, 0.0, 0.5, 0.1])


def _records(ra_true, dec_true, n_valid):
    """Five records plus two padded rows; the first ``n_valid`` are valid."""
    pull = np.array([[0.8, -1.1], [0.4, 1.5], [-0.6, 0.2], [1.2, -0.7], [-0.9, 0.5]])
    ra = ra_true + SIGMA * pull[:, 0]
    dec = dec_true + 1.2 * SIGMA * pull[:, 1]
    pad = lambda x, fill: np.concatenate([x, [fill, fill]])  # noqa: E731
    return RelativeAstromData(
        times=jnp.asarray(pad(TIMES, 0.0)),
        ra=jnp.asarray(pad(ra, 9.0)),
        dec=jnp.asarray(pad(dec, 9.0)),
        ra_err=jnp.asarray(pad(SIGMA, 1.0)),
        dec_err=jnp.asarray(pad(1.2 * SIGMA, 1.0)),
        corr=jnp.asarray(pad(RHO, 0.0)),
        planet_id=jnp.zeros(7, dtype=int),
        is_valid=jnp.arange(7) < n_valid,
    ), (ra, dec)


def _old_record_loglike(ra_pred, dec_pred, measured):
    """Bivariate Gaussian log density of each record (SciPy), shape ``(5,)``."""
    ra, dec = measured
    out = []
    for i in range(5):
        sx, sy = SIGMA[i], 1.2 * SIGMA[i]
        cov = [[sx**2, RHO[i] * sx * sy], [RHO[i] * sx * sy, sy**2]]
        out.append(
            stats.multivariate_normal.logpdf(
                [ra[i] - ra_pred[i], dec[i] - dec_pred[i]], mean=[0.0, 0.0], cov=cov
            )
        )
    return np.array(out)


def _increments(target, theta, ra_pred, dec_pred, measured, truth):
    """Old and new log-likelihood increments of each record added in time order."""
    old = _old_record_loglike(ra_pred, dec_pred, measured)
    floors = []
    for k in range(1, 6):
        data, _ = _records(*truth, k)
        previous = float(target.log_likelihood(theta, _records(*truth, k - 1)[0]))
        total = float(target.log_likelihood(theta, data))
        rest, selected = target.log_likelihood_split(
            theta, data, jnp.arange(7) == k - 1
        )
        floors.append(_floor(old[:k]))
        yield (
            k,
            old[k - 1],
            total - previous,
            (float(rest), float(selected)),
            floors[-1],
        )


def _truth_orbit():
    return dict(a=1.8, e=0.35, cos_i=0.4, node=1.2, omega=2.4, M0=0.9)


def test_extended_thiele_innes_likelihood_increments_equal_the_old_ones():
    """Fails if the extended target's chart point predicts a different sky track.

    The old prediction is the Thiele-Innes form ``east = B X + G Y``, ``north = A X +
    F Y`` with the plane coordinates of mean anomaly ``2 pi ((t - t_ref) / P - tau)``.
    Records come from a nearby chart point; every increment matches at the floor.
    """
    target = physical_prior_target(1.0, DIST_PC)
    truth = _truth_orbit()
    period = 480.0
    a = (MU_SUN * (period / two_pi) ** 2) ** (1.0 / 3.0)
    base = _ti_theta(
        np.array(
            [
                [
                    period,
                    truth["e"],
                    truth["cos_i"],
                    truth["node"],
                    truth["omega"],
                    0.31,
                    1.0,
                ]
            ]
        )
    )[0]

    def predict(theta):
        x, y = _plane_xy(
            TIMES, theta[0], two_pi / np.exp(theta[1]), theta[2] * np.exp(theta[1])
        )
        return (theta[3] * x + theta[4] * y) / DIST_PC, (
            theta[5] * x + theta[6] * y
        ) / DIST_PC

    truth_xy = predict(base)
    assert a > 0.0
    worst, worst_floor = 0.0, 0.0
    for shift in (
        np.zeros(7),
        np.array([0.03, 0.01, 0.01, 0.02, -0.03, 0.02, 0.01]),
        np.array([-0.05, -0.02, -0.02, -0.04, 0.03, -0.02, 0.03]),
    ):
        theta = base + shift
        ra_pred, dec_pred = predict(theta)
        _, measured = _records(*truth_xy, 5)
        for k, want, got, (_, selected), floor in _increments(
            target, jnp.asarray(theta), ra_pred, dec_pred, measured, truth_xy
        ):
            assert abs(got - want) <= floor, (k, got - want, floor)
            assert abs(selected - want) <= floor
            worst, worst_floor = max(worst, abs(got - want)), max(worst_floor, floor)
    print(f"extended TI increments: max_error={worst:.3g} floor={worst_floor:.3g}")


# ---------------------------------------------------------------------------
# Report scores
# ---------------------------------------------------------------------------


def _curves(n_rows, n_grid=40):
    sep = np.geomspace(0.005, 2.0, n_grid)
    dmag0 = -2.5 * np.log10(5e-11 * (1.0 + (0.05 / sep) ** 6))
    return sep, dmag0, np.tile(sep, (n_rows, 1)), np.tile(dmag0, (n_rows, 1))


def _report(epochs, detected, counts):
    n = len(epochs)
    sep, dmag0, sep_grid, dmag0_grid = _curves(n)
    report = ReportData(
        epochs=jnp.asarray(epochs, float),
        sep_grid=jnp.asarray(sep_grid),
        dmag0_grid=jnp.asarray(dmag0_grid),
        limit_grid=jnp.full(sep_grid.shape, LIMIT),
        is_detected=jnp.asarray(detected, bool),
        counts=jnp.asarray(counts, float),
        is_valid=jnp.ones(n, bool),
        snr_thresh=THRESH,
    )
    return report, (sep, dmag0, np.asarray(epochs, float), np.asarray(detected, bool),
                    np.asarray(counts, float))  # fmt: skip


def _row_loglike(spec, counts_pred):
    """Report law per row: detection density above the threshold or null probability."""
    _, _, _, detected, measured = spec
    variance = LIMIT**2 + counts_pred
    floor = THRESH * LIMIT
    detection = np.where(
        measured >= floor,
        stats.norm.logpdf(measured, counts_pred, np.sqrt(variance)),
        -np.inf,
    )
    null = special.log_ndtr((floor - counts_pred) / np.sqrt(variance))
    return np.where(detected, detection, null)


def _unit_log_counts(spec, east, north, away):
    """Log expected counts at the middle of the scale range (Lambert contrast)."""
    sep, dmag0, _, _, _ = spec
    alpha = np.hypot(east, north) / DIST_PC
    dmag = _lambert_dmag(east, north, away, np.exp(LN_LAM_REF))
    log_limit = np.log(THRESH * LIMIT)
    return (
        log_limit
        + 0.4 * np.log(10.0) * np.interp(alpha, sep, dmag0)
        - 0.4 * np.log(10.0) * dmag
    )


def _scale_marginal_quadrature(spec, log_g, half, nodes, weights):
    """Null-only scale integral: log of the weighted node mean of the row law."""
    cap = np.log(THRESH * LIMIT) + 200.0 * np.log(10.0)
    ll = [
        _row_loglike(spec, np.exp(np.minimum(log_g + half * t, cap))).sum()
        for t in nodes
    ]
    return special.logsumexp(np.log(weights) + np.array(ll)) - np.log(2.0), np.array(ll)


def _scale_marginal_closed(spec, log_g, half):
    """Closed form with a detection: Gaussian in the log scale, nulls at its peak."""
    _, _, _, detected, measured = spec
    var = LIMIT**2 + np.where(detected, measured, 0.0)
    g = np.exp(np.clip(log_g, -300.0, 300.0))
    s_gg = np.sum(np.where(detected, g**2 / var, 0.0))
    s_cg = np.sum(np.where(detected, measured * g / var, 0.0))
    s_cc = np.sum(np.where(detected, measured**2 / var, 0.0))
    t_hat = np.log(s_cg / s_gg)
    sd = np.sqrt(s_gg) / s_cg
    chi2 = s_cc - s_cg**2 / s_gg
    log_norm = np.sum(np.where(detected, -0.5 * np.log(two_pi * var), 0.0))
    inside = special.ndtr((half - t_hat) / sd) - special.ndtr((-half - t_hat) / sd)
    nulls = _row_loglike(spec, np.exp(t_hat) * g)[~detected].sum()
    terms = np.array([log_norm, -0.5 * chi2, np.log(sd), np.log(inside), nulls])
    value = (
        log_norm - 0.5 * chi2 + 0.5 * np.log(two_pi) + np.log(sd) + np.log(inside)
        - np.log(2.0 * half) + nulls
    )  # fmt: skip
    return value, terms


REPORT_ORBITS = (
    # (P, e, cos_i, node, omega, tau)
    (400.0, 0.2, 0.5, -0.7, 0.9, 0.3),
    (900.0, 0.55, -0.3, 2.1, 4.0, 0.7),
    (150.0, 0.05, 0.85, 5.0, 1.5, 0.1),
    (2500.0, 0.7, 0.1, 0.4, 3.3, 0.5),
)
EPOCHS = [0.0, 200.0, 360.0, 410.0]


def _mirror_positions(period, e, cos_i, node, omega, tau):
    """East, north and the two away signs of the mirror orbits at the report epochs."""
    a = (MU_SUN * (period / two_pi) ** 2) ** (1.0 / 3.0)
    east, north, away = _sky(
        np.array(EPOCHS), a, e, cos_i, node, omega, two_pi / period, tau * period
    )
    return east, north, (away, -away)


def _report_cases():
    """Null-only, detection-plus-nulls and all-null-with-late-rows reports."""
    cases = []
    for period, e, cos_i, node, omega, tau in REPORT_ORBITS:
        east, north, (away, _) = _mirror_positions(period, e, cos_i, node, omega, tau)
        spec0 = _report(EPOCHS, [False] * 4, [0.0] * 4)[1]
        log_g = _unit_log_counts(spec0, east, north, away)
        counts = float(np.exp(log_g[0] + 0.2))
        cases.append(
            (
                (period, e, cos_i, node, omega, tau),
                _report(EPOCHS, [False] * 4, [0.0] * 4),
                _report(
                    EPOCHS,
                    [True, True, False, False],
                    [max(counts, 600.0), 700.0, 0.0, 0.0],
                ),
            )
        )
    return cases


@pytest.mark.medium(
    reason="Scores four orbits and both report kinds through compiled chart reports."
)
@pytest.mark.parametrize("scale_range", [LN_LAM_RANGE])
def test_mirror_averaged_report_scores_equal_the_old_ones(scale_range):
    """Fails if the chart report score drifts from the element-by-element old score.

    For each of four orbits the chart point stands for two mirror orbits with equal
    weight; the old score is ``log(0.5 sum_b marginal_b)`` with each marginal built
    from the independent kinematics and Lambert photometry above, by the 24-node
    Gauss-Legendre rule when no row is a detection and by the closed Gaussian form
    when one is. The new ``log_report`` agrees at the floor for both report kinds.
    """
    target = physical_prior_target(1.0, DIST_PC, 0.10, scale_range)
    half = 0.5 * (scale_range[1] - scale_range[0])
    worst, worst_floor = 0.0, 0.0
    for orbit, (null_report, null_spec), (det_report, det_spec) in _report_cases():
        theta = jnp.asarray(_ti_theta(np.array([[*orbit, 1.0]]))[0])
        east, north, away_rows = _mirror_positions(*orbit)
        for report, spec in ((null_report, null_spec), (det_report, det_spec)):
            marginals, terms = [], []
            for away in away_rows:
                log_g = _unit_log_counts(spec, east, north, away)
                if spec[3].any():
                    value, parts = _scale_marginal_closed(spec, log_g, half)
                else:
                    value, parts = _scale_marginal_quadrature(
                        spec, log_g, half, GL_X, GL_W
                    )
                marginals.append(value)
                terms.append(parts)
            want = special.logsumexp(marginals) - np.log(2.0)
            got = float(target.log_report(theta, report))
            floor = _floor(marginals, *terms)
            assert abs(got - want) <= floor, (orbit, got, want, floor)
            worst, worst_floor = max(worst, abs(got - want)), max(worst_floor, floor)
    print(f"mirror-averaged report: max_error={worst:.3g} floor={worst_floor:.3g}")


def test_element_report_weights_equal_the_old_scale_marginal():
    """Fails if the element report weights differ from the old scale-marginal integrand.

    The refit's element route builds one orbit from ``(ln P, e, cos i, W, w, tau,
    ln mass, ln Lambda)`` with ``tp = tau P``; the independent kinematics at the
    same elements give the old weights, with the same two integration branches as
    the chart route.
    """
    target = physical_prior_target(1.0, DIST_PC, 0.10, LN_LAM_RANGE)
    half = 0.5 * (LN_LAM_RANGE[1] - LN_LAM_RANGE[0])
    worst, worst_floor = 0.0, 0.0
    for orbit, (null_report, null_spec), (det_report, det_spec) in _report_cases():
        period, e, cos_i, node, omega, tau = orbit
        a = (MU_SUN * (period / two_pi) ** 2) ** (1.0 / 3.0)
        n = two_pi / period
        mass_kg = n**2 * a**3 / G
        el = jnp.array(
            [
                np.log(period),
                e,
                cos_i,
                node,
                omega,
                tau,
                np.log(mass_kg / Msun2kg),
                np.nan,
            ]
        )
        east, north, away = _sky(
            np.array(EPOCHS), a, e, cos_i, node, omega, n, tau * period
        )
        for report, spec in ((null_report, null_spec), (det_report, det_spec)):
            log_g = _unit_log_counts(spec, east, north, away)
            if spec[3].any():
                want, parts = _scale_marginal_closed(spec, log_g, half)
            else:
                want, parts = _scale_marginal_quadrature(spec, log_g, half, GL_X, GL_W)
            got = float(report_log_weights(target, el[None], report)[0])
            floor = _floor(want, parts)
            assert abs(got - want) <= floor, (orbit, got, want, floor)
            worst, worst_floor = max(worst, abs(got - want)), max(worst_floor, floor)
    print(f"element report weights: max_error={worst:.3g} floor={worst_floor:.3g}")


def test_scale_integrand_equals_the_old_report_law_row_by_row():
    """Fails if a detection, null, padded or censored row scores differently.

    The integrand of the scale integral is the report law at the expected counts: a
    detection scores the Gaussian density of its counts when they exceed the
    threshold (zero density below), a null the probability of staying under it, with
    variance ``limit^2 + N``. Eleven expected-count values spanning the scale range
    match SciPy at the floor.
    """
    spec_rows = _report(
        [0.0] * 5, [True, True, True, False, False], [800.0, 620.0, 540.0, 0.0, 0.0]
    )
    report, spec = spec_rows
    report = ReportData(
        **{**{f: getattr(report, f) for f in (
            "epochs", "sep_grid", "dmag0_grid", "limit_grid", "is_detected", "counts"
        )}, "is_valid": jnp.array([True, True, True, True, False])},
        snr_thresh=THRESH,
    )  # fmt: skip
    worst, worst_floor = 0.0, 0.0
    for scale in np.exp(np.linspace(-1.125, 1.125, 11)):
        counts_pred = scale * np.array([300.0, 450.0, 700.0, 380.0, 1.0e4])
        limit = jnp.full(5, LIMIT)
        got = float(report.log_likelihood(jnp.asarray(counts_pred), limit))
        rows = _row_loglike(spec, counts_pred)
        want = rows[:4].sum()
        floor = _floor(rows[:4])
        if np.isneginf(want):
            assert np.isneginf(got)
            continue
        assert abs(got - want) <= floor, (scale, got, want, floor)
        worst, worst_floor = max(worst, abs(got - want)), max(worst_floor, floor)
    print(f"scale integrand: max_error={worst:.3g} floor={worst_floor:.3g}")


def test_quadrature_error_of_the_24_node_scale_rule_is_bounded():
    """Bounds the error of the existing 24-node rule against a much finer integral.

    The integrand is the null-only report law over the log scale. The rule's log
    marginal differs from a 400-node Gauss-Legendre rule and from adaptive
    ``scipy.integrate.quad`` by less than QUADRATURE_BOUND (in log marginal) on
    every orbit. Parity above keeps the 24 nodes; this is the approximation error,
    separate from parity.
    """
    half = 0.5 * (LN_LAM_RANGE[1] - LN_LAM_RANGE[0])
    fine_x, fine_w = np.polynomial.legendre.leggauss(400)
    worst_fine, worst_quad = 0.0, 0.0
    for orbit, (_, spec), _ in _report_cases():
        east, north, (away, mirror) = _mirror_positions(*orbit)
        for sign in (away, mirror):
            log_g = _unit_log_counts(spec, east, north, sign)
            coarse, _ = _scale_marginal_quadrature(spec, log_g, half, GL_X, GL_W)
            fine, _ = _scale_marginal_quadrature(spec, log_g, half, fine_x, fine_w)
            cap = np.log(THRESH * LIMIT) + 200.0 * np.log(10.0)

            def density(t, log_g=log_g, cap=cap, spec=spec):
                return np.exp(
                    _row_loglike(spec, np.exp(np.minimum(log_g + half * t, cap))).sum()
                )

            exact = np.log(
                integrate.quad(density, -1.0, 1.0, epsabs=0.0, epsrel=1e-13)[0] / 2.0
            )
            worst_fine = max(worst_fine, abs(coarse - fine))
            worst_quad = max(worst_quad, abs(coarse - exact))
    print(f"24-node error: vs 400 nodes {worst_fine:.3g}, vs quad {worst_quad:.3g}")
    assert worst_fine <= QUADRATURE_BOUND and worst_quad <= QUADRATURE_BOUND


# Measured worst error is about 0.27 in the log marginal: the null-only integrand is
# a sharp step in the log scale that 24 nodes resolve coarsely.
QUADRATURE_BOUND = 0.35
