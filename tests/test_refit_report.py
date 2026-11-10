"""Report term, photometric scale and first tier of the windowed physical-prior refit.

Each test names the property it protects. The unit tests are cheap (no refit); the
medium tests run small refits; the slow tests grade the refit against the golden
reference that carries the report term (``tools/refit_golden.py report``).
"""

import functools
import json
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hwoutils.constants import G, Msun2kg, Rearth2AU, two_pi  # noqa: E402
from scipy import stats  # noqa: E402

from photomancy.orbit.data import NullData, RelativeAstromData  # noqa: E402
from photomancy.orbit.forward import (  # noqa: E402
    predict_photometry,
    predict_relative_astrometry,
)
from photomancy.orbit.likelihoods import loglike_null  # noqa: E402
from photomancy.orbit.refit import (  # noqa: E402
    ECC_ALPHA,
    ECC_BETA,
    ReportData,
    _element_terms,
    _report_xy,
    _ti_report_terms,
    collapsed_refit,
    draw_report_scale,
    first_tier,
    importance_refit,
    loglike_report,
    physical_prior_target,
    prior_cloud,
    report_log_weights,
    ti_to_elements,
)
from photomancy.orbit.updater import UpdaterStatus  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "refit_report_golden.json"
DIST_PC = 10.0
THRESH = 5.0
LAM = 0.3 * Rearth2AU**2
LN_LAM = np.log(LAM)
SCALE_RANGE = (LN_LAM - 1.125, LN_LAM + 1.125)
CHART = jnp.array([0.2, np.log(400.0), 0.3, 0.5, -0.7, 0.9, 0.4])


def _curves(n_rows, limit=100.0, n_grid=40):
    """Sensitivity grids: counts reach the limit at contrast 5e-11 (1 + (0.05/a)^6)."""
    sep = np.geomspace(0.005, 2.0, n_grid)
    dmag0 = -2.5 * np.log10(5e-11 * (1.0 + (0.05 / sep) ** 6))
    tile = lambda x: jnp.asarray(np.tile(x, (n_rows, 1)))  # noqa: E731
    return tile(sep), tile(dmag0), tile(np.full(n_grid, limit))


def _report(epochs, detected, counts, valid=None, limit=100.0, n_grid=40):
    n = len(epochs)
    sep, dmag0, lim = _curves(n, limit, n_grid)
    return ReportData(
        epochs=jnp.asarray(epochs, float),
        sep_grid=sep,
        dmag0_grid=dmag0,
        limit_grid=lim,
        is_detected=jnp.asarray(detected, bool),
        counts=jnp.asarray(counts, float),
        is_valid=jnp.ones(n, bool) if valid is None else jnp.asarray(valid, bool),
        snr_thresh=THRESH,
    )


def _target(scale_range=SCALE_RANGE):
    return physical_prior_target(1.0, DIST_PC, 0.10, scale_range)


def test_report_likelihood_tends_to_the_null_form_as_the_limit_grows():
    """With the planet-free noise far above the planet counts, the null form.

    The two differ by the planet's shot noise, which the null form leaves out, so
    they agree to 1e-6 only once ``limit`` is huge against the expected counts.
    """
    sep, dmag0, lim = _curves(3, limit=1e9)
    rep = ReportData(
        epochs=jnp.zeros(3),
        sep_grid=sep,
        dmag0_grid=dmag0,
        limit_grid=lim,
        is_detected=jnp.zeros(3, bool),
        counts=jnp.zeros(3),
        is_valid=jnp.ones(3, bool),
        snr_thresh=THRESH,
    )
    null = NullData(
        epochs=jnp.zeros(3),
        sep_grid=sep,
        dmag0_grid=dmag0,
        is_valid=jnp.ones(3, bool),
        snr_thresh=THRESH,
    )
    alpha = jnp.array([0.08, 0.15, 0.3])
    # planets at and under the limit: a flux ratio of 1, 0.63 and 0.4 per row
    dmag0 = jnp.asarray(
        [np.interp(a, np.asarray(sep[0]), np.asarray(dmag0[0])) for a in alpha]
    )
    d_mag = dmag0 + jnp.array([0.0, 0.5, 1.0])
    assert float(loglike_report(alpha, d_mag, rep)) == pytest.approx(
        float(loglike_null(alpha, d_mag, null)), abs=1e-6
    )


def test_report_likelihood_carries_the_planet_shot_noise():
    """A null and a detection match the reporting law with ``s^2 = limit^2 + N``.

    Fails if the planet's shot noise is dropped from the variance: ``N`` is
    chosen equal to ``limit^2`` so the variance doubles, and the result differs
    from the shot-noise-free value by far more than the tolerance.
    """
    limit = 100.0
    rep = _report([0.0, 1.0], [False, True], [0.0, 11000.0], limit=limit)
    alpha = jnp.array([0.1, 0.1])
    # expected counts 1e4 at row one (a null that it nearly rules out) and row two
    dmag0 = np.interp(0.1, np.asarray(rep.sep_grid[0]), np.asarray(rep.dmag0_grid[0]))
    d_mag = dmag0 - 2.5 * np.log10(1e4 / (THRESH * limit))
    n_pred = THRESH * limit * 10 ** (-0.4 * (d_mag - dmag0))
    s = np.sqrt(limit**2 + n_pred)
    want = stats.norm.logcdf((THRESH * limit - n_pred) / s) + stats.norm.logpdf(
        11000.0, n_pred, s
    )
    got = float(loglike_report(alpha, jnp.full(2, d_mag), rep))
    assert got == pytest.approx(want, abs=1e-9)
    no_shot = stats.norm.logcdf((THRESH * limit - n_pred) / limit) + stats.norm.logpdf(
        11000.0, n_pred, limit
    )
    assert abs(want - no_shot) > 0.5


def test_report_detection_below_the_threshold_has_no_density_and_padding_is_free():
    """A detection under ``snr_thresh * limit`` is impossible; padded rows add zero."""
    rep = _report([0.0, 1.0], [True, False], [100.0, 0.0])
    alpha, d_mag = jnp.array([0.1, 0.1]), jnp.array([24.0, 24.0])
    assert float(loglike_report(alpha, d_mag, rep)) == -np.inf
    padded = _report([0.0, 1.0], [True, False], [100.0, 0.0], valid=[False, True])
    one = _report([1.0], [False], [0.0])
    assert float(loglike_report(alpha, d_mag, padded)) == pytest.approx(
        float(loglike_report(alpha[1:], d_mag[1:], one)), abs=1e-15
    )


@pytest.mark.medium
def test_chart_report_equals_the_element_report_for_both_mirror_orbits():
    """The chart route and the element route give one report for each mirror orbit.

    The chart route shares one Kepler solve across draws and takes the
    line-of-sight offset from the Thiele-Innes constants; the element route goes
    through the forward model. Row ``b`` of the chart terms is the orbit of
    ``ti_to_elements(theta, b)``, and the two mirror orbits give different
    reports (they differ in phase angle), so the sign cannot be absorbed.
    """
    target = _target()
    rep = _report(
        [0.0, 200.0, 360.0, 410.0], [True, True, False, False], [900.0, 700.0, 0.0, 0.0]
    )
    for theta in (
        CHART,
        jnp.array([0.4, np.log(300.0), 0.7, -0.8, 0.3, 0.6, 0.9]),
    ):
        chart = _ti_report_terms(target, theta, _report_xy(target, theta, rep), rep)
        for b in (0, 1):
            el = ti_to_elements(theta, b)
            np.testing.assert_allclose(
                chart[b], _element_terms(target, el, rep), rtol=1e-9, atol=1e-9
            )
        assert float(jnp.max(jnp.abs(chart[0] - chart[1]))) > 1.0


@pytest.mark.medium
def test_chart_mirror_rows_follow_the_element_branches_for_any_constants():
    """Row ``b`` is the orbit of ``ti_to_elements(theta, b)`` for any constants."""
    target = _target(SCALE_RANGE)
    rep = _report(
        [0.0, 200.0, 360.0, 410.0], [True, True, False, False], [900.0, 700.0, 0.0, 0.0]
    )
    rng = np.random.default_rng(0)
    for _ in range(12):
        theta = jnp.asarray(
            [
                rng.uniform(0, 0.6),
                np.log(rng.uniform(150, 600)),
                rng.uniform(),
                *rng.normal(0, 0.5, 4),
            ]
        )
        chart = _ti_report_terms(target, theta, _report_xy(target, theta, rep), rep)
        for b in (0, 1):
            el = ti_to_elements(theta, b)
            np.testing.assert_allclose(
                chart[b], _element_terms(target, el, rep), rtol=1e-9, atol=1e-6
            )


def _counts_at_scale(el, ln_lam, epoch):
    """Expected counts of ``el`` at scale ``ln_lam`` and ``epoch`` (D2 sensitivity)."""
    period, mass = jnp.exp(el[0]), jnp.exp(el[6]) * Msun2kg
    a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
    alpha, d_mag = predict_photometry(
        jnp.array([epoch]), a, el[1], el[2], el[3], jnp.cos(el[4]), jnp.sin(el[4]),
        period * el[5], mass, jnp.exp(ln_lam), DIST_PC,
    )  # fmt: skip
    sep, dmag0, _ = _curves(1)
    dmag0_at = np.interp(float(alpha[0]), np.asarray(sep[0]), np.asarray(dmag0[0]))
    return THRESH * 100.0 * 10.0 ** (-0.4 * (float(d_mag[0]) - dmag0_at))


def _grid_marginal(target, el, rep, n=4001):
    """Report likelihood of ``el`` integrated over the scale on a fine grid (NumPy)."""
    period, mass = jnp.exp(el[0]), jnp.exp(el[6]) * Msun2kg
    a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
    ln_lam = np.linspace(float(target.ln_lam_lo), float(target.ln_lam_hi), n)
    alpha, d_mag = jax.vmap(
        lambda lam: predict_photometry(
            rep.epochs,
            a,
            el[1],
            el[2],
            el[3],
            jnp.cos(el[4]),
            jnp.sin(el[4]),
            period * el[5],
            mass,
            lam,
            DIST_PC,
        )
    )(jnp.exp(ln_lam))
    ll = np.asarray(jax.vmap(lambda a, d: loglike_report(a, d, rep))(alpha, d_mag))
    return ln_lam, ll


@pytest.mark.medium
def test_element_report_weights_match_an_independent_scale_integral():
    """`report_log_weights` is the log of the report law averaged over the scale.

    The forward model gives the report log likelihood at 4001 scales uniform in
    the logarithm; their log mean is the reference. A detection plus nulls (the
    closed form, which fixes the detection variance at the measured counts and
    takes the nulls at the peak) agrees within 0.05; a report with no detection
    (quadrature) agrees to 1e-3.
    """
    target = _target()
    el = ti_to_elements(CHART, 1)
    counts = _counts_at_scale(el, LN_LAM + 0.2, 0.0)
    for rep, tol in (
        (_report([0.0, 200.0], [True, False], [counts, 0.0]), 0.05),
        (_report([200.0, 300.0], [False, False], [0.0, 0.0]), 1e-3),
    ):
        _, ll = _grid_marginal(target, el, rep)
        want = float(np.logaddexp.reduce(ll) - np.log(ll.size))
        got = float(report_log_weights(target, el[None], rep)[0])
        assert got == pytest.approx(want, abs=tol)


def test_prior_cloud_follows_the_declared_prior():
    """The cloud has the declared eccentricity, period, mass window and scale prior.

    Fails if the eccentricity is not the truncated Beta(0.867, 3.03), the mass
    leaves its window, or the scale is not uniform over its range.
    """
    target = physical_prior_target(0.8, DIST_PC, 0.10, (-3.0, -1.0))
    n = 40_000
    el = np.asarray(prior_cloud(target, jax.random.key(0), n))
    e_max = float(target.model.e_max)
    xs = np.linspace(0.0, e_max, 20001)
    pdf = stats.beta.pdf(xs, ECC_ALPHA, ECC_BETA)
    pdf[0] = 0.0
    mean_e = np.trapezoid(xs * pdf, xs) / np.trapezoid(pdf, xs)
    assert el[:, 1].max() < e_max
    assert abs(el[:, 1].mean() - mean_e) < 4 * el[:, 1].std() / np.sqrt(n)
    lo, hi = np.log(0.72), np.log(0.88)
    assert el[:, 6].min() >= lo and el[:, 6].max() <= hi
    ln_p = el[:, 0]
    m = target.model
    assert abs(ln_p.mean() - 0.5 * (m.lnP_lo + m.lnP_hi)) < 4 * ln_p.std() / np.sqrt(n)
    assert el[:, 7].min() >= -3.0 and el[:, 7].max() <= -1.0
    assert stats.kstest(el[:, 7], stats.uniform(-3.0, 2.0).cdf).pvalue > 0.01


@pytest.mark.medium
def test_first_tier_of_an_empty_report_is_the_prior():
    """No valid row leaves every weight equal: the cloud is the prior, full size."""
    target = _target()
    cloud = prior_cloud(target, jax.random.key(0), 2048)
    empty = ReportData.zeros(4, 16, THRESH)
    out = first_tier(target, cloud, empty, jax.random.key(1), n_particles=512)
    assert float(out.ess) == pytest.approx(2048.0)
    assert int(out.status) == int(UpdaterStatus.OK)
    assert float(out.log_evidence) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("off", [0.0, 0.43])
def test_scale_draw_recovers_a_planet_off_the_old_nodes(off):
    """A detection at the true counts puts the drawn scale on the truth, on or off 0.25.

    The planet's scale is ``off`` away from the middle of the range, where a
    9-node grid 0.25 apart has a node when ``off`` is zero and none when it is
    0.43. The counts are the noiseless expectation, so the Gaussian peak is the
    truth and the 6000 draws have mean within four standard errors of it, with
    the spread of ``1 / SNR``. Fails if the scale is snapped to a grid.
    """
    target = _target()
    truth = LN_LAM + off
    el = ti_to_elements(CHART, 1)
    counts = _counts_at_scale(el, truth, 0.0)
    rep = _report([0.0, 200.0], [True, False], [counts, 0.0])
    many = jnp.tile(el, (6000, 1))
    drawn = np.asarray(draw_report_scale(target, many, rep, jax.random.key(0)))[:, 7]
    snr = counts / np.sqrt(100.0**2 + counts)
    assert abs(drawn.mean() - truth) < 4.0 * drawn.std() / np.sqrt(6000)
    assert 0.5 / snr < drawn.std() < 1.5 / snr
    assert len(np.unique(np.round(drawn, 6))) > 1000


@pytest.mark.medium
def test_scale_draw_follows_the_scale_posterior():
    """Drawn scales match the posterior on a fine grid, with and without a detection.

    The reference is the report likelihood at 4001 scales (uniform prior): the
    mean of 6000 draws agrees within four standard errors plus 0.01 (the closed
    form's own error). The no-detection case is drawn on a fine grid of cells
    and agrees to four standard errors alone. Fails if the report is ignored in
    the draw or the truncation to the range is lost.
    """
    target = _target()
    el = ti_to_elements(CHART, 1)
    counts = _counts_at_scale(el, LN_LAM + 0.2, 0.0)
    for rep, slack in (
        (_report([0.0, 200.0], [True, False], [counts, 0.0]), 0.01),
        (_report([0.0, 200.0], [False, False], [0.0, 0.0]), 0.0),
    ):
        ln_lam, ll = _grid_marginal(target, el, rep)
        post = np.exp(ll - ll.max())
        post /= post.sum()
        mean = float(post @ ln_lam)
        sd = float(np.sqrt(post @ (ln_lam - mean) ** 2))
        many = jnp.tile(el, (6000, 1))
        drawn = np.asarray(draw_report_scale(target, many, rep, jax.random.key(1)))[
            :, 7
        ]
        assert abs(drawn.mean() - mean) < 4.0 * sd / np.sqrt(6000) + slack
        assert drawn.min() >= SCALE_RANGE[0] and drawn.max() <= SCALE_RANGE[1]


def _null_probability(target, elements, rep):
    """Null probability at each epoch of ``rep``, per element, at its own scale."""
    del target

    def one(el):
        period = jnp.exp(el[0])
        mass = jnp.exp(el[6]) * Msun2kg
        a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
        alpha, d_mag = predict_photometry(
            rep.epochs, a, el[1], el[2], el[3], jnp.cos(el[4]), jnp.sin(el[4]),
            period * el[5], mass, jnp.exp(el[7]), DIST_PC,
        )  # fmt: skip

        def row(i):
            single = jax.tree.map(lambda x: x[i][None], rep)
            return jnp.exp(loglike_report(alpha[i][None], d_mag[i][None], single))

        return jax.vmap(row)(jnp.arange(rep.epochs.shape[0]))

    return jax.lax.map(one, elements, batch_size=2**12)


@pytest.mark.medium
def test_first_tier_predicts_the_realized_detection_rate():
    """Mean predicted detection probability matches the realized fraction.

    Worlds are drawn from the declared prior (not selected on detectability) and
    observed at three epochs under the reporting law; the worlds whose three
    reports are all nulls share one history, so the first tier gives them one
    predicted probability of a detection at a fourth epoch. It is within three
    binomial standard errors of the fraction that realize a detection. Fails if
    the null likelihood omits the shot noise or the cloud is not the prior.
    """
    target = _target()
    epochs = np.array([0.0, 120.0, 240.0, 360.0])
    world = _report(epochs, [False] * 4, np.zeros(4))
    truth = prior_cloud(target, jax.random.key(1), 1500)
    p_null = np.asarray(_null_probability(target, truth, world))
    draws = np.asarray(jax.random.uniform(jax.random.key(3), p_null.shape))
    detected = draws >= p_null
    kept = ~detected[:, :3].any(axis=1)
    assert kept.sum() >= 200
    cloud = prior_cloud(target, jax.random.key(2), 2**14)
    history = _report(epochs[:3], [False] * 3, np.zeros(3))
    post = first_tier(target, cloud, history, jax.random.key(4), n_particles=2**14)
    predicted = float(
        np.mean(1.0 - np.asarray(_null_probability(target, post.elements, world))[:, 3])
    )
    realized = detected[kept, 3].mean()
    se = np.sqrt(predicted * (1.0 - predicted) / kept.sum())
    assert abs(predicted - realized) < 3.0 * se


# -- refits with the report term -----------------------------------------------------


def _fixture(k):
    """Records, report and target of the stored fixture at ``k`` records."""
    golden = json.loads(GOLDEN.read_text())
    raw, rep = golden["records"][str(k)], golden["report"][str(k)]
    fx = golden["fixture"]
    records = RelativeAstromData(
        **{
            n: jnp.asarray(raw[n])
            for n in ("times", "ra", "dec", "ra_err", "dec_err", "corr")
        },
        planet_id=jnp.zeros(len(raw["times"]), int),
        is_valid=jnp.asarray(raw["is_valid"], bool),
    )
    report = ReportData(
        **{
            n: jnp.asarray(rep[n])
            for n in (
                "epochs",
                "sep_grid",
                "dmag0_grid",
                "limit_grid",
                "is_detected",
                "counts",
                "is_valid",
            )
        },
        snr_thresh=fx["snr_thresh"],
    )
    target = physical_prior_target(
        fx["mass_msun"],
        fx["dist_pc"],
        fx["mass_half_width"],
        ln_lam_range=fx["ln_lam_range"],
    )
    return golden, records, report, target


@pytest.mark.medium
@pytest.mark.parametrize("off", [0.0, 0.43])
def test_a_detection_pins_the_photometric_scale_on_or_off_the_old_nodes(off):
    """Detections at the true scale put the refit's scale near it, on or off a node.

    The stored counts are those of a planet at the middle of the scale range;
    multiplying them by ``exp(off)`` moves the planet ``off`` away (0.43 sits
    between the 0.25-spaced nodes of the earlier scheme). The mean scale of the
    refit cloud is within 0.3 of the truth and moves with it. Fails if the report
    term is not weighted into the cloud, or the scale is snapped to a grid.
    """
    golden, records, report, target = _fixture(1)
    scaled = jnp.where(report.is_detected, report.counts * np.exp(off), report.counts)
    report = eqx.tree_at(lambda r: r.counts, report, scaled)
    cloud = collapsed_refit(
        target,
        records,
        jax.random.key(0),
        n_particles=2048,
        n_round=2**15,
        report=report,
    )
    assert int(cloud.status) in (int(UpdaterStatus.OK), int(UpdaterStatus.LOW_ESS))
    ln_lam = np.asarray(cloud.elements[:, 7])
    mid = 0.5 * sum(golden["fixture"]["ln_lam_range"])
    assert abs(ln_lam.mean() - (mid + off)) < 0.3
    assert len(np.unique(np.round(ln_lam, 6))) > 500


@pytest.mark.medium
def test_the_refit_with_a_report_runs_under_vmap():
    """``importance_refit`` with a report vmaps over keys and returns finite clouds.

    The updater maps the refit over targets; a report with its static threshold
    must trace. Fails if the report path mixes Python control flow with traced
    values.
    """
    _, records, report, target = _fixture(1)
    keys = jax.random.split(jax.random.key(0), 2)
    fn = jax.vmap(
        lambda key: importance_refit(
            target,
            records,
            report,
            key,
            n_particles=1024,
            n_round=2**14,
            max_rounds=3,
            n_constants=2,
        )
    )
    cloud = fn(keys)
    assert cloud.elements.shape == (2, 1024, 8)
    assert bool(jnp.all(jnp.isfinite(cloud.elements)))
    assert bool(jnp.all(jnp.isfinite(cloud.log_evidence)))


# -- acceptance against the golden reference -----------------------------------------

KEYS = 8
N_CLOUD = 4096
PDET_TOL = 0.05
FUSED = {
    1: dict(n_round=2**17),
    2: dict(n_round=2**16, n_last=2**18),
    3: dict(n_round=2**16, n_last=2**18),
}


def _positions(elements, future_d):
    def one(el):
        period, mass = jnp.exp(el[0]), jnp.exp(el[6]) * Msun2kg
        a = (G * mass / (two_pi / period) ** 2) ** (1.0 / 3.0)
        ra, dec = predict_relative_astrometry(
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
    """Eight keys of the refit at ``k`` records, with the report, and the golden."""
    golden, records, report, target = _fixture(k)
    ref, fx = golden["reference"][str(k)], golden["fixture"]
    future = _report([fx["future_d"]], [False], [0.0], n_grid=128)
    out = {key: [] for key in ("ess", "pdet", "mean", "spread", "ratio", "status")}
    for i in range(KEYS):
        cloud = importance_refit(
            target,
            records,
            report,
            jax.random.key(i),
            n_particles=N_CLOUD,
            **FUSED[k],
        )
        xy = _positions(cloud.elements, fx["future_d"])
        null = np.asarray(_null_probability(target, cloud.elements, future))[:, 0]
        out["status"].append(int(cloud.status))
        out["ess"].append(float(cloud.ess))
        out["pdet"].append(float(np.mean(1.0 - null)))
        out["mean"].append(xy.mean(axis=0))
        out["spread"].append(xy.std(axis=0))
        out["ratio"].append(np.exp(float(cloud.log_evidence) - ref["log_evidence"]))
    sd = np.sqrt(np.diag(np.asarray(ref["position_cov"])))
    return ref, sd, {key: np.asarray(v) for key, v in out.items()}


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
def test_refit_with_the_report_matches_the_golden_reference(k):
    """Eight keys with the report on agree with the exact reference.

    The detection probability at a future epoch is within 0.05 of the golden value
    plus three of its standard errors; the mean predicted position is within a
    quarter of a predictive standard deviation, with the golden's own sampling
    error added, and its spread within 0.8 to 1.25; the mean evidence ratio is
    within three standard errors of one, counting the golden's relative error
    (about one over the root of its Kish size). Fails if the report term is
    omitted, loses the planet shot noise, or the scale is not redrawn.
    """
    ref, sd, g = _graded(k)
    rel = 1.0 / np.sqrt(ref["kish"])
    assert (
        np.abs(g["pdet"] - ref["p_detect"]).max() <= PDET_TOL + 3.0 * ref["p_detect_se"]
    )
    shift = np.abs(g["mean"] - ref["position_mean"]) / sd
    assert shift.max() <= 0.25 + 3.0 * rel
    ratio = g["spread"] / sd
    assert ratio.min() >= 0.8 - 3.0 * rel and ratio.max() <= 1.25 + 3.0 * rel
    ev = g["ratio"]
    se = np.hypot(ev.std(ddof=1) / np.sqrt(KEYS), rel)
    assert abs(ev.mean() - 1.0) <= 3.0 * se


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
def test_keys_with_the_report_reach_a_usable_size(k):
    """At most two of eight keys report a size under 400 or a failed status.

    The updater needs a Kish size of at least 400 to trust a refit; the rate of
    keys under it must stay at the measured one in ten or better. Fails if the
    report term collapses the weights onto a handful of proposals.
    """
    _, _, g = _graded(k)
    low = (g["ess"] < 400.0) | (g["status"] != int(UpdaterStatus.OK))
    assert int(low.sum()) <= 2
