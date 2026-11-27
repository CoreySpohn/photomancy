"""The report path of the general orbit engine: a free planet radius and ``ReportTerm``.

One report fit is shared: a Thiele-Innes block on a one-planet Lambertian scene, the
planet radius free (the photometric leaf), two relative astrometry rows, and a report of
two detections and a null with two padded rows, its leaf proposal on, inside a
``SheetSum`` of the orbit row. The compiled structures are the fit density, one
completion and the vmapped run.
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hwoutils.testing import ulps
from optixstuff.spectral import Band, Quadrature, Transmission
from orbix.system import OrbitLaw, ThieleInnesBlock
from priorities import LogUniform
from scipy.special import log_ndtr
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.population import kipping_eccentricity
from skyscapes.scene import FlatStar, Planet, Scene

from photomancy import (
    CountRateTable,
    PrecomputedReportPredictor,
    ReportData,
    ReportRule,
    ReportTerm,
)
from photomancy.infer import Fit, Importance
from photomancy.measurements import AstrometryData, RelativeAstrometryTerm
from photomancy.orbit.completion import CompletedProposal, Completion
from photomancy.orbit.proposals import ScaleRotateProposal, ShapePriorProposal
from photomancy.orbit.sheets import SheetSum

LEAF = "planets[0].Rp_Rearth"
ORBITS = "dynamics.orbits"
BAND = Band(550.0, 650.0)
THRESHOLD = 5.0
LIMIT_COUNTS = 100.0
# Two astrometric epochs and a report of two detections and a null at 410 d; the
# report scores counts only, so the positions of the detections are the two records.
REC_TIMES = np.array([0.0, 200.0])
REC_RA = np.array([-0.08128829720329389, 0.09146848442849184])
REC_DEC = np.array([-0.04738471947299113, 0.059272793717665855])
REP_EPOCHS = np.array([0.0, 200.0, 410.0, 0.0, 0.0])
REP_COUNTS = np.array([1711.0280560521403, 1572.1488432460212, 0.0, 0.0, 0.0])
REP_DETECTED = np.array([True, True, False, False, False])
REP_VALID = np.array([True, True, True, False, False])


def scene():
    """One Lambertian planet (``Ag = 0.3``) at ten parsecs around one solar mass."""
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    return Scene.from_classical(
        FlatStar(1.0),
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=0.0,
        dist_pc=10.0,
        a_AU=1.0,
        e=0.1,
        i_rad=float(np.arccos(0.35)),
        W_rad=1.0,
        w_rad=2.0,
        M0_rad=1.0,
        order=(0,),
    )


def table():
    """One mode: counts per unit contrast reach the limit at 5e-11 (1 + (0.05/a)^6)."""
    sep = np.geomspace(0.005, 2.0, 128)
    contrast = 5e-11 * (1.0 + (0.05 / sep) ** 6)
    bandpass = Transmission.from_band(BAND)
    return CountRateTable(
        jnp.asarray(sep)[None],
        jnp.asarray(THRESHOLD * LIMIT_COUNTS / contrast)[None],
        jnp.array([0.03]),
        Quadrature.build((BAND,), bandpass),
        (bandpass,),
        Epoch(),
    )


def rows(
    counts=REP_COUNTS,
    detected=REP_DETECTED,
    valid=REP_VALID,
    epochs=REP_EPOCHS,
    *,
    positions=False,
):
    """Report rows; with ``positions`` the detections carry the astrometric records.

    The two detections are at the epochs of the records, so with ``positions`` their
    positions are the records with errors of 3.5 and 5 mas and a correlation of 0.3.
    """
    n = len(epochs)
    det = np.asarray(detected) & np.asarray(valid)
    ra, dec = np.zeros(n), np.zeros(n)
    err_ra, err_dec, corr = np.zeros(n), np.zeros(n), np.zeros(n)
    if positions:
        ra[:2], dec[:2] = REC_RA, REC_DEC
        err_ra[:], err_dec[:], corr[:] = 0.0035, 0.005, 0.3
    return ReportData(
        time_d=np.asarray(epochs, float),
        science_s=np.ones(n),
        limit_counts=np.full(n, LIMIT_COUNTS),
        mode=np.zeros(n, int),
        valid=np.asarray(valid),
        is_detected=det,
        counts=np.where(det, counts, 0.0),
        east_arcsec=np.where(det, ra, 0.0),
        north_arcsec=np.where(det, dec, 0.0),
        east_err_arcsec=np.where(det, err_ra, 0.0),
        north_err_arcsec=np.where(det, err_dec, 0.0),
        corr=np.where(det, corr, 0.0),
    )


def report_term(report_rows=None, *, leaf=LEAF, leaf_exponent=2.0):
    """The report term, with the leaf proposal on by default."""
    return ReportTerm(
        ReportRule(threshold=THRESHOLD),
        rows() if report_rows is None else report_rows,
        PrecomputedReportPredictor(table(), 0),
        leaf_path=leaf,
        leaf_exponent=leaf_exponent,
    )


def sheet_sum(term):
    """``term`` summed over the two sheets of the orbit row."""
    return SheetSum((term,), ORBITS, 0)


def build_fit(term=None):
    """Orbit block, free radius (log uniform over a factor 3), astrometry and report.

    The report term is held in the ``SheetSum`` of the orbit row.
    """
    law = OrbitLaw(
        period_d=LogUniform(50.0, 5000.0), eccentricity=kipping_eccentricity(0.9)
    )
    block = ThieleInnesBlock(law, LogUniform(0.9, 1.1), rows=(0,), system="dynamics")
    records = AstrometryData(
        time_d=jnp.asarray(REC_TIMES),
        east_arcsec=jnp.asarray(REC_RA),
        north_arcsec=jnp.asarray(REC_DEC),
        east_err_arcsec=jnp.full(2, 0.0035),
        north_err_arcsec=jnp.full(2, 0.0035),
        corr=jnp.zeros(2),
        planet=jnp.zeros(2, int),
        valid=jnp.ones(2, bool),
    )
    free = {
        ORBITS: block,
        LEAF: LogUniform(float(np.exp(-0.5625)), float(np.exp(0.5625))),
    }
    term = report_term() if term is None else term
    return Fit(scene(), free, (RelativeAstrometryTerm(records), sheet_sum(term)))


@pytest.fixture(scope="module")
def fit():
    """The shared report fit."""
    return build_fit()


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.medium(reason="the report fit density compiles once, 8 s cold")
def test_the_leaf_record_never_enters_the_fit_density(fit):
    """Fails if ``leaf_path`` or ``leaf_exponent`` reaches the report likelihood.

    The record is a proposal density only: the fit density at four prior draws is the
    same, bit for bit, with the record on, off, and with another exponent.
    """
    x = fit.sample_prior(jax.random.key(0), (4,))
    density = eqx.filter_jit(lambda f, x: f.log_density(x))
    on = density(fit, x)
    off = density(build_fit(report_term(leaf=None)), x)
    other = density(build_fit(report_term(leaf_exponent=1.0)), x)
    assert np.all(np.isfinite(np.asarray(on)))
    np.testing.assert_array_equal(on, off)
    np.testing.assert_array_equal(on, other)


@pytest.mark.float64_only(reason="float64 posterior coordinates and log weights")
def test_unproposed_free_entries_are_refused_by_name():
    """Fails if a free leaf no term proposes is drawn, or few_epoch takes a free leaf.

    A completion has no way to draw such a leaf, and scale-and-rotate draws only the
    orbit block; both refusals name the leaf. Fails also if a second ``SheetSum`` over
    the orbit row is accepted (a product of two sheet sums); the refusal names the
    sheet.
    """
    bare = build_fit(report_term(leaf=None))
    with pytest.raises(ValueError, match=r"planets\[0\]\.Rp_Rearth"):
        Completion.from_fit(bare)
    with pytest.raises(ValueError, match=r"planets\[0\]\.Rp_Rearth"):
        ScaleRotateProposal().sample(jax.random.key(0), build_fit(), 4)
    with pytest.raises(ValueError, match=r"dynamics\.orbits\[0\]\.sheet"):
        bare.append_term(sheet_sum(report_term(leaf=None)))


@pytest.mark.float64_only(reason="float64 posterior coordinates and log weights")
@pytest.mark.medium(reason="compiles one screened completion on the report fit, 15 s")
def test_a_pool_with_no_live_constant_weighs_zero_without_nan(fit):
    """Fails if a ``shape`` whose pool lies outside the prior gets a weight or a NaN.

    At a 4,900 d period the constants that fit the two rows imply a star mass far
    below the law's window, so no pool member is live: the weight is ``-inf`` and the
    kept draw, its density and its pool Kish size (zero) are free of NaN. A ``shape``
    at the true period beside it keeps a finite weight, so the guard cannot pass by
    killing everything.
    """
    comp = Completion.from_fit(fit)
    block, model = comp.block, comp.model
    x = jnp.array(
        [
            [np.log(2 * np.pi / 4900.0), 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
            [np.log(2 * np.pi / 372.0), 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    shape = block.chart.unconstrain(x, block.chart_support(model))[:, :3]
    z, lw, log_density, kish = eqx.filter_jit(comp.complete)(
        jax.random.key(0), shape, 32, 4
    )
    assert float(lw[0]) == -np.inf
    assert float(kish[0]) == 0.0
    assert 1.0 <= float(kish[1]) <= 32.0
    assert np.isfinite(float(lw[1]))
    assert not np.any(np.isnan(np.asarray(z)))
    assert not np.any(np.isnan(np.asarray(log_density)))


@pytest.mark.float64_only(reason="float64 posterior coordinates and log weights")
@pytest.mark.slow(
    reason="compiles the completed report under filter_vmap and a loop, 36.17 s on CI"
)
def test_the_report_run_under_vmap_equals_the_loop():
    """Fails if a batch of report fits differs from the per-world loop.

    Two worlds differ in their detection counts; the completed run (64 draws, the
    screened stage, leaves and both sheets) under ``filter_vmap`` equals the loop at
    rtol 1e-9 with a shared key.
    """
    imp = Importance(CompletedProposal(ShapePriorProposal()), 64)
    fits = [build_fit(report_term(rows(REP_COUNTS * s))) for s in (1.0, 0.9)]
    dyn, static = eqx.partition(fits, eqx.is_array)
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *dyn)
    key = jax.random.key(0)
    batched = eqx.filter_vmap(
        lambda d: imp._run(eqx.combine(d, static[0]), key).log_weights
    )(stacked)
    loop = jnp.stack([imp._run(f, key).log_weights for f in fits])
    # most prior shapes have no live constant; enough keep a weight to compare
    assert np.sum(np.isfinite(np.asarray(loop))) >= 8
    np.testing.assert_allclose(batched, loop, rtol=1e-9)


GOLDEN = Path(__file__).parent / "data" / "report_term_golden.json"
REPORTS = {
    "fixture": {},
    "censored": dict(counts=np.array([1711.03, 300.0, 0.0, 0.0, 0.0])),
    "null_only": dict(detected=np.zeros(5, bool)),
    "positions": dict(positions=True),
}


@eqx.filter_jit
def _scores(fit, both, one, z, astro=None):
    """Two-sheet value, both single sheets and the leaf record at flat draws ``z``.

    With ``astro`` the sheets are of ``one`` plus ``astro``.
    """

    def per(zi):
        x, _ = fit.from_flat(zi)
        model = fit.model_at(x)
        record = both.leaf_proposal(model)
        pair = (model, both.reflected(model))
        sheets = jnp.stack(
            [one(m) + (0.0 if astro is None else astro(m)) for m in pair]
        )
        predictions = [one.predict(m) for m in pair]
        counts = jnp.stack([p.counts for p in predictions])
        variance = jnp.stack([p.variance for p in predictions])
        return (
            both(model),
            sheets,
            record.mean,
            record.sd,
            record.log_pi,
            counts,
            variance,
        )

    return jax.vmap(per)(z)


def _golden_null_shift(rows, counts, variance):
    """Independent Gaussian CDF correction for the new detection boundary."""
    nominal = THRESHOLD * np.asarray(rows.limit_counts)
    grid = np.float32(THRESHOLD) * np.asarray(rows.limit_counts, np.float32)
    previous = np.nextafter(grid, np.float32(-np.inf))
    boundary = (grid.astype(np.float64) + previous.astype(np.float64)) / 2
    sigma = np.sqrt(np.where(variance > 0, variance, 1))
    change = log_ndtr((boundary - counts) / sigma) - log_ndtr(
        (nominal - counts) / sigma
    )
    return np.sum(np.where(rows.valid & ~rows.is_detected, change, 0.0), axis=-1)


@pytest.mark.float64_only(reason="stored float64 V&V reference values")
@pytest.mark.medium(reason="one compiled scorer for three reports at six draws, 9 s")
@pytest.mark.parametrize("name", list(REPORTS))
def test_report_term_matches_the_stored_golden(fit, name):
    """Fails if the report law, the sheet sum, its reflection or the leaf record drifts.

    Values at six stored draws (two from the prior, four from the posterior) of four
    reports: the fixture (two detections, a null, two padded rows), a detection under
    the threshold (``-inf`` everywhere), a null-only report (in these two the record
    has no peak), and the fixture with real detection positions. The stored sheets of
    the fixture agree with the previous report scorer to 4e-5 nat above -150 nat (it
    interpolates a different table). Checks, at rtol 1e-10 (compiled against the
    eager values stored): the ``SheetSum`` value, each sheet alone, the record. The
    stored sheets use the nominal threshold. An independent Gaussian CDF difference
    at the midpoint below the threshold grid point corrects only the null masses.
    For the position report, counts plus ``RelativeAstrometryTerm`` agree with the
    stored position scores. Combining the sheets allows four roundings of the
    uncancelled logaddexp/log(2) scale for the log-sum-exp and subtraction,
    including a log mean near zero. Padding is exactly inert.
    """
    golden = json.loads(GOLDEN.read_text())
    want = golden["reports"][name]
    z = jnp.asarray(golden["z"])
    kw = REPORTS[name]
    one = report_term(rows(**kw))
    both = sheet_sum(one)
    two, sheets, mean, sd, log_pi, counts, variance = _scores(fit, both, one, z)
    delta = _golden_null_shift(one.data, np.asarray(counts), np.asarray(variance))
    expected_sheets = np.asarray(want["sheets"]) + delta
    old_two = np.logaddexp(*np.asarray(want["sheets"]).T) - np.log(2)
    np.testing.assert_allclose(old_two, want["two_sheet"], rtol=1e-10)
    expected_two = np.logaddexp(*expected_sheets.T) - np.log(2)
    np.testing.assert_allclose(two, expected_two, rtol=1e-10)
    np.testing.assert_allclose(sheets, expected_sheets, rtol=1e-10)
    np.testing.assert_allclose(mean, want["mean"], rtol=1e-10)
    np.testing.assert_allclose(sd, want["sd"], rtol=1e-10)
    np.testing.assert_allclose(log_pi, want["log_pi"], rtol=1e-10)
    if name == "positions":
        astro = RelativeAstrometryTerm(one.data.astrometry())
        joint = _scores(fit, both, one, z, astro)[1]
        np.testing.assert_allclose(
            joint, np.asarray(want["with_positions"]) + delta, rtol=1e-10
        )
    log_two = np.log(2.0)
    added = np.logaddexp(sheets[:, 0], sheets[:, 1])
    summed = added - log_two
    finite = np.isfinite(summed)
    np.testing.assert_array_equal(two[~finite], summed[~finite])
    tolerance = ulps(4, np.maximum(np.abs(added[finite]), log_two))
    assert np.all(np.abs(two[finite] - summed[finite]) <= tolerance)
    counts = np.where(REP_VALID, kw.get("counts", REP_COUNTS), 999.0)
    epochs = np.where(REP_VALID, REP_EPOCHS, 12345.0)
    detected = np.where(REP_VALID, kw.get("detected", REP_DETECTED), True)
    garbage = sheet_sum(
        report_term(
            rows(counts, detected, epochs=epochs, positions=kw.get("positions", False))
        )
    )
    np.testing.assert_array_equal(_scores(fit, garbage, one, z)[0], two)
    if name == "censored":
        assert np.all(np.isneginf(np.asarray(two)))
    # a peak only where every detection is above the threshold
    peaks = np.isfinite(np.asarray(log_pi))
    assert np.any(peaks) if name in ("fixture", "positions") else not np.any(peaks)


def test_report_planet_and_sheet_sum_planet_outside_the_model_are_refused(precision):
    """Planet 1 or -1 of a one-planet model raises for the report and the sheet sum.

    Truth: the scene has one planet and one orbit row, so valid indices are
    ``[0, 1)``; without the check planet 1 scores planet 1's clamped position and
    ``SheetSum`` reflects nothing.
    Fails if: ``ReportTerm.check`` or ``SheetSum.check`` does not range the planet.
    """
    model = scene()
    good = report_term(leaf=None)
    good.check(model)
    sheet_sum(good).check(model)
    bad = eqx.tree_at(
        lambda t: t.predictor, good, PrecomputedReportPredictor(table(), 1)
    )
    with pytest.raises(ValueError, match="planet"):
        bad.check(model)
    if precision == np.float64:
        with pytest.raises(ValueError, match="planet"):
            Fit(model, {}, (bad,))
    for planet in (1, -1):
        with pytest.raises(ValueError, match="SheetSum planet"):
            SheetSum((good,), ORBITS, planet).check(model)
        if precision == np.float64:
            with pytest.raises(ValueError, match="SheetSum planet"):
                Fit(model, {}, (SheetSum((good,), ORBITS, planet),))


def test_report_planet_check_holds_under_jit_and_a_late_epoch_model_passes():
    """A static planet 2 is refused under ``jit``; a model without day 0 passes.

    Truth: the model has one planet, so planet 2 is outside ``[0, 1)`` however the
    model leaves are traced; the check compares the host index and never builds a
    tracer. A predictor model that has no positions before day 1 predicts every valid
    row of rows at later epochs, so its planet count is read at those epochs.
    Fails if: the selector is made with ``jnp`` inside ``check``, or the count is
    probed at day zero.
    """
    model = scene()
    good = report_term(leaf=None)
    bad = eqx.tree_at(
        lambda t: t.predictor, good, PrecomputedReportPredictor(table(), 2)
    )
    with pytest.raises(ValueError, match="planet"):
        jax.jit(lambda m: bad.check(m))(model)

    class Late(eqx.Module):
        inner: object

        def positions(self, *, time_d):
            if np.any(np.asarray(time_d) < 1.0):
                raise ValueError("no positions before day 1")
            return self.inner.positions(time_d=time_d)

    late_rows = rows(epochs=np.array([5.0, 200.0, 410.0, 5.0, 5.0]))
    late = eqx.tree_at(lambda t: t.data, good, late_rows)
    # ``Late`` has no ``n_planets``, so the count comes from its positions.
    late.check(Late(model))
    far = eqx.tree_at(
        lambda t: t.predictor, late, PrecomputedReportPredictor(table(), 1)
    )
    with pytest.raises(ValueError, match="planet"):
        far.check(Late(model))


def test_sheet_sum_requires_a_resolvable_path_and_counts_the_last_axis(precision):
    """An unresolvable ``orbit_path`` refuses at fit setup; planets index the last axis.

    Truth: reflection indexes the last batch axis of the quaternions, so orbits of
    shape ``(2, 1)`` hold one planet and ``(1, 2)`` hold two. A path that does not
    resolve fails at scoring, so it must fail at ``Fit``.
    Fails if: the first axis is counted or an unresolved path is skipped.
    """
    from hwoutils.model_paths import assign, resolve

    model = scene()
    good = report_term(leaf=None)
    for path in ("dynamics.orbit", "planets[99].orbits"):
        with pytest.raises(ValueError, match="does not resolve"):
            SheetSum((good,), path, 0).check(model)
        if precision == np.float64:
            with pytest.raises(ValueError, match="does not resolve"):
                Fit(model, {}, (SheetSum((good,), path, 0),))
    orbits = resolve(model, ORBITS)
    for shape, planet, accepted in (((2, 1, 4), 1, False), ((1, 2, 4), 1, True)):
        stacked = eqx.tree_at(lambda o: o.quat, orbits, jnp.zeros(shape))
        changed = assign(model, ORBITS, stacked)
        term = SheetSum((good,), ORBITS, planet)
        if accepted:
            term.check(changed)
        else:
            with pytest.raises(ValueError, match="SheetSum planet"):
                term.check(changed)
