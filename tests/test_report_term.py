"""Verification of the normalized counts/position report law and its surrogate."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from optixstuff.spectral import Band, Quadrature, Resolution, Transmission
from scipy.special import ndtr
from scipy.stats import norm

from photomancy import (
    AbstractReportPredictor,
    CountRateTable,
    PrecomputedReportPredictor,
    ReportData,
    ReportPrediction,
    ReportRule,
    ReportTerm,
)


class _Scene(eqx.Module):
    contrast: jax.Array
    epoch: Epoch = eqx.field(static=True, default=Epoch())

    def positions(self, time_d):
        east = jnp.stack((0.1 + 0.01 * time_d, 0.15 + 0.01 * time_d))
        return jnp.stack((east, -east))

    def ephemeris(self, time_d):
        return time_d

    def band_contrasts(self, quadrature, time_d, *, transmission=None, resolution=None):
        # Mode, planet and time dependence makes all three row indices observable.
        scale = (
            jnp.asarray([b.center_nm for b in quadrature.bands])[:, None, None] / 600.0
        )
        return self.contrast * scale * jnp.stack((1 + 0.01 * time_d, 2 + 0.01 * time_d))


class _ExtraVariance(AbstractReportPredictor):
    base: PrecomputedReportPredictor
    extra: jax.Array

    def __call__(self, model, rows):
        prediction = self.base(model, rows)
        return eqx.tree_at(
            lambda p: p.variance, prediction, prediction.variance + self.extra
        )


@pytest.fixture(scope="module")
def report_case():
    """Two modes, one padded grid bank and three shared exposure rows."""
    edges = (Band(550.0, 650.0), Band(850.0, 950.0))
    bands = tuple(Transmission.from_band(b) for b in edges)
    table = CountRateTable(
        jnp.array([[0.05, 0.25, 0.4, jnp.inf], [0.05, 0.25, jnp.inf, jnp.inf]]),
        jnp.array([[100.0, 200.0, 400.0, 0.0], [300.0, 500.0, 0.0, 0.0]]),
        jnp.array([0.03, 0.05]),
        Quadrature.build(edges, *bands),
        bands,
        Epoch(),
    )
    rows = ReportData.planned(
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([1.0, 2.0, 1.0]),
        jnp.array([2.0, 3.0, 2.0]),
        jnp.array([0, 1, 0]),
    )
    model = _Scene(jnp.array(0.05))
    predictor = PrecomputedReportPredictor(table, 1)
    term = ReportTerm(ReportRule(threshold=4.5, kappa=1.3, corr=0.4), rows, predictor)
    return model, term


def test_normalization():
    """Fails if selection is conditioned twice or the null uses the wrong tail.

    Fixed 256-node Gauss-Legendre quadrature integrates standard-normal counts
    from threshold through at least 12 sigma, leaving <2e-33 tail mass. The
    1e-9 absolute bound allows quadrature and floating-point rounding.
    """
    rule = ReportRule()
    nodes, weights = np.polynomial.legendre.leggauss(256)
    for counts, s, limit in ((0.0, 0.5, 2.0), (10.0, 4.0, 2.0), (100.0, 10.0, 2.0)):
        a = (rule.threshold * limit - counts) / s
        upper = max(a, 0.0) + 12
        z = (nodes + 1) * (upper - a) / 2 + a
        prediction = ReportPrediction(
            jnp.array(counts), jnp.array(s * s), jnp.array(0.0), jnp.array(0.0)
        )
        density = np.exp(rule.log_counts_density(counts + s * z, limit, prediction)) * s
        mass = np.dot(weights, density) * (upper - a) / 2
        null = float(jnp.exp(rule.log_null_probability(limit, prediction)))
        assert abs(null + mass - 1) < 1e-9
    for kwargs in ({"threshold": 0.0}, {"kappa": np.nan}, {"corr": 1.0}):
        with pytest.raises(ValueError):
            ReportRule(**kwargs)


def test_sampler_law(report_case):
    """Fails if counts selection or observable-dependent covariance is wrong.

    n=2000 draws resolve fraction errors of about 0.045 at four binomial
    sigma. The truncated mean uses its analytic variance / detection count.
    Whitened position means/variances use 4/sqrt(m), 4*sqrt(2/(m-1)); the
    covariance uses 4*sqrt((1+rho^2)/m). Error-law parity allows 16 x64 eps.
    """
    model, term = report_case
    # Move row zero's count mean near threshold to exercise both outcomes.
    model = eqx.tree_at(lambda m: m.contrast, model, jnp.array(0.025))
    keys = jax.random.split(jax.random.key(0), 2000)
    samples = eqx.filter_jit(jax.vmap(lambda key: term.draw(key, model)))(keys).data
    prediction = term.predict(model)
    n, s, limit = map(
        float,
        (
            prediction.counts[0],
            jnp.sqrt(prediction.variance[0]),
            term.data.limit_counts[0],
        ),
    )
    alpha = (term.rule.threshold * limit - n) / s
    p = ndtr(-alpha)
    detected = np.asarray(samples.is_detected[:, 0])
    assert abs(detected.mean() - p) < 4 * np.sqrt(p * (1 - p) / len(keys))
    counts = np.asarray(samples.counts[:, 0])[detected]
    z = (counts - n) / s
    mills = norm.pdf(alpha) / p
    variance = 1 + alpha * mills - mills**2
    assert abs(z.mean() - mills) < 4 * np.sqrt(variance / len(z))
    sigma = (
        term.rule.kappa
        * term.predictor.table.lod_arcsec[0]
        * np.sqrt(limit**2 + counts)
        / counts
    )
    np.testing.assert_allclose(
        np.asarray(samples.east_err_arcsec[:, 0])[detected],
        sigma,
        rtol=16 * np.finfo(float).eps,
    )
    np.testing.assert_array_equal(samples.east_err_arcsec, samples.north_err_arcsec)
    x = (
        np.asarray(samples.east_arcsec[:, 0])[detected] - prediction.east_arcsec[0]
    ) / sigma
    y = (
        np.asarray(samples.north_arcsec[:, 0])[detected] - prediction.north_arcsec[0]
    ) / sigma
    m = len(x)
    assert np.all(np.abs(np.mean([x, y], axis=1)) < 4 / np.sqrt(m))
    assert np.all(np.abs(np.var([x, y], axis=1, ddof=1) - 1) < 4 * np.sqrt(2 / (m - 1)))
    assert abs(float(np.mean(x * y)) - term.rule.corr) < 4 * np.sqrt(
        (1 + term.rule.corr**2) / m
    )
    for name in (
        "counts",
        "east_arcsec",
        "north_arcsec",
        "east_err_arcsec",
        "north_err_arcsec",
        "corr",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(samples, name))[:, 0][~detected], 0.0
        )
    assert np.isfinite(term.draw(keys[0], model)(model))


def test_surrogate_and_routing(report_case):
    """Fails if the surrogate changes the rule, data, units or mode/planet routing.

    Identical predictors agree exactly (0 ulp); independent rate arithmetic
    allows 8 x64 eps. Both paths use the same band-integrated contrast reader.
    """
    model, planned = report_case
    table = planned.predictor.table
    term = planned.draw(jax.random.key(0), model)
    exact = ReportTerm(
        term.rule,
        term.data,
        _ExtraVariance(term.predictor, jnp.array(0.0)),
        table.lod_arcsec[term.data.mode],
    )
    surrogate = exact.surrogate(table, 1)
    assert surrogate.data is exact.data
    assert surrogate.rule is exact.rule
    np.testing.assert_array_equal(exact(model), surrogate(model))
    np.testing.assert_array_equal(exact.log_ratio(model, surrogate), 0.0)
    prediction = surrogate.predict(model)
    east = 0.15 + 0.01 * np.asarray(term.data.time_d)
    rates = [
        np.interp(
            np.sqrt(2) * x,
            [0.05, 0.25, 0.4] if mode == 0 else [0.05, 0.25],
            [100.0, 200.0, 400.0] if mode == 0 else [300.0, 500.0],
        )
        for x, mode in zip(east, term.data.mode, strict=True)
    ]
    contrast = (
        model.contrast
        * np.array([1.0, 1.5, 1.0])
        * (2 + 0.01 * np.asarray(term.data.time_d))
    )
    np.testing.assert_allclose(
        prediction.counts,
        contrast * np.asarray(rates) * term.data.science_s,
        rtol=8 * np.finfo(float).eps,
    )
    np.testing.assert_array_equal(prediction.east_arcsec, east)
    np.testing.assert_array_equal(prediction.north_arcsec, -east)
    # Pin the actual System seam, rather than relying only on a toy model.
    from test_terms import _system

    system = _system()
    sys_table = CountRateTable(
        table.separation_arcsec,
        table.counts_per_contrast_s,
        table.lod_arcsec,
        table.quadrature,
        table.transmissions,
        system.epoch,
    )
    actual = PrecomputedReportPredictor(sys_table, 0)(system, term.data)
    with pytest.raises(ValueError, match="epoch"):
        PrecomputedReportPredictor(table, 0)(system, term.data)
    contrasts = jnp.stack(
        [
            system.band_contrasts(
                table.quadrature.band(i),
                term.data.time_d,
                resolution=Resolution(),
                transmission=b,
            )[0]
            for i, b in enumerate(table.transmissions)
        ]
    )
    pos = np.asarray(system.positions(term.data.time_d))[:, 0]
    rate = np.array(
        [
            np.interp(
                sep,
                np.asarray(table.separation_arcsec[mode]),
                np.asarray(table.counts_per_contrast_s[mode]),
                left=0.0,
                right=0.0,
            )
            for sep, mode in zip(np.hypot(*pos), term.data.mode, strict=True)
        ]
    )
    expected = contrasts[term.data.mode, 0, jnp.arange(3)] * rate * term.data.science_s
    np.testing.assert_allclose(actual.counts, expected, rtol=8 * np.finfo(float).eps)
    np.testing.assert_allclose(
        actual.east_arcsec,
        system.positions(term.data.time_d)[0, 0],
        rtol=8 * np.finfo(float).eps,
    )


def test_variance_log_ratio(report_case):
    """Fails if log ratios use the other data or silently omit extra variance.

    All rows are detections so only counts densities differ. The closed-form
    normal log-density difference is checked at 1e-10 absolute (x64 arithmetic).
    """
    model, term = report_case
    prediction = term.predict(model)
    rows = eqx.tree_at(
        lambda r: (
            r.is_detected,
            r.counts,
            r.east_arcsec,
            r.north_arcsec,
            r.east_err_arcsec,
            r.north_err_arcsec,
            r.corr,
        ),
        term.data,
        (
            jnp.ones(3, bool),
            prediction.counts + 1.0,
            prediction.east_arcsec,
            prediction.north_arcsec,
            jnp.full(3, 0.02),
            jnp.full(3, 0.03),
            jnp.full(3, 0.2),
        ),
    )
    term = eqx.tree_at(lambda t: t.data, term, rows)
    other = ReportTerm(
        term.rule,
        ReportData.zeros(3),
        _ExtraVariance(term.predictor, jnp.array(8.0)),
        jnp.full(3, 0.03),
    )
    expected = 0.5 * jnp.log((prediction.variance + 8) / prediction.variance) - 0.5 * (
        1 / prediction.variance - 1 / (prediction.variance + 8)
    )
    np.testing.assert_allclose(
        term.log_ratio(model, other), expected, rtol=0, atol=1e-10
    )
    assert np.all(np.abs(expected) > 0)


def test_masks_and_edges(report_case):
    """Fails if padding leaks, null NaNs disappear, or the rate leaks past a grid.

    Masked contributions and unchanged entries are exact (0 ulp). The finite
    false-positive density at zero expected counts follows the Gaussian law.
    NaN propagation is required even for nulls and below-threshold detections.
    """
    model, term = report_case
    invalid = ReportData.zeros(3)
    invalid = jax.tree.map(
        lambda x: (
            jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x
        ),
        invalid,
    )
    empty = eqx.tree_at(lambda t: t.data, term, invalid)
    assert empty(model) == 0.0
    np.testing.assert_array_equal(
        jax.grad(lambda c: empty(_Scene(c)))(model.contrast), 0.0
    )
    sampled = empty.draw(jax.random.key(0), model)
    for a, b in zip(
        jax.tree.leaves(empty.data), jax.tree.leaves(sampled.data), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    rows = eqx.tree_at(
        lambda r: (r.time_d, r.science_s, r.limit_counts),
        term.data,
        (jnp.full(3, 100.0), jnp.array([1.0, 0.0, 1.0]), jnp.array([2.0, 2.0, 0.0])),
    )
    outside = eqx.tree_at(lambda t: t.data, term, rows)
    np.testing.assert_array_equal(outside.predict(model).counts, 0.0)
    assert np.isfinite(outside(model))
    np.testing.assert_array_equal(
        outside.draw(jax.random.key(0), model).data.valid, [True, False, False]
    )
    prediction = outside.predict(model)
    # A detection outside the instrument's table can still be a noise false positive.
    assert np.isfinite(
        term.rule.log_counts_density(
            jnp.array([10.0, 10.0, 10.0]), rows.limit_counts, prediction
        )[0]
    )
    below = eqx.tree_at(
        lambda r: (r.is_detected, r.counts, r.east_err_arcsec, r.north_err_arcsec),
        term.data,
        (jnp.ones(3, bool), jnp.ones(3), jnp.ones(3), jnp.ones(3)),
    )
    assert jnp.isneginf(eqx.tree_at(lambda t: t.data, term, below)(model))
    for name in (
        "counts",
        "east_arcsec",
        "north_arcsec",
        "east_err_arcsec",
        "north_err_arcsec",
        "corr",
    ):
        poisoned = eqx.tree_at(
            lambda r, name=name: getattr(r, name), term.data, jnp.full(3, jnp.nan)
        )
        assert jnp.isnan(eqx.tree_at(lambda t: t.data, term, poisoned)(model))
    broken = eqx.tree_at(lambda p: p.contrast, model, jnp.array(jnp.nan))
    assert jnp.isnan(term.draw(jax.random.key(0), broken)(model))


def test_transformations_and_gradient(report_case):
    """Fails if JIT, vmap key routing or a null/detection gradient is unsafe.

    Eager/JIT allow 32 x64 eps for fusion; vmap/loop sampling agrees at the
    same floor. A central step 1e-6 in an order-one contrast multiplier has
    O(h^2) error; 1e-6 relative tests the analytic gradient away from knots.
    """
    model, term = report_case
    prediction = term.predict(model)
    rows = eqx.tree_at(
        lambda r: (
            r.is_detected,
            r.counts,
            r.east_arcsec,
            r.north_arcsec,
            r.east_err_arcsec,
            r.north_err_arcsec,
        ),
        term.data,
        (
            jnp.array([True, False, True]),
            prediction.counts + 2.0,
            prediction.east_arcsec + 0.002,
            prediction.north_arcsec - 0.001,
            jnp.full(3, 0.02),
            jnp.full(3, 0.03),
        ),
    )
    term = eqx.tree_at(lambda t: t.data, term, rows)
    eager = term(model)
    compiled = eqx.filter_jit(lambda t, m: t(m))(term, model)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True):
        np.testing.assert_allclose(a, b, rtol=32 * np.finfo(float).eps, atol=1e-14)
    keys = jax.random.split(jax.random.key(0), 3)
    batched = eqx.filter_jit(jax.vmap(lambda key: term.draw(key, model)))(keys)
    for i, key in enumerate(keys):
        one = term.draw(key, model)
        for a, b in zip(
            jax.tree.leaves(one.data), jax.tree.leaves(batched.data), strict=True
        ):
            np.testing.assert_allclose(
                a, b[i], rtol=32 * np.finfo(float).eps, atol=1e-14
            )

    def score(x):
        return term(eqx.tree_at(lambda m: m.contrast, model, x * model.contrast))

    gradient = jax.grad(score)(jnp.array(1.0))
    central = (score(1.0 + 1e-6) - score(1.0 - 1e-6)) / (2e-6)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, central, rtol=1e-6)


def test_epoch_mismatch_raises(report_case):
    """Fails if a table is paired with a model on a different time origin."""
    model, term = report_case
    other = Epoch(2460000.5)
    bad = _Scene(model.contrast, other)
    with pytest.raises(ValueError, match="epoch"):
        term(bad)
    same = _Scene(model.contrast, Epoch())
    assert float(term(same)) == float(term(model))
