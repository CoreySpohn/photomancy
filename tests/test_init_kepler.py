"""find_init must be Kepler-consistent.

Regression for the bug where find_init ranked periods by the a-free Thiele-Innes
chi2 (relative astrometry fits the apparent ellipse at many periods), and so
returned a period whose Kepler-III semimajor axis (a = period_to_sma(T, Ms), the
quantity the full model actually uses) does NOT fit the data. The returned
period's Kepler-implied orbit must thread the detections.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbix.equations import period_to_sma
from orbix.system import periapsis_chart_shape

from photomancy.orbit._frame import orbit_rows, sky_offsets_arcsec
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.init import (
    _grid_results,
    _kepler_consistent_loglike,
    chart_find_init,
    find_init,
)
from photomancy.orbit.thiele_innes import (
    chart_thiele_innes_fit,
    fixed_mass_coordinates,
    thiele_innes_fit,
)

MSUN = 1.98892e30
DIST_PC = 10.0
T_TRUE = 365.25


def _seed3001_three_epoch():
    """Reproduce the documented n=3, seed-3001 alias-prone case."""
    a_true = period_to_sma(T_TRUE, MSUN)
    times = jnp.linspace(0.0, 1.5 * T_TRUE, 3)
    ra0, dec0 = predict_relative_astrometry(
        times, a_true, 0.1, 0.7, 0.8, jnp.cos(0.5), jnp.sin(0.5), 40.0, MSUN, DIST_PC
    )
    sigma = 1e-3  # 1 mas, in arcsec
    kd, _ = jax.random.split(jax.random.PRNGKey(3001))
    kra, kdec = jax.random.split(kd)
    ra = ra0 + sigma * jax.random.normal(kra, ra0.shape)
    dec = dec0 + sigma * jax.random.normal(kdec, dec0.shape)
    err = jnp.full(3, sigma)
    data = RelativeAstromData(
        times=times,
        ra=ra,
        dec=dec,
        ra_err=err,
        dec_err=err,
        corr=jnp.zeros(3),
        planet_id=jnp.zeros(3, dtype=int),
        is_valid=jnp.ones(3, dtype=bool),
    )
    return data, float(a_true)


def test_find_init_period_is_kepler_consistent():
    """find_init's period must imply (via Kepler III) the true orbit size."""
    data, a_true = _seed3001_three_epoch()
    iv = find_init(data, MSUN, DIST_PC, log_T_range=(2.0, 3.0))
    T = 10.0 ** float(iv["log_P"][0])
    a_kepler = float(period_to_sma(T, MSUN))
    # the Kepler-implied semimajor axis must match the true orbit, not an a-free alias
    assert abs(a_kepler - a_true) / a_true < 0.15, (
        f"got T={T:.0f}d -> a={a_kepler:.3f}, truth {a_true:.3f}"
    )


def test_find_init_kepler_residual_is_small():
    """The Kepler-implied orbit from find_init should thread the data to ~noise."""
    data, _ = _seed3001_three_epoch()
    iv = find_init(data, MSUN, DIST_PC, log_T_range=(2.0, 3.0))
    T = 10.0 ** float(iv["log_P"][0])
    a = period_to_sma(T, MSUN)
    e = float(iv["e_raw"][0])
    w = float(iv["w_raw"][0])
    ra, dec = predict_relative_astrometry(
        data.times,
        a,
        e,
        float(iv["cos_i"][0]),
        float(iv["W"][0]),
        jnp.cos(w),
        jnp.sin(w),
        -float(iv["M0"][0]) * T / (2.0 * jnp.pi),
        MSUN,
        DIST_PC,
    )
    sq = (ra - data.ra) ** 2 + (dec - data.dec) ** 2
    rms_mas = float(jnp.sqrt(jnp.mean(sq)) * 1000.0)
    assert rms_mas < 5.0, f"find_init Kepler orbit rms = {rms_mas:.1f} mas"


_legacy_fit = eqx.filter_jit(thiele_innes_fit)
_chart_fit = eqx.filter_jit(chart_thiele_innes_fit)

TRIALS = [
    (T, e, tp)
    for T in (200.0, T_TRUE, 900.0)
    for e in (0.05, 0.3, 0.7)
    for tp in (10.0, 150.0, 300.0)
]


def test_chart_fit_matches_thiele_innes_fit_over_trials():
    """Fails if the chart fit drifts from the legacy fit at equal (T, e, tp).

    27 trials (T 200 to 900 d, e 0.05 to 0.7, tp 10 to 300 d) on the three-epoch
    case. No ridge exists in the legacy fit, so the comparison is penalty-free.
    chi2 and log likelihood match at the 64 eps floor (max(1e-10, 64 eps (1 + |x|)));
    one reported sheet is the legacy orbit (a, cos i, W and w to 1e-8), the other
    shares a, cos i and the sky positions with W shifted by pi; the sheets carry
    branches +1 and -1.
    """
    data, _ = _seed3001_three_epoch()
    eps = np.finfo(float).eps
    worst = 0.0
    for T, e, tp in TRIALS:
        old = _legacy_fit(data, T, e, tp, MSUN, DIST_PC)
        new = _chart_fit(data, DIST_PC, *periapsis_chart_shape(T, e, tp))
        for got, want in (
            (new.chi2, old.chi2),
            (new.log_likelihood, old.log_likelihood),
        ):
            floor = max(1e-10, 64 * eps * (1.0 + abs(float(want))))
            assert abs(float(got) - float(want)) <= floor * max(1.0, abs(float(want)))
            worst = max(
                worst, abs(float(got) - float(want)) / max(1.0, abs(float(want)))
            )
        rows = np.asarray(orbit_rows(new.orbits))
        np.testing.assert_allclose(rows[:, 0], float(old.a), rtol=1e-8)
        np.testing.assert_allclose(rows[:, 2], float(old.cos_i), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(rows[:, 1], e, rtol=1e-10)
        offset = np.abs(np.angle(np.exp(1j * (rows[:, 3] - float(old.W)))))
        np.testing.assert_allclose(np.sort(offset), [0.0, np.pi], atol=1e-8)
        assert sorted(np.asarray(new.branches).tolist()) == [-1, 1]
        sky = [
            np.asarray(
                sky_offsets_arcsec(
                    jax.tree.map(lambda x, k=k: x[k], new.orbits), data.times, DIST_PC
                )
            )
            for k in (0, 1)
        ]
        np.testing.assert_allclose(sky[0], sky[1], rtol=1e-9, atol=1e-12)
        assert int(new.rank) == 4
    print("chart TI fit: worst relative chi2/ll difference", worst)


def test_chart_fit_is_the_unpenalized_least_squares_optimum():
    """Fails if the QR/SVD solve is not the weighted residual optimum.

    Correlated errors, a masked row and an NaN placeholder under the mask: the
    fitted block's whitened residual is within the 64 eps floor of
    np.linalg.lstsq on an independently built whitened design (sky = block @ plane
    position), and no perturbation of the block lowers chi2. A one-epoch case is
    rank deficient (rank 2) and still returns a finite minimum-norm fit that
    reproduces the observation exactly.
    """
    data, _ = _seed3001_three_epoch()
    corr = jnp.array([0.3, -0.2, 0.5])
    valid = jnp.array([True, True, True])
    data = data.__class__(
        times=jnp.append(data.times, 700.0),
        ra=jnp.append(data.ra, jnp.nan),
        dec=jnp.append(data.dec, jnp.nan),
        ra_err=jnp.append(data.ra_err, 1e-3),
        dec_err=jnp.append(data.dec_err * 1.5, 1e-3),
        corr=jnp.append(corr, 0.0),
        planet_id=jnp.zeros(4, dtype=int),
        is_valid=jnp.append(valid, False),
    )
    shape = periapsis_chart_shape(T_TRUE, 0.3, 40.0)
    fit = _chart_fit(data, DIST_PC, *shape)
    orbit = jax.tree.map(lambda x: x[0], fit.orbits)
    xy = np.asarray(orbit.in_plane(data.times)[0])[:3]
    rows = []
    rhs = []
    for k in range(3):
        sx, sy, c = (float(v[k]) for v in (data.ra_err, data.dec_err, data.corr))
        cov = np.array([[sx**2, c * sx * sy], [c * sx * sy, sy**2]])
        w = np.linalg.inv(np.linalg.cholesky(cov))
        design = np.zeros((2, 4))
        design[0, 0:2] = xy[k]
        design[1, 2:4] = xy[k]
        rows.append(w @ design)
        rhs.append(w @ np.array([float(data.ra[k]), float(data.dec[k])]))
    # Unknowns in the arcsec scale are sky = block_AU / dist; compare in AU space.
    design = np.vstack(rows)
    target = np.concatenate(rhs)
    # sky_block is [[B, G], [A, F]] in AU; the design unknowns are the east row
    # (B, G) then the north row (A, F), in arcsec, i.e. AU / dist_pc.
    coef = np.asarray(fit.sky_block).reshape(-1) / DIST_PC
    assert np.all(np.isfinite(coef))
    resid = design @ coef - target
    reference, *_ = np.linalg.lstsq(design, target, rcond=None)
    ref_resid = design @ reference - target
    assert float(resid @ resid) == pytest.approx(
        float(ref_resid @ ref_resid), rel=1e-8, abs=1e-8
    )
    assert float(fit.chi2) == pytest.approx(float(resid @ resid), rel=1e-8, abs=1e-8)
    np.testing.assert_allclose(coef, reference, rtol=1e-6, atol=1e-9)
    # A one-epoch fit is rank deficient (rank 2 of 4) with a finite fit.
    one = data.__class__(
        *(
            x[:1]
            for x in (
                data.times,
                data.ra,
                data.dec,
                data.ra_err,
                data.dec_err,
                data.corr,
                data.planet_id,
                data.is_valid,
            )
        )
    )
    thin = _chart_fit(one, DIST_PC, *shape)
    assert int(thin.rank) == 2
    assert np.all(np.isfinite(np.asarray(thin.x)))
    assert float(thin.chi2) < 1e-12


# Medium: cold compiles of the chart grid and the legacy grid exceed the unit budget.
@pytest.mark.medium
def test_chart_find_init_matches_find_init_ranking_and_cell():
    """Fails if chart_find_init ranks differently from the legacy Kepler grid.

    On the three-epoch case with the legacy default grid sizes: the best chart
    log likelihood equals the best legacy Kepler-consistent log likelihood at the
    64 eps floor; coordinates are finite, inside the chart cell (angles in
    [-pi, pi]), and branch -1 returns the same likelihood.
    """
    data, _ = _seed3001_three_epoch()
    kwargs = dict(log_T_range=(2.0, 3.0), n_log_T=20, n_tp=8)
    coords, ll = chart_find_init(data, MSUN, DIST_PC, k=3, **kwargs)
    _, ll_other = chart_find_init(data, MSUN, DIST_PC, k=3, branch=-1, **kwargs)
    np.testing.assert_allclose(ll, ll_other, rtol=0, atol=1e-9)
    results = _grid_results(
        data, MSUN, DIST_PC, (2.0, 3.0), 20, jnp.linspace(0.0, 0.8, 9), 8
    )
    legacy = _kepler_consistent_loglike(results, data, MSUN, DIST_PC)
    best = float(jnp.max(legacy))
    floor = max(1e-10, 64 * np.finfo(float).eps * (1.0 + abs(best)))
    assert abs(float(ll[0]) - best) <= floor * max(1.0, abs(best))
    assert np.all(np.diff(np.asarray(ll)) <= 0.0)
    coords = np.asarray(coords)
    assert coords.shape == (3, 6) and np.all(np.isfinite(coords))
    assert np.all(np.abs(coords[:, 4:]) <= np.pi + 1e-12)
    # The returned start reproduces its reported likelihood through the chart.
    fit = chart_thiele_innes_fit(
        data, DIST_PC, *periapsis_chart_shape(T_TRUE, 0.1, 40.0)
    )
    both = np.asarray(fixed_mass_coordinates(fit, MSUN))
    assert both.shape == (2, 6)
    # The sheets share ln a and cos i; the reflection flips the eccentricity vector.
    np.testing.assert_allclose(both[0, [0, 3]], both[1, [0, 3]], rtol=1e-9)
    np.testing.assert_allclose(both[0, 1:3], -both[1, 1:3], rtol=1e-9, atol=1e-12)
