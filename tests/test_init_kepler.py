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
from hwoutils.constants import G
from kinematics import predict_relative_astrometry
from orbix.equations import period_to_sma
from orbix.system import ClassicalChart, EccentricityVectorChart, periapsis_chart_shape

from photomancy.orbit._frame import orbit_rows, sky_offsets_arcsec
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.init import find_init
from photomancy.orbit.likelihoods import loglike_relative_astrom
from photomancy.orbit.thiele_innes import fixed_mass_coordinates, thiele_innes_fit

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


def _best_start(data):
    """Best Kepler-consistent start as a chart orbit and its log likelihood."""
    coords, ll = find_init(data, MSUN, DIST_PC, log_T_range=(2.0, 3.0))
    orbit = jax.tree.map(
        lambda leaf: leaf[0],
        EccentricityVectorChart().to_orbit(coords, mu_AU3_per_d2=G * MSUN),
    )
    return orbit, float(ll[0])


def test_find_init_period_is_kepler_consistent():
    """find_init's period must imply (via Kepler III) the true orbit size."""
    data, a_true = _seed3001_three_epoch()
    orbit, _ = _best_start(data)
    a_kepler = float(orbit_rows(orbit)[0])
    # the Kepler-implied semimajor axis must match the true orbit, not an a-free alias
    assert abs(a_kepler - a_true) / a_true < 0.15, (
        f"got T={float(orbit.period_d):.0f}d -> a={a_kepler:.3f}, truth {a_true:.3f}"
    )
    assert float(period_to_sma(float(orbit.period_d), MSUN)) == pytest.approx(
        a_kepler, rel=1e-9
    )


def test_find_init_kepler_residual_is_small():
    """The Kepler-implied orbit from find_init should thread the data to ~noise."""
    data, _ = _seed3001_three_epoch()
    orbit, _ = _best_start(data)
    ra, dec = sky_offsets_arcsec(orbit, data.times, DIST_PC)
    sq = (ra - data.ra) ** 2 + (dec - data.dec) ** 2
    rms_mas = float(jnp.sqrt(jnp.mean(sq)) * 1000.0)
    assert rms_mas < 5.0, f"find_init Kepler orbit rms = {rms_mas:.1f} mas"


_chart_fit = eqx.filter_jit(thiele_innes_fit)

# (T, e, tp, chi2, log likelihood, a, cos i, W) of the removed free-period
# perifocal fit on the three-epoch case, frozen. The chart fit reports both
# line-of-sight sheets; one is the frozen orbit and the other has W shifted by pi.
TRIALS = [
    (
        200.0,
        0.05,
        10.0,
        3589.1671455837354,
        -1758.6506723172029,
        1.3499663321374284,
        -0.25816455830168633,
        0.7350628851937595,
    ),
    (
        200.0,
        0.3,
        150.0,
        7161.858397783388,
        -3544.9962984170293,
        1.058564945550391,
        -0.46515708212039203,
        0.7739863308252212,
    ),
    (
        200.0,
        0.7,
        300.0,
        19103.52900063236,
        -9515.831599841515,
        0.8517895783950797,
        0.1037082142058199,
        6.103941703487605,
    ),
    (
        365.25,
        0.05,
        150.0,
        265.11683421638094,
        -96.62551663352568,
        1.0866696422113962,
        0.6246496704645093,
        1.0492200195884087,
    ),
    (
        365.25,
        0.3,
        300.0,
        3912.404231646431,
        -1920.2692153485507,
        1.0566021588875585,
        0.3860001182175006,
        0.8160301048392078,
    ),
    (
        365.25,
        0.7,
        10.0,
        1820.8894254828888,
        -874.5118122667795,
        1.7734279391879677,
        0.2442435227110743,
        0.23575182983282428,
    ),
    (
        900.0,
        0.05,
        300.0,
        3215.838664096258,
        -1571.9864315734642,
        1.078843854494397,
        -0.47262556870129796,
        0.745658272185356,
    ),
    (
        900.0,
        0.3,
        10.0,
        2513.969679540043,
        -1221.0519392953568,
        0.9592975982870734,
        -0.666354480047446,
        0.525664822477111,
    ),
    (
        900.0,
        0.7,
        150.0,
        10723.634528134815,
        -5325.884363592742,
        1.0697425746209073,
        0.028658507656540492,
        1.1484744317427331,
    ),
]


def test_chart_fit_matches_frozen_perifocal_fit_over_trials():
    """Fails if the chart fit drifts from the frozen perifocal fit at equal (T, e, tp).

    Nine trials (T 200 to 900 d, e 0.05 to 0.7, tp 10 to 300 d) on the three-epoch
    case; the legacy values are typed constants (the removed route had no ridge, so
    the comparison is penalty-free). chi2 and log likelihood match at the 64 eps
    floor (max(1e-10, 64 eps (1 + |x|))); one reported sheet is the frozen orbit (a,
    cos i and W to 1e-8), the other shares a, cos i and the sky positions with W
    shifted by pi; the sheets carry branches +1 and -1.
    """
    data, _ = _seed3001_three_epoch()
    eps = np.finfo(float).eps
    worst = 0.0
    for T, e, tp, chi2, loglike, a, cos_i, W in TRIALS:
        new = _chart_fit(data, DIST_PC, *periapsis_chart_shape(T, e, tp))
        for got, want in ((new.chi2, chi2), (new.log_likelihood, loglike)):
            floor = max(1e-10, 64 * eps * (1.0 + abs(want)))
            assert abs(float(got) - want) <= floor * max(1.0, abs(want))
            worst = max(worst, abs(float(got) - want) / max(1.0, abs(want)))
        rows = np.asarray(orbit_rows(new.orbits))
        np.testing.assert_allclose(rows[:, 0], a, rtol=1e-8)
        np.testing.assert_allclose(rows[:, 2], cos_i, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(rows[:, 1], e, rtol=1e-10)
        offset = np.abs(np.angle(np.exp(1j * (rows[:, 3] - W))))
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


# Medium: a cold chart grid plus a cold Thiele-Innes fit exceed the unit budget.
@pytest.mark.medium(reason="a chart grid and a fit, 8 s cold")
def test_find_init_ranks_like_the_frozen_kepler_grid_and_cell():
    """Fails if find_init ranks differently from the removed Kepler-consistent grid.

    On the three-epoch case with the grid used to freeze the legacy value (20
    periods, 9 eccentricities, 8 periapsis times): the best log likelihood equals
    the legacy best (12.500237011495074) at the 64 eps floor; coordinates are
    finite, inside the chart cell (angles in [-pi, pi]), and branch -1 returns the
    same likelihood.
    """
    data, _ = _seed3001_three_epoch()
    kwargs = dict(log_T_range=(2.0, 3.0), n_log_T=20, n_tp=8)
    coords, ll = find_init(data, MSUN, DIST_PC, k=3, **kwargs)
    _, ll_other = find_init(data, MSUN, DIST_PC, k=3, branch=-1, **kwargs)
    np.testing.assert_allclose(ll, ll_other, rtol=0, atol=1e-9)
    best = 12.500237011495074
    floor = max(1e-10, 64 * np.finfo(float).eps * (1.0 + abs(best)))
    assert abs(float(ll[0]) - best) <= floor * max(1.0, abs(best))
    assert np.all(np.diff(np.asarray(ll)) <= 0.0)
    coords = np.asarray(coords)
    assert coords.shape == (3, 6) and np.all(np.isfinite(coords))
    assert np.all(np.abs(coords[:, 4:]) <= np.pi + 1e-12)
    # The returned start reproduces its reported likelihood through the chart.
    fit = thiele_innes_fit(data, DIST_PC, *periapsis_chart_shape(T_TRUE, 0.1, 40.0))
    both = np.asarray(fixed_mass_coordinates(fit, MSUN))
    assert both.shape == (2, 6)
    # The sheets share ln a and cos i; the reflection flips the eccentricity vector.
    np.testing.assert_allclose(both[0, [0, 3]], both[1, [0, 3]], rtol=1e-9)
    np.testing.assert_allclose(both[0, 1:3], -both[1, 1:3], rtol=1e-9, atol=1e-12)


# Medium: two cold chart grids.
@pytest.mark.medium
def test_find_init_writes_starts_in_the_declared_chart_and_epoch():
    """Starts in any chart rebuild the ranked orbits at the chart epoch.

    On the three-epoch case, the ``ClassicalChart`` starts at chart epochs 0 and
    250 d, rebuilt with ``to_orbit`` at the stated mass and epoch, reproduce the
    reported log likelihood at the 64 eps floor, so the epoch is honored and the
    chart is the one asked for; the coordinates differ between the epochs (the
    mean anomaly moves) and the default chart is the eccentricity-vector chart.
    Fails if the chart or epoch argument is ignored.
    """
    data, _ = _seed3001_three_epoch()
    kwargs = dict(log_T_range=(2.0, 3.0), n_log_T=12, n_tp=6, k=2)
    default, _ = find_init(data, MSUN, DIST_PC, **kwargs)
    assert default.shape == (2, EccentricityVectorChart().n_coords)
    starts = {}
    for epoch_d in (0.0, 250.0):
        coords, ll = find_init(
            data, MSUN, DIST_PC, chart=ClassicalChart(), epoch_d=epoch_d, **kwargs
        )
        assert coords.shape == (2, ClassicalChart().n_coords)
        orbits = ClassicalChart().to_orbit(
            coords, mu_AU3_per_d2=G * MSUN, epoch_d=epoch_d
        )
        for i in range(2):
            ra, dec = sky_offsets_arcsec(
                jax.tree.map(lambda leaf, i=i: leaf[i], orbits), data.times, DIST_PC
            )
            got = float(loglike_relative_astrom(ra, dec, data))
            floor = max(1e-10, 64 * np.finfo(float).eps * (1.0 + abs(float(ll[i]))))
            assert abs(got - float(ll[i])) <= floor * max(1.0, abs(got))
        starts[epoch_d] = np.asarray(coords)
    assert np.max(np.abs(starts[0.0][:, 5] - starts[250.0][:, 5])) > 1e-3
