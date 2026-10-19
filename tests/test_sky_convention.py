"""Sky-axis regressions anchored to published elements and independent geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg
from orbix import KeplerianOrbit
from orbix.kepler import markley_solve_trig
from sky_reference import sky_position

from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.diagnostics import orbits_from_samples
from photomancy.orbit.eig import _predict_astrom_pure, _predict_sep_dmag_pure
from photomancy.orbit.forward import (
    _reflex_proper_motion,
    predict_photometry,
    predict_pm_anomaly,
    predict_relative_astrometry,
    predict_stellar_astrometry,
)
from photomancy.orbit.init import ti_to_init
from photomancy.orbit.ofti import _scale_rotate_score
from photomancy.orbit.thiele_innes import thiele_innes_fit

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def k16():
    """Alpha Cen AB: Kervella et al. (2016), A&A 594, A107, K16 elements.

    Epochs are days from 1955.604; mass is derived from the quoted angular
    semimajor axis, parallax and period to enforce Kepler consistency.
    """
    distance = 1000 / 747.17
    a = 17.592 * distance
    period = 79.929 * 365.25
    w = np.deg2rad(232.006)
    return dict(
        T=period,
        a=a,
        e=0.5208,
        cos_i=np.cos(np.deg2rad(79.320)),
        W=np.deg2rad(205.064),
        cos_w=np.cos(w),
        sin_w=np.sin(w),
        tp=0.0,
        dist_pc=distance,
        Ms=4 * np.pi**2 * a**3 / (G * period**2),
    )


def reference(times, params):
    """Evaluate the independent oracle with a forward-model parameter dict."""
    return sky_position(times, **{k: v for k, v in params.items() if k != "Ms"})


def forward_params(params):
    """Select the relative predictor's physical arguments."""
    return {k: v for k, v in params.items() if k != "T"}


def astrometry(times, positions, ra_err=1e-4, dec_err=3e-4, corr=0.0):
    """Construct data with explicit east/north coordinates and uncertainties."""
    n = len(times)
    return RelativeAstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(positions[0]),
        dec=jnp.asarray(positions[1]),
        ra_err=jnp.broadcast_to(ra_err, (n,)),
        dec_err=jnp.broadcast_to(dec_err, (n,)),
        corr=jnp.broadcast_to(corr, (n,)),
        planet_id=jnp.zeros(n, dtype=int),
        is_valid=jnp.ones(n, dtype=bool),
    )


def fit(data, p):
    """Fit orientation at the independently known nonlinear parameters."""
    return thiele_innes_fit(data, p["T"], p["e"], p["tp"], p["Ms"], p["dist_pc"])


def test_k16_fit_recovers_orientation(k16):
    """Kervella et al. (2016), A&A 594, A107: recover K16, not its reflection.

    Green (1985), ch. 17: astrometry permits only the simultaneous
    (Omega + 180, omega + 180) node ambiguity, not i -> 180 - i.
    """
    times = np.linspace(0, k16["T"] * 1.4, 32)
    truth = reference(times, k16)
    result = fit(astrometry(times, truth), k16)
    angles = np.array(
        [
            np.rad2deg(np.arccos(result.cos_i)),
            np.rad2deg(result.W) % 360,
            np.rad2deg(np.arctan2(result.sin_w, result.cos_w)) % 360,
        ]
    )
    expected = np.array([79.320, 205.064, 232.006])
    branches = np.array([expected, expected + np.array([0, 180, 180])])
    errors = (angles - branches + 180) % 360 - 180
    assert np.min(np.max(np.abs(errors), axis=1)) < 1e-7, f"Recovered {angles} deg"
    assert float(result.chi2) < 1e-12

    # The fitted angles pass unchanged through initialization and diagnostics.
    sites = ti_to_init(result, k16["Ms"])
    samples = dict(
        T=result.T,
        e=result.e,
        cos_i=sites["cos_i"],
        W=sites["W"],
        cos_w=jnp.cos(sites["w_raw"]),
        sin_w=jnp.sin(sites["w_raw"]),
        tp=result.tp,  # Keep the epoch: initialization clips M0 off its boundary.
    )
    orbit = orbits_from_samples(samples, k16["Ms"])
    positions = orbit.position_arcsec(
        trig_solver=markley_solve_trig,
        t_jd=jnp.asarray(times),
        Ms_kg=k16["Ms"],
        dist_pc=k16["dist_pc"],
    )
    np.testing.assert_allclose(np.asarray(positions)[:, 0], truth, atol=1e-7, rtol=0)


@pytest.mark.parametrize("oracle", ["numpy", "orbix"])
def test_relative_astrometry_east_north(k16, oracle):
    """Green (1985), ch. 17; Brandt et al. (2021), Eqs. 6-9: RA is east."""
    times = jnp.asarray(np.linspace(-0.2, 1.3, 25) * k16["T"])
    if oracle == "numpy":
        expected = reference(times, k16)
    else:
        orbit = KeplerianOrbit.from_period(
            T_d=k16["T"],
            e=k16["e"],
            cos_i=k16["cos_i"],
            W_rad=k16["W"],
            cos_w=k16["cos_w"],
            sin_w=k16["sin_w"],
            tp_d=k16["tp"],
            Ms_kg=k16["Ms"],
        )
        expected = np.asarray(
            orbit.position_arcsec(
                trig_solver=markley_solve_trig,
                t_jd=times,
                Ms_kg=k16["Ms"],
                dist_pc=k16["dist_pc"],
            )
        )[:, 0]
    actual = predict_relative_astrometry(times, **forward_params(k16))
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("correlated", [False, True])
def test_ti_axis_uncertainties_and_covariance(k16, correlated):
    """Brandt et al. (2021), Eqs. 6-9: TI coefficients respect RA/Dec errors.

    Compare fitted sky positions and Gaussian residual statistics to a NumPy
    generalized least-squares oracle with unequal, epoch-dependent errors.
    """
    times = np.linspace(0, 1.2 * k16["T"], 19)
    sigma_ra = np.linspace(0.001, 0.004, len(times))
    sigma_dec = np.linspace(0.008, 0.002, len(times))
    rho = np.linspace(-0.7, 0.6, len(times)) if correlated else np.zeros(len(times))
    observed = reference(times, k16) + np.random.default_rng(82).normal(
        size=(2, len(times))
    ) * [sigma_ra, sigma_dec]
    data = astrometry(times, observed, sigma_ra, sigma_dec, rho)

    # Span the apparent orbit with independent basis predictions, then whiten
    # each (RA, Dec) observation by its own Cholesky covariance factor.
    basis_params = dict(k16, a=k16["dist_pc"], cos_i=1.0, W=0.0, cos_w=1.0, sin_w=0.0)
    Y, X = reference(times, basis_params)
    design = np.zeros((len(times), 2, 4))
    design[:, 0, :2] = np.column_stack([X, Y])
    design[:, 1, 2:] = np.column_stack([X, Y])
    cov = np.zeros((len(times), 2, 2))
    cov[:, 0, 0], cov[:, 1, 1] = sigma_ra**2, sigma_dec**2
    cov[:, 0, 1] = cov[:, 1, 0] = rho * sigma_ra * sigma_dec
    chol = np.linalg.cholesky(cov)
    white_design = np.linalg.solve(chol, design).reshape(-1, 4)
    white_data = np.linalg.solve(chol, observed.T[..., None]).reshape(-1)
    coefficients = np.linalg.lstsq(white_design, white_data, rcond=None)[0]
    expected = (design @ coefficients).T
    result = fit(data, k16)
    fitted_params = dict(
        k16,
        a=float(result.a),
        cos_i=float(result.cos_i),
        W=float(result.W),
        cos_w=float(result.cos_w),
        sin_w=float(result.sin_w),
    )
    np.testing.assert_allclose(
        reference(times, fitted_params), expected, atol=1e-8, rtol=0
    )
    chi2 = np.sum((white_data - white_design @ coefficients) ** 2)
    ll = (
        -0.5 * chi2
        - 0.5 * np.linalg.slogdet(cov)[1].sum()
        - len(times) * np.log(2 * np.pi)
    )
    np.testing.assert_allclose(result.chi2, chi2, atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.log_likelihood, ll, atol=1e-7, rtol=0)


def test_ofti_hits_independent_target():
    """Blunt et al. (2017), AJ 153, 229: scale and rotate in sky position angle.

    Green (1985), ch. 17: the node angle runs north through east.
    """
    target = np.array([[0.08], [0.03]])
    times = np.array([31.0])
    data = astrometry(times, target)
    row, ll = _scale_rotate_score(
        0.2,
        0.5,
        np.cos(1.0),
        np.sin(1.0),
        1.3,
        *target[:, 0],
        data,
        times[0],
        G * Msun2kg,
        Msun2kg,
        10.0,
        -30.0,
        30.0,
        0.99,
    )
    a, e, cos_i, W, cos_w, sin_w, tp = np.asarray(row)
    period = 2 * np.pi * np.sqrt(a**3 / (G * Msun2kg))
    actual = sky_position(
        times,
        T=period,
        a=a,
        e=e,
        cos_i=cos_i,
        W=W,
        cos_w=cos_w,
        sin_w=sin_w,
        tp=tp,
        dist_pc=10.0,
    )
    np.testing.assert_allclose(actual, target, atol=1e-12, rtol=0)
    assert np.isfinite(ll)


def test_eig_astrometry_east_north(k16):
    """Brandt et al. (2021), Eqs. 6-9: EIG uses (east, north) observables."""
    phys = dict(
        log_P=np.log10(k16["T"]),
        e_raw=k16["e"],
        cos_i=k16["cos_i"],
        W=k16["W"],
        w_raw=np.arctan2(k16["sin_w"], k16["cos_w"]),
        M0=0.0,
    )
    actual = _predict_astrom_pure(
        phys, 1234.0, lambda x: x, lambda x: x, k16["Ms"], k16["dist_pc"]
    )
    np.testing.assert_allclose(
        actual, reference([1234.0], k16)[:, 0], atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("predictor", ["eig", "forward"])
def test_standard_lambert_phase(predictor):
    """Sobolev (1975); Brown (2005), Eq. 4: alpha=30 deg gives Phi=0.88084.

    For a=1 AU, i=60 deg, Omega=omega=0 and M=90 deg, the raw
    (north, east, away) position is (0, 0.5, sqrt(3)/2) AU.
    """
    period = 2 * np.pi / np.sqrt(G * Msun2kg)
    Lambda = 1e-9
    if predictor == "eig":
        phys = dict(
            log_P=np.log10(period), e_raw=0.0, cos_i=0.5, W=0.0, w_raw=0.0, M0=np.pi / 2
        )
        sep, dmag = _predict_sep_dmag_pure(
            phys, 0.0, lambda x: x, lambda x: x, Msun2kg, 10.0, Lambda
        )
    else:
        sep, dmag = np.asarray(
            predict_photometry(
                jnp.array([period / 4]),
                1.0,
                0.0,
                0.5,
                0.0,
                1.0,
                0.0,
                0.0,
                Msun2kg,
                Lambda,
                10.0,
            )
        )[:, 0]
    alpha = np.pi / 6
    phase = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
    assert phase == pytest.approx(0.88084, abs=5e-6)
    assert float(sep) == pytest.approx(0.05, abs=1e-12)
    assert float(10 ** (-0.4 * dmag) / Lambda) == pytest.approx(phase, abs=1e-12)


@pytest.mark.parametrize("observable", ["stellar", "proper_motion", "pm_anomaly"])
def test_reflex_observables_east_north(k16, observable):
    """Green (1985), ch. 17: barycentric reflex retains the east/north axes."""
    Mp = 0.1 * k16["Ms"]
    fraction = -Mp / (k16["Ms"] + Mp)
    hip, gaia, window, n = 500.0, 9000.0, 1000.0, 25
    times = np.linspace(gaia - window / 2, gaia + window / 2, n)
    positions = fraction * reference(times, k16)
    params = dict(forward_params(k16), Mp=Mp)
    if observable == "stellar":
        actual = predict_stellar_astrometry(jnp.asarray(times), **params)
        expected = positions
    else:
        dt = times - times.mean()
        expected = positions @ dt / (dt @ dt)
        if observable == "proper_motion":
            actual = _reflex_proper_motion(gaia, window, n, **params)
        else:
            actual = predict_pm_anomaly(hip, gaia, window, n, **params)
            ends = fraction * reference([hip, gaia], k16)
            expected -= (ends[:, 1] - ends[:, 0]) / (gaia - hip)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
