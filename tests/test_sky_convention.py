"""Sky-axis regressions anchored to published elements and independent geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg, Rearth2AU
from orbix.system import FrameOrbit, phase_angle_rad
from sky_reference import position_au, sky_position

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

_FORWARD_TIMES = np.array([-120.0, 0.0, 137.0, 690.0])
_ECCENTRIC_ELEMENTS = (2.7, 0.6, 0.4, 1.1, 0.6, 0.8, -43.0)


@pytest.mark.parametrize(
    "elements, mass_ratio, relative, stellar, anomaly",
    [
        pytest.param(
            (1.3, 0.0, 1.0, 0.3, 1.0, 0.0, 17.0),
            0.001,
            [
                [
                    -0.099948653732590889,
                    0.010652383819374982,
                    0.10321978551077628,
                    0.10049197648748968,
                ],
                [
                    0.028744853748847676,
                    0.10345301696405339,
                    -0.012715183015172887,
                    -0.026783626745417014,
                ],
            ],
            [
                [
                    9.9848804927663229e-05,
                    -1.0641742077297684e-05,
                    -0.00010311666884193434,
                    -0.00010039158490258709,
                ],
                [
                    -2.8716137611236439e-05,
                    -0.00010334966729675663,
                    1.2702480534638249e-05,
                    2.6756869875541474e-05,
                ],
            ],
            [-6.8076138122183554e-08, 5.9287394533732639e-08],
            id="circular-face-on",
        ),
        pytest.param(
            _ECCENTRIC_ELEMENTS,
            0.003,
            [
                [
                    0.091789842799087640,
                    0.0062827931267316825,
                    -0.14472636179985801,
                    -0.25165442563755791,
                ],
                [
                    0.061613483320555405,
                    -0.039842104664956760,
                    -0.094401615838050415,
                    -0.013869845333618919,
                ],
            ],
            [
                [
                    -0.00027454589072508761,
                    -1.8792003370084792e-05,
                    0.00043288044406737187,
                    0.00075270516142838856,
                ],
                [
                    -0.00018428758720006602,
                    0.00011916880757215381,
                    0.00028235777419157647,
                    4.1485080758581009e-05,
                ],
            ],
            [-1.0836480091115674e-06, -5.4599438127197698e-07],
            id="eccentric-inclined",
        ),
        pytest.param(
            (4.1, 0.25, 1e-6, 2.2, -0.8, 0.6, 29.0),
            0.1,
            [
                [
                    -0.076159654905933419,
                    -0.14543872744042433,
                    -0.19610128333880375,
                    -0.075873012087053063,
                ],
                [
                    0.055436000177148295,
                    0.10586402730342588,
                    0.14274122119320648,
                    0.055228045776032686,
                ],
            ],
            [
                [
                    0.0069236049914484926,
                    0.013221702494584030,
                    0.017827389394436705,
                    0.0068975465533684605,
                ],
                [
                    -0.0050396363797407546,
                    -0.0096240024821296257,
                    -0.012976474653927862,
                    -0.0050207314341847902,
                ],
            ],
            [-5.9921181833747623e-06, 4.3615644482251724e-06],
            id="near-edge-on-finite-mass",
        ),
    ],
)
def test_forward_predictors_keep_their_values(
    elements, mass_ratio, relative, stellar, anomaly
):
    """Pin pre-change outputs generated in x64, stored with 17 significant digits.

    Fails if the relative, stellar or PM-anomaly predictor changes mass, sign,
    units, epochs or shape. Absolute floors are 1e-14 arcsec (relative),
    1e-16 arcsec (stellar) and 1e-19 arcsec/day (anomaly); rtol is 1e-11.
    """
    params = dict(
        zip(("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp"), elements, strict=True)
    )
    params.update(Ms=Msun2kg, Mp=mass_ratio * Msun2kg, dist_pc=12.5)
    times = jnp.asarray(_FORWARD_TIMES)
    actual_relative = predict_relative_astrometry(times, **params)
    actual_stellar = predict_stellar_astrometry(times, **params)
    actual_anomaly = predict_pm_anomaly(0.0, 9000.0, 1023.0, 25, **params)
    for actual, expected, atol in (
        (actual_relative, relative, 1e-14),
        (actual_stellar, stellar, 1e-16),
        (actual_anomaly, anomaly, 1e-19),
    ):
        assert np.shape(actual) == np.shape(expected)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=atol)


@pytest.mark.parametrize("cos_i", [0.4, 1.0, -1.0])
def test_relative_astrometry_element_gradients(cos_i):
    """Compare independent element derivatives to x64 central differences.

    Fails if any of a, e, cos_i, W, cos_w, sin_w or tp loses its derivative.
    At exactly face-on inclinations the existing reverse-mode cos_i gradient is NaN;
    the other six finite derivatives must still match. Steps of 1e-5 times
    max(1, abs(element)) balance truncation and roundoff (rtol 2e-6, atol 1e-9).
    """
    elements = jnp.asarray(_ECCENTRIC_ELEMENTS).at[2].set(cos_i)

    def score(x):
        ra, dec = predict_relative_astrometry(
            jnp.asarray(_FORWARD_TIMES),
            *x,
            Msun2kg,
            12.5,
            Mp=0.003 * Msun2kg,
        )
        return jnp.sum(ra + 0.37 * dec)

    indices = np.array([0, 1, 3, 4, 5, 6] if abs(cos_i) == 1 else range(7))
    steps = 1e-5 * np.maximum(1.0, np.abs(elements))
    central = []
    for i in indices:
        offset = jnp.zeros_like(elements).at[i].set(steps[i])
        central.append(
            (score(elements + offset) - score(elements - offset)) / (2 * steps[i])
        )
    actual = jax.grad(score)(elements)
    np.testing.assert_allclose(actual[indices], central, rtol=2e-6, atol=1e-9)


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
    sites = ti_to_init(result)
    samples = dict(
        T=result.T,
        e=result.e,
        cos_i=sites["cos_i"],
        W=sites["W"],
        cos_w=jnp.cos(sites["w_raw"]),
        sin_w=jnp.sin(sites["w_raw"]),
        tp=result.tp,  # Keep the epoch: initialization clips M0 off its boundary.
    )
    orbit = orbits_from_samples(samples, k16["Ms"], reference_day=0.0)
    positions, _ = orbit.at(jnp.asarray(times))
    offsets = np.asarray(positions)[0, :, :2].T / k16["dist_pc"]
    np.testing.assert_allclose(offsets, truth, atol=1e-7, rtol=0)


@pytest.mark.parametrize("oracle", ["numpy", "orbix"])
def test_relative_astrometry_east_north(k16, oracle):
    """Green (1985), ch. 17; Brandt et al. (2021), Eqs. 6-9: RA is east."""
    times = jnp.asarray(np.linspace(-0.2, 1.3, 25) * k16["T"])
    if oracle == "numpy":
        expected = reference(times, k16)
    else:
        orbit = FrameOrbit.from_classical(
            a_AU=k16["a"],
            e=k16["e"],
            i_rad=jnp.arccos(k16["cos_i"]),
            W_rad=k16["W"],
            w_rad=jnp.arctan2(k16["sin_w"], k16["cos_w"]),
            M0_rad=0.0,
            epoch_d=k16["tp"],
            mu_AU3_per_d2=G * k16["Ms"],
        )
        position, _ = orbit.at(times)
        expected = np.asarray(position)[..., :2].T / k16["dist_pc"]
    actual = predict_relative_astrometry(times, **forward_params(k16))
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("cos_i", [0.7, -0.7])
def test_raw_rows_and_forward_match_classical_ti(k16, cos_i):
    """Classical TI formulas give east, north, away raw rows for either sense."""
    p = dict(k16, cos_i=cos_i)
    times = jnp.asarray(np.linspace(-0.2, 1.3, 25) * p["T"])
    expected = position_au(
        times, **{k: v for k, v in p.items() if k not in ("Ms", "dist_pc")}
    )
    orbit = FrameOrbit.from_classical(
        a_AU=p["a"],
        e=p["e"],
        i_rad=jnp.arccos(p["cos_i"]),
        W_rad=p["W"],
        w_rad=jnp.arctan2(p["sin_w"], p["cos_w"]),
        M0_rad=0.0,
        epoch_d=p["tp"],
        mu_AU3_per_d2=G * p["Ms"],
    )
    rows, _ = orbit.at(times)
    beta = phase_angle_rad(rows)
    radius = jnp.linalg.norm(rows, axis=-1)
    np.testing.assert_allclose(rows.T, expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.cos(beta), expected[2] / radius, atol=1e-12, rtol=0)
    actual = predict_relative_astrometry(times, **forward_params(p))
    np.testing.assert_allclose(actual, expected[:2] / p["dist_pc"], atol=1e-9, rtol=0)


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
@pytest.mark.parametrize("case", ["eccentric", "near-full", "near-new"])
def test_standard_lambert_phase(predictor, case):
    """Fails if photometry changes phase convention, radius units or inverse square.

    Exact-algebra basis: independent Cartesian positions, beta=atan2(hypot(x,y),z),
    Phi=(sin(beta)+(pi-beta)*cos(beta))/pi, C=Ag*(Rp_AU/r)^2*Phi.
    Float64 tolerance is 2e-15 absolute in contrast divided by area: near new
    phase loses relative precision through cancellation in the Lambert numerator.
    Six cases cover two prediction paths and three distinct orbit/phase regimes.
    """
    a, e, cos_i, W, omega, M0 = 1.7, 0.4, 0.4, 0.6, 0.3, 1.1
    if case != "eccentric":
        a, e, cos_i, W, omega = 1.0, 0.0, 1e-3, 0.0, 0.0
        M0 = np.pi / 2 if case == "near-full" else 3 * np.pi / 2
    period = 2 * np.pi * np.sqrt(a**3 / (G * Msun2kg))
    Ag, Rp_AU = 0.3, 1.4 * Rearth2AU
    area = Ag * Rp_AU**2
    if predictor == "eig":
        phys = dict(
            log_P=np.log10(period), e_raw=e, cos_i=cos_i, W=W, w_raw=omega, M0=M0
        )
        sep, dmag = _predict_sep_dmag_pure(
            phys, 0.0, lambda x: x, lambda x: x, Msun2kg, 10.0, area
        )
    else:
        sep, dmag = np.asarray(
            predict_photometry(
                jnp.array([0.0]),
                a,
                e,
                cos_i,
                W,
                np.cos(omega),
                np.sin(omega),
                -M0 * period / (2 * np.pi),
                Msun2kg,
                area,
                10.0,
            )
        )[:, 0]
    position = position_au(
        [0.0],
        T=period,
        a=a,
        e=e,
        cos_i=cos_i,
        W=W,
        cos_w=np.cos(omega),
        sin_w=np.sin(omega),
        tp=-M0 * period / (2 * np.pi),
    )[:, 0]
    radius = np.linalg.norm(position)
    beta = np.arctan2(np.hypot(*position[:2]), position[2])
    phase = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    expected = Ag * (Rp_AU / radius) ** 2 * phase
    np.testing.assert_allclose(sep, np.hypot(*position[:2]) / 10.0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        10 ** (-0.4 * dmag) / area, expected / area, atol=2e-15, rtol=2e-13
    )


@pytest.mark.parametrize("observable", ["stellar", "proper_motion", "pm_anomaly"])
def test_reflex_observables_east_north(k16, observable):
    """Green (1985), ch. 17: barycentric reflex retains the east/north axes."""
    # The K16 mass is the Kepler total; split it so Ms + Mp keeps the period.
    Ms = k16["Ms"] / 1.1
    Mp = 0.1 * Ms
    fraction = -Mp / (Ms + Mp)
    hip, gaia, window, n = 500.0, 9000.0, 1000.0, 25
    times = np.linspace(gaia - window / 2, gaia + window / 2, n)
    positions = fraction * reference(times, k16)
    params = dict(forward_params(k16), Ms=Ms, Mp=Mp)
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
