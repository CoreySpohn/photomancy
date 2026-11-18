"""Likelihood composition and relative astrometry across the system interface."""

from dataclasses import replace

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from hwoutils.spectral import Band, Quadrature, Transmission
from kinematics import (
    predict_pm_anomaly,
    predict_relative_astrometry,
    predict_rv,
    predict_stellar_astrometry,
)
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene.planet import Planet
from skyscapes.scene.star import FlatStar
from skyscapes.scene.system import System

from photomancy import (
    AbstractTerm,
    PhotometryTerm,
    ProperMotionAnomalyTerm,
    RadialVelocityTerm,
    RelativeAstrometryTerm,
    StellarAstrometryTerm,
    TermLogDensity,
)
from photomancy.exposure import ExposureLogDensity, ExposureModel, ExposureTerm
from photomancy.likelihoods import gaussian_loglike
from photomancy.orbit.data import (
    PhotometryData,
    PMAnomalyData,
    RelativeAstromData,
    RVData,
    StellarAstromData,
)
from photomancy.orbit.likelihoods import (
    loglike_pm_anomaly,
    loglike_relative_astrom,
    loglike_rv_marginalized,
    loglike_stellar_astrom,
)
from photomancy.orbit.units import RV_AU_PER_DAY_TO_M_PER_S
from photomancy.system_terms import _path_proper_motion

# a [AU], e, node [rad], inclination [rad], periapsis [rad], M0 [rad], t0 [day].
_ELEMENTS = np.array([1.7, 0.31, 1.1, 0.8, 0.6, 0.4, 24.0])
_TIMES = np.array([31.0, 120.0, 290.0, 470.0])


_MASS = 317.8
_EPOCH = Epoch(2451545.0)


def _system(elements=_ELEMENTS, mass=_MASS, *, epoch=_EPOCH):
    """Build a finite-mass planet with explicit orbital elements and stellar context."""
    a, e, node, inc, peri, mean, epoch_d = elements
    planet = Planet(
        Rp_Rearth=jnp.ones(1),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    return System.from_classical(
        FlatStar(1.0),
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=mass * Mearth2kg / Msun2kg,
        dist_pc=10.0,
        a_AU=a,
        e=e,
        W_rad=node,
        i_rad=inc,
        w_rad=peri,
        M0_rad=mean,
        epoch_d=epoch_d,
        epoch=epoch,
        order=(0,),
    )


def _data(max_n=None):
    """Nonzero residuals and unequal correlated errors exercise every element."""
    return RelativeAstromData.from_sep_pa(
        times=_TIMES,
        sep=[0.13, 0.11, 0.09, 0.15],
        sep_err=[0.004] * 4,
        pa_deg=[45.0, 120.0, 210.0, 310.0],
        pa_err_deg=[1.0] * 4,
        max_n=max_n,
    )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("pa_deg, axis", [(90.0, 0), (0.0, 1)])
def test_sign_anchor(pa_deg, axis):
    """Truth source: PA is measured east of north; node sets periastron's PA.

    Tolerance basis: 1e-14 absolute for the vanishing trigonometric component;
    only the positive sign is asserted on the nonzero axis, independent of scale.
    Fails if: either conversion or system-term prediction flips or swaps axes.
    """
    data = RelativeAstromData.from_sep_pa(
        times=[0.0], sep=[0.1], sep_err=[0.001], pa_deg=[pa_deg], pa_err_deg=[0.1]
    )
    observed = np.array([data.ra[0], data.dec[0]])
    system = _system([1.0, 0.2, np.deg2rad(pa_deg), 0.7, 0.0, 0.0, 0.0])
    _, prediction = RelativeAstrometryTerm(data)(system)
    for position in (observed, np.asarray(prediction)[:, 0]):
        assert position[axis] > 0
        assert abs(position[1 - axis]) < 1e-14


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.parametrize("max_n", [None, 3])
def test_sep_pa_covariance(max_n):
    """Truth source: 200,000 independent Gaussian polar draws transformed by NumPy.

    Tolerance basis: standard-deviation sampling error is 1/sqrt(2*N)=0.00158;
    1% relative is over six standard errors. Correlation error is at most
    1/sqrt(N)=0.00224; 0.014 absolute exceeds six standard errors. Relative
    radial error 0.001 and angular error 0.002 rad keep curvature negligible.
    Fails if: propagated errors, covariance sign, angle units or padding differ.
    """
    sep, sep_err, pa, pa_err = 0.4, 0.0004, 0.6, 0.002
    data = RelativeAstromData.from_sep_pa(
        times=[25.0],
        sep=[sep],
        sep_err=[sep_err],
        pa_deg=[np.rad2deg(pa)],
        pa_err_deg=[np.rad2deg(pa_err)],
        planet_id=[2],
        max_n=max_n,
    )
    draws = np.random.default_rng(427).normal(
        loc=[sep, pa], scale=[sep_err, pa_err], size=(200_000, 2)
    )
    east = draws[:, 0] * np.sin(draws[:, 1])
    north = draws[:, 0] * np.cos(draws[:, 1])
    np.testing.assert_allclose(
        [data.ra_err[0], data.dec_err[0]],
        [east.std(ddof=1), north.std(ddof=1)],
        rtol=0.01,
    )
    assert float(data.corr[0]) == pytest.approx(
        np.corrcoef(east, north)[0, 1], abs=0.014
    )
    np.testing.assert_array_equal(data.is_valid, [True] + [False] * ((max_n or 1) - 1))
    assert int(data.planet_id[0]) == 2


@vv.case("scene-fits", "code-verification", basis="equivalence")
@vv.seam("skyscapes", "photomancy")
def test_system_element_parity():
    """Truth source: element-based relative astrometry scored by the same likelihood.

    Tolerance basis: rtol 1e-9 in x64 allows floating-point differences between
    Markley and differentiable Kepler solvers and the angular-unit conversions.
    Fails if: system and element paths disagree in epoch, mass, scale or axes.
    """
    data = _data()
    a, e, node, inc, peri, mean, epoch = _ELEMENTS
    motion = np.sqrt(G * (Msun2kg + _MASS * Mearth2kg) / a**3)
    expected_positions = predict_relative_astrometry(
        data.times,
        a=a,
        e=e,
        cos_i=np.cos(inc),
        W=node,
        cos_w=np.cos(peri),
        sin_w=np.sin(peri),
        tp=epoch - mean / motion,
        Ms=Msun2kg,
        dist_pc=10.0,
        Mp=_MASS * Mearth2kg,
    )
    expected = loglike_relative_astrom(*expected_positions, data)
    actual, prediction = RelativeAstrometryTerm(data)(_system())
    difference = float(actual - expected)
    print(
        f"parity: loglike difference={difference:.17g}; expected={float(expected):.17g}"
    )
    position_difference = np.max(np.abs(prediction - jnp.stack(expected_positions)))
    print(f"parity: max position difference={position_difference:.17g} arcsec")
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
    np.testing.assert_allclose(prediction, expected_positions, rtol=1e-9, atol=1e-14)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("padding", [False, True])
def test_element_gradients_and_padding(padding):
    """Truth source: central differences in all seven stored orbital elements.

    Tolerance basis: steps of 1e-5 times max(1, abs(element)) balance x64
    roundoff and O(h^2) truncation; rtol 2e-6, atol 1e-6 allow cancellation.
    Padded and unpadded values/gradients agree to 1e-12 relative (same sums).
    Fails if: differentiation, masked rows or padded predictions produce NaNs
    or change the score; all-invalid data must have exactly zero gradient.
    """
    data = _data(7 if padding else None)
    if padding:
        # Invalid payloads must be sanitized before propagation and division.
        data = eqx.tree_at(
            lambda d: (d.times, d.ra, d.dec, d.ra_err, d.dec_err, d.corr, d.planet_id),
            data,
            tuple(
                jnp.where(data.is_valid, x, fill)
                for x, fill in zip(
                    (
                        data.times,
                        data.ra,
                        data.dec,
                        data.ra_err,
                        data.dec_err,
                        data.corr,
                        data.planet_id,
                    ),
                    (jnp.nan, jnp.nan, jnp.nan, 0.0, 0.0, jnp.nan, -999),
                    strict=True,
                )
            ),
        )
    term = RelativeAstrometryTerm(data)
    score = jax.jit(lambda elements: term(_system(elements))[0])
    elements = jnp.asarray(_ELEMENTS)
    actual = jax.jit(jax.grad(score))(elements)
    steps = 1e-5 * np.maximum(1.0, np.abs(_ELEMENTS))
    offsets = np.diag(steps)
    central = (
        jax.vmap(score)(elements + offsets) - jax.vmap(score)(elements - offsets)
    ) / (2 * steps)
    np.testing.assert_allclose(actual, central, rtol=2e-6, atol=1e-6)
    value, prediction = term(_system())
    assert np.all(np.isfinite(prediction))
    if padding:

        def unpadded(x):
            return RelativeAstrometryTerm(_data())(_system(x))[0]

        np.testing.assert_allclose(value, unpadded(elements), rtol=1e-12)
        np.testing.assert_allclose(actual, jax.grad(unpadded)(elements), rtol=1e-12)
        np.testing.assert_array_equal(prediction[:, 4:], 0.0)
        empty = RelativeAstrometryTerm(
            eqx.tree_at(lambda d: d.is_valid, data, jnp.zeros(7, dtype=bool))
        )

        def empty_score(x):
            return empty(_system(x))[0]

        assert float(empty_score(elements)) == 0.0
        np.testing.assert_array_equal(jax.grad(empty_score)(elements), 0.0)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_planet_and_epoch_routing():
    """Truth source: each planet has a distinct affine time-dependent position.

    Tolerance basis: exact equality for integer arithmetic represented in x64.
    Fails if: the planet or time index is reused, broadcast or interchanged.
    """

    class LinearSystem(eqx.Module):
        def positions(self, time_d):
            east = jnp.array([10.0, 30.0])[:, None] + time_d
            return jnp.stack((east, -2 * east))

    data = eqx.tree_at(lambda d: d.planet_id, _data(), jnp.array([1, 0, 1, 0]))
    _, prediction = RelativeAstrometryTerm(data)(LinearSystem())
    east = _TIMES + np.array([30.0, 10.0, 30.0, 10.0])
    np.testing.assert_array_equal(prediction, [east, -2 * east])


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("with_prior", [False, True])
def test_mixed_sum_and_restriction(with_prior):
    """Truth source: separately evaluated image/astrometry terms and one prior.

    Tolerance basis: rtol 1e-12 for the same scalar addition in x64; predictions
    and object identities are exact. Fails if: mixed routing, auxiliary order,
    prior counting, restriction, subclass behavior or empty checks regress.
    """
    model = ExposureModel(jnp.array(2.0), (jnp.array(3.0),), _system())
    image = ExposureTerm(
        lambda instrument, exposure, scene: instrument * exposure + scene.dist_pc,
        gaussian_loglike(jnp.array(15.0), 1.0),
    )
    astrometry = RelativeAstrometryTerm(_data(), select=lambda m: m.scene)
    prior = (lambda m: -(m.instrument**2)) if with_prior else None
    ld = TermLogDensity((image, astrometry), prior)
    parts = [term(model) for term in ld.terms]
    prior_value = prior(model) if with_prior else 0.0
    value, predictions = ld.value_and_aux(model)
    np.testing.assert_allclose(
        value, sum(p[0] for p in parts) + prior_value, rtol=1e-12
    )
    assert float(ld(model)) == float(value)
    for i, (part, prediction) in enumerate(parts):
        view = ld.restrict((i,))
        assert view.terms[0] is ld.terms[i]
        assert view.prior is prior
        np.testing.assert_allclose(view(model), part + prior_value, rtol=1e-12)
        np.testing.assert_array_equal(predictions[i], prediction)
        np.testing.assert_array_equal(view.value_and_aux(model)[1][0], prediction)
    reversed_aux = ld.restrict((1, 0)).value_and_aux(model)[1]
    np.testing.assert_array_equal(reversed_aux[0], predictions[1])
    np.testing.assert_array_equal(reversed_aux[1], predictions[0])
    assert isinstance(image, AbstractTerm)
    assert isinstance(astrometry, AbstractTerm)
    assert isinstance(ExposureLogDensity((image,)).restrict((0,)), ExposureLogDensity)
    with pytest.raises(TypeError):
        AbstractTerm()
    for factory in (lambda: TermLogDensity(()), lambda: ld.restrict(())):
        with pytest.raises(ValueError, match="at least one term"):
            factory()
    for indices in ((-1,), (2,)):
        with pytest.raises(IndexError):
            ld.restrict(indices)


@vv.case("scene-fits", "code-verification", basis="equivalence")
@vv.seam("skyscapes", "photomancy")
@pytest.mark.parametrize("kind", ["stellar", "rv", "pma"])
def test_reflex_system_element_parity(kind):
    """Truth source: existing element-array predictors with the same finite mass.

    Tolerance basis: x64 rtol 1e-12, likelihood atol 1e-12; prediction absolute
    floors are 1e-18 arcsec, 1e-12 m/s, and 1e-20 arcsec/day. These allow only
    floating-point solver and unit-conversion differences, not model changes.
    Fails if: the system seam changes mass, sign, units, epoch or window model.
    """
    term = _new_term(kind)
    data = term.data
    args = _element_args()
    if kind == "stellar":
        prediction = jnp.stack(predict_stellar_astrometry(data.times, **args))
        expected = loglike_stellar_astrom(*prediction, data)
        atol = 1e-18
    elif kind == "rv":
        a, e, _, inc, peri, mean, epoch = _ELEMENTS
        motion = np.sqrt(G * (Msun2kg + _MASS * Mearth2kg) / a**3)
        prediction = (
            predict_rv(
                data.times,
                2 * np.pi / motion,
                Msun2kg,
                _MASS * Mearth2kg * np.sin(inc),
                e,
                np.cos(peri),
                np.sin(peri),
                epoch - mean / motion,
                Mp=_MASS * Mearth2kg,
            )
            * RV_AU_PER_DAY_TO_M_PER_S
        )
        expected = loglike_rv_marginalized(
            data.rv,
            prediction,
            data.rv_err,
            data.inst_ids,
            data.n_inst,
            term.jitters,
            data.is_valid,
        )
        atol = 1e-12
    else:
        prediction = predict_pm_anomaly(
            data.t_hip,
            data.t_gaia,
            data.gaia_window,
            data.n_epochs,
            **args,
        )
        expected = loglike_pm_anomaly(prediction, data)
        atol = 1e-20
    actual, result = term(_system())
    print(
        f"{kind} parity: likelihood difference={float(actual - expected):.17g}; "
        f"max prediction difference={float(jnp.max(jnp.abs(result - prediction))):.17g}"
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result, prediction, rtol=1e-12, atol=atol)


def _element_args():
    """Map stored elements to the element-array forward parameterization."""
    a, e, node, inc, peri, mean, epoch = _ELEMENTS
    motion = np.sqrt(G * (Msun2kg + _MASS * Mearth2kg) / a**3)
    return dict(
        a=a,
        e=e,
        cos_i=np.cos(inc),
        W=node,
        cos_w=np.cos(peri),
        sin_w=np.sin(peri),
        tp=epoch - mean / motion,
        Ms=Msun2kg,
        Mp=_MASS * Mearth2kg,
        dist_pc=10.0,
    )


@vv.case("scene-fits", "code-verification", basis="equivalence")
@pytest.mark.parametrize("n_epochs", [2, 25])
def test_pm_anomaly_shared_ephemeris(n_epochs, monkeypatch):
    """Fails if sharing a solve changes the PM anomaly or solves more than once.

    Truth is three independent scene calls: window, Hipparcos and Gaia.
    The 1e-12 relative tolerance permits only x64 rounding for both the slope
    and likelihood. Two samples and the standard window cover slice endpoints.
    """
    term = ProperMotionAnomalyTerm(replace(_new_term("pma").data, n_epochs=n_epochs))
    system = _system()
    calls = []
    original = System.ephemeris

    def record(self, time_d, **kwargs):
        calls.append(time_d)
        return original(self, time_d, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(System, "ephemeris", record)
        actual, prediction = term(system)
    assert len(calls) == 1
    data = term.data
    ts = jnp.linspace(
        data.t_gaia - 0.5 * data.gaia_window,
        data.t_gaia + 0.5 * data.gaia_window,
        n_epochs,
    )
    mu_gaia = _path_proper_motion(ts, *system.star_reflex(time_d=ts))
    p_hip = system.star_reflex(time_d=jnp.atleast_1d(data.t_hip))[:, 0]
    p_gaia = system.star_reflex(time_d=jnp.atleast_1d(data.t_gaia))[:, 0]
    expected_prediction = mu_gaia - (p_gaia - p_hip) / (data.t_gaia - data.t_hip)
    expected = loglike_pm_anomaly(expected_prediction, data)
    np.testing.assert_allclose(prediction, expected_prediction, rtol=1e-12, atol=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def _new_term(kind, max_n=4):
    """Small observations with residuals and nontrivial covariance/instruments."""
    if kind == "stellar":
        return StellarAstrometryTerm(
            StellarAstromData.pad(
                times=_TIMES,
                ra=[1e-4, -2e-4, 3e-4, 2e-4],
                dec=[-1e-4, 1e-4, 2e-4, -2e-4],
                ra_err=[5e-5] * 4,
                dec_err=[8e-5] * 4,
                corr=[0.2, -0.3, 0.1, 0.0],
                max_n=max_n,
            )
        )
    if kind == "rv":
        return RadialVelocityTerm(
            RVData.pad(
                times=_TIMES,
                rv=[2.0, -4.0, 8.0, 1.0],
                rv_err=[1.5] * 4,
                inst_ids=[0, 1, 0, 1],
                n_inst=3,
                max_n=max_n,
            ),
            jitters=jnp.array([0.5, 0.8, 1.0]),
        )
    if kind == "pma":
        return ProperMotionAnomalyTerm(
            PMAnomalyData(
                t_hip=jnp.array(0.0),
                t_gaia=jnp.array(9000.0),
                gaia_window=jnp.array(1023.0),
                pm_anomaly=jnp.array([2e-7, -3e-7]),
                pm_anomaly_cov=jnp.array([[4e-14, 1e-14], [1e-14, 9e-14]]),
                is_valid=jnp.array(True),
            )
        )
    return PhotometryTerm(
        PhotometryData.from_dmag(
            times=_TIMES,
            dmag=[25.0, 25.2, 24.5, 26.0],
            dmag_err=[0.2] * 4,
            band_id=[0] * 4,
            is_upper_limit=[False, True, False, True],
            max_n=max_n,
        ),
        Quadrature.build((Band(550.0, 650.0),)),
        (Transmission.from_band(Band(550.0, 650.0)),),
    )


@eqx.filter_jit
def _new_term_value_grad(term, parameters):
    """Compile each observable once, reusing data shapes for validity changes."""

    def score(x):
        return term(_system(x[:7], x[7]))

    return jax.value_and_grad(score, has_aux=True)(parameters)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize("kind", ["stellar", "rv", "pma", "photometry"])
def test_new_term_gradients_and_padding(kind):
    """Truth source: central differences in all seven elements and planet mass.

    Tolerance basis: x64 central steps 1e-5*max(1, abs(parameter)), rtol 2e-6,
    atol 1e-6 cover O(h^2) truncation/cancellation. Padding agrees to 1e-12
    relative and absolute; an all-invalid score and gradient are exactly zero.
    Each observable needs its own compiled program; only the unpadded
    comparison changes shape. Benign, poisoned and all-invalid data reuse it.
    Fails if: element/mass gradients differ, padding affects a score, or NaNs
    leak through invalid epochs, uncertainties, indices or PM covariance.
    """
    term = _new_term(kind, max_n=7)
    parameters = jnp.asarray(np.r_[_ELEMENTS, _MASS])
    (value, prediction), gradient = _new_term_value_grad(term, parameters)
    steps = 1e-5 * np.maximum(1.0, np.abs(parameters))
    central = []
    for offset, step in zip(np.diag(steps), steps, strict=True):
        plus = _new_term_value_grad(term, parameters + offset)[0][0]
        minus = _new_term_value_grad(term, parameters - offset)[0][0]
        central.append((plus - minus) / (2 * step))
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient, central, rtol=2e-6, atol=1e-6)
    unpadded = _new_term(kind)
    (base, _), base_gradient = _new_term_value_grad(unpadded, parameters)
    np.testing.assert_allclose(value, base, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, base_gradient, rtol=1e-12, atol=1e-12)

    def poison(data):
        def field(x):
            if x.dtype == jnp.bool_:
                return x
            fill = -999 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.nan
            return jnp.where(data.is_valid, x, fill)

        return jax.tree.map(field, data)

    poisoned = eqx.tree_at(lambda t: t.data, term, poison(term.data))
    (poison_value, poison_prediction), poison_grad = _new_term_value_grad(
        poisoned, parameters
    )
    np.testing.assert_allclose(poison_value, value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(poison_grad, gradient, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(poison_prediction))
    if kind != "pma":
        np.testing.assert_array_equal(prediction[..., 4:], 0.0)
    empty_data = eqx.tree_at(
        lambda d: d.is_valid, term.data, jnp.zeros_like(term.data.is_valid)
    )
    empty = eqx.tree_at(lambda t: t.data, term, poison(empty_data))
    (empty_value, empty_prediction), empty_grad = _new_term_value_grad(
        empty, parameters
    )
    assert float(empty_value) == 0.0
    np.testing.assert_array_equal(empty_prediction, 0.0)
    np.testing.assert_array_equal(empty_grad, 0.0)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_contrast_detection_and_limits():
    """Truth source: Gaussian density in contrast at the system band prediction.

    Tolerance basis: x64 rtol 1e-13 for scalar arithmetic. A +12 sigma limit
    has log CDF within 1e-30 of zero; a -40 sigma limit is finite below -800.
    Fails if: detections use magnitudes or limits use the wrong tail/unstable CDF.
    """
    system = _system()
    band = Transmission.from_band(Band(550.0, 650.0))
    times = jnp.asarray(_TIMES[:3])
    rule = Quadrature.build((Band(550.0, 650.0),), system, band)
    model = system.band_contrasts(rule, times, bandpass=band)[0, 0]
    err = model / 100.0
    observed = model + jnp.array([0.7, 12.0, -40.0]) * err
    data = PhotometryData(
        times,
        observed,
        err,
        jnp.zeros(3, dtype=int),
        jnp.zeros(3, dtype=int),
        jnp.array([False, True, True]),
        jnp.ones(3, dtype=bool),
    )
    for i in range(3):
        row = jax.tree.map(lambda x, i=i: x[i : i + 1], data)
        value, prediction = PhotometryTerm(row, rule, (band,))(system)
        np.testing.assert_allclose(prediction, model[i : i + 1], rtol=1e-13)
        if i == 0:
            expected = -0.5 * ((observed[i] - model[i]) / err[i]) ** 2 - np.log(
                err[i] * np.sqrt(2 * np.pi)
            )
            np.testing.assert_allclose(value, expected, rtol=1e-13)
        elif i == 1:
            assert abs(float(value)) < 1e-30
        else:
            assert np.isfinite(value) and value < -800


@vv.case("scene-fits", "code-verification", basis="equivalence")
@vv.seam("skyscapes", "photomancy")
def test_photometry_band_planet_epoch_routing():
    """Truth source: separate system calls for two bands, planets and epochs.

    Tolerance basis: x64 rtol 1e-13 for repeated quadrature and orbit propagation.
    Wavelength-dependent albedos make the bands distinguishable.
    Fails if: any band, planet or epoch index is broadcast, swapped or reused.
    """

    class ColoredLambertian(LambertianPhysicalModel):
        def contrast(
            self, phase_angle_rad, star_planet_distance_AU, wavelength_nm, Rp_Rearth
        ):
            return (
                super().contrast(
                    phase_angle_rad, star_planet_distance_AU, wavelength_nm, Rp_Rearth
                )
                * wavelength_nm
                / 600.0
            )

    first = _system().planets[0]
    second = _system(_ELEMENTS * np.array([1.3, 1, 1, 1, 1, 2, 1])).planets[0]
    planets = tuple(
        eqx.tree_at(
            lambda p: p.physical_model, planet, ColoredLambertian(jnp.array([albedo]))
        )
        for planet, albedo in ((first, 0.2), (second, 0.4))
    )
    a, e, node, inc, peri, mean, epoch_d = _ELEMENTS
    system = System.from_classical(
        FlatStar(1.0),
        planets,
        star_mass_Msun=1.0,
        planet_mass_Msun=_MASS * Mearth2kg / Msun2kg,
        dist_pc=10.0,
        a_AU=jnp.array([a, 1.3 * a]),
        e=e,
        W_rad=node,
        i_rad=inc,
        w_rad=peri,
        M0_rad=jnp.array([mean, 2 * mean]),
        epoch_d=epoch_d,
        epoch=Epoch(2451545.0),
    )
    edges = (Band(450.0, 550.0), Band(850.0, 950.0))
    bands = (
        Transmission.from_band(edges[0]),
        Transmission([850, 885, 950], [0, 1, 0.3]),
    )
    rule = Quadrature.build(edges, system, *bands)
    band_ids, planet_ids = [1, 0, 1, 0], [0, 1, 1, 0]
    data = PhotometryData.from_dmag(
        times=_TIMES,
        dmag=[25.0] * 4,
        dmag_err=[0.2] * 4,
        band_id=band_ids,
        planet_id=planet_ids,
    )
    _, prediction = PhotometryTerm(data, rule, bands, select=lambda m: m[0])((system,))
    expected = [
        system.band_contrasts(rule.band(b), jnp.array([t]), bandpass=bands[b])[0, p, 0]
        for b, p, t in zip(band_ids, planet_ids, _TIMES, strict=True)
    ]
    np.testing.assert_allclose(prediction, expected, rtol=1e-13, atol=0)


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
def test_dmag_contrast_error_propagation():
    """Truth source: 200,000 Gaussian magnitude draws transformed by NumPy.

    Tolerance basis: sigma_dmag=0.002 makes curvature negligible; std sampling
    error 1/sqrt(2*N)=0.00158, so 1% relative exceeds six standard errors.
    The central contrast matches to 1e-14 relative in x64.
    Fails if: the magnitude sign, base, first-order error or padding is wrong.
    """
    data = PhotometryData.from_dmag(
        times=[1.0], dmag=[15.0], dmag_err=[0.002], band_id=[1], max_n=3
    )
    draws = 10 ** (-0.4 * np.random.default_rng(0).normal(15.0, 0.002, 200_000))
    np.testing.assert_allclose(data.contrast[0], 1e-6, rtol=1e-14)
    np.testing.assert_allclose(data.contrast_err[0], draws.std(ddof=1), rtol=0.01)
    np.testing.assert_array_equal(data.is_valid, [True, False, False])
    np.testing.assert_array_equal(data.planet_id, [0, 0, 0])
    np.testing.assert_array_equal(data.is_upper_limit, [False, False, False])


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_rv_default_jitter_and_selection():
    """Truth source: explicit zero jitters versus the default with a model selector.

    Tolerance basis: exact equality for identical arithmetic. Fails if: default
    jitters add noise, select is ignored, or jitter shape mismatches go unchecked.
    """
    data = _new_term("rv").data
    expected = RadialVelocityTerm(data, jitters=jnp.zeros(data.n_inst))(_system())
    actual = RadialVelocityTerm(data, select=lambda m: m[0])((_system(),))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="jitters"):
        RadialVelocityTerm(data, jitters=jnp.zeros(1))


class _SamplingSystem(eqx.Module):
    """Cheap time-dependent observables isolate the measurement sampling laws."""

    def positions(self, time_d):
        east = 0.1 + 1e-5 * time_d
        return jnp.stack((east[None, :], -2 * east[None, :]))

    def ephemeris(self, time_d):
        return time_d

    def star_reflex(self, time_d, ephemeris=None):
        return jnp.stack((1e-10 * time_d**2, -2e-10 * time_d**2))

    def star_radial_velocity(self, time_d):
        return time_d * 0.01

    def band_contrasts(self, quadrature, time_d, *, bandpass=None):
        return jnp.full((len(quadrature.bands), 1, time_d.size), 1e-10)


@pytest.mark.parametrize("kind", ["relative", "stellar", "rv", "pma", "photometry"])
def test_term_sampling_laws(kind):
    """Fails if sampling loses masks, padding, jitter, covariance or censoring.

    Each of five measurement types needs its own sampler structure. With n=2000
    keys the Gaussian means and variances use 4/sqrt(n) and
    4*sqrt(2/(n-1)); cross moments use 4/sqrt(n). This resolves mean biases
    of about 0.09 sigma. Censoring fractions use four binomial standard errors.
    JIT/eager differences allow 32 x64 eps from fused arithmetic.
    """
    from jax.scipy.special import ndtr

    term = (
        RelativeAstrometryTerm(_data(7))
        if kind == "relative"
        else _new_term(kind, max_n=7)
    )
    if kind != "pma":
        valid = term.data.is_valid

        def poison(x):
            if x.dtype == jnp.bool_:
                return x
            fill = -999 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.nan
            return jnp.where(valid, x, fill)

        term = eqx.tree_at(lambda t: t.data, term, jax.tree.map(poison, term.data))
    model = _SamplingSystem()
    prediction = term(model)[1]
    keys = jax.random.split(jax.random.key(0), 2000)
    draws = eqx.tree_at(
        lambda t: t.data,
        term,
        eqx.filter_jit(jax.vmap(lambda key: term.sample(key, model).data))(keys),
    )
    one = term.sample(keys[0], model)
    compiled_one = eqx.tree_at(
        lambda t: t.data, term, jax.tree.map(lambda x: x[0], draws.data)
    )
    for a, b in zip(
        jax.tree.leaves(one.data), jax.tree.leaves(compiled_one.data), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=32 * np.finfo(float).eps, atol=1e-20)
    assert np.isfinite(one(model)[0])
    np.testing.assert_array_equal(one.data.is_valid, term.data.is_valid)
    for a, b in zip(jax.tree.leaves(one.data), jax.tree.leaves(term.data), strict=True):
        assert a.shape == b.shape
        np.testing.assert_array_equal(
            np.asarray(a)[~term.data.is_valid], np.asarray(b)[~term.data.is_valid]
        )
    data = term.data
    if kind in ("relative", "stellar"):
        east = (draws.data.ra[:, :4] - prediction[0, :4]) / data.ra_err[:4]
        north = (draws.data.dec[:, :4] - prediction[1, :4]) / data.dec_err[:4]
        z = jnp.stack(
            (east, (north - data.corr[:4] * east) / jnp.sqrt(1 - data.corr[:4] ** 2)),
            axis=-1,
        )
        assert np.all(
            np.abs(np.mean(z[..., 0] * z[..., 1], axis=0)) < 4 / np.sqrt(2000)
        )
    elif kind == "rv":
        sigma = jnp.sqrt(data.rv_err[:4] ** 2 + term.jitters[data.inst_ids[:4]] ** 2)
        z = (draws.data.rv[:, :4] - prediction[:4]) / sigma
    elif kind == "pma":
        z = np.linalg.solve(
            np.linalg.cholesky(data.pm_anomaly_cov),
            (draws.data.pm_anomaly - prediction).T,
        ).T
        empty = eqx.tree_at(lambda t: t.data.is_valid, term, jnp.array(False))
        np.testing.assert_array_equal(
            empty.sample(keys[0], model).data.pm_anomaly, data.pm_anomaly
        )
    else:
        # Uncensored rows test Gaussian residuals; censored rows test their event law.
        z = (
            draws.data.contrast[:, [0, 2]] - prediction[jnp.array([0, 2])]
        ) / data.contrast_err[jnp.array([0, 2])]
        for row in (1, 3):
            p = float(
                ndtr((data.contrast[row] - prediction[row]) / data.contrast_err[row])
            )
            actual = np.asarray(draws.data.is_upper_limit[:, row])
            assert abs(actual.mean() - p) <= 4 * np.sqrt(p * (1 - p) / 2000)
            np.testing.assert_array_equal(
                np.asarray(draws.data.contrast[:, row])[actual],
                float(data.contrast[row]),
            )
            assert np.all(
                np.asarray(draws.data.contrast[:, row])[~actual]
                >= float(data.contrast[row])
            )
    z = np.asarray(z)
    assert np.all(np.abs(z.mean(axis=0)) < 4 / np.sqrt(2000))
    assert np.all(np.abs(z.var(axis=0, ddof=1) - 1) < 4 * np.sqrt(2 / 1999))


def test_density_sampling_split_keys():
    """Fails if terms reuse a key or sampling changes the prior.

    Equality with each split subkey is exact algebra. Two identical terms over
    n=2000 keys have cross moment within 4/sqrt(n) of zero (four MC errors).
    """
    term = _new_term("rv")
    density = TermLogDensity((term, term), prior=lambda model: jnp.array(-2.0))
    model = _SamplingSystem()
    key = jax.random.key(0)
    draw = density.sample(key, model)
    assert draw.prior is density.prior
    for sampled, subkey in zip(draw.terms, jax.random.split(key, 2), strict=True):
        np.testing.assert_array_equal(
            sampled.data.rv, term.sample(subkey, model).data.rv
        )
    assert not np.array_equal(draw.terms[0].data.rv, draw.terms[1].data.rv)
    draws = eqx.filter_jit(
        jax.vmap(lambda k: tuple(t.data for t in density.sample(k, model).terms))
    )(jax.random.split(key, 2000))
    sigma = jnp.sqrt(term.data.rv_err[0] ** 2 + term.jitters[0] ** 2)
    prediction = term(model)[1][0]
    z = [(d.rv[:, 0] - prediction) / sigma for d in draws]
    assert abs(float(jnp.mean(z[0] * z[1]))) < 4 / np.sqrt(2000)
