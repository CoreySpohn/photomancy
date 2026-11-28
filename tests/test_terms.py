"""Likelihood composition and relative astrometry across the system interface."""

from dataclasses import replace

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import AU2m, G, Mearth2kg, Msun2kg, d2s
from hwoutils.epoch import Epoch
from hwoutils.leaves import canonical
from hwoutils.precision import PrecisionError
from hwoutils.testing import ulps
from kinematics import (
    predict_pm_anomaly,
    predict_relative_astrometry,
    predict_rv,
    predict_stellar_astrometry,
)
from optixstuff.spectral import Band, Quadrature, Resolution, Transmission
from priorities import Normal
from scipy import integrate, stats
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import Scene
from skyscapes.scene.planet import Planet
from skyscapes.scene.star import FlatStar
from system_helpers import _TIMES, _data, _new_term
from term_utils import ExposureRecord, ExposureRender, both, named_model

from photomancy import (
    PhotometryTerm,
    ProperMotionAnomalyTerm,
    RadialVelocityTerm,
    RelativeAstrometryTerm,
)
from photomancy.infer import AbstractTerm, Fit, ForwardTerm
from photomancy.infer.likelihoods import gaussian_loglike
from photomancy.measurements import (
    AstrometryData,
    DMagData,
    DMagTerm,
    PhotometryData,
    ProperMotionAnomalyData,
    RadialVelocityData,
    loglike_dmag,
    loglike_pm_anomaly,
    loglike_relative_astrom,
    loglike_rv_marginalized,
    loglike_stellar_astrom,
)
from photomancy.measurements.proper_motion import _path_proper_motion

# a [AU], e, node [rad], inclination [rad], periapsis [rad], M0 [rad], t0 [day].
_ELEMENTS = np.array([1.7, 0.31, 1.1, 0.8, 0.6, 0.4, 24.0])


class _Holder(eqx.Module):
    """A model that holds the system one path step below the root."""

    system: object


_MASS = 317.8
_EPOCH = Epoch(2451545.0)


def _system(elements=_ELEMENTS, mass=_MASS, *, epoch=_EPOCH):
    """Build a finite-mass planet with explicit orbital elements and stellar context."""
    a, e, node, inc, peri, mean, epoch_d = elements
    planet = Planet(
        Rp_Rearth=jnp.ones(1),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    return Scene.from_classical(
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


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("pa_deg, axis", [(90.0, 0), (0.0, 1)])
def test_sign_anchor(pa_deg, axis):
    """Truth source: PA is measured east of north; node sets periastron's PA.

    Floating-point basis: 16 eps of the nonzero component for angle conversion,
    the trigonometric factors and rotation of the projected position;
    only the positive sign is asserted on the nonzero axis, independent of scale.
    Fails if: either conversion or system-term prediction flips or swaps axes.
    """
    data = AstrometryData.from_separation_pa(
        time_d=[0.0],
        separation_arcsec=[0.1],
        separation_err_arcsec=[0.001],
        pa_deg=[pa_deg],
        pa_err_deg=[0.1],
    )
    observed = np.array([data.east_arcsec[0], data.north_arcsec[0]])
    system = _system([1.0, 0.2, np.deg2rad(pa_deg), 0.7, 0.0, 0.0, 0.0])
    prediction = RelativeAstrometryTerm(data).predict(system)
    for position in (observed, np.asarray(prediction)[:, 0]):
        assert position[axis] > 0
        assert abs(position[1 - axis]) < float(ulps(16, position[axis]))


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
    data = AstrometryData.from_separation_pa(
        time_d=[25.0],
        separation_arcsec=[sep],
        separation_err_arcsec=[sep_err],
        pa_deg=[np.rad2deg(pa)],
        pa_err_deg=[np.rad2deg(pa_err)],
        planet=[2],
        max_n=max_n,
    )
    draws = np.random.default_rng(427).normal(
        loc=[sep, pa], scale=[sep_err, pa_err], size=(200_000, 2)
    )
    east = draws[:, 0] * np.sin(draws[:, 1])
    north = draws[:, 0] * np.cos(draws[:, 1])
    np.testing.assert_allclose(
        [data.east_err_arcsec[0], data.north_err_arcsec[0]],
        [east.std(ddof=1), north.std(ddof=1)],
        rtol=0.01,
    )
    assert float(data.corr[0]) == pytest.approx(
        np.corrcoef(east, north)[0, 1], abs=0.014
    )
    np.testing.assert_array_equal(data.valid, [True] + [False] * ((max_n or 1) - 1))
    assert int(data.planet[0]) == 2


@vv.case("scene-fits", "code-verification", basis="equivalence")
@vv.seam("skyscapes", "photomancy")
def test_system_element_parity(precision):
    """Truth source: element-based relative astrometry scored by the same likelihood.

    Floating-point basis: 64 eps of apparent semimajor axis for the independent
    mean-anomaly, Kepler and sky rotations. Scoring retains its x64 bound.
    Fails if: system and element paths disagree in epoch, mass, scale or axes.
    """
    data = _data()
    a, e, node, inc, peri, mean, epoch = _ELEMENTS
    motion = np.sqrt(G * (Msun2kg + _MASS * Mearth2kg) / a**3)
    expected_positions = predict_relative_astrometry(
        data.time_d,
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
    term, system = RelativeAstrometryTerm(data), _system()
    prediction = term.predict(system)
    np.testing.assert_allclose(
        prediction, expected_positions, rtol=0, atol=float(ulps(64, a / 10.0))
    )
    if precision == np.float64:
        expected = loglike_relative_astrom(*expected_positions, data)
        np.testing.assert_allclose(term(system), expected, rtol=1e-9, atol=0)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.float64_only(reason="float64 central differences at relative step 1e-5")
@pytest.mark.parametrize(
    "padding",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.slow(
                reason="compiles the masked orbit likelihood and validates gradients "
                "in orbital elements, 5 s projected on CI"
            ),
        ),
    ],
)
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
            lambda d: (
                d.time_d,
                d.east_arcsec,
                d.north_arcsec,
                d.east_err_arcsec,
                d.north_err_arcsec,
                d.corr,
                d.planet,
            ),
            data,
            tuple(
                jnp.where(data.valid, x, fill)
                for x, fill in zip(
                    (
                        data.time_d,
                        data.east_arcsec,
                        data.north_arcsec,
                        data.east_err_arcsec,
                        data.north_err_arcsec,
                        data.corr,
                        data.planet,
                    ),
                    (jnp.nan, jnp.nan, jnp.nan, 0.0, 0.0, jnp.nan, -999),
                    strict=True,
                )
            ),
        )
    term = RelativeAstrometryTerm(data)
    score = jax.jit(lambda elements: term(_system(elements)))
    elements = jnp.asarray(_ELEMENTS)
    actual = jax.jit(jax.grad(score))(elements)
    steps = 1e-5 * np.maximum(1.0, np.abs(_ELEMENTS))
    offsets = np.diag(steps)
    central = (
        jax.vmap(score)(elements + offsets) - jax.vmap(score)(elements - offsets)
    ) / (2 * steps)
    np.testing.assert_allclose(actual, central, rtol=2e-6, atol=1e-6)
    value, prediction = both(term, _system())
    assert np.all(np.isfinite(prediction))
    if padding:

        def unpadded(x):
            return RelativeAstrometryTerm(_data())(_system(x))

        np.testing.assert_allclose(value, unpadded(elements), rtol=1e-12)
        np.testing.assert_allclose(actual, jax.grad(unpadded)(elements), rtol=1e-12)
        np.testing.assert_array_equal(prediction[:, 4:], 0.0)
        empty = RelativeAstrometryTerm(
            eqx.tree_at(lambda d: d.valid, data, jnp.zeros(7, dtype=bool))
        )

        def empty_score(x):
            return empty(_system(x))

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

    data = eqx.tree_at(lambda d: d.planet, _data(), jnp.array([1, 0, 1, 0]))
    prediction = RelativeAstrometryTerm(data).predict(LinearSystem())
    east = _TIMES + np.array([30.0, 10.0, 30.0, 10.0])
    np.testing.assert_array_equal(prediction, [east, -2 * east])


@pytest.mark.float64_only(reason="float64 Fit coordinates")
@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_mixed_terms_sum_in_a_fit():
    """Truth source: separately evaluated image and astrometry terms and one prior.

    Tolerance basis: rtol 1e-12 for the same scalar addition in x64. Fails if
    ``Fit`` routes a term to the wrong model, drops or double counts a term, or
    counts the prior in the likelihood.
    """
    model = ExposureRecord(jnp.array(2.0), (jnp.array(3.0),), _system())
    image = ForwardTerm(
        ExposureRender(
            lambda instrument, exposure, scene: instrument * exposure + scene.dist_pc
        ),
        gaussian_loglike(jnp.array(15.0), 1.0),
    )
    astrometry = RelativeAstrometryTerm(_data(), system_path="scene")
    law = Normal(jnp.asarray(2.0), jnp.asarray(1.0))
    fit = Fit(model, {"instrument": law}, (image, astrometry))
    x = {"instrument": jnp.asarray(2.0)}
    parts = [both(term, model)[0] for term in fit.terms]
    np.testing.assert_allclose(fit.log_likelihood(x), sum(parts), rtol=1e-12)
    np.testing.assert_allclose(
        fit.log_density(x), sum(parts) + law.log_prob(x["instrument"]), rtol=1e-12
    )
    assert isinstance(image, AbstractTerm)
    assert isinstance(astrometry, AbstractTerm)
    with pytest.raises(TypeError):
        AbstractTerm()


@vv.case("scene-fits", "code-verification", basis="equivalence")
@vv.seam("skyscapes", "photomancy")
@pytest.mark.parametrize("kind", ["stellar", "rv", "pma"])
def test_reflex_system_element_parity(kind, precision):
    """Truth source: existing element-array predictors with the same finite mass.

    Floating-point basis: 64 eps of peak observable for mean anomaly, the
    Kepler solve, mass weighting and rotation. For a fitted slope the scale
    is the reflex semimajor axis times 3/window + 2/baseline, bounding the
    sum of absolute regression and differencing weights. Scores retain
    their independent x64 roundoff check.
    Fails if: the system seam changes mass, sign, units, epoch or window model.
    """
    term = _new_term(kind)
    data = term.data
    args = _element_args()
    if kind == "stellar":
        prediction = jnp.stack(predict_stellar_astrometry(data.time_d, **args))
        if precision == np.float64:
            expected = loglike_stellar_astrom(*prediction, data)
    elif kind == "rv":
        a, e, _, inc, peri, mean, epoch = _ELEMENTS
        motion = np.sqrt(G * (Msun2kg + _MASS * Mearth2kg) / a**3)
        prediction = predict_rv(
            data.time_d,
            2 * np.pi / motion,
            Msun2kg,
            _MASS * Mearth2kg * np.sin(inc),
            e,
            np.cos(peri),
            np.sin(peri),
            epoch - mean / motion,
            Mp=_MASS * Mearth2kg,
        ) * (AU2m / d2s)
        if precision == np.float64:
            expected = loglike_rv_marginalized(
                data.rv,
                prediction,
                data.rv_err,
                data.inst_ids,
                data.n_inst,
                term.jitters,
                data.valid,
            )
    else:
        prediction = predict_pm_anomaly(
            data.t_hip,
            data.t_gaia,
            data.gaia_window,
            data.n_epochs,
            **args,
        )
        if precision == np.float64:
            expected = loglike_pm_anomaly(prediction, data)
    scale = jnp.max(jnp.abs(prediction))
    if kind == "pma":
        reflex = args["a"] / args["dist_pc"] * args["Mp"] / (args["Mp"] + args["Ms"])
        scale = reflex * (3 / data.gaia_window + 2 / (data.t_gaia - data.t_hip))
    system = _system()
    np.testing.assert_allclose(
        term.predict(system), prediction, rtol=0, atol=float(ulps(64, scale))
    )
    if precision == np.float64:
        np.testing.assert_allclose(term(system), expected, rtol=1e-12, atol=1e-12)


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
def test_pm_anomaly_shared_ephemeris(n_epochs, monkeypatch, precision):
    """Fails if sharing a solve changes the PM anomaly or solves more than once.

    Truth is three independent scene calls: window, Hipparcos and Gaia.
    For the 25-sample float32 window, four ULPs allow for rounding differences
    between the batched and singleton ephemeris solves. Predictions agree exactly
    in the other cases. The relative 1e-12 score tolerance allows float64 roundoff in
    the ephemeris and covariance reduction. Two samples and the standard window
    cover slice endpoints.
    """
    term = ProperMotionAnomalyTerm(replace(_new_term("pma").data, n_epochs=n_epochs))
    system = _system()
    calls = []
    original = Scene.ephemeris

    def record(self, time_d, **kwargs):
        calls.append(time_d)
        return original(self, time_d, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Scene, "ephemeris", record)
        prediction = term.predict(system)
        assert len(calls) == 1
        if precision == np.float64:
            calls.clear()
            score = term(system)
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
    if precision == np.float32 and n_epochs == 25:
        assert prediction.shape == expected_prediction.shape
        assert jnp.all(
            jnp.abs(prediction - expected_prediction) <= ulps(4, expected_prediction)
        )
    else:
        np.testing.assert_array_equal(prediction, expected_prediction)
    if precision == np.float64:
        expected = loglike_pm_anomaly(expected_prediction, data)
        np.testing.assert_allclose(score, expected, rtol=1e-12, atol=0)


@eqx.filter_jit
def _new_term_value_grad(term, parameters):
    """Compile each observable once, reusing data shapes for validity changes."""

    def score(x):
        return both(term, _system(x[:7], x[7]))

    return jax.value_and_grad(score, has_aux=True)(parameters)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.float64_only(reason="float64 central differences at relative step 1e-5")
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
            return jnp.where(data.valid, x, fill)

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
        lambda d: d.valid, term.data, jnp.zeros_like(term.data.valid)
    )
    empty = eqx.tree_at(lambda t: t.data, term, poison(empty_data))
    (empty_value, empty_prediction), empty_grad = _new_term_value_grad(
        empty, parameters
    )
    assert float(empty_value) == 0.0
    np.testing.assert_array_equal(empty_prediction, 0.0)
    np.testing.assert_array_equal(empty_grad, 0.0)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="measurement log densities require float64")
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
    model = system.band_contrasts(
        rule, times, resolution=Resolution(), transmission=band
    )[0, 0]
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
        value, prediction = both(PhotometryTerm(row, rule, (band,)), system)
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

    Floating-point basis: 32 eps for orbital phase, Lambert factors, band
    weights and the weighted-sum division in batched versus scalar evaluation.
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
    system = Scene.from_classical(
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
        time_d=_TIMES,
        dmag=[25.0] * 4,
        dmag_err=[0.2] * 4,
        band_id=band_ids,
        planet=planet_ids,
    )
    prediction = PhotometryTerm(data, rule, bands, system_path="system").predict(
        _Holder(system)
    )
    expected = [
        system.band_contrasts(
            rule.band(b),
            jnp.array([t]),
            resolution=Resolution(),
            transmission=bands[b],
        )[0, p, 0]
        for b, p, t in zip(band_ids, planet_ids, _TIMES, strict=True)
    ]
    np.testing.assert_allclose(
        prediction,
        expected,
        rtol=float(ulps(32, jnp.ones((), prediction.dtype))),
        atol=0,
    )


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
def test_dmag_contrast_error_propagation():
    """Truth source: 200,000 Gaussian magnitude draws transformed by NumPy.

    Tolerance basis: sigma_dmag=0.002 makes curvature negligible; std sampling
    error 1/sqrt(2*N)=0.00158, so 1% relative exceeds six standard errors.
    The central contrast allows two eps for magnitude scaling and exponentiation.
    Fails if: the magnitude sign, base, first-order error or padding is wrong.
    """
    data = PhotometryData.from_dmag(
        time_d=[1.0], dmag=[15.0], dmag_err=[0.002], band_id=[1], max_n=3
    )
    draws = 10 ** (-0.4 * np.random.default_rng(0).normal(15.0, 0.002, 200_000))
    np.testing.assert_allclose(
        data.contrast[0],
        1e-6,
        rtol=float(ulps(2, jnp.ones((), data.contrast.dtype))),
        atol=0,
    )
    np.testing.assert_allclose(data.contrast_err[0], draws.std(ddof=1), rtol=0.01)
    np.testing.assert_array_equal(data.valid, [True, False, False])
    np.testing.assert_array_equal(data.planet, [0, 0, 0])
    np.testing.assert_array_equal(data.is_upper_limit, [False, False, False])


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_rv_default_jitter_and_selection():
    """Truth source: explicit zero jitters versus the default with a model path.

    Tolerance basis: exact equality for identical arithmetic. Fails if: default
    jitters add noise, the system path is ignored, or a jitter shape mismatch goes
    unchecked.
    """
    data = _new_term("rv").data
    zero = RadialVelocityTerm(data, jitters=jnp.zeros(data.n_inst))
    selected = RadialVelocityTerm(data, system_path="system")
    expected = both(zero, _system())
    actual = both(selected, _Holder(_system()))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="jitters"):
        RadialVelocityTerm(data, jitters=jnp.zeros(1))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_rv_marginal_integrates_to_one_over_the_contrast():
    """Truth source: the flat-offset marginal is a density of the data modulo a shift.

    Two epochs of one instrument (errors 1.5 and 2.5 m/s, jitter 0.7 m/s) leave one
    contrast direction, ``(1, -1) / sqrt(2)``; the likelihood is constant along the
    shift ``(1, 1) / sqrt(2)``. Adaptive quadrature over the contrast, at an
    arbitrary shift, must give 1 (tolerance 1e-9 against a quadrature error
    estimate under 1e-10). A lone epoch of a second instrument removes no dimension
    and adds nothing. Fails if: the ``0.5 sum_j log N_j`` normalization is missing
    (the integral is ``1 / sqrt(2)``) or is taken over all valid epochs instead of
    per instrument (``sqrt(3 / 2)``).
    """
    data = RadialVelocityData.pad(
        max_n=8,
        time_d=[0.0, 1.0, 2.0],
        rv=[0.0, 0.0, 0.0],
        rv_err=[1.5, 2.5, 1.0],
        inst_ids=[0, 0, 1],
        instruments=("a", "b"),
    )
    jitters = jnp.array([0.7, 0.3])
    model = jnp.where(data.valid, 3.0, 0.0)

    @jax.jit
    def likelihood(s):
        shift = 4.0 * jnp.array([1.0, 1.0]) / jnp.sqrt(2.0)
        pair = shift + s * jnp.array([1.0, -1.0]) / jnp.sqrt(2.0)
        rv = data.rv.at[:2].set(model[:2] + pair).at[2].set(-11.0)
        return jnp.exp(
            loglike_rv_marginalized(
                rv, model, data.rv_err, data.inst_ids, 2, jitters, data.valid
            )
        )

    area, error = integrate.quad(
        lambda s: float(likelihood(s)), -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13
    )
    print(f"RV contrast integral: {area:.12f} (quadrature error {error:.1e})")
    assert error < 1e-10
    np.testing.assert_allclose(area, 1.0, rtol=0, atol=1e-9)


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

    def band_contrasts(self, quadrature, time_d, *, transmission=None, resolution=None):
        return jnp.full(
            (len(quadrature.bands), 1, time_d.size), 1e-10, dtype=time_d.dtype
        )


@pytest.mark.parametrize("kind", ["relative", "stellar", "rv", "pma", "photometry"])
def test_term_sampling_laws(kind, precision):
    """Fails if sampling loses masks, padding, jitter, covariance or censoring.

    Each of five measurement types needs its own sampler structure. With n=2000
    keys the Gaussian means and variances use 4/sqrt(n) and
    4*sqrt(2/(n-1)); cross moments use 4/sqrt(n). This resolves mean biases
    of about 0.09 sigma. Censoring fractions use four binomial standard errors.
    Floating-point basis: 32 eps of the signal scale for coordinate propagation
    and the arithmetic of a correlated draw, including near-zero components.
    """
    from jax.scipy.special import ndtr

    term = (
        RelativeAstrometryTerm(_data(7))
        if kind == "relative"
        else _new_term(kind, max_n=7)
    )
    if kind != "pma":
        valid = term.data.valid

        def poison(x):
            if x.dtype == jnp.bool_:
                return x
            fill = -999 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.nan
            return jnp.where(valid, x, fill)

        term = eqx.tree_at(lambda t: t.data, term, jax.tree.map(poison, term.data))
    model = _SamplingSystem()
    prediction = term.predict(model)
    keys = jax.random.split(jax.random.key(0), 2000)
    draws = eqx.tree_at(
        lambda t: t.data,
        term,
        eqx.filter_jit(jax.vmap(lambda key: term.draw(key, model).data))(keys),
    )
    one = term.draw(keys[0], model)
    compiled_one = eqx.tree_at(
        lambda t: t.data, term, jax.tree.map(lambda x: x[0], draws.data)
    )
    for a, b in zip(
        jax.tree.leaves(one.data), jax.tree.leaves(compiled_one.data), strict=True
    ):
        if jnp.issubdtype(a.dtype, jnp.floating):
            scale = jnp.nanmax(jnp.abs(a), initial=0.0)
            np.testing.assert_allclose(a, b, rtol=0, atol=float(ulps(32, scale)))
        else:
            np.testing.assert_array_equal(a, b)

    if precision == np.float64:
        assert np.isfinite(one(model))
    np.testing.assert_array_equal(one.data.valid, term.data.valid)
    for a, b in zip(jax.tree.leaves(one.data), jax.tree.leaves(term.data), strict=True):
        assert a.shape == b.shape
        np.testing.assert_array_equal(
            np.asarray(a)[~term.data.valid], np.asarray(b)[~term.data.valid]
        )
    data = term.data
    if kind in ("relative", "stellar"):
        east = (
            draws.data.east_arcsec[:, :4] - prediction[0, :4]
        ) / data.east_err_arcsec[:4]
        north = (
            draws.data.north_arcsec[:, :4] - prediction[1, :4]
        ) / data.north_err_arcsec[:4]
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
        empty = eqx.tree_at(lambda t: t.data.valid, term, jnp.array(False))
        np.testing.assert_array_equal(
            empty.draw(keys[0], model).data.pm_anomaly, data.pm_anomaly
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


def test_a_valid_row_selecting_a_missing_planet_is_refused_by_check(precision):
    """Planet 1 or -1 on a one-planet system raises in ``check`` for both selectors.

    Truth: the system has one planet, so valid selectors are ``[0, 1)``; negative
    values do not wrap. A padded row with planet 5 is not read.
    Fails if: the selector is clamped (JAX gather) and the row silently scores planet 0.
    """
    system = _system()
    astrometry = _data()
    photometry = _new_term("photometry")
    for planet in (1, -1):
        row = eqx.tree_at(lambda d: d.planet, astrometry, jnp.full(4, planet))
        with pytest.raises(ValueError, match="planet"):
            RelativeAstrometryTerm(row).check(system)
        if precision == np.float64:
            with pytest.raises(ValueError, match="planet"):
                Fit(system, {}, (RelativeAstrometryTerm(row),))
        data = eqx.tree_at(lambda d: d.planet, photometry.data, jnp.full(4, planet))
        with pytest.raises(ValueError, match="planet"):
            eqx.tree_at(lambda t: t.data, photometry, data).check(system)
    RelativeAstrometryTerm(astrometry).check(system)
    photometry.check(system)
    padded = eqx.tree_at(
        lambda d: (d.planet, d.valid),
        astrometry,
        (jnp.full(4, 5), jnp.zeros(4, bool)),
    )
    RelativeAstrometryTerm(padded).check(system)


def test_selector_checks_skip_traced_inputs_and_a_band_only_model_is_accepted(
    precision,
):
    """Traced selectors skip ``check``; a model with only ``band_contrasts`` fits.

    Truth: under ``jit`` the indices are unknown, so no check can run and none may
    raise; a model that implements only the documented ``band_contrasts`` has its
    planet count from the planet axis of that result (one planet here).
    Fails if: ``check`` converts a tracer to NumPy, or the count is read from
    ``positions`` (absent on this model).
    """
    system = _system()
    astrometry = _data()
    photometry = _new_term("photometry")

    def run(planet, valid):
        row = eqx.tree_at(lambda d: (d.planet, d.valid), astrometry, (planet, valid))
        RelativeAstrometryTerm(row).check(system)
        data = eqx.tree_at(
            lambda d: (d.planet, d.valid), photometry.data, (planet, valid)
        )
        eqx.tree_at(lambda t: t.data, photometry, data).check(system)
        return jnp.zeros(())

    jax.jit(run)(jnp.full(4, 7), jnp.ones(4, bool))

    class BandOnly(eqx.Module):
        scale: jax.Array

        def band_contrasts(self, quadrature, time_d, *, resolution, transmission):
            return self.scale * jnp.ones((len(quadrature.bands), 1, time_d.size))

    model = BandOnly(jnp.asarray(1e-10))
    photometry.check(model)
    if precision == np.float64:
        Fit(model, {}, (photometry,))
    bad = eqx.tree_at(lambda d: d.planet, photometry.data, jnp.ones(4, int))
    with pytest.raises(ValueError, match="planet"):
        eqx.tree_at(lambda t: t.data, photometry, bad).check(model)


def test_a_valid_band_id_outside_the_quadrature_is_refused_by_check(precision):
    """Band -1 or 1 with one band raises; band 0 and padded rows pass.

    Truth: one band, so valid indices are ``[0, 1)``; JAX would clamp -1 and 1 to
    band 0 and score the wrong filter.
    Fails if: ``band_id`` is not range checked.
    """
    system = _system()
    term = _new_term("photometry")
    for band in (1, -1):
        data = eqx.tree_at(lambda d: d.band_id, term.data, jnp.full(4, band))
        bad = eqx.tree_at(lambda t: t.data, term, data)
        with pytest.raises(ValueError, match="band"):
            bad.check(system)
        if precision == np.float64:
            with pytest.raises(ValueError, match="band"):
                Fit(system, {}, (bad,))
    padded = eqx.tree_at(
        lambda d: (d.band_id, d.valid),
        term.data,
        (jnp.full(4, 9), jnp.zeros(4, bool)),
    )
    eqx.tree_at(lambda t: t.data, term, padded).check(system)


def test_a_traced_planet_count_and_a_late_epoch_model_pass_check(precision):
    """A traced ``n_planets`` skips the check; a model covering only later days fits.

    Truth: under ``jit`` the count is unknown, so nothing can be tested and nothing may
    raise. A band-only model that has no prediction before day 1 still predicts every
    valid row, whose epochs are all after day 1, so the planet count is read there, not
    at a synthetic day zero; with no valid row the model is not called at all.
    Fails if: ``int()`` is taken of a traced count, or the forward call is made at
    day zero or with no valid selector.
    """
    system = _system()
    astrometry = _data()
    photometry = _new_term("photometry")

    class Counted(eqx.Module):
        n_planets: jax.Array

        def positions(self, *, time_d):
            return jnp.zeros((2, 1, time_d.size))

    def run(n):
        RelativeAstrometryTerm(astrometry).check(Counted(n))
        photometry.check(Counted(n))
        return n

    jax.jit(run)(jnp.asarray(1))

    class Late(eqx.Module):
        scale: jax.Array

        def band_contrasts(self, quadrature, time_d, *, resolution, transmission):
            if np.any(np.asarray(time_d) < 1.0):
                raise ValueError("no prediction before day 1")
            return self.scale * jnp.ones((len(quadrature.bands), 1, time_d.size))

    late = eqx.tree_at(
        lambda d: (d.time_d, d.valid),
        photometry.data,
        (jnp.array([5.0, 6.0, 7.0, 8.0]), jnp.ones(4, bool)),
    )
    term = eqx.tree_at(lambda t: t.data, photometry, late)
    term.check(Late(jnp.asarray(1e-10)))
    if precision == np.float64:
        Fit(Late(jnp.asarray(1e-10)), {}, (term,))
    unused = eqx.tree_at(lambda d: d.valid, late, jnp.zeros(4, bool))
    eqx.tree_at(lambda t: t.data, term, unused).check(Late(jnp.asarray(1e-10)))
    assert system is not None


@pytest.mark.float64_only(reason="float64 density of float32 measurement operands")
@pytest.mark.parametrize("kind", ["relative", "stellar", "pma"])
def test_narrow_bivariate_measurement_density(kind):
    """The east residual 1 + 2**-25 rounds to 1 if subtracted in float32.

    The independent bivariate normal uses the exact stored covariance. Eight
    float64 ULPs allow its factorization, quadratic form and normalization.
    """
    small = 2.0**-25
    observed = jnp.array([-small, 0.5], jnp.float32)
    predicted = jnp.array([1.0, -0.75], jnp.float32)
    sx, sy, rho = 0.75, 1.5, 0.25
    covariance = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
    if kind == "pma":
        data = replace(
            ProperMotionAnomalyData.zeros(),
            pm_anomaly=observed,
            pm_anomaly_cov=jnp.asarray(covariance, jnp.float32),
            valid=jnp.array(True),
        )
        value = loglike_pm_anomaly(predicted, data)
    else:
        data = replace(
            AstrometryData.zeros(1),
            east_arcsec=observed[:1],
            north_arcsec=observed[1:],
            east_err_arcsec=jnp.array([sx], jnp.float32),
            north_err_arcsec=jnp.array([sy], jnp.float32),
            corr=jnp.array([rho], jnp.float32),
            valid=jnp.array([True]),
        )
        score = (
            loglike_relative_astrom if kind == "relative" else loglike_stellar_astrom
        )
        value = score(predicted[:1], predicted[1:], data)
    expected = stats.multivariate_normal.logpdf(
        np.asarray(observed, np.float64), np.asarray(predicted, np.float64), covariance
    )
    assert value.dtype == np.float64
    np.testing.assert_allclose(value, expected, rtol=0, atol=float(ulps(8, expected)))


@pytest.mark.float64_only(reason="float64 density of float32 measurement operands")
def test_narrow_rv_series_are_centered_in_float64():
    """For two equal-noise velocities only the normalized contrast is observed.

    The projected residual is sqrt(2) * (1 + 2**-25). Eight float64 ULPs
    cover the two segment means, quadratic and Gaussian log normalization.
    """
    small = 2.0**-25
    observed = jnp.array([-small, 1.0], jnp.float32)
    model = -observed
    errors = jnp.full(2, 0.75, jnp.float32)
    jitter = jnp.array([1.0], jnp.float32)
    value = loglike_rv_marginalized(
        observed, model, errors, jnp.zeros(2, int), 1, jitter, jnp.ones(2, bool)
    )
    expected = stats.norm.logpdf(np.sqrt(2) * (1 + small), scale=1.25)
    assert value.dtype == np.float64
    np.testing.assert_allclose(value, expected, rtol=0, atol=float(ulps(8, expected)))


@pytest.mark.float64_only(reason="float64 density of float32 measurement operands")
@pytest.mark.parametrize("upper", [False, True])
def test_narrow_photometric_measurement_density(upper):
    """Compare the density and censored mass to SciPy at the float64 residual.

    The negative observation is below a float32 half-ULP at the prediction.
    Eight float64 ULPs allow standardization and the normal log PDF or CDF.
    """
    term = canonical(_new_term("photometry"), precision="float32")
    data = jax.tree.map(lambda x: x[:1], term.data)
    observed = jnp.array([-(2.0**-60)], jnp.float32)
    sigma = jnp.array([1e-10], jnp.float32)
    term = eqx.tree_at(
        lambda t: t.data,
        term,
        replace(
            data,
            contrast=observed,
            contrast_err=sigma,
            is_upper_limit=jnp.array([upper]),
        ),
    )
    prediction = term.predict(_SamplingSystem())
    assert prediction.dtype == np.float32
    value = term(_SamplingSystem())
    law = stats.norm(loc=float(prediction[0]), scale=float(sigma[0]))
    expected = (law.logcdf if upper else law.logpdf)(float(observed[0]))
    assert value.dtype == np.float64
    np.testing.assert_allclose(value, expected, rtol=0, atol=float(ulps(8, expected)))


@pytest.mark.float64_only(reason="float64 density of float32 measurement operands")
@pytest.mark.parametrize("detected", [False, True])
def test_narrow_magnitude_measurement_density(detected):
    """The null interpolates the limiting magnitude and scores in flux space.

    A quarter float32 ULP in both the observation and the curve intercept
    survives the float64 Gaussian calculation. Sixteen ULPs cover the
    interpolation, base-10 exponential and normal log CDF.
    """
    small = 2.0**-25
    data = replace(
        DMagData.zeros(1),
        sep_grid=jnp.array([[0.0, 2.0]], jnp.float32),
        dmag0_grid=jnp.array([[-small, 1.0]], jnp.float32),
        dmag_obs=jnp.array([-small], jnp.float32),
        dmag_err=jnp.array([0.75], jnp.float32),
        is_detected=jnp.array([detected]),
        valid=jnp.array([True]),
    )
    one = jnp.ones(1, jnp.float32)
    value = loglike_dmag(one, one, data)
    if detected:
        expected = stats.norm.logpdf(-small, loc=1.0, scale=0.75)
    else:
        limit_mag = (1.0 - small) / 2
        expected = stats.norm.logcdf(5.0 * (1 - 10.0 ** (-0.4 * (1 - limit_mag))))
    assert value.dtype == np.float64
    np.testing.assert_allclose(value, expected, rtol=0, atol=float(ulps(16, expected)))


@pytest.mark.float32_only(reason="measurement densities require float64")
def test_measurement_densities_refuse_single_precision():
    """Each measurement score refuses before the forward model is evaluated."""
    relative = RelativeAstrometryTerm(_data())
    rv = _new_term("rv")
    pma = _new_term("pma")
    magnitude = DMagTerm(DMagData.zeros(1))
    for term in (
        relative,
        _new_term("stellar"),
        rv,
        pma,
        magnitude,
        _new_term("photometry"),
    ):
        with pytest.raises(PrecisionError, match="Enable x64"):
            term(None)
    for call in (
        lambda: loglike_relative_astrom(None, None, None),
        lambda: loglike_stellar_astrom(None, None, None),
        lambda: loglike_pm_anomaly(None, None),
        lambda: loglike_rv_marginalized(None, None, None, None, 1, None, None),
        lambda: loglike_dmag(None, None, None),
        lambda: relative.linear_gaussian(None),
    ):
        with pytest.raises(PrecisionError, match="Enable x64"):
            call()


def test_narrow_proper_motion_draw_keeps_the_covariance_dtype():
    """A narrow forward stays narrow even with float64 available.

    Fails if the masked fallback covariance widens the Cholesky and draw.
    Working dtype and preservation of an invalid row are exact.
    """
    term = canonical(_new_term("pma"), precision="float32")
    model = canonical(_system(), precision="float32")
    key = jax.random.key(0)
    invalid = eqx.tree_at(lambda t: t.data.valid, term, jnp.array(False))
    assert term.predict(model).dtype == np.float32
    compiled = eqx.filter_jit(lambda t, m: t.draw(key, m))
    for test_term in (term, invalid):
        for evaluate in (lambda t, m: t.draw(key, m), compiled):
            value = evaluate(test_term, model).data.pm_anomaly
            assert value.dtype == np.float32
            if test_term is invalid:
                np.testing.assert_array_equal(value, invalid.data.pm_anomaly)


def test_photometry_draw_uses_the_common_threshold_grid(monkeypatch, precision):
    """A draw that rounds to its upper limit becomes a measured detection.

    Zero noise isolates the comparison. The previous float32 value stays
    censored. Fails if the wide draw below its limit rounds to equality but
    retains the upper-limit flag, or if deciding on the grid rounds the sample.
    """
    term = _new_term("photometry")
    data = eqx.tree_at(
        lambda d: (d.contrast, d.is_upper_limit),
        term.data,
        (jnp.ones_like(term.data.contrast), term.data.valid),
    )
    term = eqx.tree_at(lambda t: t.data, term, data)
    monkeypatch.setattr(
        jax.random, "normal", lambda key, shape, dtype: jnp.zeros(shape, dtype)
    )
    monkeypatch.setattr(
        type(term), "predict", lambda self, value: jnp.full_like(data.contrast, value)
    )

    def draw(value):
        return term.draw(jax.random.key(0), value).data

    last = np.nextafter(np.float32(1), np.float32(-np.inf))
    for evaluate in (draw, jax.jit(draw)):
        for value, censored in ((1.0, False), (1 - 2**-26, False), (last, True)):
            expected = jnp.asarray(value, dtype=precision)
            sampled = evaluate(expected)
            np.testing.assert_array_equal(
                sampled.is_upper_limit, np.full(data.valid.shape, censored)
            )
            np.testing.assert_array_equal(
                sampled.contrast, np.full(data.valid.shape, 1 if censored else expected)
            )


@pytest.mark.float64_only(reason="curvature checks forward owners in float64")
def test_measurement_curvature_owners_exclude_data_and_include_band_response():
    """Narrow measurements are admissible when the model and calibrations are wide."""
    terms = (
        RelativeAstrometryTerm(_data()),
        DMagTerm(DMagData.zeros(1)),
        *(_new_term(kind) for kind in ("stellar", "rv", "pma", "photometry")),
    )
    for term in terms:
        narrow_data = canonical(term.data, precision="float32")
        term = eqx.tree_at(lambda t: t.data, term, narrow_data)
        term._check_forward_precision("curvature", "term")

    photometry = _new_term("photometry")
    bands = canonical(photometry.transmissions, precision="float32")
    narrow = eqx.tree_at(lambda t: t.transmissions, photometry, bands)
    with pytest.raises(PrecisionError, match=r"term\.transmissions\[0\]"):
        narrow._check_forward_precision("curvature", "term")


class _PlainForward(eqx.Module):
    """A forward that defines none of the optional attributes of a term."""

    def __call__(self, model):
        return model.x


def test_a_forward_without_footprint_check_or_assumptions_keeps_the_empty_defaults():
    """Fails if ``ForwardTerm`` invents a footprint, an assumption or a check.

    Truth source: the released term. A forward that defines none of
    ``footprint_paths``, ``assumptions`` and ``check`` gives empty tuples and a
    ``check`` that accepts the model. Tolerance basis: exact.
    """
    term = ForwardTerm(_PlainForward(), gaussian_loglike(jnp.zeros(()), jnp.ones(())))
    assert (term.footprint_paths, term.assumptions, term.reads) == ((), (), ())
    assert term.check(named_model()) is None
