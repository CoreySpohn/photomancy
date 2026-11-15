"""Likelihood composition and relative astrometry across the system interface."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg
from orbix import KeplerianOrbit
from orbix.kepler import markley_solve_trig
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene.planet import Planet
from skyscapes.scene.star import FlatStar
from skyscapes.scene.system import System

from photomancy import AbstractTerm, RelativeAstrometryTerm, TermLogDensity
from photomancy.exposure import ExposureLogDensity, ExposureModel, ExposureTerm
from photomancy.likelihoods import gaussian_loglike
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.likelihoods import loglike_relative_astrom

# a [AU], e, node [rad], inclination [rad], periapsis [rad], M0 [rad], t0 [day].
_ELEMENTS = np.array([1.7, 0.31, 1.1, 0.8, 0.6, 0.4, 24.0])
_TIMES = np.array([31.0, 120.0, 290.0, 470.0])


def _system(elements=_ELEMENTS):
    """Build a massless planet with explicit orbital elements and stellar context."""
    a, e, node, inc, peri, mean, epoch = elements
    orbit = KeplerianOrbit(a, e, node, inc, peri, mean, epoch)
    planet = Planet(
        Rp_Rearth=jnp.ones(1),
        Mp_Mearth=jnp.zeros(1),
        orbit=orbit,
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    return System(FlatStar(Msun2kg, 10.0, 1.0), (planet,), markley_solve_trig)


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
    motion = np.sqrt(G * Msun2kg / a**3)
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
        Mp=0.0,
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
        def positions(self, times):
            east = jnp.array([10.0, 30.0])[:, None] + times
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
        lambda instrument, exposure, scene: instrument * exposure + scene.star.dist_pc,
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
