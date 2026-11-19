"""Pointwise observable scores for synthetic one- and two-planet systems."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg, two_pi
from orbix.system import Body, ClassicalChart, EccentricityVectorChart, System
from skyscapes.scene import Scene
from term_utils import null_dmag_term
from test_system_density import _floor
from test_system_priors import _orbits, _scale_coordinates, _scene
from test_terms import _data, _new_term

from photomancy.measurements import (
    DMagTerm,
    ImagingData,
    NullData,
    PMAnomalyData,
    ProperMotionAnomalyTerm,
    RadialVelocityTerm,
    RelativeAstromData,
    RelativeAstrometryTerm,
    RVData,
    StellarAstromData,
    StellarAstrometryTerm,
)
from photomancy.orbit.inference import build_chart_logdensity
from photomancy.priors import orbit_log_prior

DYNAMICAL = ("rv", "relative_astrom", "stellar_astrom", "pm_anomaly")
CHANNELS = (*DYNAMICAL, "null", "imaging", "custom")

# Frozen from the retired NumPyro joint at the four _orbits(23.0) points: per-channel
# log factors in CHANNELS order, their sum, and the constrained joint transformed to
# the physical measure (the reference measure of the physical-orbit density).
_FACTORS = [
    [
        -13.65882781405432,
        -8.01563595197823,
        27.379562458845115,
        27.424456247836655,
        -2.2505191679061682,
        -33.94885489118106,
        -0.07123173736508238,
    ],
    [
        -7.861283711114213,
        -21.676237792774554,
        28.337968122265124,
        27.432471436302908,
        -0.0001157326822710765,
        -47.92106506840513,
        -0.0201692217160477,
    ],
    [
        -6.505474449349913,
        -41.26712469420502,
        27.26816594769562,
        27.42542569570769,
        -0.0002980509959258402,
        -26.60136077113804,
        -0.01964650830939847,
    ],
    [
        -10.414682236896756,
        -110.95304960656905,
        27.641056812284845,
        27.402684121535767,
        -0.00017467008139153704,
        -42.17993640196971,
        -0.24224498015664636,
    ],
]
_TOTAL = [
    -3.141050855803087,
    -21.70843196812418,
    -19.70031283059499,
    -108.74634696185295,
]
_COMMON = [
    -6.457427490045797,
    -26.849742187702834,
    -26.799394651432088,
    -118.5752126335516,
]
# Frozen independent-pair factors in DYNAMICAL order for the two-planet fixture.
_PAIR_FACTORS = [
    -5.697835665953152,
    -16.947408675704036,
    31.03823489541007,
    27.592546878285674,
]


class Measurements(eqx.Module):
    """One container for every channel's data."""

    rv: RVData
    relative_astrom: RelativeAstromData
    stellar_astrom: StellarAstromData
    pm_anomaly: PMAnomalyData
    null: NullData
    imaging: ImagingData


@pytest.fixture(scope="module")
def data():
    """Synthetic correlated measurements with one padded row per time series."""
    relative = _data(max_n=5)
    # Use a wider astrometric error across the four very different orbits.
    relative = eqx.tree_at(
        lambda d: (d.ra_err, d.dec_err),
        relative,
        (jnp.full(5, 0.03), jnp.full(5, 0.04)),
    )
    curve = dict(
        epochs=[9.0, 48.0, 121.0],
        sep_grid=jnp.tile(jnp.array([0.005, 0.08, 0.6]), (3, 1)),
        dmag0_grid=jnp.array([[19, 20, 19.5], [20, 21, 21.5], [20, 19, 20.5]]),
        snr_thresh=4.0,
        max_n=4,
        max_pts=3,
    )
    return Measurements(
        rv=_new_term("rv", max_n=5).data,
        relative_astrom=relative,
        stellar_astrom=_new_term("stellar", max_n=5).data,
        pm_anomaly=replace(_new_term("pma").data, n_epochs=9),
        null=NullData.pad(**curve),
        imaging=ImagingData.pad(
            **curve,
            is_detected=[True, False, True],
            dmag_obs=[22.0, 0.0, 19.0],
            dmag_err=[0.4, 1.0, 0.7],
        ),
    )


def _dynamical_terms(data):
    return (
        RadialVelocityTerm(data.rv),
        RelativeAstrometryTerm(data.relative_astrom),
        StellarAstrometryTerm(data.stellar_astrom),
        ProperMotionAnomalyTerm(data.pm_anomaly),
    )


def _check_scores(name, actual, expected, components):
    """Bound each point separately by accumulated float64 roundoff."""
    errors = np.abs(np.atleast_1d(actual) - np.atleast_1d(expected))
    floors = np.array([_floor(row) for row in components])
    print(f"{name}: max_error={errors.max():.7g}, max_floor={floors.max():.7g}")
    assert np.all(errors <= floors), (name, errors, floors)


@pytest.mark.medium(reason="Compiles the full joint in two charts.")
def test_all_channels_and_reference_measure(data):
    """Fails if an observable, its multiplicity, or a reference Jacobian changes.

    The constrained joint is per dlog10(P) de dc dW dw dM dlog10(body) dj.
    Transform it to the physical measure independently of either chart. Each
    score allows max(1e-10, 64 eps sum(1 + abs(contribution))) per orbit.
    """
    mass, radius, albedo = 23.0, 0.00012, 0.32
    jitter = jnp.array([0.4, 0.7, 0.1])
    observed_e = jnp.array(0.31)

    def custom_term(scene):
        residual = scene.dynamics.orbits.e[0] - observed_e
        return -(residual**2), residual

    terms = (
        *_dynamical_terms(data),
        null_dmag_term(data.null),
        DMagTerm(data.imaging),
        custom_term,
    )
    orbits = _orbits(mass)
    factors, total, common = (jnp.asarray(x) for x in (_FACTORS, _TOTAL, _COMMON))
    assert np.ptp(factors[:, CHANNELS.index("null")]) > 1e-3
    physical = dict(
        mass_Mearth=mass, radius_AU=radius, albedo=albedo, jitter_m_per_s=jitter
    )
    log_scales = jax.tree.map(jnp.log, physical)
    for chart in (ClassicalChart(), EccentricityVectorChart()):
        density = build_chart_logdensity(
            chart,
            _scene,
            lambda s: orbit_log_prior(s.dynamics.orbits)[0],
            star_mass_kg=Msun2kg,
            terms=terms,
            scalars=_scale_coordinates(0.8),
            body_inputs=("radius_AU", "albedo"),
            term_inputs={0: ("jitter_m_per_s",)},
        )

        def evaluate(x, density=density):
            params = dict(orbit=x, **log_scales)
            value, aux = density.value_and_aux(params)
            return jnp.stack(aux["term_values"]), value, density.log_volume(params)

        points = chart.from_orbit(orbits)
        outside = points[:1].at[:, -1].add(two_pi)
        scores, joint, volume = eqx.filter_jit(jax.vmap(evaluate))(
            jnp.concatenate([points, outside])
        )
        assert jnp.isneginf(joint[-1])
        actual = scores[:-1]
        reference_value = joint[:-1] - volume[:-1]
        label = type(chart).__name__
        for i, name in enumerate(CHANNELS):
            _check_scores(f"{label} {name}", actual[:, i], factors[:, i], factors[:, i])
        _check_scores(f"{label} summed", actual.sum(axis=1), total, factors)
        parts = jnp.column_stack((factors, common - total))
        _check_scores(f"{label} full", reference_value, common, parts)
    # A vector-valued user factor would turn the joint into a vector.
    bad = eqx.tree_at(
        lambda d: d.terms, density, (lambda s: (s.dynamics.orbits.e, None),)
    )
    bad = eqx.tree_at(lambda d: d.term_inputs, bad, {})
    with pytest.raises(ValueError, match="scalar log-likelihood"):
        eqx.filter_eval_shape(bad, dict(orbit=points[0], **log_scales))


class _IndependentPairs(eqx.Module):
    """A sum of separate star-companion solutions in observation order."""

    pairs: tuple

    def ephemeris(self, time_d):
        return tuple(pair.solve(time_d) for pair in self.pairs)

    def positions(self, time_d):
        return jnp.stack(
            [jnp.stack(e.offset_arcsec(1, 0)) for e in self.ephemeris(time_d)], axis=1
        )

    def star_reflex(self, time_d):
        return sum(
            jnp.stack(e.offset_arcsec(0, "barycenter")) for e in self.ephemeris(time_d)
        )

    def star_radial_velocity(self, time_d):
        return sum(
            e.radial_velocity_m_per_s(0, "barycenter") for e in self.ephemeris(time_d)
        )


@pytest.fixture(scope="module")
def two_planets(data):
    """One geometry in both conventions; the relative data alternate planet IDs."""
    masses = jnp.array([100.0, 320.0])
    orbits = jax.tree.map(lambda x: x[:2], _orbits(jnp.tile(masses, 2)))
    scenes = tuple(
        _scene(
            jax.tree.map(lambda leaf, i=i: leaf[i], orbits),
            mass_Mearth=masses[i],
            radius_AU=0.0001,
            albedo=0.3,
        )
        for i in range(2)
    )
    pairs = _IndependentPairs(tuple(s.dynamics for s in scenes))
    jacobi = System.from_classical(
        Body(1.0),
        Body(masses * Mearth2kg / Msun2kg),
        pairs.pairs[0].frame,
        a_AU=orbits.a_AU,
        e=orbits.e,
        **{
            k: v
            for k, v in orbits.to_classical().items()
            if k in {"i_rad", "W_rad", "w_rad", "M0_rad"}
        },
    )
    scene = Scene(
        scenes[0].star,
        tuple(s.planets[0] for s in reversed(scenes)),
        jacobi,
        body_index=(1, 0),
    )
    data = eqx.tree_at(
        lambda d: d.relative_astrom.planet_id, data, jnp.array([0, 1, 0, 1, 0])
    )
    return data, pairs, scene


def test_independent_pair_scores(two_planets):
    """Fails if the multi-planet terms use the outer pair's wrong mass or origin.

    The frozen reference first sums stellar RV, reflex and PMA, and selects each
    planet's relative rows. Scores use the per-factor float64 density floor.
    """
    data, pairs, _ = two_planets
    jitter = jnp.array([0.4, 0.7, 0.1])

    @eqx.filter_jit
    def compare(data, pairs):
        rv, *others = _dynamical_terms(data)
        return jnp.stack(
            [rv(pairs, jitter_m_per_s=jitter), *(term(pairs) for term in others)]
        )

    scores, expected = compare(data, pairs), jnp.asarray(_PAIR_FACTORS)
    for name, score, target in zip(DYNAMICAL, scores, expected, strict=True):
        _check_scores(f"independent {name}", score, target, [target])


@pytest.mark.medium(
    reason="Solves both multi-planet conventions and independent PMA epochs."
)
def test_jacobi_predictions(two_planets):
    """Fails if a term silently reverts to pair masses or a star-centred reflex.

    Reference the orbix ephemeris at the actual Jacobi masses and times.
    Prediction rtol is 1e-12, with atol in m/s, arcsec, arcsec and arcsec/day.
    """
    data, pairs, scene = two_planets
    jacobi = scene.dynamics
    time_d = data.rv.times
    eph = jacobi.solve(time_d)
    relative = jnp.stack(eph.offset_arcsec(jnp.arange(1, 3), 0))
    pm = data.pm_anomaly
    window_times = jnp.linspace(
        pm.t_gaia - pm.gaia_window / 2, pm.t_gaia + pm.gaia_window / 2, pm.n_epochs
    )
    window = jnp.stack(jacobi.solve(window_times).offset_arcsec(0, "barycenter"))
    # A least-squares line with free intercept is independent of the chosen origin.
    fit = np.polynomial.polynomial.polyfit(
        np.asarray(window_times - pm.t_gaia), np.asarray(window).T, 1
    )
    end_times = jnp.stack([pm.t_hip, pm.t_gaia])
    ends = jnp.stack(jacobi.solve(end_times).offset_arcsec(0, "barycenter"))
    expected = (
        eph.radial_velocity_m_per_s(0, "barycenter")[:-1],
        relative[:, data.relative_astrom.planet_id, jnp.arange(time_d.size)][:, :-1],
        jnp.stack(eph.offset_arcsec(0, "barycenter"))[:, :-1],
        fit[1] - (ends[:, 1] - ends[:, 0]) / (pm.t_gaia - pm.t_hip),
    )
    terms = _dynamical_terms(data)
    actual = eqx.filter_jit(lambda model: tuple(t.predict(model) for t in terms))(
        jacobi
    )
    for name, prediction, target, floor in zip(
        DYNAMICAL, actual, expected, (1e-12, 1e-14, 1e-18, 1e-20), strict=True
    ):
        observed = prediction if name == "pm_anomaly" else prediction[..., :-1]
        error = np.max(np.abs(observed - target))
        print(f"Jacobi {name}: max_error={error:.7g}, atol={floor:g}, rtol=1e-12")
        np.testing.assert_allclose(observed, target, rtol=1e-12, atol=floor)
    scene_ids = 1 - data.relative_astrom.planet_id[:-1]
    np.testing.assert_allclose(
        terms[1].predict(scene)[:, :-1],
        relative[:, scene_ids, jnp.arange(time_d.size - 1)],
        rtol=1e-12,
        atol=1e-14,
    )
    pair_positions = terms[1].predict(pairs)
    assert (
        np.max(np.abs(actual[1] - pair_positions)) > 1e-7
    )  # arcsec, resolved physics change
    for term in (null_dmag_term(data.null), DMagTerm(data.imaging)):
        with pytest.raises(ValueError, match=r"(?i)multi-planet"):
            term(jacobi, area_AU2=1e-8)


def test_radiative_padding_and_dark_limit(data):
    """Fails if invalid curves affect gradients, or a dark planet yields infinite dMag.

    Keep the padded rows at fixed shape, poison all unused floating payloads,
    and compare scores, returned predictions and the gradient in log area.
    """
    mass = 23.0
    orbit = jax.tree.map(lambda x: x[1], _orbits(mass))
    model = _scene(orbit, mass_Mearth=mass, radius_AU=0.00012, albedo=0.32).dynamics
    log_area = jnp.log(0.32 * 0.00012**2)

    def score(log_area, term):
        area = jnp.exp(log_area)
        return term(model, area_AU2=area), term.predict(model, area_AU2=area)

    evaluate = eqx.filter_jit(jax.value_and_grad(score, has_aux=True))
    for term in (DMagTerm(data.imaging), null_dmag_term(data.null)):
        poisoned_data = jax.tree.map(
            lambda a: (
                a.at[-1].set(jnp.nan) if jnp.issubdtype(a.dtype, jnp.floating) else a
            ),
            term.data,
        )
        poisoned = eqx.tree_at(lambda t: t.data, term, poisoned_data)
        (clean, expected), clean_grad = evaluate(log_area, term)
        (actual, predicted), gradient = evaluate(log_area, poisoned)
        _check_scores("masked " + type(term).__name__, actual, clean, [clean])
        np.testing.assert_allclose(
            gradient, clean_grad, atol=_floor(clean_grad), rtol=0
        )
        np.testing.assert_array_equal(predicted, expected)
        np.testing.assert_array_equal(jnp.stack(predicted)[:, -1], 0.0)
        masked = eqx.tree_at(
            lambda t: t.data.is_valid, poisoned, jnp.zeros_like(term.data.is_valid)
        )
        (value, prediction), gradient = evaluate(log_area, masked)
        np.testing.assert_array_equal(jnp.stack(prediction), 0.0)
        np.testing.assert_array_equal([value, gradient], 0.0)
        _, dark_dmag = term.predict(model, area_AU2=0.0)
        np.testing.assert_allclose(dark_dmag[:-1], 750.0, rtol=0, atol=1e-12)
