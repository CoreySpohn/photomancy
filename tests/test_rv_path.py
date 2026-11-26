"""Regression coverage for RV units, masks, priors, and orbital dynamics."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import AU2m, G, Mearth2kg, Msun2kg, d2s
from hwoutils.epoch import Epoch
from hwoutils.testing import ulps
from kinematics import predict_rv
from orbix.equations import period_to_sma
from orbix.system import Body, FrameOrbit, SkyFrame, System

from photomancy import RadialVelocityTerm
from photomancy.measurements import RadialVelocityData
from photomancy.orbit.sheets import SheetSum


def _circular_system(periods, masses_kg, cos_i=0.0):
    """Planets on circular orbits at their own two-body semi-major axes."""
    mu = G * (Msun2kg + masses_kg)
    orbits = FrameOrbit.from_classical(
        period_to_sma(periods, Msun2kg + masses_kg),
        0.2 if cos_i == 0.0 else 0.0,
        jnp.arccos(cos_i),
        0.0,
        0.0,
        0.0,
        mu_AU3_per_d2=mu,
    )
    return System(Body(1.0), Body(masses_kg / Msun2kg), orbits, SkyFrame(10.0, Epoch()))


@pytest.mark.parametrize("n_planets", [1, 2])
@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_rv_likelihood_prefers_true_amplitude_in_m_per_s(n_planets):
    """A 10 m/s signal favors its true amplitude over half or twice that mass."""
    times = jnp.linspace(0.0, 800.0, 60)
    periods = jnp.array([300.0, 170.0])[:n_planets]
    masses = jnp.array([100.0, 15.0])[:n_planets]
    clean = sum(
        predict_rv(times, T, Msun2kg, Mp * Mearth2kg, 0.2, 1.0, 0.0, 0.0)
        for T, Mp in zip(periods, masses, strict=True)
    ) * (AU2m / d2s)
    assert 8.0 < float(jnp.max(jnp.abs(clean))) < 15.0
    rv = RadialVelocityData(
        time_d=times,
        rv=clean + 17.0,
        rv_err=jnp.full(times.shape, 0.2),
        inst_ids=jnp.zeros(times.shape, dtype=int),
        valid=jnp.ones(times.shape, dtype=bool),
        instruments=("a",),
    )
    term = RadialVelocityTerm(rv, jitters=jnp.array([0.01]))

    def likelihood(scale):
        return float(term(_circular_system(periods, masses * scale * Mearth2kg)))

    assert likelihood(1.0) > likelihood(0.5)
    assert likelihood(1.0) > likelihood(2.0)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_invalid_rv_rows_have_finite_likelihood_and_gradient(invalid_value):
    """Invalid observations, predictions, errors, and jitters cannot poison sums."""
    from photomancy.measurements import loglike_rv_marginalized

    def evaluate(obs, model, err, jitter):
        return loglike_rv_marginalized(
            obs,
            model,
            err,
            jnp.array([0, 1]),
            2,
            jitter,
            jnp.array([True, False]),
        )

    args = (
        jnp.array([3.0, invalid_value]),
        jnp.array([1.0, invalid_value]),
        jnp.array([0.5, invalid_value]),
        jnp.array([0.2, invalid_value]),
    )
    value, gradients = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1, 2, 3)))(
        *args
    )
    expected = loglike_rv_marginalized(
        args[0][:1],
        args[1][:1],
        args[2][:1],
        jnp.array([0]),
        1,
        args[3][:1],
        jnp.array([True]),
    )
    assert jnp.isfinite(value)
    assert float(value) == pytest.approx(float(expected), abs=1e-14)
    for gradient in gradients:
        assert jnp.all(jnp.isfinite(gradient))
        assert float(gradient[1]) == 0.0


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_single_rv_observation_integrates_to_one():
    """Integrating the sole observation's flat offset leaves log likelihood zero."""
    from photomancy.measurements import loglike_rv_marginalized

    result = loglike_rv_marginalized(
        jnp.array([7.0]),
        jnp.array([-2.0]),
        jnp.ones(1),
        jnp.zeros(1, dtype=int),
        1,
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
    )
    assert float(result) == 0.0


@pytest.mark.parametrize("n_inst", [2, 3])
@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_rv_offset_marginalization_matches_quadrature(n_inst):
    """Three observations match integrating both populated instrument offsets.

    The flat offset integral of instrument ``j`` runs along ``(1, ..., 1)``, which is
    ``sqrt(N_j)`` times its unit vector, so the normalized marginal is the integral
    times ``sqrt(N_j)``: ``sqrt(2)`` here (two epochs and one).
    """
    import numpy as np
    from scipy.integrate import quad

    from photomancy.measurements import loglike_rv_marginalized

    obs = np.array([1.4, -0.3, 2.1])
    model = np.array([0.2, -0.7, 1.0])
    err = np.array([0.7, 0.8, 0.9])
    ids = np.array([0, 0, 1])
    jitters = np.array([0.3, 0.4, 0.5])[:n_inst]
    variance = err**2 + jitters[ids] ** 2
    # The double integral factorizes because the offsets have disjoint rows.
    integral = 1.0
    for instrument in (0, 1):
        selected = ids == instrument

        def gaussian(offset, selected=selected):
            residual = obs[selected] - model[selected] - offset
            return np.exp(-0.5 * np.sum(residual**2 / variance[selected])) / np.sqrt(
                np.prod(2 * np.pi * variance[selected])
            )

        integral *= quad(gaussian, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
        integral *= np.sqrt(np.sum(selected))
    result = loglike_rv_marginalized(
        jnp.asarray(obs),
        jnp.asarray(model),
        jnp.asarray(err),
        jnp.asarray(ids),
        n_inst,
        jnp.asarray(jitters),
        jnp.ones(3, dtype=bool),
    )
    assert float(jnp.exp(result)) == pytest.approx(integral, rel=1e-6)


def test_predict_rv_known_mass_matches_exact_two_body_K():
    """Known mass gives the two-body K at Mp/Ms=0.1 (Lovis/Fischer Eq. 14).

    Floating-point basis: 16 eps for the two cube roots, mass ratio,
    eccentricity and inclination factors and the periapsis rate. The
    reference cube roots avoid a rounded fractional power on masses in kg.
    """
    from hwoutils.constants import G, two_pi

    period, eccentricity, cos_i = 400.0, 0.35, 0.4
    mass = 0.1 * Msun2kg
    minimum_mass = mass * np.sqrt(1.0 - cos_i**2)
    expected_K = (
        np.cbrt(two_pi * G / period)
        * minimum_mass
        / np.cbrt(Msun2kg + mass) ** 2
        / np.sqrt(1.0 - eccentricity**2)
    )
    velocity = predict_rv(
        jnp.array([0.0]),
        period,
        Msun2kg,
        minimum_mass,
        eccentricity,
        1.0,
        0.0,
        0.0,
        Mp=mass,
    )[0]
    measured_K = -velocity / (1.0 + eccentricity)
    assert float(measured_K) == pytest.approx(
        float(expected_K), rel=float(ulps(16, jnp.ones_like(measured_K))), abs=0
    )


def test_predict_rv_matches_orbit_radial_velocity():
    """Compare minimum-mass RV to the orbit's stellar velocity in m/s.

    Fails if true mass, inclination, sign or units differ between the paths.
    Floating-point basis: 64 eps of peak velocity for phase reduction, the
    eccentric-anomaly solve (sensitivity up to 2.5), rotations and conversion.
    """
    from hwoutils.constants import G
    from orbix.equations import period_to_sma
    from orbix.system import FrameOrbit, radial_velocity_m_per_s

    period, e, cos_i, node, cos_w, sin_w, tp = (430.0, 0.6, 0.4, 1.1, 0.6, 0.8, -43.0)
    mass = 0.1 * Msun2kg
    times = jnp.array([-120.0, 0.0, 137.0, 690.0])
    orbit = FrameOrbit.from_classical(
        period_to_sma(period, Msun2kg + mass),
        e,
        jnp.arccos(cos_i),
        node,
        jnp.arctan2(sin_w, cos_w),
        0.0,
        mu_AU3_per_d2=G * (Msun2kg + mass),
        epoch_d=tp,
    )
    _, relative_velocity = orbit.at(times)
    expected = radial_velocity_m_per_s(-mass / (Msun2kg + mass) * relative_velocity)
    actual = predict_rv(
        times,
        period,
        Msun2kg,
        mass * jnp.sqrt(1 - cos_i**2),
        e,
        cos_w,
        sin_w,
        tp,
        Mp=mass,
    ) * (AU2m / d2s)
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=float(ulps(64, jnp.max(jnp.abs(expected))))
    )


def test_rv_term_uses_total_mass(precision):
    """The RV term uses the planet's true mass in both a and K.

    Fails if the semi-major axis or the semi-amplitude drops the planet mass: a
    0.1 Msun planet on a 400 d circular, inclined orbit reproduces the exact
    two-body velocity curve, so the marginalized log likelihood of data equal to
    that curve is the zero-residual value to 1e-10. Floating-point basis:
    32 eps of peak RV for mean anomaly, circular propagation, projection
    and conversion to m/s; the quarter-period zero uses the same scale.
    """
    from hwoutils.constants import two_pi

    from photomancy.measurements import loglike_rv_marginalized

    times = jnp.array([0.0, 100.0, 200.0, 350.0])
    periods = jnp.array([400.0])
    masses = jnp.array([0.1]) * Msun2kg
    cos_i = 0.4
    mass = 0.1 * Msun2kg
    K = np.cbrt(two_pi * G / 400.0) * mass * np.sqrt(1 - cos_i**2)
    K /= np.cbrt(Msun2kg + mass) ** 2
    velocities = -K * np.cos(two_pi * np.asarray(times, dtype=np.float64) / 400.0)
    data = RadialVelocityData(
        time_d=times,
        rv=velocities * (AU2m / d2s),
        rv_err=jnp.full(4, 0.5),
        inst_ids=jnp.zeros(4, dtype=int),
        valid=jnp.ones(4, dtype=bool),
        instruments=("a",),
    )
    term = RadialVelocityTerm(data, jitters=jnp.array([0.01]))
    system = _circular_system(periods, masses, cos_i)
    prediction = term.predict(system)
    np.testing.assert_allclose(
        prediction,
        velocities * (AU2m / d2s),
        rtol=0,
        atol=float(ulps(32, K * (AU2m / d2s))),
    )
    if precision == np.float64:
        expected = loglike_rv_marginalized(
            data.rv,
            data.rv,
            data.rv_err,
            data.inst_ids,
            1,
            jnp.array([0.01]),
            data.valid,
        )
        assert float(term(system)) == pytest.approx(float(expected), abs=1e-10)


def test_reflex_orbit_closes_at_total_mass_period():
    """Stellar propagation uses the same two-body mass as Kepler's third law.

    Floating-point basis: 32 eps of the apparent reflex semimajor axis for
    period, phase reduction, Kepler solve and sky projection.
    """
    from kinematics import predict_stellar_astrometry
    from orbix.equations import period_to_sma

    period, mass, tp = 400.0, 0.1 * Msun2kg, 21.0
    a = period_to_sma(period, Msun2kg + mass)
    ra, dec = predict_stellar_astrometry(
        jnp.array([tp, tp + period]),
        a,
        0.35,
        0.4,
        0.7,
        jnp.cos(0.6),
        jnp.sin(0.6),
        tp,
        Msun2kg,
        mass,
        10.0,
    )
    scale = a * mass / (Msun2kg + mass) / 10.0
    np.testing.assert_allclose(
        [ra[1], dec[1]], [ra[0], dec[0]], rtol=0, atol=float(ulps(32, scale))
    )


@pytest.mark.parametrize("known_mass", [False, True])
def test_predict_rv_is_receding_positive_stellar_velocity(known_mass):
    """RV is +dz_star/dt in the right-handed (east, north, away) frame.

    The minimum-mass approximation contributes (2/3) Mp/Ms relative.
    Floating-point basis: 64 eps of peak speed for two periods of phase
    reduction, the Kepler solve, its derivative and sky projection.
    """
    from hwoutils.constants import G
    from orbix.equations import period_to_sma
    from orbix.system import FrameOrbit

    period, e, cos_i, node, omega, tp = 430.0, 0.37, 0.43, 1.1, 0.72, 31.0
    mass = (0.1 if known_mass else 1e-4) * Msun2kg
    sin_i = jnp.sqrt(1 - cos_i**2)
    a = period_to_sma(period, Msun2kg + mass)
    orbit = FrameOrbit.from_classical(
        a,
        e,
        jnp.arccos(cos_i),
        node,
        omega,
        0.0,
        mu_AU3_per_d2=G * (Msun2kg + mass),
        epoch_d=tp,
    )

    def stellar_z(time):
        position, _ = orbit.at(time)
        z_planet = position[2]
        return -(mass / (Msun2kg + mass)) * z_planet

    times = jnp.linspace(0.0, 2.0 * period, 31)
    velocity = jax.vmap(jax.grad(stellar_z))(times)
    prediction = predict_rv(
        times,
        period,
        Msun2kg,
        mass * sin_i,
        e,
        jnp.cos(omega),
        jnp.sin(omega),
        tp,
        **({"Mp": mass} if known_mass else {}),
    )
    tolerance = 0.0 if known_mass else (2.0 / 3.0) * (mass / Msun2kg)
    assert bool(jnp.any(velocity > 0))
    assert bool(jnp.any(velocity < 0))
    np.testing.assert_allclose(
        prediction,
        velocity,
        rtol=tolerance,
        atol=float(ulps(64, jnp.max(jnp.abs(velocity)))),
    )


def test_minimum_mass_rv_matches_inclined_true_mass():
    """Fails if minimum-mass RV needs inclination or loses its mass scaling.

    The expected fractional excess is (2/3) Mp/Ms to leading order;
    concavity makes the exact excess smaller. A small planet isolates this
    approximation from the independently propagated true stellar velocity.
    Floating-point basis: 64 eps of peak speed for two periods of mean
    anomaly, the Kepler solve and rotated velocity components.
    """
    from hwoutils.constants import G, two_pi
    from orbix.system import FrameOrbit

    mass = 1e-5 * Msun2kg
    period, inc, peri = 370.0, 0.73, 1.1
    times = jnp.linspace(-130.0, 620.0, 19)
    mu = G * (Msun2kg + mass)
    a = (mu * (period / two_pi) ** 2) ** (1.0 / 3.0)
    orbit = FrameOrbit.from_classical(
        a, 0.4, inc, 0.3, peri, 0.0, mu_AU3_per_d2=mu, epoch_d=17.0
    )
    true_rv = -mass / (Msun2kg + mass) * orbit.at(times)[1][..., 2]
    minimum = predict_rv(
        times,
        period,
        Msun2kg,
        mass * jnp.sin(inc),
        0.4,
        jnp.cos(peri),
        jnp.sin(peri),
        17.0,
    )
    np.testing.assert_allclose(
        minimum,
        true_rv,
        rtol=(2.0 / 3.0) * mass / Msun2kg,
        atol=float(ulps(64, jnp.max(jnp.abs(true_rv)))),
    )


@pytest.mark.medium(reason="4096 vmapped draws of two RV children, 5.5 s cold")
def test_sheet_sum_draw_samples_the_density_it_scores():
    """``SheetSum.draw`` marginalizes the sheet: a fair coin shared by every child.

    Truth source: ``__call__`` is the log of the mean of the two sheets' likelihoods,
    so the data it scores are drawn from the sheet mixture with weights one half
    each. The two sheets predict opposite signs for the velocity change between two
    epochs (about 3 m/s here against 0.01 m/s noise), so the sign of the change
    identifies the sheet.
    Tolerance basis: monte-carlo. 4096 draws, a binomial fraction within
    ``3 sqrt(0.25 / 4096)`` = 0.0234 of one half (a 0.3 percent false-rejection
    rate); the second child must agree with the first in every draw, exactly. The
    children's noise is independent: their difference has standard deviation
    ``sqrt(2) * 0.01`` m/s, to a relative ``4 / sqrt(2 * 8191)`` = 0.031 (four sigma
    of a sample standard deviation from 8192 differences).
    Fails if: every draw keeps the model's own sheet (all changes positive), each
    child flips its own sheet (the children then disagree in about half the draws), or
    the children share one noise key (their difference is then zero).
    """
    mass = 300.0 * Mearth2kg
    mu = G * (Msun2kg + mass)
    orbit = FrameOrbit.from_classical(
        period_to_sma(jnp.array([300.0]), Msun2kg + mass),
        0.2,
        jnp.arccos(0.3),
        0.4,
        0.7,
        0.0,
        mu_AU3_per_d2=mu,
    )
    system = System(
        Body(1.0), Body(jnp.array([mass / Msun2kg])), orbit, SkyFrame(10.0, Epoch())
    )
    data = RadialVelocityData(
        time_d=jnp.array([0.0, 60.0]),
        rv=jnp.zeros(2),
        rv_err=jnp.full(2, 0.01),
        inst_ids=jnp.zeros(2, dtype=int),
        valid=jnp.ones(2, dtype=bool),
        instruments=("rv",),
    )
    term = RadialVelocityTerm(data, jitters=jnp.array([0.0]))
    sheets = SheetSum((term, term), "orbits", 0)
    keys = jax.random.split(jax.random.key(0), 4096)
    drawn = jax.vmap(lambda k: sheets.draw(k, system))(keys)
    first, second = (np.diff(t.data.rv, axis=-1)[:, 0] for t in drawn.terms)
    assert np.all(np.abs(first) > 1.0)
    assert abs(float(np.mean(first > 0)) - 0.5) < 3 * np.sqrt(0.25 / 4096)
    np.testing.assert_array_equal(np.sign(first), np.sign(second))
    difference = np.asarray(drawn.terms[0].data.rv - drawn.terms[1].data.rv)
    spread = float(np.std(difference)) / (np.sqrt(2.0) * 0.01)
    assert abs(spread - 1.0) < 4 / np.sqrt(2 * (difference.size - 1))


class _Ramp(eqx.Module):
    """A star whose velocity is ``slope`` m/s after the epoch day, zero at it."""

    slope: jax.Array

    def star_radial_velocity(self, time_d):
        return self.slope * (time_d > 0)


def _rv_row_term(times, rvs, ids, instruments, jitters=None, max_n=None):
    n = len(times)
    return RadialVelocityTerm(
        RadialVelocityData.pad(
            time_d=times,
            rv=rvs,
            rv_err=[1.0] * n,
            inst_ids=ids,
            instruments=instruments,
            max_n=max_n or n,
        ),
        jitters=None if jitters is None else jnp.asarray(jitters),
    )


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_rv_offset_is_one_latent_per_instrument_and_merged_terms_score_the_union():
    """Two terms of one instrument are refused; their merge scores the joint rows.

    Truth: hand marginal of two unit-error rows observed at (0, 10) m/s whose
    prediction changes by 0 or 10 m/s: ``-0.5 * 50 - 0.5 log 2 - 0.5 log 2 pi + 0.5
    log 2`` is -25.918939 and, with a zero residual contrast, -0.918939 (1e-9). A
    single row alone integrates to 0, which is why a split pair scores 0 at both.
    Fails if: ``marginalizes`` is empty (the split pair is accepted and scores 0), or
    the merge drops the offset sharing.
    """
    from photomancy.infer import Fit

    first = _rv_row_term([0.0], [0.0], [0], ("a",))
    second = _rv_row_term([100.0], [10.0], [0], ("a",))
    assert first.marginalizes == ("rv_offset::a",)
    with pytest.raises(ValueError, match="rv_offset::a"):
        Fit(_Ramp(jnp.asarray(0.0)), {}, (first, second))
    merged = first.merged(second)
    together = _rv_row_term([0.0, 100.0], [0.0, 10.0], [0, 0], ("a",))
    half_log_2pi = 0.5 * np.log(2.0 * np.pi)
    # -0.5 * (spread + log A + sum log var + (N - J) log 2 pi - log N) with N = 2,
    # J = 1, A = 2, unit variances: the spread is 50 or 0.
    for slope, expected in ((0.0, -25.0 - half_log_2pi), (10.0, -half_log_2pi)):
        model = _Ramp(jnp.asarray(slope))
        assert float(first(model) + second(model)) == 0.0
        np.testing.assert_allclose(float(merged(model)), expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(merged(model), together(model), rtol=1e-12)
    assert Fit(_Ramp(jnp.asarray(0.0)), {}, (merged,)).terms == (merged,)


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_rv_merge_joins_instruments_by_name_and_remaps_ids():
    """Terms of ``("a",)`` and ``("b", "a")`` merge to ``("a", "b")`` without mixing.

    Truth: the term built directly from all rows with the ids written by hand
    (1e-12). Capacity of the first term (8) is kept; unequal fixed jitters of a shared
    name, other system paths and a foreign type are refused.
    Fails if: the other term's ``inst_ids`` are not remapped.
    """
    first = _rv_row_term([0.0, 100.0], [0.5, 2.0], [0, 0], ("a",), [0.3], max_n=8)
    other = _rv_row_term(
        [50.0, 100.0, 0.0], [4.0, 1.0, -2.0], [1, 0, 1], ("b", "a"), [0.7, 0.3]
    )
    merged = first.merged(other)
    assert merged.data.instruments == ("a", "b")
    assert merged.data.valid.shape == (8,) and int(merged.data.valid.sum()) == 5
    direct = RadialVelocityTerm(
        RadialVelocityData.pad(
            time_d=[0.0, 100.0, 50.0, 100.0, 0.0],
            rv=[0.5, 2.0, 4.0, 1.0, -2.0],
            rv_err=[1.0] * 5,
            inst_ids=[0, 0, 0, 1, 0],
            instruments=("a", "b"),
            max_n=8,
        ),
        jitters=jnp.array([0.3, 0.7]),
    )
    model = _Ramp(jnp.asarray(3.0))
    np.testing.assert_allclose(merged(model), direct(model), rtol=1e-12)
    clashing = _rv_row_term([0.0], [0.0], [0], ("a",), [0.4])
    with pytest.raises(ValueError, match="fixed jitter"):
        first.merged(clashing)
    with pytest.raises(ValueError, match="system paths"):
        first.merged(dataclasses.replace(first, system_path="other"))
    with pytest.raises(TypeError):
        first.merged(object())


def test_rv_data_refuses_ids_outside_the_instruments_but_not_padded_rows():
    """``n_inst = 1`` with ids ``[0, 1]`` raises; a padded row may hold -1.

    Fails if: the range check is removed (the second row would be silently clamped
    onto a wrong instrument by the segment sums).
    """
    rows = {"time_d": jnp.zeros(2), "rv": jnp.zeros(2), "rv_err": jnp.ones(2)}
    with pytest.raises(ValueError, match="inst_ids"):
        RadialVelocityData(
            **rows,
            inst_ids=jnp.array([0, 1]),
            valid=jnp.ones(2, bool),
            instruments=("a",),
        )

    def host_delivery(ids, valid):
        return RadialVelocityData(
            **{**rows, "rv": np.array([100_000.0, 100_000.002])},
            inst_ids=ids,
            valid=valid,
            instruments=("a",),
        ).rv

    for evaluate in (lambda f: f, jax.jit):
        with pytest.raises(ValueError, match="inst_ids"):
            evaluate(lambda: host_delivery(np.array([0, 1]), [True, True]))()
        np.testing.assert_array_equal(
            evaluate(lambda: host_delivery(np.array([0, -1]), [True, False]))()[0],
            100_000.0,
        )

    padded = RadialVelocityData(
        **rows,
        inst_ids=jnp.array([0, -1]),
        valid=jnp.array([True, False]),
        instruments=("a",),
    )
    assert padded.n_inst == 1
    with pytest.raises(ValueError, match="unique"):
        RadialVelocityData.zeros(2, ("a", "a"))
    for bad in ((1, "1"), (), ["a"]):
        with pytest.raises(ValueError, match="nonempty tuple of unique strings"):
            RadialVelocityData(
                **rows,
                inst_ids=jnp.zeros(2, int),
                valid=jnp.zeros(2, bool),
                instruments=bad,
            )
    with pytest.raises(ValueError, match="nonempty tuple"):
        RadialVelocityData.zeros(2, ())


@pytest.mark.parametrize(
    ("factory", "rows"),
    [
        (
            RadialVelocityData.pad,
            dict(time_d=[0.0], rv=[0.0], rv_err=[1.0], inst_ids=[0]),
        ),
        (RadialVelocityData.zeros, {}),
    ],
)
def test_rv_factories_reject_a_bare_instrument_name(factory, rows):
    """Fails if a factory splits a spectrograph name into one offset per letter."""
    with pytest.raises(TypeError, match="instruments"):
        factory(max_n=1, instruments="HARPS", **rows)


def test_rv_merge_of_traced_terms_names_the_host_requirement():
    """Merging inside ``jit`` raises a ValueError that says both terms must be concrete.

    Fails if: the raw tracer conversion error of ``np.asarray`` surfaces instead.
    """
    first = _rv_row_term([0.0], [0.0], [0], ("a",))
    second = _rv_row_term([100.0], [10.0], [0], ("a",))
    with pytest.raises(ValueError, match="both terms must be concrete"):
        jax.jit(lambda term: first.merged(term))(second)


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_rv_score_is_unchanged_by_a_large_common_offset():
    """Two velocities 0.002 m/s apart score the same with 1e5 m/s added to both.

    Truth: the hand marginal. With N = 2, J = 1, w = 1 / 0.002^2 each, the weighted
    spread about the mean is ``w * 2 * (0.001)^2 = 0.5``, A = 2 w, so the score is
    ``-0.5 * (0.5 + log A + 2 log 0.002^2 + log 2 pi - log 2)`` (1e-7 absolute, above
    the 1e-11 rounding of the shifted inputs). The
    uncentred sums ``C - B^2 / A`` lose it: the shifted score was 5.2957 against 5.0457.
    Fails if: the squared residuals are summed before centring within the instrument.
    """
    from photomancy.measurements import loglike_rv_marginalized

    sigma = 0.002
    w = 1.0 / sigma**2
    expected = -0.5 * (
        0.5 + np.log(2 * w) + 2 * np.log(sigma**2) + np.log(2 * np.pi) - np.log(2.0)
    )
    for shift in (0.0, 1.0e5):
        value = loglike_rv_marginalized(
            jnp.array([shift, shift + 0.002]),
            jnp.zeros(2),
            jnp.full(2, sigma),
            jnp.zeros(2, dtype=int),
            1,
            jnp.zeros(1),
            jnp.ones(2, dtype=bool),
        )
        data = RadialVelocityData.pad(
            time_d=[0.0, 1.0],
            rv=[shift, shift + 0.002],
            rv_err=[sigma] * 2,
            inst_ids=[0, 0],
            instruments=("a",),
            max_n=2,
        )
        actual = RadialVelocityTerm(data)(_Ramp(jnp.asarray(0.0)))
        np.testing.assert_allclose([value, actual], expected, rtol=0, atol=1e-7)


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_a_single_row_instrument_scores_zero_whatever_its_offset(precision):
    """One row of 90000 m/s with a tiny error contributes exactly nothing.

    Truth: an instrument with one valid row has no residual contrast, so with its
    offset marginalized the score is ``-0.5 * (log A + log var - log 1) = 0`` for any
    value or error. The centred sums gave -3.4 for this float64 row.
    Fails if: the residual is not anchored to a row of the instrument before the sums.
    """
    from photomancy.measurements import loglike_rv_marginalized

    err, atol = (
        (5.5979819791232904e-12, 1e-12) if precision == jnp.float64 else (1e-3, 1e-4)
    )
    args = (
        jnp.array([9.0e4], precision),
        jnp.zeros(1, precision),
        jnp.array([err], precision),
        jnp.zeros(1, dtype=int),
        1,
        jnp.zeros(1, precision),
        jnp.ones(1, dtype=bool),
    )
    for call in (
        loglike_rv_marginalized,
        jax.jit(loglike_rv_marginalized, static_argnums=4),
    ):
        np.testing.assert_allclose(float(call(*args)), 0.0, rtol=0, atol=atol)


def test_host_rv_offsets_keep_the_measured_separation_across_batches():
    """Two spectrographs resolve 2 mm/s at opposing 100 km/s zero points.

    Floating-point basis: each offset is the host float64 subtraction rounded
    once to storage precision. IDs are interleaved and the second batch
    supplies another calibration order and an independent velocity origin.
    """
    origins = np.array([1.0e5, -1.0e5])
    ids = np.array([1, 0, 1, 0])
    absolute = origins[ids] + [0.0, 0.0, 0.002, -0.002]
    expected = absolute - origins[ids]
    one = _rv_row_term([0, 0, 1, 1], absolute, ids, ("a", "b"), max_n=5)
    first = _rv_row_term([0, 0], absolute[:2], ids[:2], ("a", "b"))
    second = _rv_row_term([1, 1], absolute[2:], 1 - ids[2:], ("b", "a"))
    merged = first.merged(second)
    for data in (one.data, merged.data, dataclasses.replace(one.data)):
        np.testing.assert_array_equal(
            data.rv_offset[:4], expected.astype(data.rv_offset.dtype)
        )
        assert data.rv_reference == tuple(origins)
        np.testing.assert_array_equal(data.rv[:4], absolute.astype(data.rv.dtype))
    changed = dataclasses.replace(one.data, rv=[*absolute, 0.0])
    np.testing.assert_array_equal(changed.rv_offset, one.data.rv_offset)
    term = dataclasses.replace(
        first, data=dataclasses.replace(first.data, rv_offset=jnp.full(2, 1.0e-12))
    )
    np.testing.assert_array_equal(
        term.merged(first).data.rv_offset[:2], term.data.rv_offset
    )

    # The device batch has an artificial zero reference. The two host contrasts
    # fall on different points from their rounded absolute velocities in float32.
    device = _rv_row_term([0], jnp.array([1e5]), [0], ("a",))
    for deltas in ([0.0078125, 0.015625], [0.002, 0.004]):
        delivery = 1e5 + np.array(deltas)
        host = _rv_row_term([1, 2], delivery, [0, 0], ("a",))
        merged = device.merged(host)
        expected = np.r_[0.0, delivery - 1e5].astype(merged.data.rv_offset.dtype)
        np.testing.assert_array_equal(merged.data.rv_offset, expected)
        np.testing.assert_array_equal(
            jax.jit(lambda data: data.rv_offset)(merged.data), expected
        )


def test_rv_residual_anchors_observation_and_model_separately(precision):
    """A two-m/s contrast survives even when the model has a huge zero point.

    The host observation contrast and a constant model are exact in either
    working dtype. In float64, the score is the unit-variance Gaussian for
    the orthonormal two-row contrast, allowing eight roundings for the
    weighted sum, logs and normalization. Fails if the large model zero point
    is subtracted before the measured separation is formed.
    """
    from photomancy.measurements.radial_velocity import _centered_rv_residual

    class ConstantVelocity(eqx.Module):
        velocity: jax.Array

        def star_radial_velocity(self, *, time_d):
            return jnp.broadcast_to(self.velocity, time_d.shape)

    origin = float(2**54 if precision == np.float64 else 2**25)
    data = _rv_row_term([0, 1], [origin, origin - 2], [0, 0], ("a",)).data
    model = ConstantVelocity(jnp.asarray(origin, dtype=precision))

    def residual(data, model):
        return _centered_rv_residual(
            data.rv_offset,
            model.star_radial_velocity(time_d=data.time_d),
            data.inst_ids,
            data.valid,
            data.n_inst,
        )

    for evaluate in (residual, jax.jit(residual)):
        np.testing.assert_array_equal(evaluate(data, model), [0, -2])
    if precision == np.float64:
        term = RadialVelocityTerm(data)
        expected = -0.5 * (np.log(2 * np.pi) + 2)
        for evaluate in (term, eqx.filter_jit(term)):
            np.testing.assert_allclose(
                evaluate(model), expected, rtol=0, atol=float(ulps(8, abs(expected)))
            )


def test_rv_construction_accepts_traced_rows():
    """A sequence of traced RVs remains differentiable in its working precision."""

    def build(first, second):
        return RadialVelocityData.pad(
            time_d=[0.0, 1.0],
            rv=[first, second],
            rv_err=[1.0, 1.0],
            inst_ids=[0, 0],
            instruments=("a",),
            max_n=3,
        ).rv

    np.testing.assert_array_equal(jax.jit(build)(2.0, 3.0), [2.0, 3.0, 0.0])

    def from_host(ids):
        return RadialVelocityData.pad(
            time_d=[0.0, 1.0],
            rv=[1.0e5, 1.0e5 + 0.002],
            rv_err=[1.0, 1.0],
            inst_ids=ids,
            instruments=("a",),
            max_n=2,
        )

    with pytest.raises(ValueError, match="concrete instrument IDs"):
        jax.jit(from_host)(jnp.zeros(2, dtype=int))


@pytest.mark.float32_only(reason="overflow on narrowing host observations")
def test_rv_refuses_unrepresentable_storage():
    """A finite host velocity or offset cannot become infinite in stored data."""
    largest = float(np.finfo(np.float32).max)
    for velocities in ([largest * 2], [largest, -largest], [largest, 1.5 * largest]):
        n = len(velocities)
        with pytest.raises(ValueError, match="do not fit"):
            _rv_row_term(range(n), velocities, [0] * n, ("a",))

    def build(values):
        n = len(values)
        return _rv_row_term(range(n), values, [0] * n, ("a",)).data.rv

    just_above = np.nextafter(largest, np.inf)
    for evaluate in (lambda f: f, jax.jit):
        np.testing.assert_array_equal(
            evaluate(lambda: build([just_above]))(), [largest]
        )
        with pytest.raises(ValueError, match="offsets overflow"):
            evaluate(lambda: build([1e308, -1e308]))()
        with pytest.raises(ValueError, match="do not fit"):
            evaluate(lambda: build([largest, -largest]))()
        with pytest.raises(ValueError, match="do not fit"):
            evaluate(
                lambda: (
                    RadialVelocityData(
                        time_d=[0],
                        rv_offset=[largest * 2],
                        rv_reference=(0.0,),
                        rv_err=[1.0],
                        inst_ids=[0],
                        valid=[True],
                        instruments=("a",),
                    ).rv_offset
                )
            )()


@pytest.mark.float64_only(reason="narrowing a float64 data object to float32")
def test_absolute_rv_reconstruction_checks_the_host_origin_after_narrowing():
    """An origin kept on the host can exceed the precision of narrowed offsets."""
    from hwoutils.leaves import canonical

    data = _rv_row_term([0], [1.0e40], [0], ("a",)).data
    narrowed = canonical(data, precision="float32")
    np.testing.assert_array_equal(narrowed.rv_offset, [0.0])
    with pytest.raises(ValueError, match="Absolute RVs do not fit float32"):
        _ = narrowed.rv
