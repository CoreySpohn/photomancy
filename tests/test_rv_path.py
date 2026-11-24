"""Regression coverage for RV units, masks, priors, and orbital dynamics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import AU2m, G, Mearth2kg, Msun2kg, d2s
from hwoutils.epoch import Epoch
from kinematics import predict_rv
from orbix.equations import period_to_sma
from orbix.system import Body, FrameOrbit, SkyFrame, System
from term_utils import both

from photomancy import RadialVelocityTerm
from photomancy.measurements import RadialVelocityData
from photomancy.orbit.sheets import SheetSum

jax.config.update("jax_enable_x64", True)


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
        n_inst=1,
    )
    term = RadialVelocityTerm(rv, jitters=jnp.array([0.01]))

    def likelihood(scale):
        return float(term(_circular_system(periods, masses * scale * Mearth2kg)))

    assert likelihood(1.0) > likelihood(0.5)
    assert likelihood(1.0) > likelihood(2.0)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
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
    """Known mass gives the exact two-body K at Mp/Ms=0.1 (Lovis/Fischer Eq. 14)."""
    from hwoutils.constants import G, two_pi

    period, eccentricity, cos_i = 400.0, 0.35, 0.4
    mass = 0.1 * Msun2kg
    minimum_mass = mass * jnp.sqrt(1.0 - cos_i**2)
    expected_K = (
        (two_pi * G / period) ** (1.0 / 3.0)
        * minimum_mass
        / (Msun2kg + mass) ** (2.0 / 3.0)
        / jnp.sqrt(1.0 - eccentricity**2)
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
    assert float(measured_K) == pytest.approx(float(expected_K), rel=1e-12, abs=0)


def test_predict_rv_matches_orbit_radial_velocity():
    """Compare minimum-mass RV to the orbit's stellar velocity in m/s.

    Fails if true mass, inclination, sign or units differ between the paths.
    The x64 tolerance (rtol 1e-12, atol 1e-10 m/s) allows solver and conversion
    rounding for a finite-mass, eccentric, inclined orbit.
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
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_rv_term_uses_total_mass():
    """The RV term uses the planet's true mass in both a and K.

    Fails if the semi-major axis or the semi-amplitude drops the planet mass: a
    0.1 Msun planet on a 400 d circular, inclined orbit reproduces the exact
    two-body velocity curve, so the marginalized log likelihood of data equal to
    that curve is the zero-residual value to 1e-10.
    """
    from hwoutils.constants import two_pi

    from photomancy.measurements import loglike_rv_marginalized

    times = jnp.array([0.0, 100.0, 200.0, 350.0])
    periods = jnp.array([400.0])
    masses = jnp.array([0.1]) * Msun2kg
    cos_i = 0.4
    K = (two_pi * G / periods) ** (1.0 / 3.0) * masses * jnp.sqrt(1 - cos_i**2)
    K /= (Msun2kg + masses) ** (2.0 / 3.0)
    velocities = -jnp.sum(
        K[:, None] * jnp.cos(two_pi * times / periods[:, None]), axis=0
    )
    data = RadialVelocityData(
        time_d=times,
        rv=velocities * (AU2m / d2s),
        rv_err=jnp.full(4, 0.5),
        inst_ids=jnp.zeros(4, dtype=int),
        valid=jnp.ones(4, dtype=bool),
        n_inst=1,
    )
    term = RadialVelocityTerm(data, jitters=jnp.array([0.01]))
    value, prediction = both(term, _circular_system(periods, masses, cos_i))
    np.testing.assert_allclose(prediction, data.rv, rtol=1e-9, atol=1e-9)
    expected = loglike_rv_marginalized(
        data.rv,
        data.rv,
        data.rv_err,
        data.inst_ids,
        1,
        jnp.array([0.01]),
        data.valid,
    )
    assert float(value) == pytest.approx(float(expected), abs=1e-10)


def test_reflex_orbit_closes_at_total_mass_period():
    """Stellar propagation uses the same two-body mass as Kepler's third law."""
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
    assert float(ra[1]) == pytest.approx(float(ra[0]), rel=1e-12, abs=1e-15)
    assert float(dec[1]) == pytest.approx(float(dec[0]), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("known_mass", [False, True])
def test_predict_rv_is_receding_positive_stellar_velocity(known_mass):
    """RV is +dz_star/dt in the right-handed (east, north, away) frame."""
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
    tolerance = 1e-12 if known_mass else (2.0 / 3.0) * (mass / Msun2kg)
    assert bool(jnp.any(velocity > 0))
    assert bool(jnp.any(velocity < 0))
    assert jnp.allclose(prediction, velocity, rtol=tolerance, atol=0.0)


def test_minimum_mass_rv_matches_inclined_true_mass():
    """Fails if minimum-mass RV needs inclination or loses its mass scaling.

    The expected fractional excess is (2/3) Mp/Ms to leading order;
    concavity makes the exact excess smaller. A small planet isolates this
    approximation from the independently propagated true stellar velocity.
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
    np.testing.assert_allclose(minimum, true_rv, rtol=(2.0 / 3.0) * mass / Msun2kg)


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
        n_inst=1,
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
