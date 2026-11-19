"""Custom orbit likelihood terms and independent planet-scale coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from kinematics import predict_photometry, predict_relative_astrometry
from orbit_fit import classical_coordinates, scale_problem

from photomancy import NullTerm
from photomancy.orbit.data import NullData, RelativeAstromData
from photomancy.orbit.likelihoods import loglike_null, loglike_relative_astrom

jax.config.update("jax_enable_x64", True)

DIST_PC = 10.0
PERIOD_D = 900.0
ORBIT = classical_coordinates(1.6, 0.2, 0.3, 0.8, 0.5, 40.0, PERIOD_D)


def _rows(scene):
    """One planet's ``(a, e, cos_i, W, cos_w, sin_w, tp)`` from a scene."""
    orbit = jax.tree.map(lambda leaf: leaf[0], scene.dynamics.orbits)
    elements = orbit.to_periapsis_elements()
    return tuple(
        elements[k] for k in ("a_AU", "e", "cos_i", "W_rad", "cos_w", "sin_w", "tp_d")
    )


def test_physical_scale_coordinates_decode_to_radius_albedo_and_area():
    """Fails for a dropped radius or albedo decode in ``to_physical``.

    The sampled log radius and log albedo of three points decode to
    ``Rp``, ``Ag`` and ``Lambda = Ag Rp**2`` to 1e-12 relative, and the flat chart
    vector round-trips (the mass round-trip test does not cover these scalars).
    """
    problem = scale_problem(
        (), ORBIT, star_mass_kg=Msun2kg, dist_pc=DIST_PC, period_range_d=(600, 1600)
    )
    log_radius = jnp.array([-9.5, -7.9, -6.8])
    log_albedo = jnp.array([-1.2, -0.4, -2.3])
    z = jax.vmap(
        lambda r, a: problem.flatten({"orbit": ORBIT, "radius_AU": r, "albedo": a})
    )(log_radius, log_albedo)
    physical = jax.vmap(problem.to_physical)(z)
    np.testing.assert_allclose(jnp.log(physical["Rp"]), log_radius, rtol=1e-12)
    np.testing.assert_allclose(jnp.log(physical["Ag"]), log_albedo, rtol=1e-12)
    np.testing.assert_allclose(
        physical["Lambda"], physical["Ag"] * physical["Rp"] ** 2, rtol=1e-12
    )
    back = jax.vmap(problem.unflatten)(z)
    np.testing.assert_allclose(back["radius_AU"], log_radius, rtol=1e-12)
    np.testing.assert_allclose(back["albedo"], log_albedo, rtol=1e-12)


def _custom_null(data):
    """Null term from the independent kinematics and the flux-space null law."""

    def term(scene, *, radius_AU, albedo):
        rows = _rows(scene)
        alpha, dmag = predict_photometry(
            data.epochs, *rows, Msun2kg, albedo * radius_AU**2, DIST_PC
        )
        return loglike_null(alpha, dmag, data), None

    return term


@pytest.mark.medium(reason="two chart densities with a null curve, 9 s cold")
def test_custom_null_matches_builtin_logdensity():
    """Fails if a caller-defined term, its scalar bindings or the prior differ.

    The built-in null term reads the scene's own albedo and radius; the custom
    term reads the same coordinates through ``term_inputs`` and predicts with the
    independent perifocal kinematics. At eight random points the two joint
    densities agree to 1e-10 (floating-point floor of the sums).
    """
    data = NullData.pad(
        epochs=jnp.array([0.0, 100.0]),
        sep_grid=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
        dmag0_grid=jnp.full((2, 2), 23.0),
    )
    kwargs = dict(star_mass_kg=Msun2kg, dist_pc=DIST_PC)
    builtin = scale_problem((NullTerm(data),), ORBIT, **kwargs)
    custom = scale_problem(
        (_custom_null(data),),
        ORBIT,
        term_inputs={0: ("radius_AU", "albedo")},
        **kwargs,
    )
    assert builtin.position_template.keys() == custom.position_template.keys()
    z = builtin.flatten(builtin.position_template)
    points = z + 0.3 * jax.random.normal(jax.random.key(0), (8, z.size))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(custom.logdensity))(points),
        jax.jit(jax.vmap(builtin.logdensity))(points),
        rtol=0.0,
        atol=1e-10,
    )


def _custom_astrom(data, offsets):
    """Average sky positions at three exposure nodes before evaluating the law."""

    def term(scene):
        ra, dec = predict_relative_astrometry(
            (data.times[:, None] + offsets).ravel(),
            *_rows(scene),
            Msun2kg,
            DIST_PC,
        )
        return loglike_relative_astrom(
            ra.reshape(-1, 3).mean(axis=1), dec.reshape(-1, 3).mean(axis=1), data
        ), None

    return term


@pytest.mark.medium(reason="three chart densities with an exposure-averaged law")
def test_custom_term_is_prior_plus_custom_and_closures_are_isolated():
    """Fails if a custom term adds a hidden factor or closures share stale data.

    The joint with one custom term equals the data-free joint plus the term's own
    value to 1e-12; a second term over shifted exposure nodes gives the density
    for its own nodes and differs from the first by more than 1e-6.
    """
    data = RelativeAstromData(
        times=jnp.array([100.0]),
        ra=jnp.array([0.03]),
        dec=jnp.array([-0.02]),
        ra_err=jnp.array([0.02]),
        dec_err=jnp.array([0.02]),
        corr=jnp.zeros(1),
        planet_id=jnp.zeros(1, dtype=int),
        is_valid=jnp.ones(1, dtype=bool),
    )
    nodes = jnp.array([-10.0, 0.0, 10.0])
    kwargs = dict(star_mass_kg=Msun2kg, dist_pc=DIST_PC)
    prior = scale_problem((), ORBIT, **kwargs)
    first = scale_problem((_custom_astrom(data, nodes),), ORBIT, **kwargs)
    second = scale_problem((_custom_astrom(data, nodes + 50.0),), ORBIT, **kwargs)
    z = prior.flatten(prior.position_template)
    z = z + 0.1 * jax.random.normal(jax.random.key(0), z.shape)
    value, aux = first.density.value_and_aux(first.unflatten(z))
    np.testing.assert_allclose(
        first.logdensity(z), prior.logdensity(z) + aux["term_values"][0], atol=1e-12
    )
    value_second, aux_second = second.density.value_and_aux(second.unflatten(z))
    np.testing.assert_allclose(
        value_second, prior.logdensity(z) + aux_second["term_values"][0], atol=1e-12
    )
    assert abs(float(value_second - value)) > 1e-6
