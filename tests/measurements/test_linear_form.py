"""``linear_form`` of the astrometry terms: a quadratic in the sky constants."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden_systems import model_at, systems
from orbix.system import System, ThieleInnesChart
from orbix.system.core import Body, SkyFrame

from photomancy.measurements import (
    RelativeAstromData,
    RelativeAstrometryTerm,
    StellarAstromData,
    StellarAstrometryTerm,
)

jax.config.update("jax_enable_x64", True)

CHART = ThieleInnesChart()
TIMES = np.array([31.0, 120.0, 290.0, 470.0, 650.0])


def _with_constants(orbits, row, constants):
    """Orbits whose row ``row`` keeps its shape and takes the sky constants."""
    shape = jax.tree.map(lambda a: a[row], orbits)
    coords = CHART.from_orbit(shape).at[3:].set(constants)
    new = CHART.to_orbit(
        coords, mu_AU3_per_d2=shape.mu_AU3_per_d2, epoch_d=shape.epoch_d
    )
    return jax.tree.map(lambda a, b: a.at[row].set(b), orbits, new)


def _two_planet_system():
    return System.from_classical(
        Body(jnp.asarray(1.0)),
        Body(jnp.array([3e-4, 1e-3])),
        SkyFrame(jnp.asarray(8.0)),
        a_AU=jnp.array([1.2, 3.1]),
        e=jnp.array([0.2, 0.4]),
        i_rad=jnp.array([0.9, 1.2]),
        W_rad=jnp.array([0.5, 2.0]),
        w_rad=jnp.array([1.0, 0.3]),
        M0_rad=jnp.array([0.7, 2.2]),
    )


def _relative_data(planet_id):
    return RelativeAstromData.pad(
        times=TIMES,
        ra=[0.10, -0.05, 0.12, 0.2, -0.1],
        dec=[0.02, 0.11, -0.07, 0.15, 0.05],
        ra_err=[0.004, 0.005, 0.003, 0.006, 0.004],
        dec_err=[0.005, 0.004, 0.004, 0.003, 0.006],
        corr=[0.2, -0.3, 0.0, 0.5, -0.1],
        planet_id=planet_id,
        max_n=7,
    )


def _stellar_data():
    return StellarAstromData.pad(
        times=TIMES,
        ra=[1e-4, -2e-4, 3e-4, 2e-4, -1e-4],
        dec=[-1e-4, 1e-4, 2e-4, -2e-4, 1e-4],
        ra_err=[5e-5] * 5,
        dec_err=[8e-5] * 5,
        corr=[0.2, -0.3, 0.1, 0.0, 0.4],
        max_n=7,
    )


def _assert_quadratic(term, model, replace, row, scale_AU):
    """The term at random constants equals the form's quadratic at those constants.

    Draw constants from a Gaussian of scale ``scale_AU`` (seed 0), write them into the
    model through the chart and compare to ``form.log_likelihood``. The form is built
    once at the base model: the nonlinear shape does not change with the constants.
    """
    form = term.linear_form(model, row=row)
    keys = jax.random.split(jax.random.key(0), 3)
    for key in keys:
        c = scale_AU * jax.random.normal(key, (4,))
        actual = float(term(replace(model, c)))
        np.testing.assert_allclose(
            float(form.log_likelihood(c)), actual, rtol=1e-11, atol=1e-9
        )


def test_relative_form_is_the_term_on_the_golden_system():
    """Fails if the design, target, precision or arcsec scale of the form is wrong.

    Golden system g1 (a scene, one planet, four records and padding): the quadratic
    at three random sets of constants reproduces the term value to rtol 1e-11. The
    two sides sum the same Gaussian rows in different orders, so rounding only.
    """
    _, scene, entry, terms = next(s for s in systems() if s[0] == "g1")
    model = model_at(scene, entry(0))
    term = terms[0]

    def replace(model, c):
        orbits = _with_constants(model.dynamics.orbits, 0, c)
        return eqx.tree_at(lambda m: m.dynamics.orbits, model, orbits)

    _assert_quadratic(term, model, replace, 0, 0.5)


def test_relative_form_picks_one_row_of_a_chain():
    """Fails if the row's own shape is not used, or other planets' records slip in.

    Two planets; the records belong to planet 1. Constants of row 1 are varied while
    row 0 stays. A record of planet 0 must fail at run time.
    """
    system = _two_planet_system()
    term = RelativeAstrometryTerm(_relative_data([1] * 5))

    def replace(model, c):
        return eqx.tree_at(
            lambda m: m.orbits, model, _with_constants(model.orbits, 1, c)
        )

    _assert_quadratic(term, system, replace, 1, 1.0)
    mixed = RelativeAstrometryTerm(_relative_data([1, 0, 1, 1, 1]))
    with pytest.raises(Exception, match="another planet"):
        mixed.linear_form(system, row=1)


def test_stellar_form_holds_the_other_planets():
    """Fails if the reflex weight or the other planets' reflex is wrong in the form.

    Two planets with masses 3e-4 and 1e-3 solar: the star's reflex depends on both.
    Constants of row 1 vary with row 0 held, and the form must match the term.
    Then the form of row 0 is checked the same way.
    """
    system = _two_planet_system()
    term = StellarAstrometryTerm(_stellar_data())
    for row in (0, 1):

        def replace(model, c, row=row):
            return eqx.tree_at(
                lambda m: m.orbits, model, _with_constants(model.orbits, row, c)
            )

        _assert_quadratic(term, system, replace, row, 1.0)


def test_stellar_form_on_the_free_mass_golden_system():
    """Fails if the form misreads a scene's planet mass or distance (golden g2)."""
    _, scene, entry, terms = next(s for s in systems() if s[0] == "g2")
    model = model_at(scene, entry(1))
    term = terms[0]

    def replace(model, c):
        orbits = _with_constants(model.dynamics.orbits, 0, c)
        return eqx.tree_at(lambda m: m.dynamics.orbits, model, orbits)

    _assert_quadratic(term, model, replace, 0, 0.5)


def test_conditional_adds_the_arcsec_scale():
    """Fails if ``conditional`` drops ``log_scale``: its evidence is off by dist^2.

    With a near-flat prior the evidence is the likelihood at the posterior mean plus
    the prior density there, minus half log det of the posterior information.
    """
    _, scene, entry, terms = next(s for s in systems() if s[0] == "g1")
    form = terms[0].linear_form(model_at(scene, entry(0)))
    prior = 1e-12
    mean, chol, log_z = form.conditional(prior)
    expected = (
        form.log_likelihood(mean)
        + 0.5 * 4 * jnp.log(prior)
        - jnp.sum(jnp.log(jnp.diag(chol)))
        - 0.5 * prior * jnp.sum(mean**2)
    )
    np.testing.assert_allclose(float(log_z), float(expected), rtol=1e-9)
