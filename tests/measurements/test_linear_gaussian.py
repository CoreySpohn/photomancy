"""``linear_gaussian`` of relative astrometry: a Gaussian in the sky constants."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from golden_systems import model_at, systems
from hwoutils.leaves import canonical
from hwoutils.model_paths import assign
from orbix.system import (
    ClassicalChart,
    OrbitBlock,
    OrbitLaw,
    System,
    ThieleInnesBlock,
    linear_gaussian_conditional,
)
from orbix.system.core import Body, SkyFrame

from photomancy.infer import (
    AbstractLinearGaussianTerm,
    AbstractTerm,
    Fit,
    LinearGaussian,
)
from photomancy.measurements import (
    AstrometryData,
    RelativeAstrometryTerm,
    StellarAstrometryTerm,
)

TIMES = np.array([31.0, 120.0, 290.0, 470.0, 650.0])
OPEN_LAW = OrbitLaw(priorities.HalfNormal(0.3), a_AU=priorities.LogNormal(0.0, 1.0))
CONSTANTS = ("A_AU", "B_AU", "F_AU", "G_AU")


def _system(k=1):
    return System.from_classical(
        Body(jnp.asarray(1.0)),
        Body(jnp.array([3e-4, 1e-3][:k])),
        SkyFrame(jnp.asarray(8.0)),
        a_AU=jnp.array([1.2, 3.1][:k]),
        e=jnp.array([0.2, 0.4][:k]),
        i_rad=jnp.array([0.9, 1.2][:k]),
        W_rad=jnp.array([0.5, 2.0][:k]),
        w_rad=jnp.array([1.0, 0.3][:k]),
        M0_rad=jnp.array([0.7, 2.2][:k]),
    )


def _relative_data(planet):
    return AstrometryData.pad(
        time_d=TIMES,
        east_arcsec=[0.10, -0.05, 0.12, 0.2, -0.1],
        north_arcsec=[0.02, 0.11, -0.07, 0.15, 0.05],
        east_err_arcsec=[0.004, 0.005, 0.003, 0.006, 0.004],
        north_err_arcsec=[0.005, 0.004, 0.004, 0.003, 0.006],
        corr=[0.2, -0.3, 0.0, 0.5, -0.1],
        planet=planet,
        max_n=7,
    )


def _gaussian(form, constants):
    """The Gaussian log density of the form's records, written out here."""
    resid = form.target - form.design @ constants
    determinant = jnp.linalg.det(form.precision)
    active = determinant > 0
    quad = jnp.einsum("tk,tkl,tl->t", resid, form.precision, resid)
    safe = jnp.where(active, determinant, 1.0)
    return float(
        jnp.sum(
            jnp.where(
                active, -jnp.log(2 * jnp.pi) + 0.5 * jnp.log(safe) - 0.5 * quad, 0
            )
        )
    )


def _through_the_block(model, system, constants):
    """The model with sky constants written the way a Thiele-Innes block writes them."""
    block = ThieleInnesBlock(
        OPEN_LAW, priorities.Uniform(0.5, 2.0), rows=(0,), system=system
    )
    x = block.inverse(model)
    x = {**x, **{n: jnp.asarray(constants[k])[None] for k, n in enumerate(CONSTANTS)}}
    for path, value in block.forward(x, model).items():
        model = assign(model, path, value)
    return model


def _assert_pinned(term, model, system, scale_AU):
    """The term equals the form's Gaussian at constants written by the block.

    Constants of scale ``scale_AU`` (seed 0) go in through ``ThieleInnesBlock.forward``,
    which also rewrites the star mass and the mean motion, as a fit does. The form is
    built once at the base model.
    """
    form = term.linear_gaussian(model, planet=0)
    for key in jax.random.split(jax.random.key(0), 3):
        c = scale_AU * jax.random.normal(key, (4,))
        actual = float(term(_through_the_block(model, system, c)))
        np.testing.assert_allclose(_gaussian(form, c), actual, rtol=1e-11, atol=1e-9)


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_relative_form_is_the_term_on_the_golden_system():
    """Fails if the design, target or precision units of the form are wrong.

    Golden system g1 (a scene, one planet, four records and padding): the Gaussian of
    the form at three random sets of constants, written through the Thiele-Innes
    block, reproduces the term value to rtol 1e-11. Both sides sum the same Gaussian
    rows, so rounding only; the form is in arcsec with no log scale to add.
    """
    _, scene, entry, terms = next(s for s in systems() if s[0] == "g1")
    _assert_pinned(terms[0], model_at(scene, entry(0)), "dynamics", 0.5)


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_relative_form_on_a_bare_system_with_records_of_its_row():
    """The form scores float32 astrometry in float64 and rejects another planet."""
    system = _system()
    term = RelativeAstrometryTerm(
        canonical(_relative_data([0] * 5), precision="float32")
    )
    _assert_pinned(term, system, "", 1.0)
    mixed = RelativeAstrometryTerm(_relative_data([0, 1, 0, 0, 0]))
    with pytest.raises(Exception, match="another planet"):
        mixed.linear_gaussian(system, planet=0)


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_a_fit_on_a_bare_system_joins_the_block_path():
    """Fails if an orbit block on a bare system (``system=""``) cannot be fit.

    The block's paths come from ``join``, so the empty root gives ``orbits`` rather
    than ``.orbits``; the fit draws it and evaluates a finite density.
    """
    block = OrbitBlock(ClassicalChart(), OPEN_LAW, rows=(0,), system="")
    term = RelativeAstrometryTerm(_relative_data([0] * 5))
    fit = Fit(_system(), {"orbits": block}, (term,))
    x = fit.sample_prior(jax.random.key(0), (3,))
    assert x["orbits"][block.coordinate_names[0]].shape == (3, 1)
    assert np.all(np.isfinite(fit.log_density(x)))


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_relative_form_refuses_a_system_with_more_than_one_planet_row():
    """Fails if a two-planet form is built: it errs by 0.2 nat at a 10 percent move.

    With two rows the other row's offset, its mass weight and its mean motion move
    with the constants under a block that writes the star mass, which a form at fixed
    values cannot follow.
    """
    term = RelativeAstrometryTerm(_relative_data([1] * 5))
    with pytest.raises(Exception, match="more than one planet row"):
        term.linear_gaussian(_system(k=2), planet=1)


@pytest.mark.float64_only(reason="measurement log densities require float64")
def test_the_linear_capability_is_a_type_and_the_stellar_term_does_not_have_it():
    """Fails if a consumer must probe attributes, or the stellar term claims a form.

    The stellar reflex depends on the star mass and on every row's mass, which move
    with the constants under ``ThieleInnesBlock``, so its form is not linear in them.
    """
    relative = RelativeAstrometryTerm(_relative_data([0] * 5))
    stellar = StellarAstrometryTerm(
        AstrometryData.pad(
            time_d=TIMES,
            east_arcsec=[1e-4] * 5,
            north_arcsec=[1e-4] * 5,
            east_err_arcsec=[5e-5] * 5,
            north_err_arcsec=[8e-5] * 5,
            corr=[0.0] * 5,
            max_n=7,
        )
    )
    assert isinstance(relative, AbstractLinearGaussianTerm)
    assert isinstance(stellar, AbstractTerm)
    assert not isinstance(stellar, AbstractLinearGaussianTerm)
    assert not hasattr(stellar, "linear_gaussian")
    form = relative.linear_gaussian(_system(), planet=0)
    assert isinstance(form, LinearGaussian)
    assert {f.name for f in dataclasses.fields(form)} == {
        "design",
        "target",
        "precision",
    }


@pytest.mark.float64_only(reason="float64 conditional evidence")
def test_conditional_marginal_is_the_density_of_the_data_in_arcsec():
    """Fails if the form is not in the data's units (the evidence is off by dist^2).

    ``linear_gaussian_conditional`` on the form with a near-flat prior gives the
    Gaussian density of the data at the posterior mean plus the prior density there,
    minus half log det of the posterior information.
    """
    _, scene, entry, terms = next(s for s in systems() if s[0] == "g1")
    form = terms[0].linear_gaussian(model_at(scene, entry(0)))
    prior = 1e-12
    mean, chol, log_z = linear_gaussian_conditional(
        form.design, form.target, form.precision, prior
    )
    expected = (
        _gaussian(form, mean)
        + 0.5 * 4 * jnp.log(prior)
        - jnp.sum(jnp.log(jnp.diag(chol)))
        - 0.5 * prior * jnp.sum(mean**2)
    )
    np.testing.assert_allclose(float(log_z), float(expected), rtol=1e-9)
