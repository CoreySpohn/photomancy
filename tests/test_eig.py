"""Tests for the analytic Fisher EIG (Bayesian experimental design)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg, Rearth2AU, two_pi
from hwoutils.leaves import canonical
from kinematics import predict_photometry, predict_relative_astrometry
from orbix.equations import period_a
from orbix.system import ClassicalChart, OrbitBlock, OrbitLaw
from priorities import LogUniform
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.population import kipping_eccentricity
from skyscapes.scene import FlatStar, Planet, Scene

from photomancy.infer import Fit, Posterior
from photomancy.infer.eig import eig_alias_breaking, eig_detectability, eig_geometric
from photomancy.orbit import eig_astrometry

jax.config.update("jax_enable_x64", True)

DIST_PC = 10.0
MASS_RATIO = 0.1
# a (AU), e, i, W, w, M0 (rad) at day zero, the truth of the one-mode posterior.
ELEMENTS = (1.5, 0.3, 0.8, 1.1, 0.7, 0.4)
LAMBDA = 0.3 * Rearth2AU**2
KEY = jax.random.key(0)


def test_geometric_eig_matches_analytic_gaussian_update():
    """The log-det shrinkage equals the 1-D conjugate-Gaussian information gain."""
    eig, cov_new = eig_geometric(jnp.array([[4.0]]), jnp.array([[1.0]]), 1.0)
    # prec_new = 1/4 + 1 -> cov_new = 0.8; eig = 0.5 * log(4 / 0.8) = 0.5 * log 5
    assert jnp.allclose(eig, 0.5 * jnp.log(5.0))
    assert float(cov_new[0, 0]) < 4.0

    eig_precise, _ = eig_geometric(jnp.array([[4.0]]), jnp.array([[1.0]]), 0.1)
    assert eig_precise > eig  # a more precise measurement is more informative


def test_alias_breaking_eig_zero_when_modes_agree():
    """Zero when modes predict the same observable; positive when they differ."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(
        eig_alias_breaking(weights, jnp.array([[1.0], [1.0]]), 1.0), 0.0
    )
    assert eig_alias_breaking(weights, jnp.array([[1.0], [5.0]]), 1.0) > 0.0


def test_detectability_eig_zero_when_modes_agree():
    """No information when all modes agree on detectability; positive when split."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(eig_detectability(weights, jnp.array([1.0, 1.0])), 0.0)
    assert eig_detectability(weights, jnp.array([1.0, 0.0])) > 0.0


def _one_mode_posterior():
    """Draws placed symmetrically about one orbit, so one mode's mean is the truth.

    A planet of mass ratio 0.1 on the ``ELEMENTS`` orbit at ten parsecs, with
    ``Ag Rp^2 = LAMBDA``. Twelve draws at plus and minus 0.01 along each block
    coordinate have exactly that orbit as their weighted mean.
    """
    a, e, i, W, w, M0 = ELEMENTS
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    model = Scene.from_classical(
        FlatStar(1.0),
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=MASS_RATIO,
        dist_pc=DIST_PC,
        a_AU=a,
        e=e,
        i_rad=i,
        W_rad=W,
        w_rad=w,
        M0_rad=M0,
        order=(0,),
    )
    law = OrbitLaw(
        period_d=LogUniform(300.0, 3000.0), eccentricity=kipping_eccentricity()
    )
    block = OrbitBlock(ClassicalChart(), law, rows=(0,), system="dynamics")
    fit = Fit(canonical(model), {"dynamics.orbits": block}, ())
    truth = block.chart.unconstrain(
        block.chart.from_orbit(model.dynamics.orbits), block.chart_support(model)
    )[0]
    steps = 0.01 * jnp.concatenate([jnp.eye(truth.size), -jnp.eye(truth.size)])
    y = truth + steps
    draws = {
        "dynamics.orbits": {
            name: y[:, k][:, None] for k, name in enumerate(block.coordinate_names)
        }
    }
    n = y.shape[0]
    return Posterior(
        samples=draws,
        log_weights=jnp.full(n, -jnp.log(n)),
        log_target=fit.log_density(draws),
        log_evidence=jnp.array(0.0),
        log_evidence_se=jnp.array(0.0),
        fit=fit,
        info=None,
        kind="monte_carlo",
    )


def _truth_elements():
    """Periapsis-time elements of the truth for the independent kinematics."""
    a, e, i, W, w, M0 = ELEMENTS
    period = period_a(a, G * Msun2kg * (1.0 + MASS_RATIO))
    tp = -M0 * period / two_pi
    return (a, e, np.cos(i), W, np.cos(w), np.sin(w), tp)


@pytest.fixture(scope="module")
def one_mode():
    """The one-mode posterior shared by the contrast-curve tests."""
    return _one_mode_posterior()


def _score(posterior, epochs, curve):
    return eig_astrometry(
        posterior,
        epochs,
        1e-6,
        key=jax.random.key(0),
        n_modes=1,
        contrast_curve=curve,
    )


@pytest.mark.medium(reason="the scene model and the EIG forward compile, 8 s cold")
def test_predictions_are_the_orbit_at_the_total_mass(one_mode):
    """Fails if the EIG forward drops the planet mass, the scale or the epoch axis.

    At the mode mean (the truth) the predicted east and north offsets, separation and
    dMag equal the independent single-orbit kinematics at total mass ``Ms + Mp`` with
    ``Ag Rp^2 = LAMBDA`` read from the scene, to rtol 1e-12, at two epochs.
    """
    epochs = jnp.array([173.0, 411.0])
    curve = (jnp.array([0.0, 10.0]), jnp.array([80.0, 80.0]))
    result = _score(one_mode, epochs, curve)
    args = _truth_elements()
    mp = MASS_RATIO * Msun2kg
    astrometry = predict_relative_astrometry(epochs, *args, Msun2kg, DIST_PC, Mp=mp)
    sep, dmag = predict_photometry(epochs, *args, Msun2kg, LAMBDA, DIST_PC, Mp=mp)
    np.testing.assert_allclose(
        result.eig.predictions[:, 0], jnp.stack(astrometry, axis=-1), rtol=1e-12
    )
    np.testing.assert_allclose(result.separation_arcsec[:, 0], sep, rtol=1e-12)
    np.testing.assert_allclose(result.dmag[:, 0], dmag, rtol=1e-12)
    np.testing.assert_array_equal(result.eig.detectability, jnp.ones((2, 1)))


@pytest.mark.medium(reason="three EIG evaluations on the shared posterior")
def test_the_contrast_curve_decides_each_report(one_mode):
    """Fails if a planet outside the curve scores, or a repeat call keeps a stale curve.

    Periapsis and apoapsis put the planet inside and outside a separation grid placed
    between them: neither epoch is reported and both score zero. On a grid covering
    both, a deep limit reports both epochs with a positive score and a shallow limit,
    on the next call, reports neither.
    """
    period = period_a(ELEMENTS[0], G * Msun2kg * (1.0 + MASS_RATIO))
    tp = _truth_elements()[-1]
    epochs = jnp.array([tp, tp + period / 2])
    wide = (jnp.array([0.0, 10.0]), jnp.array([80.0, 80.0]))
    sep = _score(one_mode, epochs, wide).separation_arcsec[:, 0]
    assert sep[0] < sep[1]

    between = jnp.array([1.05 * sep[0], 0.95 * sep[1]])
    outside = _score(one_mode, epochs, (between, jnp.array([50.0, 50.0])))
    np.testing.assert_array_equal(outside.eig.detectability, jnp.zeros((2, 1)))
    np.testing.assert_array_equal(outside.eig.total, jnp.zeros(2))

    covering = jnp.array([0.5 * sep[0], 2.0 * sep[1]])
    deep, shallow = [
        _score(one_mode, epochs, (covering, jnp.full(2, depth)))
        for depth in (50.0, 1.0)
    ]
    assert jnp.all((1.0 < deep.dmag) & (deep.dmag < 50.0))
    np.testing.assert_array_equal(deep.eig.detectability, jnp.ones((2, 1)))
    assert jnp.all(deep.eig.total > 0.0)
    np.testing.assert_array_equal(shallow.eig.detectability, jnp.zeros((2, 1)))
    np.testing.assert_array_equal(shallow.eig.total, jnp.zeros(2))


@pytest.mark.medium(reason="a scale-and-rotate run, the clustering and the EIG forward")
def test_modes_of_a_run_score_every_candidate():
    """Fails if the dead draws of a run poison the modes or the scores.

    A ``few_epoch`` cloud of a one-row Thiele-Innes fit (the engine tests' fixture)
    holds NaN draws at zero weight and is wide enough for four modes; they score
    twelve candidate epochs, each finite and non-negative, and the epochs differ.
    """
    from test_orbit_engines import _fit, _term

    from photomancy.infer import run
    from photomancy.orbit.backends import few_epoch

    posterior = run(_fit(_term(n_rows=1)), few_epoch(n_draws=2**12), KEY)
    assert np.isneginf(np.asarray(posterior.log_weights)).any()
    epochs = jnp.linspace(0.0, 2000.0, 12)
    result = eig_astrometry(
        posterior, epochs, 0.004**2, key=jax.random.key(1), n_modes=4
    )
    total = np.asarray(result.eig.total)
    assert total.shape == (12,)
    assert np.all(np.isfinite(total) & (total >= 0.0)), total
    assert np.ptp(total) > 0.0
