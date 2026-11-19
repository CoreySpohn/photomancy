"""Null / imaging likelihoods: gradient safety and padding.

The contrast-curve likelihood interpolates a dMag0 detection limit at the planet's
predicted separation. The grids are padded (trailing sep=0, dmag0=-inf), which made
``jnp.interp`` return a finite value but a nan gradient w.r.t. the separation; these
tests pin the gradient finite and the padding helpers.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from kinematics import (  # noqa: E402
    predict_photometry,
)

from photomancy.measurements import (  # noqa: E402
    MAX_CC_PTS,
    MAX_IMG,
    ImagingData,
    NullData,
    loglike_imaging,
    loglike_null,
)

MSUN_KG = 1.989e30


def _predict(data, cos_i):
    """Predicted (separation, dMag) over a container's epochs at a test orbit."""
    return predict_photometry(
        data.epochs,
        2.0,
        0.1,
        cos_i,
        2.0,
        jnp.cos(0.5),
        jnp.sin(0.5),
        0.0,
        MSUN_KG,
        1e-4,
        10.0,
    )


def test_loglike_null_gradient_finite_on_placeholder():
    """The masked-placeholder gradient (what initialize_model checks) is finite."""
    null0 = NullData.zeros()
    grad = jax.grad(lambda ci: loglike_null(*_predict(null0, ci), null0))(0.5)
    assert jnp.isfinite(grad)


def test_loglike_imaging_gradient_finite_on_placeholder():
    """Same gradient-safety for the unified imaging likelihood placeholder."""
    img0 = ImagingData.zeros()
    grad = jax.grad(lambda ci: loglike_imaging(*_predict(img0, ci), img0))(0.5)
    assert jnp.isfinite(grad)


def test_loglike_null_gradient_finite_on_real_padded_curve():
    """Real padded contrast-curve data also has a finite geometry gradient."""
    seps = jnp.broadcast_to(jnp.linspace(0.05, 0.5, 40), (3, 40))
    contrast = jnp.full((3, 40), 1e-9)
    data = NullData.from_contrast_curves(
        jnp.array([100.0, 500.0, 900.0]), seps, contrast
    )
    grad = jax.grad(lambda ci: loglike_null(*_predict(data, ci), data))(0.5)
    assert jnp.isfinite(grad)


def test_null_epochs_penalize_only_a_planet_bright_enough_to_see():
    """Fails if a null epoch penalizes the wrong planets or pushes the wrong way.

    One null epoch of an all-null dMag container (a 1e-10 contrast limit, dMag 25,
    from 0.2 to 1 arcsec): a planet 5 mag brighter than the limit at 0.2 arcsec is
    penalized by more than 10 nats; a planet 5 mag fainter, or a bright one inside
    the inner or beyond the outer working angle, by less than 0.01. At the limit the
    gradient in dMag is positive (fainter is more likely).
    """
    contrast = jnp.array([[1e-8, 1e-9, 1e-10, 1e-10, 1e-10]])
    nulls = ImagingData.from_detections_and_nulls(
        jnp.zeros(0),
        jnp.zeros(0),
        jnp.zeros(0),
        jnp.zeros((0, 5)),
        jnp.zeros((0, 5)),
        jnp.array([0.0]),
        jnp.array([[0.05, 0.1, 0.2, 0.5, 1.0]]),
        -2.5 * jnp.log10(contrast),
        max_n=1,
        max_pts=5,
    )

    def score(separation, dmag):
        return loglike_imaging(jnp.array([separation]), jnp.array([dmag]), nulls)

    assert score(0.2, 20.0) < -10.0
    for separation, dmag in ((0.2, 30.0), (5.0, 15.0), (0.01, 15.0)):
        assert score(separation, dmag) > -0.01, (separation, dmag)
    assert jax.grad(score, argnums=1)(0.2, 25.0) > 0.0


def test_nulldata_pad_produces_max_sized_valid_container():
    """NullData.pad pads epochs + grids to MAX with a correct validity mask."""
    epochs = jnp.array([100.0, 500.0, 900.0])
    sep = jnp.broadcast_to(jnp.linspace(0.05, 0.5, 40), (3, 40))
    dmag0 = jnp.full((3, 40), 25.0)
    padded = NullData.pad(epochs=epochs, sep_grid=sep, dmag0_grid=dmag0)

    assert padded.epochs.shape == (MAX_IMG,)
    assert padded.sep_grid.shape == (MAX_IMG, MAX_CC_PTS)
    assert int(padded.is_valid.sum()) == 3
    assert bool(padded.is_valid[:3].all())
    assert not bool(padded.is_valid[3:].any())


def test_imagingdata_pad_produces_max_sized_valid_container():
    """ImagingData.pad pads all per-epoch fields to MAX with a validity mask."""
    padded = ImagingData.pad(
        epochs=jnp.array([200.0, 800.0]),
        sep_grid=jnp.broadcast_to(jnp.linspace(0.05, 0.5, 40), (2, 40)),
        dmag0_grid=jnp.full((2, 40), 25.0),
        is_detected=jnp.array([True, False]),
        dmag_obs=jnp.array([22.0, 0.0]),
        dmag_err=jnp.array([0.1, 1.0]),
    )
    assert padded.epochs.shape == (MAX_IMG,)
    assert padded.dmag_obs.shape == (MAX_IMG,)
    assert int(padded.is_valid.sum()) == 2
    assert bool(padded.is_detected[0]) and not bool(padded.is_detected[1])
