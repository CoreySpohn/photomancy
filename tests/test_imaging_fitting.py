"""dMag detection and null likelihoods: gradient safety and padding.

The contrast-curve likelihood interpolates a dMag0 detection limit at the planet's
predicted separation. The grids are padded (trailing sep=0, dmag0=-inf), which made
``jnp.interp`` return a finite value but a nan gradient w.r.t. the separation; these
tests pin the gradient finite and the padding helpers.
"""

import jax
import jax.numpy as jnp
from kinematics import (
    predict_photometry,
)

from photomancy.measurements import (
    DMagData,
    loglike_dmag,
)

MSUN_KG = 1.989e30


def _predict(data, cos_i):
    """Predicted (separation, dMag) over a container's epochs at a test orbit."""
    return predict_photometry(
        data.time_d,
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


def test_loglike_dmag_gradient_finite_on_placeholder():
    """The masked-placeholder gradient of the dMag likelihood is finite."""
    img0 = DMagData.zeros(64)
    grad = jax.grad(lambda ci: loglike_dmag(*_predict(img0, ci), img0))(0.5)
    assert jnp.isfinite(grad)


def test_contrast_curve_nulls_convert_and_keep_a_finite_gradient():
    """Fails if the contrast-curve constructor misconverts or breaks the gradient.

    Each contrast limit becomes the dMag limit ``-2.5 log10(contrast)``, every epoch
    is a valid null and the padding is invalid; the geometry gradient of the padded
    curves is finite.
    """
    seps = jnp.broadcast_to(jnp.linspace(0.05, 0.5, 40), (3, 40))
    contrast = jnp.full((3, 40), 1e-9).at[1].set(1e-10)
    data = DMagData.from_contrast_curves(
        jnp.array([100.0, 500.0, 900.0]), seps, contrast, max_n=64
    )
    assert jnp.allclose(data.dmag0_grid[:3, :40], -2.5 * jnp.log10(contrast))
    assert bool(data.valid[:3].all()) and not bool(data.valid[3:].any())
    assert not bool(data.is_detected.any())
    grad = jax.grad(lambda ci: loglike_dmag(*_predict(data, ci), data))(0.5)
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
    nulls = DMagData.from_detections_and_nulls(
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
        return loglike_dmag(jnp.array([separation]), jnp.array([dmag]), nulls)

    assert score(0.2, 20.0) < -10.0
    for separation, dmag in ((0.2, 30.0), (5.0, 15.0), (0.01, 15.0)):
        assert score(separation, dmag) > -0.01, (separation, dmag)
    assert jax.grad(score, argnums=1)(0.2, 25.0) > 0.0


def test_dmag_data_pad_produces_max_sized_valid_container():
    """DMagData.pad pads all per-epoch fields to ``max_n`` with a validity mask."""
    max_n = 64
    padded = DMagData.pad(
        time_d=jnp.array([200.0, 800.0]),
        sep_grid=jnp.broadcast_to(jnp.linspace(0.05, 0.5, 40), (2, 40)),
        dmag0_grid=jnp.full((2, 40), 25.0),
        is_detected=jnp.array([True, False]),
        dmag_obs=jnp.array([22.0, 0.0]),
        dmag_err=jnp.array([0.1, 1.0]),
        max_n=max_n,
    )
    assert padded.time_d.shape == (max_n,)
    assert padded.dmag_obs.shape == (max_n,)
    assert int(padded.valid.sum()) == 2
    assert bool(padded.is_detected[0]) and not bool(padded.is_detected[1])
