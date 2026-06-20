"""Null / imaging orbit fitting: gradient-safety, padding, and end-to-end recovery.

The contrast-curve likelihood interpolates a dMag0 detection limit at the planet's
predicted separation. The grids are padded (trailing sep=0, dmag0=-inf), which made
``jnp.interp`` return a finite value but a nan gradient w.r.t. the separation; these
tests pin the gradient finite, the padding helpers, and a full joint fit.
"""

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbix.equations import period_to_sma  # noqa: E402

from photomancy.backends import LaplaceMixtureBackend  # noqa: E402
from photomancy.orbit.data import (  # noqa: E402
    MAX_CC_PTS,
    MAX_IMG,
    ImagingData,
    NullData,
    RelativeAstromData,
)
from photomancy.orbit.forward import (  # noqa: E402
    predict_photometry,
    predict_relative_astrometry,
)
from photomancy.orbit.inference import build_orbit_logdensity  # noqa: E402
from photomancy.orbit.init import find_init_top_k  # noqa: E402
from photomancy.orbit.laplace import _pad_orbit_data  # noqa: E402
from photomancy.orbit.likelihoods import (  # noqa: E402
    loglike_imaging,
    loglike_null,
)
from photomancy.posterior import MixturePosterior  # noqa: E402

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


def test_pad_orbit_data_handles_non_max_null_container():
    """_pad_orbit_data pads a non-MAX null container instead of crashing."""
    small = NullData(
        epochs=jnp.array([100.0, 500.0]),
        sep_grid=jnp.broadcast_to(jnp.linspace(0.05, 0.5, 40), (2, 40)),
        dmag0_grid=jnp.full((2, 40), 25.0),
        is_valid=jnp.ones(2, dtype=bool),
    )
    _, _, _, _, padded_null, _ = _pad_orbit_data(None, None, None, None, small, None)
    assert padded_null.epochs.shape[0] == MAX_IMG
    assert int(padded_null.is_valid.sum()) == 2


def test_joint_astrom_imaging_fit_recovers_period():
    """A joint astrometry + imaging fit runs end-to-end and recovers the period.

    Imaging detection epochs pass through the photometry likelihood inside a real
    fit, and the TI-seeded Laplace mixture recovers the true period to within 10%.
    """
    msun, dist, t_days = 1.989e30, 10.0, 1096.0
    log_p_range = (float(np.log10(600.0)), float(np.log10(1600.0)))
    e, cos_i, big_omega, little_omega = 0.15, 0.5, 2.3, 0.8
    a_au = float(period_to_sma(t_days, msun))
    tp = -1.5 / (2.0 * jnp.pi / t_days)
    cos_w, sin_w = float(jnp.cos(little_omega)), float(jnp.sin(little_omega))
    lam = 0.3 * (1e-4) ** 2

    rng = np.random.default_rng(0)
    aerr = 5.0e-3
    t_ast = np.sort(rng.uniform(0.0, 3.0 * t_days, 6))
    ra, dec = predict_relative_astrometry(
        jnp.asarray(t_ast), a_au, e, cos_i, big_omega, cos_w, sin_w, tp, msun, dist
    )
    astrom = RelativeAstromData(
        times=jnp.asarray(t_ast),
        ra=jnp.asarray(np.asarray(ra) + rng.normal(0.0, aerr, 6)),
        dec=jnp.asarray(np.asarray(dec) + rng.normal(0.0, aerr, 6)),
        ra_err=jnp.full(6, aerr),
        dec_err=jnp.full(6, aerr),
        corr=jnp.zeros(6),
        planet_id=jnp.zeros(6, dtype=int),
        is_valid=jnp.ones(6, dtype=bool),
    )

    t_det = jnp.asarray(np.sort(rng.uniform(0.0, 3.0 * t_days, 3)))
    _, dmag_true = predict_photometry(
        t_det, a_au, e, cos_i, big_omega, cos_w, sin_w, tp, msun, lam, dist
    )
    npts = 40
    sep = jnp.broadcast_to(jnp.linspace(0.01, 0.6, npts), (3, npts))
    img = ImagingData.from_detections_and_nulls(
        det_epochs=t_det,
        det_dmag_obs=dmag_true,
        det_dmag_err=jnp.full(3, 0.1),
        det_sep_grid=sep,
        det_dmag0_grid=jnp.full((3, npts), 30.0),
        null_epochs=jnp.zeros((0,)),
        null_sep_grid=jnp.zeros((0, npts)),
        null_dmag0_grid=jnp.zeros((0, npts)),
    )

    problem = build_orbit_logdensity(
        msun,
        dist,
        relative_astrom_data=astrom,
        imaging_data=img,
        log_P_range=log_p_range,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_dicts = find_init_top_k(astrom, msun, dist, k=5, log_T_range=log_p_range)
    inits = jnp.stack([problem.init_to_z(d) for d in init_dicts])

    post = LaplaceMixtureBackend(min_eigenvalue=1.0, n_steps=300).run(
        problem.logdensity, inits
    )

    assert isinstance(post, MixturePosterior)
    best = post.means[jnp.argmax(post.log_evidences)]
    phys = problem.to_physical(best)
    assert abs(float(phys["T"]) - t_days) / t_days < 0.10
    assert jnp.isfinite(phys["Lambda"]) and float(phys["Lambda"]) > 0.0
