"""Tests for photomancy.orbit.laplace -- MAP + Laplace approximation."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from orbix.equations import period_to_sma  # noqa: E402

from photomancy.orbit.data import AstromData  # noqa: E402
from photomancy.orbit.forward import predict_astrometry  # noqa: E402
from photomancy.orbit.init import find_init  # noqa: E402
from photomancy.orbit.laplace import (  # noqa: E402
    LaplaceResult,
    map_laplace_fit,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Msun2kg = 1.989e30
DIST_PC = 10.0
TRUE_T = 1096.0
LOG_P_RANGE = (float(jnp.log10(600.0)), float(jnp.log10(1600.0)))
ASTROM_ERR = 5.0e-3


def _make_astrom(n_obs, seed=42):
    """Generate synthetic astrometry data."""
    a_true = float(period_to_sma(TRUE_T, Msun2kg))
    tp_true = -1.5 / (2.0 * jnp.pi / TRUE_T)
    np.random.seed(seed)
    times = np.sort(np.random.uniform(0, 3 * TRUE_T, 20))[:n_obs]
    ra_true, dec_true = predict_astrometry(
        jnp.array(times),
        a_true,
        0.15,
        0.5,
        2.3,
        jnp.cos(0.8),
        jnp.sin(0.8),
        tp_true,
        Msun2kg,
        DIST_PC,
    )
    ra = jnp.array(np.array(ra_true) + np.random.randn(n_obs) * ASTROM_ERR)
    dec = jnp.array(np.array(dec_true) + np.random.randn(n_obs) * ASTROM_ERR)
    return AstromData(
        times=jnp.array(times),
        ra=ra,
        dec=dec,
        ra_err=jnp.full(n_obs, ASTROM_ERR),
        dec_err=jnp.full(n_obs, ASTROM_ERR),
        corr=jnp.zeros(n_obs),
        planet_id=jnp.zeros(n_obs, dtype=int),
        is_valid=jnp.ones(n_obs, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_map_laplace_fit_smoke():
    """map_laplace_fit returns a LaplaceResult with correct fields."""
    astrom = _make_astrom(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
        result = map_laplace_fit(
            Msun2kg,
            DIST_PC,
            astrom_data=astrom,
            log_P_range=LOG_P_RANGE,
            init_vals=init_vals,
            n_steps=200,
        )
    assert isinstance(result, LaplaceResult)
    assert result.z_map.shape[0] == result.n_params
    assert result.covariance.shape == (result.n_params, result.n_params)
    assert result.cholesky.shape == (result.n_params, result.n_params)
    assert len(result.param_names) > 0


def test_map_converges():
    """MAP on n=5 data recovers T within 10% of truth."""
    astrom = _make_astrom(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
        result = map_laplace_fit(
            Msun2kg,
            DIST_PC,
            astrom_data=astrom,
            log_P_range=LOG_P_RANGE,
            init_vals=init_vals,
        )
    samples = result.sample(jax.random.PRNGKey(0), n=500)
    T_median = float(jnp.median(samples["T"]))
    assert abs(T_median - TRUE_T) / TRUE_T < 0.10, (
        f"T_median={T_median:.1f} vs truth={TRUE_T:.1f}"
    )


def test_covariance_positive_definite():
    """Regularised covariance is positive-definite for n=1,2,3."""
    for n_obs in [1, 2, 3]:
        astrom = _make_astrom(n_obs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
            result = map_laplace_fit(
                Msun2kg,
                DIST_PC,
                astrom_data=astrom,
                log_P_range=LOG_P_RANGE,
                init_vals=init_vals,
                n_steps=100,
            )
        eigvals = jnp.linalg.eigvalsh(result.covariance)
        assert jnp.all(eigvals > 0), f"n={n_obs}: eigenvalues={eigvals}"


def test_samples_shape():
    """sample() returns arrays with the correct number of samples."""
    astrom = _make_astrom(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
        result = map_laplace_fit(
            Msun2kg,
            DIST_PC,
            astrom_data=astrom,
            log_P_range=LOG_P_RANGE,
            init_vals=init_vals,
            n_steps=100,
        )
    samples = result.sample(jax.random.PRNGKey(1), n=100)
    # Physical params should include T at minimum
    assert "T" in samples
    assert samples["T"].shape == (100,)


def test_log_prob_finite():
    """log_prob() returns finite values at the MAP point."""
    astrom = _make_astrom(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
        result = map_laplace_fit(
            Msun2kg,
            DIST_PC,
            astrom_data=astrom,
            log_P_range=LOG_P_RANGE,
            init_vals=init_vals,
            n_steps=100,
        )
    lp = result.log_prob(result.z_map)
    assert jnp.isfinite(lp), f"log_prob at MAP = {lp}"
    # log_prob at MAP should be the maximum (zero Mahalanobis distance)
    # so only the normalisation constant remains
    d = result.n_params
    expected = -0.5 * d * jnp.log(2 * jnp.pi) - jnp.sum(
        jnp.log(jnp.diag(result.cholesky))
    )
    assert jnp.allclose(lp, expected, atol=1e-6)
