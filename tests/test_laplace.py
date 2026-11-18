"""Tests for photomancy.orbit.laplace -- MAP + Laplace approximation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg, two_pi
from hwoutils.epoch import Epoch
from numpyro.infer.util import log_density
from orbix.equations import period_to_sma
from orbix.system import Body, ClassicalChart, SkyFrame, System
from scipy.optimize import minimize
from sky_reference import sky_position

from photomancy import RelativeAstrometryTerm
from photomancy.orbit.data import OrbitData, RelativeAstromData
from photomancy.orbit.diagnostics import sample_physical
from photomancy.orbit.inference import (
    ChartOrbitProblem,
    build_chart_logdensity,
    build_numpyro_logdensity,
)
from photomancy.orbit.init import find_init
from photomancy.orbit.laplace import (
    map_laplace_fit,
    map_laplace_mixture_fit,
    numpyro_map_laplace_fit,
)
from photomancy.orbit.model import build_model
from photomancy.posterior import GaussianPosterior
from photomancy.priors import orbit_log_prior

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIST_PC = 10.0
TRUE_T = 1096.0
LOG_P_RANGE = (float(jnp.log10(600.0)), float(jnp.log10(1600.0)))
ASTROM_ERR = 5.0e-3


def _make_astrom(n_obs, seed=42):
    """Independent sky truth: Green (1985), Spherical Astronomy, ch. 17."""
    a_true = float(period_to_sma(TRUE_T, Msun2kg))
    tp_true = -1.5 / (2.0 * jnp.pi / TRUE_T)
    np.random.seed(seed)  # noqa: NPY002 - Preserve seeded fixture stream.
    times = np.sort(
        np.random.uniform(0, 3 * TRUE_T, 20)  # noqa: NPY002 - Preserve seeded times.
    )[:n_obs]
    ra_true, dec_true = sky_position(
        times,
        T=TRUE_T,
        a=a_true,
        e=0.15,
        cos_i=0.5,
        W=2.3,
        cos_w=np.cos(0.8),
        sin_w=np.sin(0.8),
        tp=tp_true,
        dist_pc=DIST_PC,
    )
    # Keep the seeded perturbations on their original physical axes:
    # raw propagation used north first; observables are now (east, north).
    north_noise = (
        np.random.randn(n_obs) * ASTROM_ERR  # noqa: NPY002 - Preserve seeded noise.
    )
    east_noise = (
        np.random.randn(n_obs) * ASTROM_ERR  # noqa: NPY002 - Preserve seeded noise.
    )
    ra = jnp.asarray(ra_true + east_noise)
    dec = jnp.asarray(dec_true + north_noise)
    return RelativeAstromData(
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


@pytest.fixture(scope="module")
def fitted():
    """Fit the five-epoch scenario once for all posterior contract checks."""
    astrom = _make_astrom(5)
    init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
    result = numpyro_map_laplace_fit(
        Msun2kg,
        DIST_PC,
        relative_astrom_data=astrom,
        log_P_range=LOG_P_RANGE,
        init_vals=init_vals,
    )
    problem = build_numpyro_logdensity(
        Msun2kg, DIST_PC, relative_astrom_data=astrom, log_P_range=LOG_P_RANGE
    )
    return result, problem


# Medium: this consumer can pay for the shared fit and blind initializer.
@pytest.mark.medium
def test_map_laplace_fit_smoke(fitted):
    """Fails if an orbit fit loses its Gaussian covariance or finite evidence."""
    result, _ = fitted
    assert isinstance(result, GaussianPosterior)
    d = result.mean.shape[0]
    assert result.cov.shape == (d, d)
    assert jnp.isfinite(result.evidence)


# Medium: this consumer can pay for the shared fit and blind initializer.
@pytest.mark.medium
def test_map_converges(fitted):
    """Fails if the five-epoch MAP period misses truth by more than 10%."""
    result, problem = fitted
    samples = sample_physical(result, problem, jax.random.PRNGKey(0), 500)
    T_median = float(jnp.median(samples["T"]))
    assert abs(T_median - TRUE_T) / TRUE_T < 0.10, (
        f"T_median={T_median:.1f} vs truth={TRUE_T:.1f}"
    )


# Medium: Three sparse-epoch fits include distinct initializer shapes.
@pytest.mark.medium
def test_covariance_positive_definite():
    """Regularised covariance is positive-definite for n=1,2,3."""
    for n_obs in [1, 2, 3]:
        astrom = _make_astrom(n_obs)
        init_vals = find_init(astrom, Msun2kg, DIST_PC, log_T_range=LOG_P_RANGE)
        result = numpyro_map_laplace_fit(
            Msun2kg,
            DIST_PC,
            relative_astrom_data=astrom,
            log_P_range=LOG_P_RANGE,
            init_vals=init_vals,
            n_steps=100,
        )
        eigvals = jnp.linalg.eigvalsh(result.cov)
        assert jnp.all(eigvals > 0), f"n={n_obs}: eigenvalues={eigvals}"


# Medium: this consumer can pay for the shared fit and blind initializer.
@pytest.mark.medium
def test_samples_shape(fitted):
    """Fails if the fitted posterior no longer samples flat z-space."""
    result, _ = fitted
    samples = result.sample(jax.random.PRNGKey(0), 100)
    assert samples.shape == (100, result.mean.shape[0])


# Medium: this consumer can pay for the shared fit and blind initializer.
@pytest.mark.medium
def test_log_prob_finite(fitted):
    """Fails if the MAP density differs from the Gaussian normalizer."""
    result, _ = fitted
    lp = result.log_prob(result.mean)
    assert jnp.isfinite(lp), f"log_prob at MAP = {lp}"
    # log_prob at the mean is the maximum (zero Mahalanobis distance),
    # so only the normalisation constant remains.
    d = result.mean.shape[0]
    chol = jnp.linalg.cholesky(result.cov)
    expected = -0.5 * d * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(chol)))
    assert jnp.allclose(lp, expected, atol=1e-5)


# Medium: optimize and differentiate both independent orbital likelihoods.
@pytest.mark.medium
def test_chart_map_and_laplace_match_numpyro():
    """Fails if the chart fit shifts the mode or curvature of the pulled-back joint."""
    astrom = _make_astrom(12, seed=0)
    frame = SkyFrame(DIST_PC, Epoch())

    def build(orbit):
        return System(
            Body(1.0), Body(jnp.zeros(1)), jax.tree.map(lambda x: x[None], orbit), frame
        )

    def prior(system):
        orbit = jax.tree.map(lambda leaf: leaf[0], system.orbits)
        return orbit_log_prior(orbit, period_range_d=(600.0, 1600.0))

    density = build_chart_logdensity(
        ClassicalChart(),
        build,
        prior,
        star_mass_kg=Msun2kg,
        terms=(RelativeAstrometryTerm(astrom),),
    )
    log_a = jnp.log(period_to_sma(TRUE_T, Msun2kg))
    x0 = jnp.array([log_a, -1.6, 0.55, 2.25, 0.75, 1.55])
    params = {"orbit": x0}
    problem = ChartOrbitProblem(density, params)
    fitted = map_laplace_fit(problem, problem.flatten(params), min_eigenvalue=1e-10)

    model = build_model(has_relative_astrom=True, log_P_range=LOG_P_RANGE)
    model_args = (Msun2kg, DIST_PC, OrbitData(relative_astrom=astrom))

    def negative_joint(x):
        # Kepler's law gives dlog10(P)/dln(a) = 1.5 / ln(10).
        e, cos_i = jax.nn.sigmoid(x[1]), jnp.tanh(x[2])
        log_p = jnp.log10(TRUE_T) + 1.5 * (x[0] - log_a) / jnp.log(10.0)
        sites = dict(log_P=log_p, e_raw=e, cos_i=cos_i, W=x[3], w_raw=x[4], M0=x[5])
        joint, _ = log_density(
            model, model_args, {}, jax.tree.map(jnp.atleast_1d, sites)
        )
        jac = 1.5 / jnp.log(10.0) * e * (1 - e) * (1 - cos_i**2)
        return -joint - jnp.log(jac)

    # Independent optimizer and direct Hessian factorization, without eigenvalue
    # clipping: these epochs resolve all six coordinates.
    reference = minimize(
        jax.jit(jax.value_and_grad(negative_joint)),
        x0,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8},
    )
    precision = np.asarray(jax.jit(jax.hessian(negative_joint))(reference.x))
    root = np.linalg.cholesky(precision)
    # Measure residual gradients and MAP displacements in posterior sigma.
    # A 1e-10 nat objective floor resolves displacements sqrt(2e-10) sigma.
    sigma_floor = np.sqrt(2e-10)
    assert np.linalg.norm(np.linalg.solve(root, reference.jac)) < sigma_floor
    mode_delta = np.linalg.norm(root.T @ (fitted.mean - reference.x))
    # Whitening gives a unit covariance. Compare principal variances to
    # 1 ppm, well below the uncertainty of the Gaussian approximation.
    cov_delta = np.linalg.norm(root.T @ fitted.cov @ root - np.eye(6), ord=2)
    log_z = -reference.fun + 3 * np.log(two_pi) - np.log(np.diag(root)).sum()
    evidence_delta = abs(float(fitted.evidence) - log_z)
    print(
        f"MAP delta {mode_delta:.5g} sigma; whitened covariance delta "
        f"{cov_delta:.5g}; Laplace log Z delta {evidence_delta:.5g} nat"
    )
    assert mode_delta < sigma_floor
    assert cov_delta < 1e-6
    # A 1 ppm relative variance shift changes 0.5 log(det cov) by 0.5 ppm
    # per dimension at first order.
    assert evidence_delta < 3.1e-6

    # Relative astrometry is invariant under the simultaneous pi shift of
    # node and periapsis. The two resolved modes have the same evidence.
    starts = jnp.stack([x0, x0.at[3:5].add(jnp.pi)])
    mixture = map_laplace_mixture_fit(problem, starts, min_eigenvalue=1e-10)
    mirror_free_means = mixture.means.at[1, 3:5].add(-jnp.pi)
    displacements = (mirror_free_means - reference.x) @ root
    assert np.max(np.linalg.norm(displacements, axis=1)) < sigma_floor
    np.testing.assert_allclose(mixture.log_evidences, log_z, rtol=0, atol=3.1e-6)
