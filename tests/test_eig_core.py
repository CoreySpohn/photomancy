"""Tests for the domain-agnostic EIG layer."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils import PrecisionError
from hwoutils.testing import ulps
from jax.scipy.stats import norm
from nmc_reference import (
    class_mi_mc,
    detection_channel_mi_mc,
    mixture_mode_mi_mc,
    probit_site_moments_mc,
    reported_mixture_mode_mi_mc,
)
from priorities import Mixture, MultivariateNormal

from photomancy.infer.eig import (
    LinearizedEIG,
    _ep_probit_update,
    _null_update,
    _probit_gaussian_mass,
    eig_alias_breaking,
    eig_alias_breaking_from_covariances,
    eig_class,
    eig_contrastive,
    eig_detectability,
    eig_detection_channel,
    eig_detection_class,
    eig_geometric,
    eig_linearized,
    eig_nested,
)

_posterior_information = pytest.mark.float64_only(
    reason="posterior weights, moments and information gain require float64"
)


def _mixture(means, covs, log_evidences):
    """A ``priorities.Mixture`` of Gaussian modes from moments and log weights."""
    return Mixture(
        mixing_logits=jnp.asarray(log_evidences),
        component_distribution=MultivariateNormal(
            loc=jnp.asarray(means), scale_tril=jnp.linalg.cholesky(jnp.asarray(covs))
        ),
    )


def _two_mode_posterior():
    return _mixture(
        means=jnp.array([[1.0], [2.0]]),
        covs=jnp.array([[[1.0]], [[1.0]]]),
        log_evidences=jnp.array([0.0, 0.0]),
    )


@_posterior_information
def test_alias_breaking_eig_zero_when_modes_agree():
    """Zero when modes predict the same observable; positive when they differ."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(
        eig_alias_breaking(weights, jnp.array([[1.0], [1.0]]), 1.0), 0.0
    )
    assert eig_alias_breaking(weights, jnp.array([[1.0], [5.0]]), 1.0) > 0.0


@_posterior_information
def test_detectability_eig_zero_when_modes_agree():
    """Zero when all modes agree on detectability; positive when split."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(eig_detectability(weights, jnp.array([1.0, 1.0])), 0.0)
    assert eig_detectability(weights, jnp.array([1.0, 0.0])) > 0.0


@_posterior_information
def test_eig_linearized_linear_forward_more_informative_scores_higher():
    """A larger Jacobian and wider mode disagreement give more information."""
    posterior = _two_mode_posterior()

    def forward(z, c):
        return jnp.array([c * z[0]])

    res = eig_linearized(posterior, jnp.array([0.5, 2.0]), forward, 1.0)
    assert res.total.shape == (2,)
    assert jnp.all(jnp.isfinite(res.total))
    assert jnp.all(res.total >= -1e-6)
    assert res.total[1] > res.total[0]
    assert res.predictions.shape == (2, 2, 1)


@_posterior_information
def test_eig_linearized_detectable_splits_modes():
    """A detectable callback that splits the modes drives the detectability output."""
    posterior = _two_mode_posterior()

    def forward(z, c):
        return jnp.array([c * z[0]])

    def detectable(z, c):
        return jnp.where(z[0] < 1.5, 1.0, 0.0)  # mode 0 detectable, mode 1 not

    res = eig_linearized(
        posterior, jnp.array([1.0]), forward, 1.0, detectable=detectable
    )
    assert res.detectability.shape == (1, 2)
    assert jnp.allclose(res.detectability[0], jnp.array([1.0, 0.0]))
    assert jnp.isfinite(res.total[0])


# ---------------------------------------------------------------------------
# Exact discrete-tier forms (the unified-currency upgrades)
# ---------------------------------------------------------------------------

LN2 = float(jnp.log(2.0))


@_posterior_information
def test_detectability_eig_exact_at_deterministic_split():
    """A deterministic 50/50 detection split carries exactly ln 2 nats.

    I(D; M) = H_b(sum w_k d_k) - sum w_k H_b(d_k); the old variance surrogate
    0.5*log(1 + Var_w(d)/0.25) reported 0.5*log(2) here -- a 2x undercount.
    """
    weights = jnp.array([0.5, 0.5])
    det = jnp.array([0.0, 1.0])
    assert jnp.allclose(eig_detectability(weights, det), LN2)


@_posterior_information
def test_detectability_eig_bounded_by_mode_entropy():
    """I(D; M) <= H(w): the detection bit cannot exceed the mode uncertainty."""
    weights = jnp.array([0.9, 0.1])
    h_w = -jnp.sum(weights * jnp.log(weights))
    det = jnp.array([0.0, 1.0])
    val = eig_detectability(weights, det)
    assert val <= h_w + 1e-12
    assert val > 0.0


@_posterior_information
def test_detectability_eig_soft_probabilities_match_direct_sum():
    """The binary-entropy form equals the direct double-sum mutual information."""
    weights = jnp.array([0.3, 0.5, 0.2])
    det = jnp.array([0.1, 0.7, 0.95])
    p_bar = jnp.sum(weights * det)
    direct = 0.0
    for k in range(3):
        for d_out, p_out in ((1, det[k]), (0, 1.0 - det[k])):
            p_marg = p_bar if d_out == 1 else 1.0 - p_bar
            direct += jnp.where(
                p_out > 0, weights[k] * p_out * jnp.log(p_out / p_marg), 0.0
            )
    assert jnp.allclose(eig_detectability(weights, det), direct, rtol=1e-10)


@_posterior_information
def test_alias_breaking_eig_capped_at_mode_entropy():
    """Two perfectly resolved 50/50 modes are worth ln 2 nats, not log(separation).

    The moment-matched bound 0.5*log(1 + Var_w/sigma^2) grows without bound as
    modes separate; the true I(M; Y) <= H(w).
    """
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [100.0]])
    assert jnp.allclose(eig_alias_breaking(weights, y_preds, 1.0), LN2)

    # Far below full separation the bound itself (not the cap) is active.
    y_close = jnp.array([[0.0], [0.5]])
    val = eig_alias_breaking(weights, y_close, 1.0)
    assert val < LN2
    assert jnp.allclose(val, 0.5 * jnp.log1p(0.0625), rtol=1e-10)


@_posterior_information
def test_alias_breaking_eig_uses_per_mode_predictive_widths():
    """(K, n_obs) obs_variance = per-mode predictive J Sigma J^T + R.

    Wide within-mode predictives swallow the mode separation: the alias term
    must shrink relative to the R-only evaluation.
    """
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [1.0]])
    narrow = eig_alias_breaking(weights, y_preds, 0.01)
    pred_var = jnp.array([[1.01], [1.01]])  # J Sigma J^T = 1.0 on top of R = 0.01
    wide = eig_alias_breaking(weights, y_preds, pred_var)
    assert wide < narrow
    expected = 0.5 * (jnp.log(0.25 + 1.01) - jnp.log(1.01))
    assert jnp.allclose(wide, expected, rtol=1e-10)


@_posterior_information
def test_alias_breaking_eig_distinguishes_unequal_widths():
    """Modes with equal means but different predictive widths are informative."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [0.0]])
    pred_var = jnp.array([[0.01], [4.0]])
    assert eig_alias_breaking(weights, y_preds, pred_var) > 0.1


@_posterior_information
def test_geometric_eig_qoi_projection_ignores_nuisance_gain():
    """A nuisance-only observation shows zero marginal gain on the QoI block.

    EIG_g = 0.5*[logdet(Phi Sigma Phi^T) - logdet(Phi Sigma' Phi^T)] with Phi the
    QoI selector; the full-theta logdet would book the nuisance shrinkage as gain.
    """
    cov = jnp.eye(2)
    jac_nuisance = jnp.array([[0.0, 1.0]])  # measures only parameter 1
    phi_qoi = jnp.array([[1.0, 0.0]])  # QoI is parameter 0

    eig_full, _ = eig_geometric(cov, jac_nuisance, 0.01)
    assert eig_full > 1.0  # full-theta logdet books the nuisance shrinkage

    eig_marg, cov_new = eig_geometric(cov, jac_nuisance, 0.01, qoi_projection=phi_qoi)
    assert jnp.allclose(eig_marg, 0.0, atol=1e-10)
    assert cov_new.shape == (2, 2)  # update itself stays full-space


@_posterior_information
def test_geometric_eig_identity_projection_matches_full():
    """Phi = I reproduces the unprojected gain."""
    cov = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    jac = jnp.array([[1.0, 0.5], [0.2, 2.0]])
    eig_full, _ = eig_geometric(cov, jac, 0.5)
    eig_proj, _ = eig_geometric(cov, jac, 0.5, qoi_projection=jnp.eye(2))
    assert jnp.allclose(eig_full, eig_proj, rtol=1e-10)


@_posterior_information
@pytest.mark.parametrize("gap", [1e-6, 1e-12])
@pytest.mark.parametrize("row", [(1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_geometric_eig_near_singular_prior_matches_the_scalar_closed_form(gap, row):
    """One scalar observation of a nearly degenerate prior gains its closed form.

    Truth source: for one observation ``y = j . z`` with variance ``v``, the gain is
    ``0.5 log1p(j Sigma j^T / v)``. With correlation ``rho = 1 - gap`` the quadratic
    form along ``(1, -1)`` is ``2 - 2 rho``, exact in float64 (Sterbenz), so the
    reference is a float64 expression accurate to a few units in the last place.
    Tolerance basis: floating-point. Eight units in the last place along the well
    measured directions; along the thin direction the data perturb the answer by
    about eps / gap relative, so the bound is eight times that condition number.
    Fails if the prior is inverted outright: adding the information to a precision
    with entries near 1 / gap loses the measured direction (a zero or negative gain
    at gap 1e-12, relative errors near 1e-5 elsewhere).
    """
    rho = 1.0 - gap
    cov = jnp.array([[1.0, rho], [rho, 1.0]])
    jac = jnp.array([row])
    variance = 0.5
    eig, cov_new = eig_geometric(cov, jac, variance)
    j0, j1 = row
    quadratic = j0 * j0 + j1 * j1 + 2.0 * rho * j0 * j1
    if row == (1.0, -1.0):
        quadratic = 2.0 - 2.0 * rho
    want = 0.5 * np.log1p(quadratic / variance)
    eps = float(ulps(1, 1.0))
    rtol = 8 * eps / gap if row == (1.0, -1.0) else 8 * eps
    np.testing.assert_allclose(float(eig), want, rtol=rtol, atol=0)
    assert bool(jnp.all(jnp.isfinite(cov_new)))


@_posterior_information
def test_geometric_eig_correlated_qoi_marginal_gain_positive():
    """A correlated nuisance measurement still informs the QoI marginal."""
    cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    jac = jnp.array([[0.0, 1.0]])
    phi = jnp.array([[1.0, 0.0]])
    eig_marg, _ = eig_geometric(cov, jac, 0.01, qoi_projection=phi)
    # Sigma'_00 = 1 - 0.8^2 * (1 / (1 + 0.01)) -> gain ~ 0.5 log(1/0.366)
    assert eig_marg > 0.4


# ---------------------------------------------------------------------------
# Nested-MC reference cross-checks (the tier-Z calibrator in miniature)
# ---------------------------------------------------------------------------


@_posterior_information
def test_alias_bound_upper_bounds_true_mixture_mi():
    """The capped alias term is a valid upper bound on the exact I(M; Y).

    Checked against a brute-force MC of the mixture MI across the overlap ->
    resolved range; the bound must sit above the MC truth at every separation.
    """
    weights = jnp.array([0.5, 0.5])
    variances = jnp.ones((2, 1))
    for i, sep in enumerate([0.5, 1.0, 2.0, 4.0, 20.0]):
        y_preds = jnp.array([[0.0], [sep]])
        bound = eig_alias_breaking(weights, y_preds, variances)
        mc, se = mixture_mode_mi_mc(jax.random.PRNGKey(i), weights, y_preds, variances)
        # 1e-12 covers the ulp-equality case where both saturate at exactly H(w).
        assert bound >= mc - 4.0 * se - 1e-12, f"bound violated at sep={sep}"


@_posterior_information
def test_alias_cap_tight_in_resolved_limit():
    """At full separation the MC truth reaches H(w) and the cap is exact."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [50.0]])
    variances = jnp.ones((2, 1))
    bound = eig_alias_breaking(weights, y_preds, variances)
    mc, se = mixture_mode_mi_mc(jax.random.PRNGKey(0), weights, y_preds, variances)
    assert jnp.allclose(bound, LN2)
    assert jnp.abs(mc - LN2) < 4.0 * se + 1e-3


@_posterior_information
def test_detectability_eig_matches_binary_channel_limits():
    """Exact detection MI hits both closed-form limits of the binary channel."""
    weights = jnp.array([0.5, 0.5])
    # Deterministic split: ln 2. Noisy but symmetric d = {0.2, 0.8}:
    # I = H_b(0.5) - H_b(0.2) (binary symmetric channel with crossover 0.2).
    hb = lambda p: -p * jnp.log(p) - (1 - p) * jnp.log(1 - p)  # noqa: E731
    val = eig_detectability(weights, jnp.array([0.2, 0.8]))
    assert jnp.allclose(val, LN2 - hb(0.2), rtol=1e-10)


# ---------------------------------------------------------------------------
# eig_class: classification gain I(C; Y) over caller-supplied P(c | k)
# ---------------------------------------------------------------------------


@_posterior_information
def test_detection_class_eig_identity_classes_equals_detectability():
    """With C = M (identity class map) the closed form reduces to I(D; M)."""
    weights = jnp.array([0.3, 0.7])
    det = jnp.array([0.1, 0.9])
    val = eig_detection_class(weights, det, jnp.eye(2))
    assert jnp.allclose(val, eig_detectability(weights, det), rtol=1e-10)


@_posterior_information
def test_detection_class_eig_merged_classes_lose_information():
    """Merging modes into one class destroys the detection information (DPI)."""
    weights = jnp.array([0.5, 0.5])
    det = jnp.array([0.0, 1.0])
    one_class = jnp.array([[1.0], [1.0]])
    assert jnp.allclose(eig_detection_class(weights, det, one_class), 0.0, atol=1e-12)

    # Three modes, two classes: I(C; D) <= I(M; D).
    weights3 = jnp.array([0.3, 0.4, 0.3])
    det3 = jnp.array([0.05, 0.6, 0.95])
    classes3 = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert (
        eig_detection_class(weights3, det3, classes3)
        <= eig_detectability(weights3, det3) + 1e-12
    )


@_posterior_information
def test_detection_class_eig_certain_detection_is_uninformative():
    """All modes surely detected -> the detection bit carries nothing."""
    weights = jnp.array([0.5, 0.5])
    det = jnp.ones(2)
    assert jnp.allclose(eig_detection_class(weights, det, jnp.eye(2)), 0.0, atol=1e-12)


@_posterior_information
def test_class_eig_saturates_at_class_prior_entropy():
    """I(C; Y) <= H(C): a huge-separation observation yields H(C), not more."""
    weights = jnp.array([0.5, 0.5])
    class_probs = jnp.eye(2)
    y_preds = jnp.array([[0.0], [200.0]])
    val = eig_class(
        weights, class_probs, y_preds, 1.0, key=jax.random.PRNGKey(0), n_samples=256
    )
    h_class = -jnp.sum(0.5 * jnp.log(0.5) * jnp.ones(2))
    assert val <= h_class + 1e-9
    assert jnp.allclose(val, LN2, atol=5e-3)


@_posterior_information
def test_class_eig_identical_predictions_zero():
    """Modes that predict identically cannot classify: I(C; Y) ~ 0."""
    weights = jnp.array([0.5, 0.5])
    val = eig_class(
        weights,
        jnp.eye(2),
        jnp.array([[1.0], [1.0]]),
        1.0,
        key=jax.random.PRNGKey(0),
        n_samples=256,
    )
    assert jnp.allclose(val, 0.0, atol=1e-9)


@_posterior_information
def test_class_eig_matches_nmc_reference_mid_separation():
    """The stratified estimator agrees with brute-force nested MC of I(C; Y)."""
    weights = jnp.array([0.4, 0.35, 0.25])
    class_probs = jnp.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    y_preds = jnp.array([[0.0, 0.5], [1.5, -0.5], [3.0, 1.0]])
    variances = jnp.array([[1.0, 2.0], [1.5, 1.0], [1.0, 1.0]])
    val = eig_class(
        weights,
        class_probs,
        y_preds,
        variances,
        key=jax.random.PRNGKey(0),
        n_samples=4096,
    )
    mc, se = class_mi_mc(
        jax.random.PRNGKey(1), weights, class_probs, y_preds, variances, 40000
    )
    assert jnp.abs(val - mc) < 4.0 * se + 5e-3


@_posterior_information
def test_class_eig_dpi_under_capped_alias():
    """Class-pure modes: I(C; Y) <= I(M; Y) <= capped alias bound (DPI chain)."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [2.0]])
    variances = jnp.ones((2, 1))
    val = eig_class(
        weights,
        jnp.eye(2),
        y_preds,
        variances,
        key=jax.random.PRNGKey(0),
        n_samples=4096,
    )
    bound = eig_alias_breaking(weights, y_preds, variances)
    assert val <= bound + 5e-3


@_posterior_information
def test_class_eig_wider_separation_more_informative():
    """Monotone in mode separation for class-pure modes."""
    weights = jnp.array([0.5, 0.5])
    vals = [
        eig_class(
            weights,
            jnp.eye(2),
            jnp.array([[0.0], [sep]]),
            1.0,
            key=jax.random.PRNGKey(0),
            n_samples=1024,
        )
        for sep in (0.5, 2.0, 8.0)
    ]
    assert vals[0] < vals[1] < vals[2]


# ---------------------------------------------------------------------------
# EP null tier: probit-Gaussian mass, site moment update, detection-channel EIG
# ---------------------------------------------------------------------------


def test_probit_gaussian_mass_matches_mc():
    """The closed-form probit-Gaussian integral equals the MC expectation."""
    mean = jnp.array([0.4, -0.2])
    cov = jnp.array([[0.8, 0.3], [0.3, 0.6]])
    a, b = 0.3, jnp.array([1.2, -0.7])
    mass = _probit_gaussian_mass(mean, cov, a, b)
    mc_mass, _, _ = probit_site_moments_mc(jax.random.PRNGKey(0), mean, cov, a, b)
    assert jnp.abs(mass - mc_mass) < 5.0e-3


@_posterior_information
def test_ep_probit_update_matches_mc_tilted_moments():
    """The EP site update returns the exact tilted mean and covariance."""
    mean = jnp.array([0.4, -0.2])
    cov = jnp.array([[0.8, 0.3], [0.3, 0.6]])
    a, b = -0.5, jnp.array([1.5, 0.4])
    log_mass, mean_new, cov_new = _ep_probit_update(mean, cov, a, b)
    mc_mass, mc_mean, mc_cov = probit_site_moments_mc(
        jax.random.PRNGKey(0), mean, cov, a, b, n_samples=400000
    )
    assert jnp.abs(jnp.exp(log_mass) - mc_mass) < 5.0e-3
    assert jnp.all(jnp.abs(mean_new - mc_mean) < 2.0e-2)
    assert jnp.all(jnp.abs(cov_new - mc_cov) < 2.0e-2)


@_posterior_information
def test_ep_probit_update_hard_truncation_limit():
    """A steep site reproduces the truncated-Gaussian Mills-ratio moments.

    For X ~ N(mu, sigma^2) truncated to X <= c: with alpha = (c - mu)/sigma and
    lam = phi(alpha)/Phi(alpha), E[X] = mu - sigma*lam and
    Var[X] = sigma^2 (1 - alpha*lam - lam^2).
    """
    mu, sigma, c = 1.0, 0.7, 1.3
    steep = 400.0  # site Phi(steep * (c - x)) -> indicator{x <= c}
    log_mass, mean_new, cov_new = _ep_probit_update(
        jnp.array([mu]), jnp.array([[sigma**2]]), steep * c, jnp.array([-steep])
    )
    alpha = (c - mu) / sigma
    lam = jnp.exp(norm.logpdf(alpha) - norm.logcdf(alpha))
    assert jnp.allclose(mean_new[0], mu - sigma * lam, rtol=1.0e-3)
    assert jnp.allclose(
        cov_new[0, 0], sigma**2 * (1.0 - alpha * lam - lam**2), rtol=5.0e-3
    )
    assert jnp.allclose(jnp.exp(log_mass), norm.cdf(alpha), rtol=1.0e-3)


@_posterior_information
def test_ep_probit_update_flat_site_is_identity():
    """A site that is ~1 over the bulk leaves mass and moments unchanged."""
    mean = jnp.array([0.0])
    cov = jnp.array([[1.0]])
    log_mass, mean_new, cov_new = _ep_probit_update(mean, cov, 30.0, jnp.array([0.1]))
    assert jnp.exp(log_mass) > 0.999
    assert jnp.allclose(mean_new, mean, atol=1.0e-4)
    assert jnp.allclose(cov_new, cov, rtol=1.0e-3)


@_posterior_information
def test_null_update_kills_the_detectable_mode():
    """After an observed null, the solidly detectable mode loses its weight.

    Detection site p_det = Phi(a + b theta): mode at theta = +3 is deep in the
    detectable regime, mode at theta = -3 is deep in the undetectable regime.
    """
    weights = jnp.array([0.5, 0.5])
    means = jnp.array([[3.0], [-3.0]])
    covs = jnp.array([[[0.2]], [[0.2]]])
    a, b = 0.0, jnp.array([3.0])
    w_new, m_new, _c_new = _null_update(weights, means, covs, a, b)
    assert w_new[1] > 0.999
    assert jnp.allclose(jnp.sum(w_new), 1.0, rtol=1.0e-9)
    # The surviving mode was never near the boundary: moments unchanged.
    assert jnp.allclose(m_new[1], means[1], atol=1.0e-3)


@_posterior_information
def test_null_update_shaves_a_mode_cut_by_the_limit():
    """A mode straddling the detection boundary is shifted and narrowed."""
    weights = jnp.array([1.0])
    means = jnp.array([[0.0]])
    covs = jnp.array([[[1.0]]])
    a, b = 0.0, jnp.array([2.0])  # boundary right through the mode
    _w_new, m_new, c_new = _null_update(weights, means, covs, a, b)
    assert m_new[0, 0] < -0.3  # pushed toward the undetectable side
    assert c_new[0, 0, 0] < 1.0  # tail shaved


@_posterior_information
def test_detection_channel_eig_reduces_to_detectability_eig_at_zero_width():
    """With vanishing within-mode spread the channel MI is the discrete I(D; M)."""
    weights = jnp.array([0.4, 0.6])
    means = jnp.array([[1.0], [-1.5]])
    covs = jnp.array([[[1.0e-8]], [[1.0e-8]]])
    a, b = 0.2, jnp.array([1.3])
    val = eig_detection_channel(weights, means, covs, a, b)
    d_k = norm.cdf(a + means @ b)
    assert jnp.allclose(val, eig_detectability(weights, d_k), atol=1.0e-6)


@_posterior_information
def test_detection_channel_eig_adds_within_mode_information():
    """The chain rule adds a nonnegative within-mode term to the smeared I(D; M)."""
    weights = jnp.array([0.5, 0.5])
    means = jnp.array([[0.5], [-0.5]])
    covs = jnp.array([[[1.5]], [[1.5]]])
    a, b = 0.0, jnp.array([2.0])
    val = eig_detection_channel(weights, means, covs, a, b)
    d_smeared = jax.vmap(lambda m, c: _probit_gaussian_mass(m, c, a, b))(means, covs)
    discrete_only = eig_detectability(weights, d_smeared)
    assert val >= discrete_only - 1.0e-12
    assert val > discrete_only + 1.0e-3  # boundary cuts both modes: strictly more


@_posterior_information
def test_detection_channel_eig_matches_nmc():
    """The Gauss-Hermite channel MI agrees with brute-force nested MC."""
    weights = jnp.array([0.35, 0.65])
    means = jnp.array([[0.8, 0.0], [-0.6, 0.5]])
    covs = jnp.array([[[0.7, 0.2], [0.2, 0.5]], [[1.1, -0.3], [-0.3, 0.9]]])
    a_arr = jnp.array([0.3, -0.2])
    b_arr = jnp.array([[1.4, -0.5], [0.9, 1.1]])
    val = eig_detection_channel(weights, means, covs, a_arr, b_arr)
    mc, se = detection_channel_mi_mc(
        jax.random.PRNGKey(0), weights, means, covs, a_arr, b_arr
    )
    assert jnp.abs(val - mc) < 4.0 * se + 2.0e-3


@_posterior_information
def test_fisher_is_blind_to_the_null_but_the_channel_mi_is_not():
    """The sec-8.1 counterexample: Bernoulli FIM ~ 0 at both modes, MI = ln 2.

    One mode solidly detectable, one solidly not: the detection outcome resolves
    the mode with probability ~1 (ln 2 nats) while the per-mode Fisher information
    of the Bernoulli channel is numerically zero at both mode means.
    """
    weights = jnp.array([0.5, 0.5])
    means = jnp.array([[6.0], [-6.0]])
    covs = jnp.array([[[0.3]], [[0.3]]])
    a, b = 0.0, jnp.array([2.0])

    val = eig_detection_channel(weights, means, covs, a, b)
    assert jnp.abs(val - LN2) < 1.0e-3

    def bernoulli_fim(theta):
        z = a + b[0] * theta
        dp = b[0] * norm.pdf(z)
        return dp**2 / (norm.cdf(z) * norm.cdf(-z))  # p(1-p), saturation-stable

    assert bernoulli_fim(6.0) < 1.0e-20
    assert bernoulli_fim(-6.0) < 1.0e-20


# ---------------------------------------------------------------------------
# Full predictive covariance: correlated components and observable rotations
# ---------------------------------------------------------------------------

# Equal-weight, zero-mean components with identical marginal variances and
# opposite correlations, observed through the identity with isotropic noise.
# Only the joint observation can tell them apart.
OPPOSITE_COVS = jnp.array([[[1.0, 0.9], [0.9, 1.0]], [[1.0, -0.9], [-0.9, 1.0]]])
OPPOSITE_NOISE = 0.1
ROTATION_45 = jnp.array([[1.0, 1.0], [-1.0, 1.0]]) / jnp.sqrt(2.0)


def _opposite_correlation_posterior():
    return _mixture(
        means=jnp.zeros((2, 2)),
        covs=OPPOSITE_COVS,
        log_evidences=jnp.log(jnp.array([0.5, 0.5])),
    )


@_posterior_information
def test_mode_information_invariant_under_observable_rotation():
    """Rotating the observables of one physical experiment preserves every term.

    An orthogonal rotation of the observation with isotropic noise is the same
    experiment, so the mode-information bound must not change, and it must sit
    above a full-covariance Monte Carlo estimate of I(M; Y), which is positive
    here. Tolerance: floating-point for the invariance; monte-carlo (4 sigma,
    200000 draws, seed 0) for the bound.
    """
    posterior = _opposite_correlation_posterior()
    original = eig_linearized(
        posterior, jnp.array([0.0]), lambda z, c: z, OPPOSITE_NOISE
    )
    rotated = eig_linearized(
        posterior, jnp.array([0.0]), lambda z, c: ROTATION_45 @ z, OPPOSITE_NOISE
    )
    for name in ("total", "geometric", "alias"):
        assert jnp.allclose(
            getattr(original, name), getattr(rotated, name), rtol=0.0, atol=1e-12
        ), name

    predictive = OPPOSITE_COVS + OPPOSITE_NOISE * jnp.eye(2)
    mc, se = reported_mixture_mode_mi_mc(
        jax.random.PRNGKey(0), jnp.array([0.5, 0.5]), jnp.zeros((2, 2)), predictive
    )
    assert mc > 0.2
    assert original.alias[0] >= mc - 4.0 * se


@_posterior_information
def test_covariance_bound_closed_form_for_opposite_correlations():
    """The moment-matched bound uses full determinants.

    The mixture covariance is ``1.1 I`` and each component determinant is
    ``1.1**2 - 0.9**2 = 0.4``, so the bound is ``0.5*log(1.21 / 0.4)``. The
    per-observable (diagonal) form reports zero here. Tolerance: exact-algebra.
    """
    predictive = OPPOSITE_COVS + OPPOSITE_NOISE * jnp.eye(2)
    val = eig_alias_breaking_from_covariances(
        jnp.array([0.5, 0.5]), jnp.zeros((2, 2)), predictive
    )
    assert jnp.allclose(val, 0.5 * jnp.log(1.21 / 0.4), rtol=1e-12)


@_posterior_information
def test_covariance_bound_rotation_invariant_with_offset_means():
    """An orthogonal change of observable basis leaves the bound unchanged."""
    weights = jnp.array([0.2, 0.5, 0.3])
    means = jnp.array([[0.3, -0.4], [1.0, 0.2], [-0.5, 0.7]])
    covs = jnp.array(
        [
            [[0.8, 0.3], [0.3, 0.5]],
            [[1.2, -0.6], [-0.6, 0.9]],
            [[0.4, 0.1], [0.1, 1.5]],
        ]
    )
    rot = jnp.array([[jnp.cos(0.7), -jnp.sin(0.7)], [jnp.sin(0.7), jnp.cos(0.7)]])
    base = eig_alias_breaking_from_covariances(weights, means, covs)
    turned = eig_alias_breaking_from_covariances(
        weights, means @ rot.T, jnp.einsum("ij,kjl,ml->kim", rot, covs, rot)
    )
    assert base > 0.0
    assert jnp.allclose(base, turned, rtol=1e-12)


@_posterior_information
def test_covariance_bound_upper_bounds_full_covariance_mc():
    """The bound sits above the exact I(M; Y) and is not equal to it.

    Reference: a full-covariance Monte Carlo of the mixture mutual information
    (200000 draws, seed 0). Tolerance: monte-carlo, 4 sigma on each side.
    """
    weights = jnp.array([0.5, 0.5])
    predictive = OPPOSITE_COVS + OPPOSITE_NOISE * jnp.eye(2)
    bound = eig_alias_breaking_from_covariances(weights, jnp.zeros((2, 2)), predictive)
    mc, se = reported_mixture_mode_mi_mc(
        jax.random.PRNGKey(0), weights, jnp.zeros((2, 2)), predictive
    )
    assert mc > 0.2  # the correlation difference is informative
    # The value quoted in the mathematical-foundations explanation page.
    assert abs(mc - 0.331) < 4.0 * se + 5e-4
    assert bound >= mc - 4.0 * se
    assert bound > mc + 4.0 * se  # an upper bound here, not the exact value


@_posterior_information
def test_covariance_bound_zero_for_identical_distributions():
    """Identical means and full covariances carry no mode information."""
    cov = jnp.array([[1.0, 0.7], [0.7, 2.0]])
    val = eig_alias_breaking_from_covariances(
        jnp.array([0.2, 0.3, 0.5]),
        jnp.tile(jnp.array([[0.4, -1.0]]), (3, 1)),
        jnp.tile(cov[None], (3, 1, 1)),
    )
    assert jnp.allclose(val, 0.0, atol=1e-12)


@_posterior_information
def test_covariance_bound_matches_variance_interface_in_one_dimension():
    """In one dimension the covariance form equals the variance form and closed form."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [0.5]])
    variances = jnp.array([[1.0], [1.3]])
    from_cov = eig_alias_breaking_from_covariances(
        weights, y_preds, variances[:, :, None]
    )
    from_var = eig_alias_breaking(weights, y_preds, variances)
    expected = 0.5 * (jnp.log(0.0625 + 1.15) - 0.5 * jnp.log(1.3))
    assert jnp.allclose(from_cov, from_var, rtol=1e-12)
    assert jnp.allclose(from_cov, expected, rtol=1e-12)


@_posterior_information
def test_variance_interface_is_the_diagonal_covariance_case():
    """Per-mode variances mean diagonal predictive covariances, in any dimension.

    The between-mode spread still enters with its full outer product, so for
    diagonal inputs in two dimensions the result is not the per-observable sum.
    """
    weights = jnp.array([0.3, 0.7])
    y_preds = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    variances = jnp.array([[1.0, 2.0], [0.5, 1.5]])
    diag_covs = jax.vmap(jnp.diag)(variances)
    assert jnp.allclose(
        eig_alias_breaking(weights, y_preds, variances),
        eig_alias_breaking_from_covariances(weights, y_preds, diag_covs),
        rtol=1e-12,
    )
    # Scalar and (n_obs,) measurement variances broadcast to every mode.
    assert jnp.allclose(
        eig_alias_breaking(weights, y_preds, jnp.array([0.5, 2.0])),
        eig_alias_breaking_from_covariances(
            weights, y_preds, jnp.tile(jnp.diag(jnp.array([0.5, 2.0]))[None], (2, 1, 1))
        ),
        rtol=1e-12,
    )


@_posterior_information
def test_covariance_bound_capped_at_mode_entropy():
    """Resolved correlated components are worth H(w), not the Gaussian bound."""
    weights = jnp.array([0.2, 0.3, 0.5])
    means = jnp.array([[0.0, 0.0], [80.0, 0.0], [0.0, 80.0]])
    covs = jnp.tile(OPPOSITE_COVS[:1] + OPPOSITE_NOISE * jnp.eye(2), (3, 1, 1))
    val = eig_alias_breaking_from_covariances(weights, means, covs)
    h_w = -jnp.sum(weights * jnp.log(weights))
    assert jnp.allclose(val, h_w, rtol=1e-12)


@_posterior_information
def test_covariance_bound_ignores_zero_mass_components():
    """A zero-weight component is absent: adding one leaves the bound unchanged."""
    weights = jnp.array([0.4, 0.6])
    means = jnp.array([[0.0, 0.0], [0.8, -0.3]])
    covs = OPPOSITE_COVS + OPPOSITE_NOISE * jnp.eye(2)
    base = eig_alias_breaking_from_covariances(weights, means, covs)
    padded = eig_alias_breaking_from_covariances(
        jnp.array([0.4, 0.0, 0.6]),
        jnp.array([[0.0, 0.0], [1.0e6, -1.0e6], [0.8, -0.3]]),
        jnp.stack([covs[0], 1.0e-6 * jnp.eye(2), covs[1]]),
    )
    assert jnp.allclose(base, padded, rtol=1e-12)


@_posterior_information
def test_covariance_bound_rejects_non_positive_definite_input():
    """Indefinite, singular, or nonfinite covariances give NaN, not a clipped number."""
    weights = jnp.array([0.5, 0.5])
    means = jnp.zeros((2, 2))
    good = jnp.eye(2)
    for bad in (
        jnp.array([[1.0, 2.0], [2.0, 1.0]]),  # indefinite
        jnp.array([[1.0, 1.0], [1.0, 1.0]]),  # singular
        jnp.array([[jnp.nan, 0.0], [0.0, 1.0]]),
    ):
        val = eig_alias_breaking_from_covariances(
            weights, means, jnp.stack([good, bad])
        )
        assert jnp.isnan(val)


@_posterior_information
def test_covariance_bound_rejects_mismatched_shapes():
    """Shapes must be (K,), (K, n_obs), (K, n_obs, n_obs); no 2-D covariance guess."""
    weights = jnp.array([0.5, 0.5])
    means = jnp.zeros((2, 2))
    with pytest.raises(ValueError, match="predictive_covariances"):
        eig_alias_breaking_from_covariances(weights, means, jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="predictive_covariances"):
        eig_alias_breaking_from_covariances(weights, means, jnp.ones((2, 3, 3)))


def _float64_covariance_bound(weights, means, covs):
    """Independent NumPy float64 evaluation of the moment-matched Gaussian bound."""
    weights = np.asarray(weights, float)
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    centered = means - weights @ means
    mixture = np.einsum("k,kij->ij", weights, covs) + np.einsum(
        "k,ki,kj->ij", weights, centered, centered
    )
    logdets = np.array([np.linalg.slogdet(c)[1] for c in covs])
    bound = 0.5 * (np.linalg.slogdet(mixture)[1] - weights @ logdets)
    entropy = -np.sum(weights[weights > 0] * np.log(weights[weights > 0]))
    return min(bound, entropy)


@_posterior_information
def test_covariance_bound_on_elongated_gaussians():
    """Strongly elongated components agree with independent NumPy log determinants.

    The error in a log determinant scales as the dimension times the condition
    number times the relative error in the matrix. Allow 16 roundings for
    centering, the two rank-one updates, scaling, Cholesky and log subtraction.
    Fails if elongated but positive-definite modes return NaN or negative gain.
    """
    rng = np.random.default_rng(3)
    weights = np.array([0.5, 0.5])
    for trial in range(200):
        scale = 10.0 ** rng.uniform(1.0, 4.5)
        # Alternate coincident and independent elongation axes.
        v = rng.normal(size=(2, 2))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        if trial % 2:
            v[1] = v[0]
        covs = scale**2 * np.einsum("ki,kj->kij", v, v) + np.eye(2)
        means = rng.normal(size=(2, 2)) * (trial % 3 > 0)
        val = eig_alias_breaking_from_covariances(weights, means, covs)
        ref = _float64_covariance_bound(weights, means, covs)
        # The spectral norm is at most scale**2 + 1 and the unit noise
        # floor bounds the inverse norm by one.
        assert 0 <= val <= np.log(2.0)
        assert abs(val - ref) <= ulps(16 * 2, scale**2 + 1.0)


@_posterior_information
def test_covariance_bound_is_exact_for_mixed_units():
    """Changing the units of two observables preserves their mode information.

    Sixteen roundings cover the two unit conversions, mixture rescaling,
    two-dimensional Cholesky and subtraction at unit log-determinant scale.
    Fails if a spread in observable units is mistaken for a singular covariance.
    """
    scales = np.array([1e-4, 1e3])  # e.g. arcsec astrometry beside m/s RV
    base = np.array([[[1.0, 0.6], [0.6, 1.0]], [[1.0, -0.6], [-0.6, 1.0]]])
    covs = base * np.outer(scales, scales)
    means = np.zeros((2, 2))
    weights = np.array([0.5, 0.5])
    for bound in (
        eig_alias_breaking_from_covariances,
        jax.jit(eig_alias_breaking_from_covariances),
    ):
        with pytest.raises(PrecisionError, match="canonical"):
            bound(weights, means, jnp.asarray(covs, jnp.float32))
    val = eig_alias_breaking_from_covariances(weights, means, covs)
    ref = _float64_covariance_bound(weights, means, base)
    assert abs(val - ref) <= ulps(16, np.float64(1.0))


# ---------------------------------------------------------------------------
# Reporting model: mode-dependent Bernoulli reporting of a Gaussian observation
# ---------------------------------------------------------------------------


def _separated_null_posterior():
    return _mixture(
        means=jnp.array([[-10.0], [10.0]]),
        covs=jnp.ones((2, 1, 1)),
        log_evidences=jnp.log(jnp.array([0.5, 0.5])),
    )


def _entropy(p):
    p = jnp.asarray(p)
    return -jnp.sum(jnp.where(p > 0.0, p * jnp.log(jnp.where(p > 0.0, p, 1.0)), 0.0))


@_posterior_information
def test_always_null_record_carries_no_information():
    """A record that is never reported is identical under every mode: zero gain.

    Two resolved modes would be worth ln 2 if observed, but a detection
    probability of zero for both means the only possible record is the same
    null. Tolerance: floating-point.
    """
    res = eig_linearized(
        _separated_null_posterior(),
        jnp.array([0.0]),
        lambda z, c: z,
        1.0,
        detectable=lambda z, c: jnp.array(0.0),
    )
    for name in ("total", "geometric", "alias"):
        assert jnp.allclose(getattr(res, name), 0.0, rtol=0.0, atol=1e-12), name


@_posterior_information
def test_always_reported_record_recovers_the_continuous_score():
    """Detection probability one everywhere reproduces the score without reporting."""
    posterior = _opposite_correlation_posterior()
    plain = eig_linearized(posterior, jnp.array([0.0]), lambda z, c: z, OPPOSITE_NOISE)
    reported = eig_linearized(
        posterior,
        jnp.array([0.0]),
        lambda z, c: z,
        OPPOSITE_NOISE,
        detectable=lambda z, c: jnp.array(1.0),
    )
    for name in ("total", "geometric", "alias"):
        assert jnp.allclose(
            getattr(plain, name), getattr(reported, name), rtol=1e-12, atol=1e-14
        ), name


@_posterior_information
def test_deterministic_binary_split_is_worth_ln2_of_mode_information():
    """One mode always reported, the other never: the report alone names the mode.

    The modes overlap in the observable, but the reporting bit already carries
    all ``H(w) = ln 2`` of mode information, and the reported value cannot add
    more. The within-mode gain accrues only for the reported mode:
    ``0.5 * 0.5*log(1 + c**2 var / R)``. Tolerance: exact-algebra.
    """
    posterior = _two_mode_posterior()  # modes at 1 and 2, unit variance
    c = 1.5
    res = eig_linearized(
        posterior,
        jnp.array([c]),
        lambda z, cand: jnp.array([cand * z[0]]),
        1.0,
        detectable=lambda z, cand: jnp.where(z[0] < 1.5, 1.0, 0.0),
    )
    assert jnp.allclose(res.alias[0], LN2, rtol=1e-12)
    assert jnp.allclose(res.geometric[0], 0.5 * 0.5 * jnp.log1p(c**2), rtol=1e-12)


def _reporting_posterior(x_means, x_vars, det_probs, weights):
    """Modes over ``z = (x, s)``: ``x`` is observed, ``s`` sets the report chance."""
    n_modes = len(weights)
    means = jnp.stack([jnp.asarray(x_means), jnp.asarray(det_probs)], axis=1)
    covs = jnp.zeros((n_modes, 2, 2))
    covs = covs.at[:, 0, 0].set(jnp.asarray(x_vars))
    covs = covs.at[:, 1, 1].set(1.0e-4)
    posterior = _mixture(
        means=means, covs=covs, log_evidences=jnp.log(jnp.asarray(weights))
    )
    return posterior


def _score_reporting(posterior, obs_var):
    return eig_linearized(
        posterior,
        jnp.array([0.0]),
        lambda z, c: z[:1],
        obs_var,
        detectable=lambda z, c: z[1],
    )


@_posterior_information
def test_informative_nondetection_matches_direct_enumeration():
    """With an uninformative value, the mode gain is exactly the report bit I(M; D).

    Every mode predicts the same observable distribution, so a reported value
    says nothing further and the nondetections carry real information. The
    reference enumerates the joint table ``p(k, D)`` directly; the restricted
    reporting model is also checked by Monte Carlo (200000 draws, seed 0).
    Tolerance: exact-algebra; monte-carlo at 4 sigma.
    """
    weights = jnp.array([0.3, 0.3, 0.4])
    det = jnp.array([0.9, 0.2, 0.5])
    posterior = _reporting_posterior([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], det, weights)
    res = _score_reporting(posterior, 0.5)

    p_bar = jnp.sum(weights * det)
    enumerated = 0.0
    for k in range(3):
        for p_given_k, p_marginal in ((det[k], p_bar), (1.0 - det[k], 1.0 - p_bar)):
            enumerated += weights[k] * p_given_k * jnp.log(p_given_k / p_marginal)
    assert enumerated > 0.05
    assert jnp.allclose(res.alias[0], enumerated, rtol=1e-10)

    mc, se = reported_mixture_mode_mi_mc(
        jax.random.PRNGKey(0),
        weights,
        jnp.zeros((3, 1)),
        jnp.full((3, 1, 1), 1.5),
        det_probs=det,
    )
    assert jnp.abs(res.alias[0] - mc) < 4.0 * se


@_posterior_information
def test_resolved_reports_match_direct_enumeration():
    """Resolved modes: a report names its mode, a null leaves the null posterior.

    Enumeration gives ``I(M; D, Y) = H(w) - (1 - p_det) H(w0)`` with
    ``w0_k`` proportional to ``w_k (1 - d_k)``: the value is worth ``H(w)`` only
    when it is seen. Tolerance: exact-algebra; monte-carlo at 4 sigma
    (200000 draws, seed 0).
    """
    weights = jnp.array([0.3, 0.3, 0.4])
    det = jnp.array([0.9, 0.2, 0.5])
    x_means = jnp.array([0.0, 60.0, 120.0])
    posterior = _reporting_posterior(x_means, [1.0, 1.0, 1.0], det, weights)
    res = _score_reporting(posterior, 0.5)

    p_bar = jnp.sum(weights * det)
    w_null = weights * (1.0 - det) / (1.0 - p_bar)
    enumerated = _entropy(weights) - (1.0 - p_bar) * _entropy(w_null)
    assert jnp.allclose(res.alias[0], enumerated, rtol=1e-10)
    assert res.alias[0] < _entropy(weights)

    mc, se = reported_mixture_mode_mi_mc(
        jax.random.PRNGKey(0),
        weights,
        x_means[:, None],
        jnp.full((3, 1, 1), 1.5),
        det_probs=det,
    )
    assert jnp.abs(res.alias[0] - mc) < 4.0 * se


@_posterior_information
def test_mixed_reporting_bounds_the_monte_carlo_mode_information():
    """Overlapping modes with mixed reporting: the score bounds the exact MI.

    Reference: a Monte Carlo of ``I(M; D, Y)`` under the restricted reporting
    model (200000 draws, seed 0). The within-mode gain is the reporting-weighted
    conjugate Gaussian gain ``sum_k w_k d_k 0.5*log(1 + var_k / R)``.
    Tolerance: monte-carlo at 4 sigma; exact-algebra for the within-mode term.
    """
    weights = jnp.array([0.25, 0.45, 0.3])
    det = jnp.array([0.85, 0.15, 0.6])
    x_means = jnp.array([0.0, 1.0, 2.5])
    x_vars = jnp.array([0.4, 1.2, 0.7])
    obs_var = 0.3
    posterior = _reporting_posterior(x_means, x_vars, det, weights)
    res = _score_reporting(posterior, obs_var)

    mc, se = reported_mixture_mode_mi_mc(
        jax.random.PRNGKey(0),
        weights,
        x_means[:, None],
        (x_vars + obs_var)[:, None, None],
        det_probs=det,
    )
    assert res.alias[0] >= mc - 4.0 * se
    assert res.alias[0] <= _entropy(weights) + 1e-12
    within = jnp.sum(weights * det * 0.5 * jnp.log1p(x_vars / obs_var))
    assert jnp.allclose(res.geometric[0], within, rtol=1e-8)


# ---------------------------------------------------------------------------
# Golden: the result before the Mixture input port
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).parent / "data" / "eig_mixture_golden.json"


def _golden_forward(z, c):
    return jnp.array([c * z[0] + z[1], z[1] - 0.5 * c * z[2]])


def _golden_detectable(z, c):
    return jax.nn.sigmoid(z[0] - 0.2 * c)


@pytest.mark.float64_only(reason="stored float64 V&V reference values")
@pytest.mark.parametrize("name", ["two_modes_plain", "three_modes_reporting"])
def test_eig_linearized_matches_golden_of_the_legacy_input(name):
    """The Mixture input reproduces the result captured from the legacy input.

    The golden (``scripts/one_stack/capture_eig_mixture_golden.py`` on the base
    commit) holds the full result as the fields of ``LinearizedEIG``.
    Tolerance: rtol 1e-12, since the Cholesky round trip of the covariances
    changes them by about 1e-16.
    """
    fixture = json.loads(GOLDEN.read_text())[name]
    mixture = _mixture(fixture["means"], fixture["covs"], fixture["log_weights"])
    res = eig_linearized(
        mixture,
        jnp.asarray(fixture["candidates"]),
        _golden_forward,
        0.5,
        detectable=_golden_detectable if fixture["reporting"] else None,
    )
    assert set(fixture["result"]) == set(LinearizedEIG.__dataclass_fields__)
    for key, expected in fixture["result"].items():
        np.testing.assert_allclose(
            np.asarray(getattr(res, key)),
            np.asarray(expected),
            rtol=1e-12,
            atol=1e-13,
        )


@pytest.mark.float32_only(reason="float64 posterior information refuses this process")
def test_posterior_information_refuses_a_float32_process():
    """Fails if an estimator reduces weights or solves a covariance in float32."""
    w = jnp.ones(1)
    y = jnp.ones((1, 1))
    covs = y[None, :, :]
    key = jax.random.key(0)
    for fn, args, kwargs in [
        (eig_geometric, (y, y, 1.0), {}),
        (eig_alias_breaking_from_covariances, (w, y, covs), {}),
        (eig_alias_breaking, (w, y, 1.0), {}),
        (eig_detectability, (w, w), {}),
        (eig_detection_class, (w, w, y), {}),
        (eig_class, (w, y, y, 1.0), {"key": key}),
        (eig_detection_channel, (w, y, covs, 0.0, w), {}),
        (eig_linearized, (_two_mode_posterior(), w, _golden_forward, 1.0), {}),
        (eig_contrastive, (None, None, key), {"n_outer": 1}),
        (eig_nested, (None, None, key), {"n_outer": 1, "n_inner": 1, "min_n_eff": 1}),
    ]:
        with pytest.raises(PrecisionError, match=fn.__name__):
            fn(*args, **kwargs)
