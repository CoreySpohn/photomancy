"""Tests for the domain-agnostic EIG layer."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402
from nmc_reference import (  # noqa: E402
    class_mi_mc,
    detection_channel_mi_mc,
    mixture_mode_mi_mc,
    probit_site_moments_mc,
)

from photomancy.eig import (  # noqa: E402
    alias_breaking_eig,
    class_eig,
    detectability_eig,
    detection_channel_eig,
    detection_class_eig,
    ep_probit_update,
    evaluate_candidates,
    geometric_eig,
    null_update,
    probit_gaussian_mass,
)
from photomancy.posterior import MixturePosterior


def _two_mode_posterior():
    return MixturePosterior(
        means=jnp.array([[1.0], [2.0]]),
        covs=jnp.array([[[1.0]], [[1.0]]]),
        log_evidences=jnp.array([0.0, 0.0]),
    )


def test_geometric_eig_matches_analytic_gaussian_update():
    """The log-det shrinkage equals the 1-D conjugate-Gaussian information gain."""
    eig, cov_new = geometric_eig(jnp.array([[4.0]]), jnp.array([[1.0]]), 1.0)
    assert jnp.allclose(eig, 0.5 * jnp.log(5.0))
    assert float(cov_new[0, 0]) < 4.0


def test_alias_breaking_eig_zero_when_modes_agree():
    """Zero when modes predict the same observable; positive when they differ."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(
        alias_breaking_eig(weights, jnp.array([[1.0], [1.0]]), 1.0), 0.0
    )
    assert alias_breaking_eig(weights, jnp.array([[1.0], [5.0]]), 1.0) > 0.0


def test_detectability_eig_zero_when_modes_agree():
    """Zero when all modes agree on detectability; positive when split."""
    weights = jnp.array([0.5, 0.5])
    assert jnp.allclose(detectability_eig(weights, jnp.array([1.0, 1.0])), 0.0)
    assert detectability_eig(weights, jnp.array([1.0, 0.0])) > 0.0


def test_evaluate_candidates_linear_forward_more_informative_scores_higher():
    """A larger Jacobian and wider mode disagreement give more information."""
    posterior = _two_mode_posterior()

    def forward(z, c):
        return jnp.array([c * z[0]])

    res = evaluate_candidates(posterior, jnp.array([0.5, 2.0]), forward, 1.0)
    assert res["total_eig"].shape == (2,)
    assert jnp.all(jnp.isfinite(res["total_eig"]))
    assert jnp.all(res["total_eig"] >= -1e-6)
    assert res["total_eig"][1] > res["total_eig"][0]
    assert res["predictions"].shape == (2, 2, 1)


def test_evaluate_candidates_detectable_splits_modes():
    """A detectable callback that splits the modes drives the detectability output."""
    posterior = _two_mode_posterior()

    def forward(z, c):
        return jnp.array([c * z[0]])

    def detectable(z, c):
        return jnp.where(z[0] < 1.5, 1.0, 0.0)  # mode 0 detectable, mode 1 not

    res = evaluate_candidates(
        posterior, jnp.array([1.0]), forward, 1.0, detectable=detectable
    )
    assert res["detectability"].shape == (1, 2)
    assert jnp.allclose(res["detectability"][0], jnp.array([1.0, 0.0]))
    assert jnp.isfinite(res["total_eig"][0])


# ---------------------------------------------------------------------------
# Exact discrete-tier forms (the unified-currency upgrades)
# ---------------------------------------------------------------------------

LN2 = float(jnp.log(2.0))


def test_detectability_eig_exact_at_deterministic_split():
    """A deterministic 50/50 detection split carries exactly ln 2 nats.

    I(D; M) = H_b(sum w_k d_k) - sum w_k H_b(d_k); the old variance surrogate
    0.5*log(1 + Var_w(d)/0.25) reported 0.5*log(2) here -- a 2x undercount.
    """
    weights = jnp.array([0.5, 0.5])
    det = jnp.array([0.0, 1.0])
    assert jnp.allclose(detectability_eig(weights, det), LN2)


def test_detectability_eig_bounded_by_mode_entropy():
    """I(D; M) <= H(w): the detection bit cannot exceed the mode uncertainty."""
    weights = jnp.array([0.9, 0.1])
    h_w = -jnp.sum(weights * jnp.log(weights))
    det = jnp.array([0.0, 1.0])
    val = detectability_eig(weights, det)
    assert val <= h_w + 1e-12
    assert val > 0.0


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
    assert jnp.allclose(detectability_eig(weights, det), direct, rtol=1e-10)


def test_alias_breaking_eig_capped_at_mode_entropy():
    """Two perfectly resolved 50/50 modes are worth ln 2 nats, not log(separation).

    The moment-matched bound 0.5*log(1 + Var_w/sigma^2) grows without bound as
    modes separate; the true I(M; Y) <= H(w).
    """
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [100.0]])
    assert jnp.allclose(alias_breaking_eig(weights, y_preds, 1.0), LN2)

    # Far below full separation the bound itself (not the cap) is active.
    y_close = jnp.array([[0.0], [0.5]])
    val = alias_breaking_eig(weights, y_close, 1.0)
    assert val < LN2
    assert jnp.allclose(val, 0.5 * jnp.log1p(0.0625), rtol=1e-10)


def test_alias_breaking_eig_uses_per_mode_predictive_widths():
    """(K, n_obs) obs_variance = per-mode predictive J Sigma J^T + R.

    Wide within-mode predictives swallow the mode separation: the alias term
    must shrink relative to the R-only evaluation.
    """
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [1.0]])
    narrow = alias_breaking_eig(weights, y_preds, 0.01)
    pred_var = jnp.array([[1.01], [1.01]])  # J Sigma J^T = 1.0 on top of R = 0.01
    wide = alias_breaking_eig(weights, y_preds, pred_var)
    assert wide < narrow
    expected = 0.5 * (jnp.log(0.25 + 1.01) - jnp.log(1.01))
    assert jnp.allclose(wide, expected, rtol=1e-10)


def test_alias_breaking_eig_distinguishes_unequal_widths():
    """Modes with equal means but different predictive widths are informative."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [0.0]])
    pred_var = jnp.array([[0.01], [4.0]])
    assert alias_breaking_eig(weights, y_preds, pred_var) > 0.1


def test_geometric_eig_qoi_projection_ignores_nuisance_gain():
    """A nuisance-only observation shows zero marginal gain on the QoI block.

    EIG_g = 0.5*[logdet(Phi Sigma Phi^T) - logdet(Phi Sigma' Phi^T)] with Phi the
    QoI selector; the full-theta logdet would book the nuisance shrinkage as gain.
    """
    cov = jnp.eye(2)
    jac_nuisance = jnp.array([[0.0, 1.0]])  # measures only parameter 1
    phi_qoi = jnp.array([[1.0, 0.0]])  # QoI is parameter 0

    eig_full, _ = geometric_eig(cov, jac_nuisance, 0.01)
    assert eig_full > 1.0  # full-theta logdet books the nuisance shrinkage

    eig_marg, cov_new = geometric_eig(cov, jac_nuisance, 0.01, qoi_projection=phi_qoi)
    assert jnp.allclose(eig_marg, 0.0, atol=1e-10)
    assert cov_new.shape == (2, 2)  # update itself stays full-space


def test_geometric_eig_identity_projection_matches_full():
    """Phi = I reproduces the unprojected gain."""
    cov = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    jac = jnp.array([[1.0, 0.5], [0.2, 2.0]])
    eig_full, _ = geometric_eig(cov, jac, 0.5)
    eig_proj, _ = geometric_eig(cov, jac, 0.5, qoi_projection=jnp.eye(2))
    assert jnp.allclose(eig_full, eig_proj, rtol=1e-10)


def test_geometric_eig_correlated_qoi_marginal_gain_positive():
    """A correlated nuisance measurement still informs the QoI marginal."""
    cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    jac = jnp.array([[0.0, 1.0]])
    phi = jnp.array([[1.0, 0.0]])
    eig_marg, _ = geometric_eig(cov, jac, 0.01, qoi_projection=phi)
    # Sigma'_00 = 1 - 0.8^2 * (1 / (1 + 0.01)) -> gain ~ 0.5 log(1/0.366)
    assert eig_marg > 0.4


# ---------------------------------------------------------------------------
# Nested-MC reference cross-checks (the tier-Z calibrator in miniature)
# ---------------------------------------------------------------------------


def test_alias_bound_upper_bounds_true_mixture_mi():
    """The capped alias term is a valid upper bound on the exact I(M; Y).

    Checked against a brute-force MC of the mixture MI across the overlap ->
    resolved range; the bound must sit above the MC truth at every separation.
    """
    weights = jnp.array([0.5, 0.5])
    variances = jnp.ones((2, 1))
    for i, sep in enumerate([0.5, 1.0, 2.0, 4.0, 20.0]):
        y_preds = jnp.array([[0.0], [sep]])
        bound = alias_breaking_eig(weights, y_preds, variances)
        mc, se = mixture_mode_mi_mc(jax.random.PRNGKey(i), weights, y_preds, variances)
        # 1e-12 covers the ulp-equality case where both saturate at exactly H(w).
        assert bound >= mc - 4.0 * se - 1e-12, f"bound violated at sep={sep}"


def test_alias_cap_tight_in_resolved_limit():
    """At full separation the MC truth reaches H(w) and the cap is exact."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [50.0]])
    variances = jnp.ones((2, 1))
    bound = alias_breaking_eig(weights, y_preds, variances)
    mc, se = mixture_mode_mi_mc(jax.random.PRNGKey(0), weights, y_preds, variances)
    assert jnp.allclose(bound, LN2)
    assert jnp.abs(mc - LN2) < 4.0 * se + 1e-3


def test_detectability_eig_matches_binary_channel_limits():
    """Exact detection MI hits both closed-form limits of the binary channel."""
    weights = jnp.array([0.5, 0.5])
    # Deterministic split: ln 2. Noisy but symmetric d = {0.2, 0.8}:
    # I = H_b(0.5) - H_b(0.2) (binary symmetric channel with crossover 0.2).
    hb = lambda p: -p * jnp.log(p) - (1 - p) * jnp.log(1 - p)  # noqa: E731
    val = detectability_eig(weights, jnp.array([0.2, 0.8]))
    assert jnp.allclose(val, LN2 - hb(0.2), rtol=1e-10)


# ---------------------------------------------------------------------------
# class_eig: classification gain I(C; Y) over caller-supplied P(c | k)
# ---------------------------------------------------------------------------


def test_detection_class_eig_identity_classes_equals_detectability():
    """With C = M (identity class map) the closed form reduces to I(D; M)."""
    weights = jnp.array([0.3, 0.7])
    det = jnp.array([0.1, 0.9])
    val = detection_class_eig(weights, det, jnp.eye(2))
    assert jnp.allclose(val, detectability_eig(weights, det), rtol=1e-10)


def test_detection_class_eig_merged_classes_lose_information():
    """Merging modes into one class destroys the detection information (DPI)."""
    weights = jnp.array([0.5, 0.5])
    det = jnp.array([0.0, 1.0])
    one_class = jnp.array([[1.0], [1.0]])
    assert jnp.allclose(detection_class_eig(weights, det, one_class), 0.0, atol=1e-12)

    # Three modes, two classes: I(C; D) <= I(M; D).
    weights3 = jnp.array([0.3, 0.4, 0.3])
    det3 = jnp.array([0.05, 0.6, 0.95])
    classes3 = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert (
        detection_class_eig(weights3, det3, classes3)
        <= detectability_eig(weights3, det3) + 1e-12
    )


def test_detection_class_eig_certain_detection_is_uninformative():
    """All modes surely detected -> the detection bit carries nothing."""
    weights = jnp.array([0.5, 0.5])
    det = jnp.ones(2)
    assert jnp.allclose(detection_class_eig(weights, det, jnp.eye(2)), 0.0, atol=1e-12)


def test_class_eig_saturates_at_class_prior_entropy():
    """I(C; Y) <= H(C): a huge-separation observation yields H(C), not more."""
    weights = jnp.array([0.5, 0.5])
    class_probs = jnp.eye(2)
    y_preds = jnp.array([[0.0], [200.0]])
    val = class_eig(
        weights, class_probs, y_preds, 1.0, key=jax.random.PRNGKey(0), n_samples=256
    )
    h_class = -jnp.sum(0.5 * jnp.log(0.5) * jnp.ones(2))
    assert val <= h_class + 1e-9
    assert jnp.allclose(val, LN2, atol=5e-3)


def test_class_eig_identical_predictions_zero():
    """Modes that predict identically cannot classify: I(C; Y) ~ 0."""
    weights = jnp.array([0.5, 0.5])
    val = class_eig(
        weights,
        jnp.eye(2),
        jnp.array([[1.0], [1.0]]),
        1.0,
        key=jax.random.PRNGKey(0),
        n_samples=256,
    )
    assert jnp.allclose(val, 0.0, atol=1e-9)


def test_class_eig_matches_nmc_reference_mid_separation():
    """The stratified estimator agrees with brute-force nested MC of I(C; Y)."""
    weights = jnp.array([0.4, 0.35, 0.25])
    class_probs = jnp.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    y_preds = jnp.array([[0.0, 0.5], [1.5, -0.5], [3.0, 1.0]])
    variances = jnp.array([[1.0, 2.0], [1.5, 1.0], [1.0, 1.0]])
    val = class_eig(
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


def test_class_eig_dpi_under_capped_alias():
    """Class-pure modes: I(C; Y) <= I(M; Y) <= capped alias bound (DPI chain)."""
    weights = jnp.array([0.5, 0.5])
    y_preds = jnp.array([[0.0], [2.0]])
    variances = jnp.ones((2, 1))
    val = class_eig(
        weights,
        jnp.eye(2),
        y_preds,
        variances,
        key=jax.random.PRNGKey(0),
        n_samples=4096,
    )
    bound = alias_breaking_eig(weights, y_preds, variances)
    assert val <= bound + 5e-3


def test_class_eig_wider_separation_more_informative():
    """Monotone in mode separation for class-pure modes."""
    weights = jnp.array([0.5, 0.5])
    vals = [
        class_eig(
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
    mass = probit_gaussian_mass(mean, cov, a, b)
    mc_mass, _, _ = probit_site_moments_mc(jax.random.PRNGKey(0), mean, cov, a, b)
    assert jnp.abs(mass - mc_mass) < 5.0e-3


def test_ep_probit_update_matches_mc_tilted_moments():
    """The EP site update returns the exact tilted mean and covariance."""
    mean = jnp.array([0.4, -0.2])
    cov = jnp.array([[0.8, 0.3], [0.3, 0.6]])
    a, b = -0.5, jnp.array([1.5, 0.4])
    log_mass, mean_new, cov_new = ep_probit_update(mean, cov, a, b)
    mc_mass, mc_mean, mc_cov = probit_site_moments_mc(
        jax.random.PRNGKey(0), mean, cov, a, b, n_samples=400000
    )
    assert jnp.abs(jnp.exp(log_mass) - mc_mass) < 5.0e-3
    assert jnp.all(jnp.abs(mean_new - mc_mean) < 2.0e-2)
    assert jnp.all(jnp.abs(cov_new - mc_cov) < 2.0e-2)


def test_ep_probit_update_hard_truncation_limit():
    """A steep site reproduces the truncated-Gaussian Mills-ratio moments.

    For X ~ N(mu, sigma^2) truncated to X <= c: with alpha = (c - mu)/sigma and
    lam = phi(alpha)/Phi(alpha), E[X] = mu - sigma*lam and
    Var[X] = sigma^2 (1 - alpha*lam - lam^2).
    """
    mu, sigma, c = 1.0, 0.7, 1.3
    steep = 400.0  # site Phi(steep * (c - x)) -> indicator{x <= c}
    log_mass, mean_new, cov_new = ep_probit_update(
        jnp.array([mu]), jnp.array([[sigma**2]]), steep * c, jnp.array([-steep])
    )
    alpha = (c - mu) / sigma
    lam = jnp.exp(norm.logpdf(alpha) - norm.logcdf(alpha))
    assert jnp.allclose(mean_new[0], mu - sigma * lam, rtol=1.0e-3)
    assert jnp.allclose(
        cov_new[0, 0], sigma**2 * (1.0 - alpha * lam - lam**2), rtol=5.0e-3
    )
    assert jnp.allclose(jnp.exp(log_mass), norm.cdf(alpha), rtol=1.0e-3)


def test_ep_probit_update_flat_site_is_identity():
    """A site that is ~1 over the bulk leaves mass and moments unchanged."""
    mean = jnp.array([0.0])
    cov = jnp.array([[1.0]])
    log_mass, mean_new, cov_new = ep_probit_update(mean, cov, 30.0, jnp.array([0.1]))
    assert jnp.exp(log_mass) > 0.999
    assert jnp.allclose(mean_new, mean, atol=1.0e-4)
    assert jnp.allclose(cov_new, cov, rtol=1.0e-3)


def test_null_update_kills_the_detectable_mode():
    """After an observed null, the solidly detectable mode loses its weight.

    Detection site p_det = Phi(a + b theta): mode at theta = +3 is deep in the
    detectable regime, mode at theta = -3 is deep in the undetectable regime.
    """
    weights = jnp.array([0.5, 0.5])
    means = jnp.array([[3.0], [-3.0]])
    covs = jnp.array([[[0.2]], [[0.2]]])
    a, b = 0.0, jnp.array([3.0])
    w_new, m_new, c_new = null_update(weights, means, covs, a, b)
    assert w_new[1] > 0.999
    assert jnp.allclose(jnp.sum(w_new), 1.0, rtol=1.0e-9)
    # The surviving mode was never near the boundary: moments unchanged.
    assert jnp.allclose(m_new[1], means[1], atol=1.0e-3)


def test_null_update_shaves_a_mode_cut_by_the_limit():
    """A mode straddling the detection boundary is shifted and narrowed."""
    weights = jnp.array([1.0])
    means = jnp.array([[0.0]])
    covs = jnp.array([[[1.0]]])
    a, b = 0.0, jnp.array([2.0])  # boundary right through the mode
    w_new, m_new, c_new = null_update(weights, means, covs, a, b)
    assert m_new[0, 0] < -0.3  # pushed toward the undetectable side
    assert c_new[0, 0, 0] < 1.0  # tail shaved


def test_detection_channel_eig_reduces_to_detectability_eig_at_zero_width():
    """With vanishing within-mode spread the channel MI is the discrete I(D; M)."""
    weights = jnp.array([0.4, 0.6])
    means = jnp.array([[1.0], [-1.5]])
    covs = jnp.array([[[1.0e-8]], [[1.0e-8]]])
    a, b = 0.2, jnp.array([1.3])
    val = detection_channel_eig(weights, means, covs, a, b)
    d_k = norm.cdf(a + means @ b)
    assert jnp.allclose(val, detectability_eig(weights, d_k), atol=1.0e-6)


def test_detection_channel_eig_adds_within_mode_information():
    """The chain rule adds a nonnegative within-mode term to the smeared I(D; M)."""
    weights = jnp.array([0.5, 0.5])
    means = jnp.array([[0.5], [-0.5]])
    covs = jnp.array([[[1.5]], [[1.5]]])
    a, b = 0.0, jnp.array([2.0])
    val = detection_channel_eig(weights, means, covs, a, b)
    d_smeared = jax.vmap(lambda m, c: probit_gaussian_mass(m, c, a, b))(means, covs)
    discrete_only = detectability_eig(weights, d_smeared)
    assert val >= discrete_only - 1.0e-12
    assert val > discrete_only + 1.0e-3  # boundary cuts both modes: strictly more


def test_detection_channel_eig_matches_nmc():
    """The Gauss-Hermite channel MI agrees with brute-force nested MC."""
    weights = jnp.array([0.35, 0.65])
    means = jnp.array([[0.8, 0.0], [-0.6, 0.5]])
    covs = jnp.array([[[0.7, 0.2], [0.2, 0.5]], [[1.1, -0.3], [-0.3, 0.9]]])
    a_arr = jnp.array([0.3, -0.2])
    b_arr = jnp.array([[1.4, -0.5], [0.9, 1.1]])
    val = detection_channel_eig(weights, means, covs, a_arr, b_arr)
    mc, se = detection_channel_mi_mc(
        jax.random.PRNGKey(0), weights, means, covs, a_arr, b_arr
    )
    assert jnp.abs(val - mc) < 4.0 * se + 2.0e-3


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

    val = detection_channel_eig(weights, means, covs, a, b)
    assert jnp.abs(val - LN2) < 1.0e-3

    def bernoulli_fim(theta):
        z = a + b[0] * theta
        dp = b[0] * norm.pdf(z)
        return dp**2 / (norm.cdf(z) * norm.cdf(-z))  # p(1-p), saturation-stable

    assert bernoulli_fim(6.0) < 1.0e-20
    assert bernoulli_fim(-6.0) < 1.0e-20
