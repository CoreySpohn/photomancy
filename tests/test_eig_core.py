"""Tests for the domain-agnostic EIG layer."""

import jax
import jax.numpy as jnp
from nmc_reference import class_mi_mc, mixture_mode_mi_mc

from photomancy.eig import (
    alias_breaking_eig,
    class_eig,
    detectability_eig,
    detection_class_eig,
    evaluate_candidates,
    geometric_eig,
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
