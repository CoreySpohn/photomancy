"""Domain-agnostic analytic Expected Information Gain over a Posterior.

Scores candidate observations from a ``MixturePosterior`` (per-mode Gaussians) as
estimator tiers of one mutual information ``I((M, D, theta); Y)``: the Gaussian tier is
the analytic Laplace covariance update (``Sigma^-1 += J^T R^-1 J``), the discrete tier
carries the exact detection-channel MI, the ``H(w)``-capped alias bound, and the class
information ``I(C; Y)`` over caller-supplied per-mode class weights. The forward model
is a caller-supplied ``forward(z, candidate) -> observables`` (differentiable in the
flat position ``z``); the orbit / disk / imaging layers supply their own. This core
knows no domain.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jax.scipy.stats import norm


def _binary_entropy(p):
    """Entropy of a Bernoulli(p) in nats; exact 0 at p in {0, 1} via xlogy."""
    return -xlogy(p, p) - xlogy(1.0 - p, 1.0 - p)


def _categorical_entropy(p, axis=-1):
    """Entropy of a categorical distribution along ``axis`` in nats."""
    return -jnp.sum(xlogy(p, p), axis=axis)


def geometric_eig(cov_old, jacobian, obs_variance, qoi_projection=None):
    """Information gain from covariance shrinkage at one mode (nats).

    Args:
        cov_old: Current covariance. Shape ``(d, d)``.
        jacobian: Forward-model Jacobian ``dy/dz``. Shape ``(n_obs, d)``.
        obs_variance: Scalar or ``(n_obs,)`` measurement variances.
        qoi_projection: Optional QoI selector ``Phi``. Shape ``(q, d)``. When given,
            the gain is the marginal-covariance log-det difference
            ``0.5*[logdet(Phi Sigma Phi^T) - logdet(Phi Sigma' Phi^T)]``, so shrinking
            a nuisance block is not booked as gain. The covariance update itself stays
            full-space.

    Returns:
        ``(eig, cov_new)`` -- scalar information gain and the updated covariance.
    """
    prec_old = jnp.linalg.inv(cov_old)
    obs_var = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(obs_variance)), (jacobian.shape[0],)
    )
    fim = jacobian.T @ jnp.diag(1.0 / obs_var) @ jacobian
    cov_new = jnp.linalg.inv(prec_old + fim)
    if qoi_projection is None:
        eig = 0.5 * (jnp.linalg.slogdet(cov_old)[1] - jnp.linalg.slogdet(cov_new)[1])
    else:
        phi = jnp.asarray(qoi_projection)
        marg_old = phi @ cov_old @ phi.T
        marg_new = phi @ cov_new @ phi.T
        eig = 0.5 * (jnp.linalg.slogdet(marg_old)[1] - jnp.linalg.slogdet(marg_new)[1])
    return eig, cov_new


def alias_breaking_eig(weights, y_preds, obs_variance):
    """Mode-discrimination information from the continuous channel (nats).

    A per-dimension moment-matched upper bound on ``I(M; Y)`` intersected with the
    exact bound ``I(M; Y) <= H(w)`` -- two well-separated 50/50 modes are worth
    ``ln 2``, not ``log(separation)``.

    Args:
        weights: Mode weights. Shape ``(K,)``.
        y_preds: Predictions at each mode. Shape ``(K, n_obs)``.
        obs_variance: Measurement variance ``R`` as a scalar or ``(n_obs,)``, or the
            per-mode predictive variances ``diag(J_k Sigma_k J_k^T) + R`` as
            ``(K, n_obs)`` so within-mode posterior spread widens the predictive
            instead of being ignored.

    Returns:
        Scalar alias-breaking EIG.
    """
    y_preds = jnp.asarray(y_preds)
    pred_var = jnp.broadcast_to(jnp.asarray(obs_variance), y_preds.shape)
    y_mean = jnp.sum(weights[:, None] * y_preds, axis=0)
    y_var = jnp.sum(weights[:, None] * (y_preds - y_mean[None, :]) ** 2, axis=0)
    within = jnp.sum(weights[:, None] * pred_var, axis=0)
    log_within = jnp.sum(weights[:, None] * jnp.log(pred_var), axis=0)
    bound = 0.5 * jnp.sum(jnp.log(y_var + within) - log_within)
    return jnp.minimum(bound, _categorical_entropy(weights))


def detectability_eig(weights, det_weights):
    """Exact detection-channel mutual information ``I(D; M)`` (nats).

    ``I(D; M) = H_b(sum_k w_k d_k) - sum_k w_k H_b(d_k)`` for the binary detection
    outcome ``D`` -- closed form, saturating at ``min(H_b(p_det), H(w))``. (The former
    variance surrogate ``0.5*log(1 + Var_w(d)/0.25)`` undercounted by up to 2x.)

    Args:
        weights: Mode weights. Shape ``(K,)``.
        det_weights: Per-mode detection probability in ``[0, 1]``. Shape ``(K,)``.

    Returns:
        Scalar detection-channel EIG.
    """
    p_det = jnp.sum(weights * det_weights)
    return _binary_entropy(p_det) - jnp.sum(weights * _binary_entropy(det_weights))


def detection_class_eig(weights, det_weights, class_probs):
    """Closed-form classification gain ``I(C; D)`` of a detection-only observation.

    The ``Y = D`` special case of :func:`class_eig`, exact:
    ``P(c | D) propto sum_k w_k d_k^D (1 - d_k)^(1 - D) P(c | k)``. Saturates at the
    class-prior entropy ``H(C)`` and reduces to :func:`detectability_eig` when the
    class map is the mode identity.

    Args:
        weights: Mode weights. Shape ``(K,)``.
        det_weights: Per-mode detection probability in ``[0, 1]``. Shape ``(K,)``.
        class_probs: Per-mode class weights ``P(c | k)``. Shape ``(K, C)``.

    Returns:
        Scalar classification EIG.
    """
    p_det = jnp.sum(weights * det_weights)
    tiny = jnp.finfo(jnp.result_type(p_det)).tiny
    q_prior = weights @ class_probs
    q_det = ((weights * det_weights) @ class_probs) / jnp.maximum(p_det, tiny)
    q_non = ((weights * (1.0 - det_weights)) @ class_probs) / jnp.maximum(
        1.0 - p_det, tiny
    )
    h_cond = p_det * _categorical_entropy(q_det) + (1.0 - p_det) * _categorical_entropy(
        q_non
    )
    return _categorical_entropy(q_prior) - h_cond


def class_eig(weights, class_probs, y_preds, obs_variance, *, key, n_samples=64):
    """Classification gain ``I(C; Y)`` of a candidate observation (nats).

    The mixture-predictive reweighting estimator: the class posterior after seeing
    ``y`` is ``sum_k w_k(y) P(c | k)`` with ``w_k(y) propto w_k N(y; y_k, S_k)``, and

    ``I(C; Y) = H(sum_k w_k P(. | k)) - E_{y ~ pbar}[H(sum_k w_k(y) P(. | k))]``,

    estimated with ``n_samples`` stratified draws per mode from the per-mode
    predictive (exact stratification over the mixture index). Saturates at the
    class-prior entropy by construction; when modes are class-pure the data
    processing inequality gives ``I(C; Y) <= I(M; Y)``, so the capped alias term is
    its upper surrogate. The QoI semantics live entirely in the caller-supplied
    ``class_probs``; this core knows no domain.

    Args:
        weights: Mode weights. Shape ``(K,)``.
        class_probs: Per-mode class weights ``P(c | k)``. Shape ``(K, C)``.
        y_preds: Predictions at each mode. Shape ``(K, n_obs)``.
        obs_variance: Measurement variance ``R`` as a scalar or ``(n_obs,)``, or the
            per-mode predictive variances ``diag(J_k Sigma_k J_k^T) + R`` as
            ``(K, n_obs)``.
        key: PRNG key for the stratified predictive draws.
        n_samples: Draws per mode.

    Returns:
        Scalar classification EIG (clipped at 0, the estimand's exact floor).
    """
    y_preds = jnp.asarray(y_preds)
    class_probs = jnp.asarray(class_probs)
    pred_var = jnp.broadcast_to(jnp.asarray(obs_variance), y_preds.shape)
    pred_sd = jnp.sqrt(pred_var)
    log_w = jnp.where(
        weights > 0.0, jnp.log(jnp.where(weights > 0.0, weights, 1.0)), -jnp.inf
    )

    eps = jax.random.normal(key, (n_samples, *y_preds.shape))
    ys = y_preds[None, :, :] + pred_sd[None, :, :] * eps

    def _posterior_class_entropy(y):
        log_lik = jnp.sum(norm.logpdf(y[None, :], y_preds, pred_sd), axis=-1)
        w_post = jax.nn.softmax(log_w + log_lik)
        return _categorical_entropy(w_post @ class_probs)

    h_post = jax.vmap(jax.vmap(_posterior_class_entropy))(ys)
    expected_h = jnp.sum(weights * jnp.mean(h_post, axis=0))
    h_prior = _categorical_entropy(weights @ class_probs)
    return jnp.maximum(h_prior - expected_h, 0.0)


_EIG_JIT_CACHE: dict = {}


def _build_eig_batch_fn(forward, detectable=None):
    """JIT-compile the batched EIG once for a given forward / detectable pair."""

    def _pred_and_jac(z, candidate):
        y = forward(z, candidate)
        jac = jax.jacrev(lambda zz: forward(zz, candidate))(z)
        return y, jac

    def _one_candidate(candidate, means, covs, weights, obs_var):
        y_preds, jacobians = jax.vmap(lambda z: _pred_and_jac(z, candidate))(means)
        geom_eigs = jax.vmap(lambda cov, jac: geometric_eig(cov, jac, obs_var)[0])(
            covs, jacobians
        )
        # Posterior-width-aware per-mode predictive: diag(J_k Sigma_k J_k^T) + R.
        pred_var = (
            jax.vmap(lambda jac, cov: jnp.einsum("ij,jk,ik->i", jac, cov, jac))(
                jacobians, covs
            )
            + obs_var
        )
        if detectable is None:
            det_w = jnp.ones(means.shape[0])
            weighted_geom = jnp.sum(weights * geom_eigs)
            alias_val = alias_breaking_eig(weights, y_preds, pred_var)
        else:
            det_w = jax.vmap(lambda z: detectable(z, candidate))(means)
            weighted_geom = jnp.sum(weights * det_w * geom_eigs)
            alias_val = alias_breaking_eig(
                weights, y_preds, pred_var
            ) + detectability_eig(weights, det_w)
        return weighted_geom + alias_val, weighted_geom, alias_val, y_preds, det_w

    @jax.jit
    def _batch(candidates, means, covs, weights, obs_var):
        return jax.vmap(lambda c: _one_candidate(c, means, covs, weights, obs_var))(
            candidates
        )

    return _batch


def evaluate_candidates(
    posterior, candidates, forward, obs_variance, *, detectable=None, cache_key=None
):
    """Analytic EIG for a batch of candidate observations against a mixture posterior.

    Args:
        posterior: A ``MixturePosterior`` (``means (K, d)``, ``covs (K, d, d)``,
            ``log_evidences (K,)``).
        candidates: Batch of candidate designs (opaque to the core). Shape ``(N, ...)``.
        forward: ``forward(z, candidate) -> y (n_obs,)``, differentiable in flat ``z``.
        obs_variance: Scalar or ``(n_obs,)`` measurement variance.
        detectable: Optional ``detectable(z, candidate) -> float`` in ``[0, 1]``;
            non-detectable modes contribute no astrometric information and the exact
            detection-channel MI is added. When omitted, all modes detectable.
        cache_key: Optional hashable key; when given, the compiled batch function is
            cached and reused (so the warm path stays fast).

    Returns:
        Dict with ``total_eig (N,)``, ``geometric_eig (N,)``, ``alias_eig (N,)``,
        ``predictions (N, K, n_obs)``, ``detectability (N, K)``.
    """
    obs_var = jnp.atleast_1d(jnp.asarray(obs_variance))
    batch_fn = None
    if cache_key is not None:
        batch_fn = _EIG_JIT_CACHE.get(cache_key)
    if batch_fn is None:
        batch_fn = _build_eig_batch_fn(forward, detectable)
        if cache_key is not None:
            _EIG_JIT_CACHE[cache_key] = batch_fn

    weights = jnp.exp(posterior.log_weights)
    total, geom, alias, preds, det_w = batch_fn(
        jnp.asarray(candidates), posterior.means, posterior.covs, weights, obs_var
    )
    return {
        "total_eig": total,
        "geometric_eig": geom,
        "alias_eig": alias,
        "predictions": preds,
        "detectability": det_w,
    }
