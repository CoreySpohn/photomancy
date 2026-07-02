"""Monte Carlo mutual-information references for the analytic EIG forms.

Test-only utilities (not public API): brute-force estimators of the same mutual
informations the closed forms in ``photomancy.eig`` compute, for validating
exactness and upper-bound properties on small toys. Each estimator returns
``(estimate, standard_error)`` so tests can assert within MC error bars.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm


def _mode_loglik(y, means, sds):
    """Per-mode diagonal-Gaussian log-likelihoods of one draw. Shape ``(K,)``."""
    return jnp.sum(norm.logpdf(y[None, :], means, sds), axis=-1)


def mixture_mode_mi_mc(key, weights, means, variances, n_samples=20000):
    """MC estimate of I(M; Y) for a diagonal Gaussian mixture.

    I(M; Y) = E_{k~w, y~p_k}[log p(y | k) - log pbar(y)] with every density exact.

    Args:
        key: PRNG key.
        weights: Mode weights. Shape ``(K,)``.
        means: Per-mode predictive means. Shape ``(K, n_obs)``.
        variances: Per-mode predictive variances. Shape ``(K, n_obs)``.
        n_samples: Number of joint draws.

    Returns:
        ``(estimate, standard_error)`` in nats.
    """
    sds = jnp.sqrt(variances)
    k_key, y_key = jax.random.split(key)
    ks = jax.random.choice(k_key, weights.shape[0], (n_samples,), p=weights)
    eps = jax.random.normal(y_key, (n_samples, means.shape[1]))
    ys = means[ks] + sds[ks] * eps

    def one(y, k):
        comp = _mode_loglik(y, means, sds)
        return comp[k] - logsumexp(jnp.log(weights) + comp)

    vals = jax.vmap(one)(ys, ks)
    return jnp.mean(vals), jnp.std(vals) / jnp.sqrt(n_samples)


def class_mi_mc(key, weights, class_probs, means, variances, n_samples=20000):
    """MC estimate of I(C; Y) for per-mode class weights over a Gaussian mixture.

    Samples (k, c, y) from the joint p(k) p(c | k) p(y | k) and averages
    log p(y | c) - log pbar(y), with p(y | c) = sum_k w_k P(c|k) p(y|k) / P(c).
    This is a different decomposition than the entropy-difference estimator in
    ``photomancy.eig.class_eig``, so agreement is a real cross-check.

    Args:
        key: PRNG key.
        weights: Mode weights. Shape ``(K,)``.
        class_probs: Per-mode class weights ``P(c | k)``. Shape ``(K, C)``.
        means: Per-mode predictive means. Shape ``(K, n_obs)``.
        variances: Per-mode predictive variances. Shape ``(K, n_obs)``.
        n_samples: Number of joint draws.

    Returns:
        ``(estimate, standard_error)`` in nats.
    """
    sds = jnp.sqrt(variances)
    k_key, c_key, y_key = jax.random.split(key, 3)
    ks = jax.random.choice(k_key, weights.shape[0], (n_samples,), p=weights)
    c_uniform = jax.random.uniform(c_key, (n_samples,))
    c_cdf = jnp.cumsum(class_probs[ks], axis=-1)
    cs = jnp.sum(c_uniform[:, None] > c_cdf, axis=-1)
    eps = jax.random.normal(y_key, (n_samples, means.shape[1]))
    ys = means[ks] + sds[ks] * eps

    log_w = jnp.log(weights)
    log_p_class = logsumexp(log_w[:, None] + jnp.log(class_probs), axis=0)

    def one(y, c):
        comp = _mode_loglik(y, means, sds)
        log_joint_c = logsumexp(log_w + jnp.log(class_probs[:, c]) + comp)
        log_marg = logsumexp(log_w + comp)
        return log_joint_c - log_p_class[c] - log_marg

    vals = jax.vmap(one)(ys, cs)
    return jnp.mean(vals), jnp.std(vals) / jnp.sqrt(n_samples)
