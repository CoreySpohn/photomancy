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


def probit_site_moments_mc(key, mean, cov, a, b, n_samples=200000):
    """MC moments of the tilted density N(theta; mean, cov) * Phi(a + b @ theta).

    Importance-weights standard-normal draws by the probit site. Returns
    ``(mass, tilted_mean, tilted_cov)`` -- the normalizer and the first two moments
    of the truncated/tilted distribution the EP site update approximates exactly.
    """
    from jax.scipy.stats import norm as _norm

    chol = jnp.linalg.cholesky(cov)
    eps = jax.random.normal(key, (n_samples, mean.shape[0]))
    thetas = mean[None, :] + eps @ chol.T
    site = _norm.cdf(a + thetas @ b)
    mass = jnp.mean(site)
    w = site / jnp.sum(site)
    t_mean = jnp.sum(w[:, None] * thetas, axis=0)
    centered = thetas - t_mean[None, :]
    t_cov = (w[:, None] * centered).T @ centered
    return mass, t_mean, t_cov


def detection_channel_mi_mc(key, weights, means, covs, a_arr, b_arr, n_samples=200000):
    """MC estimate of I((M, theta); D) for per-mode linearized detection sites.

    D ~ Bernoulli(Phi(a_k + b_k @ theta)) given mode k and position theta. Uses
    I = E[log p(D | theta, k) - log pbar(D)] with pbar(D) computed exactly from the
    per-mode probit-Gaussian masses.

    Returns:
        ``(estimate, standard_error)`` in nats.
    """
    from jax.scipy.stats import norm as _norm

    k_key, t_key, d_key = jax.random.split(key, 3)
    n_modes = weights.shape[0]
    ks = jax.random.choice(k_key, n_modes, (n_samples,), p=weights)
    chols = jax.vmap(jnp.linalg.cholesky)(covs)
    eps = jax.random.normal(t_key, (n_samples, means.shape[1]))
    thetas = means[ks] + jnp.einsum("nij,nj->ni", chols[ks], eps)
    p_det = _norm.cdf(jax.vmap(jnp.dot)(b_arr[ks], thetas) + a_arr[ks])
    ds = jax.random.bernoulli(d_key, p_det)

    # Exact marginal detection probability via the probit-Gaussian integral.
    smear = jnp.sqrt(1.0 + jnp.einsum("kj,kjl,kl->k", b_arr, covs, b_arr))
    d_k = _norm.cdf((a_arr + jnp.einsum("kj,kj->k", b_arr, means)) / smear)
    p_bar = jnp.sum(weights * d_k)

    p_out = jnp.where(ds, p_det, 1.0 - p_det)
    p_marg = jnp.where(ds, p_bar, 1.0 - p_bar)
    vals = jnp.log(p_out) - jnp.log(p_marg)
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
