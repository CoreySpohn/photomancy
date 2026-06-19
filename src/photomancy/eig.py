"""Domain-agnostic analytic Expected Information Gain over a Posterior.

Scores candidate observations from a ``MixturePosterior`` (per-mode Gaussians) via the
analytic Laplace covariance update (``Sigma^-1 += J^T R^-1 J``). The forward model is a
caller-supplied ``forward(z, candidate) -> observables`` (differentiable in the flat
position ``z``); the orbit / disk / imaging layers supply their own. This core knows no
domain.
"""

import jax
import jax.numpy as jnp


def geometric_eig(cov_old, jacobian, obs_variance):
    """Information gain from covariance shrinkage at one mode (nats).

    Args:
        cov_old: Current covariance. Shape ``(d, d)``.
        jacobian: Forward-model Jacobian ``dy/dz``. Shape ``(n_obs, d)``.
        obs_variance: Scalar or ``(n_obs,)`` measurement variances.

    Returns:
        ``(eig, cov_new)`` -- scalar information gain and the updated covariance.
    """
    prec_old = jnp.linalg.inv(cov_old)
    obs_var = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(obs_variance)), (jacobian.shape[0],)
    )
    fim = jacobian.T @ jnp.diag(1.0 / obs_var) @ jacobian
    cov_new = jnp.linalg.inv(prec_old + fim)
    eig = 0.5 * (jnp.linalg.slogdet(cov_old)[1] - jnp.linalg.slogdet(cov_new)[1])
    return eig, cov_new


def alias_breaking_eig(weights, y_preds, obs_variance):
    """Information gain from prediction disagreement across modes (nats).

    Args:
        weights: Mode weights. Shape ``(K,)``.
        y_preds: Predictions at each mode. Shape ``(K, n_obs)``.
        obs_variance: Scalar or ``(n_obs,)`` measurement variances.

    Returns:
        Scalar alias-breaking EIG.
    """
    obs_var = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(obs_variance)), (y_preds.shape[-1],)
    )
    y_mean = jnp.sum(weights[:, None] * y_preds, axis=0)
    y_var = jnp.sum(weights[:, None] * (y_preds - y_mean[None, :]) ** 2, axis=0)
    return 0.5 * jnp.sum(jnp.log1p(y_var / obs_var))


def detectability_eig(weights, det_weights):
    """Information gain from detection disagreement across modes (nats).

    Args:
        weights: Mode weights. Shape ``(K,)``.
        det_weights: Per-mode detectability in ``[0, 1]``. Shape ``(K,)``.

    Returns:
        Scalar detection-disagreement EIG.
    """
    p_det = jnp.sum(weights * det_weights)
    det_var = jnp.sum(weights * (det_weights - p_det) ** 2)
    return 0.5 * jnp.log1p(det_var / 0.25)


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
        if detectable is None:
            det_w = jnp.ones(means.shape[0])
            weighted_geom = jnp.sum(weights * geom_eigs)
            alias_val = alias_breaking_eig(weights, y_preds, obs_var)
        else:
            det_w = jax.vmap(lambda z: detectable(z, candidate))(means)
            weighted_geom = jnp.sum(weights * det_w * geom_eigs)
            alias_val = alias_breaking_eig(
                weights, y_preds, obs_var
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
            non-detectable modes contribute no astrometric information and a
            detection-disagreement term is added. When omitted, all modes detectable.
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
