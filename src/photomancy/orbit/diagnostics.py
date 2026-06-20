"""Physical-space diagnostics for fitted orbit posteriors.

The orbit-specific views over a fitted posterior, as free functions on a
``(posterior, problem)`` pair: draw physical samples and summarize the modes. The
posterior is a generic ``MixturePosterior`` / ``GaussianPosterior`` (pure stats); the
``OrbitProblem`` supplies the ``to_physical`` coordinate map. Domain-agnostic posterior
operations (manifold projection, variance-capped sampling) live in
``photomancy.posterior_utils``.
"""

import jax
import jax.numpy as jnp


def sample_physical(posterior, problem, key, n):
    """Draw ``n`` posterior samples and map them to physical orbital parameters.

    Args:
        posterior: A fitted posterior (``MixturePosterior`` / ``GaussianPosterior``).
        problem: The :class:`~photomancy.orbit.inference.OrbitProblem` whose
            ``to_physical`` maps a flat ``z`` to named physical parameters.
        key: PRNG key.
        n: Number of samples.

    Returns:
        Dict ``{name: (n,)}`` over the problem's physical parameter names (``T``, ``e``,
        ``a``, ``cos_i``, ...).
    """
    z = posterior.sample(key, n)
    return jax.vmap(problem.to_physical)(z)


def mode_summary(posterior, problem):
    """Per-mode ``weight``, ``log_evidence``, and physical MAP ``params`` (diagnostics).

    Args:
        posterior: A :class:`~photomancy.posterior.MixturePosterior`.
        problem: The :class:`~photomancy.orbit.inference.OrbitProblem` supplying
            ``to_physical``.

    Returns:
        A list with one dict per mode -- ``{"weight", "log_evidence", "params"}`` --
        where ``params`` maps each physical parameter name to its MAP value.
    """
    weights = jnp.exp(posterior.log_weights)
    out = []
    for k in range(posterior.n_modes):
        phys = problem.to_physical(posterior.means[k])
        out.append(
            {
                "weight": float(weights[k]),
                "log_evidence": float(posterior.log_evidences[k]),
                "params": {kk: float(jnp.squeeze(v)) for kk, v in phys.items()},
            }
        )
    return out
