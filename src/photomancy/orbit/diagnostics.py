"""Physical-space diagnostics for fitted orbit posteriors.

Chart-space mixtures expose the predictions and likelihoods of the fitted system's
terms at each mode mean. The fitted problem supplies ``to_physical`` for posterior
samples and mode summaries. Each function first calls the problem's ``require``, so
a posterior that declares a different chart or epoch than the problem (or declares
none) is refused with a ``ValueError`` instead of being decoded in the wrong
coordinates.
"""

import jax
import jax.numpy as jnp
from hwoutils.constants import G
from orbix.equations import period_to_sma

from photomancy.orbit._frame import frame_from_tp


def mode_observables(posterior, problem):
    """Return term log likelihoods and predictions at each chart-space mode.

    The chart problem uses the same flat coordinates as the mixture.
    The result has the same ``term_values`` and ``predictions`` tuples as
    ``SystemLogDensity.value_and_aux``, with a leading mode axis on each array.
    """
    problem.require(posterior)
    return jax.vmap(lambda z: problem.density.value_and_aux(problem.unflatten(z))[1])(
        posterior.means
    )


def sample_physical(posterior, problem, key, n):
    """Draw ``n`` posterior samples and map them to physical orbital parameters.

    Args:
        posterior: A fitted posterior (``MixturePosterior`` / ``GaussianPosterior``).
        problem: The fitted orbit problem whose
            ``to_physical`` maps a flat ``z`` to named physical parameters.
        key: PRNG key.
        n: Number of samples.

    Returns:
        Dict ``{name: (n,)}`` over the problem's physical parameter names (``T``, ``e``,
        ``a``, ``cos_i``, ...).
    """
    problem.require(posterior)
    z = posterior.sample(key, n)
    return jax.vmap(problem.to_physical)(z)


def orbits_from_samples(samples, Ms_kg, *, reference_day):
    """Turn physical samples into a ``(K,)``-batched ``orbix.system.FrameOrbit``.

    The bridge from a fitted posterior to orbit-track visualization and any
    other propagation consumer: ``sample_physical`` output (or any dict with
    the ``T``/``e``/``cos_i``/``W``/``cos_w``/``sin_w``/``tp`` keys the
    forward models use) becomes one batched orbit object.

    The forward models interpret ``tp`` and observation times as days from the
    same zero; the sampling model defines ``tp = -M0 * T / (2*pi)``. Samples
    do not identify a calendar epoch. ``reference_day`` locates that zero in
    the returned orbit's time coordinate, in float days from the caller's
    ``hwoutils.epoch.Epoch``. Periapsis is at ``reference_day + tp``; evaluate
    with ``orbit.at(reference_day + observation_times)``. Use zero when the
    caller's Epoch is already the observation-time origin.

    Args:
        samples: Dict of physical parameters keyed as ``to_physical`` emits
            them, each value scalar or ``(K,)``. Includes true planet mass
            ``Mp`` in kg when available; otherwise assumes negligible planet mass.
        Ms_kg: Stellar mass in kg (the fit argument, not a fitted parameter).
        reference_day: Sample day zero in the caller's float-day coordinate.

    Returns:
        A :class:`orbix.system.FrameOrbit`, batched over the sample values.
        Scalar inputs produce an unbatched orbit.
    """
    total_mass = Ms_kg + samples.get("Mp", 0.0)
    return frame_from_tp(
        period_to_sma(samples["T"], total_mass),
        samples["e"],
        samples["cos_i"],
        samples["W"],
        samples["cos_w"],
        samples["sin_w"],
        reference_day + samples["tp"],
        G * total_mass,
    )


def mode_scalars(posterior, problem, name):
    """One physical scalar per mixture mode, with the mode weights.

    The recurring "period per mode against its posterior weight" extraction,
    generalized to any physical parameter: evaluate ``to_physical`` at each
    mode mean and pull one named entry.

    Args:
        posterior: A :class:`~photomancy.posterior.MixturePosterior`.
        problem: The fitted orbit problem supplying ``to_physical``.
        name: Physical parameter name (``"T"``, ``"e"``, ...).

    Returns:
        Tuple ``(values, weights)``, each shape ``(n_modes,)``.
    """
    problem.require(posterior)
    values = jnp.stack(
        [
            jnp.squeeze(problem.to_physical(posterior.means[k])[name])
            for k in range(posterior.n_modes)
        ]
    )
    return values, jnp.exp(posterior.log_weights)


def mode_summary(posterior, problem):
    """Per-mode ``weight``, ``log_evidence``, and physical MAP ``params`` (diagnostics).

    Args:
        posterior: A :class:`~photomancy.posterior.MixturePosterior`.
        problem: The fitted orbit problem supplying
            ``to_physical``.

    Returns:
        A list with one dict per mode -- ``{"weight", "log_evidence", "params"}`` --
        where ``params`` maps each physical parameter name to its MAP value.
    """
    problem.require(posterior)
    weights = jnp.exp(posterior.log_weights)
    out = []
    for k in range(posterior.n_modes):
        phys = problem.to_physical(posterior.means[k])
        out.append(
            {
                "weight": float(weights[k]),
                "log_evidence": float(posterior.log_evidences[k]),
                "params": {kk: jnp.squeeze(v).tolist() for kk, v in phys.items()},
            }
        )
    return out
