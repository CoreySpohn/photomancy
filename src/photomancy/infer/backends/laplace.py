"""Laplace backend: MAP + Hessian -> Gaussian mixture Posterior.

The MAP is found with ``optax.lbfgs`` (a quasi-Newton line search that
converges to high precision). The covariance is the inverse of the
negative-log-density Hessian at the MAP, computed by Hessian-vector products
(forward-over-reverse AD) and regularized by flooring its eigenvalues at
``min_eigenvalue`` -- so a direction the data barely constrains gets a bounded
variance instead of inverting to an enormous one.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from photomancy.infer.backends._flat import flat_density, start_points
from photomancy.infer.backends.base import AbstractBackend
from photomancy.infer.backends.importance import ImportanceInfo, modes_importance
from photomancy.infer.posterior import Posterior


def _map_optimize(neg_logdensity, init, n_steps):
    """Minimize ``neg_logdensity`` from ``init`` with ``optax.lbfgs``."""
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(neg_logdensity)

    def step(carry, _):
        z, state = carry
        value, grad = value_and_grad(z, state=state)
        updates, state = opt.update(
            grad, state, z, value=value, grad=grad, value_fn=neg_logdensity
        )
        z = optax.apply_updates(z, updates)
        return (z, state), None

    (z_map, _), _ = jax.lax.scan(step, (init, opt.init(init)), None, length=n_steps)
    return z_map


def _hessian(neg_logdensity, z_map):
    """Symmetrized Hessian of ``neg_logdensity`` at ``z_map`` (forward over reverse)."""
    grad_fn = jax.grad(neg_logdensity)

    def hvp(v):
        return jax.jvp(grad_fn, (z_map,), (v,))[1]

    d = z_map.shape[0]
    hess = jax.vmap(hvp)(jnp.eye(d, dtype=z_map.dtype))
    return 0.5 * (hess + hess.T)


@eqx.filter_jit
def _laplace_mode(logdensity, init, n_steps, min_eigenvalue):
    """One mode with the diagnostics that decide whether it is a usable proposal.

    Returns ``(z_map, cov, log_z, min_raw_eigenvalue, grad_norm, newton_decrement)``:
    the covariance floors the Hessian eigenvalues at ``min_eigenvalue``; the raw
    eigenvalue and the gradient are those of the unregularized negative log density at
    ``z_map``, and the Newton decrement ``0.5 g' cov g`` is the number of nats the point
    still lies below the Gaussian's peak.
    """

    def neg(z):
        return -logdensity(z)

    z_map = _map_optimize(neg, init, n_steps)
    grad = jax.grad(neg)(z_map)
    eigvals, eigvecs = jnp.linalg.eigh(_hessian(neg, z_map))
    floored = jnp.maximum(eigvals, min_eigenvalue)
    cov = (eigvecs / floored) @ eigvecs.T
    d = z_map.shape[0]
    log_z = (
        logdensity(z_map)
        + 0.5 * d * jnp.log(2.0 * jnp.pi)
        - 0.5 * jnp.sum(jnp.log(floored))
    )
    return (
        z_map,
        cov,
        log_z,
        jnp.min(eigvals),
        jnp.linalg.norm(grad),
        0.5 * grad @ cov @ grad,
    )


def _merge_modes(flat_means, flat_covs, log_density, valid, periods, merge_distance):
    """Greedy merge of coincident modes; returns ``(kept, mode_index)``.

    Starts are visited in descending log density, valid ones only. A start joins the
    nearest kept mode when its Mahalanobis distance under that mode's precision is at
    most ``merge_distance`` (coordinates wrapped on the periodic entries); otherwise it
    is kept. ``mode_index[i]`` is the index of the kept mode start ``i`` converged to,
    -1 for a start that is not a mode.
    """
    n, d = flat_means.shape
    period = jnp.asarray([0.0 if p is None else p for p in periods], flat_means.dtype)
    safe = jnp.where(period > 0, period, 1.0)
    delta = flat_means[:, None, :] - flat_means[None, :, :]
    delta = jnp.where(period > 0, delta - safe * jnp.round(delta / safe), delta)
    eye = jnp.eye(d, dtype=flat_means.dtype)
    covs = jnp.where(valid[:, None, None], flat_covs, eye)
    solved = jnp.linalg.solve(covs[None], delta[..., None])[..., 0]
    distance = jnp.sqrt(jnp.maximum(jnp.sum(delta * solved, axis=-1), 0.0))
    order = jnp.argsort(jnp.where(valid, -log_density, jnp.inf))

    def visit(kept, i):
        near = jnp.where(kept, distance[i], jnp.inf)
        j = jnp.argmin(near)
        merges = valid[i] & (near[j] <= merge_distance)
        keep = valid[i] & ~merges
        index = jnp.where(keep, i, jnp.where(merges, j, -1))
        return kept.at[i].set(keep), index

    kept, index_in_order = jax.lax.scan(visit, jnp.zeros(n, bool), order)
    return kept, jnp.zeros(n, index_in_order.dtype).at[order].set(index_in_order)


class LaplaceInfo(eqx.Module):
    """The modes of a :class:`Laplace` run and which of them the proposal kept.

    The ``flat_`` fields are in the unconstrained flat vector of ``fit.to_flat``
    (sorted entry keys, each leaf row-major); ``modes`` is in stored coordinates.

    Attributes:
        flat_means: Mode locations ``(K, d)``.
        flat_covs: Floored inverse Hessians ``(K, d, d)``.
        modes: The mode of every start as an entry dict in stored coordinates with a
            leading axis ``K``; usable as ``init=`` of a ``Laplace`` of ``n_modes=K``,
            which reproduces the kept modes. Rows that are not finite are NaN.
        mode_index: The index of the kept mode each start converged to ``(K,)``, -1
            where the start is not a mode. Coincident starts share one kept mode.
        mode_log_evidence: Laplace log evidence of each mode, which sets its weight in
            the proposal ``(K,)``.
        kept: Whether each mode entered the proposal ``(K,)``: a mode, and not merged
            into a better one within ``merge_distance``.
        n_modes_kept: Number of modes kept; zero means the slice is NaN.
        min_raw_eigenvalue: Smallest eigenvalue of each mode's unregularized Hessian of
            the negative log density ``(K,)``; a large negative value is an indefinite
            point, not a mode.
        grad_norm: Norm of the gradient of the negative log density at each mode
            ``(K,)``.
        newton_decrement: ``0.5 g' cov g`` at each mode, in nats ``(K,)``.
        importance: The :class:`~photomancy.infer.backends.importance.ImportanceInfo`.
    """

    flat_means: jax.Array
    flat_covs: jax.Array
    modes: dict
    mode_index: jax.Array
    mode_log_evidence: jax.Array
    kept: jax.Array
    n_modes_kept: jax.Array
    min_raw_eigenvalue: jax.Array
    grad_norm: jax.Array
    newton_decrement: jax.Array
    importance: ImportanceInfo


class Laplace(AbstractBackend):
    """A Gaussian at each mode, wrapped on periodic coordinates, importance weighted.

    1. From each of ``n_modes`` starts the density is maximized with L-BFGS. The
       proposal is one Gaussian at the mode with the inverse Hessian as covariance, in
       the unconstrained coordinates, wrapped on every coordinate the fit declares
       periodic (several modes make a mixture weighted by the Laplace evidences).
    2. The draws are weighted ``density / proposal`` with the wrapped proposal density.
       The weights correct the Gaussian to the posterior; the evidence is the importance
       mean and is consistent whenever the weights have a finite mean.
    3. The evidence follows the contract on
       :attr:`~photomancy.infer.posterior.Posterior.log_evidence` (the k-hat is
       ``info.importance.khat``): a Gaussian that misses the tail of the posterior
       gives weights of infinite variance and an optimistic error. The draws and
       weights are still returned and are not gated; the operations of a ``Posterior``
       gate them.
    4. Expected to be poor: a posterior that is flat or nearly flat in an angle (a
       Gaussian cannot match a flat density on a circle), a posterior with more than one
       mode, and a mode near a zero-density region of a block.
    5. In those cases the cloud is a seed for
       ``AdaptiveImportance(MixtureProposal.from_cloud(key, z, log_weights,
       n_centers=...), n_draws)``, with ``z = fit.to_flat(posterior.samples)`` the
       flat vectors of the cloud.
    6. Started from the prior, the modes found can miss the posterior: on a concentrated
       orbit fit two of four keys kept only modes whose cloud had a Kish size near 1
       and a k-hat above 3 (log evidence about 1200 nats low), and the other two a
       k-hat of 0.7 to 0.9 with an evidence about three stated errors low. All four
       return a NaN evidence; seed ``AdaptiveImportance`` from the cloud (item 5), or
       start from a draw that is already near the posterior.
    7. Widening the covariance is not a remedy: it makes the weights worse.

    Starts that converge to the same mode are merged before the proposal is built, so a
    duplicate does not double the mode's weight: starts are visited in descending log
    density and one joins an already kept mode when its Mahalanobis distance under that
    mode's precision is at most ``merge_distance`` (wrapped on periodic coordinates).
    A merged start has ``kept = False`` and ``info.mode_index`` names the mode it
    joined. The merge sees only the starts it is given: a mode no start reaches is
    missing from the proposal and the evidence, and nothing in the run can show it.

    A mode is dropped from the proposal when it is not a mode: when the unregularized
    Hessian has an eigenvalue below ``-min_eigenvalue`` (an indefinite point, as where
    an optimizer stalls against a wall) or the Newton decrement exceeds
    ``max_decrement`` nats (the gradient has not converged). A direction with
    curvature between ``-min_eigenvalue`` and ``min_eigenvalue`` is flat, not wrong,
    and gets the variance ``1 / min_eigenvalue``. With no mode left the slice is NaN
    (draws, weights, evidence and standard error) and ``info.n_modes_kept`` is zero;
    ``info.min_raw_eigenvalue`` and ``info.grad_norm`` say why.

    ``run(fit, Laplace(init=posterior), key)`` is the route to an evidence for a NUTS
    result.

    Args:
        n_draws: Importance draws.
        n_modes: Number of starts and proposal components.
        n_steps: L-BFGS iterations per start.
        min_eigenvalue: Floor on the precision eigenvalues of the proposal covariance
            and the indefiniteness threshold, in units of inverse squared unconstrained
            coordinate (the standard deviation of a flat direction is at most
            ``1 / sqrt(min_eigenvalue)``). It trades a narrow proposal, whose weights
            have a heavy right tail that a few thousand draws under-sample (a bias of
            about 0.02 nat on a prior-only orbit fit at 0.1), against a wide one,
            whose weights vary more (variance). The default 0.03 is unbiased on that
            fit.
        max_decrement: Largest Newton decrement, in nats, of a mode that is kept.
        merge_distance: Mahalanobis distance, in standard deviations of the better
            mode, below which two converged starts are one mode.
        init: ``None`` for prior draws as starts, one constrained draw (repeated for
            every start), an entry dict whose every leaf has a leading axis of length
            ``n_modes`` (one explicit start each, so each alias keeps its own mode),
            or a ``Posterior`` whose weights pick the starts; the starts are placed in
            the flat vector by ``fit.to_flat``.
    """

    n_draws: int = eqx.field(static=True, default=2000)
    n_modes: int = eqx.field(static=True, default=1)
    n_steps: int = eqx.field(static=True, default=100)
    min_eigenvalue: float = 0.03
    max_decrement: float = 1.0
    merge_distance: float = eqx.field(static=True, default=1.0)
    init: Posterior | dict | None = None

    @eqx.filter_jit
    def _run(self, fit, key, *, batch_size=None):
        """Find the modes, then draw and weight from the mixture of the kept ones."""
        k_start, k_draw = jax.random.split(key)
        density = flat_density(fit)
        starts = start_points(fit, self.init, k_start, self.n_modes)
        means, covs, log_z, min_eig, grad_norm, decrement = jax.vmap(
            lambda z0: _laplace_mode(density, z0, self.n_steps, self.min_eigenvalue)
        )(starts)
        is_mode = (
            jnp.all(jnp.isfinite(means), axis=-1)
            & jnp.isfinite(log_z)
            & (min_eig >= -self.min_eigenvalue)
            & (decrement <= self.max_decrement)
        )
        kept, mode_index = _merge_modes(
            means,
            covs,
            jax.vmap(density)(means),
            is_mode,
            fit.periods(),
            self.merge_distance,
        )
        draws = modes_importance(
            fit, k_draw, means, covs, log_z, kept, self.n_draws, batch_size
        )
        info = LaplaceInfo(
            means,
            covs,
            fit.from_flat(means)[0],
            mode_index,
            log_z,
            kept,
            jnp.sum(kept),
            min_eig,
            grad_norm,
            decrement,
            draws.info,
        )
        return eqx.tree_at(lambda d: d.info, draws, info)
