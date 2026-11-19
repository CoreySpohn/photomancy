"""The one posterior type, built only by :func:`photomancy.infer.run`.

A :class:`Posterior` is a weighted cloud of draws of a :class:`Fit`
with the fit it targets: ``samples`` keyed by the fit's free paths, normalized log
weights, the fit's own log density at each draw, and the evidence and its standard error
where the backend has them. Every backend fills the same fields, so a consumer reads one
layout whichever method ran.

Operations. ``sample`` and ``resample`` draw from the cloud, ``with_priors`` and
``append`` reweight it to a changed ``Fit`` without evaluating the old terms,
``model`` and ``forecast`` read the stored draws through the fit. Each acts on one
slice (one world) and raises on a batched posterior; map worlds with
``eqx.filter_vmap``. A refusal is a NaN slice, never a raise on a traced value.

Stored block coordinates are not reduced to a cell: the angle coordinates of an orbit
chart (``W_rad``, ``w_rad``, ``M0_rad``) are plain reals whose density repeats every
``2 pi``, so a sampler can wander across cells. Take no moment, quantile or covariance
of such a coordinate (read the orbit through ``model`` instead). For the same reason
the importance evidence of ``Laplace`` and ``Pathfinder`` on such a block is wrong
unless the proposal sits well inside one cell; ``SMC`` evidence is not affected.

Layout. The draw axis is flat and in stored order (chains are concatenated, never an
axis). Leading axes are batch axes: a stack of worlds is the same type with a longer
prefix, produced by mapping ``run`` over a stacked fit with ``eqx.filter_vmap``, and
every operation is mapped over it the same way. Shape checks read only the trailing
axes of ``log_weights`` and the batch prefix they define.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from priorities import Mixture, MultivariateNormal

from photomancy.infer.diagnostics import kish_ess, mc_se, pareto_khat, weighted_mean
from photomancy.infer.evidence import importance_evidence
from photomancy.infer.fit import Fit
from photomancy.infer.particles import (
    categorical_indices,
    systematic_indices,
    take_draws,
)

__all__ = ["Forecast", "Posterior", "cluster_to_mixture", "ravel_samples"]

_KINDS = ("monte_carlo", "quadrature")


class Forecast(eqx.Module):
    """Weighted forecast of a predictive distribution over the stored draws.

    ``noise_variance`` is the weighted mean of the per-draw predictive variance and
    ``spread_variance`` the weighted variance of the per-draw predictive mean, so
    ``noise_variance + spread_variance`` is the total variance of the mixture over
    the posterior (the law of total variance).

    Attributes:
        mean: Weighted mean of the per-draw predictive means.
        noise_variance: Weighted mean of the per-draw predictive variances.
        spread_variance: Weighted variance of the per-draw predictive means.
    """

    mean: object
    noise_variance: object
    spread_variance: object

    @property
    def variance(self):
        """Total predictive variance, ``noise_variance + spread_variance``."""
        return jax.tree.map(jnp.add, self.noise_variance, self.spread_variance)


def _reweight_slice(log_weights, log_ratio, min_n_eff, max_khat):
    """Reweight one slice by ``log_ratio`` and gate on the combined weights.

    Returns the new normalized log weights (all NaN on refusal), the log evidence
    increment and the standard error of that increment (both NaN on refusal). The
    increment is the self-normalized mean of the ratio under the old weights, so its
    error is the delta-method error of that mean; it is zero for a ratio of one.
    """
    lw = jnp.asarray(log_weights, jnp.float64)
    ratio = jnp.asarray(log_ratio, jnp.float64)
    live = ~jnp.isneginf(lw)
    bad = jnp.any(jnp.isnan(lw)) | jnp.any(jnp.isnan(ratio))
    bad = bad | jnp.any(jnp.isposinf(ratio) & live)
    combined = jnp.where(live, lw + ratio, -jnp.inf)
    log_z_new, _ = importance_evidence(combined)
    log_z_old, _ = importance_evidence(lw)
    log_total = log_z_new + jnp.log(lw.shape[-1])
    no_support = ~jnp.isfinite(log_total)
    new = jnp.where(bad | no_support, -jnp.inf, combined - log_total)
    top = jnp.max(jnp.where(live, ratio, -jnp.inf))
    scaled = jnp.exp(ratio - jnp.where(jnp.isfinite(top), top, 0.0))
    increment_se = mc_se(lw, scaled) / weighted_mean(lw, scaled)
    n_eff = kish_ess(new)
    khat = pareto_khat(new)
    khat_fails = jnp.isfinite(max_khat) & ~(khat <= max_khat)
    refuse = bad | no_support | ~(n_eff >= min_n_eff) | khat_fails
    nan = jnp.asarray(jnp.nan, jnp.float64)
    return (
        jnp.where(refuse, nan, new),
        jnp.where(refuse, nan, log_z_new - log_z_old),
        jnp.where(refuse, nan, increment_se),
    )


class Posterior(eqx.Module):
    """A weighted cloud of draws of a ``Fit`` and the density it targets.

    Attributes:
        samples: Entry dict with exactly the keys of ``fit.free``. A leaf-law entry is
            ``(*batch, N, *leaf_shape)``; a block entry is a dict of coordinate arrays
            ``(*batch, N, R)``. The draw axis is flat, in the backend's stored order.
        log_weights: Normalized log weights ``(*batch, N)``, float64
            (``logsumexp`` over ``N`` is zero). A slice that has no support is NaN.
        log_target: ``fit.log_density`` at the draws ``(*batch, N)``, float64: the
            normalized prior plus terms against Lebesgue measure on the stored
            coordinates, exactly ``-inf`` off the prior's support. It is never a
            proposal density.
        log_evidence: Log marginal likelihood ``(*batch,)``, float64, NaN when absent
            (the MCMC backends have none, and no estimate is invented).
        log_evidence_se: Standard error of ``log_evidence`` ``(*batch,)``, float64, NaN
            when the backend has none.
        fit: The ``Fit`` whose density the weights target.
        info: The backend's diagnostics module (chain index, divergence flags, thin
            factor, ladder, stage traces), or ``None``. Chain structure lives here and
            never in ``samples``.
        kind: ``"monte_carlo"`` (a random estimator) or ``"quadrature"`` (a
            positive-weight rule whose error is judged by refinement).
    """

    samples: dict
    log_weights: jax.Array
    log_target: jax.Array
    log_evidence: jax.Array
    log_evidence_se: jax.Array
    fit: Fit
    info: eqx.Module | None
    kind: str = eqx.field(static=True)

    def __check_init__(self):
        """Refuse a malformed posterior on the host (trailing axes only).

        Raises:
            ValueError: For a wrong ``kind``, key set, dtype or shape.
        """
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if set(self.samples) != set(self.fit.free):
            raise ValueError(
                "samples must have exactly the keys of fit.free: "
                f"{sorted(self.samples)} against {sorted(self.fit.free)}"
            )
        for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
            dtype = jnp.asarray(getattr(self, name)).dtype
            if dtype != jnp.float64:
                raise ValueError(f"{name} must be float64, got {dtype}")
        shape = jnp.shape(self.log_weights)
        if not shape:
            raise ValueError("log_weights needs a draw axis")
        batch, n = shape[:-1], shape[-1]
        if jnp.shape(self.log_target) != shape:
            raise ValueError(
                f"log_target has shape {jnp.shape(self.log_target)}, "
                f"log_weights {shape}"
            )
        for name in ("log_evidence", "log_evidence_se"):
            if jnp.shape(getattr(self, name)) != batch:
                raise ValueError(
                    f"{name} has shape {jnp.shape(getattr(self, name))}, "
                    f"expected the batch shape {batch}"
                )
        for key, entry in self.samples.items():
            for leaf in jax.tree.leaves(entry):
                if jnp.ndim(leaf) < len(batch) + 1 or jnp.shape(leaf)[
                    : len(batch) + 1
                ] != (*batch, n):
                    raise ValueError(
                        f"samples[{key!r}] has a leaf of shape {jnp.shape(leaf)}; "
                        f"expected leading axes {(*batch, n)} (batch, then draws)"
                    )

    # ----- draws -----------------------------------------------------------------

    def _batch_ndim(self):
        return jnp.ndim(self.log_weights) - 1

    def _require_slice(self, name):
        if self._batch_ndim():
            raise ValueError(
                f"Posterior.{name} takes one slice; map it over the batch with "
                "eqx.filter_vmap"
            )

    def sample(self, key, sample_shape=()):
        """Draw independent samples from the weighted cloud, with replacement.

        Categorical draws on the normalized weights (inverse CDF, so memory is linear in
        the draw count): duplicates occur, and for a
        correlated cloud (MCMC chains) the draws bootstrap the chain. Use
        :meth:`resample` to build a new fixed-size cloud. A slice with no support
        returns NaN draws. Takes one slice; map worlds with ``eqx.filter_vmap``.

        Args:
            key: PRNG key.
            sample_shape: Static shape of the draws.

        Returns:
            Entry dict, leaves ``(*sample_shape, *leaf_shape)``.
        """
        self._require_slice("sample")
        sample_shape = tuple(sample_shape)
        lw = jnp.asarray(self.log_weights)
        none = jnp.any(jnp.isnan(lw))
        idx = categorical_indices(lw, key, sample_shape)
        out = take_draws(self.samples, idx)
        return jax.tree.map(lambda leaf: jnp.where(none, jnp.nan, leaf), out)

    def resample(self, key, n=None):
        """Return a cloud of ``n`` systematically resampled draws with uniform weights.

        Fixed size (``n`` defaults to the current size), one random offset per slice.
        The draws are mutually dependent and chain structure is dropped, so ``info``
        becomes ``None``; ``log_target`` is indexed, the evidence and the fit are kept.
        A slice with no support keeps NaN weights.

        Args:
            key: PRNG key.
            n: Static number of draws, default the stored count.
        """
        self._require_slice("resample")
        lw = jnp.asarray(self.log_weights)
        size = lw.shape[-1] if n is None else int(n)
        none = jnp.any(jnp.isnan(lw))
        safe = jnp.where(none, -math.log(lw.shape[-1]), lw)
        idx = systematic_indices(key, safe, size)
        weights = jnp.full(size, jnp.where(none, jnp.nan, -math.log(size)))
        return Posterior(
            samples=take_draws(self.samples, idx),
            log_weights=weights.astype(jnp.float64),
            log_target=jnp.take(self.log_target, idx, axis=0),
            log_evidence=self.log_evidence,
            log_evidence_se=self.log_evidence_se,
            fit=self.fit,
            info=None,
            kind=self.kind,
        )

    # ----- reweighting -----------------------------------------------------------

    def _check_gate(self, min_n_eff, max_khat):
        given = (min_n_eff is not None, max_khat is not None)
        if self.kind == "quadrature":
            if any(given):
                raise ValueError(
                    "a quadrature posterior has no Monte Carlo gate: pass "
                    "min_n_eff=None and max_khat=None (the error is judged by "
                    "refinement)"
                )
        elif not all(given):
            raise ValueError("min_n_eff and max_khat are required for monte_carlo")

    def _reweighted(self, new_fit, log_ratio, min_n_eff, max_khat):
        self._check_gate(min_n_eff, max_khat)
        quadrature = self.kind == "quadrature"
        lo = 0.0 if quadrature else min_n_eff
        hi = jnp.inf if quadrature else max_khat
        new_lw, increment, increment_se = _reweight_slice(
            self.log_weights, log_ratio, lo, hi
        )
        if quadrature:
            refused = jnp.any(jnp.isnan(new_lw))
            increment_se = jnp.where(refused, jnp.nan, 0.0)
        return Posterior(
            samples=self.samples,
            log_weights=new_lw,
            log_target=jnp.asarray(self.log_target, jnp.float64) + log_ratio,
            log_evidence=self.log_evidence + increment,
            log_evidence_se=jnp.sqrt(self.log_evidence_se**2 + increment_se**2),
            fit=new_fit,
            info=None,
            kind=self.kind,
        )

    def with_priors(self, priors, *, min_n_eff, max_khat):
        """Swap priors by importance reweighting; no term is evaluated.

        The new ``Fit`` comes from ``fit.with_priors(priors)`` (which refuses a
        changed free set and a block with another map). Weights become
        ``log_weights + log pi_new - log pi_old`` and ``log_target`` follows; the
        evidence gains ``log sum w pi_new / pi_old``. Its standard error is the old
        error and the delta-method error of that self-normalized mean under the old
        weights, added in quadrature, so an identity swap leaves both unchanged. Exact
        in expectation only for normalized priors.

        The gate runs on the combined reweighted weights (``log_weights`` plus the
        log ratio), never the ratio alone, so it also re-gates the input cloud's own
        tail: an identity swap on a heavy-tailed cloud refuses. A slice that fails
        (Kish size below ``min_n_eff``, k-hat above ``max_khat`` by ``~(khat <=
        max_khat)`` so NaN refuses, zero total weight, a NaN ratio) comes back with
        all-NaN ``log_weights``, evidence and standard error; the caller decides per
        world with ``jnp.isnan``. ``max_khat=jnp.inf`` skips the k-hat test and
        ``min_n_eff=0`` the size test. Clouds under about 25 draws always fail a
        k-hat gate (fewer than five tail draws). The size is the plain Kish size:
        multiply by a chain efficiency yourself for correlated draws.

        Where the gate does not apply: k-hat is not valid on nested-sampling clouds
        (pass ``max_khat=jnp.inf`` and gate on ``min_n_eff``) and a ``quadrature``
        cloud has no gate (see below). A ``MAP`` posterior has one draw, so any
        ``min_n_eff`` above 1 refuses it. NaN is propagated, never repaired. Stored
        block angle coordinates are not reduced to a cell, so take no moment,
        quantile or covariance of one; the swap itself is unaffected because the
        priors and terms are periodic.

        Args:
            priors: Replacement entries, a subset of ``fit.free`` with the same kinds.
            min_n_eff: Smallest acceptable Kish size of the new weights, or ``None``
                for a ``quadrature`` posterior (which has no gate).
            max_khat: Largest acceptable Pareto k-hat, or ``None`` for ``quadrature``.

        Raises:
            ValueError: On the host, for a static fault in ``priors`` or gate
                arguments that do not match ``kind``.
        """
        self._require_slice("with_priors")
        new_fit = self.fit.with_priors(priors)
        old = self.fit.log_prior(self.samples)
        new = new_fit.log_prior(self.samples)
        ratio = jnp.where(jnp.isneginf(old), -jnp.inf, new - old)
        return self._reweighted(new_fit, ratio, min_n_eff, max_khat)

    def append(self, term, *, min_n_eff, max_khat):
        """Add one observation by importance reweighting; no old term is evaluated.

        The new ``Fit`` comes from ``fit.append(term)``. The term is evaluated at the
        model of every stored draw, weights gain ``+ term``, ``log_target`` follows and
        the evidence gains ``log sum w exp(term)``, with the standard error updated as
        in :meth:`with_priors`. The gate is the same, on the combined weights.

        The result is consistent with bias of order ``1 / N_eff``, not exact, and it
        holds only when (a) the new datum is conditionally independent of the old given
        the model, (b) the term reads only parameters already in the cloud (``free`` is
        never changed), (c) the held values and builds are the cloud's (the term is
        evaluated on ``fit.model_at``), and (d) the gate passes.

        Args:
            term: An :class:`~photomancy.infer.fit.AbstractTerm`, normalized.
            min_n_eff: As in :meth:`with_priors`.
            max_khat: As in :meth:`with_priors`.

        Raises:
            ValueError: On the host, for a term whose reads meet a free path or gate
                arguments that do not match ``kind``.
        """
        self._require_slice("append")
        new_fit = self.fit.append(term)
        only = Fit(self.fit.model, self.fit.free, (term,))
        ratio = jnp.asarray(only.log_likelihood(self.samples), jnp.float64)
        return self._reweighted(new_fit, ratio, min_n_eff, max_khat)

    # ----- reading the draws through the fit -------------------------------------

    def model(self, i):
        """Return the model of stored draw ``i``, every held leaf included."""
        self._require_slice("model")
        x = jax.tree.map(lambda a: jnp.take(a, i, axis=0), self.samples)
        return self.fit.model_at(x)

    def forecast(self, fn, *, batch_size=None):
        """Weighted forecast and its total-variance split over the stored draws.

        ``fn(model) -> (mean, variance)`` is a predictive mean and variance of a future
        datum given the model (pytrees of arrays, equal structure); the datum is taken
        independent of the old data given the model. The result is the weighted mean
        of the means, the weighted mean of the variances (``noise_variance``) and the
        weighted variance of the means (``spread_variance``); their sum is the
        total variance. ``fn`` is traced over ``fit.model_at(x)``, so it must be
        jittable and may use only builds made from ``fit.model``; a path the fit varies
        must not be read through a host build.

        Subsample with ``p.resample(key, m).forecast(fn)``: the variance of a mean over
        ``m`` resampled draws is inflated by about ``1 + N_eff / m`` relative to the
        weighted cloud.

        Args:
            fn: ``model -> (mean, variance)``.
            batch_size: Draws evaluated at once (``jax.lax.map`` chunk), default all
                at once with ``jax.vmap``.
        """
        self._require_slice("forecast")
        fit = self.fit

        def one(x):
            return fn(fit.model_at(x))

        if batch_size is None:
            means, variances = jax.vmap(one)(self.samples)
        else:
            means, variances = jax.lax.map(one, self.samples, batch_size=batch_size)
        w = jnp.exp(jnp.asarray(self.log_weights, jnp.float64))

        def average(a):
            return jnp.tensordot(w, jnp.asarray(a, jnp.float64), axes=(0, 0))

        mean = jax.tree.map(average, means)
        noise = jax.tree.map(average, variances)
        spread = jax.tree.map(
            lambda m, mu: average((jnp.asarray(m) - mu) ** 2), means, mean
        )
        return Forecast(mean, noise, spread)


def ravel_samples(posterior, keys=None):
    """Flatten each draw to a vector: ``(*batch, N, D)`` and an unravel function.

    The columns are the entries' leaves in the deterministic order of
    ``jax.tree.leaves`` (sorted keys, then each leaf row-major). ``unravel`` maps an
    array ``(*lead, D)`` back to the entry dict of those keys, leaves shaped
    ``(*lead, *leaf_shape)``.

    Args:
        posterior: A :class:`Posterior`.
        keys: Entries to flatten, default all of ``posterior.samples``.
    """
    samples = (
        posterior.samples
        if keys is None
        else {key: posterior.samples[key] for key in keys}
    )
    batch = jnp.ndim(posterior.log_weights) - 1
    leaves, treedef = jax.tree.flatten(samples)
    trailing = [jnp.shape(leaf)[batch + 1 :] for leaf in leaves]
    sizes = [math.prod(t) for t in trailing]
    lead = jnp.shape(posterior.log_weights)
    flat = jnp.concatenate(
        [
            jnp.reshape(leaf, (*lead, size))
            for leaf, size in zip(leaves, sizes, strict=True)
        ],
        axis=-1,
    )
    offsets = [sum(sizes[:i]) for i in range(len(sizes) + 1)]

    def unravel(vector):
        parts = [
            jnp.reshape(vector[..., a:b], (*jnp.shape(vector)[:-1], *t))
            for a, b, t in zip(offsets[:-1], offsets[1:], trailing, strict=True)
        ]
        return jax.tree.unflatten(treedef, parts)

    return flat, unravel


def cluster_to_mixture(posterior, k, key, *, keys=None, iters=25, cov_floor=1e-6):
    """Cluster a posterior into a ``k``-mode Gaussian ``priorities.Mixture``.

    Weighted k-means (fixed iterations) assigns the draws, flattened by
    :func:`ravel_samples`, to ``k`` clusters. Each mode is the cluster's weighted mean
    and covariance and its weight is the cluster's mass, so the mixture follows the
    posterior's modes. The result is a distribution over the flat vector: use it as the
    law of one free leaf of shape ``(D,)`` in the next ``Fit``. It has no inverse CDF,
    so it serves the gradient and sample backends and not nested sampling.

    Args:
        posterior: An unbatched :class:`Posterior` (one slice; map over worlds).
        k: Number of modes.
        key: PRNG key for the k-means start.
        keys: Entries to cluster, default all.
        iters: k-means iterations.
        cov_floor: Diagonal added to every covariance.

    Returns:
        A :class:`priorities.Mixture` of :class:`priorities.MultivariateNormal`.
    """
    if jnp.ndim(posterior.log_weights) != 1:
        raise ValueError("cluster_to_mixture takes one slice; map it over a batch")
    x, _ = ravel_samples(posterior, keys)
    n, d = x.shape
    w = jax.nn.softmax(posterior.log_weights)
    centers = x[jax.random.choice(key, n, (k,), replace=False)]

    def assign(centers):
        sq = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        return jax.nn.one_hot(jnp.argmin(sq, axis=1), k) * w[:, None]

    def step(centers, _):
        mass_w = assign(centers)
        mass = jnp.sum(mass_w, axis=0)
        new = (mass_w.T @ x) / jnp.maximum(mass[:, None], 1e-30)
        return jnp.where(mass[:, None] > 0, new, centers), None

    centers, _ = jax.lax.scan(step, centers, None, length=iters)
    mass_w = assign(centers)
    mass = jnp.sum(mass_w, axis=0)
    means = (mass_w.T @ x) / jnp.maximum(mass[:, None], 1e-30)

    def cov(mu, wcol):
        diff = x - mu
        return (diff * wcol[:, None]).T @ diff / jnp.maximum(jnp.sum(wcol), 1e-30)

    covs = jax.vmap(cov)(means, mass_w.T) + cov_floor * jnp.eye(d)[None]
    log_mass = jnp.log(jnp.maximum(mass, 1e-30))
    return Mixture(log_mass, MultivariateNormal(means, jnp.linalg.cholesky(covs)))
