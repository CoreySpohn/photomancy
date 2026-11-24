# Fits and posteriors

This page builds one small fit by hand and follows it from the data to a posterior, an
evidence, and an update with new data. The problem is the simplest one with a known answer:
three noisy measurements of one number. Every quantity the code prints can be checked
against a closed form, so each step of the machinery can be seen doing what it claims.
The [orbit fitting example](../examples/orbit_fitting) runs the same steps on a planet.

## A model holds every quantity

The model is an Equinox module whose leaves are arrays. Here it holds one number, `mu`,
the quantity the measurements are of.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Model(eqx.Module):
    mu: jax.Array


model = Model(mu=jnp.asarray(0.0))
```

A real model is larger. An orbit fit holds an orbix `System` with the star, the planets and
the frame, and a scene fit holds a skyscapes `Scene`. A node of the model is named by a
dotted path such as `"mu"` or `"dynamics.orbits"`, read and written with
`hwoutils.model_paths`. The fit below refers to what it frees by these paths.

## Data enter as terms

A term holds one set of measurements and scores them at a model. It returns the
log-likelihood of its data, with every normalizing constant included, so that the sum of
the terms is a normalized density of the data.

```python continuation
from typing import ClassVar

from photomancy.infer import AbstractTerm


class GaussianTerm(AbstractTerm):
    """Measurements of mu with independent Gaussian errors."""

    y: jax.Array
    sigma: float
    reads: ClassVar[tuple[str, ...]] = ()

    def __call__(self, model):
        r = (self.y - model.mu) / self.sigma
        return jnp.sum(-0.5 * r**2 - jnp.log(self.sigma) - 0.5 * jnp.log(2 * jnp.pi))

    def draw(self, key, model):
        noise = self.sigma * jax.random.normal(key, self.y.shape)
        return eqx.tree_at(lambda t: t.y, self, model.mu + noise)


term = GaussianTerm(y=jnp.array([0.9, 1.3, 1.1]), sigma=0.5)
```

The method `draw` returns a copy of the term holding a simulated data set at a given
model, which is how a test or a survey simulation makes data from a known truth. The field
`reads` lists the model paths of any cached build the term holds. This term caches
nothing, so it reads nothing beyond the model it is handed. The library ships terms for the
measurements of direct imaging in `photomancy.measurements`, each paired with a data class
of the same stem: `RelativeAstrometryTerm` with `AstrometryData`, `RadialVelocityTerm`
with `RadialVelocityData`, `DMagTerm` with `DMagData`, and `ReportTerm` with `ReportData`.

## A fit names what is free

A `Fit` combines three things: the model, a dictionary of the free entries, and the terms.
The dictionary maps a model path to the prior over that node. Here the one free entry is
`mu`, with a standard normal prior from the `priorities` library.

```python continuation
from priorities import Normal

from photomancy.infer import Fit

fit = Fit(model, {"mu": Normal(0.0, 1.0)}, (term,))
print(fit.log_density({"mu": jnp.asarray(1.0)}))
```

Every path that is not in the dictionary keeps the value it has in the model. The fit is a
log density over the free entries, `log_density = log_prior + sum of terms`, and because
the prior and each term are normalized, it is a normalized joint density of the free
values and the data. Its integral over `mu` is the evidence of the data.

A free entry takes one of two forms. A `priorities` distribution, as above, is a law over
one whole array leaf. A block (`hwoutils.blocks.AbstractBlock`) owns a larger node and
writes it from its own coordinates with its own normalized density. The orbit blocks of
orbix are the main example: a `ThieleInnesBlock` frees one planet's orbit and also writes
the star mass that the orbit implies.

## A backend turns the fit into a posterior

The function `run` takes a fit, a backend and a random key, and returns a `Posterior`. The
backend below is importance sampling with the prior as the proposal: it draws `mu` from
the prior and weights each draw by its likelihood.

```python continuation
from photomancy.infer import Importance, PriorProposal, run

posterior = run(fit, Importance(PriorProposal(), n_draws=4000), jax.random.key(0))
print(posterior.samples["mu"].shape, posterior.log_weights.shape)
```

A posterior is a cloud of weighted draws. The field `samples` holds the draws under the
same keys as the free entries, `log_weights` holds their normalized log weights, and
`log_target` holds the fit's own log density at each draw. The fit itself rides along as
`posterior.fit`, so everything downstream can rebuild the model at any draw.

## The evidence comes with an error

The posterior also carries the log evidence and its standard error. For this problem the
evidence has a closed form: the three measurements are jointly Gaussian with covariance
`sigma**2 * I + 1`, the measurement noise plus the prior variance shared by all three.

```python continuation
import numpy as np
from scipy.stats import multivariate_normal

exact = multivariate_normal(np.zeros(3), 0.25 * np.eye(3) + np.ones((3, 3)))
log_z_exact = exact.logpdf([0.9, 1.3, 1.1])

print(posterior.log_evidence, posterior.log_evidence_se, log_z_exact)
assert abs(posterior.log_evidence - log_z_exact) < 4 * posterior.log_evidence_se
```

The importance estimate agrees with the exact value within its stated error. The same
fields exist for every backend, and a backend that has no evidence estimator, such as a
Markov chain sampler, returns NaN in both rather than an invented number.

## A NaN evidence is a refusal

A finite `log_evidence` means that three things held. The same statement is on
`Posterior.log_evidence`, and the rest of the documentation refers here.

1. **Tail.** The importance estimate of the evidence is trustworthy only when the
   weights have a finite variance. The library tests this from the run itself with the
   Pareto k-hat of the weights (Vehtari et al. 2024): a k-hat above
   `khat_threshold(n_draws)` means a few draws carry almost all of the weight and the
   stated error is not an error bar.
2. **Proposal validity.** The proposal reports its weights valid, `info.proposal_valid`.
   A proposal that has no condition of its own always reports valid. The completed orbit
   proposal of `any_epoch` has one. It draws a pool of constants for every orbit shape,
   and the pool fraction of a draw is the Kish size of its pool divided by the pool
   size. The proposal reports valid only when the plain mean of that fraction over the
   live draws is at least 0.5 (`min_pool_fraction`).
3. **No refusal.** The run reached its target (a tempering ladder that arrived at
   temperature one, a mode kept, a nested-sampling run that converged rather than being
   cut by its sample cap, no NaN weight) and every later gate passed.

When any of the three fails, the backend returns NaN for the evidence and its error and
still returns the draws and their weights.

The contract has one limit that no check can see. A fit that misses a mode entirely,
because no start or proposal ever visited it, gives a finite evidence that is wrong. The
tail test cannot flag it, because the weights of the visited mode look healthy.

Making the data a hundred times more precise shows the refusal. Prior draws then almost
never land where the likelihood is, and a handful of draws take all the weight.

```python continuation
from photomancy.infer import khat_threshold

sharp = GaussianTerm(y=jnp.array([1.02, 0.98, 1.01]), sigma=0.01)
fit_sharp = Fit(model, {"mu": Normal(0.0, 1.0)}, (sharp,))
refused = run(fit_sharp, Importance(PriorProposal(), n_draws=4000), jax.random.key(0))

print(refused.info.khat, khat_threshold(4000), refused.log_evidence)
assert np.isnan(refused.log_evidence)
assert np.all(np.isfinite(refused.log_weights))
```

The draws and their weights are still returned, because a caller may want them as the
seed of a better run. Only the number that would have been wrong is withheld.

## New data reweight the cloud

When one more measurement arrives, the posterior does not have to be fitted again. Each
stored draw already carries a model, so the new term can be evaluated at every draw and
added to its log weight. `Posterior.append_term` does this and returns a posterior over
the fit with the extra term.

```python continuation
new = GaussianTerm(y=jnp.array([1.2]), sigma=0.5)
moved = posterior.append_term(new, min_n_eff=500, max_khat=0.7)

exact4 = multivariate_normal(np.zeros(4), 0.25 * np.eye(4) + np.ones((4, 4)))
log_z4_exact = exact4.logpdf([0.9, 1.3, 1.1, 1.2])

print(len(moved.fit.terms), moved.log_evidence, log_z4_exact)
assert abs(moved.log_evidence - log_z4_exact) < 4 * moved.log_evidence_se
```

The evidence gains the log of the weighted mean of the new likelihood, which is the
evidence of the new datum given the old ones, and it again agrees with the closed form.

Reweighting is only as good as the overlap between the old cloud and the new posterior,
so `append_term` takes a gate. The arguments `min_n_eff` and `max_khat` set the smallest
acceptable effective sample size and the largest acceptable k-hat of the new weights. The
gate judges the combined weights (the old weights plus the new term), so a cloud whose own
weights have a heavy tail can fail it even for a weak datum, and `Posterior.resample`
first gives a cloud of equal weights when that is a concern. When the gate fails, every
weight and the evidence come back NaN, and the cloud is not used.

`Posterior.replace_term` is the companion for a term that changes rather than one that is
added. Its main use is a padded data set: a term built with room for more rows holds the
next row as invalid padding, and replacing it with the same term where that row is valid
adds the datum without changing the structure of the fit, so a batch of fits over many
simulated worlds does not recompile.

## A prior can be swapped

`Posterior.with_priors` reweights the cloud to a different prior in the same way, with no
term evaluated. A swap can narrow a prior, because the draws that remain still follow the
new prior inside the old support, but it cannot widen one: a cloud has no draws where the
old prior had no mass, so a widening needs a refit and comes back refused. Any swap whose
containment cannot be shown is refused the same way. A leaf prior is compared by its
support, and a mixture covers only its components with nonzero weight, not the gap
between them. A block is compared law by law, and every other field must be unchanged: a
sky-position block at another epoch is refused, and so is an orbit prior moved from a law
on the semi-major axis to a law on the period, even an equivalent one, so express both
priors in the same parameterization. A block that can decide containment itself may
define `support_contains(new)`.

The field `info.log_mass_ratio` is the log of the mean ratio of the new density to the old
one under the old weights, read before the decision, refused or not. When the cloud is
the old prior (a fit with no terms), it is the log of the mass of the new prior on the old
support; on a posterior it is the evidence increment the swap would have applied.

## Keeping a posterior on disk

A posterior has no save method yet. Until persistence lands, write it with Equinox. The
file holds the arrays only, so reading it needs a template `Posterior` with the same
PyTree structure throughout: the same `Fit`, the same draw count and backend, and the
same diagnostics in `info` and the same static metadata, such as `kind`. A posterior
returned by `with_priors` or `append_term` carries a `ReweightInfo` where a fresh run
carries its backend's info, so build its template by the same chain of calls. The file is
specific to the library version that wrote it.

```python continuation
import tempfile
from pathlib import Path

import equinox as eqx

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "posterior.eqx"
    eqx.tree_serialise_leaves(path, posterior)
    like = run(fit, Importance(PriorProposal(), n_draws=4000), jax.random.key(1))
    restored = eqx.tree_deserialise_leaves(path, like)

assert all(
    bool(jnp.array_equal(a, b))
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(posterior), strict=True)
)
```

## A refused reweighting falls back to a refit

The precise measurements above are a datum that reweighting cannot absorb: the cloud drawn
under the broad posterior has almost no draws where the sharp likelihood lives. The
function `update` makes the cheap move when it works and refits when it does not. It calls
`append_term` (or `replace_term`, when given an `index`) with the gate, and on a refusal
it runs the given backend on the new fit.

```python continuation
from photomancy.infer import Laplace, update

updated = update(
    posterior,
    sharp,
    Laplace(n_draws=4000),
    jax.random.key(1),
    min_n_eff=500,
    max_khat=0.7,
)

y_all = [0.9, 1.3, 1.1, 1.02, 0.98, 1.01]
noise = np.diag([0.25] * 3 + [1e-4] * 3)
log_z6_exact = multivariate_normal(np.zeros(6), noise + np.ones((6, 6))).logpdf(y_all)

print(updated.log_evidence, log_z6_exact)
assert abs(updated.log_evidence - log_z6_exact) < 1e-6
```

Here the reweighting is refused and the refit with `Laplace` returns the evidence of all
six measurements, exact for this Gaussian problem. The refit must return the same layout
as the cloud, the same number of draws and the same kind of posterior, so the backend is
built with the posterior's draw count. Because the choice between the two branches is one
`jax.lax.cond`, `update` can be traced and mapped over a stack of fits.

The `info` of the result is an `UpdateInfo`. Its `reweight` field holds the gate reading
(`n_eff`, `khat`, `refused`), where `refused` is true exactly when the refit ran. Its
`refit` field is the backend's own info, the same as `run(...).info` where the refit ran
and filled with NaN, zero or false elsewhere, so `info.refit.proposal_valid` says whether
the refit's proposal could vouch for its weights.

## The backends

Every backend is passed to `run` in the same way and returns a `Posterior`. They differ in
how the draws are made and in whether an evidence comes with them.

| Backend | How it draws | Evidence |
|---|---|---|
| `Importance(proposal)` | independent draws from one proposal, weighted to the fit | importance mean, refused under the contract above |
| `AdaptiveImportance(proposal)` | rounds of Gaussian mixtures refitted to the weighted cloud, tempered when the proposal is the prior | importance mean of the last round, refused under the contract above |  XXhat limit |
| `Laplace()` | a Gaussian at each mode, weighted to the fit | importance mean, refused under the contract above |
| `Pathfinder()` | Pathfinder Gaussians along the optimizer path, weighted to the fit | importance mean, refused under the contract above |
| `NUTS()`, `MCLMC()` | Markov chains with equal weights | none (NaN) |
| `NestedSampling()` | jaxns nested sampling of the unit cube | nested-sampling estimate, NaN when cut by the sample cap |
| `MAP()` | a staged optimization, one draw at the optimum | none (NaN) |
| `Grid(axes)` | a quadrature rule on a product grid | the quadrature sum, with no standard error |

The proposals for the importance backends are `PriorProposal`, `MixtureProposal`, and
`DefensiveProposal`, which mixes a small share of the prior into another proposal. The
orbit package adds proposals and configured backends of its own, described in
[Orbital parameterizations](orbital-parameterizations).
