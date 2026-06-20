# Mathematical foundations

This page gives the rigorous treatment behind the [architecture overview](architecture):
the fit, the posterior approximations, the evidence, the expected information gain, and
the shared structure that lets them compose. The notation follows the code, so each
formula corresponds to a piece of the engine.

## The fit: a posterior over a partitioned scene

A fit targets the posterior over the fitted parameters $z$ given data $D$,

$$ p(z \mid D) \propto p(D \mid z)\, p(z). $$

photomancy works with the unnormalized log-posterior, the logdensity

$$ \ell(z) = \log p(D \mid z) + \log p(z). $$

Here $z$ is the flat vector of the differentiable leaves of the scene. A scene is an
Equinox PyTree, and a fit partitions it into $z$ (the inexact-array leaves) and a static
remainder $S_0$. Scoring a position recombines them, $S = \mathrm{combine}(z, S_0)$,
pushes $S$ through the forward model $f$ to predicted data, and evaluates the likelihood,
$\log p(D \mid z) = \log L\big(f(S)\big)$. The forward model and likelihood operate on the
structured scene, while the sampler manipulates only $z$.

## The Laplace approximation

The maximum a posteriori estimate maximizes the logdensity,

$$ z_\star = \arg\max_z \ell(z), $$

found by gradient-based optimization. Expanding $\ell$ to second order about $z_\star$
gives a Gaussian posterior,

$$ p(z \mid D) \approx \mathcal{N}(z_\star, \Sigma), \qquad \Sigma = H^{-1}, \quad
H = -\nabla^2 \ell(z_\star), $$

where $H$ is the curvature of the logdensity at the mode, equal to the precision of the
Gaussian. For a Gaussian likelihood this curvature is the prior precision plus the observed
information the data carry about $z$, so $\Sigma$ is an inverse-information matrix, the
Bayesian analogue of the Cramer-Rao bound. The engine forms $H$ by Hessian-vector products
(forward-over-reverse automatic
differentiation), symmetrizes it, and floors its eigenvalues before inverting,

$$ H = Q \Lambda Q^\top, \qquad
\Sigma = \big(Q \, \max(\Lambda, \lambda_{\min}) \, Q^\top\big)^{-1}, $$

so a direction the data barely constrains receives a bounded variance rather than
inverting to an enormous one.

The same expansion yields the Laplace approximation to the log marginal likelihood, the
evidence,

$$ \log Z \approx \ell(z_\star) + \frac{d}{2}\log(2\pi) + \frac{1}{2}\log|\Sigma|, $$

with $d$ the dimension of $z$. This is the per-fit evidence the Gaussian backend carries.

## Mixtures and total evidence

A multi-start fit returns $K$ modes, each a Gaussian $\mathcal{N}(z_k, \Sigma_k)$ with its
own Laplace evidence $\log Z_k$. The mixture posterior weights the modes by their evidence,

$$ p(z \mid D) \approx \sum_{k=1}^{K} w_k \, \mathcal{N}(z_k, \Sigma_k), \qquad
w_k = \frac{e^{\log Z_k}}{\sum_j e^{\log Z_j}}, $$

and the total evidence marginalizes over them,

$$ \log Z = \log \sum_{k} e^{\log Z_k}. $$

This mixture is the substrate the analytic information gain consumes, because it carries
both the per-mode covariances (continuous-parameter uncertainty) and the weights (discrete
uncertainty across modes, such as period aliases).

The sampling backends reach the same posterior and evidence by other routes. Sequential
Monte Carlo accumulates the evidence from its tempering normalizers, nested sampling
integrates the likelihood over prior mass, and the No-U-Turn and microcanonical Langevin
samplers return samples without an evidence. Each returns the same `Posterior` interface, so nothing downstream
depends on which route produced it.

## Sequential updating

A posterior is a distribution over the same $z$ as a prior, so the engine converts one
into the other. Folding a new batch of data into the running belief $B_k$ is then Bayes'
rule with the previous posterior as the prior,

$$ B_{k+1}(z) \propto p(D_{k+1} \mid z)\, B_k(z), \qquad B_0(z) = p(z), $$

so information accumulates across observations and the belief sharpens. A Gaussian
posterior carries its full covariance forward, and a mixture carries its modes,
preserving multimodality that a single Gaussian would collapse.

## Expected information gain

The value of a candidate observation $Y$ is the information it is expected to provide. The
expected information gain is the mutual information between the parameters and the
observation, equivalently the expected reduction in posterior entropy,

$$ \mathrm{EIG}(Y) = \mathbb{E}_{Y}\big[\, D_{\mathrm{KL}}\!\big(p(z \mid Y) \,\|\, p(z)\big)
\,\big] = I(z; Y). $$

**Continuous-parameter gain.** For an observation linearized about a mode,
$y = J z + \varepsilon$ with Jacobian $J = \partial y / \partial z$ and measurement
covariance $R$, the matrix $\mathcal{I} = J^\top R^{-1} J$ is the Fisher information the
observation carries about $z$, and the posterior precision updates by adding it to the prior
precision,

$$ \Sigma_{\mathrm{new}}^{-1} = \Sigma_{\mathrm{old}}^{-1} + J^\top R^{-1} J. $$

The information gain is the resulting log-volume contraction of the covariance,

$$ \mathrm{EIG}_{\mathrm{geom}} = \tfrac{1}{2}\big(\log|\Sigma_{\mathrm{old}}|
- \log|\Sigma_{\mathrm{new}}|\big) = \tfrac{1}{2}\log\big|\, I
+ \Sigma_{\mathrm{old}} \mathcal{I} \,\big|. $$

Maximizing this gain over candidate observations is Bayesian D-optimal design, the choice
that maximizes the determinant of the posterior information $\Sigma_{\mathrm{new}}^{-1}$. The
Fisher form $J^\top R^{-1} J$, rather than the full Hessian used at the mode, is the right
curvature here because the gain is scored before the data exist, so the residual term the
observed information carries averages to zero under the model and leaves the expected Fisher
information. This term is evaluated per mode and dominates once a single mode is well
isolated.

**Alias-breaking gain.** When several modes survive, they predict different observations,
and an observation that separates them sharpens the weights. With per-mode predictions
$y_k$, weights $w_k$, mean $\bar{y} = \sum_k w_k y_k$, and weighted variance
$\mathrm{Var}(y) = \sum_k w_k (y_k - \bar{y})^2$ per observable, the gain from resolving
which mode is correct is

$$ \mathrm{EIG}_{\mathrm{alias}} = \tfrac{1}{2} \sum_{\text{obs}}
\log\!\Big(1 + \frac{\mathrm{Var}(y)}{R}\Big). $$

An epoch where the modes disagree strongly relative to the noise is worth the most. This is
the part of the gain that the Fisher information cannot describe, a property of the discrete
spread across modes rather than the local curvature within one, so the two terms partition
the total into a within-mode Fisher part and a between-mode part.

**Detectability.** When detection is itself uncertain, each mode carries a detection
probability $d_k \in [0, 1]$. With $\bar{d} = \sum_k w_k d_k$ and
$\mathrm{Var}(d) = \sum_k w_k (d_k - \bar{d})^2$, a detection-disagreement term

$$ \mathrm{EIG}_{\mathrm{det}} = \tfrac{1}{2}\log\!\Big(1 + \frac{\mathrm{Var}(d)}{1/4}\Big) $$

adds the information from learning whether the planet is detectable at all, where $1/4$ is
the maximum variance of a Bernoulli. The continuous-parameter term is weighted by
detectability, so a mode that cannot be seen contributes no parameter information.

## How the pieces interact

One quantity recurs at every scale, the mutual information between a question $Q$ and an
observation, $I(Q; Y)$, and changing what $Q$ is reuses the same machinery. Taking $Q$ to
be the continuous parameters gives the geometric gain, the question of which epoch sharpens
the orbit. Taking $Q$ to be a discrete label, which mode is correct or whether a planet or
a spectral feature is present, gives the alias-breaking and detection terms, and
integrating the likelihood over the prior turns the same question into the evidence and the
Bayes factor $\log Z_1 - \log Z_0$. Chaining the belief through sequential updating carries
each answer into the next observation.

Because the posterior, the evidence, and the information gain all read one `Posterior`
interface, a method written for one question at one scale transfers to the others. The
architecture overview describes how the code layers compose; this shared information
currency is why they compose.
