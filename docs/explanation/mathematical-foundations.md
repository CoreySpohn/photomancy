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

The value of a candidate observation $Y$ is the information it is expected to provide
about a declared quantity of interest, evaluated under the current belief. For a quantity
$\Psi$, the expected information gain is the mutual information, equivalently the expected
reduction in posterior entropy,

$$ \mathrm{EIG}_{\Psi}(Y) = \mathbb{E}_{Y}\big[\, D_{\mathrm{KL}}\!\big(p(\Psi \mid Y) \,\|\, p(\Psi)\big)
\,\big] = I(\Psi; Y), $$

and the scheduling questions of the library are different choices of $\Psi$ over the same
posterior: the continuous parameters, the discrete mode label, the binary detection
outcome, or a science-level classification. Over a mixture posterior the joint gain
decomposes exactly by the chain rule of mutual information,

$$ I\big((M, z); Y\big) = I(M; Y) + \sum_k w_k \, I(z; Y \mid M = k), $$

a between-mode discrete term plus an evidence-weighted within-mode continuous term. The
terms below are the estimators of these pieces, and `evaluate_candidates` assembles them.

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

When the science target is a sub-block of the parameters, the gain that matters is the
marginal one. With a selector $\Phi$ onto the block of interest,

$$ \mathrm{EIG}_{\mathrm{marg}} = \tfrac{1}{2}\big(\log|\Phi \Sigma_{\mathrm{old}} \Phi^\top|
- \log|\Phi \Sigma_{\mathrm{new}} \Phi^\top|\big), $$

so an observation that only sharpens a nuisance parameter books no gain (the
`qoi_projection` of `geometric_eig`). The full-vector log-determinant would reward
nuisance shrinkage indiscriminately.

**Mode-discrimination gain.** When several modes survive, they predict different
observations, and an observation that separates them sharpens the weights. The mode
information $I(M; Y)$ has no closed form for a Gaussian mixture, so the library reports an
upper bound on it, the intersection of two bounds, which `alias_breaking_eig_from_covariances`
returns. Each mode predicts the observation as $\mathcal{N}(y_k, S_k)$, where the full
predictive covariance $S_k = J_k \Sigma_k J_k^\top + R$ carries the within-mode spread and its
correlations as well as the measurement noise. Because $I(M; Y) = H(Y) - \sum_k w_k H(Y \mid k)$
and a Gaussian has the largest entropy for a given covariance, the moment-matched bound is

$$ I(M; Y) \;\le\; \tfrac{1}{2} \Big[ \log\big|S_{\mathrm{mix}}\big|
- \sum_k w_k \log|S_k| \Big], \qquad
S_{\mathrm{mix}} = \sum_k w_k \big[ S_k + (y_k - \bar y)(y_k - \bar y)^\top \big], $$

with $\bar y = \sum_k w_k y_k$; and globally the information about the mode label cannot
exceed the mode entropy, $I(M; Y) \le H(w)$, which is applied after the Gaussian bound. Two
perfectly resolved 50/50 modes are worth $\ln 2$ nats, however far apart their predictions
sit. The determinants must be taken over the full covariances, because two modes can share
their means and marginal variances and differ only in their correlations: with
$S_{1,2} = \begin{pmatrix} 1.1 & \pm 0.9 \\ \pm 0.9 & 1.1 \end{pmatrix}$ the bound is
$\tfrac{1}{2}\log(1.21/0.4) \approx 0.553$ nats, a Monte Carlo evaluation of the exact mode
information gives about $0.331$ nats, and a per-observable (diagonal) evaluation would report
zero in these coordinates and $0.553$ after a $45^\circ$ rotation of the same experiment. The
full-determinant form is invariant under any invertible linear change of observable basis.
The result is an upper bound rather than an estimate: it is tight when the modes are resolved
(the $H(w)$ cap) or identical (zero), and it overstates the information of partially
overlapping modes. The variance interface `alias_breaking_eig` is the diagonal special case,
for predictive covariances known to be diagonal. This is the part of the gain that the Fisher
information cannot describe, a property of the discrete spread across modes rather than the
local curvature within one, so the chain rule above partitions the total into a within-mode
Fisher part and a between-mode part.

**Detection-channel gain.** When detection is itself uncertain, each mode carries a
detection probability $d_k \in [0, 1]$, and the binary outcome $D$ carries exactly

$$ \mathrm{EIG}_{\mathrm{det}} = I(D; M) = H_b\Big(\sum_k w_k d_k\Big) - \sum_k w_k H_b(d_k),
\qquad H_b(p) = -p \log p - (1 - p)\log(1 - p), $$

the closed-form mutual information of the detection channel (`detectability_eig`). It
saturates at $\min\big(H_b(\bar d),\, H(w)\big)$: once every mode agrees on
detectability, the detection bit teaches nothing more. `evaluate_candidates` combines this
bit with the continuous value under a restricted reporting model, in which mode $k$ is
reported with probability $d_k$ independently of the Gaussian value within that mode, and a
nondetection carries no value. The chain rule then gives

$$ I(M; D, Y) = I(M; D) + p_{\mathrm{det}}\, I(M; Y \mid D = 1), \qquad
p_{\mathrm{det}} = \sum_k w_k d_k, $$

where the reported mixture has weights proportional to $w_k d_k$ and its mode information
takes the bound above. The within-mode gain of mode $k$ is weighted by $w_k d_k$, because only
a reported value sharpens the parameters. A record that is never reported scores zero, a
record that is always reported recovers the continuous score, and a nondetection is
informative whenever the $d_k$ differ. The model does not describe selection on the value
itself, such as a flux or contrast threshold that cuts through the predictive distribution of
one mode, because that selection changes the conditional distribution of the reported value.

**Classification gain.** The science question is often a discrete classification $C$ of
the state, such as which class of atmosphere is present. The caller supplies per-mode
class weights $P(c \mid k)$, which is where the domain knowledge enters, and the engine
computes

$$ I(C; Y) = H\Big(\sum_k w_k P(\cdot \mid k)\Big)
- \mathbb{E}_{y \sim \bar p}\Big[ H\Big(\sum_k w_k(y)\, P(\cdot \mid k)\Big) \Big],
\qquad w_k(y) \propto w_k \, \mathcal{N}\big(y;\, y_k,\, s_k^2\big), $$

by stratified draws from the mixture predictive $\bar p$ (`class_eig`); for a
detection-only observation the posterior reweighting is closed-form
(`detection_class_eig`). The gain is bounded by the class entropy $H(C)$, so it
saturates by construction: a settled classification scores zero no matter how much more
precise further data would make the parameters. When modes are class-pure the data
processing inequality gives $I(C; Y) \le I(M; Y)$, so the capped mode-discrimination
bound is a consistent upper surrogate of the classification gain.

## How the pieces interact

One quantity recurs at every scale, the mutual information between a question $Q$ and an
observation, $I(Q; Y)$, and changing what $Q$ is reuses the same machinery. Taking $Q$ to
be the continuous parameters gives the geometric gain, the question of which epoch sharpens
the orbit. Taking $Q$ to be a discrete label, which mode is correct, whether the planet is
detected, or which class the system belongs to, gives the mode-discrimination, detection,
and classification terms, and integrating the likelihood over the prior turns the same
question into the evidence and the Bayes factor $\log Z_1 - \log Z_0$. Chaining the belief
through sequential updating carries each answer into the next observation.

The evidence plays a different role from the gain, and the two must not be swapped. The
expected gain is the acquisition score: it saturates as its question settles, which is
what lets it allocate a finite observing budget across the remaining ignorance. The
accumulated evidence, the running Bayes factor, is the certificate: it hardens without
bound as data accumulate, and it decides when to stop observing and announce. Maximizing
the expected Bayes factor instead of the mutual information keeps paying full,
undiminished value to re-confirm a settled conclusion, while reading a saturated score as
proof mistakes exhausted utility for certainty. One evidence computation serves both
roles; the roles stay distinct.

Because the posterior, the evidence, and the information gain all read one `Posterior`
interface, a method written for one question at one scale transfers to the others. The
architecture overview describes how the code layers compose; this shared information
currency is why they compose.
