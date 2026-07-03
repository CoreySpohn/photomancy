# Orbital parameterizations

Gradient-based inference is only as good as the coordinates it runs in. For
Keplerian orbits the choice of element set determines whether the posterior
geometry near common configurations, circular orbits above all, is smooth,
funneled, or outright non-differentiable. This page records how photomancy
parameterizes orbits today, what the geometry of those choices is, and the
direction the library is moving. It is the reference to consult before
adding a sampler, a prior, or a new orbit forward model.

## Coordinates in the current engine

Three tiers of the orbit stack use three different coordinate systems, each
matched to its job:

- **The NumPyro model** samples $\log_{10} P$ uniformly, eccentricity from
  the Kipping (2013) Beta prior by default, the angles $\omega$, $\Omega$,
  and $M_0$ uniformly, and $\cos i$ uniformly. These are the classical
  elements, chosen for transparent priors.
- **The discovery tier** (`grid_search`) scans the eccentricity vector
  $(e_x, e_y) = e(\cos\omega, \sin\omega)$ together with $\log_{10} P$,
  $\cos i$, $\Omega$, and an epoch fraction. The scan is quasi-random and
  gradient-free.
- **The linear tier** (`thiele_innes`) fits the Thiele-Innes constants
  $(A, B, F, G)$ by ordinary least squares at trial values of the nonlinear
  parameters. The design matrix stays well conditioned at any eccentricity;
  the classical angles are extracted only afterwards.

## The geometry of the circular limit

The classical pair $(e, \omega)$ degenerates as $e \to 0$: the argument of
periastron loses meaning, the posterior develops a ridge over $\omega$, and
any quantity computed through the inverse map $(e_x, e_y) \mapsto
(e, \omega)$ hits non-differentiable functions ($\sqrt{\cdot}$ and
$\mathrm{atan2}$) exactly at the origin. Three practical consequences:

1. **Samplers pay a geometry tax.** Hamiltonian samplers spend leapfrog
   steps exploring the $\omega$ ridge and the periodic seams of the angle
   coordinates. Reparameterizations that remove these features are a
   long-standing recommendation in the exoplanet literature (Ford 2006).
2. **Curvature-based machinery breaks at the mode it needs most.** A
   Hessian evaluated through the polar chart at a circular
   maximum a posteriori point is undefined; Laplace approximations,
   Fisher-information calculations, and information-gain scores built on
   them inherit the failure precisely for near-circular targets.
3. **Posterior transport amplifies noise.** The Jacobian of the map from
   the eccentricity vector to $(e, \omega)$ has a condition number that
   grows as $2/e$, so pushing a Gaussian posterior into the classical chart
   near the circular limit manufactures spurious uncertainty. Marginals for
   $e$ and $\omega$ near zero eccentricity should be reported from samples,
   not from transformed Gaussians.

Nonsingular element sets that remove the $e = 0$ (and $i = 0$) degeneracies
date to the early machine-computation era: Cohen and Hubbard (1962)
introduced a fully nonsingular set, and the equinoctial elements were
consolidated under that name by Broucke and Cefola (1972). The eccentricity
vector used by the discovery tier is the in-plane half of that construction.

A subtlety worth stating plainly: the singularities live in **inverse chart
maps, prior densities, and sampler geometry, not in forward propagation**.
Forward models built natively from any standard element set, including the
classical one at exactly $e = 0$, are smooth. Failures appear when a forward
model is composed through a chart inversion. The design rule is to build the
forward model directly in the sampling coordinates and to convert between
element sets only for reporting.

## Guidance by task

| Task | Coordinates | Reason |
|---|---|---|
| Gradient-based refinement from an informed start | eccentricity vector | smooth through $e = 0$; no angle seams |
| Laplace, Fisher information, expected information gain | eccentricity vector | finite curvature at circular modes |
| Blind discovery scans | eccentricity-vector box | gradient-free, uniform coverage |
| Markov chains started from the prior | classical elements, or temper first | see the caution below |
| Few-epoch linear fits | Thiele-Innes constants | linear, well conditioned at any $e$ |
| Nested sampling | whichever chart gives a tractable unit-cube prior transform | slice sampling is gradient-free |

**Caution on cold starts.** Decoupled nonsingular coordinates are locally
clean but globally disconnected: a chain started far from the solution can
converge into a poor local optimum and stay there. The correlated classical
angles, usually treated as a defect, act as a connecting ridge during global
search. Vector-coordinate chains should therefore always be seeded from the
discovery or linear tier, which is the standard pipeline in photomancy.

## Direction

A forward model parameterized natively by the eccentricity vector and the
mean argument of latitude, with per-orbit coefficients precomputed at
construction, is planned for the orbit stack. In preliminary internal
benchmarks it delivers order-of-magnitude gains in effective samples per
gradient evaluation for near-circular orbits, with exact finite Hessians at
$e = 0$; reproducible benchmarks will accompany the feature. Two boundary
cases are known and deliberately deferred: full equinoctial elements are
singular for exactly retrograde orbits ($i = 180^\circ$), and the quaternion
elements of Cohen and Hubbard (1962) are the principled remedy should
face-on retrograde populations become a target.

## References

- Broucke, R. A. and Cefola, P. J. (1972), "On the equinoctial orbit
  elements", *Celestial Mechanics* 5, 303.
- Cohen, C. J. and Hubbard, E. C. (1962), "A nonsingular set of orbit
  elements", *Astronomical Journal* 67, 10.
- Ford, E. B. (2006), "Improving the efficiency of Markov chain Monte Carlo
  for analyzing the orbits of extrasolar planets", *Astrophysical Journal*
  642, 505.
- Kipping, D. M. (2013), "Parametrizing the exoplanet eccentricity
  distribution with the beta distribution", *MNRAS* 434, L51.
