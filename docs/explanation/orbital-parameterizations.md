# Orbital parameterizations

Gradient-based inference is only as good as the coordinates it runs in. For
Keplerian orbits the choice of element set determines whether the posterior
geometry near common configurations, circular orbits above all, is smooth,
funneled, or outright non-differentiable. This page records how photomancy
parameterizes orbits today, what the geometry of those choices is, and the
direction the library is moving. It is the reference to consult before
adding a sampler, a prior, or a new orbit forward model.

## Coordinates in the current engine

The orbit fit and the discovery algorithms use different coordinates:

- **Chart inference** uses an orbix Classical, EccentricityVector, Frame, or
  Cartesian chart and explicit scalar coordinates. All backends evaluate the
  same system and data terms.
- **The discovery tier** (`grid_search`) scans the eccentricity vector
  $(e_x, e_y) = e(\cos\omega, \sin\omega)$ together with $\log_{10} P$,
  $\cos i$, $\Omega$, and an epoch fraction. The scan is quasi-random and
  gradient-free.
- **The linear tier** (`thiele_innes`) fits the Thiele-Innes constants
  $(A, B, F, G)$ by ordinary least squares at trial values of the nonlinear
  parameters. The design matrix stays well conditioned at any eccentricity;
  the classical angles are extracted only afterwards.

## Posterior tracks and time origins

`orbits_from_samples(samples, Ms_kg, *, reference_day)` returns an
`orbix.system.FrameOrbit` batched over the posterior draws. It uses `T`, `e`,
`cos_i`, `W`, `cos_w`, `sin_w`, and `tp`, plus true planet mass `Mp` when
present. The semimajor axis follows from `T` and the total mass.

The forward models use days from a common observation-time zero. The sampling
model defines `tp = -M0 * T / (2*pi)`, with no calendar epoch in the samples.
Supply `reference_day` as that zero in float days from your `hwoutils.epoch.Epoch`.
Evaluate tracks with `orbit.at(reference_day + observation_times)`; the returned
position and velocity arrays have shape `(K, N, 3)` in AU and AU/day, with
coordinates `(east, north, away)`. Use `reference_day=0.0` when the Epoch already
matches the observation-time origin. Scalar samples produce an unbatched orbit.

## Fitting in chart coordinates

Construct a one-planet density with
{py:func}`~photomancy.orbit.inference.build_chart_logdensity`, then pass it and a
representative parameter dictionary to {py:class}`~photomancy.orbit.ChartOrbitProblem`.
The dictionary contains an `orbit` vector in the chosen chart and a leaf for
each sampled body property or instrument nuisance. The same system builder and
epoch determine the likelihood, posterior predictions, and EIG.

For an interior chart position, `start = problem.flatten(position)` yields the
flat vector expected by MAP, Laplace, NUTS, MCLMC, and Pathfinder. Pass
`problem.logdensity` as the scalar callable, for example
`NUTSBackend().run(problem.logdensity, start, key)`. The orbit Laplace facade
accepts the problem directly as `map_laplace_fit(problem, start)`.

For normalized prior draws, use {py:class}`~photomancy.priors.FactorizedOrbitPrior`
and give each independent scalar coordinate a
{py:class}`~photomancy.priors.LogUniformPrior` or
{py:class}`~photomancy.priors.HalfNormalPrior`. The sampled companion mass enters
the two-body orbit in both the prior transform and likelihood.
`orbit_tempered_smc(problem, SMCBackend(), n_particles=128, key=key)` draws initial
particles from this prior and tempers the data terms.
`orbit_nested_sampling(problem, key=key)` integrates those terms over the same
prior. Both return chart rows. Use `problem.to_physical(posterior.samples[i])`
to decode row `i` with the original fit's chart and epoch.

## Custom likelihoods and planet scales

A custom likelihood is a term. Pass `build_chart_logdensity` a plain callable that
takes the built system (or scene) and returns `(log_likelihood, aux)`, with no
prior terms. Planet scales are explicit independent scalar coordinates:

- `scalars={name: ScalarCoordinate(prior)}`: a sampled radius (`radius_AU`),
  albedo (`albedo`), mass (`mass_Mearth`) or jitter (`jitter_m_per_s`). The
  default is logarithmic; `ScalarCoordinate(prior, logarithmic=False)` samples the
  value itself. A {py:class}`~photomancy.priors.LogUniformPrior` or
  {py:class}`~photomancy.priors.HalfNormalPrior` also supports nested sampling.
- `body_inputs=("radius_AU", "albedo")`: scalars passed to the scene builder.
- `term_inputs={0: ("radius_AU", "albedo")}`: scalars passed as keyword
  arguments to the term at that index, so a callable
  `term(system, *, radius_AU, albedo)` reads the sampled scales directly. Terms
  never share state, so each closure carries its own data.

Several terms add. A prior-only fit passes `terms=()`. Nested sampling integrates
the terms over the factorized prior and returns chart rows;
`problem.to_physical` decodes `Rp`, `Ag`, `Lambda = Ag * Rp**2` and the mass.

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

## Element sets as charts

Established orbit-fitting codes such as radvel (Fulton et al. 2018) and
orbitize! (Blunt et al. 2020) let the user choose among several element
bases, which is valuable because no single basis serves every task. The
maintenance cost of that flexibility depends entirely on where the basis
concept lives in the architecture. photomancy's position:

- **One hub, thin spokes.** The forward model exists in a single internal
  representation, with per-orbit constants precomputed at construction.
  A basis is a *chart*: a bijection onto the hub plus a prior density
  expressed in its own coordinates. Charts convert into the hub, never into
  each other, so there is no quadratic growth of conversion paths.
- **Charts end at the problem seam.** A chart is a small builder that
  assembles the fit problem (logdensity, unflattening, constraints,
  parameter names). Every sampler backend downstream is chart-blind: it
  sees a flat vector and a logdensity. Adding a chart therefore touches no
  backend, no likelihood, and no reporting code.
- **Charts are compiled away.** Because problems are assembled at trace
  time and JIT-compiled, a chart adds no runtime dispatch and no per-call
  conversion cost; the coordinate change fuses into the model graph.
- **Priors bind to charts explicitly.** Where an exact correspondence
  exists it is used directly (a Rayleigh eccentricity prior is a pair of
  independent Gaussians on the eccentricity vector); otherwise the density
  is reweighted with Jacobians obtained by automatic differentiation, never
  by hand.
- **A closed menu.** The supported charts are the ones with a task behind
  them: classical elements, the eccentricity vector, Thiele-Innes
  constants, and the discovery-scan box. Preferences about units, epochs,
  or which parameters to hold fixed are handled by the prior layer and by
  deterministic transforms, not by new bases.

Each chart ships with the same small test contract: an exact round trip
against the hub, forward-model gradient checks at the degenerate loci
($e = 0$, $i = 0$, near-retrograde), and prior-equivalence checks against
the classical chart.

## References

- Blunt, S. et al. (2020), "orbitize!: A comprehensive orbit-fitting
  software package for the high-contrast imaging community",
  *Astronomical Journal* 159, 89.
- Broucke, R. A. and Cefola, P. J. (1972), "On the equinoctial orbit
  elements", *Celestial Mechanics* 5, 303.
- Fulton, B. J. et al. (2018), "RadVel: the radial velocity modeling
  toolkit", *PASP* 130, 044504.
- Cohen, C. J. and Hubbard, E. C. (1962), "A nonsingular set of orbit
  elements", *Astronomical Journal* 67, 10.
- Ford, E. B. (2006), "Improving the efficiency of Markov chain Monte Carlo
  for analyzing the orbits of extrasolar planets", *Astrophysical Journal*
  642, 505.
- Kipping, D. M. (2013), "Parametrizing the exoplanet eccentricity
  distribution with the beta distribution", *MNRAS* 434, L51.
