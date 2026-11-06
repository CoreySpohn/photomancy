# Orbit updater acceptance protocol

This document fixes, before any panel result exists, how a fixed-budget array
orbit updater is graded against reference posteriors. It states the designs
(regimes), the model pairs, the worlds and seeds, every criterion with its
threshold and formula, the reference method and budget, the verdict table, the
rule for an incomplete panel, and what a passing panel may claim. A reader needs
nothing else to reproduce a verdict.

The updater under test assimilates one astrometric epoch at a time into a
fixed-capacity weighted particle belief. Its budget ceiling is 8,192 particles,
32 tempering rungs and 4 mutation moves per rung, in 64-bit floating point. The
panel grades the posterior that this finite budget actually returns, not the
posterior an unbounded sampler would return.

## Status and amendments

- **Status: draft.** Values marked *provisional* below may be re-sized on the
  development panel by the rules in "Provisional values", only inside the
  bounds stated there.
  After re-sizing, the parameter block's `status` becomes `frozen` in a
  dedicated commit, and that commit is the frozen protocol.
- The protocol commit recorded in every verdict row is the latest commit that
  changed this file. `python tools/updater_panel.py --dry-run` prints it.
- After the freeze, the panel may be rerun only after an amendment to this file,
  approved by the maintainer and committed before the rerun. The first result
  stays in the verdict table beside any rerun.
- The panel is a script (`tools/updater_panel.py`), never a test. Its committed
  outputs are the verdict table and the development-panel record.

## Parameter block

The block below is the single machine-readable source for every number the
panel uses. `tools/updater_panel.py` parses it; the prose quotes it. Derived
values (critical counts, the DKW bound, the z quantile) are recomputed from
their formulas by the tool, and a unit test checks that they equal the values
stated here under `stated`.

<!-- updater-protocol-parameters -->
```json
{
  "protocol": "orbit-updater-acceptance",
  "version": 1,
  "status": "draft",
  "distance_pc": 10.0,
  "updater_budget": {
    "particles": 8192,
    "rungs": 32,
    "moves_per_rung": 4,
    "doubled_budget": "particles x 2, rungs and moves unchanged",
    "float": "float64"
  },
  "regimes": [
    {"id": "R1", "name": "short arc", "cadence_days": [0.0, 30.0, 60.0],
     "sigma_mas": 3.5, "truncation": {}},
    {"id": "R2", "name": "multi-revolution", "cadence_days": [0.0, 365.0, 730.0],
     "sigma_mas": 3.5, "truncation": {"period_days": [50.0, 500.0]}},
    {"id": "R3", "name": "near-circular", "cadence_days": [0.0, 180.0, 360.0],
     "sigma_mas": 3.5, "truncation": {"e_max": 0.05}},
    {"id": "R4", "name": "high error", "cadence_days": [0.0, 180.0, 360.0],
     "sigma_mas": 15.0, "truncation": {}}
  ],
  "astrometry": {"radec_correlation": 0.0, "errors_known_to_updater": true},
  "model_pairs": {
    "toy": {
      "graded_by": "mechanics tests",
      "dimension": 2,
      "prior": {"family": "normal", "mean": 0.0, "scale": 1.0},
      "likelihood": {"family": "normal", "scale": 0.5},
      "reference": "closed-form conjugate posterior"
    },
    "ti_window": {
      "graded_by": "panel",
      "mass_model": "implied-mass window",
      "parameterization": ["e", "lnP", "tau", "east_X", "east_Y", "north_X", "north_Y"],
      "prior": {
        "e": "uniform(0, e_max)",
        "lnP": "uniform(ln P_lo, ln P_hi), P in days",
        "tau": "uniform(0, 1), periastron phase relative to t_ref",
        "ti_constants": "independent normal(0, s_ti^2), AU",
        "window": "mu_lo <= (2 pi / P)^2 a(TI)^3 <= mu_hi"
      },
      "defaults": {"s_ti_au": 2.0, "e_max": 0.9, "period_days": [50.0, 5000.0],
                   "mass_window_msun": [0.3333333333333333, 3.0], "t_ref_days": 0.0},
      "likelihood": "independent Gaussian east and north offsets, known sigma",
      "reference": "lambert_depth_fit"
    },
    "physical_fixed": {
      "graded_by": "panel",
      "mass_model": "fixed stellar mass",
      "parameterization": ["a", "e", "cos_i", "W", "omega", "M_ref"],
      "prior": {
        "ln_a": "uniform on the semimajor axes of the period range at fixed mass",
        "e": "Beta(0.867, 3.03) (kipping13) restricted to e < e_max",
        "cos_i": "uniform(-1, 1)",
        "W": "uniform(0, 2 pi)",
        "omega": "uniform(0, 2 pi)",
        "M_ref": "uniform(0, 2 pi), mean anomaly at the anchor epoch"
      },
      "defaults": {"stellar_mass_msun": 1.0, "ecc_prior": "kipping13", "e_max": 0.9,
                   "period_days": [50.0, 5000.0]},
      "likelihood": "independent Gaussian east and north offsets, known sigma",
      "reference": "ofti_scale_and_rotate_importance"
    }
  },
  "panel": {
    "mass_models": ["ti_window", "physical_fixed"],
    "epoch_counts": [2, 3],
    "quantities": ["e", "lnP", "future_east", "future_north"],
    "updates": ["initializer", "epoch_2", "epoch_3"],
    "worlds_per_cell": 400,
    "alias_subset_worlds": 400,
    "development_worlds_per_regime_and_mass_model": 38,
    "tuning_worlds_per_regime_and_mass_model": 6,
    "smoke_worlds": 24
  },
  "prng": {
    "implementation": "threefry2x32",
    "key_type": "typed key, jax.random.key(seed, impl='threefry2x32'), x64 enabled",
    "root_seed": 271828,
    "stages": {"development": 1, "panel": 2, "smoke": 3, "exact": 4},
    "purposes": {
      "truth": 0, "noise": 1, "updater_a": 2, "updater_b": 3,
      "updater_doubled": 4, "reference_1": 5, "reference_2": 6,
      "reference_3": 7, "reference_4": 8, "pit": 9,
      "cert_doubled_1": 10, "cert_doubled_2": 11,
      "cert_proposal_1": 12, "cert_proposal_2": 13, "exact_run": 14
    },
    "simulation_seeds": {"sizing": 6, "alias_freeze": 7}
  },
  "visibility": {"wavelength_nm": 500.0, "aperture_m": 6.0,
                 "iwa_lambda_over_d": 3.0, "owa_lambda_over_d": 30.0},
  "criteria": {
    "coverage": {"interval": 0.90, "reported_interval": 0.50, "family_alpha": 0.05},
    "calibration": {"family_alpha": 0.05},
    "alias_accuracy": {"bins": 8, "quantities": ["lnP", "e", "future_pa"],
                       "family_alpha": 0.05, "delta2": 0.0005, "delta2_max": 0.001,
                       "min_power": 0.8,
                       "alternative": {"damaged_fraction": 0.02, "moved_mass": 0.2},
                       "freeze_panels": 400},
    "failed_updates": {"acceptable_rate": 0.02, "family_alpha": 0.05,
                       "upper_bound_confidence": 0.95},
    "schedule_stability": {"comparisons": ["independent_keys", "doubled_budget"],
                           "tolerance": 0.05, "tolerance_max": 0.05,
                           "max_count": 20, "max_count_cap": 20}
  },
  "reference": {
    "ti_window": {"proposals_per_fit": 1048576, "max_proposals_per_fit": 8388608,
                  "allocation": "equal per Lambert family",
                  "depth_half_width": "(1 + e_max) a_max",
                  "max_revolutions": "ceil(baseline / P_lo)"},
    "physical_fixed": {"proposals_per_fit": 4194304, "max_proposals_per_fit": 33554432,
                       "anchor": "highest-SNR valid epoch"},
    "repeat_noise": {"runs": 4, "bound_fraction_of_delta2": 0.25},
    "convergence": {"bound_fraction_of_delta2": 0.25},
    "cross_method": {"bound_fraction_of_delta2": 0.5,
                     "ti_window": "admissible_region_fit, same model, budget and z_max_au",
                     "physical_fixed": "same weights, anchor at the lowest-SNR valid epoch"},
    "exact_checks": {"sampler_repeats": 16, "family_alpha": 0.01,
                     "bins_per_case": 24, "interval_end_points_per_case": 4,
                     "cases": 2, "quadrature_error_factor": 2.0,
                     "bin_mass_floor": 0.002}
  },
  "provisional": [
    "criteria.alias_accuracy.delta2",
    "reference.ti_window.proposals_per_fit",
    "reference.physical_fixed.proposals_per_fit"
  ],
  "limits": {"working_days": 14, "compute_wait_days": 7},
  "stated": {
    "coverage_critical_count": 339,
    "overcoverage_report_count": 379,
    "dkw_epsilon_5dp": 0.09904,
    "alias_z_3dp": 3.078,
    "failed_updates_critical_count": 17,
    "certification_miss_probability_3dp": 0.046,
    "exact_check_compared_values": 56,
    "exact_check_t_critical_3dp": 4.938
  }
}
```

## Regimes

A regime is a **design**: a cadence, an astrometric error known to the updater,
a mass model and an optional prior truncation. The truncation applies equally to
the truth generator, the updater and the reference, so within a regime all three
share one prior. Truths are drawn from that prior; no world is ever selected,
filtered or reweighted on a property of its truth after drawing, because a
truth-filtered panel is miscalibrated even for an exact posterior.

| Regime | Cadence (days) | Error per axis | Truncation | What it stresses |
|---|---|---|---|---|
| R1 short arc | 0, 30, 60 | 3.5 mas | none | Curvature-limited arcs, broad period posteriors |
| R2 multi-revolution | 0, 365, 730 | 3.5 mas | $50 \le P \le 500$ d | Period aliases across many revolutions |
| R3 near-circular | 0, 180, 360 | 3.5 mas | $e \le 0.05$ | The circular limit of the parameterization |
| R4 high error | 0, 180, 360 | 15 mas | none | The top of the supported error range |

- The distance is 10 pc in every regime, and errors are uncorrelated between
  east and north.
- The nominal error, 3.5 mas, is about one fifth of the diffraction width
  $\lambda/D$ = 17.2 mas at 500 nm for a 6 m aperture, a centroid at modest
  signal-to-noise.
- The supported error range is taken to end at 15 mas per axis. Above it a
  single epoch no longer separates the planet from the inner edge of the prior:
  the smallest semimajor axis in the default TI prior (period 50 days, one third
  of a solar mass) subtends 18 mas at 10 pc, and 15 mas is 0.87 $\lambda/D$.
- The cadences are test designs, not observing recommendations.
- Each world receives three updates: a one-epoch initializer and then epochs 2
  and 3. The posterior after epoch 2 and after epoch 3 are graded separately
  (epoch counts 2 and 3).

The **future epoch** $t_f$ of a posterior is the next epoch of its cadence: for
the two-epoch posterior it is the third cadence epoch; for the three-epoch
posterior it is the third epoch plus the last cadence step (90, 1095, 540 and
540 days for R1 to R4). The future position quantities and the schedule
probability are evaluated at $t_f$.

## Model pairs

A model pair is an updater and the reference it is graded against. Both sides
share one prior, likelihood, support and parameterization. A comparison is
valid only within one pair. Each pair's declaration (its entry in the parameter
block, with the regime truncations applied) is hashed with SHA-256 over
canonical JSON (sorted keys, no whitespace); the dry run prints the hashes, and
a binding with a different hash is a different model.

All pairs share the astrometric likelihood: for accepted epochs $k$ with
measured offsets $(x_k, y_k)$ in arcsec, model offsets $(\hat x_k, \hat y_k)$
and known per-axis error $\sigma$,

$$
\ln L(\theta) = -\sum_k \left[\frac{(x_k - \hat x_k)^2 + (y_k - \hat y_k)^2}{2\sigma^2}
+ \ln(2\pi\sigma^2)\right],
$$

which is `photomancy.orbit.likelihoods.loglike_relative_astrom` with zero
correlation. There is no detection selection, association ambiguity or nuisance
calibration, and the distance is fixed.

### Toy pair (conjugate Gaussian)

The mechanics tests, not the panel, use this pair. The parameter is
$x \in \mathbb{R}^2$ with prior $\mathcal{N}(0, I)$; each record is
$y_k \sim \mathcal{N}(x, 0.5^2 I)$. The reference is the closed-form conjugate
posterior. It isolates the updater's bookkeeping (padding, deduplication, mixed
lanes) from orbit geometry.

### TI pair (implied-mass window)

- **Parameterization.** $\theta = (e, \ln P, \tau, A_x, A_y, B_x, B_y)$, the
  independent Thiele-Innes basis of `photomancy.orbit.few_epoch.TIBasisModel`.
  East is $A_x X + A_y Y$ and north is $B_x X + B_y Y$ (in AU, divided by the
  distance for arcsec), with $X = \cos E - e$, $Y = \sqrt{1-e^2}\sin E$, mean
  anomaly $M = 2\pi[(t - t_\mathrm{ref})/P - \tau]$ and $t_\mathrm{ref} = 0$, the
  first cadence epoch.
- **Prior.** $e \sim U(0, e_\mathrm{max})$; $\ln P \sim U(\ln P_\mathrm{lo},
  \ln P_\mathrm{hi})$ with $P$ in days; $\tau \sim U(0, 1)$; each TI constant
  $\sim \mathcal{N}(0, s_\mathrm{TI}^2)$ in AU; restricted to the implied-mass
  window $\mu_\mathrm{lo} \le (2\pi/P)^2 a^3 \le \mu_\mathrm{hi}$, where $a$ is
  the semimajor axis implied by the TI constants
  (`few_epoch.ti_semimajor_axis`).
- **Defaults, as the code has them** (`TIBasisModel` field defaults):
  $s_\mathrm{TI} = 2$ AU, $e_\mathrm{max} = 0.9$, $P$ from 50 to 5,000 days,
  $\mu_\mathrm{lo} = GM_\odot/3$ and $\mu_\mathrm{hi} = 3GM_\odot$ (a window of
  one third to three solar masses), $t_\mathrm{ref} = 0$. Regimes R2 and R3
  change only `lnP_hi` and `e_max`.
- **Support.** `TIBasisModel.in_support`: $0 < e < e_\mathrm{max}$,
  $\ln P_\mathrm{lo} < \ln P < \ln P_\mathrm{hi}$, the mass window, all finite.
- **Truths.** Drawn in JAX by `tools/updater_panel.py` (`ti_truth`): blocks of
  1,024 candidates, block $j$ keyed by `fold_in(truth_key, j)`, with $e$, $\ln P$
  and $\tau$ uniform on the declared ranges and the TI constants normal; the
  first candidate that satisfies `TIBasisModel.in_support` is the truth. This is
  exact rejection sampling of the declared prior. (`TIBasisModel.sample_prior`
  is not used: it reseeds NumPy from a 31-bit integer and floors $e$ at
  $10^{-6}$.)
- **Reference.** `few_epoch.lambert_depth_fit` on the same model instance (see
  "Reference method and budget").

### Physical pair (fixed mass)

- **Parameterization.** $(a, e, \cos i, \Omega, \omega, M_\mathrm{ref})$ with
  the stellar mass fixed at $1\,M_\odot$ (`hwoutils.constants.Msun2kg`), the
  forward model `photomancy.orbit.forward.predict_relative_astrometry`, and the
  period tied to $a$ by Kepler's third law. $M_\mathrm{ref}$ is the mean
  anomaly at the anchor epoch; a uniform phase at one epoch is a uniform phase
  at every epoch, so the anchor choice does not change the prior.
- **Prior.** $\ln a$ uniform between the semimajor axes of $P_\mathrm{lo}$ and
  $P_\mathrm{hi}$ at fixed mass (equivalently $\ln P$ uniform on that range);
  $e \sim \mathrm{Beta}(0.867, 3.03)$ (the `kipping13` prior of
  `photomancy.orbit.priors`) restricted to $e < e_\mathrm{max}$; $\cos i \sim
  U(-1, 1)$; $\Omega, \omega, M_\mathrm{ref} \sim U(0, 2\pi)$. These are the
  arguments of `photomancy.orbit.ofti.ofti` with `ecc_prior="kipping13"`,
  `e_max` = 0.9 and `log_P_range` = $(\log_{10} 50, \log_{10} 5000)$, the
  regime truncations applied to the same arguments.
- **Support.** $P_\mathrm{lo} \le P \le P_\mathrm{hi}$ and $0 \le e <
  e_\mathrm{max}$.
- **Truths.** Drawn in JAX by `tools/updater_panel.py` (`physical_truth`)
  under the world's truth key, $e$ by rejection from the Beta distribution in
  blocks of 1,024 until $e < e_\mathrm{max}$, with $M_\mathrm{ref}$ the mean
  anomaly at the first cadence epoch.
- **Chart.** The updater may run in any chart (for example
  $\sqrt e(\cos\omega, \sin\omega)$ and a mean longitude) provided its prior
  density carries that chart's Jacobian, so that the prior measure above is
  unchanged.
- **Reference.** The OFTI scale-and-rotate proposal with exact importance
  weights (next section).

## Reference method and budget

Both references are importance samplers whose only approximation is Monte Carlo
error. Each reference fit reports its Kish effective sample size
$\mathrm{ESS} = 1 / \sum_i w_i^2$ for normalized weights $w_i$.

### TI reference

`few_epoch.lambert_depth_fit` conditions on the first and last accepted epochs,
samples their noise-free positions from the Gaussian error kernel (which
cancels those epochs' likelihood exactly), samples two line-of-sight depths and
the log gravitational parameter, solves the Lambert problem in every
(revolutions, way, branch) family, and weights by prior, Jacobian and the
likelihood of the remaining epoch. Exactness needs the proposal to cover the
whole support, so the protocol sets the two arguments the defaults do not
guarantee:

- `z_max_au` $= (1 + e_\mathrm{max})\, a_\mathrm{max}$ with
  $a_\mathrm{max} = (\mu_\mathrm{hi} P_\mathrm{hi}^2 / 4\pi^2)^{1/3}$, the
  largest possible distance from the star. The default of 8 AU does not cover
  the default prior (it needs 15.7 AU).
- `n_max_rev` $= \lceil (t_\mathrm{last} - t_\mathrm{first}) / P_\mathrm{lo}
  \rceil$, so every alias family that fits the baseline is enumerated.
- `n_per_family` $= \lfloor N_\mathrm{prop} / F \rfloor$ with
  $F = 2 + 4\,$`n_max_rev` families and *provisional* $N_\mathrm{prop} = 2^{20}$
  = 1,048,576 proposals per fit, at most $2^{23}$.

The dry run prints both arguments for every regime and epoch count.

### Physical reference

The public `ofti` function accepts proposals against the full likelihood,
including the anchor epoch onto which each orbit is scaled and rotated. The
proposal already draws the anchor position from that epoch's error kernel, so
this counts the anchor epoch twice: in a one-epoch fit the accepted anchor
positions scatter by $\sigma/\sqrt 2$ rather than $\sigma$ (a scatter ratio of
0.708 was measured at the commit this protocol was written against). Its
default acceptance threshold, the largest log likelihood of the first batch, is
a further approximation whenever a later proposal exceeds it.

The reference therefore uses OFTI's proposal with the exact importance weight.
`ofti.ScaleAndRotate.sample_and_score` draws the shape parameters from the prior
and an anchor position $t$ from the anchor epoch's error kernel, and solves for
$(a, \Omega)$ so that the orbit passes through $t$. With $\rho = |t|$, the
Jacobian determinant of the anchor position with respect to the solved
coordinates is $|\partial(x, y)/\partial(\ln a, \Omega)| = \rho^2$ (its inverse,
$\rho^{-2}$, is the determinant of the map from $t$ to $(\ln a, \Omega)$). The
proposal density in $(\ln a, \Omega)$ is therefore the error kernel times
$\rho^2$, and for the prior above the normalized weight is

$$
w \propto \frac{L_\mathrm{other}(\theta)}{\rho^2}\,\mathbb{1}[\theta \in \mathrm{support}],
$$

where $L_\mathrm{other}$ is the likelihood of every accepted epoch except the
anchor (obtained by scoring with the anchor epoch masked invalid; the anchor
values still drive the proposal). The anchor is the highest signal-to-noise
valid epoch, the `ofti` default rule. The budget is *provisional*
$N_\mathrm{prop} = 2^{22}$ = 4,194,304 proposals per fit, at most $2^{25}$.

### Reference certification

A reference is certified for a mass model only after all of the checks below
pass. Checks 1 to 3 use the alias statistic of the next section, with the runs
named in each check as the two sides, pooled over the regimes for each mass
model, epoch count and alias quantity: all 38 development worlds per regime,
152 per mass model. If 2 percent of worlds carried a reference failure, a
152-world sample would contain none of them with probability
$0.98^{152} = 0.046$. Each check compares the upper bound
$\overline{D2} + z\,\mathrm{SE}$ with a fraction of $\delta_2$.

1. **Repeat noise.** Four independent runs at the budget,
   $D2_w = \sum_b (R_{1,b} - R_{3,b})(R_{2,b} - R_{4,b})$; bound at most
   $\delta_2 / 4$. This measures Monte Carlo noise only.
2. **Budget agreement.** Two runs at the budget against two runs at twice the
   budget; bound at most $\delta_2 / 4$.
3. **Proposal agreement.** Two runs of a sampler with a different proposal and
   the same target: for the TI pair `few_epoch.admissible_region_fit` (one
   conditioned epoch, a velocity box proposal) on the same model instance,
   budget and `z_max_au`; for the physical pair the same exact weights with the
   anchor at the lowest signal-to-noise valid epoch. Bound at most
   $\delta_2 / 2$.
4. **Exact checks** (next subsection) pass for the reference.

Checks 2 and 3 are evidence, not bounds. Agreement between two budgets does not
bound the bias of either without a convergence model, and the two proposals
share the target code (the physical pair shares its weighting formula; the TI
samplers share the model and support functions), so a shared error passes both.
A mode that the updater and the reference both miss does not change $D2$ and is
not detectable by the panel. The exact checks are the only accuracy anchor of
the references. A reference that fails a check at its maximum budget is not
certified, and the alias criterion of its mass model is `incomplete` (see
"Incomplete verdicts").

### Exact checks

Two low-dimensional cases are integrated numerically to a demonstrated accuracy
and compared with the sampler on the quantities the panel grades: the eight-bin
masses of $\ln P$, $e$ and the future position angle, and the end points of the
central 90 percent intervals of the future east and north positions.

- **Physical pair, one epoch.** One epoch at $t = 0$ with $\sigma$ = 3.5 mas,
  regime R1 support (so the period truncation is active), future epoch 30 days.
  The posterior is the full physical prior times the one-epoch likelihood. The
  integral is nested: an outer tensor rule over the shape coordinates (Gauss
  rule on the truncated Beta CDF coordinate of $e$, which carries the prior and
  the $e < e_\mathrm{max}$ truncation; Gauss rule on $\cos i$; periodic
  trapezoid rules on $\omega$ and $M_\mathrm{ref}$), and for each shape an inner
  rule over the anchor position $t$ in polar coordinates, with density
  $\mathcal{N}(t; \mathrm{obs}, \Sigma)\,\rho^{-2}$ and the period truncation
  applied exactly as the integration limits
  $r_0 a_\mathrm{lo} \le \rho \le r_0 a_\mathrm{hi}$.
- **TI pair, reduced case.** Two epochs (0 and 60 days, 3.5 mas), the TI model
  with $P$ from 200 to 400 days and a wide mass window ($\mu$ from
  $10^{-3}$ to $10^{3}\,GM_\odot$), future epoch 120 days. Given $(e, \ln P,
  \tau)$ the positions are linear in the four TI constants, whose prior is
  Gaussian, so they are integrated analytically, and the bin masses and the
  future-position distribution (a Gaussian per node) follow from a
  three-dimensional grid: Gauss rules in $e$ and $\ln P$, a periodic trapezoid
  rule in $\tau$. The window's contribution is bounded per node, not ignored:
  since $\lVert c\rVert/\sqrt 2 \le a \le \lVert c\rVert$ for the TI vector $c$,
  a ball of radius $r$ around the conditional mean lies inside the window when
  its image interval does, and the mass outside is at most the
  $\chi^2_4$ tail at $r^2 / \lambda_\mathrm{max}$; the weighted sum of these
  bounds must be below $10^{-6}$.
- **Refinement.** Each rule is evaluated at three successive levels, each
  doubling the nodes per dimension, and the table of every graded value at each
  level is committed. The quadrature error estimate $\epsilon_q$ of a value is
  twice the change between the two finest levels; the finest level must change
  the value by less than the previous refinement did.
- **Comparison.** The sampler (reference, and the updater for information) runs
  16 times with independent keys at its budget. The compared values are, per
  case, the 24 bin masses (8 bins of 3 quantities) and the 4 interval end
  points, so $m = 56$ over the two cases. A value passes when
  $|\bar x - x_q| \le \epsilon_q + \max(t^\star s_x/\sqrt{16},\ f)$, with
  $\bar x$ and $s_x$ the mean and standard deviation over the 16 runs, $x_q$
  the finest quadrature value, and $f = 0.002$ for bin masses ($f = 0$ for end
  points). The critical value is the Student $t$ quantile with 15
  degrees of freedom at $1 - 0.01/(2m)$, $t^\star$ = **4.938** (Bonferroni at
  family level 0.01), so a correct sampler with Gaussian Monte Carlo errors
  and negligible quadrature error fails the whole set of exact checks with
  probability at most 0.01. The floor $f$ handles nearly empty bins, whose
  estimates are far from Gaussian (all 16 runs can return zero); eight bins
  each off by 0.002 add only $3.2\times 10^{-5}$ to $D2$, far below
  $\delta_2$. `--sizing` runs a seeded control (multinomial bin estimates from
  Dirichlet(0.7) masses and Gaussian end points, 20,000 simulated sets): the
  family rejection rate is 0.0099 at ESS 1,000. Limitation: at lower ESS the
  bin estimates are less Gaussian and the rate rises (0.015 at ESS 300), so the
  0.01 calibration holds only for sampler runs with ESS of about 1,000 or more. The tolerance follows
  from the quadrature error and the measured Monte Carlo spread, not from the
  astrometric error. Every value must pass.

## Worlds, seeds and random numbers

- **Panel.** 4 regimes x 2 mass models x 400 worlds = 3,200 worlds; every world
  is graded at both epoch counts and for all four quantities, so each of the 64
  cells has 400 worlds. The alias criterion also uses all 400 worlds.
- **Development.** 38 worlds per regime and mass model (304) on the development
  stage, disjoint from the panel by construction. Tuning of the updater uses
  worlds 0 to 5 of each (48); the freeze summaries and the reference
  certification use all 38.
- **Smoke.** 24 worlds (3 per regime and mass model) on their own stage, for a
  quick end-to-end run of the panel code.
- **Every world counts in every denominator.** A world whose update returns a
  status other than `OK` is scored on the belief it returned.

Random numbers use typed JAX keys of the threefry2x32 implementation, created
explicitly (`jax.random.key(root_seed, impl="threefry2x32")`, independent of the
ambient default), in 64-bit mode. Every draw has a key derived by folding
semantic coordinates into the root key, never by a counter or by execution
order:

```text
world_key = fold_in(fold_in(fold_in(fold_in(key(root_seed), stage),
                                    mass_model_index), regime_index), world_index)
purpose_key = fold_in(world_key, purpose)
draw_key    = fold_in(purpose_key, index)
```

The last index is the update (0 initializer, 1 epoch 2, 2 epoch 3) for updater
runs, and the epoch index (1 for the two-epoch posterior, 2 for the three-epoch
posterior) for reference, certification and PIT draws; truth and noise use 0.

Stage, purpose and index values are in the parameter block (mass models and
regimes are indexed in the order listed there). The development and panel
stages differ in the second fold, so their keys never coincide; development
seeds therefore cannot leak into the panel. Every run role has its own
coordinates: the method and budget of a certification run are in its purpose
(`reference_1` to `reference_4` for the repeat runs, `cert_doubled_*` for twice
the budget, `cert_proposal_*` for the other proposal) and the epoch count in the
last index. The exact checks use the `exact` stage, world index 0, regime index
0 and purpose `exact_run` with the repeat number (0 to 7) as the index. Seeded
NumPy simulations (sizing, the alias freeze) use
`numpy.random.default_rng([root_seed, s, ...])` with the stream numbers `s` under
`simulation_seeds`, followed by canonical indices (for the freeze, the group's
regime, mass model and epoch-count indices in parameter-block order and the
quantity index), never by input order. A unit test enumerates every declared run
role through the production key function, checks that all keys differ, checks
that a deliberately broken key function (one truth key for every world) is
caught, and checks that the keys do not change with JAX's default generator
setting.

## Criteria

Each criterion is a column of the verdict table with its own verdict. The
updater passes only if every criterion passes in every group. Family error rates
are split by Bonferroni over the stated number of tests, which holds under any
dependence between tests. What each per-test level rests on differs:

- **Exact.** The coverage and failed-update cutoffs are exact binomial tails for
  independent worlds at the stated null rate, and the calibration band is the
  exact finite-sample DKW inequality.
- **Tolerance.** Schedule stability is an engineering tolerance on run-to-run
  differences, not a hypothesis test, and no error rate is claimed for it.
- **Approximate.** The alias bound is a normal approximation to the mean of 400
  independent world statistics. Its operating characteristics (pass rate of an
  exact updater, power against the stated alternative) are validated by
  simulation, not guaranteed.

### Coverage

For each cell (regime, mass model, epoch count, quantity; 64 cells) and world,
form the central 90 percent interval of the quantity from the weighted
posterior: with particles sorted by value and cumulative normalized weight
$C_j$, the lower end is the first value with $C_j \ge 0.05$ and the upper end
the first with $C_j \ge 0.95$. The world covers when the truth lies in the
closed interval.

- Per-cell level $\alpha_c = 0.05/64$. The critical count is the largest $c$
  with $\Pr[\mathrm{Bin}(400, 0.90) \le c] \le \alpha_c$, which is **339**. A
  cell fails when 339 or fewer of its 400 worlds cover.
- Over-coverage is reported, not graded: the smallest $k$ with
  $\Pr[\mathrm{Bin}(400, 0.90) \ge k] \le \alpha_c$ is **379**.
- The 50 percent central interval coverage is reported.

### Calibration

For each cell and world, the randomized PIT of the truth $x^\star$ is
$u = F(x^\star{-}) + V\,[F(x^\star) - F(x^\star{-})]$, with
$F(x) = \sum_i w_i \mathbb{1}[x_i \le x]$, $F(x{-})$ the same with strict
inequality, and $V \sim U(0,1)$ drawn under the PIT purpose key. The statistic
is the sup-distance between the empirical CDF of the 400 PITs and the uniform
CDF. By the Dvoretzky-Kiefer-Wolfowitz inequality,

$$
\epsilon = \sqrt{\frac{\ln(2/\alpha)}{2n}}, \qquad \alpha = 0.05/64,\ n = 400,
$$

which evaluates to **0.09904**; a cell fails when the sup-distance exceeds
$\epsilon$. The tool computes $\epsilon$ from the formula, never from the
rounded value. A finite-budget updater with low ESS gives under-dispersed PITs;
the test is of that updater, which is the intent.

### Alias accuracy

For each group (regime, mass model, epoch count; 16 groups) and alias quantity
($\ln P$, $e$ and the future position angle measured east of north; 3
quantities), bin each posterior into 8 equal-width bins over the regime's
support ($[\ln P_\mathrm{lo}, \ln P_\mathrm{hi}]$, $[0, e_\mathrm{max}]$,
$[0, 2\pi)$). For each of the 400 worlds $w$, with bin masses $U_a, U_b$ from the
two independent updater runs and $R_a, R_b$ from two independent reference runs,

$$
D2_w = \sum_{b=1}^{8} (U_{a,b} - R_{a,b})(U_{b,b} - R_{b,b}).
$$

Because the four runs are independent given the data, $D2_w$ is an unbiased
estimate of the squared $L_2$ distance between the *expected* binned outputs of
the updater and of the reference, with no floor from the reference's Monte
Carlo noise. That distance equals the distance to the exact posterior only as
far as each algorithm's expected output is unbiased; self-normalized importance
weights carry a bias of order 1/ESS, and nothing in the panel bounds it beyond
the exact checks.
The group statistic is the mean $\overline{D2}$ with standard error
$\mathrm{SE} = s / \sqrt{400}$.

- The group fails when $\overline{D2} + z\,\mathrm{SE} > \delta_2$, with
  $z = \Phi^{-1}(1 - 0.05/48) =$ **3.078** (48 = 16 groups x 3 quantities).
  This is the approximate bound described above.
- *Provisional* $\delta_2 = 5 \times 10^{-4}$, never above
  $\delta_2^{\max} = 10^{-3}$.
- Required power: at least 0.8 against the stated alternative, an updater
  that, in 2 percent of worlds, moves $\min(0.2, m_2)$ of posterior mass from
  the second-largest bin (mass $m_2$) into the largest.
- All 400 worlds are used, so the chance that a sample contains no damaged world
  under that alternative is $0.98^{400} = 3.1 \times 10^{-4}$, below the per-test
  level $0.05/48 = 1.04 \times 10^{-3}$. (A 200-world sample would miss every
  damaged world with probability 0.018.)
- The total variation $\mathrm{TV}_w = \tfrac12 \sum_b |U_{a,b} - R_{a,b}|$ is
  reported (median and 95th percentile) but not graded, because TV against a
  noisy reference is biased upward.

### Failed updates

For each regime, mass model and update (initializer, epoch 2, epoch 3; 24
groups of 400 independent worlds), count the worlds whose update status is not
`OK` (low ESS, no support, budget exhausted, incomplete ladder). An acceptable
updater fails at most 2 percent of updates. At per-group level $0.05/24$ the
critical count is the smallest $c$ with $\Pr[\mathrm{Bin}(400, 0.02) > c] \le
0.05/24$, which is **17**: a group fails when more than 17 of its 400 worlds
fail. The one-sided 95 percent Clopper-Pearson upper bound on the rate is
reported.

### Schedule stability

For each world, $p$ is the posterior predictive probability that the planet is
visible at $t_f$: the weighted fraction of particles whose projected separation
lies between 3 and 30 $\lambda/D$ at 500 nm for a 6 m aperture (51.6 to 516 mas;
the dry run prints the values).

- **Comparisons.** For each world and each of two comparisons, the two
  independent-key runs, and run $a$ at the nominal budget against the doubled
  budget, $d = |p_a - p_b|$.
- **Tolerance.** A group (regime, mass model, epoch count) fails a comparison
  when more than 20 of its 400 worlds have $d > 0.05$.
- **What it measures.** Run-to-run stability of the scheduling input at the
  declared budget: whether two runs that should agree give the same answer to
  within 5 percentage points in at least 95 percent of worlds. It is not a test
  of calibration and uses no claimed ESS. For orientation, an updater whose
  estimate behaves like $n$ independent draws at $p = 0.5$ exceeds the tolerance
  in a world with probability $2[1 - \Phi(0.05/\sqrt{0.5/n})]$, about 0.025 at
  $n = 1{,}000$ and 0.22 at $n = 300$ (`--sizing` prints these with the group
  failure probabilities), so the tolerance requires a budget whose predictive
  probabilities are effectively that precise.
- **Bounds.** The tolerance (0.05) and the count (20) may only be tightened
  before the freeze, never loosened.

### Mechanics

The updater's mechanics tests (padding after a fit and on an empty belief leave
every leaf bitwise equal; mixed active and inactive lanes under `vmap`; same
keys give the same belief; a duplicate record adds no likelihood; replay; the
conjugate Gaussian pair; a wrong-target mutant caught) are green at the graded
code commit. Any red test fails this criterion.

### Detectable alternatives

`python tools/updater_panel.py --sizing` recomputes these from the formulas
(and the alias case by a seeded simulation):

- A cell whose true coverage is 0.82 fails with probability 0.935 (0.85: 0.466;
  0.88: 0.030). Exact.
- A failed-update group at a true rate of 0.05 fails with probability 0.709,
  at 0.06 with 0.919.
- A schedule comparison in which 10 percent of worlds exceed the tolerance
  fails with probability 0.9998; 5 percent, with 0.44. Exact for independent
  worlds.
- An updater that moves $\min(0.2, m_2)$ of the period mass in 2 percent of
  worlds fails the
  alias test with probability above 0.99 at $\delta_2 = 5\times 10^{-4}$ and
  $10^{-3}$, at updater ESS 300 or 1,000 and reference ESS 4,400 (Dirichlet(0.7)
  bin masses, 400 worlds); an exact updater passes. Simulated.

## Provisional values

Three values may be re-sized on the development panel, before the freeze, by
these rules only and only inside the stated bounds. Each re-sizing is a commit to
the parameter block, made before any panel world is run, together with the
development record that justifies it. If development precision cannot meet a
bound, the response is more sampling (a larger reference budget, up to its
maximum) or an `incomplete` verdict for the affected criterion, never a looser
gate.

1. **$\delta_2$, within $[5\times 10^{-4}, 10^{-3}]$.** An exact updater
   should pass each of the 48 alias tests with probability at least
   $1 - 0.05/48$. For an exact updater $\overline{D2} \approx \mathcal{N}(0,
   \mathrm{SE}_0^2)$, so this needs $\delta_2 \ge 2 z\, \mathrm{SE}_0$. The
   candidate is $\max(5\times 10^{-4},\ 2 z \max \mathrm{SE}_0)$ over groups and
   quantities. It is accepted only if it is at most $10^{-3}$ and, in every
   group and quantity, the one-sided 95 percent Clopper-Pearson lower bound on
   the simulated power against the stated alternative is at least 0.8;
   otherwise the alias criterion is `incomplete` for the groups that miss. The
   calculation is `python tools/updater_panel.py --freeze SUMMARY.json`:
   - *Inputs*: exactly one entry for each of the 16 groups (the tool refuses a
     missing, duplicate or unknown group), each listing all of its development
     worlds by id, exactly the ids 0 to 37 (the tool refuses a missing,
     duplicate or unknown id and sorts the worlds by id before resampling); a
     world carries its updater ESS, its reference ESS and its reference bin
     masses for each alias quantity.
   - *Aggregation*: 400 simulated panels per group and quantity; each panel
     resamples 400 whole development worlds with replacement, keeping each
     world's masses and both ESS values together, and draws the four runs as
     multinomials at those ESS values. $\mathrm{SE}_0$ is the mean panel
     standard error; power is the fraction of panels that fail when the stated
     damage is applied.
   - *Seeds*: stream `alias_freeze` with the canonical group and quantity
     indices.
   - *Output*: the candidate $\delta_2$; per group and quantity
     $\mathrm{SE}_0$, the power with its lower bound, and the exact-updater pass
     rate with its two-sided 95 percent interval (reported, not claimed as a
     level: 400 panels cannot show a rate of $1 - 0.05/48$); and `frozen` or
     `incomplete` with the reasons.
2. **TI reference budget, from $2^{20}$ to at most $2^{23}$ proposals.**
3. **Physical reference budget, from $2^{22}$ to at most $2^{25}$ proposals.**
   Each budget is doubled from its starting value until the reference is
   certified ("Reference certification"); the frozen value is the smallest
   certified budget. Running the certification and the exact checks needs the
   development-panel code (truth draws, reference runs, the quadrature and the
   binned summaries), which is required work that must exist and be committed
   before the freeze.

The schedule tolerance and count may be tightened, not loosened. Proposal
scales, the tempering ladder and the moves of the updater may be tuned on the
development panel; the budget ceiling may not be raised.

## Compute

Per key and model pair, the panel makes 4 regimes x 400 worlds x 3 updates =
4,800 updates. Two independent keys and one doubled-budget run (which costs two)
give 19,200 update-equivalents per pair, each priced at that pair's measured
cost. Each pair needs 4 x 400 x 2 x 2 = 6,400 reference fits (regimes, worlds,
epoch counts, independent runs), and the development certification 3,040
fit-equivalents more (four repeat runs, two runs at twice the budget and two
other-proposal runs, for 152 worlds and two epoch counts). The certification
needs only the references, so it can run while the updater is being written.

`tools/updater_latency_probe.py` times the vmapped astrometric likelihood over
8,192 particles, with compilation timed apart from the steady state and every
call blocked until ready, and scales it to $8192 \times 32 \times 4 =
1{,}048{,}576$ evaluations per update. It also prices one proposal of each
reference. Its output, `docs/updater-latency-probe.json`, records the machine,
library versions and budget; the dry run prices the tally from it only when the
recorded budget equals the protocol's. The likelihood
alone is a lower bound on an update: proposals, reweighting and resampling add
to it. Latency is reported against the 100 ms target and never enters a
verdict.

## Verdict table

The panel writes `docs/updater-verdict.csv`, one row per graded unit:

- `quantity` in {e, lnP, future_east, future_north}: the 64 coverage and
  calibration cells (alias columns filled for e and lnP);
- `quantity` = future_pa: the alias rows for the position angle (16);
- `quantity` = updates, with `epoch_count` 1, 2 or 3 naming the update: the
  failed-update groups (24);
- `quantity` = schedule: the schedule groups (16), each holding both
  comparisons.

Columns, in order: `regime`, `mass_model`, `epoch_count`, `quantity`,
`worlds`, `covered_count`, `covered_50_count`, `pit_sup_distance`, `d2_mean`,
`d2_upper`, `exhausted_count`, `exhausted_cp_upper`,
`schedule_deviations_keys`, `schedule_deviations_budget`, `tv_median`,
`tv_p95`, `latency_ms_per_update`, `memory_mb`, `seeds`, `verdict_coverage`,
`verdict_calibration`, `verdict_alias`, `verdict_failed_updates`,
`verdict_schedule`, `verdict_mechanics`, `protocol_commit`, `code_commit`.

A column that does not apply to a row is empty and its verdict is `na`. A
verdict is `pass`, `fail`, `incomplete` or `na`. `seeds` records the root seed
and stage. `latency_ms_per_update` is the warm batched amortized cost per active
update for the intended batch, and `memory_mb` the peak resident memory for the
intended world, target and design batch; both are reported against a target of
100 ms per update and decide nothing in the verdict.

## Incomplete verdicts

- A group is graded only when all of its 400 worlds have been run. Otherwise
  its verdict is `incomplete`.
- The alias criterion of a mass model is `incomplete` when its reference is not
  certified at the maximum budget, and of a group when the frozen $\delta_2$
  cannot meet its bound or power floor.
- The alias criterion of a mass model is also `incomplete` when the exact
  checks fail for its reference.
- The work has a limit of 14 working days from the first commit of this
  protocol. The clock stops while a panel job waits for a compute slot, for at
  most 7 further days. A panel cut short by either limit is recorded as it
  stands.
- The panel verdict is `fail` if any graded group fails, otherwise `incomplete`
  if any group is incomplete, otherwise `pass`. An incomplete panel is never
  recorded as a fail or a pass; it goes to the maintainer with the partial table.
- Dropping a world, a regime or a cell to complete a panel is not allowed.

## What a passing panel may claim

A pass means: under each declared model pair, at the frozen budget, the
updater's interval coverage, PIT calibration, binned alias agreement with the
reference, failed-update rate and schedule stability meet the thresholds above,
in the four designs above. It does not mean more:

- Each pair declares an astrometry-only likelihood, a distance fixed at 10 pc,
  known errors and its mass model as a prior (the implied-mass window, or a
  fixed mass). Without these, an exact scale degeneracy remains, and nothing is
  claimed about fits that free the distance or the errors.
- Photometry is outside the panel. If photometry is added later, only the
  product $A_g R_p^2$ is identifiable from it; no claim needs radius or albedo
  alone.
- With known mass and distance, two astrometric epochs give four constraints on
  six orbital elements, so the orbit is not locally identified at two epochs;
  three generic epochs can identify it locally. (With reflected-light
  photometry, a free amplitude and a phase law with $q$ free shape parameters,
  the local minimum is $\max(3, q + 1)$ epochs.) The two-epoch cells grade
  calibration, never informativeness.
- Alias accuracy is agreement with the reference. A mode that the updater and
  the reference both miss is not detected; the exact checks are the only
  anchor of reference accuracy, and they cover two low-dimensional cases.
- Schedule stability is a tolerance on run-to-run agreement, not a statement
  about calibration or about the true visibility probability.
- Full Fisher rank does not imply a unique solution, and no complete
  enumeration of solutions is assumed, so alias accuracy is graded against the
  reference posterior, not against a list of roots.
- In the TI parameterization a mirror pair of orbits (reflection through the
  sky plane) is one point, so no mirror-symmetry test applies.
- Near-circular orbits have undefined periastron angles, so no periastron angle
  is graded in any regime.
- The cadences, errors and truncations are test designs. A pass does not certify
  other cadences, errors, priors, distances or masses, multi-planet systems,
  detection selection, or population-level inference.
- Latency and memory are reported, not graded.
