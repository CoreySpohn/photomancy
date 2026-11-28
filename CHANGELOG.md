# Changelog

## Unreleased

* **orbit:** Rename `TIBasisModel` parameters `(A, F, B, G)` -> `(east_X, east_Y, north_X, north_Y)`; coefficient values and order are unchanged.
* **orbit:** `ti_semimajor_axis` now takes `(east_X, east_Y, north_X, north_Y)`
  where it took `(A, F, B, Gc)`; the values and their order are unchanged, so
  positional calls still work and keyword calls must be renamed. The few-epoch
  fitters (`admissible_region_fit`, `lambert_depth_fit`) label their posterior
  columns with `TI_PARAM_NAMES = ("e", "lnP", "tau", "east_X", "east_Y",
  "north_X", "north_Y")` instead of `("e", "lnP", "tau", "A", "F", "B", "G")`,
  so code that indexes a saved few-epoch posterior by the old names must use the
  new ones. Such posteriors live in Thiele-Innes coordinates and cannot be
  converted by `migrate_sky_convention`; refit them.
* **orbit:** Add `migrate_sky_convention(samples)` for saved physical-element
  samples from astrometry-only fits made with earlier releases: it applies
  `cos_i -> -cos_i` and `W -> (pi/2 - W) mod 2 pi` and raises on posteriors stored
  in transformed sampling coordinates. See the how-to page on migrating saved
  fits.
* **deps:** Require `orbix>=3.0.0`, whose raw position and velocity rows are right-handed (east, north, away), with dz/dt positive when receding.
* **orbit:** Relative and stellar astrometry, Thiele-Innes fits, OFTI and EIG now
  use (Delta alpha*, Delta Dec) = (east, north) and the standard phase angle
  beta, with cos(beta) = z/r (z away).
  Astrometry-only fits made with earlier releases carry mirrored elements and
  convert by i' = 180 deg - i, Omega' = (90 deg - Omega) mod 360, with a, e,
  omega and the epoch terms unchanged. EIG and forward-model photometry
  phases were inverted, so fits and forecasts that used photometry must be
  refit rather than converted. Pass east offsets (Delta RA cos Dec), not
  Delta RA, as `RelativeAstromData.ra`.

## 2.0.0 (2026-11-28)


### ⚠ BREAKING CHANGES

* **image:** read roll from the exposure pointing
* **disk:** isotropic inclination, one-turn position angle and sign-conditioned slopes in default_disk_prior
* **precision:** route float64 inference conversions through one helper, refuse narrow posteriors with PrecisionError and enforce the precision guard
* **infer:** run float32 forwards inside float64 fits with destination-typed writes, float64 scoring and owner-checked curvature
* **photomancy:** decide RV, roll, image-support, censoring and detection-bracket admission identically in both lanes, follow edited orbit walls, require float64 for image scoring
* **image:** ImageTerm(mask=None) is removed, use ImageTerm.on_support(model, ...) or an explicit mask. A term returns NaN where a proposal loses a scored pixel, and an observation, mask or prediction of the wrong shape is refused. Requires a coronagraphoto with planet_source_support.
* **infer:** a MAP stage fit whose coordinates differ from the run fit raises ValueError, Fit.draw_schema and Fit.check_draws validate draws, and init draws, hand-built Posterior samples and leaf replacements must match the trailing shapes of the free entries.
* **detection:** profile_amplitude returns a fourth value, resolved, and the default n_newton is 8. The detection z is NaN where the profile is unresolved.
* **infer:** Posterior.with_priors now refuses (NaN weights and evidence, info.support_covered false) a swap to a law that fills the gap between a mixture's components or the support of a zero-weight component, any change of a block's non-law leaves, static fields or structure (another time_d of a SkyPositionBlock, a numeric bound a block builds its law from, an a_AU orbit law replaced by a period_d law even when equivalent); a block may define support_contains(new) to decide containment itself.
* **infer:** StageTrace gains regressed and final_loss; a stage whose returned loss is non-finite sets failed, and one worse than the trace best by more than the stall tolerance sets regressed. The returned point is still the last iterate.
* **postproc:** MawetArm takes every option by keyword and MatchedFilterArm every option after the provider, so a positional value can no longer bind to an unrelated option (MawetArm(4.0) now raises TypeError).
* **infer:** LikelihoodEstimate gains converged, positive_definite, finite, log_likelihood, n_steps, stopped_early and n_newton; min_eigenvalue is now the smallest eigenvalue of the correlation-form information and grad_norm the scaled gradient norm; measure's max_decrement default is 1e-3; map_fit ends a stage with StageTrace.failed on a non-finite loss.
* **infer:** with_priors refuses a swap whose new support is not inside the old (NaN weights and evidence, info.refused true); ReweightInfo gains support_covered and log_mass_ratio; Posterior.resample returns kind monte_carlo for a grid; SheetSum.draw samples one sheet with probability one half shared by its children instead of the stored sheet.
* **measurements:** RadialVelocityData takes instruments (unique names) in place of n_inst, and zeros requires them; two RadialVelocityTerm of one instrument are refused by Fit and update merges them; update raises when a new term shares latents with two held terms; plot_rv takes m/s (renamed rv_m_per_s) and selects observations by valid
* **postproc:** MawetArm and MatchedFilterArm no longer take fwhm_px; they read FrameSet.fwhm_px and take aperture_fwhm (a multiple of it). Both arms refuse frames at different telescope_pa_deg (ValueError when concrete, NaN statistics under a trace). A candidate whose filter footprint touches an excluded pixel now returns NaN, and reference apertures whose footprint touches one are dropped. AbstractFilter subclasses must implement prepare_validity and evaluate_validity. MawetArm.detect reads science.fwhm_px as a concrete host value, so under jit the FrameSet must be closed over.
* **infer:** Fit converts the leaves of its model with hwoutils.leaves.canonical and fit.model is the converted tree, not the object passed in.
* **infer:** Fit refuses a model with NumPy or Python-number leaves (convert with hwoutils.leaves.canonical); a stage prepare and callback records receive JAX arrays instead of NumPy copies; Laplace finds each mode with the scaled MAP stage and its stopping rule.
* **infer:** the default unscaled MAP stage no longer stops early and runs all n_steps; the scaled stage stops only when the loss has stalled and the scaled gradient norm is below 1e-3 (stopped_early certifies that). detect checks the render-once path with one more render and profiles a nonlinear prediction by Newton steps through the fit's likelihood. batch_size=None in detect and likelihood_ratio_map now means all at once (jax.vmap).
* **infer:** the default MAP stage now ends early (five steps without a 1e-7 gain, read every five steps), so n_steps is the most steps and a run returns the converged point instead of stepping on.
* **infer:** MAP scales its coordinates by the information scale by default (scale=False restores the old path); detect and likelihood_ratio_map default to batch_size=1; measure maximizes with the MAP optimizer, adds grad_norm and newton_decrement to LikelihoodEstimate, requires a small Newton decrement for regular, and raises MemoryLimit above max_bytes.
* **measurements:** PhotometryTerm and CountRateTable no longer take a reads argument and no longer refuse a resolution without reads, since table knots are static and no fit can free them.
* **exposure:** photomancy.exposure is removed. Score detector frames with photomancy.image.ImageTerm, and any other forward model or noise law with photomancy.infer.ForwardTerm(forward, likelihood), where the forward is an eqx.Module that renders model.exposures[i] (what ExposureTerm's render and exposure_index did).
* **measurements:** the band-contrast helper behind PhotometryTerm and PrecomputedReportPredictor now requires a Resolution, so a precomputed report predictor of a scene with a tabulated source needs CountRateTable(..., resolution=Resolution.build(quadrature, *scene.spectral_tables())) and raises LookupError without it; PhotometryTerm.reads is now a field filled by Target.terms from Scene.knot_leaves, ReportTerm.reads is a property returning the table's reads, and CountRateTable gains resolution and reads fields.
* **infer:** NUTS and MCLMC now raise ValueError before sampling when Fit.walls is not empty (a block law with bounds its coordinates do not map, such as a bounded law on the Cartesian chart or any Thiele-Innes block); use a block whose coordinates carry the bounds, a law without them, or Importance or NestedSampling. Fit.walls is new.
* **infer:** Fit now refuses a leaf law on a field of a stored orbit (dynamics.orbits.a_AU, ex, n_rad_per_d, quat) and a leaf law on a star or planet mass when no entry writes the orbits; free the orbits through an orbit block, which re-rates held rows from the free masses.
* **measurements:** PhotometryTerm holds a resolution of the system's tabulated spectra, and TargetData.terms builds its quadrature from the filter curves alone, with each band trimmed to the filter's nonzero extent. A system passed to band_contrasts by the term must accept a resolution keyword.
* **image:** refuse a mask that scores unsupported pixels or a planet outside the source domain at fit setup
* **postproc:** read Coronagraph.coronagraph_at in the yippy template
* **measurements:** name Transmission tuples transmissions and pass transmission to band_contrasts
* **postprocessing:** state background_variance_factor in the calculator example
* **postproc:** frame_time_s, valid, *Arm, east/north_arcsec and separation_arcsec names
* **infer:** renames, old -> new: AbstractConditionalTerm -> AbstractProposingTerm; LeafConditional -> LeafProposal; AbstractLinearTerm -> AbstractLinearGaussianTerm; LinearForm -> LinearGaussian; the linear-form and leaf-conditional term methods -> linear_gaussian and leaf_proposal; planet_id and row -> planet; evaluate_candidates (dict) -> eig_linearized (LinearizedEIG); evaluate_orbit_candidates -> eig_astrometry(posterior, time_d, obs_variance_arcsec2, *, key, n_modes=4, ...) returning AstrometryEIG; FitStage -> MAPStage (free -> groups, logdensity -> fit); MCLMC n_tune -> n_warmup; Pathfinder maxiter, maxcor -> n_steps, history_size; NestedInfo.ess -> n_eff; cluster_to_mixture k, iters -> n_modes, n_steps; midpoint -> midpoint_rule; gauss_legendre -> gauss_legendre_rule; geometric_eig -> eig_geometric, alias_breaking_eig -> eig_alias_breaking, alias_breaking_eig_from_covariances -> eig_alias_breaking_from_covariances, detectability_eig -> eig_detectability, detection_class_eig -> eig_detection_class, class_eig -> eig_class, detection_channel_eig -> eig_detection_channel; probit_gaussian_mass, ep_probit_update and null_update are private; max_n has no default.
* **infer:** cluster_to_mixture standardizes and wraps its distances and refines the clusters by Gaussian reassignment, so the mixture it returns differs from the unscaled k-means result for the same key; it also accepts init=.
* **infer:** LaplaceInfo.means and covs and PathfinderInfo.means and covs are renamed flat_means and flat_covs; LaplaceInfo gains mode_index and modes.
* **infer:** update(...).info is an UpdateInfo; gate reads move from info.n_eff, info.khat and info.refused to info.reweight.n_eff, info.reweight.khat and info.reweight.refused, and info.refit holds the refit backend's info.
* **infer:** ForwardTerm raises TypeError for a likelihood that is not an eqx.Module (a lambda or closure), and the likelihood method sample(key, predicted) is now draw(key, predicted), also for ExposureTerm.
* **infer:** removed the *Backend classes (LaplaceBackend, LaplaceMixtureBackend, PathfinderBackend, PathfinderMixtureBackend, NUTSBackend, MCLMCBackend, JaxnsBackend; use run(fit, Laplace|Pathfinder|NUTS|MCLMC|NestedSampling)), LegacyBackend, laplace_fit, laplace_covariance, build_scene_nested_model and jaxns_prior, TermLogDensity (use Fit with ForwardTerm), and the modules photomancy.infer.legacy_posterior (GaussianPosterior, MixturePosterior, SamplePosterior, PointPosterior, AbstractPosterior; use Posterior), photomancy.infer.model (build_scene_logdensity, build_gaussian_fit, AbstractLogDensity; use Fit), photomancy.infer.priors (IndependentPrior; use priorities laws) and photomancy.infer.posterior_utils (project_samples, sample_capped). infer exports the Info classes MAPInfo, MCLMCInfo, NUTSInfo and NestedInfo.
* **infer:** AbstractProposal.weigh returns a fourth value, valid, and Completion.complete returns the pool Kish size of each draw as a fourth value. A completed orbit proposal whose pool is less than half effective gives a NaN evidence (info.proposal_valid false); the draws and weights are still returned.
* **infer:** evaluate_candidates scores a priorities.Mixture of MultivariateNormal modes (what cluster_to_mixture returns) instead of a MixturePosterior or a duck of it.
* **exposure:** ExposureLogDensity is removed; build Fit(ExposureModel(...), free, terms) and run it with MAP, giving each stage a Fit over a subset of the terms.
* **infer:** MAPBackend is removed; run a staged fit as run(fit, MAP(...)).
* **disk:** build_disk_logdensity, disk_fit_leaves, build_retrieval_logdensity and abundance_fit_leaves are removed. default_disk_prior and default_abundance_prior return a dict of laws keyed by model path for Fit.free, abundance_paths replaces abundance_fit_leaves, and radiance_forward and contrast_forward return eqx.Module forwards. Fit with a ForwardTerm and run with MAP or Laplace replace the log-density builders.
* **measurements:** the data classes name their time field time_d (pad, from_sep_pa, from_dmag, from_contrast_curves, det_time_d, null_time_d) and MAX_* leave the package exports
* **infer:** with_priors, append_term, replace_term and update return the gate reading as ReweightInfo(n_eff, khat, refused)
* **orbit:** any_epoch(n_draws) sizes the returned cloud only; the adaptation rounds keep n_adapt_draws
* **infer:** remove the SMC backend
* **infer:** n_adapt_draws, defensive_fraction, reached_target, n_eff_per_round, append_term, model_at and update(posterior)
* **orbit:** any_epoch, few_epoch(n_draws), ShapePriorProposal, n_pool and classical_elements; orbit exports its factories
* **infer:** proposals weigh their draws; Fit.from_flat and LeafConditional.sample give the orbit engines public homes
* **measurements:** ReportData, ReportRule, CountRateTable, PrecomputedReportPredictor and log_ratio
* **measurements:** AstrometryData, RadialVelocityData and ProperMotionAnomalyData name the data classes
* **measurements:** DMagData and loglike_dmag replace ImagingData, NullData and the null kernel; the dMag area is a model leaf
* **orbit:** score candidate epochs from a Posterior's clustered modes
* **orbit:** SheetSum owns the sheet sum, ReportTerm loses orbit_path, Fit refuses two sums over one latent
* **orbit:** delete the legacy orbit engines, chart problems and root priors
* **measurements:** ReportTerm scores counts only and ReportRows.astrometry hands detection positions to the astrometry term
* **infer:** importance evidence and its error are NaN above the k-hat limit, ImportanceInfo.khat
* **orbit:** CompletedProposal takes n_constants=None (8 with no term left to score, 32 otherwise) and n_scored=4; Completion.complete takes n_scored; Completion.view and the theta view class are gone (a proposal over theta is given the Completion itself, which answers periods(), block and model); Completion.from_fit refuses a free entry that is neither an orbit coordinate, a constant nor a proposed leaf, in place of the ThetaPrior count check
* **orbit:** delete migrate_sky_convention and scan the package for deleted names
* **measurements:** normalize the flat-offset RV marginal with 0.5 sum_j log N_j and re-baseline its goldens
* **measurements:** delete the jitter_m_per_s and area_AU2 call keywords of the RV and dMag terms
* **orbit:** move system_density into photomancy.orbit and drop its root exports
* **orbit:** delete OrbitUpdater with its kernel, belief, fixed-mass and conjugate targets, tests, tools and protocol docs
* **infer:** with_priors re-expresses draws across a change of support
* **infer:** Importance and AdaptiveImportance, Laplace and Pathfinder as importance over mixtures, batch_size on run
* **infer:** periodic cell reduction, Fit.periods, chain-aware from_unit, proposals and next_temperature
* **infer:** categorical_indices takes the key first
* **measurements:** DMagTerm refuses to draw a null epoch
* **measurements:** relative astrometry linear form in arcsec, capability by type, mapped orbit block in tests
* **measurements:** move data, kernels and terms into photomancy.measurements, select becomes system_path, DMagTerm replaces NullTerm
* **detection:** detect and measure take a Fit and measure returns a LikelihoodEstimate
* **image:** ImageTerm runs on a scene-rooted model, ImageModel holds the instrument, exposures returns
* **infer:** EIG estimators take a term and the nested form excludes the outer draw
* **infer:** value_and_prediction leaves AbstractTerm and stays a private legacy method
* **infer:** a reweight keeps its evidence error and every operation takes one slice
* **infer:** posterior operations: sample, resample, with_priors, append, model, forecast, and the two EIG estimators
* **infer:** a NaN target refuses the slice, importance_evidence lives in evidence, init is typed, batching tested
* rename ExposureModel.exposures to nuisance
* rename ImagingTerm to DMagTerm and mark it legacy
* **infer:** add ImageTerm, detect and measure over the expected electron rates
* **infer:** add Posterior, run and generic backends over Fit; move the old posterior types to legacy_posterior
* **infer:** add the Haar constant to the legacy chart densities now that orbix volumes are normalized
* **infer:** add Fit over free paths and blocks with scalar AbstractTerm terms
* **infer:** split the conditional mc_se into ratio_mc_se
* **priors:** name sample_shape and q as the distribution API does, and document the coronagraphoto exposure calls
* **priors:** take eccentricity laws from skyscapes and adopt the orbix element API
* **postproc:** rename MatchedFilterPostProc to MatchedFilterPostProcessing
* drop the photomancy.infer re-exports from the package root
* **postproc:** move post-processing in from coronalyze and drop the imaging extra
* take generic priors from the priorities library and guard the photomancy.infer import graph
* make blackjax and jaxns optional extras imported when a backend runs
* move the generic inference machinery into photomancy.infer
* import grids and spectral types from optixstuff and declare hwoutils[jax]
* **orbit:** build and read frame orbits through the orbix FrameOrbit methods
* take the merged skyscapes Scene in disk fitting and tests
* delete the NumPyro orbit model, bridge, forward predictors and legacy routes
* declare chart and epoch on every chart posterior and check them in EIG, diagnostics and initialization
* back the fixed-mass and Thiele-Innes targets with the chart density and declare chart and epoch on samples
* run MAP, Laplace, sampling, SMC and nested backends on the chart density
* evaluate band contrasts through one multi-band quadrature
* pass star_planet_distance_AU to skyscapes contrast_cube
* stamp report tables with their epoch and refuse a mismatched model epoch
* share one reporting law between refit and report, and move particle weights and resampling into photomancy.particles
* take projected-orbit scale, gap and mirror completion from orbix
* fit Thiele-Innes constants with the orbix masked solve, treat separations beyond the contrast curve as undetectable in orbit EIG, and name the atmosphere forward distance dist_AU
* remove photomancy.fields; use hwoutils random fields, conversions and eyepiece colours
* Thiele-Innes inversion from orbix, one contrast-curve interpolation, no deprecated eccentricity transform
* photometry through skyscapes and geometry through orbix, one eccentricity prior and one proper-motion anomaly
* fit disks through the scene photon radiance on a caller-supplied sky grid
* **orbit:** orbits_from_samples returns a FrameOrbit and takes the reference day

### Features

* a chart-coordinate system density with an orbit prior against the reference measure ([e787070](https://github.com/CoreySpohn/photomancy/commit/e787070b6c45c849ede02fdb68b63726499cd2ec))
* add chart routes for Thiele-Innes, few-epoch, projected, grid search, OFTI and initialization ([7bdabdf](https://github.com/CoreySpohn/photomancy/commit/7bdabdf1b94c6a68f1173179f006947bb7dc0f20))
* back the fixed-mass and Thiele-Innes targets with the chart density and declare chart and epoch on samples ([049808e](https://github.com/CoreySpohn/photomancy/commit/049808edafe0382d6beed4fa1af7065f0c549224))
* complete physical priors, nuisance coordinates and Frame and Cartesian charts in the chart density ([15a16ef](https://github.com/CoreySpohn/photomancy/commit/15a16ef98c38d9988b40ecafe9500d97146ded99))
* declare chart and epoch on every chart posterior and check them in EIG, diagnostics and initialization ([be54fbe](https://github.com/CoreySpohn/photomancy/commit/be54fbe76be35cf79ddb7698fb44627051ee74de))
* delete the NumPyro orbit model, bridge, forward predictors and legacy routes ([815da7d](https://github.com/CoreySpohn/photomancy/commit/815da7d2b41dbbe7426e9b5af5a523cb5736a73c))
* **detection:** detect and measure take a Fit and measure returns a LikelihoodEstimate ([79b86ba](https://github.com/CoreySpohn/photomancy/commit/79b86bad2dbe22c441d1d4fe85eb0bf2008f0ccd))
* **detection:** report regular and the smallest eigenvalue from measure, invert in correlation form ([dda1ad2](https://github.com/CoreySpohn/photomancy/commit/dda1ad21594ca29117cb77da8f792b4ea760d1c8))
* **disk:** fit the disk and the atmosphere abundances through Fit and ForwardTerm ([b5b80eb](https://github.com/CoreySpohn/photomancy/commit/b5b80ebca143dab54e1f9519581da81ad08fd28c))
* **disk:** isotropic inclination, one-turn position angle and sign-conditioned slopes in default_disk_prior ([383f24c](https://github.com/CoreySpohn/photomancy/commit/383f24c6fba18779e56ee64b74b566cc0c08fcde))
* **exposure:** delete ExposureModel and ExposureTerm ([a8bf3e7](https://github.com/CoreySpohn/photomancy/commit/a8bf3e702d4922626b25cc044e1dddf652abfee9))
* **exposure:** fit exposures through run and MAP ([f911580](https://github.com/CoreySpohn/photomancy/commit/f9115805e33f4beb284eb764b5439c319a031f90))
* fit disks through the scene photon radiance on a caller-supplied sky grid ([da4291a](https://github.com/CoreySpohn/photomancy/commit/da4291a83e4ff579929af983fe9b8981169d8f2a))
* **fit:** terms name their assumptions, and Fit and the detection results record them ([ad9a95e](https://github.com/CoreySpohn/photomancy/commit/ad9a95eff7b96653935e3e6403a3d11108f16c2d))
* **image:** add static_beyond_speckle to ImageTerm to score static light beyond a speckle field ([496d0bd](https://github.com/CoreySpohn/photomancy/commit/496d0bda1ea48b128579636f1f72ced9f9349c97))
* **image:** explicit scored mask, shape checks and the source-domain call ([92097fa](https://github.com/CoreySpohn/photomancy/commit/92097faefafadb476508dcbcf5b656b131b9e2eb))
* **image:** ImageTerm runs on a scene-rooted model, ImageModel holds the instrument, exposures returns ([d973eb5](https://github.com/CoreySpohn/photomancy/commit/d973eb5fbec311833f58aded8ae33f94464aa361))
* **image:** ImageTerm takes the observed Image, checked against the detector grid and roll, and mask=None scores every pixel ([7c4fb20](https://github.com/CoreySpohn/photomancy/commit/7c4fb20635fcf8176d838b32aca465c30a8bb0da))
* **image:** refuse a mask that scores unsupported pixels or a planet outside the source domain at fit setup ([7af4bfd](https://github.com/CoreySpohn/photomancy/commit/7af4bfd027792aa115098a77b60ab8d5358e7229))
* **infer:** add a pool-validity condition to the importance evidence ([82236bc](https://github.com/CoreySpohn/photomancy/commit/82236bcec3bc146c0f2727c9dbd912f1545fab94))
* **infer:** add AbstractConditionalTerm and LeafConditional, a leaf record that is never a target ([4bb234a](https://github.com/CoreySpohn/photomancy/commit/4bb234a70fcb3807adc397b877191582a99c4c0e))
* **infer:** add conditional mc_se and tie handling in pareto_khat ([4f214e1](https://github.com/CoreySpohn/photomancy/commit/4f214e10a285ee406bf29fc313d3708eadb7b814))
* **infer:** add Fit over free paths and blocks with scalar AbstractTerm terms ([258275b](https://github.com/CoreySpohn/photomancy/commit/258275b85cc179e9c8d31ee8b21b0b6aab0e71d0))
* **infer:** add Fit.information and term Fisher roots, fold the dense Hessian into fit ([07a441b](https://github.com/CoreySpohn/photomancy/commit/07a441b6401fcc85e0ea9039289f5231468865d2))
* **infer:** add ForwardTerm, the likelihood of a forward module as a Fit term ([9e56691](https://github.com/CoreySpohn/photomancy/commit/9e56691be1feaaf1d50dd94713b3494cdf55ec72))
* **infer:** add free diagnostics (khat, mc_se, weighted moments and quantiles) ([95b4be8](https://github.com/CoreySpohn/photomancy/commit/95b4be87eeb6b2fba44955c12084884765bc7795))
* **infer:** add ImageTerm, detect and measure over the expected electron rates ([a0c1ee3](https://github.com/CoreySpohn/photomancy/commit/a0c1ee31c606bf5a9c855eb83d81d354975be9ac))
* **infer:** add Posterior, run and generic backends over Fit; move the old posterior types to legacy_posterior ([787c3a4](https://github.com/CoreySpohn/photomancy/commit/787c3a428213b76704154564c8df0b73a6232230))
* **infer:** cluster_to_mixture takes init centers and clusters in standardized coordinates ([be63802](https://github.com/CoreySpohn/photomancy/commit/be6380287dd236b70782fc08e1200204b2414731))
* **infer:** delete the legacy inference stack ([05ccc93](https://github.com/CoreySpohn/photomancy/commit/05ccc93c821733535df73897ace642e4de0190dd))
* **infer:** evaluate_candidates takes a priorities.Mixture ([5ee08b7](https://github.com/CoreySpohn/photomancy/commit/5ee08b7ca72e94897606ffb9ebacdac4fa557a98))
* **infer:** Fit converts model leaves at construction ([c7b44ae](https://github.com/CoreySpohn/photomancy/commit/c7b44aedb0b0d980410fbbede780c87ff83b5d4e))
* **infer:** Fit refuses an entry that breaks a tie a model node declares ([aa3b5d1](https://github.com/CoreySpohn/photomancy/commit/aa3b5d1d50b702e7fcc0ccac67142cc93f026a4e))
* **infer:** Fit refuses non-JAX leaves, stages get device arrays, Laplace stops by the MAP rule ([d1332a5](https://github.com/CoreySpohn/photomancy/commit/d1332a5bb8c42871f97af66f81983c8bd56daec9))
* **infer:** Fit.to_flat places stored draws in the flat unconstrained vector, the inverse of from_flat ([7cc33e2](https://github.com/CoreySpohn/photomancy/commit/7cc33e22019a52ff5135edaa403421b8d8acd7da))
* **infer:** Importance and AdaptiveImportance, Laplace and Pathfinder as importance over mixtures, batch_size on run ([4c94787](https://github.com/CoreySpohn/photomancy/commit/4c94787457bd710bb80251094cd03c8a65b79d31))
* **infer:** importance evidence and its error are NaN above the k-hat limit, ImportanceInfo.khat ([e9fde69](https://github.com/CoreySpohn/photomancy/commit/e9fde69e549de1e42ebf9dc15bae40dea6ac362a))
* **infer:** Laplace and Pathfinder start from several explicit entry dicts ([e17046e](https://github.com/CoreySpohn/photomancy/commit/e17046e88c1ef24ef4a111b6b35d888fdc4abb3b))
* **infer:** MAP stages score a Fit view and closed forms read pre-step predictions ([0d981b0](https://github.com/CoreySpohn/photomancy/commit/0d981b0183c8bc03dee895fd0456c065d4943099))
* **infer:** merge coincident Laplace modes and expose mode_index and modes ([776588f](https://github.com/CoreySpohn/photomancy/commit/776588f0e6709a1bbda14911f7f6675c5b1ce1e6))
* **infer:** n_adapt_draws, defensive_fraction, reached_target, n_eff_per_round, append_term, model_at and update(posterior) ([ee9cda8](https://github.com/CoreySpohn/photomancy/commit/ee9cda8fe6c5276100d2bfd5352637af9ae11657))
* **infer:** non-temperable proposals refit on the fit density, defensive proposal keeps its prior ([ead49d4](https://github.com/CoreySpohn/photomancy/commit/ead49d4676d5f2526d204759182502a60fa8bad6))
* **infer:** NUTS and MCLMC refuse a fit whose blocks declare walls ([cb28a1f](https://github.com/CoreySpohn/photomancy/commit/cb28a1f1281b233de63a5075f19e5a26f90a0d89))
* **infer:** periodic cell reduction, Fit.periods, chain-aware from_unit, proposals and next_temperature ([7303bdb](https://github.com/CoreySpohn/photomancy/commit/7303bdb716c8f12dd960c88fd49e1cf125d0bfa9))
* **infer:** posterior operations: sample, resample, with_priors, append, model, forecast, and the two EIG estimators ([384e5e8](https://github.com/CoreySpohn/photomancy/commit/384e5e80288891fc3d104fc8bb02280e7d5c23cd))
* **infer:** proposals weigh their draws; Fit.from_flat and LeafConditional.sample give the orbit engines public homes ([835b40a](https://github.com/CoreySpohn/photomancy/commit/835b40ac4b2c788cb5ff56ffcd3b7871c5ff1d84))
* **infer:** remove the SMC backend ([6852529](https://github.com/CoreySpohn/photomancy/commit/6852529ff97715c94bb7ec437e28ee0406ac521c))
* **infer:** replace_term and update, batch_size on the reweighting operations and both EIG estimators ([5281ed9](https://github.com/CoreySpohn/photomancy/commit/5281ed93371904c454a3b202355d531afceaa97d))
* **infer:** return UpdateInfo from update with the refit's own info ([5ecf83a](https://github.com/CoreySpohn/photomancy/commit/5ecf83a57e0a656b06e91d4041dfaf481cd9ad5c))
* **infer:** run float32 forwards inside float64 fits with destination-typed writes, float64 scoring and owner-checked curvature ([11d6d84](https://github.com/CoreySpohn/photomancy/commit/11d6d84d372ceef425ef48fc2a481e4101c6ecd1))
* **infer:** scale MAP coordinates, render detect once, converge measure with a stationarity check ([e016b66](https://github.com/CoreySpohn/photomancy/commit/e016b6650c3721a49eacea03459991aafba14404))
* **infer:** whitened residual on ImageTerm, the Gaussian likelihood and ForwardTerm, and Fit.residuals per term ([783a60b](https://github.com/CoreySpohn/photomancy/commit/783a60b4000032906239951e4a3f7a89d047f576))
* **infer:** with_priors re-expresses draws across a change of support ([fecd5e8](https://github.com/CoreySpohn/photomancy/commit/fecd5e89e1d04be96e4ffd3354d2d76275d139e4))
* **infer:** with_priors, append_term, replace_term and update return the gate reading as ReweightInfo(n_eff, khat, refused) ([a75bfe7](https://github.com/CoreySpohn/photomancy/commit/a75bfe7307a979d75a68150464702c8ca2a89b16))
* make blackjax and jaxns optional extras imported when a backend runs ([a163942](https://github.com/CoreySpohn/photomancy/commit/a1639423e307c826ac4540d74f04e8841bbbf8a3))
* **measurements:** AstrometryData, RadialVelocityData and ProperMotionAnomalyData name the data classes ([c1ee4ce](https://github.com/CoreySpohn/photomancy/commit/c1ee4ceb61e1db30291bb6bd5b1bc5243b9d525c))
* **measurements:** delete the jitter_m_per_s and area_AU2 call keywords of the RV and dMag terms ([fdd529e](https://github.com/CoreySpohn/photomancy/commit/fdd529ec9ab3b4cffc89fae819475593598fd1fd))
* **measurements:** DMagData and loglike_dmag replace ImagingData, NullData and the null kernel; the dMag area is a model leaf ([349cfaa](https://github.com/CoreySpohn/photomancy/commit/349cfaa1d279790c8cbf831a4f4ef4e404cd1d2a))
* **measurements:** integrate report predictions over the source tables and declare knot reads ([75d11af](https://github.com/CoreySpohn/photomancy/commit/75d11af4a0b8a16b40d1ed49f97ab59c275c6115))
* **measurements:** integrate tabulated spectra in photometry terms ([2afc8fe](https://github.com/CoreySpohn/photomancy/commit/2afc8fe3f8ede945a48db4de069517374206e7e5))
* **measurements:** linear_form on the two astrometry terms ([fde9d33](https://github.com/CoreySpohn/photomancy/commit/fde9d33c6c35159e8c7419ace3d41c99f46297a2))
* **measurements:** name RV instruments, share one offset latent and merge epochs in update ([eec83b4](https://github.com/CoreySpohn/photomancy/commit/eec83b48a4b0bb5a0aa7bf5315c2a554650f744a))
* **measurements:** remove the knot reads of the photometry and report terms ([4fa0882](https://github.com/CoreySpohn/photomancy/commit/4fa0882133e76f44b1a90b94309e29c670f61054))
* **measurements:** ReportData, ReportRule, CountRateTable, PrecomputedReportPredictor and log_ratio ([67b8a93](https://github.com/CoreySpohn/photomancy/commit/67b8a932777e393316f8b438845ea7365a06d208))
* **measurements:** ReportTerm scores counts only and ReportRows.astrometry hands detection positions to the astrometry term ([1cf1d9e](https://github.com/CoreySpohn/photomancy/commit/1cf1d9eceac60a381293ef67a6fd3128fa8687a3))
* **measurements:** ReportTerm sums the two orbit sheets and proposes its photometric leaf ([1ba5d5d](https://github.com/CoreySpohn/photomancy/commit/1ba5d5d944acbf3c853ccbd64410c10ebf2b08aa))
* **measurements:** the data classes name their time field time_d (pad, from_sep_pa, from_dmag, from_contrast_curves, det_time_d, null_time_d) and MAX_* leave the package exports ([430b97d](https://github.com/CoreySpohn/photomancy/commit/430b97d52cbc3e163b1aedd6a0c3a92834936d6b))
* move the generic inference machinery into photomancy.infer ([40098e2](https://github.com/CoreySpohn/photomancy/commit/40098e2c4333bcb2f0cc1c3e9f2c894ed3973ae9))
* **orbit:** any_epoch, few_epoch(n_draws), ShapePriorProposal, n_pool and classical_elements; orbit exports its factories ([98128e3](https://github.com/CoreySpohn/photomancy/commit/98128e38d39b8fde35ba7f9deb89c0bb06f6760e))
* **orbit:** build and read frame orbits through the orbix FrameOrbit methods ([dc7870c](https://github.com/CoreySpohn/photomancy/commit/dc7870c80803c0068d8931fe2756ff4b6a60a642))
* **orbit:** classical_elements returns the argument of periapsis w_rad ([a2a6b7c](https://github.com/CoreySpohn/photomancy/commit/a2a6b7c65b056afd2343bd709e343debbaf9e10a))
* **orbit:** completed proposal, scale-rotate proposal and the few_epoch and collapsed engines ([2414455](https://github.com/CoreySpohn/photomancy/commit/24144554036b69c5fe356e6469fc2aaf11b364fb))
* **orbit:** completion screens a pool of constants by the prior and scores a few with their proposed leaves ([c8a24d4](https://github.com/CoreySpohn/photomancy/commit/c8a24d4f3baaebabf5469b13b25ffbb999a24a88))
* **orbit:** delete migrate_sky_convention and scan the package for deleted names ([4af2030](https://github.com/CoreySpohn/photomancy/commit/4af2030abf05d3dc27ec1be0df1707dc2e73698f))
* **orbit:** delete OrbitUpdater with its kernel, belief, fixed-mass and conjugate targets, tests, tools and protocol docs ([4013b0c](https://github.com/CoreySpohn/photomancy/commit/4013b0ce2cb36d235832c3811b1a44d35ed62b80))
* **orbit:** delete the legacy orbit engines, chart problems and root priors ([09477e2](https://github.com/CoreySpohn/photomancy/commit/09477e2aaea8bf107678ba4456c4ea3b7341e375))
* **orbit:** move system_density into photomancy.orbit and drop its root exports ([75c7803](https://github.com/CoreySpohn/photomancy/commit/75c7803e8498c3ee023361fa6b8016b508227d75))
* **orbit:** orbits_from_samples returns a FrameOrbit and takes the reference day ([6d2bc9d](https://github.com/CoreySpohn/photomancy/commit/6d2bc9d6f31af23488ded60407ca992f0a0a9b71))
* **orbit:** refit any_epoch kernels at twice the Silverman bandwidth in every round ([10095c7](https://github.com/CoreySpohn/photomancy/commit/10095c7bc61017d66f91751631fb7935370e0f65))
* **orbit:** score candidate epochs from a Posterior's clustered modes ([c8a37f7](https://github.com/CoreySpohn/photomancy/commit/c8a37f77a092f2db42b4cada573cfeeadc8302d6))
* **orbit:** SheetSum owns the sheet sum, ReportTerm loses orbit_path, Fit refuses two sums over one latent ([a06de52](https://github.com/CoreySpohn/photomancy/commit/a06de5287df964720d2eabe681cb32819e42983c))
* **postproc:** frame_time_s, valid, *Arm, east/north_arcsec and separation_arcsec names ([89d5e3b](https://github.com/CoreySpohn/photomancy/commit/89d5e3b5f0387214664032416811eadac4ead7fb))
* **postproc:** make the arm options keyword-only ([933081f](https://github.com/CoreySpohn/photomancy/commit/933081f64370e533ef33e9bd0846493c090ae494))
* **postproc:** move post-processing in from coronalyze and drop the imaging extra ([2bb75df](https://github.com/CoreySpohn/photomancy/commit/2bb75df9cf4ab064c4d312d4132c19631a3b1491))
* **postproc:** propagate validity through filter footprints, require one orientation, read the resolution element from the frames ([aecf8b9](https://github.com/CoreySpohn/photomancy/commit/aecf8b917eb2b6b2363aa61202505bfa1423563d))
* **postproc:** read Coronagraph.coronagraph_at in the yippy template ([8653b88](https://github.com/CoreySpohn/photomancy/commit/8653b880c43eb17d98a1dcef69f92d3d700566c5))
* run MAP, Laplace, sampling, SMC and nested backends on the chart density ([c969f08](https://github.com/CoreySpohn/photomancy/commit/c969f086afe53fd099410f2531ffa3bbd91fd6d4))
* score every observable channel as an explicit term of the chart density ([130c291](https://github.com/CoreySpohn/photomancy/commit/130c291b5c27709be31fb90f478a1a9ade597562))
* stamp report tables with their epoch and refuse a mismatched model epoch ([b06f614](https://github.com/CoreySpohn/photomancy/commit/b06f61494d8dcfee8def0077b2c9e3884ee4a750))
* take generic priors from the priorities library and guard the photomancy.infer import graph ([aa13648](https://github.com/CoreySpohn/photomancy/commit/aa136482fed824e05474c984e6ba44c5c3282a17))
* **viz:** corner plot from a Posterior through posterior_physical ([b076c2f](https://github.com/CoreySpohn/photomancy/commit/b076c2f9c872a13abc6ff567413c3182e3dd36ea))


### Bug Fixes

* **detection:** cap the float32 resolution tolerance at one eps of the objective ([d35d0f5](https://github.com/CoreySpohn/photomancy/commit/d35d0f52032f49766f1a4a2e579df7e79741d4fb))
* **detection:** resolve float32 profiles at the objective rounding floor; refuse a bare RV instrument string ([3987fd5](https://github.com/CoreySpohn/photomancy/commit/3987fd5419c41dbbcbf6545d6f54bd037fb841cd))
* **detection:** safeguarded amplitude profiler with a resolved flag ([eadf5d0](https://github.com/CoreySpohn/photomancy/commit/eadf5d0a256690c32ca99ecb6e5f05395a663978))
* **detection:** treat a non-finite render as a linearity departure and re-profile with one program shape ([d011209](https://github.com/CoreySpohn/photomancy/commit/d0112092deaf884aae8e07fdaa4b23474d8b3414))
* fit Thiele-Innes constants with the orbix masked solve, treat separations beyond the contrast curve as undetectable in orbit EIG, and name the atmosphere forward distance dist_AU ([484ef90](https://github.com/CoreySpohn/photomancy/commit/484ef905deb3da0537ec7de33918fe1e4abd4960))
* **image:** coerce NumPy and JAX booleans for static_beyond_speckle and test the mask=None scored set ([27604b8](https://github.com/CoreySpohn/photomancy/commit/27604b8187009712f6acbf2d8b84928902282330))
* **image:** excluded pixels carry no NaN gradient into a finite score ([b856882](https://github.com/CoreySpohn/photomancy/commit/b85688241d298604fee1f9c34fbaaf3a50c8688f))
* **image:** mask=None scores the supported pixels, and an observed Image must be electrons on the detector grid at the term's roll ([6e5268e](https://github.com/CoreySpohn/photomancy/commit/6e5268e7c121988201aa9a6b61fabdf7b03ae6fb))
* **image:** the residual whitens the floored rates log_prob scores and is safe to differentiate ([34888a9](https://github.com/CoreySpohn/photomancy/commit/34888a941627cdcd8760b911ba38979a95dec11b))
* **infer:** a NaN target refuses the slice, importance_evidence lives in evidence, init is typed, batching tested ([9d5c05d](https://github.com/CoreySpohn/photomancy/commit/9d5c05d27d1cc0a2ecfc192774863a947023d346))
* **infer:** a reweight keeps its evidence error and every operation takes one slice ([8326f8e](https://github.com/CoreySpohn/photomancy/commit/8326f8e9486d3f41d3ceb1158d3e51252bb1cc57))
* **infer:** adaptive importance refuses an unfinished ladder, rungs by Kish size, proposals by capability, cdf pick, laplace floor 0.03 ([a9408d6](https://github.com/CoreySpohn/photomancy/commit/a9408d6357ff23f24d90deec809bebb8efcbc5df))
* **infer:** add the Haar constant to the legacy chart densities now that orbix volumes are normalized ([ad0ed1c](https://github.com/CoreySpohn/photomancy/commit/ad0ed1c1d70f7ce3642c7f929bf4f1040d2fde86))
* **infer:** cap k-hat by the largest observed exceedance so bounded weights are not refused ([968b5cd](https://github.com/CoreySpohn/photomancy/commit/968b5cd33b436f762e435e0c3c605eddce12bf9c))
* **infer:** categorical_indices takes the key first ([48bbbb4](https://github.com/CoreySpohn/photomancy/commit/48bbbb4cb364c4b3107d0bbcd84b737b3af1ce28))
* **infer:** cluster_to_mixture skips dead draws and empty clusters ([e695b9c](https://github.com/CoreySpohn/photomancy/commit/e695b9c0a6a304c1df97f7aea4559242c59b2920))
* **infer:** compare leaf rank per entry when stacking explicit starts ([6b52e3a](https://github.com/CoreySpohn/photomancy/commit/6b52e3a7b38d0905119de611347ae308e2faa7b6))
* **infer:** dead draws found by isfinite, gate counts NaN at live draws, same_layout replaces support_change ([c1d7ec3](https://github.com/CoreySpohn/photomancy/commit/c1d7ec3bfcb28a8867c71526762fd8a003781163))
* **infer:** EIG estimators take a term and the nested form excludes the outer draw ([a6926a4](https://github.com/CoreySpohn/photomancy/commit/a6926a4f7df135ebd090af8948010a5b575d3442))
* **infer:** evaluate the loss at the point map_fit returns and flag a regressed or failed stage ([bf42f5d](https://github.com/CoreySpohn/photomancy/commit/bf42f5d37ae2462bc41183c7935cbdec00733a1d))
* **infer:** Fit evaluates block walls once on the host and keeps them static so a stacked or jitted wall-free fit is not refused ([2166de9](https://github.com/CoreySpohn/photomancy/commit/2166de9131511a581ee4406b4e46ed361fb006d4))
* **infer:** Fit keeps Python int and bool leaves, block writes are converted, Laplace gate tested ([b1f3b8a](https://github.com/CoreySpohn/photomancy/commit/b1f3b8a7b0dbfd6a3b6450f9a8aa389a1b151748))
* **infer:** Fit names a static field as structure that cannot be free instead of pointing at a block ([2944842](https://github.com/CoreySpohn/photomancy/commit/2944842ddfa38418a874f00e4b3f0d8625799e55))
* **infer:** Fit passes block reads to check_ties so a written mass needs an orbit block that reads it ([552d8c0](https://github.com/CoreySpohn/photomancy/commit/552d8c05f3afa6df09e58219ff30e990a6fe81b4))
* **infer:** Fit.walls reads the blocks when their laws are concrete and takes stored walls as an init argument, so tree_at edits and dataclasses.replace stay correct ([fa9a904](https://github.com/CoreySpohn/photomancy/commit/fa9a90453ffda393ca29afefd10762aef761110f))
* **infer:** ForwardTerm reports its forward's reads so a fit refuses freeing a path the held build cached ([a88dc41](https://github.com/CoreySpohn/photomancy/commit/a88dc418f82f95aa5ac876504a4262486c7bda7d))
* **infer:** keep int subclasses static in Fit, convert block writes in the prior swap, isolate the Laplace gate test ([999738c](https://github.com/CoreySpohn/photomancy/commit/999738c53dbc2ac72913461413fa18e60d6f37e4))
* **infer:** MixtureProposal.from_cloud reads no dead draw ([4d011fc](https://github.com/CoreySpohn/photomancy/commit/4d011fcc95a94f2314b7543183ac4c185df4e295))
* **infer:** refuse a prior swap whose containment cannot be shown: mixtures by their weighted components, blocks by their non-law fields ([085a34e](https://github.com/CoreySpohn/photomancy/commit/085a34e45773e255d993fb45074d03f6c26e3680))
* **infer:** refuse a widened prior support, evaluate dead draws live, mark resampled grids monte_carlo, draw sheets from the scored density ([26f25d1](https://github.com/CoreySpohn/photomancy/commit/26f25d1240691ec4314775427a750fa55dd5932b))
* **infer:** refuse stage fits with another coordinate map and check every draw against the schema ([df72481](https://github.com/CoreySpohn/photomancy/commit/df72481efb79ae435eec82215ffce36322903211))
* **infer:** render each exposure once when a term gives its value and prediction together ([6a8f82d](https://github.com/CoreySpohn/photomancy/commit/6a8f82df62d2602e6c72d7d4198ce37e2d90294e))
* **infer:** report convergence of measure by unit-free fields, backtrack the Newton polish, end a stage on a non-finite loss ([4f40833](https://github.com/CoreySpohn/photomancy/commit/4f4083347f5181e7296fe631953c7b7a4e302890))
* **infer:** report the gate's real decision as refused on a quadrature slice ([2ba4ad9](https://github.com/CoreySpohn/photomancy/commit/2ba4ad9b9b7f181a6a919cd6d06b92fba9b4737d))
* **infer:** return NaN evidence from a nested sampling run cut by a hard stop ([f1775ee](https://github.com/CoreySpohn/photomancy/commit/f1775eebdcd11de7689271ca995eadbf3b507149))
* **infer:** return the lowest-value L-BFGS iterate from Laplace ([5cfac9b](https://github.com/CoreySpohn/photomancy/commit/5cfac9b258c75cb949d5c7fe42f173d9c661e08f))
* **infer:** sort the prior subkeys, walk once per density, refuse non-terms, broadcast the block volume ([32b6ab9](https://github.com/CoreySpohn/photomancy/commit/32b6ab9ef55304e9772096aa7e2e6396d86d5a98))
* **infer:** stop the default MAP stage once the objective converges ([102a62a](https://github.com/CoreySpohn/photomancy/commit/102a62a0df45203e04c3b0663bb5699f407aba24))
* **infer:** stop the default MAP stage only on a converged scaled gradient and check detect's linearity ([08ce9ae](https://github.com/CoreySpohn/photomancy/commit/08ce9ae80c752ca2cc51b95405af6c03bdba4c11))
* **infer:** value_and_prediction leaves AbstractTerm and stays a private legacy method ([f0fb352](https://github.com/CoreySpohn/photomancy/commit/f0fb35294aa3eabfcc8e6f20d1dfdfd8a9159375))
* **measurements:** apply the scene planet flux_scale in system_separation_dmag ([311b667](https://github.com/CoreySpohn/photomancy/commit/311b6673f69ca1d03443b587b8c44df1a319c92c))
* **measurements:** DMagTerm refuses to draw a null epoch ([17876ba](https://github.com/CoreySpohn/photomancy/commit/17876ba1ebb098c7aa08484c50feefa54419f4f4))
* **measurements:** host-side selector checks, epoch-based planet count, anchored RV spread, SheetSum path and axis checks ([e17109e](https://github.com/CoreySpohn/photomancy/commit/e17109e30b502f8e0be84f53771ec3667450a974))
* **measurements:** name the Julian date dialect when converting target rows ([260122c](https://github.com/CoreySpohn/photomancy/commit/260122cf339feced508506d9569113c84c8002c8))
* **measurements:** normalize the flat-offset RV marginal with 0.5 sum_j log N_j and re-baseline its goldens ([a393af3](https://github.com/CoreySpohn/photomancy/commit/a393af33019eb9d2171b0194bb63031846cfa7e4))
* **measurements:** range-check band, report and sheet planets on concrete values, centre the RV spread, validate instrument names ([33073c7](https://github.com/CoreySpohn/photomancy/commit/33073c7ff9a84d681684e16ed338ebb6ea9ca627))
* **measurements:** refuse a held resolution that declares no knot reads ([9f6991d](https://github.com/CoreySpohn/photomancy/commit/9f6991d2a724f4d00a00f7eb8a15a311ec4bbe62))
* **measurements:** refuse planet selectors outside the system ([ee688d6](https://github.com/CoreySpohn/photomancy/commit/ee688d64d04f51ae26a6eb76657016a9cb6dbca4))
* **measurements:** relative astrometry linear form in arcsec, capability by type, mapped orbit block in tests ([af915f5](https://github.com/CoreySpohn/photomancy/commit/af915f563ead305d7ae3559927ec711691efae3b))
* **orbit:** a NaN score at a live draw reaches the completed weight and refuses the slice ([cf064ba](https://github.com/CoreySpohn/photomancy/commit/cf064ba78f92e7f179665f2003abd92e1e15661e))
* **orbit:** any_epoch(n_draws) sizes the returned cloud only; the adaptation rounds keep n_adapt_draws ([22babd1](https://github.com/CoreySpohn/photomancy/commit/22babd1e85491b5c0f665c956eed6e9a7adcd859))
* **orbit:** collapsed and few_epoch return a full last round, inverse-cdf pick in completion ([940f2d9](https://github.com/CoreySpohn/photomancy/commit/940f2d9ffd70f7834529bb33852726fa4289fcb8))
* **orbit:** read the pool condition as a plain mean over the live draws ([af53f5f](https://github.com/CoreySpohn/photomancy/commit/af53f5f62e42a560e7ed1d82aabe9b669cf86da2))
* **orbit:** scale-and-rotate returns out-of-bound draws as NaN dead draws and refuses a free leaf by name ([adbe35b](https://github.com/CoreySpohn/photomancy/commit/adbe35b42797f267f58467a917469438dc087da5))
* **orbit:** skip the SheetSum planet check when the model has no orbit rows ([2815dd3](https://github.com/CoreySpohn/photomancy/commit/2815dd3b65c5c8232534239afa874bec21b77ee2))
* **photomancy:** decide RV, roll, image-support, censoring and detection-bracket admission identically in both lanes, follow edited orbit walls, require float64 for image scoring ([038e1dc](https://github.com/CoreySpohn/photomancy/commit/038e1dc2afbde9479ec2375ac4699e99526fc33b))
* **photomancy:** read contrast curves and report rates with interp_linear for jax 0.12.1 ([3f95313](https://github.com/CoreySpohn/photomancy/commit/3f9531378c989d883a639af7ee20cac79263acfe))
* **postproc:** check template validity at the real patch nodes, build even Gaussian validity on the response grid, and use nonzero aperture footprints ([851686c](https://github.com/CoreySpohn/photomancy/commit/851686c82acab41292a1dcc9821c47119b296b63))
* **postproc:** compare the span of signed-wrapped position angles so float32 and frame-order spreads above 1e-6 deg are refused ([af5d9b4](https://github.com/CoreySpohn/photomancy/commit/af5d9b4f68942d52ae4b9781d9fe097c3f1f30c2))
* **postproc:** evaluate the orientation check of closed-over angles at compile time inside a jitted arm ([57a205d](https://github.com/CoreySpohn/photomancy/commit/57a205dd6ab64dc0dd1aa533c0d71f3a86c300c0))
* **postproc:** reduce each position angle to a signed remainder before taking offsets so a full turn cannot absorb a float32 spread ([b37bd16](https://github.com/CoreySpohn/photomancy/commit/b37bd16f7231164ba565fc188efd46bb8946d5fc))
* **precision:** route float64 inference conversions through one helper, refuse narrow posteriors with PrecisionError and enforce the precision guard ([4b2f693](https://github.com/CoreySpohn/photomancy/commit/4b2f693c6fed08bcb7b6fb8d683ecaf2abe6ff28))
* **sheets:** SheetSum forwards check and footprint_paths to the terms it holds ([dde9a8d](https://github.com/CoreySpohn/photomancy/commit/dde9a8d4f431de53fe74ca7c03b5f02d2051777c))
* **sheets:** SheetSum reports the assumptions of the terms it holds ([a21448c](https://github.com/CoreySpohn/photomancy/commit/a21448c41b7d0043325a82521710155b48ab2579))


### Performance Improvements

* **detection:** stop the Newton polish at the first refused step ([d2e50f1](https://github.com/CoreySpohn/photomancy/commit/d2e50f12d4bb4c230f1b4d6239e95013e325f65e))
* **infer:** a stalled line search stops a stage whose scaled gradient is small ([232768c](https://github.com/CoreySpohn/photomancy/commit/232768ccfcefd43ab3113b82ad5605830bcf6a8b))
* **infer:** build dense Hessians one Hessian-vector product at a time ([81c0e5a](https://github.com/CoreySpohn/photomancy/commit/81c0e5a1779323cff0dfb7530eb1aa3e7b8242b3))
* **infer:** end an L-BFGS stage when its zoom line search stalls ([b695da5](https://github.com/CoreySpohn/photomancy/commit/b695da5f614849d7782df3b6aca64b4cde6c2fb2))
* **orbit:** pick refit mixture components by inverse CDF in the completed proposal ([24c3288](https://github.com/CoreySpohn/photomancy/commit/24c32887d544e4f417f122e998da7accf2526f43))


### Documentation

* **postprocessing:** state background_variance_factor in the calculator example ([5c26dad](https://github.com/CoreySpohn/photomancy/commit/5c26dadb1d972fb46933be686b036b37b372f5b5))


### Miscellaneous Chores

* release 2.0.0 ([20604b0](https://github.com/CoreySpohn/photomancy/commit/20604b0b6e1e135b6a43bff13e7b32cdcb2cc96f))


### Code Refactoring

* drop the photomancy.infer re-exports from the package root ([f10cb71](https://github.com/CoreySpohn/photomancy/commit/f10cb7114b70e0df2091acb918102231a82045b4))
* evaluate band contrasts through one multi-band quadrature ([015ce44](https://github.com/CoreySpohn/photomancy/commit/015ce440d8d38b9c8daf061bf38b6e7e763df8db))
* **image:** read roll from the exposure pointing ([e895814](https://github.com/CoreySpohn/photomancy/commit/e89581491f04de62fc239608c2dd1dae2a4265bd))
* import grids and spectral types from optixstuff and declare hwoutils[jax] ([414222a](https://github.com/CoreySpohn/photomancy/commit/414222a31c78b2879fb06e9a39a63bbfb73e33c9))
* **infer:** rename the public surface and finish the staged MAP, EIG and naming passes ([b7d052c](https://github.com/CoreySpohn/photomancy/commit/b7d052cf4c4cf5fc96a9df78d94bc857e5a47602))
* **infer:** require module likelihoods in ForwardTerm and rename sample to draw ([da8ba6f](https://github.com/CoreySpohn/photomancy/commit/da8ba6fd2a1a744551d7d517c88f1fe638d2de32))
* **infer:** split the conditional mc_se into ratio_mc_se ([40cd2fe](https://github.com/CoreySpohn/photomancy/commit/40cd2fe7f4fba4cbc6468bee7ed208bb45ba1777))
* **measurements:** move data, kernels and terms into photomancy.measurements, select becomes system_path, DMagTerm replaces NullTerm ([4100e71](https://github.com/CoreySpohn/photomancy/commit/4100e7171674e30e24d77c3bdd4622569217dcd1))
* **measurements:** name Transmission tuples transmissions and pass transmission to band_contrasts ([5722c20](https://github.com/CoreySpohn/photomancy/commit/5722c20700b87f1e4e3a2bdc029c3ba99fa7465b))
* pass star_planet_distance_AU to skyscapes contrast_cube ([f85e796](https://github.com/CoreySpohn/photomancy/commit/f85e7967a784f9aa5b0f79618df7aafc50fe0ac5))
* photometry through skyscapes and geometry through orbix, one eccentricity prior and one proper-motion anomaly ([0fd31f4](https://github.com/CoreySpohn/photomancy/commit/0fd31f4c79778f1aa51875cc74f2d70bcd9fea44))
* **postproc:** rename MatchedFilterPostProc to MatchedFilterPostProcessing ([50fe5e5](https://github.com/CoreySpohn/photomancy/commit/50fe5e54dbb85edad16209edc2f3572af118f70e))
* **priors:** name sample_shape and q as the distribution API does, and document the coronagraphoto exposure calls ([ee4edfe](https://github.com/CoreySpohn/photomancy/commit/ee4edfe55543b6b64ffbf46f942a8741c9287b54))
* **priors:** take eccentricity laws from skyscapes and adopt the orbix element API ([9349481](https://github.com/CoreySpohn/photomancy/commit/934948101c71b1da84e159347b7fc141c033a0a2))
* remove photomancy.fields; use hwoutils random fields, conversions and eyepiece colours ([bcc205f](https://github.com/CoreySpohn/photomancy/commit/bcc205f1bc11b9c1523d7072d4d7dc61394a117d))
* rename ExposureModel.exposures to nuisance ([d7d9ebc](https://github.com/CoreySpohn/photomancy/commit/d7d9ebc83bf5308b331314ca00a788960422103c))
* rename ImagingTerm to DMagTerm and mark it legacy ([33013f3](https://github.com/CoreySpohn/photomancy/commit/33013f3d8795dfa5ecc5a925c461100b32eb58b6))
* share one reporting law between refit and report, and move particle weights and resampling into photomancy.particles ([15c2556](https://github.com/CoreySpohn/photomancy/commit/15c25564a33271cf36927349c7b6fa90f03839f0))
* take projected-orbit scale, gap and mirror completion from orbix ([84d12eb](https://github.com/CoreySpohn/photomancy/commit/84d12eb8b247704710e672d1a296c7e1ca339657))
* take the merged skyscapes Scene in disk fitting and tests ([5f4d3bf](https://github.com/CoreySpohn/photomancy/commit/5f4d3bf4aeefcc0437e4e4ad2f6d29d69a48a5c7))
* Thiele-Innes inversion from orbix, one contrast-curve interpolation, no deprecated eccentricity transform ([efc04aa](https://github.com/CoreySpohn/photomancy/commit/efc04aa308c3c69de29f82f8942d9e8858ed63ab))

## [1.0.0](https://github.com/CoreySpohn/photomancy/compare/v0.6.0...v1.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* every term samples its own data; a coronagraphic report term with an exact law and a tabulated surrogate
* system terms, disk term and target loader use days from the scene epoch

### Features

* a general term type, a sum of terms, and a relative-astrometry term scored on a system ([c6fe7b1](https://github.com/CoreySpohn/photomancy/commit/c6fe7b16698ac5aa683bbbae7c08aa5bbf14deb0))
* every term samples its own data; a coronagraphic report term with an exact law and a tabulated surrogate ([9a1f7ed](https://github.com/CoreySpohn/photomancy/commit/9a1f7edfb88a17aca66f7f0d35059af9692bf3c4))
* photometry, radial-velocity, stellar-astrometry and proper-motion-anomaly terms on a system, and a target-file loader ([f36656b](https://github.com/CoreySpohn/photomancy/commit/f36656bb8a728d474cc8d9e47bd58ac1b79578e1))
* system terms, disk term and target loader use days from the scene epoch ([e707cf6](https://github.com/CoreySpohn/photomancy/commit/e707cf6efeb6a0086abb069e75a97bf40bcb9597))


### Bug Fixes

* **orbit:** add the frame-orbit builder module the propagators import ([d606498](https://github.com/CoreySpohn/photomancy/commit/d60649806802eefd51c9ce48c8ba47ba567b9b4a))
* **terms:** annotate with declared packages only ([c7eb55f](https://github.com/CoreySpohn/photomancy/commit/c7eb55f4aebddde8f7b194e9f1671a772b4cac79))

## [0.6.0](https://github.com/CoreySpohn/photomancy/compare/v0.5.0...v0.6.0) (2026-11-15)


### Features

* **likelihoods:** Gaussian likelihood with a marginalized low-rank term and its residual basis ([6eb12a1](https://github.com/CoreySpohn/photomancy/commit/6eb12a1c0cfa00391744a09241ced71b53e58d32))
* **staged:** a stage can rebuild its log density from the current parameters ([683476c](https://github.com/CoreySpohn/photomancy/commit/683476c0917f8632ad250cdab589ebf63a782a91))


### Performance Improvements

* **photomancy:** skip unnecessary report-power bisection ([1a779e0](https://github.com/CoreySpohn/photomancy/commit/1a779e08e4cf0028d4b4aed568b2152f961e85c3))

## [0.5.0](https://github.com/CoreySpohn/photomancy/compare/v0.4.0...v0.5.0) (2026-11-10)


### Features

* **backends:** add group_labels and rescaled_transform for staged fits ([b8d129a](https://github.com/CoreySpohn/photomancy/commit/b8d129ae12e824531e6d50fe36c23c7f6203c5af))
* **backends:** add map_fit, a staged PyTree MAP fit with hold freezing ([e47be84](https://github.com/CoreySpohn/photomancy/commit/e47be848cca341a7334ab37f1b7f2156ef8d46e3))
* **backends:** add MAPBackend and PointPosterior ([3bc87d5](https://github.com/CoreySpohn/photomancy/commit/3bc87d5e071f5de5a087e367eaced15cf935186c))
* **backends:** add stage log-densities, closed forms, early stopping and callbacks to map_fit ([3b3a097](https://github.com/CoreySpohn/photomancy/commit/3b3a09791a91b0101705f9c96e73c6d5c0cd6b41))
* **backends:** add zero_gradient and drop_update freeze policies to map_fit ([e626b9f](https://github.com/CoreySpohn/photomancy/commit/e626b9f99eeeb1bf0b333991d066cc74da18d02d))
* **backends:** let map_fit callbacks fire on a step predicate ([c0f0f5c](https://github.com/CoreySpohn/photomancy/commit/c0f0f5cc3b634c2e10a45b18038aaed39b02db82))
* **core:** accept a likelihood builder and mask in build_gaussian_fit ([c6014c1](https://github.com/CoreySpohn/photomancy/commit/c6014c16f65d3af2b1e5d68fb0a498a1e42cc13f))
* **core:** add AbstractLogDensity with value_and_aux ([f0589b4](https://github.com/CoreySpohn/photomancy/commit/f0589b49bd122d76e3738e4634c506ce4606b866))
* **detection:** add profile_amplitude, likelihood_ratio_map and curvature_block ([d6b7566](https://github.com/CoreySpohn/photomancy/commit/d6b7566de5d35a48eb6165e49340c7dc9c3749a4))
* **exposure:** add ExposureLogDensity for multi-exposure fits with a toy self-fit ([6c9f3f9](https://github.com/CoreySpohn/photomancy/commit/6c9f3f9e69800f38d7df0f51d1501162a46ca4e9))
* **fields:** log density of a field map ([f45ed66](https://github.com/CoreySpohn/photomancy/commit/f45ed66b0dbfe2d3ae2ba746a11b95bc1c27dac0))
* **fields:** stationary Gaussian field in whitened coordinates ([bf550c2](https://github.com/CoreySpohn/photomancy/commit/bf550c2dd918a02855b75e49c5f85e0457578b7e))
* **fields:** whiten a field map ([9ab54d9](https://github.com/CoreySpohn/photomancy/commit/9ab54d94d5e179ee2065ee3a7c431f53801ea91c))
* **likelihoods:** masked normalized Gaussian, Laplace, Huber and Student-t likelihood builders ([71db6ed](https://github.com/CoreySpohn/photomancy/commit/71db6edc4900265d1b09a4216ffd684db5878abd))
* **orbit:** add a collapsed Metropolis-Hastings move for targets with a linear Gaussian block ([0bfc4b6](https://github.com/CoreySpohn/photomancy/commit/0bfc4b6907fdf1d4c0e1913912997c9498e7a734))
* **orbit:** add a fixed-capacity orbit updater with its belief, statuses and targets ([7823e53](https://github.com/CoreySpohn/photomancy/commit/7823e532e81facd6ee9a0f46b07d95bb247aa656))
* **orbit:** add per-target proposal scales and an exact collapsed period-alias move to the updater ([e843fa6](https://github.com/CoreySpohn/photomancy/commit/e843fa6bed85fd368a2b931382e4a40f325553a2))
* **orbit:** add resumable, sharded development and certification runners to the updater panel ([4cfb86f](https://github.com/CoreySpohn/photomancy/commit/4cfb86f8fa84a9e9d8177bd2b892d8ac4e1e3a4d))
* **orbit:** continuous photometric scale with report power ladder and per-center kernels ([ede1762](https://github.com/CoreySpohn/photomancy/commit/ede176228b21c07f6037c1ae8d03264431ad5b00))
* **orbit:** custom likelihood route with planet-scale sites in the model, inference and nested paths ([fc62ef7](https://github.com/CoreySpohn/photomancy/commit/fc62ef7031cb1a29883cd62b5db81a3ef8708f71))
* **orbit:** replace the placeholder updater kernel with tempered SMC and random-walk moves ([b16dbdb](https://github.com/CoreySpohn/photomancy/commit/b16dbdb76cb2cbdb77a0dda24bc85a7399e2ec33))
* **panel:** add the exact-check runner with quadrature refinement tables for both reference samplers ([c88b484](https://github.com/CoreySpohn/photomancy/commit/c88b48421225243ea85bd6d240c65317827e62d0))
* **panel:** add the kernel tuning driver, per-model kernel values and the measured-cost tally ([229c421](https://github.com/CoreySpohn/photomancy/commit/229c42196b6748b85cbab8ef5f976aac146660d0))
* **panel:** collapsed prior TI reference and an independent frequency-mixture sampler for check 3, with a bias-injection test ([bbe4bf4](https://github.com/CoreySpohn/photomancy/commit/bbe4bf468b0e6dbbca6c8648b6324172e2db2781))
* **panel:** grade a panel directory into the verdict table, with a completeness audit and a stage flag for status ([9f11b3b](https://github.com/CoreySpohn/photomancy/commit/9f11b3b2001bffd412d08c091d3b81707b42efc8))
* **panel:** stamp unit records with a run hash and flag stale records at the freeze ([1398e10](https://github.com/CoreySpohn/photomancy/commit/1398e104f52888f7c9699388c99f6677fb9acbb0))
* **photomancy:** windowed physical-prior importance refit with golden-reference test ([03cc3b2](https://github.com/CoreySpohn/photomancy/commit/03cc3b226233274f78f9caf2b37dce4d00788c55))
* **staged:** solver stages that optimize their free groups jointly with a line-search solver ([040a553](https://github.com/CoreySpohn/photomancy/commit/040a553c8ab0af56309028d9f29952dd2b78fbb9))


### Bug Fixes

* **orbit:** count the OFTI reference epoch once in the rejection score ([92f7b4a](https://github.com/CoreySpohn/photomancy/commit/92f7b4a8b8c685b75903a42d118e01062d2da45a))
* **orbit:** keep planet-scale sites on conversion, grade nested convergence by evidence, share model construction ([dc1d0de](https://github.com/CoreySpohn/photomancy/commit/dc1d0deaef830a2e2d91716f7ce29092a776bfd3))
* **orbit:** normalize updater weights in max-shifted form, report zero ESS without support and reject non-finite physical charts ([95c2ed5](https://github.com/CoreySpohn/photomancy/commit/95c2ed5b932e1e2e97367c063ee475c9f0d7b73f))
* **orbit:** validate ladder arguments and document the paired scale approximation ([c8fbda7](https://github.com/CoreySpohn/photomancy/commit/c8fbda749ee6bbfcff668a162a68bae954e71671))
* **panel:** enforce the proposal cap on effective budgets and test the floor for every TI reference role ([ec21b30](https://github.com/CoreySpohn/photomancy/commit/ec21b303a7211abe1ecc2ff8a60b81d734b72a7c))
* **panel:** enforce the TI ESS floor, independent exact checks with a binding window, TI budget 2^22 from a full preflight ([926a768](https://github.com/CoreySpohn/photomancy/commit/926a7684a06b6f0eb26e195b1e3f01a2c9e88031))
* **panel:** score empty reference fits as incomplete, resolve per-model kernel values in exact checks, write the manifest at start ([b330ca2](https://github.com/CoreySpohn/photomancy/commit/b330ca2b6fb9d6b6fc9fc2658c76fa00ea7eea9d))
* **panel:** unbiased replicate precision, source-fingerprinted run hashes, validated exact evidence, plan-complete freeze, refusing resume and atomic manifests ([d08e03b](https://github.com/CoreySpohn/photomancy/commit/d08e03b5d2c6bed2464c63b0bb31baac22a328a1))
* **photomancy:** draw several sets of refit constants per proposal to cut low-size refits ([7bc7d75](https://github.com/CoreySpohn/photomancy/commit/7bc7d75e5e9cb5ea4788b0fe26b789e753dce748))
* **photomancy:** guard refit round sizes and empty records, independent one-record reference ([935c6b7](https://github.com/CoreySpohn/photomancy/commit/935c6b74aa60fed1dabc241176d521f02d15648c))
* **staged:** rescaled_transform maps a line-search solver's grad and value_fn into the scaled variable ([c42efae](https://github.com/CoreySpohn/photomancy/commit/c42efae86d6d434cc8025cf8ea1dd9a5001f796e))
* **tools:** grade only bound evidence, mark missing or invalid evidence incomplete ([69f438b](https://github.com/CoreySpohn/photomancy/commit/69f438bef74302c6e4fa8e20681a42e72c2f8b13))
* **tools:** mark recompiles after a cache clear cold, clear on cgroup memory pressure, identity test on real physical units ([55fd30d](https://github.com/CoreySpohn/photomancy/commit/55fd30d6a2f11ccd029fe4639ce90d3ef470034f))
* **tools:** release JAX and Equinox compile caches between panel units above an RSS threshold ([545490d](https://github.com/CoreySpohn/photomancy/commit/545490df667e0ccca182b313085b3a47567f30b9))
* **tools:** require canonical freeze evidence, valid fit diagnostics and tracked evidence files; scope incompleteness per model ([c0803b8](https://github.com/CoreySpohn/photomancy/commit/c0803b8bde5d26d6973eb7afab56914ddbc94448))
* **updater:** resize physical reference budget to 2^24 and define exact-check convergence ([564422b](https://github.com/CoreySpohn/photomancy/commit/564422b5627735caaca10b4340c7e60607b31840))


### Performance Improvements

* **orbit:** build the Thiele-Innes design matrix from one Kepler solve ([257ff63](https://github.com/CoreySpohn/photomancy/commit/257ff63cf775b8a7453b4945c0e2c8f51cd46933))
* **orbit:** run the scale quadrature and cells only without a detection under a lane batch ([706fbef](https://github.com/CoreySpohn/photomancy/commit/706fbef9027e0f52245d5b8cdec423315aabfcf6))

## [0.4.0](https://github.com/CoreySpohn/photomancy/compare/v0.3.0...v0.4.0) (2026-11-05)


### Features

* **evidence:** convert log Bayes factors to detection significance ([24d7b10](https://github.com/CoreySpohn/photomancy/commit/24d7b1003c7bb907c4e75fc82f21a3ef2eb564a5))
* **orbit:** add migrate_sky_convention for saved fits, name the Thiele-Innes renames, fix the cos_i notebook call ([81279b3](https://github.com/CoreySpohn/photomancy/commit/81279b34f8266f5b9908e8cf196743140e72cb2d))


### Bug Fixes

* **orbit:** count marginalized offsets in the RV likelihood normalization ([f59e382](https://github.com/CoreySpohn/photomancy/commit/f59e382bcebe3c6bd6217e85c548fe96e66f0ce0))
* **orbit:** fit and predict astrometry in (east, north) with the standard phase angle ([2197b9f](https://github.com/CoreySpohn/photomancy/commit/2197b9f3ba0c9c1b280154bf5636ae0ef1b96984))
* **orbit:** keep the RV validity mask when re-padding ([e233b73](https://github.com/CoreySpohn/photomancy/commit/e233b73485660baa123bcd33606f99fa83af9153))
* **orbit:** mask invalid RV rows before forming residuals ([41f67ca](https://github.com/CoreySpohn/photomancy/commit/41f67caebcf2ff4b12a8ba10e925d63ba1493ba3))
* **orbit:** read orbix (east, north, away) rows directly, name the phase beta and the TIBasisModel coefficients by axis ([5df2b80](https://github.com/CoreySpohn/photomancy/commit/5df2b80f0ad3a536f41071ecf6f9aa71ee648880))
* **orbit:** size the cached RV jitter by the data's instrument count ([e6ecbb7](https://github.com/CoreySpohn/photomancy/commit/e6ecbb75d5b0952bcb7cf1bc687ac31398084ba9))
* **orbit:** treat RVData velocities as m/s throughout the RV likelihood ([3a6720b](https://github.com/CoreySpohn/photomancy/commit/3a6720b0813318b998d72530fa578d710b177c62))
* **orbit:** truncate eccentricity priors to [0, 1) ([667b664](https://github.com/CoreySpohn/photomancy/commit/667b66414aa2807bfd8eca197fd72ef09bf9c359))
* **orbit:** use the total mass consistently in the RV amplitude and Kepler's law ([9cf2063](https://github.com/CoreySpohn/photomancy/commit/9cf2063976fc01cb100c7d92636dea3c2215055a))
* **orbit:** use the total mass in EIG predictions, posterior diagnostics and the period inversion ([fa53ca6](https://github.com/CoreySpohn/photomancy/commit/fa53ca6f3437881833a8531076794f40371cba0c))
* **photomancy:** respect orbit precision and treat warnings as errors ([cce2655](https://github.com/CoreySpohn/photomancy/commit/cce2655026f6ae93be7f697d64e898ca1ee98d98))


### Documentation

* **examples:** only radial velocity breaks the (Omega + pi, omega + pi) ambiguity ([bce7246](https://github.com/CoreySpohn/photomancy/commit/bce7246f45a5bc9599f99e047da881f919d674cc))

## [0.3.0](https://github.com/CoreySpohn/photomancy/compare/v0.2.0...v0.3.0) (2026-10-14)


### ⚠ BREAKING CHANGES

* **deps:** JaxnsBackend drops gradient_guided and parameter_estimation, which jaxns 3 removed; build_scene_nested_model and JaxnsBackend.run now use the jaxns 3 Model (a prior model that realises its priors and returns the log-likelihood).

### Features

* **orbit:** add projected astrometric loss with analytic partials ([9d40832](https://github.com/CoreySpohn/photomancy/commit/9d408328c33ccde7edb59c61dbb1dd1b9ef7f639))


### Bug Fixes

* **deps:** port nested sampling to the jaxns 3 api ([6b0928d](https://github.com/CoreySpohn/photomancy/commit/6b0928d718972e2c4c3457d90f630286ca2e0395))
* **eig:** bound mode information with full predictive covariances ([9e880f2](https://github.com/CoreySpohn/photomancy/commit/9e880f222da6b232aa80933ee3dca983d4f00248))
* **eig:** return NaN instead of a wrong bound for predictives beyond working precision, and rank with nanargmax ([7529b3e](https://github.com/CoreySpohn/photomancy/commit/7529b3e13b1034cb7fe94c8ea9e2569593fb9175))
* **eig:** score reports by the chain rule so an always-null record carries no information ([59a9715](https://github.com/CoreySpohn/photomancy/commit/59a9715a002ecdddd19ec7f4a94c0a2afb5c51ad))


### Miscellaneous Chores

* release 0.3.0 ([8633a1e](https://github.com/CoreySpohn/photomancy/commit/8633a1e1ac84344e2818bbe5f0e473cd9e7206ef))

## [0.2.0](https://github.com/CoreySpohn/photomancy/compare/v0.1.0...v0.2.0) (2026-08-29)


### Features

* **orbit:** add orbits_from_samples and mode_scalars diagnostics ([3870b48](https://github.com/CoreySpohn/photomancy/commit/3870b48b8d38d51ab534be1dc784e0c92ec131d8))
* **viz:** add corner, EIG, and predictive panels on eyepiece ([22c491d](https://github.com/CoreySpohn/photomancy/commit/22c491d513ea2150f45d4b95fcbe36669ad5c2d9))


### Bug Fixes

* **backends:** adapt MCLMC to the BlackJAX 1.6 tuner and kernel API ([554b4de](https://github.com/CoreySpohn/photomancy/commit/554b4de012956b88ab7a480a4b752d5a6f700e81))
* **deps:** pin the orbix floor and carry viz into the test extra ([d089792](https://github.com/CoreySpohn/photomancy/commit/d0897921a5095ad99e2d63855e865b59b28f0222))
* **viz:** draw total eig as an envelope under its own components ([48304eb](https://github.com/CoreySpohn/photomancy/commit/48304ebe4e91050d163f9cbf1eb8d9caa4a9fd5e))
* **viz:** invert the detectability plane so fainter is downward in both panels ([30c5bac](https://github.com/CoreySpohn/photomancy/commit/30c5bacd5c2ceedf061febabc882248a04b959a6))
* **viz:** pin the eig legend and use bracketed units throughout ([29d42f9](https://github.com/CoreySpohn/photomancy/commit/29d42f91562735e5b166aaa27d988d1d7191db2c))

## [0.1.0](https://github.com/CoreySpohn/photomancy/compare/v0.0.1...v0.1.0) (2026-07-18)


### Features

* **atmosphere:** photomancy.atmosphere package (fixed-T abundance retrieval) ([451dfb4](https://github.com/CoreySpohn/photomancy/commit/451dfb4f693f51fbc41902080d36cb929d4e79f8))
* **backends:** add MCLMC and Pathfinder samplers ([3c2a94e](https://github.com/CoreySpohn/photomancy/commit/3c2a94e6f87a1e6114fb22296b6daf857514a5f6))
* **backends:** default NUTS to a dense mass matrix ([4d7fc5a](https://github.com/CoreySpohn/photomancy/commit/4d7fc5a5c08e9b1808a755c8a14098237a22edec))
* **backends:** filter_jit NUTS and SMC run so forward arrays trace, not bake ([93f6e8b](https://github.com/CoreySpohn/photomancy/commit/93f6e8bf5700eb73b250ab705431c134a961c363))
* **backends:** JaxnsBackend + build_scene_nested_model (nested sampling, evidence) ([61a6c1a](https://github.com/CoreySpohn/photomancy/commit/61a6c1a0708975492c7429ea4d811a63cf1974b5))
* **core:** build_gaussian_fit shared by disk + atmosphere scene fits ([a6f0e3c](https://github.com/CoreySpohn/photomancy/commit/a6f0e3c9903d3214fb42d9e5978bfa91173026f0))
* **core:** make logdensity a Module so forward arrays trace, not bake ([3f62bce](https://github.com/CoreySpohn/photomancy/commit/3f62bce135ccb1ec02ec7648fea98aee285433ae))
* **disk:** default_disk_prior + disk_fit_leaves (P2 per-domain prior registry) ([f7f0e2f](https://github.com/CoreySpohn/photomancy/commit/f7f0e2f4b418cf7c338fae08b46418793bf969c1))
* **disk:** fit a skyscapes System disk via the scene-as-PyTree engine ([b8a3a22](https://github.com/CoreySpohn/photomancy/commit/b8a3a22d68f26b73fb58d7cd228edc66b02c62a4))
* **eig:** EP probit null tier and exact detection-channel EIG ([fc72740](https://github.com/CoreySpohn/photomancy/commit/fc727408fbfc71fabb366ebed5c502d4e330945f))
* **eig:** evaluate_candidates_mixture so OFTI/grid_search drive orbit EIG ([4aef44b](https://github.com/CoreySpohn/photomancy/commit/4aef44b541fac9da23376a61c2a78b27f72f8e92))
* **eig:** exact detection MI, capped alias, QoI projection, class_eig ([054e7fd](https://github.com/CoreySpohn/photomancy/commit/054e7fd94fb4f562af6c2c6832b1990c4d39f92c))
* **eig:** generic EIG on the Posterior; orbit eig delegates ([aade648](https://github.com/CoreySpohn/photomancy/commit/aade648b7d8283212a4514dfb86714da8d4b9c8a))
* generic inference backends, posterior types, and orbit-path wiring ([c491677](https://github.com/CoreySpohn/photomancy/commit/c4916775cee71a1ec3a990a688ad8cc9a3e59f6d))
* **orbit:** absorb roberts_sequence from orbix (sole consumer) ([b379e4a](https://github.com/CoreySpohn/photomancy/commit/b379e4a669fceb0373d3fc614589d4b2cbdebb84))
* **orbit:** add Hipparcos-Gaia proper-motion-anomaly observable ([3d783c3](https://github.com/CoreySpohn/photomancy/commit/3d783c3f9ee0fe3bda841008102b7ded8b183b02))
* **orbit:** add sample_physical and mode_summary diagnostics ([1981518](https://github.com/CoreySpohn/photomancy/commit/19815189aeb48264c8bfaae42fa13e59f85d5a78))
* **orbit:** add stellar-reflex astrometry observable ([8294ead](https://github.com/CoreySpohn/photomancy/commit/8294ead4190f7e187e61bc5fe577d274a27a939b))
* **orbit:** exact few-epoch fitters (TIBasisModel, lambert_depth_fit, admissible_region_fit) ([af920ab](https://github.com/CoreySpohn/photomancy/commit/af920ab68adf8db42c1cf2d38232b95f8dfecadd))
* **orbit:** expose n_planets; add EIG, RV/joint, and multi-planet fitting tests ([c1766cd](https://github.com/CoreySpohn/photomancy/commit/c1766cd1f2a38926802865f326fc22b804a62514))
* **orbit:** OFTI/grid_search return SamplePosterior; drop ParticlePosterior ([722aeff](https://github.com/CoreySpohn/photomancy/commit/722aeff004305bc8f730017c8b754ffcc6ea533e))
* **orbit:** orbit_nested_sampling via NumPyro NestedSampler (evidence/model comparison) ([62c7e01](https://github.com/CoreySpohn/photomancy/commit/62c7e0177d6b4d3ace9a87d5e9097eb49353e208))
* **orbit:** OrbitProblem exposes unflatten/constrain; to_unconstrained + elements_to_sites ([1629f09](https://github.com/CoreySpohn/photomancy/commit/1629f09367632f6c2b4bb58591f3d84fe90e7fbf))
* **orbit:** rename evaluate_candidates_mixture to evaluate_orbit_candidates ([06113b1](https://github.com/CoreySpohn/photomancy/commit/06113b1ca00d121fc885215503ea0b06c156f7ee))
* **orbit:** return unified Posterior types; drop parallel result classes ([b02e3bf](https://github.com/CoreySpohn/photomancy/commit/b02e3bf459fd140d2011526e21d7ec79cd27544e))
* **posterior:** add general project_samples and sample_capped utilities ([5d294f5](https://github.com/CoreySpohn/photomancy/commit/5d294f59150f0d094451c901e84bf1e05791278a))
* **posterior:** add MixturePosterior.n_modes and best_mode ([4bac025](https://github.com/CoreySpohn/photomancy/commit/4bac025f2c593fd39c68006c90d4d1c8a1808af9))
* **posterior:** param_names + sample_dict + SIR on SamplePosterior; cluster_to_mixture ([c4b46ef](https://github.com/CoreySpohn/photomancy/commit/c4b46ef83924a10aa1a183534b4ced6e641bb329))
* **priors:** AbstractPrior + Uniform/Normal/LogNormal/IndependentPrior/JointPrior ([05162ff](https://github.com/CoreySpohn/photomancy/commit/05162ff4cc1f0f2688fd7a6a6352bbce3b82dfad))
* **priors:** MixturePrior + MixturePosterior/SamplePosterior.to_prior (P2 sample-&gt;prior) ([19398ec](https://github.com/CoreySpohn/photomancy/commit/19398ec288933b31c8f35c6e970df0088b99a4ae))
* **priors:** wire AbstractPrior into scene fits, jaxns adapter, and to_prior ([0cb6537](https://github.com/CoreySpohn/photomancy/commit/0cb6537810ec2159f1fff126c37e8e32ed6794cb))


### Bug Fixes

* **orbit:** make null/imaging detection fitting work (gradient-safe contrast-curve interp + pad) ([e5dc2d5](https://github.com/CoreySpohn/photomancy/commit/e5dc2d5509d0218f07eb22875e55bfec13a9dfd1))
* **posterior:** make to_prior a uniform exact-conversion contract ([2c9b019](https://github.com/CoreySpohn/photomancy/commit/2c9b019aded5356be91a958aa180f783790f5f64))

## 0.0.1 (2026-06-18)


### Features

* add Backend/Posterior protocol and Laplace backend ([c18e80c](https://github.com/CoreySpohn/photomancy/commit/c18e80c1104e447dbcd2dcdd61346959db5ec662))
* **core:** add scene-PyTree logdensity assembly ([77f80ed](https://github.com/CoreySpohn/photomancy/commit/77f80edea1cd82f5dbef604dfa92b5ae37759723))
* scaffold photomancy and port orbit fitting from orbix ([d3fe243](https://github.com/CoreySpohn/photomancy/commit/d3fe2436403568c8928788ef43f9a14fe7fae130))


### Miscellaneous Chores

* release 0.0.1 ([25be93d](https://github.com/CoreySpohn/photomancy/commit/25be93d29830aab5f51b6482c1c6950acd2eee01))

## Changelog
