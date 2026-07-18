# Changelog

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
