# Dependency floors

Floors were checked with Python 3.12 and a scratch environment using
`uv pip install --resolution lowest-direct -e ".[test,docs,dev,imaging]"` with
`--override .github/sibling-pins.txt`. Build dependencies were also constrained to
the versions below for an editable build. Each test file runs in its own pytest
process with warnings treated as errors.

The sibling overrides select the exact CI commits, even when their development
version metadata precedes a declared floor. Those rows validate the pinned
implementation rather than the corresponding released package. Transitive
requirements can also raise a resolved version above its direct floor. The test
extra lists its plotting dependencies directly so it does not impose a recursive
self-version constraint on shallow VCS checkouts.

| Dependency | Declared floor | Resolved version | Groups |
| --- | --- | --- | --- |
| blackjax | >=1.6 | 1.6 | runtime |
| coronalyze | >=1.2.0 | 1.1.2.dev52+g541a25c86 | imaging |
| equinox | >=0.13.2 | 0.13.2 | runtime |
| eyepiece | >=0.1.0 | 0.16.1.dev25+gc7a4bea5a | docs, test, viz |
| hatch-fancy-pypi-readme | >=24.1.0 | 24.1.0 | build |
| hatch-vcs | >=0.4.0 | 0.4.0 | build |
| hatchling | >=1.26.3 | 1.26.3 | build |
| hwostyle | >=1.7.0 | 1.7.1.dev6+g8f7acbaea | docs |
| hwoutils | >=1.10.0 | 1.10.1.dev16+g683aa5221 | runtime |
| hypothesis | >=6.79.1 | 6.79.1 | test |
| ineedvalidation | >=1.0.0 | not run at the floor | test |
| ipython | >=8.18 | 8.18.0 | docs |
| ipywidgets | >=8.1.1 | 8.1.1 | docs |
| jax | >=0.8.1 | 0.9.0 | runtime |
| jaxctx | >=1.2.0 | 1.2.0 | runtime |
| jaxlib | >=0.8.1 | 0.9.0 | runtime |
| jaxns | >=3.0.0 | 3.0.0 | runtime |
| matplotlib | >=3.11 | 3.11.0 | docs, test, viz |
| myst-nb | >=1.1 | 1.1.0 | docs |
| numpy | >=2.1.2 | 2.1.2 | runtime |
| numpyro | >=0.21.0 | 0.21.0 | runtime |
| optax | >=0.2.4 | 0.2.4 | runtime |
| orbix | >=3.0.0 | 2.0.1.dev17+g2f9f366a6 | runtime |
| packaging | >=24.2 | 24.2 | test |
| pre-commit | >=3.6 | 3.6.0 | dev |
| pytest | >=7.3.2 | 7.3.2 | test |
| pytest-fail-slow | >=0.1.0 | 0.1.0 | test |
| pytest-markdown-docs | >=0.5.0 | 0.5.0 | test |
| pytest-testmon | >=2.0.7 | 2.0.7 | test |
| pytest-timeout | >=1.3.1 | 1.3.1 | test |
| pytest-xdist | >=2.5 | 2.5.0 | test |
| scipy | >=1.13 | 1.13.0 | test |
| skyscapes | >=2.1.0 | 2.0.1.dev44+g5280469c2 | runtime |
| sphinx | >=7.3.7 | 7.3.7 | docs |
| sphinx-autoapi | >=3.0 | 3.0.0 | docs |
| sphinx-book-theme | >=1.1.3 | 1.1.3 | docs |
| sphinx_autodoc_typehints | >=2.0 | 2.0.0 | docs |
| sphinxcontrib-mermaid | >=0.9.2 | 0.9.2 | docs |
| tfp-nightly | >=0.26.0.dev20260621 | 0.26.0.dev20260621 | test |

NumPyro 0.15 cannot import the resolved JAX because `host_callback` was removed.
NumPyro 0.19 imports the deprecated `jax.interpreters.pxla.xla_pmap_p`; 0.21
avoids both problems. NumPy, Matplotlib, and the markdown pytest plugin floors
match the versions selected by the complete lowest-direct resolution.

## Optimizer floor bisect

Optax >=0.2.4: fixes both stalled fits; relevant upstream changes revise L-BFGS first-step scaling and line-search initialization ([upstream change](https://github.com/google-deepmind/optax/pull/1133)).

Both failures reproduced in a scratch Python 3.12 environment with the complete
lowest-direct resolution and the sibling overrides above. The RV test returned
`T=499.99999999999994` days for truth `800`; the PMa test returned
`log_Mp=2.8021538928713605` for truth `2.5021538928713607`.
Only Optax changed in each trial; installed-package snapshots confirmed that all
other versions and sibling commits stayed identical to the failing baseline.

| Optax version | Joint RV + astrometry node | Joint astrometry + PMa node |
| --- | --- | --- |
| 0.2.3 (original floor) | Failed, 35.00 s | Failed, 18.31 s |
| 0.2.8 (shared version) | Passed, 27.16 s | Passed, 11.04 s |
| 0.2.4 (new floor) | Passed, 24.97 s | Passed, 10.95 s |

The node IDs are
`tests/test_rv_fitting.py::test_joint_rv_astrometry_recovers_period_and_mass`
and
`tests/test_pm_anomaly.py::test_joint_relative_astrom_and_pma_recovers_mass`.
Optax 0.2.4 immediately follows 0.2.3, so no intervening release needs checking.
Optax alone fixes both failures; no other dependency floor needs to rise.
Tests, initial values, tolerances, and library code are unchanged.

## Test results

The previous shared-environment full-suite check passed all 335 tests across
42 files in 459.50 s. After raising the Optax floor, both affected files are
checked separately in a fresh lowest-direct environment using the same build
constraints and sibling overrides. Timings exclude compute queue waits.

| Lowest-direct test file | Passed | Failed | Pytest time |
| --- | ---: | ---: | ---: |
| `tests/test_rv_fitting.py` | 2 | 0 | 31.90 s |
| `tests/test_pm_anomaly.py` | 5 | 0 | 11.23 s |
| Total | 7 | 0 | 43.13 s |

The fresh resolution differs from the original baseline only in Optax
(`0.2.3` to `0.2.4`). Warnings remain errors. The test-budget plugin reports
that both RV tests exceed its 5 s unit budget; its configured report mode does
not fail the run.
