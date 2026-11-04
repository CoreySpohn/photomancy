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
| optax | >=0.2.3 | 0.2.3 | runtime |
| orbix | >=3.0.0 | 2.0.1.dev17+g2f9f366a6 | runtime |
| packaging | >=24.2 | 24.2 | test |
| pre-commit | >=3.6 | 3.6.0 | dev |
| pytest | >=7.3.2 | 7.3.2 | test |
| pytest-cov | >=1.0 | 1.0 | test |
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

## Test results

All 42 test files ran separately in both environments, including slow tests.
These are summed pytest times for the final attempt per file; queue waits and
superseded attempts are excluded.

| Environment | Passed | Failed | Pytest time |
| --- | ---: | ---: | ---: |
| Shared | 335 | 0 | 459.50 s |
| Lowest direct | 333 | 2 | 517.75 s |

## Open numerical failures

### Proper-motion mass recovery

The lowest-direct test run fails
`tests/test_pm_anomaly.py::test_joint_relative_astrom_and_pma_recovers_mass`.
The mass stays at the initial `log10(Mp / Mearth) = 2.8021538928713605`,
compared with truth `2.5021538928713607`; the unchanged assertion requires
an absolute error below `0.1` and observes `0.3`.

Reproducing input: stellar mass `1.989e30` kg, distance `10` pc,
`a=5.2` AU, `e=0.15`, `cos_i=0.4`, `W=0.8`, `omega=1.0`, `M0=0.6`,
and planet mass `317.8` Earth masses. There are 12 astrometric epochs from
`0` to `24.75 * 365.25` days, with `0.001` arcsec errors; the proper-motion
window is `2.8 * 365.25` days with 25 samples and errors of
`0.02e-3 / 365.25` arcsec/day. Noise keys are `0`, `1`, and `2` for east,
north, and proper motion. Only the initial log mass is displaced from truth,
by `+0.3`. The fit uses 600 L-BFGS steps and `min_eigenvalue=1e-12`.

This environment resolves JAX/JAXlib `0.9.0`, Optax `0.2.3`, Equinox `0.13.2`,
and the pinned sibling implementations in the table. The shared environment
passes the same test with JAX `0.12.0.1` and Optax `0.2.8`; the responsible
dependency or numerical difference has not been isolated. No assertion,
initial value, tolerance, or dependency bound was changed to bypass this failure.

### Joint RV and astrometry period recovery

`tests/test_rv_fitting.py::test_joint_rv_astrometry_recovers_period_and_mass`
returns `T=499.99999999999994` days for truth `800` days. The unchanged
assertion requires relative error below `0.10` and observes `0.375`.

Reproducing input: stellar mass `hwoutils.constants.Msun2kg`, distance `10` pc,
`e=0.1`, `cos_i=0.5`, `W=2.0`, `omega=0.5`,
`tp=-1.2 / (2*pi / 800)` days, and planet mass `300` Earth masses.
The RV fixture uses NumPy seed `3`, 15 epochs uniformly drawn over `0..2000`
days, and noise equal to 5% of the clean RV standard deviation. The astrometry
fixture uses NumPy seed `4`, six epochs over the same interval, and `0.003`
arcsec noise. The period prior spans `500..1200` days. Five Thiele-Innes seeds
feed `LaplaceMixtureBackend(min_eigenvalue=1.0, n_steps=300)`.
The shared environment passes this test. The period and subsequent mass
assertions, data, initializers, and dependency bounds are retained.
