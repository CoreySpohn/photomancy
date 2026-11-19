"""Pytest configuration for the photomancy test suite.

Share fitted scenarios and reuse compiled primitives across pairs of modules.
Clearing after each pair bounds memory when CI uses two workers. JAX's
persistent cache still accelerates repeat runs; its directory follows the CI
environment and defaults to a gitignored local directory.
"""

import gc
import importlib.util
import os
from pathlib import Path

import jax
import pytest
from hypothesis import settings

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

jax.config.update("jax_enable_x64", True)

_CACHE_DIR = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        Path(__file__).resolve().parent.parent / ".jax_cache",
    )
)
jax.config.update("jax_compilation_cache_dir", str(_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

_MODULES_SINCE_CLEAR = 0


@pytest.fixture(autouse=True, scope="module")
def release_compiled_programs():
    """Bound cache growth while letting adjacent files share small JAX programs."""
    global _MODULES_SINCE_CLEAR
    yield
    _MODULES_SINCE_CLEAR += 1
    if _MODULES_SINCE_CLEAR == 2:
        jax.clear_caches()
        _MODULES_SINCE_CLEAR = 0
    gc.collect()


@pytest.fixture(scope="module")
def fitted_orbit_mixture():
    """One small, fitted three-component posterior for EIG and diagnostic checks.

    These checks consume a fitted posterior; blind initialization and period
    recovery have dedicated tests. Three blind starts exercise the mixture path.
    """
    import jax.numpy as jnp
    import numpy as np
    from orbit_fit import chart_problem
    from orbix.equations import period_to_sma
    from orbix.system import ClassicalChart
    from sky_reference import sky_position

    from photomancy.measurements import RelativeAstromData
    from photomancy.orbit.init import find_init
    from photomancy.orbit.laplace import map_laplace_mixture_fit

    mass, distance, period, error = 1.989e30, 10.0, 1096.0, 5e-3
    log_range = (float(np.log10(600.0)), float(np.log10(1600.0)))
    times = period * np.array([0.0, 0.23, 0.71, 1.17, 1.89])
    ra, dec = sky_position(
        times,
        T=period,
        a=float(period_to_sma(period, mass)),
        e=0.15,
        cos_i=0.5,
        W=2.3,
        cos_w=np.cos(0.8),
        sin_w=np.sin(0.8),
        tp=-1.5 * period / (2.0 * np.pi),
        dist_pc=distance,
    )
    noise = np.random.default_rng(0).normal(0.0, error, (2, 5))
    data = RelativeAstromData(
        times=jnp.asarray(times),
        ra=jnp.asarray(ra + noise[0]),
        dec=jnp.asarray(dec + noise[1]),
        ra_err=jnp.full(5, error),
        dec_err=jnp.full(5, error),
        corr=jnp.zeros(5),
        planet_id=jnp.zeros(5, dtype=int),
        is_valid=jnp.ones(5, dtype=bool),
    )
    chart = ClassicalChart()
    starts, _ = find_init(data, mass, distance, k=3, log_T_range=log_range, chart=chart)
    problem = chart_problem(
        data,
        {"orbit": starts[0]},
        star_mass_kg=mass,
        dist_pc=distance,
        chart=chart,
        period_range_d=(600.0, 1600.0),
    )
    init = jnp.stack([problem.flatten({"orbit": start}) for start in starts])
    post = map_laplace_mixture_fit(problem, init, n_steps=100)
    return post, problem
