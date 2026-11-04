"""Pytest configuration for the photomancy test suite.

Enables JAX's persistent compilation cache so repeat local runs reuse compiled
kernels instead of recompiling the NumPyro orbit model and the JAX fitters, which
dominate the suite's wall-clock. The first cold run still pays full compilation;
subsequent runs are much faster. The cache directory is gitignored.
"""

import importlib.util
import os
from pathlib import Path

import jax
from hypothesis import settings

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".jax_cache"
jax.config.update("jax_compilation_cache_dir", str(_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
