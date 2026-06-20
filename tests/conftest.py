"""Pytest configuration for the photomancy test suite.

Enables JAX's persistent compilation cache so repeat local runs reuse compiled
kernels instead of recompiling the NumPyro orbit model and the JAX fitters, which
dominate the suite's wall-clock. The first cold run still pays full compilation;
subsequent runs are much faster. The cache directory is gitignored.
"""

from pathlib import Path

import jax

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".jax_cache"
jax.config.update("jax_compilation_cache_dir", str(_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
