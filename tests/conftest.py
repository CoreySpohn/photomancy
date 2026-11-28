"""Pytest configuration for the photomancy test suite.

Share fitted scenarios and keep compiled primitives across small modules.
Clear caches at module boundaries when a worker reaches the memory limit. JAX's
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
    from hwoutils.testbudget import memory as budget_memory

    pytest_plugins.append("hwoutils.testbudget.plugin")
else:
    budget_memory = None
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")
if importlib.util.find_spec("hwoutils.precision_guard") is not None:
    pytest_plugins.append("hwoutils.precision_guard")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

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
_WORKER_CACHE_RSS_GIB = 2.5


@pytest.fixture(autouse=True, scope="module")
def release_compiled_programs():
    """Keep reusable small programs until a worker reaches 2.5 GiB RSS.

    Check after at least two files so a worker with high residual RSS does not
    clear between every pair of neighbors. Older hwoutils clears after each pair.
    """
    global _MODULES_SINCE_CLEAR
    yield
    _MODULES_SINCE_CLEAR += 1
    if _MODULES_SINCE_CLEAR < 2:
        return
    if (
        budget_memory is not None
        and budget_memory.current_rss_bytes() < _WORKER_CACHE_RSS_GIB * budget_memory.GB
    ):
        return
    jax.clear_caches()
    _MODULES_SINCE_CLEAR = 0
    gc.collect()
