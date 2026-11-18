"""The sampler libraries are optional: imported on use, named when missing."""

import subprocess
import sys

BLOCKED_BACKENDS = """
import sys
sys.modules["blackjax"] = None
sys.modules["jaxns"] = None
sys.modules["jaxctx"] = None

import jax
import jax.numpy as jnp
from photomancy.infer.backends import (
    JaxnsBackend,
    MCLMCBackend,
    NUTSBackend,
    PathfinderBackend,
    SMCBackend,
)

def density(z):
    return -0.5 * jnp.sum(z**2)

key = jax.random.key(0)
z = jnp.zeros(2)
calls = {
    "blackjax": [
        lambda: NUTSBackend(n_warmup=2, n_samples=2).run(density, z, key),
        lambda: MCLMCBackend(n_tune=2, n_samples=2).run(density, z, key),
        lambda: PathfinderBackend().run(density, z, key),
        lambda: SMCBackend().run(density, density, jnp.zeros((4, 2)), key),
    ],
    "jaxns": [lambda: JaxnsBackend().run(None, key=key)],
}
for extra, backends in calls.items():
    for call in backends:
        try:
            call()
        except ImportError as error:
            assert f"photomancy[{extra}]" in str(error), str(error)
        else:
            raise SystemExit(f"expected ImportError naming photomancy[{extra}]")
"""


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_photomancy_loads_no_sampler_library():
    """Importing the package and its backends leaves blackjax and jaxns unloaded.

    Fails if a backend module, or anything ``import photomancy`` reaches, imports
    a sampler library at module level, which would make the extras mandatory.
    """
    result = _run(
        "import sys; import photomancy; import photomancy.infer.backends; "
        "import photomancy.orbit; "
        "loaded = {'blackjax', 'jaxns', 'jaxctx'} & set(sys.modules); "
        "assert not loaded, loaded"
    )
    assert result.returncode == 0, result.stderr


def test_backend_without_its_sampler_names_the_extra():
    """Each sampler backend raises an ImportError naming its extra when it runs.

    Fails if a backend surfaces a bare ModuleNotFoundError, or names the wrong
    extra, when blackjax or jaxns is not installed.
    """
    result = _run(BLOCKED_BACKENDS)
    assert result.returncode == 0, result.stderr
