"""The sampler libraries are optional: imported on use, named when missing."""

import subprocess
import sys
from textwrap import indent

import pytest

BLOCKED_BACKENDS = """
import sys
sys.modules["blackjax"] = None
sys.modules["jaxns"] = None
sys.modules["jaxctx"] = None

import jax
import jax.numpy as jnp
import equinox as eqx
from priorities import Uniform
from photomancy.infer.likelihoods import gaussian_loglike
from photomancy.infer import MCLMC, NUTS, Fit, NestedSampling, Pathfinder, run
from photomancy.infer.terms import ForwardTerm

class Model(eqx.Module):
    x: jax.Array

class Forward(eqx.Module):
    def __call__(self, model):
        return model.x

fit = Fit(
    Model(jnp.zeros(2)),
    {"x": Uniform(-1.0, 1.0)},
    (ForwardTerm(Forward(), gaussian_loglike(jnp.zeros(2), 1.0)),),
)
key = jax.random.key(0)
calls = {
    "blackjax": [
        lambda: run(fit, NUTS(n_warmup=2, n_draws=2), key),
        lambda: run(fit, MCLMC(n_warmup=2, n_draws=2), key),
        lambda: run(fit, Pathfinder(), key),
    ],
    "jaxns": [lambda: run(fit, NestedSampling(), key)],
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


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_backend_without_its_sampler_names_the_extra(tmp_path, precision, pytestconfig):
    """Each sampler backend raises an ImportError naming its extra when it runs.

    Fails if a backend surfaces a bare ModuleNotFoundError, or names the wrong
    extra, when blackjax or jaxns is not installed.
    """
    child = tmp_path / "test_blocked_backends.py"
    child.write_text(
        "def test_blocked_backends():\n" + indent(BLOCKED_BACKENDS, "    ")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(pytestconfig.inipath),
            f"--precision={precision.name}",
            "-q",
            "-p",
            "hwoutils.precision_guard",
            "-p",
            "no:cacheprovider",
            f"--basetemp={tmp_path / 'pytest'}",
            str(child),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
