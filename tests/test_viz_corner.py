"""Contract tests for photomancy.viz corner plots."""

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test creates."""
    yield
    plt.close("all")


def _samples(n=200, seed=3):
    """A physical-sample dict with a deterministic ll_total leak included."""
    rng = np.random.default_rng(seed)
    return {
        "T": 900.0 + 40.0 * rng.standard_normal(n),
        "e": np.clip(0.3 + 0.05 * rng.standard_normal(n), 0.0, 1.0),
        "cos_i": np.clip(0.4 + 0.1 * rng.standard_normal(n), -1.0, 1.0),
        "ll_total": rng.standard_normal(n),
    }


class _Problem:
    """A minimal OrbitProblem stand-in with a traceable to_physical."""

    param_names = ("z0", "z1")

    def to_physical(self, z):
        """Map a flat z to a physical dict, leaking ll_total like the real one."""
        return {
            "T": 900.0 + 40.0 * z[0],
            "e": 0.3 * jax.nn.sigmoid(z[1]),
            "ll_total": z[0],
        }


def test_plot_corner_samples_door_excludes_ll_total():
    """Default params keep the curated order and never show ll_total."""
    import eyepiece as ep

    from photomancy.viz import plot_corner

    result = plot_corner(_samples())
    assert isinstance(result, ep.MosaicResult)
    assert result.axes.shape == (3, 3)
    labels = [result.axes[2, j].get_xlabel() for j in range(3)]
    assert labels == ["T (days)", "e", "cos i"]


def test_plot_corner_posterior_door_requires_problem_and_key():
    """The posterior door draws via sample_physical and names what is missing."""
    from photomancy.posterior import GaussianPosterior
    from photomancy.viz import plot_corner

    posterior = GaussianPosterior(
        mean=jnp.zeros(2), cov=jnp.eye(2), evidence=jnp.array(0.0)
    )
    result = plot_corner(posterior, _Problem(), key=jax.random.PRNGKey(0), n=64)
    assert result.axes.shape == (2, 2)

    with pytest.raises(TypeError, match="problem, key"):
        plot_corner(posterior)


def test_plot_corner_ranges_pin_the_grid():
    """A range pins its column x-limits and off-diagonal row y-limits."""
    from photomancy.viz import plot_corner

    ranges = {"T": (600.0, 1600.0), "e": (0.0, 1.0)}
    result = plot_corner(_samples(), ranges=ranges)
    assert result.axes[0, 0].get_xlim() == (600.0, 1600.0)
    assert result.axes[2, 0].get_xlim() == (600.0, 1600.0)
    assert result.axes[1, 0].get_ylim() == (0.0, 1.0)
    assert result.axes[1, 1].get_xlim() == (0.0, 1.0)


def test_plot_corner_overlay_names_and_ranges():
    """Overlay takes a list of dicts, shares labels and pinned ranges."""
    from photomancy.viz import plot_corner_overlay

    result = plot_corner_overlay(
        [_samples(seed=3), _samples(seed=4)],
        names=["run a", "run b"],
        ranges={"T": (600.0, 1600.0)},
    )
    assert result.axes.shape == (3, 3)
    assert result.axes[2, 0].get_xlim() == (600.0, 1600.0)
    assert len(result.artists["hist"]) == 6
