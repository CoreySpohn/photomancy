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
        "period_d": 900.0 + 40.0 * rng.standard_normal(n),
        "e": np.clip(0.3 + 0.05 * rng.standard_normal(n), 0.0, 1.0),
        "cos_i": np.clip(0.4 + 0.1 * rng.standard_normal(n), -1.0, 1.0),
        "ll_total": rng.standard_normal(n),
    }


def test_plot_corner_samples_door_excludes_ll_total():
    """Default params keep the curated order and never show ll_total."""
    import eyepiece as ep

    from photomancy.viz import plot_corner

    result = plot_corner(_samples())
    assert isinstance(result, ep.MosaicResult)
    assert result.axes.shape == (3, 3)
    labels = [result.axes[2, j].get_xlabel() for j in range(3)]
    assert labels == ["P [days]", "e", "cos i"]


@pytest.mark.medium(
    reason="prior draws, one draw model and the corner door compile apart, 7 s cold"
)
def test_plot_corner_reads_the_weighted_draws_of_a_posterior():
    """Fails if the Posterior door ignores the weights, the orbit path or the row.

    Two prior draws of a one-planet orbit fit, all the weight on the second: every
    corner sample is that draw's orbit, read through the fit at ``dynamics.orbits``
    (period, eccentricity, cos i and the argument of periapsis to rtol 1e-12), and a
    missing path is refused.
    """
    from hwoutils.model_paths import resolve
    from orbix.system import ClassicalChart, OrbitBlock, OrbitLaw
    from priorities import LogUniform
    from skyscapes.population import kipping_eccentricity
    from system_helpers import _orbits, _scene

    from photomancy.infer import Fit, Posterior
    from photomancy.orbit import classical_elements
    from photomancy.viz import plot_corner

    model = _scene(
        jax.tree.map(lambda x: x[1], _orbits()),
        mass_Mearth=1.0,
        radius_AU=1e-4,
        albedo=0.3,
    )
    law = OrbitLaw(
        period_d=LogUniform(600.0, 1600.0), eccentricity=kipping_eccentricity()
    )
    block = OrbitBlock(ClassicalChart(), law, rows=(0,), system="dynamics")
    fit = Fit(model, {"dynamics.orbits": block}, ())
    draws = fit.sample_prior(jax.random.key(0), (2,))
    posterior = Posterior(
        samples=draws,
        log_weights=jnp.array([-jnp.inf, 0.0]),
        log_target=fit.log_density(draws),
        log_evidence=jnp.array(0.0),
        log_evidence_se=jnp.array(0.0),
        fit=fit,
        info=None,
        kind="monte_carlo",
    )
    orbit = resolve(posterior.model_at(1), "dynamics.orbits")
    expected = {
        "period_d": orbit.period_d[0],
        "e": orbit.e[0],
        "cos_i": jnp.cos(orbit.to_classical()["i_rad"][0]),
        "w_rad": orbit.to_classical()["w_rad"][0],
    }
    physical = classical_elements(
        posterior, jax.random.key(1), 16, orbit_path="dynamics.orbits"
    )
    for name, value in expected.items():
        np.testing.assert_allclose(physical[name], value, rtol=1e-12, atol=0)
    result = plot_corner(
        posterior,
        key=jax.random.key(1),
        n=16,
        orbit_path="dynamics.orbits",
        params=["period_d", "e", "cos_i"],
    )
    assert result.axes.shape == (3, 3)
    with pytest.raises(TypeError, match="orbit_path"):
        plot_corner(posterior, key=jax.random.key(1))


def test_plot_corner_ranges_pin_the_grid():
    """A range pins its column x-limits and off-diagonal row y-limits."""
    from photomancy.viz import plot_corner

    ranges = {"period_d": (600.0, 1600.0), "e": (0.0, 1.0)}
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
        ranges={"period_d": (600.0, 1600.0)},
    )
    assert result.axes.shape == (3, 3)
    assert result.axes[2, 0].get_xlim() == (600.0, 1600.0)
    assert len(result.artists["hist"]) == 6
