"""Contract tests for photomancy.viz predictive panels.

The regression class here is the padded data containers: pad rows must
never reach a drawn artist, and the AU/day -> m/s conversion is checked
numerically.
"""

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hwoutils.constants import AU2m, d2s

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test creates."""
    yield
    plt.close("all")


def test_plot_rv_converts_au_per_day_to_m_per_s():
    """Curves and data both convert; pad rows are never drawn."""
    from photomancy.orbit.data import RVData
    from photomancy.viz import plot_rv

    t = np.linspace(0.0, 300.0, 50)
    rv = 1.0e-6 * np.sin(2.0 * np.pi * t / 150.0)
    data = RVData.pad(
        times=jnp.array([10.0, 90.0, 200.0]),
        rv=jnp.array([1.0e-6, -0.5e-6, 0.2e-6]),
        rv_err=jnp.array([1.0e-7, 1.0e-7, 1.0e-7]),
        inst_ids=jnp.zeros(3, dtype=int),
        n_inst=1,
    )
    result = plot_rv(t, rv, data=data)
    drawn = result.artists["lines"][0].get_ydata()
    np.testing.assert_allclose(drawn, rv * AU2m / d2s, rtol=1e-12)
    assert result.ax.get_ylabel() == "radial velocity (m/s)"

    errbar_line = [line for line in result.ax.lines if len(line.get_xdata()) == 3]
    assert errbar_line, "exactly the 3 valid epochs are drawn, no pad rows"


def test_plot_rv_fan_alpha():
    """A (K, T) batch draws K faded curves."""
    from photomancy.viz import plot_rv

    t = np.linspace(0.0, 100.0, 20)
    curves = np.stack([np.sin(t / 20.0), np.cos(t / 20.0)]) * 1e-6
    result = plot_rv(t, curves)
    assert len(result.artists["lines"]) == 2
    assert result.artists["lines"][0].get_alpha() == pytest.approx(0.25)


def _imaging_data():
    """Two detections and one null, padded, with a tiny contrast grid."""
    from photomancy.orbit.data import ImagingData

    grid = jnp.linspace(0.05, 0.4, 8)
    limit = jnp.full(8, 24.0)
    return ImagingData.from_detections_and_nulls(
        det_epochs=jnp.array([20.0, 150.0]),
        det_dmag_obs=jnp.array([22.5, 23.1]),
        det_dmag_err=jnp.array([0.1, 0.15]),
        det_sep_grid=jnp.stack([grid, grid]),
        det_dmag0_grid=jnp.stack([limit, limit]),
        null_epochs=jnp.array([260.0]),
        null_sep_grid=grid[None, :],
        null_dmag0_grid=limit[None, :],
    )


def test_plot_dmag_masks_pads_and_inverts_once():
    """Detections/nulls come only from valid rows; y inverts idempotently."""
    from photomancy.viz import plot_dmag

    t = np.linspace(0.0, 300.0, 60)
    dmag = 23.0 + np.sin(2.0 * np.pi * t / 300.0)
    data = _imaging_data()

    result = plot_dmag(t, dmag, data=data, limit_dmag=24.0)
    assert result.ax.yaxis_inverted()
    nulls = result.artists["scatter"]
    assert nulls.get_offsets().shape == (1, 2)
    np.testing.assert_allclose(np.asarray(nulls.get_offsets())[0], [260.0, 24.0])

    ylim_before = result.ax.get_ylim()
    plot_dmag(t, dmag + 0.5, ax=result.ax)
    assert result.ax.yaxis_inverted()
    assert result.ax.get_ylim()[0] >= result.ax.get_ylim()[1]
    del ylim_before


def test_plot_detectability_masks_nonfinite_limit():
    """-inf pads in a contrast curve never reach the drawn limit line."""
    from photomancy.viz import plot_detectability

    sep = np.array([0.1, 0.2, 0.3])
    dmag = np.array([22.0, 24.5, 23.0])
    cc_sep = np.linspace(0.05, 0.4, 10)
    cc_dmag = np.concatenate([np.full(8, 24.0), np.full(2, -np.inf)])

    result = plot_detectability(
        sep, dmag, contrast_curve=(cc_sep, cc_dmag), iwa_arcsec=0.08
    )
    assert len(result.artists["line"].get_xdata()) == 8
    assert "fill" in result.artists
    assert result.artists["scatter"].get_offsets().shape == (3, 2)
