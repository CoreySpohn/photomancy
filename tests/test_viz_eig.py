"""Contract tests for photomancy.viz.plot_eig."""

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


def _result(n=40):
    """A synthetic evaluate_candidates-shaped result with a known argmax."""
    t = np.linspace(0.0, 400.0, n)
    geometric = 0.2 + 0.1 * np.sin(t / 60.0)
    alias = np.exp(-0.5 * ((t - 250.0) / 40.0) ** 2)
    return t, {
        "total_eig": geometric + alias,
        "alias_eig": alias,
        "geometric_eig": geometric,
    }


def test_plot_eig_draws_components_and_marks_best():
    """All three components draw; the best epoch gets a dashed vline."""
    import eyepiece as ep

    from photomancy.viz import plot_eig

    t, result_dict = _result()
    result = plot_eig(t, result_dict, epochs_d=[0.0, 100.0])
    assert isinstance(result, ep.PlotResult)
    assert len(result.artists["lines"]) == 4
    assert len(result.artists["line"]) == 2

    best_line = result.artists["lines"][-1]
    expected = t[int(np.argmax(result_dict["total_eig"]))]
    assert best_line.get_xdata()[0] == pytest.approx(expected)
    assert result.ax.get_ylabel() == "expected information gain (nats)"


def test_plot_eig_tolerates_missing_components():
    """A result with only total_eig still draws."""
    from photomancy.viz import plot_eig

    t, result_dict = _result()
    result = plot_eig(t, {"total_eig": result_dict["total_eig"]}, mark_best=False)
    assert len(result.artists["lines"]) == 1


def test_plot_eig_reuses_handed_axes():
    """A handed ax is drawn into; no new figure is created."""
    from photomancy.viz import plot_eig

    _, axes = plt.subplots(1, 2, layout="constrained")
    t, result_dict = _result()
    result = plot_eig(t, result_dict, ax=axes[0])
    assert result.ax is axes[0]
    w0 = axes[0].get_position(original=True).width
    w1 = axes[1].get_position(original=True).width
    assert w0 == pytest.approx(w1)
