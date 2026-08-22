"""The expected-information-gain decision panel.

One shape, reimplemented at every scheduling call site: the EIG
decomposition against candidate epoch, existing epochs as reference
lines, and the chosen next epoch marked. The input is the result dict
``evaluate_candidates`` / ``evaluate_orbit_candidates`` returns, so the
panel keys off the library's own vocabulary (``total_eig``,
``alias_eig``, ``geometric_eig``) rather than caller-side unpacking.
"""

import numpy as np

from photomancy.viz._require import eyepiece

_COMPONENTS = (
    ("total_eig", "total"),
    ("alias_eig", "alias"),
    ("geometric_eig", "geometric"),
)


def plot_eig(
    candidates_d,
    result,
    *,
    epochs_d=None,
    mark_best=True,
    ax=None,
    line_kw=None,
    legend_loc="upper right",
):
    """EIG decomposition against candidate observation epoch.

    Args:
        candidates_d: Candidate epochs in days, shape ``(N,)``.
        result: The dict ``evaluate_candidates`` /
            ``evaluate_orbit_candidates`` returns; whichever of
            ``total_eig`` / ``alias_eig`` / ``geometric_eig`` are present
            are drawn, in one stable color each (the cast is declared per
            call, so the components keep their colors across figures).
        epochs_d: Optional existing observation epochs, drawn as faint
            vertical reference lines.
        mark_best: Mark ``argmax(total_eig)`` with a dashed vertical line
            in the total component's color.
        ax: Axes to draw into. None creates a new figure.
        line_kw: Extra kwargs for each component's ``ax.plot``, applied
            last.
        legend_loc: Legend corner. Defaults to a pinned corner rather than
            ``"best"``, which matplotlib re-solves per draw and which
            therefore makes the legend jump between animation frames.

    Returns:
        An ``eyepiece.PlotResult`` whose ``"lines"`` holds the component
        curves (in the order drawn), then the best-epoch line when marked;
        epoch reference lines land under ``"line"`` as a list.
    """
    ep = eyepiece()
    t = np.asarray(candidates_d, float).reshape(-1)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(layout="constrained")

    styles = ep.SourceStyles([name for _, name in _COMPONENTS])
    lines = []
    # Total is the SUM of the two components, not a third peer measurement
    # (evaluate_candidates returns weighted_geom + alias_val, weighted_geom,
    # alias_val). Drawn as three equal lines it reads as three independent
    # metrics, and once alias goes to zero the total and geometric curves
    # land on each other and overprint into a color belonging to neither.
    # So the total is a wide, pale envelope UNDER its own components: the
    # components stay legible on top, and a component sitting inside the
    # envelope is the additive identity made visible.
    for key, name in _COMPONENTS:
        if key not in result:
            continue
        values = np.asarray(result[key], float).reshape(-1)
        is_total = key == "total_eig"
        kw = {
            "color": styles[name]["color"],
            "lw": 4.0 if is_total else 1.4,
            "alpha": 0.35 if is_total else 1.0,
            "zorder": 1.5 if is_total else 2.5,
            "label": name,
            **(line_kw or {}),
        }
        (line,) = ax.plot(t, values, **kw)
        lines.append(line)
    artists = {"lines": lines}

    if epochs_d is not None:
        import matplotlib as mpl

        reference = mpl.rcParams["text.color"]
        artists["line"] = [
            ax.axvline(float(epoch), color=reference, lw=0.6, alpha=0.25)
            for epoch in np.asarray(epochs_d, float).reshape(-1)
        ]

    if mark_best and "total_eig" in result:
        best = t[int(np.argmax(np.asarray(result["total_eig"])))]
        lines.append(ax.axvline(best, color=styles["total"]["color"], ls="--", lw=1.2))

    ax.set_xlabel("candidate epoch [days]")
    ax.set_ylabel("expected information gain [nats]")
    # A pinned corner, not the default "best": matplotlib re-solves "best"
    # on every draw, so an animated panel's legend hops between frames as
    # the curves move under it.
    ax.legend(loc=legend_loc)
    return ep.PlotResult(ax=ax, artists=artists)
