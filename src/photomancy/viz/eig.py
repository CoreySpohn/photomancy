"""The expected-information-gain decision panel.

One shape, reimplemented at every scheduling call site: the EIG
decomposition against candidate epoch, existing epochs as reference
lines, and the chosen next epoch marked. The input is the ``LinearizedEIG``
that ``eig_linearized`` returns, or the ``AstrometryEIG`` of ``eig_astrometry``,
so the panel keys off the library's own vocabulary (``total``, ``alias``,
``geometric``) rather than caller-side unpacking.
"""

import numpy as np

from photomancy.viz._require import eyepiece

_COMPONENTS = ("total", "alias", "geometric")


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
        result: The ``LinearizedEIG`` of ``eig_linearized`` or the
            ``AstrometryEIG`` of ``eig_astrometry`` (any object with the
            attributes); whichever of ``total`` / ``alias`` / ``geometric``
            are present are drawn, in one stable color each (the cast is
            declared per call, so the components keep their colors across
            figures).
        epochs_d: Optional existing observation epochs, drawn as faint
            vertical reference lines.
        mark_best: Mark ``nanargmax(total)`` with a dashed vertical line
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
    result = getattr(result, "eig", result)
    t = np.asarray(candidates_d, float).reshape(-1)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(layout="constrained")

    styles = ep.style.SourceStyles(list(_COMPONENTS))
    lines = []
    # Total is the SUM of the two components, not a third peer measurement
    # (eig_linearized returns geometric + alias, geometric,
    # alias). Drawn as three equal lines it reads as three independent
    # metrics, and once alias goes to zero the total and geometric curves
    # land on each other and overprint into a color belonging to neither.
    # So the total is a wide, pale envelope UNDER its own components: the
    # components stay legible on top, and a component sitting inside the
    # envelope is the additive identity made visible.
    for name in _COMPONENTS:
        component = getattr(result, name, None)
        if component is None:
            continue
        values = np.asarray(component, float).reshape(-1)
        is_total = name == "total"
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

    if mark_best and getattr(result, "total", None) is not None:
        best = t[int(np.nanargmax(np.asarray(result.total)))]
        lines.append(ax.axvline(best, color=styles["total"]["color"], ls="--", lw=1.2))

    ax.set_xlabel("candidate epoch [days]")
    ax.set_ylabel("expected information gain [nats]")
    # A pinned corner, not the default "best": matplotlib re-solves "best"
    # on every draw, so an animated panel's legend hops between frames as
    # the curves move under it.
    ax.legend(loc=legend_loc)
    return ep.PlotResult(ax=ax, artists=artists)
