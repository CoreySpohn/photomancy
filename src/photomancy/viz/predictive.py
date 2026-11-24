"""Predictive panels: RV curves, dMag tracks, and the detectability plane.

These functions own the domain knowledge the hand-rolled versions kept
fumbling: ``plot_rv`` takes curves and observations in m/s;
``dMag`` axes read fainter downward (``plot_dmag`` inverts once,
idempotently); and the padded data containers carry ``valid`` masks and
``-inf`` pads that must never reach a drawn artist.
"""

import numpy as np

from photomancy.viz._require import eyepiece


def _curves(values):
    """Normalize ``(T,)`` or ``(K, T)`` curve input to ``(K, T)``."""
    arr = np.atleast_2d(np.asarray(values, float))
    return arr


def _new_axes():
    """A fresh constrained-layout axes; matplotlib stays a lazy import."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(layout="constrained")
    return ax


def _track_color(ep):
    """The single track color, resolved from the active palette per call."""
    return ep.style.SourceStyles(["track"])["track"]["color"]


def plot_rv(
    t_d,
    rv_m_per_s,
    *,
    data=None,
    ax=None,
    line_kw=None,
):
    """Radial-velocity curves and observations, all in m/s.

    Curves are plotted as given (``RadialVelocityTerm.predict`` returns m/s), with no
    unit conversion; ``RadialVelocityData`` observations and uncertainties are m/s too.

    Args:
        t_d: Times in days, shape ``(T,)``.
        rv_m_per_s: Predicted curves in m/s, ``(T,)`` or ``(K, T)`` --
            a fan of posterior-predictive draws enters as the batch.
        data: Optional ``RadialVelocityData``; the rows its ``valid`` mask selects are
            drawn as errorbars, in m/s.
        ax: Axes to draw into. None creates a new figure.
        line_kw: Extra kwargs for each curve's ``ax.plot``, applied last.

    Returns:
        An ``eyepiece.PlotResult`` with ``"lines"`` (one per curve) and,
        when ``data`` is given, ``"collection"`` (the errorbar's caps).
    """
    ep = eyepiece()
    t = np.asarray(t_d, float).reshape(-1)
    curves = _curves(rv_m_per_s)

    if ax is None:
        ax = _new_axes()

    color = _track_color(ep)
    alpha = 1.0 if curves.shape[0] == 1 else 0.25
    kw = {"color": color, "lw": 1.5, "alpha": alpha, **(line_kw or {})}
    lines = [ax.plot(t, curve, **kw)[0] for curve in curves]
    artists = {"lines": lines}

    if data is not None:
        keep = np.asarray(data.valid, bool)
        errbar = ax.errorbar(
            np.asarray(data.time_d)[keep],
            np.asarray(data.rv)[keep],
            yerr=np.asarray(data.rv_err)[keep],
            fmt="o",
            color=ep.style.neutral(1.0),
            markersize=4,
            lw=1.0,
            zorder=5,
        )
        artists["collection"] = errbar.lines[2][0]

    ax.set_xlabel("time [days]")
    ax.set_ylabel("radial velocity [m/s]")
    return ep.PlotResult(ax=ax, artists=artists)


def plot_dmag(
    t_d,
    dmag,
    *,
    data=None,
    limit_dmag=None,
    ax=None,
    line_kw=None,
):
    """Delta-magnitude tracks against time, fainter drawn downward.

    Args:
        t_d: Times in days, shape ``(T,)``.
        dmag: Predicted dMag curves, ``(T,)`` or ``(K, T)``.
        data: Optional ``DMagData``; valid detections are drawn as
            errorbars, valid nulls as downward triangles at ``limit_dmag``
            (nulls are skipped when no limit is given, since a null has no
            measured dMag to sit at).
        limit_dmag: Optional detection-limit level, drawn as a dashed
            horizontal line.
        ax: Axes to draw into. None creates a new figure. The y axis is
            inverted so fainter is downward; an already-inverted axis is
            left alone, so overplotting does not flip it back.
        line_kw: Extra kwargs for each curve's ``ax.plot``, applied last.

    Returns:
        An ``eyepiece.PlotResult`` with ``"lines"`` (one per curve), plus
        ``"collection"`` (detection errorbars), ``"scatter"`` (null
        markers), and ``"line"`` (the limit line) when drawn.
    """
    ep = eyepiece()
    t = np.asarray(t_d, float).reshape(-1)
    curves = _curves(dmag)

    if ax is None:
        ax = _new_axes()

    color = _track_color(ep)
    alpha = 1.0 if curves.shape[0] == 1 else 0.25
    kw = {"color": color, "lw": 1.5, "alpha": alpha, **(line_kw or {})}
    lines = [ax.plot(t, curve, **kw)[0] for curve in curves]
    artists = {"lines": lines}

    neutral = ep.style.neutral(1.0)
    if limit_dmag is not None:
        artists["line"] = ax.axhline(
            limit_dmag, color=neutral, ls="--", lw=0.8, alpha=0.6
        )

    if data is not None:
        valid = np.asarray(data.valid)
        detected = np.asarray(data.is_detected) & valid
        nulls = ~np.asarray(data.is_detected) & valid
        epochs = np.asarray(data.time_d)
        if detected.any():
            errbar = ax.errorbar(
                epochs[detected],
                np.asarray(data.dmag_obs)[detected],
                yerr=np.asarray(data.dmag_err)[detected],
                fmt="o",
                color=neutral,
                markersize=4,
                lw=1.0,
                zorder=5,
            )
            artists["collection"] = errbar.lines[2][0]
        if nulls.any() and limit_dmag is not None:
            artists["scatter"] = ax.scatter(
                epochs[nulls],
                np.full(int(nulls.sum()), limit_dmag),
                marker="v",
                color=neutral,
                s=25,
                zorder=5,
            )

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("time [days]")
    ax.set_ylabel("delta-magnitude (fainter downward)")
    return ep.PlotResult(ax=ax, artists=artists)


def plot_detectability(
    separation_arcsec,
    dmag,
    *,
    contrast_curve=None,
    iwa_arcsec=None,
    ax=None,
    scatter_kw=None,
):
    """The separation-dMag plane: predicted points against the limit curve.

    Args:
        separation_arcsec: Separations in arcsec, any shape (flattened).
        dmag: Delta-magnitudes, same shape.
        contrast_curve: Optional ``(separation_arcsec, dmag_limit)`` pair of
            arrays, drawn as the detection-limit curve; points fainter than
            the curve (larger dMag) are undetectable, and with the y axis
            inverted those points sit visually below it.
        iwa_arcsec: Optional inner working angle; the region inside it is
            shaded as unobservable.
        ax: Axes to draw into. None creates a new figure. The y axis is
            inverted so fainter is downward, matching ``plot_dmag`` and the
            direct-imaging contrast-curve convention; an already-inverted
            axis is left alone, so overplotting does not flip it back.
        scatter_kw: Extra kwargs for the ``ax.scatter`` call, applied last.

    Returns:
        An ``eyepiece.PlotResult`` with ``"scatter"`` (the cloud), plus
        ``"line"`` (the limit curve) and ``"fill"`` (the IWA span) when
        drawn.
    """
    ep = eyepiece()
    sep = np.asarray(separation_arcsec, float).reshape(-1)
    dm = np.asarray(dmag, float).reshape(-1)

    if ax is None:
        ax = _new_axes()

    color = _track_color(ep)
    kw = {"s": 8, "alpha": 0.4, "color": color, "edgecolors": "none"}
    kw.update(scatter_kw or {})
    artists = {"scatter": ax.scatter(sep, dm, **kw)}

    neutral = ep.style.neutral(1.0)
    if contrast_curve is not None:
        cc_sep = np.asarray(contrast_curve[0], float).reshape(-1)
        cc_dmag = np.asarray(contrast_curve[1], float).reshape(-1)
        finite = np.isfinite(cc_dmag)
        (limit,) = ax.plot(
            cc_sep[finite], cc_dmag[finite], color=neutral, lw=1.2, ls="-"
        )
        artists["line"] = limit
    if iwa_arcsec is not None:
        artists["fill"] = ax.axvspan(0.0, iwa_arcsec, color=neutral, alpha=0.12, lw=0)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("separation [arcsec]")
    ax.set_ylabel("delta-magnitude (fainter downward)")
    return ep.PlotResult(ax=ax, artists=artists)
