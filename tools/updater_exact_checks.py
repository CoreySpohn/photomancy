#!/usr/bin/env python3
"""Exact checks of the updater references against numerical quadrature.

The protocol (``docs/updater-protocol.md``, "Exact checks") anchors the
accuracy of both reference samplers on low-dimensional cases, two integrated
to a demonstrated accuracy and one graded against a brute-force sampler:

- **physical_fixed**: one epoch at t = 0 (3.5 mas), regime R1 support, future
  epoch 30 days;
- **ti_window**: the TI model with periods of 200 to 400 days and the mass
  window ``reference.exact_checks.ti_mass_window_msun`` of the parameter
  block, epochs at 0 and 60 days (3.5 mas), future epoch 120 days;
- **ti_binding**: the same periods and epochs with a mass window that cuts
  the posterior (``reference.exact_checks.binding_case``) and large errors,
  graded against the mean of ``sampler_repeats`` runs of the anchor sampler,
  which draws every coordinate from the prior and weights by the likelihood
  and the window indicator directly.

Each case has 28 graded values: the eight-bin masses of ln P, e and the future
position angle (east of north), and the end points of the central 90 percent
intervals of the future east and north offsets. The quadrature is evaluated at
three refinement levels, each doubling the nodes of every dimension; the
quadrature error of a value is twice the change between the two finest levels,
and that change must be smaller than the previous one. The graded samplers
(the reference and, for the TI cases, the check-3 frequency-mixture sampler
at twice the budget; the updater for information) run ``sampler_repeats``
times with independent keys, and a value passes when
``|mean - x_q| <= eps_q + max(t* s / sqrt(R), f)``.

Usage::

    python tools/updater_exact_checks.py --out DIR [--levels 0,1,2]
        [--cases ti_window,physical_fixed,ti_binding]
        [--samplers anchor,reference,independent,updater]
        [--repeats 16] [--shard i/n]
    python tools/updater_exact_checks.py --report DIR [--write EXACT.json]

Every quadrature level and sampler repeat is one resumable work unit written
as a JSON line; ``--report`` assembles the refinement tables, the comparison
and the per-mass-model verdicts that ``updater_panel.py --freeze --exact``
reads.

Quadrature, physical case. With shape ``s = (e, cos i, omega, M_ref)`` and the
anchor position ``t`` in polar coordinates ``(rho, phi)``, the one-epoch
posterior density is ``p(s) N(t; obs, sigma^2 I) / rho`` on ``rho`` between
``r0(s) a_lo`` and ``r0(s) a_hi`` (``r0`` the anchor separation of the orbit at
``a = 1``, ``W = 0``). Writing ``|t - obs|^2 = (rho - rho_o)^2 + 2 rho rho_o
(1 - cos x)`` with ``x = phi - phi_o`` separates a Gaussian in ``rho`` from a
von Mises kernel of concentration ``K = rho rho_o / sigma^2`` in ``x``. The
outer rules are Gauss on the truncated-Beta CDF coordinate of ``e`` (split at
the e-bin edges), Gauss on ``cos i`` and periodic trapezoids on ``omega`` and
``M_ref``. The ``rho`` rule is Gauss in ``u = Phi((rho - rho_o) / sigma)``,
split at the ln P bin edges ``r0 a_j``, which removes the Gaussian factor. The
future position angle is ``phi`` plus an offset fixed by ``(s, rho)``, so the
angle-bin masses and the east and north CDFs are exact arc integrals of the
von Mises kernel; with ``y = 2 sqrt(K) sin(x / 2)`` each arc integral is a
normal CDF plus a small smooth correction integrated by a Gauss rule in
``y``.

Quadrature, TI case. Given ``(e, ln P, tau)`` the predictions are linear in the
four TI constants with a Gaussian prior, so they are integrated analytically
from dense normal equations (a code path separate from the updater's Cholesky
helper, which the collapsed reference uses): each node carries its marginal
likelihood and a Gaussian future position. Gauss rules in ``e`` and ``ln P``
are split at the bin edges, the ``tau`` rule is a periodic trapezoid; angle-bin
masses are exact Gaussian wedge probabilities (Owen's T), and the end points
come from the Gaussian mixture CDF. The mass window is bounded per node: a ball
of radius ``r`` around the conditional mean lies in the window when its image
interval does (``|c| / sqrt 2 <= a <= |c|``), and the mass outside is at most
the chi-square (4 degrees of freedom) tail at ``r^2 / lambda_max``; the
weighted sum of the bounds must be below ``1e-6``.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CASES = ("ti_window", "physical_fixed", "ti_binding")
SAMPLERS = ("anchor", "reference", "independent", "updater")
# Samplers run per case; "anchor" is the binding case's brute-force grading
# reference, "reference" and "independent" are graded (amendment A7).
CASE_SAMPLERS = {
    "ti_window": ("reference", "independent", "updater"),
    "physical_fixed": ("reference", "updater"),
    "ti_binding": ("anchor", "reference", "independent"),
}
GRADED = ("reference", "independent")
# The mass model whose verdict each case feeds.
CASE_MASS_MODEL = {
    "ti_window": "ti_window",
    "physical_fixed": "physical_fixed",
    "ti_binding": "ti_window",
}
WINDOW_BOUND_MAX = 1e-6
# Physical case: the anchor kernel is integrated over |y| <= Y_CAP (mass
# beyond is below 1e-18 of the total), which needs K >= K_MIN for a smooth
# arc integrand.
Y_CAP = 9.0
K_MIN = 100.0
RHO_WINDOW_SIGMA = 9.0

# Base node counts; level L multiplies each by 2^L.
PHYSICAL_BASE = {
    "e_per_bin": 1,
    "cos_i": 8,
    "omega": 8,
    "m_ref": 8,
    "rho_per_piece": 3,
    "arc": 4,
}
TI_BASE = {"e_per_bin": 2, "lnP_per_bin": 2, "tau": 32}


def _load_panel():
    spec = importlib.util.spec_from_file_location(
        "updater_panel", TOOLS / "updater_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panel = _load_panel()


def value_names():
    """The 28 graded value names of one case, in canonical order."""
    names = [f"{q}_bin{b}" for q in ("lnP", "e", "future_pa") for b in range(8)]
    return names + [
        f"future_{axis}_{end}" for axis in ("east", "north") for end in ("lo", "hi")
    ]


def level_counts(base, level):
    """Node counts of a refinement level."""
    return {k: v * 2**level for k, v in base.items()}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


# Mass window of the reduced TI case (solar masses) when the parameter block
# does not set ``reference.exact_checks.ti_mass_window_msun``.
TI_MASS_WINDOW_MSUN = (1e-3, 1e3)


def ti_reduced_model(protocol=None):
    """The TI model of the exact check: P 200 to 400 days, wide mass window."""
    from photomancy.orbit.few_epoch import MU_SUN, TIBasisModel

    window = TI_MASS_WINDOW_MSUN
    if protocol is not None:
        window = protocol["reference"]["exact_checks"].get(
            "ti_mass_window_msun", window
        )
    return TIBasisModel(
        lnP_lo=math.log(200.0),
        lnP_hi=math.log(400.0),
        mu_lo=window[0] * MU_SUN,
        mu_hi=window[1] * MU_SUN,
    )


def ti_binding_model(protocol):
    """The binding-window TI case: P 200 to 400 days, a narrow mass window."""
    from photomancy.orbit.few_epoch import MU_SUN, TIBasisModel

    window = protocol["reference"]["exact_checks"]["binding_case"]["mass_window_msun"]
    return TIBasisModel(
        lnP_lo=math.log(200.0),
        lnP_hi=math.log(400.0),
        mu_lo=window[0] * MU_SUN,
        mu_hi=window[1] * MU_SUN,
    )


def _key_slot(case):
    """(mass model, regime index) of a case's exact-stage keys."""
    return ("ti_window", 1) if case == "ti_binding" else (case, 0)


def case_setup(protocol, case):
    """Truth, records, support and grading edges of one exact-check case."""
    import jax
    import numpy as np
    from hwoutils.constants import mas2arcsec

    from photomancy.orbit.updater import ThieleInnesTarget

    panel.enable_x64()
    r1 = protocol["regimes"][0]
    sigma = r1["sigma_mas"] * mas2arcsec
    slot = _key_slot(case)
    truth_key = panel.draw_key(protocol, "exact", *slot, 0, "truth", 0)
    noise_key = panel.draw_key(protocol, "exact", *slot, 0, "noise", 0)
    if case == "physical_fixed":
        target = panel.update_target(protocol, r1, "physical_fixed")
        truth = target.from_physical(panel.physical_truth(protocol, r1, truth_key))
        times = np.array([0.0])
        t_f = 30.0
        p_lo, p_hi, e_max = panel.regime_bounds(protocol, r1, "physical_fixed")
    else:
        if case == "ti_binding":
            model = ti_binding_model(protocol)
            binding = protocol["reference"]["exact_checks"]["binding_case"]
            sigma = binding["sigma_mas"] * mas2arcsec
        else:
            model = ti_reduced_model(protocol)
        target = ThieleInnesTarget(model, protocol["distance_pc"])
        truth = panel.ti_truth(model, truth_key)
        times = np.array([0.0, 60.0])
        t_f = 120.0
        p_lo, p_hi, e_max = math.exp(model.lnP_lo), math.exp(model.lnP_hi), model.e_max
    ra, dec = target.predict(truth, times)
    noise = sigma * jax.random.normal(noise_key, (2, len(times)))
    n_bins = protocol["criteria"]["alias_accuracy"]["bins"]
    edges = {
        "lnP": np.linspace(math.log(p_lo), math.log(p_hi), n_bins + 1),
        "e": np.linspace(0.0, e_max, n_bins + 1),
        "future_pa": np.linspace(0.0, 2.0 * math.pi, n_bins + 1),
    }
    return {
        "case": case,
        "target": target,
        "truth": np.asarray(truth),
        "times": times,
        "ra": np.asarray(ra + noise[0]),
        "dec": np.asarray(dec + noise[1]),
        "sigma": sigma,
        "t_f": t_f,
        "period_days": (p_lo, p_hi),
        "e_max": e_max,
        "edges": edges,
    }


def records(setup, mask=None):
    """The case's epochs as ``RelativeAstromData``."""
    world = {k: setup[k] for k in ("times", "ra", "dec", "sigma")}
    return panel.astrometry(world, len(setup["times"]), mask)


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------


def gauss_pieces(edges, n):
    """Gauss-Legendre nodes and weights on each interval of ``edges``.

    Returns ``(nodes, weights, piece)`` with ``piece`` the interval index.
    """
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(n)
    lo, hi = np.asarray(edges[:-1]), np.asarray(edges[1:])
    half = 0.5 * (hi - lo)
    nodes = (0.5 * (hi + lo))[:, None] + half[:, None] * x[None, :]
    weights = half[:, None] * w[None, :]
    piece = np.repeat(np.arange(len(lo)), n)
    return nodes.reshape(-1), weights.reshape(-1), piece


def periodic_nodes(n, period):
    """Periodic trapezoid nodes and weights (normalized to total 1)."""
    import numpy as np

    return np.arange(n) * period / n, np.full(n, 1.0 / n)


def wedge_masses(mean, cov, edges):
    """Probability that a 2D Gaussian's position angle lies in each wedge.

    Position angle is measured from north (second axis) toward east (first
    axis). Each wedge must span less than pi. Uses the bivariate normal CDF in
    Owen's T form.

    Args:
        mean: Means ``(n, 2)`` (east, north).
        cov: Covariances ``(n, 2, 2)``.
        edges: Angle edges ``(B + 1,)``.

    Returns:
        Masses ``(n, B)``.
    """
    import numpy as np
    from scipy.special import ndtr, owens_t

    lo, hi = np.asarray(edges[:-1]), np.asarray(edges[1:])
    # Inward normals: sin(theta - lo) >= 0 and sin(hi - theta) >= 0.
    n1 = np.stack([np.cos(lo), -np.sin(lo)], axis=1)
    n2 = np.stack([-np.cos(hi), np.sin(hi)], axis=1)
    m1 = mean @ n1.T
    m2 = mean @ n2.T
    v1 = np.einsum("bi,nij,bj->nb", n1, cov, n1)
    v2 = np.einsum("bi,nij,bj->nb", n2, cov, n2)
    c12 = np.einsum("bi,nij,bj->nb", n1, cov, n2)
    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    rho = np.clip(c12 / (s1 * s2), -1.0 + 1e-15, 1.0 - 1e-15)
    h, k = m1 / s1, m2 / s2
    return bivariate_normal_cdf(h, k, rho, ndtr, owens_t)


def bivariate_normal_cdf(h, k, rho, ndtr, owens_t):
    """P(X <= h, Y <= k) for standard normals with correlation ``rho``."""
    import numpy as np

    tiny = 1e-300
    h = np.where(h == 0.0, tiny, h)
    k = np.where(k == 0.0, tiny, k)
    root = np.sqrt(1.0 - rho**2)
    a_h = (k - rho * h) / (h * root)
    a_k = (h - rho * k) / (k * root)
    beta = np.where(h * k > 0.0, 0.0, 0.5)
    out = 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, a_h) - owens_t(k, a_k) - beta
    return np.clip(out, 0.0, 1.0)


def mixture_quantile(weights, means, sds, level, iterations=200):
    """Quantile of a normalized Gaussian mixture by bisection."""
    import numpy as np
    from scipy.special import ndtr

    lo = float(np.min(means - 12.0 * sds))
    hi = float(np.max(means + 12.0 * sds))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if np.sum(weights * ndtr((mid - means) / sds)) < level:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(abs(lo), abs(hi), 1e-12):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# TI quadrature
# ---------------------------------------------------------------------------


def ti_quadrature(setup, counts, chunk=2**15):
    """Graded values of the TI case by quadrature at the given node counts.

    The window bound of a node is the chi-square (4 degrees of freedom) tail
    at ``r^2 / lambda_max`` for the largest ball around the conditional mean
    ``m`` whose semimajor axes stay inside the window. ``a(c)`` is the largest
    singular value of the TI matrix ``[[east_X, east_Y], [north_X,
    north_Y]]``, which is 1-Lipschitz in the Frobenius (vector) norm, so the
    ball of radius ``r`` maps into ``[a(m) - r, a(m) + r]``. The protocol's
    interval ``[(|m| - r) / sqrt 2, |m| + r]`` contains that one and gives a
    looser bound, reported as ``window_bound_norm_interval``.

    Returns:
        Dict with ``values`` (name to value), ``window_bound`` (weighted sum
        of the per-node window bounds) and ``nodes``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.special import ndtr  # noqa: F401  (used by wedge_masses)

    from photomancy.orbit.few_epoch import ti_semimajor_axis

    target = setup["target"]
    model = target.model
    edges = setup["edges"]
    times = jnp.asarray(setup["times"])
    obs = jnp.concatenate([jnp.asarray(setup["ra"]), jnp.asarray(setup["dec"])])
    prec_obs = jnp.full(obs.shape, 1.0 / setup["sigma"] ** 2)
    e_x, e_w, e_bin = gauss_pieces(edges["e"], counts["e_per_bin"])
    p_x, p_w, p_bin = gauss_pieces(edges["lnP"], counts["lnP_per_bin"])
    t_x, t_w = periodic_nodes(counts["tau"], 1.0)
    grid = np.stack(np.meshgrid(e_x, p_x, t_x, indexing="ij"), axis=-1).reshape(-1, 3)
    w_grid = (e_w[:, None, None] * p_w[None, :, None] * t_w[None, None, :]).reshape(-1)
    eb = np.broadcast_to(e_bin[:, None, None], (len(e_x), len(p_x), len(t_x))).reshape(
        -1
    )
    pb = np.broadcast_to(p_bin[None, :, None], (len(e_x), len(p_x), len(t_x))).reshape(
        -1
    )
    t_f = jnp.asarray([setup["t_f"]])
    two_pi = 2.0 * math.pi

    def design(u, at):
        # Columns: the forward model at each unit TI vector (predictions are
        # linear in the constants), stacked (east rows, then north rows).
        cols = []
        for j in range(4):
            ra, dec = target.predict(jnp.concatenate([u, jnp.eye(4)[j]]), at)
            cols.append(jnp.concatenate([ra, dec]))
        return jnp.stack(cols, axis=1)

    @jax.jit
    def node(u):
        # The Gaussian marginal and conditional of the TI constants from dense
        # normal equations, a code path independent of the updater's
        # Cholesky-based helper that the collapsed reference uses.
        h = design(u, times)
        precision = jnp.eye(4) / model.s_ti**2 + h.T @ (prec_obs[:, None] * h)
        b = h.T @ (prec_obs * obs)
        mean = jnp.linalg.solve(precision, b)
        cov_c = jnp.linalg.inv(precision)
        log_z = -0.5 * jnp.linalg.slogdet(precision)[1] + 0.5 * b @ mean
        hf = design(u, t_f)
        lam_max = jnp.linalg.eigvalsh(cov_c)[-1]
        norm_m = jnp.linalg.norm(mean)
        period = jnp.exp(u[1])
        a_lo = (model.mu_lo * (period / two_pi) ** 2) ** (1.0 / 3.0)
        a_hi = (model.mu_hi * (period / two_pi) ** 2) ** (1.0 / 3.0)
        a_m = ti_semimajor_axis(*mean)

        def tail(radius):
            x = jnp.where(radius > 0.0, radius**2 / lam_max, 0.0)
            return jnp.where(radius > 0.0, jnp.exp(-0.5 * x) * (1.0 + 0.5 * x), 1.0)

        loose = tail(jnp.minimum(norm_m - jnp.sqrt(2.0) * a_lo, a_hi - norm_m))
        sharp = tail(jnp.minimum(a_m - a_lo, a_hi - a_m))
        return log_z, hf @ mean, hf @ cov_c @ hf.T, sharp, loose

    batched = jax.jit(jax.vmap(node))
    parts = [
        batched(jnp.asarray(grid[i : i + chunk])) for i in range(0, len(grid), chunk)
    ]
    log_z, mu_f, cov_f, bound, loose = (
        np.concatenate([np.asarray(p[j]) for p in parts]) for j in range(5)
    )
    w = w_grid * np.exp(log_z - log_z.max())
    w /= w.sum()
    values = {}
    for b in range(8):
        values[f"lnP_bin{b}"] = float(w[pb == b].sum())
        values[f"e_bin{b}"] = float(w[eb == b].sum())
    masses = wedge_masses(mu_f, cov_f, edges["future_pa"])
    for b, m in enumerate(w @ masses):
        values[f"future_pa_bin{b}"] = float(m)
    for i, axis in enumerate(("east", "north")):
        sds = np.sqrt(cov_f[:, i, i])
        for end, level in (("lo", 0.05), ("hi", 0.95)):
            values[f"future_{axis}_{end}"] = mixture_quantile(w, mu_f[:, i], sds, level)
    return {
        "values": values,
        "window_bound": float(w @ bound),
        "window_bound_norm_interval": float(w @ loose),
        "angle_mass_total": float((w @ masses).sum()),
        "nodes": len(grid),
    }


# ---------------------------------------------------------------------------
# Physical quadrature
# ---------------------------------------------------------------------------


def _arc_half_mass(abs_x, kappa, gl_x, gl_w):
    """Mass of the kernel exp(K (cos x - 1)) on [0, |x|], |x| <= pi.

    With ``y = 2 sqrt(K) sin(x / 2)`` the mass is ``K^-1/2 int_0^y exp(-y^2 /
    2) g(y) dy`` with ``g = (1 - y^2 / (4 K))^-1/2``. The Gaussian part (``g =
    1``) is exact through the normal CDF; the correction ``exp(-y^2 / 2) (g -
    1)``, of order ``1 / (8 K)``, is smooth and integrated by a Gauss rule in
    ``y``. ``y`` is capped at ``Y_CAP``.
    """
    import jax.numpy as jnp
    from jax.scipy.special import ndtr

    y = jnp.minimum(2.0 * jnp.sqrt(kappa) * jnp.sin(0.5 * abs_x), Y_CAP)
    main = jnp.sqrt(2.0 * jnp.pi) * (0.5 - ndtr(-y))
    half = 0.5 * y
    yy = half[..., None] * (1.0 + gl_x)
    g_minus_1 = 1.0 / jnp.sqrt(1.0 - yy**2 / (4.0 * kappa[..., None])) - 1.0
    corr = half * jnp.sum(gl_w * jnp.exp(-0.5 * yy**2) * g_minus_1, axis=-1)
    return (main + corr) / jnp.sqrt(kappa)


def _wrap_angle(x):
    import jax.numpy as jnp

    return jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def _cumulative(x, kappa, j_pi, gl_x, gl_w):
    """Kernel mass on [-pi, x] for x in [-pi, pi)."""
    import jax.numpy as jnp

    return j_pi + jnp.sign(x) * _arc_half_mass(jnp.abs(x), kappa, gl_x, gl_w)


def _arc_mass(start, length, kappa, j_pi, gl_x, gl_w):
    """Kernel mass on the arc [start, start + length] (length in [0, 2 pi])."""
    import jax.numpy as jnp

    xa = _wrap_angle(start)
    xe = xa + length
    g_a = _cumulative(xa, kappa, j_pi, gl_x, gl_w)
    wraps = xe > jnp.pi
    g_e = _cumulative(jnp.where(wraps, xe - 2.0 * jnp.pi, xe), kappa, j_pi, gl_x, gl_w)
    return jnp.where(wraps, 2.0 * j_pi - g_a + g_e, g_e - g_a)


def physical_outer_nodes(setup, counts):
    """Outer nodes over (e, cos i, omega, M_ref) with normalized prior weights."""
    import numpy as np
    from scipy.special import betainc, betaincinv

    alpha, beta = 0.867, 3.03
    e_max = setup["e_max"]
    f_max = betainc(alpha, beta, e_max)
    u_edges = betainc(alpha, beta, setup["edges"]["e"]) / f_max
    u, u_w, e_bin = gauss_pieces(u_edges, counts["e_per_bin"])
    e = betaincinv(alpha, beta, u * f_max)
    c, c_w, _ = gauss_pieces(np.array([-1.0, 1.0]), counts["cos_i"])
    c_w = c_w / 2.0
    om, om_w = periodic_nodes(counts["omega"], 2.0 * math.pi)
    mr, mr_w = periodic_nodes(counts["m_ref"], 2.0 * math.pi)
    mesh = np.meshgrid(
        np.arange(len(e)),
        np.arange(len(c)),
        np.arange(len(om)),
        np.arange(len(mr)),
        indexing="ij",
    )
    i_e, i_c, i_o, i_m = (m.reshape(-1) for m in mesh)
    return {
        "e": e[i_e],
        "cos_i": c[i_c],
        "omega": om[i_o],
        "m_ref": mr[i_m],
        "weight": u_w[i_e] * c_w[i_c] * om_w[i_o] * mr_w[i_m],
        "e_bin": e_bin[i_e],
    }


def physical_nodes(setup, counts, chunk=4096):
    """Weighted (shape, rho) nodes of the physical case.

    Returns host arrays: ``weight`` (node mass, kernel total included),
    ``kappa``, ``offset`` (future position angle minus x), ``sep`` (future
    separation, arcsec), ``j_pi`` (half the kernel total), ``e_bin`` and
    ``lnp_bin``; also ``k_min`` and the node count.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from hwoutils.constants import G
    from orbix.equations.orbit import mean_motion

    from photomancy.orbit.forward import predict_relative_astrometry

    target = setup["target"]
    ms = float(target.Ms_kg)
    dist = float(target.dist_pc)
    mu = G * ms
    sigma = setup["sigma"]
    obs_e, obs_n = float(setup["ra"][0]), float(setup["dec"][0])
    rho_o = math.hypot(obs_e, obs_n)
    phi_o = math.atan2(obs_e, obs_n)
    ln_a_lo, ln_a_hi = (float(b) for b in target.ln_a_bounds())
    bw = (ln_a_hi - ln_a_lo) / 8.0
    w_lo = max(rho_o - RHO_WINDOW_SIGMA * sigma, 1e-12)
    w_hi = rho_o + RHO_WINDOW_SIGMA * sigma
    n_pieces = min(int(math.log(w_hi / w_lo) / bw) + 2, 8)
    z_min = (w_lo - rho_o) / sigma
    outer = physical_outer_nodes(setup, counts)
    gl_x, gl_w = (
        jnp.asarray(a) for a in np.polynomial.legendre.leggauss(counts["arc"])
    )
    r_x, r_w = np.polynomial.legendre.leggauss(counts["rho_per_piece"])
    r_x, r_w = jnp.asarray(r_x), jnp.asarray(r_w)
    t_f = setup["t_f"]

    def position(a, e, cos_i, cos_w, sin_w, m_ref, t):
        tp = -m_ref / mean_motion(a, mu)
        x, y = predict_relative_astrometry(
            jnp.reshape(t, (1,)), a, e, cos_i, 0.0, cos_w, sin_w, tp, ms, dist
        )
        return x[0], y[0]

    def one(e, cos_i, omega, m_ref, w_out):
        cos_w, sin_w = jnp.cos(omega), jnp.sin(omega)
        x0, y0 = position(1.0, e, cos_i, cos_w, sin_w, m_ref, 0.0)
        r0 = jnp.hypot(x0, y0)
        pa0 = jnp.arctan2(x0, y0)
        j0 = jnp.clip(jnp.floor((jnp.log(w_lo / r0) - ln_a_lo) / bw), 0, 7).astype(int)
        j = j0 + jnp.arange(n_pieces)
        valid = j <= 7
        a_lo_j = jnp.exp(ln_a_lo + bw * j)
        a_hi_j = jnp.exp(ln_a_lo + bw * (j + 1))
        # Pieces are clipped to the window |rho - rho_o| <= RHO_WINDOW_SIGMA sigma
        # (mass outside below 1e-18), which keeps K at its window minimum.
        z_lo = jnp.clip((r0 * a_lo_j - rho_o) / sigma, z_min, RHO_WINDOW_SIGMA)
        z_hi = jnp.clip((r0 * a_hi_j - rho_o) / sigma, z_min, RHO_WINDOW_SIGMA)
        u_lo = jax.scipy.special.ndtr(z_lo)
        u_hi = jax.scipy.special.ndtr(z_hi)
        half = jnp.where(valid, 0.5 * jnp.maximum(u_hi - u_lo, 0.0), 0.0)
        u = 0.5 * (u_hi + u_lo)[:, None] + half[:, None] * r_x
        u = jnp.clip(u, 1e-300, 1.0 - 1e-16)
        rho = rho_o + sigma * jax.scipy.special.ndtri(u)
        rho = jnp.where(half[:, None] > 0.0, rho, rho_o)
        a = rho / r0
        kappa = rho * rho_o / sigma**2
        xf, yf = jax.vmap(
            jax.vmap(lambda aa: position(aa, e, cos_i, cos_w, sin_w, m_ref, t_f))
        )(a)
        offset = phi_o + jnp.arctan2(xf, yf) - pa0
        sep = jnp.hypot(xf, yf)
        j_pi = _arc_half_mass(jnp.full_like(kappa, jnp.pi), kappa, gl_x, gl_w)
        base = w_out * half[:, None] * r_w * jnp.sqrt(2.0 * jnp.pi) * sigma / rho
        weight = base * 2.0 * j_pi
        lnp_bin = jnp.broadcast_to(j[:, None], rho.shape)
        return weight, kappa, offset, sep, j_pi, lnp_bin

    batched = jax.jit(jax.vmap(one))
    keys = ("e", "cos_i", "omega", "m_ref", "weight")
    out = {
        k: [] for k in ("weight", "kappa", "offset", "sep", "j_pi", "lnp_bin", "e_bin")
    }
    k_min = math.inf
    n_out = len(outer["e"])
    for i in range(0, n_out, chunk):
        sl = slice(i, i + chunk)
        args = [jnp.asarray(outer[k][sl]) for k in keys]
        res = [np.asarray(r).reshape(len(outer["e"][sl]), -1) for r in batched(*args)]
        keep = res[0] > 0.0
        for name, arr in zip(
            ("weight", "kappa", "offset", "sep", "j_pi", "lnp_bin"), res, strict=True
        ):
            out[name].append(arr[keep])
        out["e_bin"].append(
            np.broadcast_to(outer["e_bin"][sl][:, None], keep.shape)[keep]
        )
        if keep.any():
            k_min = min(k_min, float(res[1][keep].min()))
    nodes = {k: np.concatenate(v) for k, v in out.items()}
    nodes["k_min"] = k_min
    nodes["outer"] = n_out
    nodes["pieces"] = n_pieces
    return nodes


def physical_quadrature(setup, counts, chunk=2**18):
    """Graded values of the physical case by quadrature at the given counts."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    nodes = physical_nodes(setup, counts)
    if nodes["k_min"] < K_MIN:
        raise RuntimeError(f"kernel concentration {nodes['k_min']:.1f} below {K_MIN}")
    gl_x, gl_w = (
        jnp.asarray(a) for a in np.polynomial.legendre.leggauss(counts["arc"])
    )
    total = nodes["weight"].sum()
    values = {}
    for b in range(8):
        values[f"lnP_bin{b}"] = float(
            nodes["weight"][nodes["lnp_bin"] == b].sum() / total
        )
        values[f"e_bin{b}"] = float(nodes["weight"][nodes["e_bin"] == b].sum() / total)
    names = ("weight", "kappa", "offset", "sep", "j_pi")
    n = len(nodes["weight"])
    padded = {}
    size = -(-n // chunk) * chunk
    for k in names:
        arr = np.zeros(size)
        arr[:n] = nodes[k]
        if k in ("kappa", "sep", "j_pi"):
            arr[n:] = 1.0 if k != "kappa" else 1e6
        padded[k] = arr.reshape(-1, chunk)
    blocks = [
        {k: jnp.asarray(padded[k][i]) for k in names} for i in range(size // chunk)
    ]
    step = 2.0 * jnp.pi / 8.0

    @jax.jit
    def pa_bins(blk):
        base = blk["weight"] / (2.0 * blk["j_pi"])
        starts = jnp.arange(8) * step
        masses = jax.vmap(
            lambda s: _arc_mass(
                s - blk["offset"], step, blk["kappa"], blk["j_pi"], gl_x, gl_w
            )
        )(starts)
        return masses @ base

    pa = sum(np.asarray(pa_bins(blk)) for blk in blocks) / total
    for b in range(8):
        values[f"future_pa_bin{b}"] = float(pa[b])

    @jax.jit
    def cdf(blk, v_east, v_north):
        base = blk["weight"] / (2.0 * blk["j_pi"])
        kappa, j_pi, off, sep = blk["kappa"], blk["j_pi"], blk["offset"], blk["sep"]

        def east(v):
            r = jnp.clip(v / sep, -1.0, 1.0)
            asr = jnp.arcsin(r)
            mass = _arc_mass(
                jnp.pi - asr - off, jnp.pi + 2.0 * asr, kappa, j_pi, gl_x, gl_w
            )
            mass = jnp.where(v >= sep, 2.0 * j_pi, jnp.where(v <= -sep, 0.0, mass))
            return base @ mass

        def north(v):
            r = jnp.clip(v / sep, -1.0, 1.0)
            acr = jnp.arccos(r)
            mass = _arc_mass(
                acr - off, 2.0 * jnp.pi - 2.0 * acr, kappa, j_pi, gl_x, gl_w
            )
            mass = jnp.where(v >= sep, 2.0 * j_pi, jnp.where(v <= -sep, 0.0, mass))
            return base @ mass

        return jax.vmap(east)(v_east), jax.vmap(north)(v_north)

    def total_cdf(v):
        e_part, n_part = 0.0, 0.0
        for blk in blocks:
            a, b = cdf(blk, jnp.asarray(v[:2]), jnp.asarray(v[2:]))
            e_part = e_part + np.asarray(a)
            n_part = n_part + np.asarray(b)
        return np.concatenate([e_part, n_part]) / total

    sep_max = float(nodes["sep"].max())
    levels = np.array([0.05, 0.95, 0.05, 0.95])
    roots, iterations = illinois(total_cdf, levels, -sep_max, sep_max)
    for i, name in enumerate(
        ("future_east_lo", "future_east_hi", "future_north_lo", "future_north_hi")
    ):
        values[name] = float(roots[i])
    return {
        "values": values,
        "k_min": nodes["k_min"],
        "nodes": int(n),
        "outer_nodes": int(nodes["outer"]),
        "rho_pieces": int(nodes["pieces"]),
        "angle_mass_total": float(pa.sum()),
        "root_iterations": iterations,
    }


def illinois(fn, levels, lo, hi, tol=1e-13, max_iter=100):
    """Solve ``fn(v) = levels`` elementwise by the Illinois false-position rule.

    ``fn`` maps a vector of trial points to the vector of increasing functions
    evaluated there; the bracket ``[lo, hi]`` is shared at the start.
    """
    import numpy as np

    n = len(levels)
    a, b = np.full(n, lo), np.full(n, hi)
    fa, fb = fn(a) - levels, fn(b) - levels
    side = np.zeros(n)
    c = 0.5 * (a + b)
    for it in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        c = np.where(np.isfinite(c), c, 0.5 * (a + b))
        fc = fn(c) - levels
        left = np.sign(fc) == np.sign(fa)
        b_new = np.where(left, b, c)
        a_new = np.where(left, c, a)
        fb_new = np.where(left, fb, fc)
        fa_new = np.where(left, fc, fa)
        # Illinois: halve the retained end's value when the same side repeats.
        fb_new = np.where(left & (side == 1), 0.5 * fb_new, fb_new)
        fa_new = np.where(~left & (side == -1), 0.5 * fa_new, fa_new)
        side = np.where(left, 1, -1)
        a, b, fa, fb = a_new, b_new, fa_new, fb_new
        if np.all((np.abs(fc) < tol) | (np.abs(b - a) < tol * max(abs(lo), abs(hi)))):
            return c, it
    return c, max_iter


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def summary_values(summary):
    """The 28 graded values from a ``summarize_posterior`` result."""
    values = {}
    for q in ("lnP", "e", "future_pa"):
        for b, m in enumerate(summary["bins"][q]):
            values[f"{q}_bin{b}"] = float(m)
    for axis in ("east", "north"):
        lo, hi = summary["intervals"][f"future_{axis}"]["90"]
        values[f"future_{axis}_lo"] = lo
        values[f"future_{axis}_hi"] = hi
    return values


def reference_sample(protocol, setup, key, n_prop, chunk=2**16):
    """One run of the case's reference sampler: quantities and log weights."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    t_f = setup["t_f"]
    parts = {q: [] for q in panel.QUANTITIES}
    weights = []
    if setup["case"] != "physical_fixed":
        # The panel's TI reference (amendment A7): collapsed prior sampler.
        theta, lw = panel.ti_collapsed_reference(
            setup["target"], records(setup), key, n_prop, chunk
        )
        q, lw = panel.finite_chart_quantities(
            "ti_window", setup["target"], theta, lw, t_f, min(chunk, n_prop)
        )
        for name in panel.QUANTITIES:
            parts[name].append(q[name])
        weights.append(lw)
    else:
        import equinox as eqx

        from photomancy.orbit.ofti import ScaleAndRotate

        target = setup["target"]
        ms, dist = float(target.Ms_kg), float(target.dist_pc)
        scored = records(setup, mask=np.zeros(1, bool))
        log_p = tuple(math.log10(p) for p in setup["period_days"])
        ecc = protocol["model_pairs"]["physical_fixed"]["defaults"]["ecc_prior"]
        batch = min(chunk, n_prop)
        rows_fn = panel._physical_rows_fn()

        @eqx.filter_jit
        def score(kb):
            return ScaleAndRotate().sample_and_score(
                kb, batch, scored, 0, ms, dist, ecc, log_p, setup["e_max"]
            )

        times = jnp.asarray([0.0, t_f])
        for j in range(-(-n_prop // batch)):
            rows, ll = score(jax.random.fold_in(key, j))
            take = min(batch, n_prop - j * batch)
            e, ln_p, east, north = rows_fn(rows[:take], times, ms, dist)
            rho = jnp.hypot(east[:, 0], north[:, 0])
            lw = np.asarray(ll[:take] - 2.0 * jnp.log(rho))
            fin = np.isfinite(lw)
            for name, v in zip(
                panel.QUANTITIES, (e, ln_p, east[:, 1], north[:, 1]), strict=True
            ):
                parts[name].append(np.asarray(v)[fin])
            weights.append(lw[fin])
    return {k: np.concatenate(v) for k, v in parts.items()}, np.concatenate(weights)


def independent_sample(protocol, setup, key, n_prop, chunk=2**16):
    """The TI check-3 sampler (frequency mixture, joint weights) on the case."""
    config = protocol["reference"]["ti_window"]["independent"]
    theta, lw = panel.ti_frequency_reference(
        setup["target"], records(setup), key, n_prop, chunk, config
    )
    return panel.finite_chart_quantities(
        "ti_window", setup["target"], theta, lw, setup["t_f"], min(chunk, n_prop)
    )


def anchor_sample(setup, key, n_prop, chunk=2**16):
    """Brute-force prior sampler: the binding case's grading reference.

    Every coordinate, the TI constants included, comes from the prior, and
    the weight is the likelihood times the support (window) indicator,
    evaluated directly: no conditional Gaussian and no marginal likelihood,
    so it shares no algebra with the samplers it grades. Exact, and efficient
    only because the binding case's errors are large.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    target, data = setup["target"], records(setup)
    batch = min(chunk, n_prop)

    @eqx.filter_jit
    def run(kb):
        theta = target.sample_prior(kb, batch)
        inside = jnp.isfinite(jax.vmap(target.log_prior)(theta))
        ll = jax.vmap(lambda t: target.log_likelihood(t, data))(theta)
        return theta, jnp.where(inside, ll, -jnp.inf)

    thetas, lws = [], []
    for j in range(-(-n_prop // batch)):
        theta, lw = run(jax.random.fold_in(key, j))
        take = min(batch, n_prop - j * batch)
        thetas.append(np.asarray(theta)[:take])
        lws.append(np.asarray(lw)[:take])
    theta, lw = np.concatenate(thetas), np.concatenate(lws)
    return panel.finite_chart_quantities(
        "ti_window", target, theta, lw, setup["t_f"], batch
    )


def anchor_table(protocol, runs, t_crit=None):
    """Grading table of the binding case from its anchor runs.

    Each value's reference is the mean of the anchor runs and its error term
    ``eps_q`` is ``t* s / sqrt(R)``, the anchor's own Monte Carlo error at
    the exact checks' critical value.
    """
    import numpy as np

    t_crit = t_crit or panel.critical_values(protocol)["exact_check_t_critical"]
    table = {}
    for name in value_names():
        x = np.array([r["values"][name] for r in runs])
        sd = float(x.std(ddof=1)) if len(x) > 1 else math.inf
        table[name] = {
            "levels": [float(x.mean())],
            "eps_q": t_crit * sd / math.sqrt(len(x)),
            "converged": True,
        }
    return table


def updater_sample(protocol, setup, key, particles):
    """One run of the protocol's updater on the case (for information)."""
    import numpy as np

    from photomancy.orbit.updater import OrbitUpdater, UpdaterRecord

    kernel = protocol["updater_kernel"]

    def per_model(value):
        return value[setup["case"]] if isinstance(value, dict) else value

    updater = OrbitUpdater(
        n_particles=particles,
        n_rungs=protocol["updater_budget"]["rungs"],
        n_moves=protocol["updater_budget"]["moves_per_rung"],
        record_capacity=len(setup["times"]),
        low_ess_fraction=per_model(kernel["low_ess_fraction"]),
        target_cess_fraction=per_model(kernel["target_cess_fraction"]),
        proposal_scale=per_model(kernel["proposal_scale"]),
        alias_moves=per_model(kernel["alias_moves"]),
    )
    import jax

    belief = None
    for u in range(len(setup["times"])):
        record = UpdaterRecord(
            time=setup["times"][u],
            ra=setup["ra"][u],
            dec=setup["dec"][u],
            ra_err=setup["sigma"],
            dec_err=setup["sigma"],
            record_id=u + 1,
        )
        k_u = jax.random.fold_in(key, u)
        if u == 0:
            belief, _ = updater.init(setup["target"], record, k_u)
        else:
            belief, _ = updater.update(belief, record, k_u)
    q = panel.chart_quantities(
        setup["case"], setup["target"], belief.particles, setup["t_f"]
    )
    return q, np.asarray(belief.log_weights)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def plan(cases, levels, samplers, repeats):
    """Work units: one per case and level, and per case, sampler and repeat."""
    units = []
    for case in cases:
        for level in levels if case != "ti_binding" else ():
            units.append(
                {
                    "id": f"exact/{case}/quadrature/L{level}",
                    "case": case,
                    "kind": "quadrature",
                    "level": level,
                }
            )
        for sampler in (s for s in samplers if s in CASE_SAMPLERS[case]):
            for r in range(repeats):
                units.append(
                    {
                        "id": f"exact/{case}/{sampler}/r{r:02d}",
                        "case": case,
                        "kind": sampler,
                        "repeat": r,
                    }
                )
    return units


def _exact_definitions():
    """Functions and constants of this tool that determine an exact record."""
    return (
        value_names,
        level_counts,
        ti_reduced_model,
        case_setup,
        records,
        gauss_pieces,
        periodic_nodes,
        wedge_masses,
        bivariate_normal_cdf,
        mixture_quantile,
        ti_quadrature,
        _arc_half_mass,
        _wrap_angle,
        _cumulative,
        _arc_mass,
        physical_outer_nodes,
        physical_nodes,
        physical_quadrature,
        illinois,
        summary_values,
        reference_sample,
        independent_sample,
        anchor_sample,
        ti_binding_model,
        _key_slot,
        expected_budget,
        updater_sample,
        plan,
        ExactExecutor,
        ("constants", PHYSICAL_BASE, TI_BASE, TI_MASS_WINDOW_MSUN, Y_CAP, K_MIN),
        ("samplers", CASE_SAMPLERS, CASE_MASS_MODEL),
        ("rho_window_sigma", RHO_WINDOW_SIGMA),
    )


def exact_run_hash(protocol):
    """Hash of the protocol content and code that determine an exact record."""
    return panel.declaration_hash(
        {
            "panel": panel.run_hash(protocol),
            "exact": panel.source_fingerprint(_exact_definitions()),
        }
    )


def expected_budget(protocol, unit, proposals=None, particles=None):
    """The budget fields an exact record must carry (overrides for test runs)."""
    proposals = proposals or {}
    if unit["kind"] == "quadrature":
        base = TI_BASE if unit["case"] == "ti_window" else PHYSICAL_BASE
        return {"counts": level_counts(base, unit["level"])}
    pair = CASE_MASS_MODEL[unit["case"]]
    default = protocol["reference"][pair]["proposals_per_fit"]
    base = proposals.get(pair, default)
    if unit["kind"] == "reference":
        return {"proposals": base}
    if unit["kind"] == "independent":
        return {"proposals": 2 * base}
    if unit["kind"] == "anchor":
        binding = protocol["reference"]["exact_checks"]["binding_case"]
        return {"proposals": proposals.get("anchor", binding["anchor_proposals"])}
    return {"particles": particles or protocol["updater_budget"]["particles"]}


def unit_of(record_id):
    """The unit description encoded in an exact unit id."""
    _, case, kind, tail = record_id.split("/")
    if kind == "quadrature":
        return {"id": record_id, "case": case, "kind": kind, "level": int(tail[1:])}
    return {"id": record_id, "case": case, "kind": kind, "repeat": int(tail[1:])}


def _proposal_cap(protocol, case):
    return protocol["reference"][CASE_MASS_MODEL[case]]["max_proposals_per_fit"]


def budget_cap_violations(protocol, proposals=None):
    """Sampler budgets of the exact checks above their pair's proposal cap.

    Effective budgets, after the check-3 sampler's multiple of two.
    """
    out = []
    for case in CASES:
        for sampler in CASE_SAMPLERS[case]:
            if sampler == "updater":
                continue
            unit = {"case": case, "kind": sampler}
            n = expected_budget(protocol, unit, proposals)["proposals"]
            if n > _proposal_cap(protocol, case):
                out.append(
                    f"{case} {sampler}: {n} > cap {_proposal_cap(protocol, case)}"
                )
    return out


def record_valid(protocol, record, run_hash, proposals=None, particles=None):
    """Whether an exact record matches the current code, protocol and budget.

    A sampler record above its pair's proposal cap is never valid.
    """
    if record.get("run_hash") != run_hash:
        return False
    try:
        unit = unit_of(record["id"])
    except (KeyError, ValueError):
        return False
    if record.get("proposals", 0) > _proposal_cap(protocol, unit["case"]):
        return False
    budget = expected_budget(protocol, unit, proposals, particles)
    return all(record.get(k) == v for k, v in budget.items())


class ExactExecutor:
    """Runs exact-check units; caches each case's setup."""

    def __init__(self, protocol, proposals=None, particles=None):
        """Store the protocol and optional budget overrides."""
        panel.enable_x64()
        self.protocol = protocol
        self.proposals = proposals or {}
        self.particles = particles or protocol["updater_budget"]["particles"]
        self._setups = {}
        self.run_hash = exact_run_hash(protocol)

    def setup(self, case):
        """The case setup (truth, records, edges), built once."""
        if case not in self._setups:
            self._setups[case] = case_setup(self.protocol, case)
        return self._setups[case]

    def __call__(self, unit):
        """Run one unit and return its record."""
        import jax

        start = time.perf_counter()
        setup = self.setup(unit["case"])
        if unit["kind"] == "quadrature":
            base = TI_BASE if unit["case"] == "ti_window" else PHYSICAL_BASE
            counts = level_counts(base, unit["level"])
            fn = ti_quadrature if unit["case"] == "ti_window" else physical_quadrature
            record = {"kind": "quadrature", "level": unit["level"], "counts": counts}
            record.update(fn(setup, counts))
        else:
            key = panel.draw_key(
                self.protocol,
                "exact",
                *_key_slot(unit["case"]),
                0,
                "exact_run",
                unit["repeat"],
            )
            budget = expected_budget(
                self.protocol, unit, self.proposals, self.particles
            )
            if unit["kind"] == "reference":
                q, lw = reference_sample(self.protocol, setup, key, budget["proposals"])
            elif unit["kind"] == "independent":
                q, lw = independent_sample(
                    self.protocol,
                    setup,
                    jax.random.fold_in(key, 2),
                    budget["proposals"],
                )
            elif unit["kind"] == "anchor":
                q, lw = anchor_sample(
                    setup, jax.random.fold_in(key, 3), budget["proposals"]
                )
            else:
                q, lw = updater_sample(
                    self.protocol, setup, jax.random.fold_in(key, 1), self.particles
                )
                budget = {"particles": self.particles}
            summary = panel.summarize_posterior(
                q, lw, setup["edges"], panel.visibility_mas(self.protocol)
            )
            record = {
                "kind": unit["kind"],
                "repeat": unit["repeat"],
                "kish_ess": summary["kish_ess"],
                "values": summary_values(summary),
                **budget,
            }
        record["seconds"] = time.perf_counter() - start
        record["run_hash"] = self.run_hash
        return record


# ---------------------------------------------------------------------------
# Comparison and verdicts
# ---------------------------------------------------------------------------


def refinement(levels_values):
    """Refinement table entry of each value from three levels.

    Args:
        levels_values: List of three dicts (levels 0, 1, 2), name to value.

    Returns:
        Dict name to ``{"levels", "eps_q", "converged"}``: ``eps_q`` is twice
        the change between the two finest levels; ``converged`` requires that
        change to be smaller than the previous one (or below 1e-14).
    """
    out = {}
    for name in value_names():
        v0, v1, v2 = (lv[name] for lv in levels_values)
        d10, d21 = abs(v1 - v0), abs(v2 - v1)
        out[name] = {
            "levels": [v0, v1, v2],
            "eps_q": 2.0 * d21,
            "converged": bool(d21 < d10 or d21 <= 1e-14),
        }
    return out


def compare(protocol, table, runs, t_crit=None):
    """Per-value comparison of sampler runs with the finest quadrature.

    A value passes when ``|mean - x_q| <= eps_q + max(t* s / sqrt(R), f)``,
    ``f`` the bin-mass floor for bin masses and 0 for end points.
    """
    import numpy as np

    ex = protocol["reference"]["exact_checks"]
    t_crit = t_crit or panel.critical_values(protocol)["exact_check_t_critical"]
    rows = {}
    for name in value_names():
        x = np.array([r["values"][name] for r in runs])
        mean = float(x.mean())
        sd = float(x.std(ddof=1)) if len(x) > 1 else math.inf
        floor = ex["bin_mass_floor"] if "_bin" in name else 0.0
        x_q = table[name]["levels"][-1]
        tol = table[name]["eps_q"] + max(t_crit * sd / math.sqrt(len(x)), floor)
        rows[name] = {
            "mean": mean,
            "sd": sd,
            "quadrature": x_q,
            "deviation": abs(mean - x_q),
            "tolerance": tol,
            "pass": bool(abs(mean - x_q) <= tol),
        }
    return rows


def _manifest_commits(out_dir):
    """Distinct code and protocol commits recorded by the run's manifests."""
    code, proto = set(), set()
    for path in sorted(Path(out_dir).glob("manifest-*.json")):
        for entry in json.loads(path.read_text()).get("invocations", []):
            code.add(str(entry.get("code_commit")))
            proto.add(str(entry.get("protocol_commit")))
    return sorted(code), sorted(proto)


def report(protocol, out_dir, cases=CASES, levels=(0, 1, 2)):
    """Assemble the exact-check results and verdicts from a run directory.

    Only records run under the current code and protocol content
    (``exact_run_hash``) at the protocol budgets count as evidence; a case
    with a duplicate record, a stale or off-budget record, levels that are
    not three consecutive refinements, or a missing reference repeat cannot
    pass. The provenance block reports the run hash the records must carry
    and the commits the run's manifests recorded.
    """
    records, duplicates = panel.load_units(out_dir)
    current = exact_run_hash(protocol)
    repeats = protocol["reference"]["exact_checks"]["sampler_repeats"]
    t_crit = panel.critical_values(protocol)["exact_check_t_critical"]
    code_commits, protocol_commits = _manifest_commits(out_dir)
    result = {
        "provenance": {
            "run_hash": current,
            "manifest_code_commits": code_commits,
            "manifest_protocol_commits": protocol_commits,
        },
        "levels": list(levels),
        "cases": {},
        "verdicts": {},
    }
    levels = [int(lv) for lv in levels]
    consecutive = len(levels) == 3 and levels == list(range(levels[0], levels[0] + 3))
    over_cap = budget_cap_violations(protocol)
    for case in cases:
        reasons = []
        if over_cap:
            reasons.append(
                "protocol budgets exceed the proposal cap (the method needs an "
                f"amendment, not another doubling): {over_cap}"
            )
        if not consecutive and case != "ti_binding":
            reasons.append(f"levels {levels} are not three consecutive refinements")
        mine = {i: r for i, r in records.items() if i.split("/")[1] == case}
        dup = sorted({i for i in duplicates if i.split("/")[1] == case})
        if dup:
            reasons.append(f"{len(dup)} duplicate records: {dup[:3]}")
        invalid = sorted(
            i for i, r in mine.items() if not record_valid(protocol, r, current)
        )
        if invalid:
            reasons.append(
                f"{len(invalid)} records from other code, protocol content or "
                f"budgets: {invalid[:3]}"
            )
        units = {i: r for i, r in mine.items() if i not in invalid}
        has_quad = case != "ti_binding"
        quad_levels = levels if has_quad else []
        quad = [units.get(f"exact/{case}/quadrature/L{lv}") for lv in quad_levels]
        entry = {"quadrature": {}}
        for lv, rec in zip(quad_levels, quad, strict=True):
            if rec is None:
                reasons.append(f"quadrature level {lv} missing")
            else:
                entry["quadrature"][f"L{lv}"] = {
                    k: v for k, v in rec.items() if k != "id"
                }
        if case == "ti_window" and quad and quad[-1] is not None:
            if quad[-1]["window_bound"] >= WINDOW_BOUND_MAX:
                reasons.append(
                    "window bound {:.2e} not below {}".format(
                        quad[-1]["window_bound"], WINDOW_BOUND_MAX
                    )
                )
        table = None
        if not has_quad:
            anchors = [
                units.get(f"exact/{case}/anchor/r{r:02d}") for r in range(repeats)
            ]
            anchors = [a for a in anchors if a is not None]
            entry["anchor_runs"] = len(anchors)
            if len(anchors) < repeats:
                reasons.append(f"anchor repeats {len(anchors)} of {repeats}")
            else:
                entry["anchor_kish_ess"] = [a["kish_ess"] for a in anchors]
                table = anchor_table(protocol, anchors, t_crit)
                entry["refinement"] = table
        elif consecutive and all(q is not None for q in quad):
            table = refinement([q["values"] for q in quad])
            entry["refinement"] = table
            unconverged = [n for n, t in table.items() if not t["converged"]]
            if unconverged:
                reasons.append(f"refinement not converging for {unconverged}")
        verdict_fail = False
        for sampler in (s for s in CASE_SAMPLERS[case] if s != "anchor"):
            runs = [
                units.get(f"exact/{case}/{sampler}/r{r:02d}") for r in range(repeats)
            ]
            have = [r for r in runs if r is not None]
            entry[f"{sampler}_runs"] = len(have)
            if have:
                entry[f"{sampler}_kish_ess"] = [r["kish_ess"] for r in have]
            if table is not None and len(have) >= 2:
                rows = compare(protocol, table, have, t_crit)
                entry[f"{sampler}_comparison"] = rows
                failed = [n for n, r in rows.items() if not r["pass"]]
                entry[f"{sampler}_failed"] = failed
                if sampler in GRADED and failed and len(have) == repeats:
                    verdict_fail = True
            if sampler in GRADED and len(have) < repeats:
                reasons.append(f"{sampler} repeats {len(have)} of {repeats}")
        if verdict_fail and not reasons:
            verdict = "fail"
        elif reasons:
            verdict = "incomplete"
        else:
            verdict = "pass"
        entry["reasons"] = reasons
        result["cases"][case] = entry
        result["verdicts"][case] = verdict
    result["duplicates"] = len(duplicates)
    return result


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, default=panel.PROTOCOL)
    parser.add_argument("--out", type=Path, default=None, help="run directory")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--samplers", default=",".join(SAMPLERS))
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--ti-proposals", type=int, default=None, help="test override")
    parser.add_argument(
        "--physical-proposals", type=int, default=None, help="test override"
    )
    parser.add_argument("--particles", type=int, default=None, help="test override")
    parser.add_argument("--report", type=Path, default=None, help="assemble results")
    parser.add_argument(
        "--write", type=Path, default=None, help="write the report JSON"
    )
    args = parser.parse_args(argv)
    protocol = panel.load_protocol(args.protocol)
    cases = args.cases.split(",")
    if args.out is not None:
        repeats = (
            args.repeats or protocol["reference"]["exact_checks"]["sampler_repeats"]
        )
        units = plan(
            cases,
            [int(x) for x in args.levels.split(",")],
            args.samplers.split(","),
            repeats,
        )
        proposals = {}
        if args.ti_proposals:
            proposals["ti_window"] = args.ti_proposals
        if args.physical_proposals:
            proposals["physical_fixed"] = args.physical_proposals
        over_cap = budget_cap_violations(protocol, proposals)
        if over_cap:
            raise SystemExit(
                "refusing to run: sampler budgets exceed the proposal cap "
                f"(amend the method instead): {over_cap}"
            )
        shard = panel.parse_shard(args.shard)
        executor = ExactExecutor(protocol, proposals, args.particles)
        entry = panel.manifest_entry(
            protocol,
            None,
            {
                "proposals": proposals or "protocol",
                "particles": args.particles or "protocol",
            },
        )
        entry["exact_run_hash"] = executor.run_hash
        entry["finished"] = None

        def compatible(record):
            return record_valid(
                protocol, record, executor.run_hash, proposals, args.particles
            )

        try:
            counts = panel.run_units(
                units,
                args.out,
                shard,
                panel._ManifestFirst(executor, args.out, shard, entry),
                log=lambda m: print(m, flush=True),
                compatible=compatible,
            )
        except panel.IncompatibleRunError as err:
            raise SystemExit(f"refusing to resume: {err}") from None
        import datetime

        entry.update(counts, finished=datetime.datetime.now(datetime.UTC).isoformat())
        panel.write_manifest(args.out, shard, entry)
        print(json.dumps(counts))
    if args.report is not None:
        levels = [int(x) for x in args.levels.split(",")]
        result = report(protocol, args.report, cases, levels)
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.write is not None:
            panel.atomic_write_text(args.write, text + "\n")
        print(
            json.dumps(
                {
                    "verdicts": result["verdicts"],
                    "reasons": {c: e["reasons"] for c, e in result["cases"].items()},
                },
                indent=2,
            )
        )
    if args.out is None and args.report is None:
        parser.error("nothing to do: pass --out or --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
