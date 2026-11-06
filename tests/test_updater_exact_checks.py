"""Tests for the updater exact-check runner (tools/updater_exact_checks.py)."""

import importlib.util
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

TOOL = Path(__file__).parents[1] / "tools" / "updater_exact_checks.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("updater_exact_checks", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load_tool()


def test_wedge_masses_match_monte_carlo_and_angle_convention():
    """Gaussian wedge masses equal sampled position-angle frequencies.

    Three 2D Gaussians (one straddling the origin, two offset and correlated)
    are binned into 8 position-angle wedges measured east of north. With
    N = 40,000 draws a frequency has standard error at most 0.0025, so every
    mass must lie within 0.012 of its frequency and each row must sum to 1.
    A mutant that swaps east and north (angle measured north of east) must
    miss by more than 0.1. Fails if the wedge normals, the Owen's T form of
    the bivariate normal CDF or the angle convention is wrong.
    """
    rng = np.random.default_rng(0)
    means = np.array([[0.1, -0.2], [1.5, 0.4], [-0.3, -2.0]])
    covs = np.array(
        [[[1.0, 0.3], [0.3, 0.5]], [[0.2, -0.1], [-0.1, 0.3]], [[0.5, 0.0], [0.0, 0.1]]]
    )
    edges = np.linspace(0.0, 2.0 * math.pi, 9)
    masses = ex.wedge_masses(means, covs, edges)
    np.testing.assert_allclose(masses.sum(axis=1), 1.0, atol=1e-12)
    n = 40_000
    swapped = ex.wedge_masses(means[:, ::-1], covs[:, ::-1, ::-1], edges)
    worst_swap = 0.0
    for i in range(3):
        x = rng.multivariate_normal(means[i], covs[i], n)
        pa = np.mod(np.arctan2(x[:, 0], x[:, 1]), 2.0 * math.pi)
        freq = np.histogram(pa, edges)[0] / n
        assert np.max(np.abs(masses[i] - freq)) < 0.012
        worst_swap = max(worst_swap, np.max(np.abs(swapped[i] - freq)))
    assert worst_swap > 0.1


def test_arc_kernel_total_and_arcs_match_direct_integration():
    """The transformed arc rule reproduces the von Mises kernel integrals.

    The total of exp(K (cos x - 1)) over the circle is 2 pi I0(K) e^-K; at
    K = 150, 800 and 5,000 the rule's total must match to 1e-10 relative at
    16 nodes, and arcs that wrap through pi or cover the peak must match
    adaptive quadrature (scipy quad) to 1e-9 relative to the total. Fails if the
    Phi(y) change of variables, its Jacobian sqrt(2 pi / K) or the arc
    wrapping is wrong.
    """
    from scipy.integrate import quad
    from scipy.special import i0e

    gl_x, gl_w = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(16))
    for kappa in (150.0, 800.0, 5000.0):
        k = jnp.asarray([kappa])
        j_pi = ex._arc_half_mass(jnp.asarray([math.pi]), k, gl_x, gl_w)
        total = 2.0 * float(j_pi[0])
        exact = 2.0 * math.pi * i0e(kappa)
        assert abs(total / exact - 1.0) < 1e-10
        for start, length in ((-0.05, 0.12), (2.9, 1.0), (-math.pi, 0.4), (0.01, 3.0)):
            arc = float(
                ex._arc_mass(jnp.asarray([start]), length, k, j_pi, gl_x, gl_w)[0]
            )
            direct = quad(
                lambda x, k=kappa: math.exp(k * (math.cos(x) - 1.0)),
                start,
                start + length,
                points=[0.0, 2.0 * math.pi] if start + length > math.pi else [0.0],
                epsabs=1e-15,
                epsrel=1e-13,
                limit=500,
            )[0]
            assert abs(arc - direct) < 1e-9 * exact


def _collapsed_reference(setup, key, n):
    """Exact collapsed importance draws of the TI case: summary values."""
    from photomancy.orbit.updater import _linear_gaussian

    panel = ex.panel
    target = setup["target"]
    data = ex.records(setup)
    ones = jnp.ones(len(setup["times"]))

    @jax.jit
    def draw(key):
        k_u, k_c = jax.random.split(key)
        u = target.sample_prior(k_u, n)
        z = jax.random.normal(k_c, (n, 4))

        def one(theta, zz):
            mean, chol, log_z, _, _ = _linear_gaussian(target, theta, data, ones)
            c = mean + jax.scipy.linalg.solve_triangular(chol.T, zz, lower=False)
            theta = theta.at[3:].set(c)
            ok = jnp.isfinite(target.log_prior(theta))
            return theta, jnp.where(ok, log_z, -jnp.inf)

        return jax.vmap(one)(u, z)

    theta, lw = (np.asarray(a) for a in draw(key))
    q = panel.chart_quantities("ti_window", target, theta, setup["t_f"])
    summary = panel.summarize_posterior(
        q, lw, setup["edges"], panel.visibility_mas(panel.load_protocol())
    )
    return ex.summary_values(summary)


@pytest.mark.medium
def test_ti_quadrature_matches_an_independent_exact_sampler():
    """Level-0 TI quadrature agrees with collapsed importance sampling.

    The sampler draws (e, ln P, tau) from the prior, the TI constants from
    their conditional Gaussian, and weights by the marginal likelihood times
    the window indicator (exact, independent of the quadrature's Gaussian
    algebra). Four runs of 2^16 draws give a mean and standard deviation per
    value; every one of the 28 values must agree within 6 standard errors of
    the mean plus twice the level-0 to level-1 quadrature change plus a floor
    (1e-4 for masses, 1e-5 arcsec for end points). Bin masses must sum to 1.
    Fails if the quadrature drops or misweights the marginal likelihood, uses
    the wrong future-position covariance, or mislabels a bin or end point.
    A 64-particle updater unit must also run through the executor with the
    protocol's per-mass-model kernel values (fails if a per-model kernel
    entry reaches the updater unresolved).
    """
    protocol = ex.panel.load_protocol()
    unit = {"id": "u", "case": "ti_window", "kind": "updater", "repeat": 0}
    rec = ex.ExactExecutor(protocol, particles=64)(unit)
    assert rec["particles"] == 64
    assert abs(sum(rec["values"][f"lnP_bin{b}"] for b in range(8)) - 1.0) < 1e-9
    setup = ex.case_setup(protocol, "ti_window")
    q0 = ex.ti_quadrature(setup, ex.level_counts(ex.TI_BASE, 0))["values"]
    q1 = ex.ti_quadrature(setup, ex.level_counts(ex.TI_BASE, 1))["values"]
    for quantity in ("lnP", "e", "future_pa"):
        assert abs(sum(q0[f"{quantity}_bin{b}"] for b in range(8)) - 1.0) < 1e-12
    runs = [_collapsed_reference(setup, jax.random.key(i), 2**16) for i in range(4)]
    for name in ex.value_names():
        x = np.array([r[name] for r in runs])
        floor = 1e-4 if "_bin" in name else 1e-5
        tol = 6.0 * x.std(ddof=1) / 2.0 + 2.0 * abs(q1[name] - q0[name]) + floor
        assert abs(x.mean() - q1[name]) < tol, (name, x.mean(), q1[name], tol)


@pytest.mark.medium
def test_physical_quadrature_matches_the_reference_sampler_coarsely():
    """Level-0 physical quadrature agrees with the exact-weight reference.

    The reference (scale-and-rotate proposal, weight rho^-2 on the support)
    runs once at 2^18 proposals. Level 0 is coarse (one Gauss node per e bin,
    eight per angle), so the tolerances catch convention errors, not
    accuracy: every bin mass within 0.05, end points within 2 mas, and the
    masses of each quantity summing to 1. Fails if the rotation sign, the
    position-angle offset of the future epoch, the east and north arcs, or
    the e-coordinate transform is wrong (each moves whole bins or end points
    by more).
    """
    protocol = ex.panel.load_protocol()
    setup = ex.case_setup(protocol, "physical_fixed")
    quad = ex.physical_quadrature(setup, ex.level_counts(ex.PHYSICAL_BASE, 0))
    values = quad["values"]
    for quantity in ("lnP", "e", "future_pa"):
        assert abs(sum(values[f"{quantity}_bin{b}"] for b in range(8)) - 1.0) < 1e-10
    key = ex.panel.draw_key(protocol, "exact", "physical_fixed", 0, 0, "exact_run", 0)
    q, lw = ex.reference_sample(protocol, setup, key, 2**18)
    summary = ex.panel.summarize_posterior(
        q, lw, setup["edges"], ex.panel.visibility_mas(protocol)
    )
    sampled = ex.summary_values(summary)
    for name in ex.value_names():
        tol = 0.05 if "_bin" in name else 0.002
        assert abs(sampled[name] - values[name]) < tol, (
            name,
            sampled[name],
            values[name],
        )


def _write_units(
    tool, out_dir, offset=0.0, drop=None, stall=False, tamper=None, protocol=None
):
    """Synthetic exact records carrying the current run hash and budgets.

    ``tamper`` corrupts the evidence: ``"budget"`` gives one reference repeat
    one proposal, ``"stale"`` gives one an old run hash, ``"duplicate"``
    writes one repeat twice.
    """
    panel = tool.panel
    protocol = protocol or panel.load_protocol()
    current = tool.exact_run_hash(protocol)
    rng = np.random.default_rng(0)
    names = tool.value_names()
    truth = {
        n: (0.125 if "_bin" in n else 0.05 * (i - 20)) for i, n in enumerate(names)
    }
    path = Path(out_dir) / "units-0of1.jsonl"
    for case in tool.CASES:
        for level in range(3):
            uid = f"exact/{case}/quadrature/L{level}"
            if uid == drop:
                continue
            # Converging: offsets 1e-3, 1.1e-3, 1.11e-3; stalled: 0, 1e-4, 2e-4.
            shift = sum(1e-3 * 0.1**i for i in range(level + 1))
            if stall:
                shift = 1e-4 * level
            values = {n: v + shift for n, v in truth.items()}
            rec = {
                "id": uid,
                "kind": "quadrature",
                "level": level,
                "values": values,
                "run_hash": current,
                **tool.expected_budget(protocol, tool.unit_of(uid)),
            }
            if case == "ti_window":
                rec["window_bound"] = 1e-12
            panel.append_record(path, rec)
        for r in range(16):
            values = {
                n: v + 1.11e-3 + 1e-3 * rng.standard_normal() for n, v in truth.items()
            }
            if case == "physical_fixed":
                values["e_bin3"] += offset
            uid = f"exact/{case}/reference/r{r:02d}"
            rec = {
                "id": uid,
                "kind": "reference",
                "repeat": r,
                "kish_ess": 1e4,
                "values": values,
                "run_hash": current,
                **tool.expected_budget(protocol, tool.unit_of(uid)),
            }
            if r == 3 and tamper == "budget":
                rec["proposals"] = 1
            if r == 3 and tamper == "stale":
                rec["run_hash"] = "0" * 16
            panel.append_record(path, rec)
            if r == 3 and tamper == "duplicate":
                panel.append_record(path, rec)


def test_report_flags_offsets_and_missing_units(tmp_path):
    """The verdicts pass a correct sampler and refuse incomplete evidence.

    Synthetic records: quadrature changes 1e-4 then 1e-5 between levels,
    16 sampler repeats with 1e-3 Gaussian scatter around the quadrature.
    Fails if that sampler does not pass, if a planted offset of 0.01 in one
    bin mass (five times the floor) does not fail, if a missing quadrature
    level does not make the verdict incomplete, or if a refinement whose
    finest change is not smaller than the previous one is accepted. Evidence
    with one record at one proposal, one from another code or protocol
    content (run hash), or one written twice must not pass; nor may levels
    other than three consecutive refinements. The freeze re-assesses the
    exact checks from their run directory: a passing directory adds no exact
    reason, a tampered one does, and a results file is refused.
    """
    protocol = ex.panel.load_protocol()
    cases = {
        "good": {},
        "offset": {"offset": 0.01},
        "missing": {"drop": "exact/ti_window/quadrature/L1"},
        "stall": {"stall": True},
        "budget": {"tamper": "budget"},
        "stale": {"tamper": "stale"},
        "duplicate": {"tamper": "duplicate"},
    }
    verdicts = {}
    for label, kwargs in cases.items():
        out = tmp_path / label
        out.mkdir()
        _write_units(ex, out, protocol=protocol, **kwargs)
        verdicts[label] = ex.report(protocol, out)["verdicts"]
    assert verdicts["good"] == {"ti_window": "pass", "physical_fixed": "pass"}
    assert verdicts["offset"] == {"ti_window": "pass", "physical_fixed": "fail"}
    assert verdicts["missing"]["ti_window"] == "incomplete"
    assert verdicts["stall"] == {
        "ti_window": "incomplete",
        "physical_fixed": "incomplete",
    }
    for label in ("budget", "stale", "duplicate"):
        assert verdicts[label]["ti_window"] == "incomplete", label
    for levels in ((1, 2, 3), (0, 2, 4)):
        assert ex.report(protocol, tmp_path / "good", levels=levels)["verdicts"] == {
            "ti_window": "incomplete",
            "physical_fixed": "incomplete",
        }

    dev = tmp_path / "dev"
    dev.mkdir()

    def exact_reasons(exact_path):
        report = ex.panel.freeze_report(protocol, dev, exact_path)
        return [r for r in report["reasons"] if "exact" in r]

    assert exact_reasons(tmp_path / "good") == []
    assert any(
        "exact checks: incomplete" in r for r in exact_reasons(tmp_path / "stale")
    )
    results = tmp_path / "exact-checks.json"
    results.write_text('{"verdicts": {"ti_window": "pass", "physical_fixed": "pass"}}')
    assert any("not a run directory" in r for r in exact_reasons(results))
