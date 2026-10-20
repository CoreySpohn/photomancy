"""Exact few-epoch fitters: convention locks and grid-reference regression."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from photomancy.orbit.data import RelativeAstromData  # noqa: E402
from photomancy.orbit.few_epoch import (  # noqa: E402
    MU_SUN,
    TI_PARAM_NAMES,
    TIBasisModel,
    admissible_region_fit,
    lambert_depth_fit,
    state_to_theta,
    ti_semimajor_axis,
)

DIST_PC = 10.0
SIGMA_AU = 0.02
TIMES = jnp.array([0.0, 180.0, 400.0])

MODEL = TIBasisModel(lnP_lo=float(np.log(80.0)), lnP_hi=float(np.log(3000.0)))

TRUTH = dict(a=1.0, e=0.25, inc=0.65, W=1.1, w=0.8, tau=0.30, mu=MU_SUN)


def _newton_E(M, e, iters=60):
    """Independent Newton Kepler solve for the test truth."""
    E = M
    for _ in range(iters):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
    return E


def _truth_state(t):
    """Truth (r, v, theta) built from rotations, independent of the module."""
    a, e, inc, W, w = TRUTH["a"], TRUTH["e"], TRUTH["inc"], TRUTH["W"], TRUTH["w"]
    mu, tau = TRUTH["mu"], TRUTH["tau"]
    P = 2.0 * jnp.pi * jnp.sqrt(a**3 / mu)

    def rz(th):
        c, s = jnp.cos(th), jnp.sin(th)
        return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(th):
        c, s = jnp.cos(th), jnp.sin(th)
        return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    R = rz(W) @ rx(inc) @ rz(w)
    p_hat, q_hat = R[:, 0], R[:, 1]
    M = 2.0 * jnp.pi * ((t - MODEL.t_ref) / P - tau)
    E = _newton_E(M, e)
    X = jnp.cos(E) - e
    Y = jnp.sqrt(1.0 - e**2) * jnp.sin(E)
    r = a * (X * p_hat + Y * q_hat)
    v = (
        (2.0 * jnp.pi * a / P)
        / (1.0 - e * jnp.cos(E))
        * (-jnp.sin(E) * p_hat + jnp.sqrt(1.0 - e**2) * jnp.cos(E) * q_hat)
    )
    theta = jnp.array(
        [e, jnp.log(P), tau, a * p_hat[0], a * q_hat[0], a * p_hat[1], a * q_hat[1]]
    )
    return r, v, theta


def _make_data(n_epochs, seed=11, sigma_au=SIGMA_AU):
    """Noisy RelativeAstromData (arcsec) on the first n_epochs of TIMES."""
    rng = np.random.default_rng(seed)
    sel = {1: [0], 2: [0, 2], 3: [0, 1, 2]}[n_epochs]
    times = TIMES[np.array(sel)]
    _, _, theta = _truth_state(0.0)
    x, y = MODEL.forward_au(theta, times)
    x = np.asarray(x) + sigma_au * rng.standard_normal(len(sel))
    y = np.asarray(y) + sigma_au * rng.standard_normal(len(sel))
    n = len(sel)
    return RelativeAstromData(
        times=times,
        ra=jnp.asarray(x) / DIST_PC,
        dec=jnp.asarray(y) / DIST_PC,
        ra_err=jnp.full(n, sigma_au / DIST_PC),
        dec_err=jnp.full(n, sigma_au / DIST_PC),
        corr=jnp.zeros(n),
        planet_id=jnp.zeros(n, dtype=int),
        is_valid=jnp.ones(n, dtype=bool),
    )


def _grid_reference(data, n_e=24, n_lnP=48, n_tau=32, n_mc=64):
    """Exact RBTI grid posterior means/stds for (e, lnP, tau) under MODEL."""
    x_au = data.ra * DIST_PC
    y_au = data.dec * DIST_PC
    d = jnp.concatenate([x_au, y_au])
    times = data.times
    n_t = times.shape[0]
    s2 = MODEL.s_ti**2
    eps = jax.random.normal(jax.random.PRNGKey(3), (n_mc, 4))

    e_ax = jnp.linspace(1e-4, MODEL.e_max - 1e-4, n_e)
    lnP_ax = jnp.linspace(MODEL.lnP_lo, MODEL.lnP_hi, n_lnP)
    tau_ax = jnp.linspace(0.0, 1.0, n_tau, endpoint=False) + 0.5 / n_tau

    def cell(e, lnP, tau):
        P = jnp.exp(lnP)
        M = 2.0 * jnp.pi * ((times - MODEL.t_ref) / P - tau)
        E = _newton_E(M, e)
        X = jnp.cos(E) - e
        Y = jnp.sqrt(1.0 - e**2) * jnp.sin(E)
        Z = jnp.zeros_like(X)
        Phi = jnp.concatenate(
            [jnp.stack([X, Y, Z, Z], axis=1), jnp.stack([Z, Z, X, Y], axis=1)]
        )
        Sig = s2 * (Phi @ Phi.T) + SIGMA_AU**2 * jnp.eye(2 * n_t)
        cho = jax.scipy.linalg.cho_factor(Sig)
        alpha = jax.scipy.linalg.cho_solve(cho, d)
        logZ = -0.5 * (d @ alpha + 2.0 * jnp.sum(jnp.log(jnp.diag(cho[0]))))
        m = s2 * (Phi.T @ alpha)
        V = s2 * jnp.eye(4) - s2**2 * (Phi.T @ jax.scipy.linalg.cho_solve(cho, Phi))
        L = jnp.linalg.cholesky(V + 1e-14 * jnp.eye(4))
        ti = m[None, :] + eps @ L.T
        a = jnp.vectorize(
            lambda east_X, east_Y, north_X, north_Y: jnp.sqrt(
                (
                    east_X**2
                    + north_X**2
                    + east_Y**2
                    + north_Y**2
                    + jnp.sqrt(
                        jnp.maximum(
                            (east_X**2 + north_X**2 + east_Y**2 + north_Y**2) ** 2
                            - 4.0 * (east_X * north_Y - north_X * east_Y) ** 2,
                            0.0,
                        )
                    )
                )
                / 2.0
            )
        )(ti[:, 0], ti[:, 1], ti[:, 2], ti[:, 3])
        mu = (2.0 * jnp.pi / P) ** 2 * a**3
        frac = jnp.mean((mu >= MODEL.mu_lo) & (mu <= MODEL.mu_hi))
        return logZ + jnp.log(frac + 1e-300)

    ee, pp, tt = jnp.meshgrid(e_ax, lnP_ax, tau_ax, indexing="ij")
    lp = jax.lax.map(
        lambda etc: jax.vmap(cell)(*etc),
        tuple(x.reshape(32, -1) for x in (ee.ravel(), pp.ravel(), tt.ravel())),
    ).ravel()
    p = np.asarray(jnp.exp(lp - jax.scipy.special.logsumexp(lp))).reshape(
        n_e, n_lnP, n_tau
    )
    out = {}
    for name, ax, axis in (
        ("e", np.asarray(e_ax), (1, 2)),
        ("lnP", np.asarray(lnP_ax), (0, 2)),
        ("tau", np.asarray(tau_ax), (0, 1)),
    ):
        marg = p.sum(axis=axis)
        mean = float((marg * ax).sum())
        out[name] = (mean, float(np.sqrt((marg * (ax - mean) ** 2).sum())))
    return out


def _weighted_stats(post, col):
    """Weighted mean/std of one posterior column, plus the Kish ESS."""
    lw = np.asarray(post.log_weights)
    th = np.asarray(post.samples)[:, col]
    finite = np.isfinite(lw)
    w = np.exp(lw[finite] - lw[finite].max())
    w /= w.sum()
    mean = float((w * th[finite]).sum())
    std = float(np.sqrt((w * (th[finite] - mean) ** 2).sum()))
    return mean, std, float(1.0 / (w**2).sum())


def test_ti_parameter_names_multiply_declared_axes():
    """Named coefficients multiply X and Y on their declared sky axes."""
    names = ("e", "lnP", "tau", "east_X", "east_Y", "north_X", "north_Y")
    assert TI_PARAM_NAMES == names
    params = dict(
        zip(names, [0.3, np.log(300.0), 0.2, 1.7, -0.4, 0.6, 2.1], strict=True)
    )
    theta = jnp.array([params[name] for name in TI_PARAM_NAMES])
    M = 2 * np.pi * ((TIMES - MODEL.t_ref) / np.exp(params["lnP"]) - params["tau"])
    E = np.asarray(_newton_E(M, params["e"]))
    X = np.cos(E) - params["e"]
    Y = np.sqrt(1 - params["e"] ** 2) * np.sin(E)
    east, north = MODEL.forward_au(theta, TIMES)
    np.testing.assert_allclose(
        east, params["east_X"] * X + params["east_Y"] * Y, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        north, params["north_X"] * X + params["north_Y"] * Y, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("cos_i", [0.7, -0.7])
def test_ti_semimajor_axis_east_first(cos_i):
    """East-first determinant is -a^2 cos(i); its square preserves a."""
    a, W, w = 2.3, 1.1, 0.8
    cW, sW, cw, sw = np.cos(W), np.sin(W), np.cos(w), np.sin(w)
    coefficients = dict(
        east_X=a * (sW * cw + cW * sw * cos_i),
        east_Y=a * (-sW * sw + cW * cw * cos_i),
        north_X=a * (cW * cw - sW * sw * cos_i),
        north_Y=a * (-cW * sw - sW * cw * cos_i),
    )
    determinant = (
        coefficients["east_X"] * coefficients["north_Y"]
        - coefficients["north_X"] * coefficients["east_Y"]
    )
    assert determinant == pytest.approx(-(a**2) * cos_i, abs=1e-12, rel=0)
    assert float(ti_semimajor_axis(**coefficients)) == pytest.approx(
        a, abs=1e-12, rel=0
    )


def test_state_to_theta_convention_lock():
    """state_to_theta inverts the rotation-built truth exactly."""
    r, v, theta_true = _truth_state(0.0)
    theta = state_to_theta(r, v, TRUTH["mu"], 0.0, MODEL.t_ref)
    assert np.allclose(np.asarray(theta), np.asarray(theta_true), atol=1e-8)


def test_forward_au_matches_independent_propagation():
    """The model forward reproduces independently propagated positions."""
    _, _, theta = _truth_state(0.0)
    x, y = MODEL.forward_au(theta, TIMES)
    for i, t in enumerate(np.asarray(TIMES)):
        r, _, _ = _truth_state(float(t))
        assert np.allclose(float(x[i]), float(r[0]), atol=1e-9)
        assert np.allclose(float(y[i]), float(r[1]), atol=1e-9)


def test_lambert_depth_fit_matches_grid_reference():
    """The exact Lambert fit reproduces the certified grid posterior (n=2)."""
    data = _make_data(2)
    ref = _grid_reference(data)
    post, parts = lambert_depth_fit(
        data,
        dist_pc=DIST_PC,
        model=MODEL,
        key=jax.random.PRNGKey(0),
        n_per_family=6000,
        n_max_rev=5,
        return_parts=True,
    )
    mean_e, _, ess = _weighted_stats(post, 0)
    mean_lnP, _, _ = _weighted_stats(post, 1)
    assert ess > 100.0
    assert abs(mean_e - ref["e"][0]) < 0.06
    assert abs(mean_lnP - ref["lnP"][0]) < 0.15
    lj = parts["ln_jac"][np.isfinite(parts["ln_prior"])]
    assert float(np.std(np.asarray(lj))) > 0.1


def test_third_epoch_tightens_period():
    """Scoring a third epoch through the likelihood shrinks std(lnP)."""
    # Gentler noise and a depth proposal matched to the ~1 AU data scale keep
    # the uniform proposal's ESS usable once the third epoch's likelihood
    # multiplies in (the production answer is a data-informed proposal).
    sigma = 0.06
    post2 = lambert_depth_fit(
        _make_data(2, sigma_au=sigma),
        dist_pc=DIST_PC,
        model=MODEL,
        key=jax.random.PRNGKey(1),
        n_per_family=6000,
        n_max_rev=5,
        z_max_au=3.0,
    )
    data3 = _make_data(3, sigma_au=sigma)
    post3 = lambert_depth_fit(
        data3,
        dist_pc=DIST_PC,
        model=MODEL,
        key=jax.random.PRNGKey(2),
        n_per_family=16000,
        n_max_rev=5,
        idx=(0, 2),
        z_max_au=3.0,
    )
    _, std2, _ = _weighted_stats(post2, 1)
    _mean3, std3, ess3 = _weighted_stats(post3, 1)
    assert ess3 > 15.0
    assert std3 < 0.7 * std2
    # The posterior is multi-modal, so check the top-weight particle (a real
    # orbit) rather than the weighted-mean theta (which is not an orbit).
    top = np.asarray(post3.samples)[int(np.argmax(np.asarray(post3.log_weights)))]
    x_au, y_au = MODEL.forward_au(jnp.asarray(top), data3.times)
    resid = np.hypot(
        np.asarray(x_au / DIST_PC - data3.ra), np.asarray(y_au / DIST_PC - data3.dec)
    )
    assert float(resid[1]) < 4.0 * sigma / DIST_PC


def test_admissible_region_fit_matches_grid_reference_n1():
    """The exact AR fit reproduces the certified grid posterior (n=1)."""
    data = _make_data(1)
    ref = _grid_reference(data)
    post = admissible_region_fit(
        data,
        dist_pc=DIST_PC,
        model=MODEL,
        key=jax.random.PRNGKey(4),
        n_particles=120_000,
    )
    mean_e, _, ess = _weighted_stats(post, 0)
    mean_lnP, _, _ = _weighted_stats(post, 1)
    assert ess > 150.0
    assert abs(mean_e - ref["e"][0]) < 0.06
    assert abs(mean_lnP - ref["lnP"][0]) < 0.15


def test_conditioned_pair_validation():
    """Bad conditioned-epoch choices raise informative errors."""
    data = _make_data(2)
    with pytest.raises(ValueError, match="distinct valid"):
        lambert_depth_fit(
            data,
            dist_pc=DIST_PC,
            model=MODEL,
            key=jax.random.PRNGKey(0),
            idx=(0, 0),
            n_per_family=10,
        )
    bad = RelativeAstromData(
        times=data.times,
        ra=data.ra,
        dec=data.dec,
        ra_err=data.ra_err,
        dec_err=data.dec_err,
        corr=jnp.array([0.5, 0.0]),
        planet_id=data.planet_id,
        is_valid=data.is_valid,
    )
    with pytest.raises(ValueError, match="correlation"):
        lambert_depth_fit(
            bad,
            dist_pc=DIST_PC,
            model=MODEL,
            key=jax.random.PRNGKey(0),
            n_per_family=10,
        )
