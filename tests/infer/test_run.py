"""Posterior, run and the generic backends.

Two kinds of test. The fast ones use a one-leaf fit with a closed-form evidence: a
uniform prior on a planet radius times an unnormalized Gaussian likelihood, so every
backend has a known answer. The ``slow`` ones run each backend on the golden orbit (the
fixed-mass relative-astrometry fit whose density is pinned against the old system
density in ``test_fit.py``) and compare with the old backend on the same problem.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from orbix.system import ClassicalChart
from priorities import Uniform
from test_fit import (
    COORDS,
    GOLDEN,
    PERIOD_RANGE,
    _block,
    _chart_coords,
    _coords,
    _orbit_map,
    _relative,
    _scene,
)

from photomancy.infer import (
    MAP,
    MCLMC,
    NUTS,
    SMC,
    AbstractTerm,
    Fit,
    Grid,
    Laplace,
    NestedSampling,
    Pathfinder,
    Posterior,
    run,
)
from photomancy.infer.backends.grid import gauss_legendre, midpoint
from photomancy.infer.backends.staged import FitStage
from photomancy.infer.posterior import cluster_to_mixture, ravel_samples

jax.config.update("jax_enable_x64", True)

KEY = "planets[0].Rp_Rearth"
HIGH = 4.0
WIDTH = 0.3
# L-BFGS steps of the golden-orbit optimum comparisons. The old density cuts each angle
# to one cell, so a long run leaves the cell and stalls on its edge. Both backends take
# identical steps only while the line search is not decided by rounding: the two
# densities agree to 2e-12 absolute (1e-16 relative) but past the second step an
# accept/reject of the line search flips and the paths part (0.8 apart at step 4).
STEPS = 2


class _Gauss(AbstractTerm):
    """Unnormalized Gaussian likelihood of the planet radius, centered at ``center``."""

    center: jax.Array
    reads = ()

    def __call__(self, model):
        return -0.5 * jnp.sum(((model.planets[0].Rp_Rearth - self.center) / WIDTH) ** 2)

    def draw(self, key, model):
        return self


def _leaf_fit(center=1.5):
    return Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_Gauss(jnp.asarray(center)),))


def _exact_log_z(center=1.5):
    """Log of the mean over ``[0, HIGH]`` of ``exp(-0.5 ((r - c) / w)^2)``."""
    cdf = jax.scipy.stats.norm.cdf
    mass = cdf((HIGH - center) / WIDTH) - cdf((0.0 - center) / WIDTH)
    return float(jnp.log(mass * WIDTH * math.sqrt(2 * math.pi) / HIGH))


def _moments(post):
    """Weighted mean and standard deviation of the radius."""
    w = np.exp(np.asarray(post.log_weights))
    r = np.asarray(post.samples[KEY])[..., 0]
    mean = np.sum(w * r)
    return mean, math.sqrt(np.sum(w * (r - mean) ** 2))


def _check_contract(post, fit, n=None):
    """Fields every posterior has, whichever backend made it."""
    assert isinstance(post, Posterior)
    assert set(post.samples) == set(fit.free)
    for name in ("log_weights", "log_target", "log_evidence", "log_evidence_se"):
        assert getattr(post, name).dtype == jnp.float64
    if n is not None:
        assert post.log_weights.shape == (n,)
    np.testing.assert_allclose(
        float(jax.scipy.special.logsumexp(post.log_weights)), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(post.log_target),
        np.asarray(fit.log_density(post.samples)),
        rtol=1e-12,
    )


# ----- the Posterior type --------------------------------------------------------


def test_posterior_checks_its_fields():
    """Wrong kind, keys, dtype and shapes are refused; batch axes are draw axes."""
    fit = _leaf_fit()
    post = run(fit, Laplace(n_draws=64), jax.random.key(0))
    fields = {
        name: getattr(post, name)
        for name in (
            "samples",
            "log_weights",
            "log_target",
            "log_evidence",
            "log_evidence_se",
            "fit",
            "info",
            "kind",
        )
    }
    Posterior(**fields)
    with pytest.raises(ValueError, match="kind"):
        Posterior(**{**fields, "kind": "gibbs"})
    with pytest.raises(ValueError, match=r"keys of fit\.free"):
        Posterior(**{**fields, "samples": {"other": post.samples[KEY]}})
    with pytest.raises(ValueError, match="float64"):
        Posterior(**{**fields, "log_target": post.log_target.astype(jnp.float32)})
    with pytest.raises(ValueError, match="log_target has shape"):
        Posterior(**{**fields, "log_target": post.log_target[:-1]})
    with pytest.raises(ValueError, match="expected the batch shape"):
        Posterior(**{**fields, "log_evidence": jnp.zeros(3)})
    with pytest.raises(ValueError, match="leading axes"):
        Posterior(**{**fields, "samples": {KEY: post.samples[KEY][:-1]}})


def test_ravel_samples_round_trips_a_block_fit():
    """``ravel_samples`` flattens leaf and block entries and ``unravel`` inverts it."""
    fit = Fit(
        _scene(),
        {KEY: Uniform(0.5, 4.0), "dynamics.orbits": _block()},
        (_relative(),),
    )
    post = run(fit, Laplace(n_draws=8, n_steps=5), jax.random.key(1))
    flat, unravel = ravel_samples(post)
    assert flat.shape[:-1] == (8,)
    back = unravel(flat)
    for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(post.samples), strict=True):
        np.testing.assert_array_equal(a, b)
    only, _ = ravel_samples(post, keys=(KEY,))
    assert only.shape == (8, 1)


def test_cluster_to_mixture_follows_the_modes():
    """Two well-separated clusters give two modes with their masses."""
    fit = _leaf_fit()
    post = run(fit, Laplace(n_draws=400), jax.random.key(2))
    left = jnp.where(jnp.arange(400) % 4 == 0, 0.5, 3.0)
    split = eqx.tree_at(
        lambda p: (p.samples, p.log_weights),
        post,
        ({KEY: left[:, None]}, jnp.full(400, -math.log(400))),
    )
    mixture = cluster_to_mixture(split, 2, jax.random.key(3))
    means = np.sort(np.asarray(mixture.component_distribution.loc).ravel())
    np.testing.assert_allclose(means, [0.5, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        np.sort(np.exp(np.asarray(jax.nn.log_softmax(mixture.mixing_logits)))),
        [0.25, 0.75],
        atol=1e-6,
    )


# ----- backends on the one-leaf fit with a closed-form evidence ------------------


def test_grid_is_quadrature_with_the_exact_evidence():
    """A fine midpoint rule recovers the evidence and mean; refining converges."""
    fit = _leaf_fit()
    errors = []
    for n in (200, 800):
        post = run(fit, Grid({KEY: midpoint(0.0, HIGH, n)}), jax.random.key(0))
        assert post.kind == "quadrature"
        _check_contract(post, fit, n)
        errors.append(abs(float(post.log_evidence) - _exact_log_z()))
        assert np.isnan(float(post.log_evidence_se))
    assert errors[1] < errors[0] and errors[1] < 1e-5
    mean, sd = _moments(post)
    assert abs(mean - 1.5) < 1e-6 and abs(sd - WIDTH) < 1e-3
    gl = run(fit, Grid({KEY: gauss_legendre(0.0, HIGH, 200)}), jax.random.key(0))
    assert abs(float(gl.log_evidence) - _exact_log_z()) < 1e-6
    with pytest.raises(ValueError, match="non-positive weight"):
        run(fit, Grid({KEY: (jnp.arange(3.0), jnp.array([1.0, -1.0, 1.0]))}), None)
    with pytest.raises(ValueError, match="exactly the degrees of freedom"):
        run(fit, Grid({"x": midpoint(0.0, 1.0, 3)}), None)


def test_laplace_corrects_the_gaussian_with_importance_weights():
    """Weighted draws and evidence are right though the real-line target is skewed."""
    fit = _leaf_fit()
    post = run(fit, Laplace(n_draws=20000), jax.random.key(0))
    _check_contract(post, fit, 20000)
    assert abs(float(post.log_evidence) - _exact_log_z()) < 4 * float(
        post.log_evidence_se
    )
    mean, _ = _moments(post)
    assert abs(mean - 1.5) < 0.01
    assert post.info.means.shape == (1, 1)


def test_nuts_has_no_evidence_and_the_right_moments():
    """NUTS draws match the known mean; the evidence is NaN; chains are in info."""
    fit = _leaf_fit()
    post = run(fit, NUTS(n_warmup=200, n_draws=500, n_chains=4), jax.random.key(0))
    _check_contract(post, fit, 2000)
    assert np.isnan(float(post.log_evidence)) and np.isnan(float(post.log_evidence_se))
    assert post.kind == "monte_carlo"
    np.testing.assert_array_equal(np.asarray(post.info.chain), np.repeat(range(4), 500))
    r = np.asarray(post.samples[KEY])[:, 0].reshape(4, 500)
    se = r.mean(axis=1).std(ddof=1) / 2.0
    assert abs(r.mean() - 1.5) < 5 * max(se, 0.01)
    assert not np.any(np.asarray(post.info.divergent))


def test_mclmc_recovers_the_mean():
    """MCLMC (two or more dimensions) matches the known mean and has no evidence."""
    fit = Fit(
        _scene(),
        {KEY: Uniform(0.0, HIGH), "planets[0].physical_model.Ag": Uniform(0.1, 1.0)},
        (_Gauss(jnp.asarray(1.5)),),
    )
    post = run(fit, MCLMC(n_tune=500, n_draws=2000, n_chains=2), jax.random.key(0))
    _check_contract(post, fit, 4000)
    assert np.isnan(float(post.log_evidence))
    assert abs(_moments(post)[0] - 1.5) < 0.05
    ag = np.asarray(post.samples["planets[0].physical_model.Ag"]).ravel()
    assert abs(ag.mean() - 0.55) < 0.05


def test_smc_recovers_the_evidence():
    """Tempered SMC gives the evidence within its run-to-run scatter."""
    fit = _leaf_fit()
    runs = [
        run(fit, SMC(n_particles=2000, n_mcmc_steps=5), jax.random.key(k))
        for k in range(4)
    ]
    for post in runs:
        _check_contract(post, fit, 2000)
        assert bool(post.info.reached)
    z = np.array([float(p.log_evidence) for p in runs])
    sd = max(z.std(ddof=1), 0.01)
    assert abs(z.mean() - _exact_log_z()) < 4 * sd / 2.0 + 0.02
    assert abs(_moments(runs[0])[0] - 1.5) < 0.05


def test_pathfinder_importance_weights_target_the_fit():
    """Pathfinder's mixture proposal is corrected to the fit's density."""
    fit = _leaf_fit()
    post = run(fit, Pathfinder(n_draws=5000), jax.random.key(0))
    _check_contract(post, fit, 5000)
    assert abs(_moments(post)[0] - 1.5) < 0.02


def test_map_finds_the_mode():
    """MAP returns one draw at the optimum."""
    fit = _leaf_fit()
    post = run(fit, MAP(n_steps=100), jax.random.key(0))
    assert post.log_weights.shape == (1,)
    assert np.isnan(float(post.log_evidence))
    # the optimum of the real-line density, which carries the logit volume
    x = float(post.samples[KEY][0, 0])
    assert abs(x - 1.5) < 0.1
    np.testing.assert_allclose(
        float(post.log_target[0]), float(fit.log_density(post.samples)[0]), rtol=1e-12
    )


def test_nested_sampling_recovers_the_evidence():
    """Jaxns gives the exact evidence within its standard deviation."""
    pytest.importorskip("jaxns")
    fit = _leaf_fit()
    post = run(fit, NestedSampling(max_samples=20000), jax.random.key(0))
    _check_contract(post, fit)
    se = float(post.log_evidence_se)
    assert abs(float(post.log_evidence) - _exact_log_z()) < 5 * max(se, 0.02)
    assert abs(_moments(post)[0] - 1.5) < 0.05
    block_fit = Fit(_scene(), {"dynamics.orbits": _block()}, (_relative(),))
    with pytest.raises(ValueError, match="block"):
        run(block_fit, NestedSampling(), jax.random.key(0))


def test_a_stack_of_worlds_equals_the_per_world_loop():
    """``eqx.filter_vmap(run)`` over two stacked fits equals running each in turn."""
    centers = (1.2, 2.0)
    fits = [_leaf_fit(c) for c in centers]
    static = eqx.partition(fits[0], eqx.is_array)[1]
    stack = eqx.combine(
        jax.tree.map(
            lambda *a: jnp.stack(a), *[eqx.partition(f, eqx.is_array)[0] for f in fits]
        ),
        static,
    )
    backend = Laplace(n_draws=300)
    keys = jax.random.split(jax.random.key(7), 2)
    batched = eqx.filter_vmap(lambda f, k: run(f, backend, k))(stack, keys)
    assert batched.log_weights.shape == (2, 300)
    assert batched.log_evidence.shape == (2,)
    assert batched.samples[KEY].shape == (2, 300, 1)
    for i, fit in enumerate(fits):
        one = run(fit, backend, keys[i])
        np.testing.assert_allclose(
            np.asarray(batched.log_evidence[i]), float(one.log_evidence), rtol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(batched.samples[KEY][i]), np.asarray(one.samples[KEY]), rtol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(batched.log_target[i]), np.asarray(one.log_target), rtol=1e-9
        )


def _stack(fits):
    """Stack fits of one structure along a new leading world axis."""
    static = eqx.partition(fits[0], eqx.is_array)[1]
    arrays = [eqx.partition(f, eqx.is_array)[0] for f in fits]
    return eqx.combine(jax.tree.map(lambda *a: jnp.stack(a), *arrays), static)


_WORLD_BACKENDS = {
    "nuts": NUTS(n_warmup=60, n_draws=40, n_chains=2),
    "smc": SMC(n_particles=200, n_mcmc_steps=3),
    "pathfinder": Pathfinder(n_draws=200),
    "grid": Grid({KEY: midpoint(0.0, HIGH, 120)}),
}


@pytest.mark.parametrize("name", list(_WORLD_BACKENDS))
def test_every_batching_backend_maps_over_two_worlds(name):
    """``filter_vmap(run)`` over two stacked fits gives two posteriors of one type.

    Truth source: each world's own fit: ``log_target`` is its density at the stored
    draws and the posterior mean follows that world's data center (1.2 against 2.0).
    Tolerance basis: shapes and the density identity are exact (rtol 1e-9); the means
    differ by 0.8 between worlds, far above the sampler scatter, so each is compared to
    its center within 0.25 (200 particles, 80 draws, quadrature exact).
    Fails if: a backend closes over the fit instead of taking it as a traced argument,
    mixes the worlds or breaks under a batch prefix.
    """
    centers = (1.2, 2.0)
    fits = [_leaf_fit(c) for c in centers]
    backend = _WORLD_BACKENDS[name]
    keys = jax.random.split(jax.random.key(3), 2)
    out = eqx.filter_vmap(lambda f, k: run(f, backend, k))(_stack(fits), keys)
    n = out.log_weights.shape[-1]
    assert out.log_weights.shape == (2, n) and out.log_evidence.shape == (2,)
    assert out.samples[KEY].shape == (2, n, 1)
    for i, (fit, center) in enumerate(zip(fits, centers, strict=True)):
        np.testing.assert_allclose(
            np.asarray(out.log_target[i]),
            np.asarray(fit.log_density({KEY: out.samples[KEY][i]})),
            rtol=1e-9,
        )
        w = np.exp(np.asarray(out.log_weights[i]))
        mean = float(np.sum(w * np.asarray(out.samples[KEY][i])[:, 0]))
        assert abs(mean - center) < 0.25


class _NanAbove(_Gauss):
    """The Gaussian term, NaN for radii above 2.2 (a failed evaluation)."""

    def __call__(self, model):
        r = model.planets[0].Rp_Rearth[0]
        return jnp.where(r > 2.2, jnp.nan, super().__call__(model))


def test_a_nan_term_refuses_the_slice_instead_of_zero_weight():
    """NaN values of a term make the whole slice NaN, in weights and evidence.

    Fails if: a backend or ``run`` maps NaN to a zero weight, which would return a
    clean-looking posterior and a finite evidence from the draws that did not fail.
    """
    fit = Fit(_scene(), {KEY: Uniform(0.0, HIGH)}, (_NanAbove(jnp.asarray(1.5)),))
    start = {KEY: jnp.asarray([1.5])}
    laplace = run(fit, Laplace(n_draws=4000, init=start), jax.random.key(0))
    assert np.any(np.isnan(np.asarray(fit.log_density(laplace.samples))))
    grid = run(fit, Grid({KEY: midpoint(0.0, HIGH, 200)}), jax.random.key(0))
    for post in (laplace, grid):
        assert np.all(np.isnan(np.asarray(post.log_weights)))
        assert np.isnan(float(post.log_evidence))


def _prior_only_orbit_fit():
    return Fit(_scene(), {"dynamics.orbits": _block()}, ())


@pytest.mark.medium
def test_smc_prior_only_evidence_on_an_orbit_block_is_zero():
    """A normalized prior with no terms has evidence exactly 1.

    Truth source: the block's prior is a normalized density, so ``log Z = 0``.
    Tolerance basis: the reviewer measured 0.0000 (rounded to 4 places) with 1000
    particles; the bound 0.1 is about 20 times the SMC scatter of this smooth problem.
    Fails if: the block's volume is not normalized or SMC mishandles the angles.
    """
    post = run(
        _prior_only_orbit_fit(),
        SMC(n_particles=1000, n_mcmc_steps=3),
        jax.random.key(0),
    )
    assert abs(float(post.log_evidence)) < 0.1


@pytest.mark.xfail(
    strict=True,
    reason="L1: orbit angle coordinates are plain reals with a periodic density, so a "
    "proposal wider than one 2 pi cell inflates the importance evidence",
)
def test_laplace_prior_only_evidence_on_an_orbit_block_is_zero():
    """The same prior-only evidence through ``Laplace`` (known defect, strict xfail).

    Measured with ``min_eigenvalue=0.1``: log Z about +3.1 +- 0.3 against truth 0 with
    a passing k-hat. Remove the marker when the block declares its periods.
    """
    post = run(
        _prior_only_orbit_fit(),
        Laplace(n_draws=2000, min_eigenvalue=0.1),
        jax.random.key(0),
    )
    assert abs(float(post.log_evidence)) < 4 * float(post.log_evidence_se)


def test_run_refuses_a_fit_built_without_x64():
    """Fit refuses float32 mode at construction, so ``run`` never sees one."""
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            _leaf_fit()
    finally:
        jax.config.update("jax_enable_x64", True)


# ----- the golden orbit against the old backends --------------------------------


def _golden_fit():
    return Fit(_scene(), {"dynamics.orbits": _block()}, (_relative(),))


def _golden_start():
    point = GOLDEN["g1"]["points"][0]
    return {"dynamics.orbits": _coords(point)}


def _old_problem():
    """The old chart problem for the same relative astrometry and prior."""
    from hwoutils.constants import Msun2kg
    from orbit_fit import chart_problem

    point = jnp.asarray(GOLDEN["g1"]["points"][0])
    problem = chart_problem(
        _relative().data,
        {"orbit": point},
        star_mass_kg=Msun2kg,
        dist_pc=10.0,
        chart=ClassicalChart(),
        period_range_d=PERIOD_RANGE,
    )
    return problem, problem.flatten({"orbit": point})


def _in_block_coordinates(problem, start):
    """The old chart problem re-expressed in the new fit's block coordinates.

    Prior and density pick up ``log |dx/dy|`` of the block map; the likelihood and the
    evidence are unchanged. Vectors stay in the old chart order.
    """
    chart, support = _orbit_map()

    def carry(fn, jac):
        def wrapped(y):
            x, log_jac = chart.constrain(y[None, :], support)
            return fn(x[0]) + (log_jac[0] if jac else 0.0)

        return wrapped

    return (
        carry(problem.logdensity, True),
        carry(problem.logprior, True),
        carry(problem.loglikelihood, False),
        chart.unconstrain(start[None, :], support)[0],
    )


def _new_coords(post):
    """Chart coordinates of each draw as ``(N, 6)`` in the old coordinate order."""
    return np.asarray(_chart_coords(post.samples["dynamics.orbits"])[0])


def _wrap_angles(z):
    """Mean of the three non-angle coordinates, then unit vectors of the angles."""
    angles = z[..., 3:]
    return np.concatenate(
        [z[..., :3], np.cos(angles), np.sin(angles)],
        axis=-1,
    )


@pytest.mark.slow
def test_nuts_agrees_with_the_old_backend_on_the_golden_orbit():
    """New and old NUTS posterior means agree within five combined standard errors.

    Both sample the golden orbit from the same start with four independent chains. The
    statistic is the difference of the chain-mean averages of the three non-angle chart
    coordinates and of the cosine and sine of each angle, each against five times the
    standard error from the scatter of the four chain means (per backend, combined in
    quadrature), floored at 2 percent of the old posterior standard deviation.
    """
    from photomancy.infer.backends import NUTSBackend

    problem, start = _old_problem()
    old = np.stack(
        [
            np.asarray(
                NUTSBackend(n_warmup=400, n_samples=600)
                .run(problem.logdensity, start, jax.random.key(k))
                .samples
            )
            for k in range(4)
        ]
    )
    post = run(
        _golden_fit(),
        NUTS(n_warmup=400, n_draws=600, n_chains=4, init=_golden_start()),
        jax.random.key(11),
    )
    new = _new_coords(post).reshape(4, 600, 6)
    f_old, f_new = _wrap_angles(old), _wrap_angles(new)
    mean_old, mean_new = f_old.mean(axis=1), f_new.mean(axis=1)
    se = np.sqrt(mean_old.var(axis=0, ddof=1) / 4 + mean_new.var(axis=0, ddof=1) / 4)
    floor = 0.02 * f_old.reshape(-1, f_old.shape[-1]).std(axis=0)
    diff = np.abs(mean_old.mean(axis=0) - mean_new.mean(axis=0))
    assert np.all(diff < 5 * np.maximum(se, floor)), (diff, se, floor)


@pytest.mark.slow
def test_mclmc_agrees_with_the_old_backend_on_the_golden_orbit():
    """New and old MCLMC means agree to the same rule as NUTS (two runs each)."""
    from photomancy.infer.backends import MCLMCBackend

    problem, start = _old_problem()
    old = np.stack(
        [
            np.asarray(
                MCLMCBackend(n_tune=1500, n_samples=3000)
                .run(problem.logdensity, start, jax.random.key(k))
                .samples
            )
            for k in range(3)
        ]
    )
    post = run(
        _golden_fit(),
        MCLMC(n_tune=1500, n_draws=3000, n_chains=3, init=_golden_start()),
        jax.random.key(5),
    )
    new = _new_coords(post).reshape(3, 3000, 6)
    f_old, f_new = _wrap_angles(old), _wrap_angles(new)
    mean_old, mean_new = f_old.mean(axis=1), f_new.mean(axis=1)
    se = np.sqrt(mean_old.var(axis=0, ddof=1) / 3 + mean_new.var(axis=0, ddof=1) / 3)
    floor = 0.02 * f_old.reshape(-1, f_old.shape[-1]).std(axis=0)
    diff = np.abs(mean_old.mean(axis=0) - mean_new.mean(axis=0))
    assert np.all(diff < 5 * np.maximum(se, floor)), (diff, se, floor)


@pytest.mark.slow
def test_smc_agrees_with_the_old_backend_on_the_golden_orbit():
    """New and old SMC log evidence agree within five combined run-to-run errors.

    The old SMC runs on the old density, whose prior is normalized against the
    unnormalized Haar measure; the new fit's prior is normalized, so the new evidence
    differs by exactly the constant ``LOG_HAAR`` (added to every old chart volume).
    The comparison subtracts it.
    """
    from photomancy.infer.backends import SMCBackend
    from photomancy.priors import LOG_HAAR

    _, logprior, loglikelihood, _ = _in_block_coordinates(*_old_problem())
    perm = np.array([sorted(COORDS).index(n) for n in COORDS])
    fit = _golden_fit()
    n = 1500
    old_z, new_z = [], []
    for k in range(4):
        key = jax.random.key(k)
        draws = fit.to_real(fit.sample_prior(key, (n,)))
        particles = jax.vmap(lambda u: jax.flatten_util.ravel_pytree(u)[0])(draws)
        old = SMCBackend(n_mcmc_steps=8).run(
            logprior, loglikelihood, particles[:, perm], key
        )
        old_z.append(float(old.evidence))
        new_z.append(
            float(run(fit, SMC(n_particles=n, n_mcmc_steps=8), key).log_evidence)
        )
    old_z, new_z = np.array(old_z), np.array(new_z)
    se = math.sqrt(old_z.var(ddof=1) / 4 + new_z.var(ddof=1) / 4)
    assert abs(old_z.mean() - LOG_HAAR - new_z.mean()) < 5 * max(se, 0.05), (
        old_z,
        new_z,
    )


@pytest.mark.slow
def test_laplace_agrees_with_the_old_backend_on_the_golden_orbit():
    """The new Laplace modes equal the old MAP-plus-Hessian fit of the same density.

    Same start and optimizer settings, so the mean and covariance of the Gaussian
    proposal equal the old ``GaussianPosterior`` to optimizer tolerance.
    """
    from photomancy.infer.backends import LaplaceBackend

    logdensity, _, _, start = _in_block_coordinates(*_old_problem())
    old = LaplaceBackend(n_steps=STEPS, min_eigenvalue=1.0).run(
        logdensity, start, jax.random.key(0)
    )
    post = run(
        _golden_fit(),
        Laplace(n_draws=64, n_steps=STEPS, min_eigenvalue=1.0, init=_golden_start()),
        jax.random.key(0),
    )
    # the new flat vector follows sorted coordinate names, the old one the chart order
    perm = np.array([sorted(COORDS).index(n) for n in COORDS])
    np.testing.assert_allclose(
        np.asarray(post.info.means[0])[perm], np.asarray(old.mean), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(post.info.covs[0])[np.ix_(perm, perm)],
        np.asarray(old.cov),
        rtol=1e-3,
        atol=1e-9,
    )


@pytest.mark.slow
def test_map_equals_the_old_map_backend_on_the_golden_orbit():
    """The new MAP optimum equals the old ``MAPBackend`` optimum to 1e-6."""
    from photomancy.infer.backends import MAPBackend

    logdensity, _, _, start = _in_block_coordinates(*_old_problem())
    stage = (FitStage(STEPS, ("all",), solver=optax.lbfgs(), name="lbfgs"),)
    old = MAPBackend(labels="all", transforms={}, stages=stage).run(logdensity, start)
    post = run(
        _golden_fit(),
        MAP(n_steps=STEPS, init=_golden_start()),
        jax.random.key(0),
    )
    block = post.samples["dynamics.orbits"]
    new = np.stack([np.asarray(block[n])[0, 0] for n in COORDS])
    np.testing.assert_allclose(new, np.asarray(old.params), atol=1e-6)
