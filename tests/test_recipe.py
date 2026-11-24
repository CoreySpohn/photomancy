"""The fitting recipe end to end on a reduced scene, one test per stage boundary.

The scene is a star with two planets seen through the analytic coronagraph of
``image_scene``: a bright one (flux scale 1) and a faint one (flux scale 0.18), observed
as two rolls of counting-noise frames. The recipe is the one the how-to "Fit a scene"
teaches: an ``ImageTerm`` per exposure from the ``Image``; a stage-one fit with an
unbounded amplitude law; ``detect`` over a grid of sky positions; the admission rule;
``measure`` from the detected start; ``Fit.residuals``; ``detect`` again at the best
fit for one more source; then ``MAP`` (and short ``NUTS`` chains) with the real priors.
The disk of the recipe is the radiance fit of ``test_disk_fitting``: the analytic
coronagraph is not a scene coronagraph, so a disk cannot be rendered through it here.
"""

import coronagraphoto as cp
import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_disk_fitting
from hwoutils.model_paths import resolve
from image_scene import (
    BANDS,
    Epoch,
    FlatStar,
    FramePlan,
    GridPhysicalModel,
    Mearth2kg,
    Msun2kg,
    Planet,
    Scene,
    make_detector,
    make_instrument,
)
from orbix.system.orbit_blocks import SkyPositionBlock
from priorities import LogNormal, Normal
from scipy import stats
from test_disk_fitting import (
    GRID,
    INIT,
    TIME,
    WL,
    _data,
    _fit,
    _x,
)

import photomancy.detection as detection
from photomancy.detection import curvature_block, detect, measure
from photomancy.disk import radiance_forward
from photomancy.image import ImageTerm
from photomancy.infer import MAP, NUTS, Fit, Laplace, run

# The admission threshold: the 99th percentile of the per-map maximum of z over an
# 81-trial grid and two rolls, for white Poisson noise and a model equal to the truth.
THRESHOLD = 3.8
# Bright and faint truth flux scales; the faint planet's z on this grid sits between the
# design value 2.3 and THRESHOLD, so only the right threshold leaves it out.
BRIGHT, FAINT = 1.0, 0.18
SEPARATIONS, POSITION_ANGLES = (0.04, 0.05), (30.0, 200.0)
ROLLS_DEG = (0.0, 40.0)
EDGE = np.linspace(-0.06, 0.06, 9) + 0.0033  # off the star: a trial on it is NaN
_EAST, _NORTH = (jnp.asarray(a.ravel()) for a in np.meshgrid(EDGE, EDGE))
TRIALS = {
    "dynamics.orbits": {"east_arcsec": _EAST[:, None], "north_arcsec": _NORTH[:, None]}
}
AMP = "planets[{}].flux_scale"
STEPS = 100

# the disk fixture of the disk-fitting tests, shared by name
make_system = test_disk_fitting.make_system


def _scene(bright, faint):
    """Star and two circular face-on planets, 0.04 and 0.05 arcsec out."""
    wl = jnp.array([400.0, 500.0, 600.0, 700.0])
    grid = jnp.full((1, 4, 3), 1e-9)

    def planet(scale):
        model = GridPhysicalModel(wl, jnp.array([0.0, 90.0, 180.0]), grid)
        return Planet(jnp.ones(1), model, flux_scale=scale)

    return Scene.from_classical(
        FlatStar(flux_phot_per_nm_m2=1e6),
        (planet(bright), planet(faint)),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.full(2, Mearth2kg / Msun2kg),
        dist_pc=10.0,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.array(SEPARATIONS) * 10,
        e=jnp.zeros(2),
        W_rad=jnp.deg2rad(jnp.array(POSITION_ANGLES)),
        i_rad=jnp.zeros(2),
        w_rad=jnp.zeros(2),
        M0_rad=jnp.zeros(2),
        epoch_d=0.0,
        order=(0, 1),
    )


def _block(row, spread=0.1):
    """The sky offset of one planet as the fitted coordinates."""
    return SkyPositionBlock(
        east_arcsec=Normal(0.0, spread),
        north_arcsec=Normal(0.0, spread),
        time_d=0.0,
        rows=(row,),
        system="dynamics",
    )


def _absent(scene, rows=(0, 1)):
    """The scene with the planets in ``rows`` at zero flux."""
    zero = jnp.zeros(1)
    return eqx.tree_at(
        lambda s: tuple(s.planets[k].flux_scale for k in rows),
        scene,
        tuple(zero for _ in rows),
    )


def _terms(truth):
    """One ``ImageTerm`` per roll, from the ``Image`` that ``observe`` returns."""
    instrument = make_instrument(make_detector("poisson"))
    out = []
    for i, roll in enumerate(ROLLS_DEG):
        exposure = cp.build_exposure(
            truth,
            instrument,
            BANDS,
            time_d=0.0,
            roll_deg=roll,
            plan=FramePlan(frame_time_s=3000.0, n_frames=2),
        )
        image = cp.observe(jax.random.PRNGKey(10 + i), truth, instrument, exposure)
        out.append(ImageTerm(image, instrument=instrument, exposure=exposure))
    return tuple(out)


def _slot(model, terms, row):
    """Stage one: the source in ``row`` with an unbounded amplitude and a position."""
    law = {AMP.format(row): Normal(0.0, 10.0), "dynamics.orbits": _block(row)}
    return Fit(model, law, terms)


def _detect(fit, row):
    """The map over the trial grid and the index of its largest z."""
    found = detect(fit, TRIALS, amplitude=AMP.format(row), batch_size=9)
    return found, int(jnp.nanargmax(found.z))


def _start(found, index, row, model):
    """The start from the detect map: the trial at its peak and the amplitude there."""
    path = AMP.format(row)
    shape = jnp.shape(resolve(model, path))
    return {
        path: jnp.reshape(found.a_hat[index], shape),
        "dynamics.orbits": jax.tree.map(lambda a: a[index], TRIALS["dynamics.orbits"]),
    }


def _admit(z_max):
    """The admission rule before the fit: a source enters above the threshold."""
    return bool(z_max > THRESHOLD)


def _block_eigenvalues(fit, x, row):
    """Eigenvalues of one source's observed-information block (flux and position)."""
    path = AMP.format(row)

    def where(x):
        return (
            x[path],
            x["dynamics.orbits"]["east_arcsec"],
            x["dynamics.orbits"]["north_arcsec"],
        )

    return np.linalg.eigvalsh(np.asarray(curvature_block(fit.log_likelihood, x, where)))


def _positive(eigenvalues):
    """A block is constrained when its smallest eigenvalue clears rounding.

    The floor is relative to the largest eigenvalue: a ridge at a converged point has
    an eigenvalue of rounding size, whose sign carries no information.
    """
    return bool(eigenvalues.min() > 1e-8 * eigenvalues.max())


def _keep(estimate, eigenvalues):
    """The admission rule after the fit: converged, and a constrained block.

    Convergence is the Newton decrement, the unit-free form of the gradient norm
    judged against the widths: under 0.01 nat.
    """
    return bool(estimate.newton_decrement < 0.01) and _positive(eigenvalues)


def _truth_position(scene, row):
    """The injected sky offset of one planet, as the block writes it."""
    x = _block(row).inverse(scene)
    return float(x["east_arcsec"][0]), float(x["north_arcsec"][0])


def _nearest(east, north):
    """Index of the trial nearest to a sky offset."""
    return int(jnp.argmin((_EAST - east) ** 2 + (_NORTH - north) ** 2))


@pytest.fixture(scope="module")
def recipe():
    """Run the recipe stage by stage; each test reads the stage it checks."""
    truth = _scene(BRIGHT, FAINT)
    terms = _terms(truth)
    base = _absent(truth)

    one = _slot(base, terms, 0)
    found0, i0 = _detect(one, 0)
    start0 = _start(found0, i0, 0, base)
    estimate = measure(one, start0, n_steps=STEPS)
    held = one.model_at(estimate.value)  # the admitted source stays in the model

    two = _slot(held, terms, 1)
    found1, i1 = _detect(two, 1)

    final = Fit(
        held, {AMP.format(0): LogNormal(0.0, 1.0), "dynamics.orbits": _block(0)}, terms
    )
    posterior = run(final, MAP(n_steps=STEPS, init=estimate.value), jax.random.key(0))
    best = jax.tree.map(lambda a: a[0], posterior.samples)
    extra = _slot(final.model_at(best), terms, 1)
    found_extra, _ = _detect(extra, 1)
    return dict(
        truth=truth,
        terms=terms,
        base=base,
        one=one,
        found0=found0,
        i0=i0,
        start0=start0,
        estimate=estimate,
        found1=found1,
        i1=i1,
        final=final,
        posterior=posterior,
        best=best,
        found_extra=found_extra,
    )


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.slow(reason="the staged recipe compiles each stage's program")
def test_measure_recovers_the_bright_planet_position_and_flux(recipe):
    """``measure`` from the detected start recovers the sky offset and the flux scale.

    Truth source: the injected east and north offsets of the bright planet (as
    ``SkyPositionBlock.inverse`` writes them) and its injected flux scale. Tolerance
    basis: monte-carlo, 3 formal standard errors per coordinate for the three
    coordinates, a joint false-failure rate near 1 - 0.9973**3 = 0.8 percent for a
    correct fit; one seeded realization (frames drawn with keys 10 and 11; the formal
    errors are the square roots of the diagonal of ``LikelihoodEstimate.cov``).
    Fails if the sky offset, the amplitude or the covariance is wrong, or the fit
    stopped short of the maximum.
    """
    east, north = _truth_position(recipe["truth"], 0)
    est = recipe["estimate"]
    x = est.value
    got = np.array(
        [
            float(x["dynamics.orbits"]["east_arcsec"][0]),
            float(x["dynamics.orbits"]["north_arcsec"][0]),
            float(x[AMP.format(0)][0]),
        ]
    )
    want = np.array([east, north, BRIGHT])
    # the flat vector of the estimate lists the entries sorted by key: the block's
    # east and north offsets, then the flux scale
    sigma = np.asarray(est.sigma)
    assert bool(est.regular)
    z = np.abs(got - want) / sigma
    assert np.all(z < 3.0), (z, got, want, sigma)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(reason="the staged recipe compiles each stage's program")
def test_detect_peaks_at_the_injected_planet_and_the_start_comes_from_the_map(recipe):
    """The map's largest ``z`` is the trial nearest the bright planet, above 3.8.

    Truth source: the injected sky offset of the bright planet and the grid itself:
    the trial nearest the injection is found by distance. Tolerance basis:
    exact-algebra (the index must be equal; the peak ``z`` is about 26, far above the
    threshold). The start the recipe assembles is that trial's position and the
    amplitude at it, entry for entry, and is not the injection.
    Fails if the map peaks elsewhere, or the start is taken from anything but the map.
    """
    east, north = _truth_position(recipe["truth"], 0)
    found, peak = recipe["found0"], recipe["i0"]
    assert peak == _nearest(east, north)
    assert float(found.z[peak]) > THRESHOLD
    start = recipe["start0"]["dynamics.orbits"]
    assert float(start["east_arcsec"][0]) == float(_EAST[peak])
    assert float(start["north_arcsec"][0]) == float(_NORTH[peak])
    assert float(start["east_arcsec"][0]) != east
    np.testing.assert_allclose(
        recipe["start0"][AMP.format(0)], found.a_hat[peak], rtol=0, atol=0
    )


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.slow(reason="the staged recipe compiles each stage's program")
def test_admission_leaves_out_the_sub_threshold_planet(recipe):
    """The faint planet's ``z`` is between 2.3 and 3.8, so it is not admitted.

    Truth source: the stated null of the threshold (white Poisson noise, the model
    equal to the truth, 81 trials, 2 rolls; 99th percentile of the per-map maximum of
    ``z`` near 3.8). The map is the second detect, with the bright planet held at its
    estimate. Tolerance basis: monte-carlo, one seeded realization; the planet's
    peak ``z`` is 3.1, so the margin to the threshold is 0.7 and the design's 2.3
    would admit it.
    Fails if the rule admits a source below the threshold, or the faint planet's map
    is not above the design's 2.3 (the case would then not test the threshold).
    """
    z_max = float(jnp.nanmax(recipe["found1"].z))
    east, north = _truth_position(recipe["truth"], 1)
    near = float(recipe["found1"].z[_nearest(east, north)])
    assert 2.3 < z_max < THRESHOLD
    assert near > 2.3
    assert not _admit(z_max)
    assert _admit(float(jnp.nanmax(recipe["found0"].z)))
    # left out: the model that goes on holds the faint planet at zero flux
    assert float(recipe["final"].model.planets[1].flux_scale[0]) == 0.0


@pytest.fixture(scope="module")
def flat():
    """A fainter planet (0.03) admitted by construction, the bright one held."""
    truth = _scene(BRIGHT, 0.03)
    terms = _terms(truth)
    held = _absent(truth, rows=(1,))
    fit = _slot(held, terms, 1)
    east, north = _truth_position(truth, 1)
    start = {
        AMP.format(1): jnp.array([0.1]),
        "dynamics.orbits": {
            "east_arcsec": jnp.array([east + 0.004]),
            "north_arcsec": jnp.array([north - 0.004]),
        },
    }
    estimate = measure(fit, start, n_steps=STEPS)
    return fit, estimate, _block_eigenvalues(fit, estimate.value, 1)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(reason="a second scene through the staged recipe")
def test_a_planet_the_data_cannot_place_is_removed_by_the_block_rule(recipe, flat):
    """A faint source admitted by construction is a flat direction; the rule drops it.

    Truth source: the model. A planet of flux scale 0.03 is under one formal error
    above zero on this grid, so its position and amplitude are not constrained: the
    run ends where the planet sits off the detector and feeds the frame through its
    wing with an amplitude of order 1e4, whose formal width is larger than the value.
    The curvature of its block (flux, east, north) has an eigenvalue of rounding size
    against the largest, while the bright planet's block is well conditioned.
    Tolerance basis: floating-point (the rounding floor 1e-8 of the largest eigenvalue
    against a measured ratio of 2e-16 for the flat block and 1e-3 for the bright one;
    the Newton decrement, 3e-3 nat for the flat run and 1e-28 for the bright run,
    against 0.01). The scalar ``regular`` is true for the flat run at three of the
    four starts tried (it is a statement about positive definiteness of a matrix, not
    about identifiability), so the rule does not use it.
    Fails if the rule keeps the flat source or removes the bright one.
    """
    _, estimate, eigenvalues = flat
    assert float(estimate.newton_decrement) < 0.01  # converged by the rule's own test
    assert not _positive(eigenvalues)
    assert not _keep(estimate, eigenvalues)
    bright = _block_eigenvalues(recipe["one"], recipe["estimate"].value, 0)
    assert _positive(bright)
    assert _keep(recipe["estimate"], bright)


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.slow(reason="the staged recipe compiles each stage's program")
def test_residuals_are_within_their_chi_square_band_and_no_source_is_left(recipe):
    """The whitened residual of each exposure is chi-square, and no extra source shows.

    Truth source: the null distribution of the sum of squares of whitened residuals,
    chi-square with the number of scored pixels minus the three fitted scalars as its
    degrees of freedom (the counts are in the hundreds per pixel, so the Pearson
    residual is close to normal). Tolerance basis: monte-carlo, the central 99.8
    percent band (quantiles 0.001 and 0.999) of each of the two exposures, a
    false-failure rate of 0.4 percent for a correct model; the extra-source ``|z|``
    over the 81 trials against 3.8, one seeded realization (a two-sided false-failure
    rate near 2 percent for one noise draw; the measured minimum is -3.7, edge trial).
    Fails if a residual is mis-whitened, or the model leaves a point source the detect
    stage does not see.
    """
    final = recipe["final"]
    for residual in final.residuals(recipe["best"]):
        r = np.asarray(residual)
        scored = np.isfinite(r)
        dof = int(scored.sum()) - 3
        p = stats.chi2.cdf(float((r[scored] ** 2).sum()), dof)
        assert 0.001 < p < 0.999, (p, dof)
    z = np.asarray(recipe["found_extra"].z)
    assert np.nanmax(np.abs(z)) < THRESHOLD, np.nanmin(z)


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(reason="the staged recipe compiles each stage's program")
def test_every_optimizer_call_reports_convergence_by_the_shared_rule(
    recipe, monkeypatch
):
    """``MAP``, ``measure`` and ``Laplace`` each stop on convergence, not on the cap.

    Truth source: the stopping contract. Scaled stages stop when the loss has stalled
    and the scaled gradient norm is below 1e-3. ``MAP`` reports ``stopped_early`` in
    its stage trace with fewer than ``n_steps`` steps and a loss that never rises;
    ``measure`` is read through the trace of the optimizer it calls (a spy on
    ``map_fit``) and its own ``grad_norm`` and ``newton_decrement``; ``Laplace``
    reports ``grad_norm`` and ``newton_decrement`` per mode. Tolerance basis:
    exact-algebra for the flags and step counts; floating-point for the decrement
    bound (below 1e-6 nat, where a converged run measures 1e-28) and the loss trace
    (rises under 1e-9 of the first value).
    Fails if a call runs to its step cap, or reports a point that is not stationary.
    """
    trace = recipe["posterior"].info.stages[0]
    assert trace.stopped_early and trace.n_steps < STEPS
    assert np.all(np.diff(trace.loss) <= 1e-9 * abs(trace.loss[0]))

    seen = []
    real = detection.map_fit

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append(result.stages)
        return result

    monkeypatch.setattr(detection, "map_fit", spy)
    again = measure(recipe["one"], recipe["start0"], n_steps=STEPS)
    (stages,) = seen
    assert stages[0].stopped_early and stages[0].n_steps < STEPS
    assert float(again.newton_decrement) < 1e-6 and float(again.grad_norm) < 1e-3

    laplace = run(
        recipe["final"],
        Laplace(n_draws=64, n_steps=STEPS, init=recipe["estimate"].value),
        jax.random.key(1),
    ).info
    assert int(laplace.n_modes_kept) == 1
    assert float(laplace.newton_decrement[0]) < 1e-6
    assert float(laplace.grad_norm[0]) < 1e-3


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.slow(reason="the radiance render at four points and a MAP run")
def test_disk_aliases_after_map_have_equal_likelihood(make_system):
    """After ``MAP`` on a disk, the four relabelings of the optimum score the same.

    Truth source: the symmetry of the model. The image of ``(i, PA)`` equals that of
    ``(180 - i, PA + 180)`` and ``(i + 180, PA)``, and exchanging the two slopes leaves
    the radial profile unchanged, so the optimum and its three relabelings are one disk.
    The recipe evaluates the log density at the four and reports the highest; the
    returned alias is the one the recipe reports. Tolerance basis: floating-point,
    rtol 1e-10 on the log likelihood (the images agree to about 3e-15 in float64).
    Fails if a forward stops being invariant under a relabeling, or the recipe reports
    an alias whose density is below another's.
    """
    truth = make_system()
    forward = radiance_forward(WL, TIME, grid=GRID)
    data, sigma = _data(truth, forward)
    init = make_system(**INIT)
    fit = _fit(init, data, sigma, forward)
    point = run(fit, MAP(n_steps=800, init=_x(init)), jax.random.key(0))
    assert point.info.stages[0].stopped_early
    x = {path: leaf[0] for path, leaf in point.samples.items()}

    def alias(i, pa, a_in, a_out):
        return {
            **x,
            "midplane_inc_deg": jnp.asarray(i),
            "midplane_position_angle_deg": jnp.asarray(pa),
            "disk.alpha_in": jnp.asarray(a_in),
            "disk.alpha_out": jnp.asarray(a_out),
        }

    i, pa = float(x["midplane_inc_deg"]), float(x["midplane_position_angle_deg"])
    a_in, a_out = float(x["disk.alpha_in"]), float(x["disk.alpha_out"])
    points = {
        "as fitted": x,
        "mirror": alias(180.0 - i, pa + 180.0, a_in, a_out),
        "flipped": alias(i + 180.0, pa, a_in, a_out),
        "exchanged": alias(i, pa, a_out, a_in),
    }
    likelihood = {k: float(fit.log_likelihood(p)) for k, p in points.items()}
    np.testing.assert_allclose(
        list(likelihood.values()), likelihood["as fitted"], rtol=1e-10
    )
    density = {k: float(fit.log_density(p)) for k, p in points.items()}
    reported = max(density, key=density.get)
    assert density["as fitted"] >= density[reported] - 1e-6 * abs(density[reported]), (
        reported,
        density,
    )


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.slow(reason="four short NUTS chains through the renderer")
def test_nuts_with_the_real_priors_from_the_measure_start_mixes(recipe):
    """Four short ``NUTS`` chains from the ``measure`` estimate have ``rhat`` near 1.

    Truth source: the chain diagnostic of Vehtari et al. (rank-normalized split
    ``rhat`` of ``blackjax.diagnostics.potential_scale_reduction``) of the three
    admitted parameters, from 4 chains of 200 warmup and 200 draws. Tolerance basis:
    monte-carlo, ``rhat`` below 1.05 for each parameter, and the posterior mean of the
    flux scale within 4 posterior standard deviations of the injection (the prior is a
    wide log-normal).
    Fails if the chains do not agree, or the posterior does not cover the truth.
    """
    blackjax = pytest.importorskip("blackjax")
    final, start = recipe["final"], recipe["estimate"].value
    post = run(
        final,
        NUTS(n_warmup=200, n_draws=200, n_chains=4, init=start),
        jax.random.key(3),
    )
    chain = np.asarray(post.info.chain)
    flat = {
        "east": post.samples["dynamics.orbits"]["east_arcsec"][:, 0],
        "north": post.samples["dynamics.orbits"]["north_arcsec"][:, 0],
        "flux": post.samples[AMP.format(0)][:, 0],
    }
    for name, draws in flat.items():
        grouped = np.stack([np.asarray(draws)[chain == c] for c in range(4)])
        rhat = float(blackjax.diagnostics.potential_scale_reduction(grouped))
        assert rhat < 1.05, (name, rhat)
    flux = np.asarray(flat["flux"])
    assert abs(flux.mean() - BRIGHT) < 4.0 * flux.std()
