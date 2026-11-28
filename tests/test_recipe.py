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

from types import SimpleNamespace

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
from photomancy.detection import (
    _correlation_inverse,
    curvature_block,
    detect,
    measure,
)
from photomancy.disk import radiance_forward
from photomancy.image import ImageTerm
from photomancy.infer import MAP, NUTS, Fit, ForwardTerm, Laplace, run

pytestmark = pytest.mark.float64_only(reason="float64 Fit coordinates")

# The admission threshold: the 99th percentile of the per-map maximum of z over an
# 81-trial grid and two rolls, for white Poisson noise and a model equal to the truth.
THRESHOLD = 3.8
# The screen of the removal rule: a source whose own block, scaled by its diagonal (a
# unit-free matrix), has a smallest eigenvalue at or below this is a marginal call. It
# measures collinearity, not constraint, so it never removes a source: it sends it to
# the likelihood-ratio refit, the statistic the admission threshold is stated in.
SCREEN = 0.05
# Bright and faint truth flux scales; the faint planet's z on this grid sits between 2.3
# and THRESHOLD, so only the right threshold leaves it out.
BRIGHT, FAINT = 1.0, 0.18
SEPARATIONS, POSITION_ANGLES = (0.04, 0.05), (30.0, 200.0)
ROLLS_DEG = (0.0, 40.0)
EDGE = np.linspace(-0.06, 0.06, 9) + 0.0033  # off the star: a trial on it is NaN
_EAST, _NORTH = (jnp.asarray(a.ravel()) for a in np.meshgrid(EDGE, EDGE))
TRIALS = {
    "dynamics.orbits": {"east_arcsec": _EAST[:, None], "north_arcsec": _NORTH[:, None]}
}
AMP = "planets[{}].flux_scale"
# NumPy seeds of the frame counts, one per roll.
SEEDS = (10, 11)
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


def _block(rows, spread=0.1):
    """The sky offset of the planets in ``rows`` as the fitted coordinates."""
    return SkyPositionBlock(
        east_arcsec=Normal(0.0, spread),
        north_arcsec=Normal(0.0, spread),
        time_d=0.0,
        rows=(rows,) if isinstance(rows, int) else tuple(rows),
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
    """One ``ImageTerm`` per roll, its counts drawn on the host from a fixed seed.

    The detector has no dark current, clock-induced charge, read noise or excess
    noise, so ``observe`` draws Poisson counts at the detector's mean. The same
    counts are drawn here with NumPy at that mean, so the realization the cases
    below are stated on does not move when JAX changes its Poisson sampler.
    """
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
        rates = cp.expected(truth, instrument, exposure)
        mean, _ = instrument.detector.mean_and_variance(rates, exposure.plan)
        counts = np.random.default_rng(SEEDS[i]).poisson(np.asarray(mean))
        image = eqx.tree_at(
            lambda im: im.data, image, jnp.asarray(counts, dtype=image.data.dtype)
        )
        out.append(
            ImageTerm.on_support(truth, image, instrument=instrument, exposure=exposure)
        )
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


def _free_set(z_max):
    """The free entries of the fit that goes on: the sources the rule admits.

    ``z_max`` maps a row to the largest ``z`` of its detection map. A source has free
    entries (a flux law and a place in the position block) only when ``_admit`` says so.
    """
    rows = tuple(row for row, z in z_max.items() if _admit(z))
    free = {AMP.format(row): LogNormal(0.0, 1.0) for row in rows}
    if rows:
        free["dynamics.orbits"] = _block(rows)
    return free


def _block_eigenvalues(fit, x, row):
    """Eigenvalues of one source's block (flux, east, north) scaled by its diagonal."""
    path = AMP.format(row)

    def where(x):
        return (
            x[path],
            x["dynamics.orbits"]["east_arcsec"],
            x["dynamics.orbits"]["north_arcsec"],
        )

    return _scaled_eigenvalues(fit, x, where)


def _scaled_eigenvalues(fit, x, where):
    """Eigenvalues of the block ``where`` picks, scaled by its own diagonal."""
    block = np.asarray(curvature_block(fit.log_likelihood, x, where))
    root = np.sqrt(np.abs(np.diag(block)))
    return np.linalg.eigvalsh(block / np.outer(root, root))


def _rule(fit, start, where, amplitude, amp_index, null, n_steps, more, max_decrement):
    """The removal rule for one admitted source, in the order the how-to gives.

    ``where`` picks the source's block, ``amplitude`` its flux from an entry dictionary,
    ``amp_index`` its place in the flat vector of ``LikelihoodEstimate.sigma`` and
    ``null`` the log likelihood of the fit refit without the source.

    1. ``measure`` from the start.
    2. Not finite, not converged, or ended by the step cap (``stopped_early`` false,
       whatever its decrement): run again from the same start with ``more`` steps. If
       that is still not finite and converged, the source comes out.
    3. Converged: a marginal call when the smallest eigenvalue of the source's own block
       is at most ``SCREEN``, or the amplitude is under ``THRESHOLD`` marginal widths.
    4. A marginal call is decided by the refit: keep only if ``2 delta lnL`` against
       the fit without the source exceeds ``THRESHOLD**2``.

    ``regular`` is not used: it is true at a converged ridge.

    Returns:
        A namespace with ``keep``, ``first``, ``last``, ``marginal`` (the screen fired),
        ``by_block`` and ``by_amplitude`` (which trigger), and ``lr`` (``2 delta lnL``,
        or ``None`` when the refit did not run).
    """
    first = last = measure(fit, start, n_steps=n_steps, max_decrement=max_decrement)
    out = SimpleNamespace(
        keep=False, first=first, last=last, marginal=False, by_block=False,
        by_amplitude=False, lr=None,
    )  # fmt: skip
    if not (bool(first.finite) and bool(first.converged) and bool(first.stopped_early)):
        last = out.last = measure(fit, start, n_steps=more, max_decrement=max_decrement)
        if not (bool(last.finite) and bool(last.converged)):
            return out
    eigenvalues = _scaled_eigenvalues(fit, last.value, where)
    width = float(np.asarray(last.sigma)[amp_index])
    out.by_block = bool(eigenvalues.min() <= SCREEN)
    out.by_amplitude = bool(abs(float(amplitude(last.value))) / width < THRESHOLD)
    out.marginal = out.by_block or out.by_amplitude
    if not out.marginal:
        out.keep = True
        return out
    out.lr = 2.0 * (float(last.log_likelihood) - float(null(last.value)))
    out.keep = out.lr > THRESHOLD**2
    return out


def _keeps(fit, row, start, n_steps=STEPS, more=400, max_decrement=1e-3):
    """The removal rule for the source in ``row`` of an image fit (see ``_rule``).

    The fit without the source is the fit at zero flux: no other entry is free here, so
    there is nothing else to refit.
    """
    path = AMP.format(row)

    def zeroed(x):
        return fit.log_likelihood({**x, path: jnp.zeros_like(x[path])})

    return _rule(
        fit,
        start,
        lambda x: (
            x[path],
            x["dynamics.orbits"]["east_arcsec"],
            x["dynamics.orbits"]["north_arcsec"],
        ),
        lambda x: x[path][0],
        2,  # the flat vector lists the block's east and north, then the flux
        zeroed,
        n_steps,
        more,
        max_decrement,
    )


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

    z_max = {0: float(jnp.nanmax(found0.z)), 1: float(jnp.nanmax(found1.z))}
    final = Fit(held, _free_set({0: z_max[0]}), terms)
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
        z_max=z_max,
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
    correct fit; one seeded realization (frame counts drawn with NumPy seeds 10 and
    11; the formal errors are the square roots of the diagonal of
    ``LikelihoodEstimate.cov``).
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
    amplitude at it, entry for entry (the grid is off the injection, so a start taken
    from the injection would differ in the position).
    Fails if the map peaks elsewhere, or the start is taken from anything but the map.
    """
    east, north = _truth_position(recipe["truth"], 0)
    found, peak = recipe["found0"], recipe["i0"]
    assert peak == _nearest(east, north)
    assert float(found.z[peak]) > THRESHOLD
    start = recipe["start0"]["dynamics.orbits"]
    assert float(start["east_arcsec"][0]) == float(_EAST[peak])
    assert float(start["north_arcsec"][0]) == float(_NORTH[peak])
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
    peak ``z`` is 3.1, so the margin to the threshold is 0.7 and a threshold of 2.3
    would admit it. The free set of the fit that goes on is built through the rule
    from both candidates' maps, so the faint planet's flux law and its place in the
    position block are absent from it.
    Fails if the rule admits a source below the threshold, the faint planet has free
    entries, or the faint planet's map is not above 2.3 (the case would then not test
    the threshold).
    """
    z_max = float(jnp.nanmax(recipe["found1"].z))
    east, north = _truth_position(recipe["truth"], 1)
    near = float(recipe["found1"].z[_nearest(east, north)])
    assert 2.3 < z_max < THRESHOLD
    assert near > 2.3
    assert not _admit(z_max)
    assert _admit(float(jnp.nanmax(recipe["found0"].z)))
    free = _free_set(recipe["z_max"])
    assert AMP.format(0) in free and AMP.format(1) not in free
    assert free["dynamics.orbits"].rows == (0,)
    assert set(free) == set(recipe["final"].free)
    # left out: the model that goes on holds the faint planet at zero flux
    assert float(recipe["final"].model.planets[1].flux_scale[0]) == 0.0


@pytest.fixture(scope="module")
def flat():
    """A fainter planet (0.03) admitted by construction, the bright one held.

    Returns the fit and a function from an offset of the start (east and north, in
    arcseconds, from the truth) to the start entry dictionary.
    """
    truth = _scene(BRIGHT, 0.03)
    terms = _terms(truth)
    held = _absent(truth, rows=(1,))
    fit = _slot(held, terms, 1)
    east, north = _truth_position(truth, 1)

    def start(d_east, d_north):
        return {
            AMP.format(1): jnp.array([0.1]),
            "dynamics.orbits": {
                "east_arcsec": jnp.array([east + d_east]),
                "north_arcsec": jnp.array([north + d_north]),
            },
        }

    return fit, start


class _Gaussian(eqx.Module):
    """The log likelihood of a Gaussian with information ``precision`` at ``mean``."""

    mean: jax.Array
    precision: jax.Array

    def __call__(self, predicted):
        r = predicted - self.mean
        return -0.5 * r @ self.precision @ r


class _RootedGaussian(_Gaussian):
    """The same Gaussian with its Fisher root, so ``measure`` runs in scaled units.

    The information of ``-0.5 r' P r`` is ``J' P J``; with ``P = L L'`` the root
    ``L' predicted`` has the Jacobian ``L' J``. A likelihood without a root makes
    ``measure`` run unscaled, and an unscaled run never stops early.
    """

    def fisher_root(self, predicted):
        return jnp.linalg.cholesky(self.precision).T @ predicted


class _Source(eqx.Module):
    """The flux and sky offset of one source, one scalar each."""

    a: jax.Array
    east: jax.Array
    north: jax.Array


class _SourceForward(eqx.Module):
    def __call__(self, model):
        return jnp.stack([model.a, model.east, model.north])


# A faint source whose flux and east offset are correlated at RIDGE_R in the likelihood
# and whose north offset is independent, written in stored units 100 (flux) and 1000
# (offsets) times the correlation form's, so the units are far from unity.
RIDGE_R = 0.999
RIDGE_MEAN = (0.2, 0.01, -0.02)
RIDGE_UNITS = (100.0, 1e3, 1e3)


@pytest.fixture(scope="module")
def ridge():
    """A flat block by construction: the Gaussian likelihood of a faint, ridged source.

    The log likelihood is ``-0.5 r' P r`` about ``RIDGE_MEAN``, with ``P = D C D``:
    ``C`` the correlation form (unit diagonal, ``RIDGE_R`` between flux and east, zero
    elsewhere, so its eigenvalues are ``1 - RIDGE_R``, ``1`` and ``1 + RIDGE_R``) and
    ``D = diag(RIDGE_UNITS)``. No frame, no noise: the maximum, the information and the
    likelihood ratio against zero flux are known in closed form.

    Returns:
        The fit, a start off the maximum, the log likelihood of the fit without the
        flux (the offsets refit; the quadratic needs no start), and ``2 delta lnL``
        from the Schur complement.
    """
    d = np.array(RIDGE_UNITS)
    corr = np.array([[1.0, RIDGE_R, 0.0], [RIDGE_R, 1.0, 0.0], [0.0, 0.0, 1.0]])
    precision = corr * np.outer(d, d)
    mean = np.array(RIDGE_MEAN)
    term = ForwardTerm(
        _SourceForward(), _RootedGaussian(jnp.asarray(mean), jnp.asarray(precision))
    )
    zero = _Source(jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
    laws = {name: Normal(0.0, 10.0) for name in ("a", "east", "north")}
    fit = Fit(zero, laws, (term,))
    start = {
        "a": jnp.asarray(1.0),
        "east": jnp.asarray(mean[1] + 0.01),
        "north": jnp.asarray(mean[2] - 0.01),
    }
    without = Fit(zero, {k: laws[k] for k in ("east", "north")}, (term,))
    refit = measure(without, {"east": start["east"], "north": start["north"]})
    # the profile over the offsets: the flux's marginal variance is the (a, a) entry
    # of the inverse precision, and 2 delta lnL is its own value squared over it
    lr = mean[0] ** 2 / np.linalg.inv(precision)[0, 0]
    return fit, start, float(refit.log_likelihood), float(lr)


def _ridge_block(x):
    return x["a"], x["east"], x["north"]


def _correlation_signature(fit, estimate, scales):
    """The verdict and smallest eigenvalue of the information after a change of units.

    ``scales`` are the factors that change the coordinates, in the order of the flat
    vector; the information is rescaled by them as they are, then again with every
    coordinate multiplied by the arcsecond to radian factor.
    """
    info = np.asarray(
        fit.information(estimate.value, kind="observed", include_prior=False)
    )
    out = []
    for unit in (1.0, np.pi / 648000.0):
        c = np.asarray(scales) * unit
        _, _, definite, smallest = _correlation_inverse(
            jnp.asarray(info / np.outer(c, c))
        )
        out.append((bool(definite), float(smallest)))
    return out


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(reason="the bright planet of the staged recipe, and the compiled fit")
def test_a_planet_the_data_cannot_place_is_removed_by_the_refit_in_any_units(
    recipe, ridge
):
    """A faint source on a likelihood ridge converges, is screened and is removed.

    Truth source: a Gaussian likelihood built to be flat in one direction (the
    ``ridge`` fixture: no frame, no noise, so no realization decides the path).
    Flux and east offset are correlated at 0.999, so the block's correlation form has
    eigenvalues 0.001, 1 and 1.999 exactly, and the faint flux (0.2, a marginal width of
    0.22 and z = 0.89) is under the threshold. The fit converges to the known maximum,
    the block scaled by its own diagonal has a smallest eigenvalue of 1 - 0.999, below
    the screen, so the rule refits without the source: ``2 delta lnL`` is the squared
    flux over the (a, a) entry of the inverse precision, 0.80, far below
    ``THRESHOLD**2`` = 14.4, and the source comes out. In a quadratic ``2 delta lnL`` is
    ``z**2``, so the amplitude trigger also fires here; the block eigenvalue is what the
    case pins, and the correlated-position case below pins the block firing alone. The
    bright planet of the imaging recipe (z about 26) is not screened (block
    eigenvalue near 1) and is kept. The verdicts and the smallest eigenvalue of the
    information in correlation form do not change when the coordinates are rescaled
    (the ridge is written in stored units 100 and 1000 times the correlation form's, the
    imaging fit in arcseconds and flux scale; both then by 1e-4 and by arcseconds to
    radians).
    Tolerance basis: exact-algebra (rtol 1e-6 on the eigenvalue, on the maximum and on
    ``2 delta lnL``, the quadratic being solved to the Newton polish; the screen 0.05
    lies between 0.001 and the bright planet's 0.97); the rescaled eigenvalue equals the
    original to rtol 1e-9 (floating-point, one eigenvalue of a 3 by 3 matrix whose
    scaling rounds at 1e-16). ``regular`` is true for the ridge (the matrix is
    invertible), so the rule does not use it.
    Fails if the rule keeps the ridged source or removes the bright one, a verdict
    depends on the units (a positive-definite test or an eigenvalue taken on the
    stored-unit matrix), the screen does not see the correlation between flux and
    position (a block that drops its cross terms), or the removal is not made by the
    likelihood ratio.
    """
    fit, start, null_lnl, lr = ridge
    verdict = _rule(
        fit,
        start,
        _ridge_block,
        lambda x: x["a"],
        0,
        lambda x: null_lnl,
        100,
        400,
        1e-3,
    )
    first = verdict.first
    assert bool(first.finite) and bool(first.converged) and first is verdict.last
    assert bool(first.stopped_early) and float(first.newton_decrement) < 1e-3
    x = first.value
    got = np.array([float(x["a"]), float(x["east"]), float(x["north"])])
    # the offset from the maximum in the stored units' own widths (1 / units)
    off = (got - RIDGE_MEAN) * np.array(RIDGE_UNITS)
    np.testing.assert_allclose(off, 0.0, rtol=0, atol=1e-6)
    eigenvalues = _scaled_eigenvalues(fit, x, _ridge_block)
    np.testing.assert_allclose(eigenvalues.min(), 1.0 - RIDGE_R, rtol=1e-6)
    assert verdict.by_block and verdict.marginal and verdict.by_amplitude
    assert verdict.lr is not None and verdict.lr == pytest.approx(lr, rel=1e-6)
    assert verdict.lr < THRESHOLD**2 and not verdict.keep
    bright = _keeps(recipe["one"], 0, recipe["start0"])
    assert bright.keep and not bright.marginal and bright.lr is None
    assert bool(bright.last.regular)
    flux_offsets = np.array([1e-4, 1.0, 1.0])
    imaging = np.array([1.0, 1.0, 1e-4])  # the flat vector: east, north, flux
    for model, estimate, scales in (
        (fit, first, flux_offsets),
        (recipe["one"], bright.last, imaging),
    ):
        (base, original), *rescaled = _correlation_signature(model, estimate, scales)
        assert base == bool(estimate.positive_definite)
        np.testing.assert_allclose(original, float(estimate.min_eigenvalue), rtol=1e-9)
        for definite, smallest in rescaled:
            assert definite == base
            np.testing.assert_allclose(smallest, original, rtol=1e-9)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.slow(reason="a second scene through the staged recipe")
def test_a_source_the_block_screen_passes_is_removed_through_the_amplitude_trigger(
    flat,
):
    """A converged source with a well-conditioned block but no amplitude is removed.

    Truth source: the model (the flat scene, start 0.004 arcsec west and 0.004 north of
    the truth). The run converges at a point where the block's smallest scaled
    eigenvalue is 0.50, above the screen, so the block does not fire; the amplitude is
    under 3.8 of its marginal widths, so the amplitude trigger does, and the refit
    gives ``2 delta lnL`` of 1.1 against the zero-flux fit, below 14.4: removed.
    Tolerance basis: floating-point (the eigenvalue 0.50 against 0.05, and the ratio
    against 14.4, are far from their limits).
    Fails if the removal depends on the block screen alone (this source would be kept),
    or the amplitude trigger is absent.
    """
    fit, start = flat
    verdict = _keeps(fit, 1, start(-0.004, 0.004), n_steps=400)
    assert bool(verdict.first.converged) and verdict.first is verdict.last
    assert _block_eigenvalues(fit, verdict.last.value, 1).min() > SCREEN
    assert verdict.by_amplitude and not verdict.by_block
    print("flat1 2 delta lnL:", verdict.lr)
    assert verdict.lr < THRESHOLD**2 and not verdict.keep


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(reason="a compiled optimizer run, then a rerun, on a constructed fit")
def test_a_ridge_walk_ended_by_the_step_cap_is_rerun_because_of_the_cap(ridge):
    """A run ended by the step cap is rerun on the cap, even when it is converged.

    Truth source: the Gaussian likelihood of the ``ridge`` fixture, a quadratic whose
    maximum, information and ``2 delta lnL`` are known. The step cap is one L-BFGS
    iteration, so the cap binds by construction: the stop rule needs the loss to have
    stalled, and one step from a start 1 flux unit and 0.01 arcsec off the maximum
    cannot stall it. The Newton polish then finishes the quadratic: the estimate is
    finite, ``converged`` with a decrement at rounding level (far under the 1e-3 nat
    tolerance), and still ``stopped_early`` false. The rule must rerun it for that
    reason alone, with 400 steps; the rerun's stop rule ends it (``stopped_early``) and
    the refit then removes the source (``2 delta lnL`` 0.80 against 14.4).
    Tolerance basis: exact-algebra for the flags (the decrement of a polished quadratic
    is rounding, 1e-6 nat at most here; the cap of 1 against the 400 of the rerun).
    Fails if a run that the cap ended reports ``stopped_early`` (the cap is not
    recorded), or the rule skips the rerun because the capped run is converged.
    """
    fit, start, null_lnl, lr = ridge
    verdict = _rule(
        fit, start, _ridge_block, lambda x: x["a"], 0, lambda x: null_lnl, 1, 400, 1e-3
    )
    first, last = verdict.first, verdict.last
    assert bool(first.finite) and bool(first.converged) and int(first.n_steps) == 1
    assert float(first.newton_decrement) < 1e-6
    assert not bool(first.stopped_early)
    assert last is not first and int(last.n_steps) > 1
    assert bool(last.converged) and bool(last.stopped_early)
    assert verdict.lr == pytest.approx(lr, rel=1e-6) and not verdict.keep


class _Pair(eqx.Module):
    """An amplitude and a second parameter, one scalar each."""

    a: jax.Array
    d: jax.Array


class _PairForward(eqx.Module):
    def __call__(self, model):
        return jnp.stack([model.a, model.d])


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.slow(
    reason="compiles a correlated astrometric refit in the staged imaging recipe, 6 "
    "s projected on CI"
)
def test_a_bright_source_with_a_correlated_position_is_kept_by_the_refit():
    """A source at z = 26 whose flux and separation correlate at 0.97 is kept.

    Truth source: a Gaussian likelihood with unit marginal widths, correlation 0.97 and
    its maximum at amplitude 26 (z = 26 marginally). The correlation-form information
    has eigenvalues 1 plus and minus 0.97, so the block's smallest is 0.03, under the
    screen: the screen fires and the refit decides. Refit without the amplitude (the
    separation free) the maximum of the likelihood is ``-0.5 * 26**2``, so
    ``2 delta lnL = 676``, far above 14.4: kept. A removal at the screen would have
    removed it.
    Tolerance basis: exact-algebra (rtol 1e-6 on the eigenvalue and on 676; the
    quadratic needs no iteration beyond the polish).
    Fails if the block screen removes a well detected source, or the refit statistic is
    not the likelihood ratio of the marginal significance.
    """
    rho = 0.97
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mean = jnp.array([26.0, 3.0])
    term = ForwardTerm(_PairForward(), _Gaussian(mean, jnp.asarray(np.linalg.inv(cov))))
    laws = {"a": Normal(0.0, 100.0), "d": Normal(0.0, 100.0)}
    fit = Fit(_Pair(jnp.zeros(()), jnp.zeros(())), laws, (term,))
    start = {"a": jnp.asarray(20.0), "d": jnp.asarray(1.0)}
    without = Fit(_Pair(jnp.zeros(()), jnp.zeros(())), {"d": laws["d"]}, (term,))

    def null(x):
        return measure(without, {"d": jnp.asarray(1.0)}).log_likelihood

    verdict = _rule(
        fit,
        start,
        lambda x: (x["a"], x["d"]),
        lambda x: x["a"],
        0,
        null,
        100,
        400,
        1e-3,
    )
    eigenvalues = _scaled_eigenvalues(
        fit, verdict.last.value, lambda x: (x["a"], x["d"])
    )
    np.testing.assert_allclose(eigenvalues.min(), 1.0 - rho, rtol=1e-6)
    assert verdict.by_block and not verdict.by_amplitude
    assert verdict.lr == pytest.approx(26.0**2, rel=1e-6)
    assert verdict.keep


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
    The recipe evaluates the log density at the four and reports the highest (printed
    here); the likelihoods are what the test asserts, because the density differs
    between aliases by the prior and the volume term. Tolerance basis: floating-point,
    rtol 1e-10 on the log likelihood (the images agree to about 3e-15 in float64).
    Fails if a forward stops being invariant under a relabeling.
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
    print("alias reported:", max(density, key=density.get), density)


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
@pytest.mark.slow(reason="four short NUTS chains through the renderer")
def test_nuts_with_the_real_priors_from_dispersed_starts_mixes(recipe):
    """Four short ``NUTS`` chains from dispersed starts agree and cover the injection.

    Truth source: the split ``rhat`` of Gelman and Rubin (1992) as ``blackjax`` computes
    it, each chain halved so that a trend inside a chain shows, and the effective sample
    size of Geyer's initial monotone sequence (``blackjax.diagnostics``), for the three
    admitted parameters, from 4 chains of 200 warmup and 200 draws. The chains start at
    draws of ``N(estimate, 4 cov)`` (twice the formal width in each coordinate), so
    agreement is the sampler's and not the shared start's. Tolerance basis: monte-carlo,
    split ``rhat`` below 1.05, the effective size above 100 of the 800 draws per
    parameter, and the posterior mean of the flux scale within 4 posterior standard
    deviations of the injection (the prior is a wide log-normal); one seeded run.
    Fails if the chains do not agree, mix too slowly to be useful, or the posterior does
    not cover the truth.
    """
    blackjax = pytest.importorskip("blackjax")
    final, estimate = recipe["final"], recipe["estimate"]
    flat, unravel = jax.flatten_util.ravel_pytree(estimate.value)
    chol = jnp.linalg.cholesky(4.0 * estimate.cov)
    noise = jax.random.normal(jax.random.key(7), (4, flat.size))
    start = jax.vmap(unravel)(flat + noise @ chol.T)
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
        halves = np.concatenate([grouped[:, :100], grouped[:, 100:]])
        rhat = float(blackjax.diagnostics.potential_scale_reduction(halves))
        size = float(blackjax.diagnostics.effective_sample_size(grouped))
        assert rhat < 1.05, (name, rhat)
        assert size > 100.0, (name, size)
    flux = np.asarray(flat["flux"])
    assert abs(flux.mean() - BRIGHT) < 4.0 * flux.std()
