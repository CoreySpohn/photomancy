"""Disk fitting: recover a skyscapes Scene's disk via the scene-as-PyTree engine."""

import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import rad2deg
from hwoutils.epoch import Epoch
from hwoutils.model_paths import resolve
from optixstuff.grids import PixelGrid, SkyOrientation
from orbix.system import IsotropicInclination
from priorities import Normal

from photomancy.disk import default_disk_prior, radiance_forward
from photomancy.infer import MAP, Fit, ForwardTerm, Laplace, run
from photomancy.infer.diagnostics import weighted_mean
from photomancy.infer.likelihoods import gaussian_loglike

WL, TIME = 500.0, 0.0
GRID = PixelGrid.regular((51, 51), 0.2, "arcsec", orientation=SkyOrientation(0.0))


GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "forward_fit_golden.json").read_text()
)["disk"]
# The six free paths of default_disk_prior.
PATHS = (
    "disk.sma_AU",
    "disk.alpha_in",
    "disk.alpha_out",
    "disk.ksi0_AU",
    "midplane_inc_deg",
    "midplane_position_angle_deg",
)
INIT = dict(sma=56.0, alpha_in=4.0, alpha_out=-6.0, ksi0=1.3, incl=48.0, pa=22.0)


def _x(system):
    """The free entries of ``system``, as the dict a ``Fit`` takes."""
    return {path: jnp.asarray(resolve(system, path)) for path in PATHS}


def _data(truth, forward, rng_seed=0):
    """The truth image with 2 percent Gaussian noise, and the noise sigma."""
    clean = forward(truth)
    sigma = 0.02 * float(jnp.max(jnp.abs(clean)))
    rng = np.random.default_rng(rng_seed)
    return jnp.asarray(np.asarray(clean) + rng.normal(0.0, sigma, clean.shape)), sigma


def _fit(system, data, sigma, forward, prior_system=None):
    """Build the disk fit from the geometry of ``prior_system`` (default ``system``)."""
    prior = default_disk_prior(system if prior_system is None else prior_system)
    return Fit(system, prior, (ForwardTerm(forward, gaussian_loglike(data, sigma)),))


@pytest.fixture(scope="module")
def make_system():
    """Import the scene stack only on the worker that executes these disk tests."""
    from skyscapes.disk import GraterDisk
    from skyscapes.scene import FlatStar, Scene
    from zodi import DensityDust

    def build(sma=50.0, alpha_in=5.0, alpha_out=-5.0, ksi0=1.0, incl=55.0, pa=30.0):
        """Star + GraterDisk Scene; orientation held as arrays so it is fittable."""
        disk = GraterDisk(
            sma_AU=jnp.array(sma),
            alpha_in=jnp.array(alpha_in),
            alpha_out=jnp.array(alpha_out),
            ksi0_AU=jnp.array(ksi0),
            gamma=jnp.array(2.0),
            beta=jnp.array(1.0),
            rmin_AU=jnp.array(5.0),
            rmax_AU=jnp.array(200.0),
            wavelengths_nm=jnp.array([400.0, 1000.0]),
            g_HG_grid=jnp.array([0.3, 0.3]),
            Ag_grid=jnp.array([0.5, 0.5]),
            amplitude=DensityDust(
                level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
            ),
            n_slices_los=31,
        )
        star = FlatStar(flux_phot_per_nm_m2=1e9)
        return Scene.from_classical(
            star=star,
            planets=(),
            disk=disk,
            star_mass_Msun=1.0,
            planet_mass_Msun=jnp.empty(0),
            dist_pc=10.0,
            a_AU=jnp.empty(0),
            e=jnp.empty(0),
            i_rad=jnp.empty(0),
            W_rad=jnp.empty(0),
            w_rad=jnp.empty(0),
            M0_rad=jnp.empty(0),
            epoch=Epoch(2451545.0),
            midplane_inc_deg=jnp.array(incl),
            midplane_position_angle_deg=jnp.array(pa),
        )

    return build


@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_the_disk_has_no_orientation_leaf_to_free(make_system):
    """Fails if a disk grows an orientation leaf a fit can free but a read never uses.

    The scene's midplane leaves are the only orientation; ``Fit`` refuses an entry
    at ``disk.inclination_deg`` or ``disk.position_angle_deg`` and names the path.
    Construction only, so there is nothing to compile.
    """
    system = make_system()
    for leaf in ("inclination_deg", "position_angle_deg"):
        with pytest.raises(ValueError, match=f"unknown model path 'disk.{leaf}'"):
            Fit(system, {f"disk.{leaf}": Normal(jnp.asarray(50.0), 5.0)})


# Medium: The disk image and its six-parameter gradient exceed the unit compile budget.
@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.medium
def test_disk_fit_is_finite_and_differentiable(make_system):
    """The disk fit is finite and grad-safe at the system's own values."""
    system = make_system()
    forward = radiance_forward(WL, TIME, grid=GRID)
    image = forward(system)
    sigma = 0.02 * float(jnp.max(jnp.abs(image)))
    fit = _fit(system, image, sigma, forward)
    assert fit.walls == {}
    assert tuple(fit.free) == PATHS
    x0 = _x(system)
    edge_on = {**x0, "midplane_inc_deg": jnp.asarray(90.0)}
    assert fit.to_real(edge_on)["midplane_inc_deg"] == 0.0
    assert jnp.isfinite(fit.log_density(x0))
    grad = jax.grad(fit.log_density)(x0)
    assert all(jnp.isfinite(g) for g in grad.values())
    assert float(fit.model_at(x0).disk.sma_AU) == 50.0


# Medium: the golden needs the disk render at three points.
@pytest.mark.float64_only(reason="stored float64 V&V reference values")
@pytest.mark.medium
def test_fit_density_matches_the_prior_adjusted_golden(make_system):
    """Compare the legacy residual sum plus the normalized disk prior at three points.

    The stored values adjust the legacy densities by the change in log prior.
    Subtract the Gaussian normalization from the fit to compare in the same measure.
    Tolerance basis: floating-point (rtol 1e-12, float64).
    Fails if: a path, a law, a scale or the forward changed in the port.
    """
    forward = radiance_forward(WL, TIME, grid=GRID)
    data, sigma = _data(make_system(), forward)
    center = make_system(**INIT)
    fit = _fit(center, data, sigma, forward)
    n = GOLDEN["n_pixels"]
    assert data.size == n
    constant = -0.5 * n * math.log(2.0 * math.pi) - n * math.log(sigma)
    for name, point in GOLDEN["points"].items():
        x = _x(make_system(**point))
        np.testing.assert_allclose(
            float(fit.log_density(x)) - constant,
            GOLDEN["log_density"][name],
            rtol=1e-12,
        )


@pytest.mark.float64_only(reason="float64 posterior coordinates and log weights")
@pytest.mark.slow
def test_map_and_laplace_recover_disk_shape_and_orientation(make_system):
    """MAP and Laplace recover the truth disk shape and orientation from noise.

    Truth: the generating disk. Tolerances are the recovery bounds of ``check`` (15
    percent in the semi-major axis, 0.4 AU, 6 degrees). The default MAP must stop
    early (converged: loss stalled and scaled gradient small) on a loss trace that
    never rises by more than 1e-9 of its first value, a floating-point bound (the rise
    measured here is 8e-17 of the first value in 30 steps), and the final iterate's
    objective is no worse than the trace minimum within the same bound. Its log
    density must equal that of the unscaled run within 1e-3 nat: both reach the same
    optimum, the scale changes only the path.
    Fails if the default stage has no stopping rule: converged, it takes steps that
    raise the loss and can end in an alias of the optimum.
    """
    truth = make_system()  # sma=50, ai=5, ao=-5, ksi0=1, incl=55, pa=30
    forward = radiance_forward(WL, TIME, grid=GRID)
    data, sigma = _data(truth, forward)
    init = make_system(**INIT)
    fit = _fit(init, data, sigma, forward)
    x0 = _x(init)

    def check(x):
        assert abs(float(x["disk.sma_AU"]) - 50.0) / 50.0 < 0.15
        assert abs(float(x["disk.ksi0_AU"]) - 1.0) < 0.4
        assert abs(float(x["midplane_inc_deg"]) - 55.0) < 6.0
        assert abs(float(x["midplane_position_angle_deg"]) - 30.0) < 6.0

    # Default MAP: scaled, and stopped once the objective stops improving.
    point = run(fit, MAP(n_steps=800, init=x0), jax.random.key(0))
    trace = point.info.stages[0]
    assert trace.stopped_early and trace.n_steps < 800
    ours = {path: leaf[0] for path, leaf in point.samples.items()}
    final = -float(fit.real_log_prob(fit.to_real(ours)))
    assert np.all(np.diff(trace.loss) <= 1e-9 * abs(trace.loss[0]))
    assert final <= trace.loss.min() + 1e-9 * abs(trace.loss[0])
    check(ours)
    control = run(fit, MAP(n_steps=800, init=x0, scale=False), jax.random.key(0))
    theirs = {path: leaf[0] for path, leaf in control.samples.items()}
    assert abs(float(fit.log_density(ours)) - float(fit.log_density(theirs))) < 1e-3

    post = run(fit, Laplace(n_draws=256, n_steps=800, init=x0), jax.random.key(1))
    assert int(post.info.n_modes_kept) == 1
    check({p: weighted_mean(post.log_weights, v) for p, v in post.samples.items()})


# Medium: verify the relabelings through the disk renderer.
@pytest.mark.medium
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_disk_aliases_have_equal_likelihood_and_equal_or_zero_prior(make_system):
    """The forward and physical-coordinate prior respect the Grater symmetries.

    The (i, PA) mirrors, full turns in PA and exchanged slopes give the same
    likelihood to rtol 1e-12 (the images agree to about 3e-15 in float64).
    Fails if a supported mirror has different prior density, an out-of-range
    label has any density, or a PA half-turn alone preserves the near side.
    """
    forward = radiance_forward(WL, TIME, grid=GRID)
    data, sigma = _data(make_system(), forward)
    # Put both mirror labels inside the PA interval, away from its endpoints.
    fit = _fit(make_system(pa=60.0), data, sigma, forward)
    points = [
        make_system(),
        make_system(incl=125.0, pa=210.0),
        make_system(incl=235.0),
        make_system(pa=390.0),
        make_system(alpha_in=-5.0, alpha_out=5.0),
    ]
    values = [float(fit.log_likelihood(_x(p))) for p in points]
    np.testing.assert_allclose(values, values[0], rtol=1e-12)
    log_priors = [float(fit.log_prior(_x(p))) for p in points]
    assert math.isfinite(log_priors[0])
    assert log_priors[1] == log_priors[0]
    assert all(p == -math.inf for p in log_priors[2:])
    # A half-turn alone reverses the near side in this anisotropically scattering disk.
    half_turn = float(fit.log_likelihood(_x(make_system(pa=210.0))))
    assert not np.isclose(half_turn, values[0], rtol=1e-12, atol=0)


class _Scaled(eqx.Module):
    """A swapped forward, standing in for a coronagraph stage: 0.7 times the render."""

    base: eqx.Module

    def __call__(self, system):
        return 0.7 * self.base(system)


# Medium: this consumer can pay for the deferred scene-stack import.
@pytest.mark.medium
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_forward_is_swappable(make_system):
    """A swapped forward module fits end to end, and a closure is refused."""
    system = make_system()
    forward = _Scaled(radiance_forward(WL, TIME, grid=GRID))
    image = forward(system)
    sigma = 0.02 * float(jnp.max(jnp.abs(image)))
    fit = _fit(system, image, sigma, forward)
    x0 = _x(system)
    assert jnp.isfinite(fit.log_density(x0))
    assert all(jnp.isfinite(g) for g in jax.grad(fit.log_density)(x0).values())
    with pytest.raises(TypeError, match=r"eqx\.Module"):
        ForwardTerm(lambda s: 0.7 * forward.base(s), gaussian_loglike(image, sigma))


# Medium: this consumer can pay for the deferred scene-stack import.
@pytest.mark.medium
def test_default_disk_prior_support_and_centers(make_system):
    """Check the isotropic degree measure, one PA turn and the ordered slopes."""
    system = make_system(incl=125.0, pa=-340.0)
    prior = default_disk_prior(system)
    assert tuple(prior) == PATHS
    x = _x(system)
    assert all(jnp.isfinite(law.log_prob(x[path])) for path, law in prior.items())
    for path in ("disk.sma_AU", "disk.ksi0_AU"):
        np.testing.assert_array_max_ulp(prior[path].icdf(jnp.asarray(0.5)), x[path], 2)
    inc = prior["midplane_inc_deg"]
    np.testing.assert_array_equal(inc.support(), [0.0, 180.0])
    assert jnp.all(jnp.isneginf(inc.log_prob(jnp.asarray([0.0, 180.0]))))
    # One quarter of isotropic orientations lie within 60 degrees of the pole.
    angles = jnp.asarray([60.0, 90.0])
    quantiles = jnp.asarray([0.25, 0.5])
    np.testing.assert_array_max_ulp(inc.icdf(quantiles), angles, 2)
    np.testing.assert_array_max_ulp(inc.cdf(angles), quantiles, 2)
    peak = jnp.asarray(math.log(math.pi / 360.0))
    np.testing.assert_array_max_ulp(inc.log_prob(angles[1]), peak, 2)
    key = jax.random.key(7)
    draws = inc.sample(key, (3,), dtype=angles.dtype)
    assert draws.dtype == angles.dtype
    np.testing.assert_array_equal(
        draws, IsotropicInclination().sample(key, (3,), dtype=angles.dtype) * rad2deg
    )
    np.testing.assert_array_equal(
        prior["midplane_position_angle_deg"].support(), [-520.0, -160.0]
    )
    for path in ("disk.alpha_in", "disk.alpha_out"):
        law = prior[path]
        assert jnp.isneginf(law.log_prob(-x[path]))
        assert law.log_prob(x[path]) > law.log_prob(2 * x[path])
