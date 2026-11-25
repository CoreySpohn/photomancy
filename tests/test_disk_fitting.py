"""Disk fitting: recover a skyscapes Scene's disk via the scene-as-PyTree engine."""

import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hwoutils.model_paths import resolve
from optixstuff.grids import PixelGrid, SkyOrientation
from priorities import LogNormal, Normal

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
    """The disk fit: laws centered on ``prior_system`` (default ``system``)."""
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
    assert tuple(fit.free) == PATHS
    x0 = _x(system)
    assert jnp.isfinite(fit.log_density(x0))
    grad = jax.grad(fit.log_density)(x0)
    assert all(jnp.isfinite(g) for g in grad.values())
    assert float(fit.model_at(x0).disk.sma_AU) == 50.0


# Medium: the golden needs the disk render at three points.
@pytest.mark.float64_only(reason="stored float64 V&V reference values")
@pytest.mark.medium
def test_fit_density_matches_the_legacy_density_up_to_the_gaussian_constant(
    make_system,
):
    """The fit density minus the Gaussian normalization is the stored legacy density.

    The legacy density was the unnormalized Gaussian plus the default prior, captured
    at three points before the port (``data/forward_fit_golden.json``). ``ForwardTerm``
    with ``gaussian_loglike`` adds ``-N/2 log(2 pi) - N log(sigma)``, so subtracting it
    leaves the same prior and the same residual sum.
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
            GOLDEN["legacy_log_density"][name],
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


# Medium: the disk render at four points.
@pytest.mark.medium
@pytest.mark.float64_only(reason="float64 Fit coordinates")
def test_disk_aliases_have_equal_likelihood(make_system):
    """The relabelings of one disk give the same image and the same log likelihood.

    Truth source: the symmetry of the model, not a stored number. The image of
    (i, PA) equals that of (180 - i, PA + 180) and (i + 180, PA), and exchanging the
    two slopes with both signs flipped is the same radial profile; the four points
    (55, 30), (125, 210), (235, 30) and slopes (-5, 5) are one disk. Tolerance basis:
    floating-point, rtol 1e-12 on the log likelihood (the images agree to about
    3e-15, float64). The log prior differs between them by 11 to 51 nat, which is
    why a fit can end in an alias.
    Fails if a forward stops being invariant under one of the relabelings.
    """
    forward = radiance_forward(WL, TIME, grid=GRID)
    data, sigma = _data(make_system(), forward)
    fit = _fit(make_system(), data, sigma, forward)
    points = [
        make_system(),
        make_system(incl=125.0, pa=210.0),
        make_system(incl=235.0),
        make_system(alpha_in=-5.0, alpha_out=5.0),
    ]
    values = [float(fit.log_likelihood(_x(p))) for p in points]
    np.testing.assert_allclose(values, values[0], rtol=1e-12)


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
@pytest.mark.float64_only(reason="float64 Fit coordinates")
@pytest.mark.medium
def test_default_disk_prior_is_keyed_by_path_and_centers_on_system(make_system):
    """default_disk_prior is one law per path, centered on the system's values."""
    system = make_system()  # sma=50, ai=5, ao=-5, ksi0=1, incl=55, pa=30
    prior = default_disk_prior(system)
    assert tuple(prior) == PATHS
    assert isinstance(prior["disk.sma_AU"], LogNormal)
    assert isinstance(prior["disk.ksi0_AU"], LogNormal)
    assert isinstance(prior["midplane_inc_deg"], Normal)

    # the laws are accepted by a Fit over the system and sit on its values: each
    # median recovers the current value.
    fit = Fit(system, prior)
    x0 = _x(system)
    assert jnp.isfinite(fit.log_prior(x0))
    for path, law in prior.items():
        np.testing.assert_allclose(
            float(law.icdf(jnp.asarray(0.5))), float(x0[path]), rtol=1e-5
        )
    # weakly informative: the prior density is higher at the truth than far from it.
    far = {path: value + 30.0 for path, value in x0.items()}
    far["disk.sma_AU"] = x0["disk.sma_AU"] * 10.0
    assert fit.log_prior(x0) > fit.log_prior(far)
