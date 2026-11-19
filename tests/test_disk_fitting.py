"""Disk fitting: recover a skyscapes Scene's disk via the scene-as-PyTree engine."""

import json
import math
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from hwoutils.epoch import Epoch  # noqa: E402
from hwoutils.model_paths import resolve  # noqa: E402
from optixstuff.grids import PixelGrid, SkyOrientation  # noqa: E402
from priorities import LogNormal, Normal  # noqa: E402

from photomancy.disk import default_disk_prior, radiance_forward  # noqa: E402
from photomancy.infer import MAP, Fit, ForwardTerm, Laplace, run  # noqa: E402
from photomancy.infer.diagnostics import weighted_mean  # noqa: E402
from photomancy.infer.likelihoods import gaussian_loglike  # noqa: E402

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


# Medium: The disk image and its six-parameter gradient exceed the unit compile budget.
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


@pytest.mark.slow
def test_map_and_laplace_recover_disk_shape_and_orientation(make_system):
    """MAP and Laplace recover the truth disk shape and orientation from noise."""
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

    point = run(fit, MAP(n_steps=800, init=x0), jax.random.key(0))
    check({path: leaf[0] for path, leaf in point.samples.items()})

    post = run(fit, Laplace(n_draws=256, n_steps=800, init=x0), jax.random.key(1))
    assert int(post.info.n_modes_kept) == 1
    check({p: weighted_mean(post.log_weights, v) for p, v in post.samples.items()})


class _Scaled(eqx.Module):
    """A swapped forward, standing in for a coronagraph stage: 0.7 times the render."""

    base: eqx.Module

    def __call__(self, system):
        return 0.7 * self.base(system)


# Medium: this consumer can pay for the deferred scene-stack import.
@pytest.mark.medium
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
