"""Disk fitting: recover a skyscapes System's disk via the scene-as-PyTree engine."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbix.kepler.shortcuts.grid import get_grid_solver  # noqa: E402
from skyscapes.disk import GraterDisk  # noqa: E402
from skyscapes.scene import FlatStar, System  # noqa: E402

from photomancy.backends import LaplaceBackend  # noqa: E402
from photomancy.disk import (  # noqa: E402
    build_disk_logdensity,
    surface_brightness_forward,
)

WL, TIME = 500.0, 0.0
SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)


def _FIT(s):
    """The fitted leaves: disk radial/vertical shape + shared orientation."""
    return [
        s.disk.sma_AU,
        s.disk.alpha_in,
        s.disk.alpha_out,
        s.disk.ksi0_AU,
        s.midplane_inc_deg,
        s.midplane_pa_deg,
    ]


def _make_system(sma=50.0, alpha_in=5.0, alpha_out=-5.0, ksi0=1.0, incl=55.0, pa=30.0):
    """Star + GraterDisk System; orientation held as arrays so it is fittable."""
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
        nx=51,
        ny=51,
        pixel_scale_arcsec=0.2,
        dist_pc=10.0,
        n_slices_los=31,
    )
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    return System(
        star=star,
        planets=(),
        disk=disk,
        trig_solver=SOLVER,
        midplane_inc_deg=jnp.array(incl),
        midplane_pa_deg=jnp.array(pa),
    )


def test_build_disk_logdensity_is_finite_and_differentiable():
    """The disk logdensity builds over the fitted leaves and is grad-safe at z0."""
    system = _make_system()
    forward = surface_brightness_forward(WL, TIME)
    image = forward(system)
    sigma = 0.02 * float(jnp.max(jnp.abs(image)))
    logdensity, z0, unravel = build_disk_logdensity(
        system, image, fit_leaves=_FIT, noise_sigma=sigma, forward=forward
    )
    assert z0.shape == (6,)  # sma, alpha_in, alpha_out, ksi0, incl, pa
    assert jnp.isfinite(logdensity(z0))
    assert jnp.all(jnp.isfinite(jax.grad(logdensity)(z0)))
    # unravel recovers the fitted leaves in System structure
    assert float(unravel(z0).disk.sma_AU) == 50.0


def test_laplace_recovers_disk_shape_and_orientation():
    """Laplace MAP recovers the truth disk shape + orientation from a noisy image."""
    truth = _make_system()  # sma=50, ai=5, ao=-5, ksi0=1, incl=55, pa=30
    forward = surface_brightness_forward(WL, TIME)
    clean = forward(truth)
    sigma = 0.02 * float(jnp.max(jnp.abs(clean)))
    rng = np.random.default_rng(0)
    data = jnp.asarray(np.asarray(clean) + rng.normal(0.0, sigma, clean.shape))

    init = _make_system(
        sma=56.0, alpha_in=4.0, alpha_out=-6.0, ksi0=1.3, incl=48.0, pa=22.0
    )
    logdensity, z0, unravel = build_disk_logdensity(
        init, data, fit_leaves=_FIT, noise_sigma=sigma, forward=forward
    )
    post = LaplaceBackend(n_steps=800, min_eigenvalue=1e-6).run(logdensity, z0)
    fit = unravel(post.mean)

    assert abs(float(fit.disk.sma_AU) - 50.0) / 50.0 < 0.15
    assert abs(float(fit.disk.ksi0_AU) - 1.0) < 0.4
    assert abs(float(fit.midplane_inc_deg) - 55.0) < 6.0
    assert abs(float(fit.midplane_pa_deg) - 30.0) < 6.0


def test_forward_is_swappable():
    """A swapped forward (stand-in for a coronagraph stage) fits end to end."""
    system = _make_system()
    base = surface_brightness_forward(WL, TIME)

    def alt_forward(s):  # stand-in for a coronagraph/PSF stage
        return 0.7 * base(s)

    image = alt_forward(system)
    sigma = 0.02 * float(jnp.max(jnp.abs(image)))
    logdensity, z0, _ = build_disk_logdensity(
        system, image, fit_leaves=_FIT, noise_sigma=sigma, forward=alt_forward
    )
    assert jnp.isfinite(logdensity(z0))
    assert jnp.all(jnp.isfinite(jax.grad(logdensity)(z0)))
