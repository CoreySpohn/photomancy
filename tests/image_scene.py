"""A small synthetic coronagraph, scene and exposure for the image-term tests.

The coronagraph is analytic: a Gaussian off-axis response of 0.35 lambda/D with a
constant 0.3 sky transmission and a constant leakage, so a source at any sky position
has a closed-form image. Nothing here reads a file.
"""

import coronagraphoto as cp
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from optixstuff import ConstantThroughput, Detector, Instrument, SimplePrimary
from optixstuff.coronagraph import AbstractTableCoronagraph, SourcePSFs
from optixstuff.exposure import FramePlan
from optixstuff.grids import Frame, Image, Measure, PixelGrid
from optixstuff.performance import PerformanceCurves, SampledProfile
from optixstuff.spectral import Band
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import FlatStar, Planet, Scene

N_PIX = 24
STEP_LOD = 0.3
BANDS = (Band(500.0, 600.0),)
SIGMA_LOD = 0.35


class GaussianCoronagraph(AbstractTableCoronagraph):
    """Analytic flux-fraction Gaussian tables with constant leakage."""

    grid: PixelGrid = eqx.field(static=True)
    sky_trans: Image
    uniform_sky_profile: SampledProfile
    bandpass: Band | None = None
    band_averaged: bool = eqx.field(static=True, default=False)

    def __init__(self, n=N_PIX, step=STEP_LOD):
        """Build an n x n focal grid with the given lambda/D step."""
        self.grid = PixelGrid.focal(n, step, Frame(550.0, 6.0, "circumscribed"))
        self.sky_trans = Image(
            jnp.full((n, n), 0.3), self.grid, (), Measure.intensive(), "transmission"
        )
        self.uniform_sky_profile = SampledProfile(
            [0.0],
            [0.3],
            quantity="uniform_sky_response",
            frame=self.grid.frame,
            interpolation="linear",
        )

    def _native_source_psfs(self, x_lod, y_lod):
        x, y = self.grid.coordinates()
        r2 = (x[None] - jnp.atleast_1d(x_lod)[:, None, None]) ** 2 + (
            y[None] - jnp.atleast_1d(y_lod)[:, None, None]
        ) ** 2
        data = (
            0.3
            * self.grid.pixel_area
            / (2 * np.pi * SIGMA_LOD**2)
            * jnp.exp(-r2 / (2 * SIGMA_LOD**2))
        )
        image = Image(
            data, self.grid, ("source",), Measure.integral(), "stellar_flux_fraction"
        )
        return SourcePSFs(image, jnp.ones(data.shape[0], dtype=bool))

    def _native_stellar_map(self, stellar_diameter_lod):
        return Image(
            jnp.full(self.grid.shape, 1e-9 * self.grid.pixel_area)
            * (1 + stellar_diameter_lod**2),
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )

    def curves(self, rule=None, *, leakage_rule="aperture_mean"):
        """Closed-form circular Gaussian and constant leakage integrals."""
        from optixstuff.apertures import Circular

        rule = Circular(0.7) if rule is None else rule
        area = np.pi * rule.radius_lod**2
        tau = 0.3 * (1 - np.exp(-(rule.radius_lod**2) / (2 * SIGMA_LOD**2)))
        return PerformanceCurves(
            [0.0],
            self.grid.frame,
            rule,
            [tau],
            [area],
            [1e-9 * area],
            [1e-9],
            [1e-9 * area / tau],
            0.0,
            min(self.grid.shape) * self.grid.step / 2,
            throughput_variance=[tau],
            core_area_variance_lod2=[area],
            leakage_fraction_variance=[1e-9 * area],
            leakage_density_variance_lod2=[1e-9],
        )


class ConstantVarianceDetector(Detector):
    """A Gaussian detector whose variance does not depend on the rates."""

    variance_e2: float = eqx.field(static=True)

    def __init__(self, variance_e2, **kwargs):
        """Build the parent detector and fix the variance."""
        super().__init__(**kwargs)
        self.variance_e2 = float(variance_e2)

    def mean_and_variance(self, rates, plan, *, sum_bands=False):
        """Mean from the parent, a fixed variance everywhere."""
        mean, _ = super().mean_and_variance(rates, plan, sum_bands=sum_bands)
        return mean, jnp.full_like(mean, self.variance_e2)


def make_detector(kind="poisson", **kwargs):
    """A detector on the coronagraph grid: ``poisson``, ``read`` or ``constant``."""
    grid = GaussianCoronagraph().grid
    scale = float(grid.frame.lod_to_arcsec(550.0)) * STEP_LOD
    common = dict(pixel_scale_arcsec=scale, shape=(N_PIX, N_PIX), qe=0.8)
    if kind == "poisson":
        return Detector(photon_counting=True, **common)
    if kind == "read":
        return Detector(read_noise_e_rms=3.0, **common)
    return ConstantVarianceDetector(kwargs.get("variance_e2", 1.0), **common)


def make_instrument(detector):
    """40 m2 collecting area, 0.4 throughput, the analytic coronagraph."""
    return Instrument(
        SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=40.0),
        (ConstantThroughput(0.4),),
        GaussianCoronagraph(),
        detector,
    )


def make_scene(*, contrast=1e-10, sky_pa_deg=0.0, separation_arcsec=0.05, model=None):
    """One planet on a face-on circular orbit at the given sky position.

    ``model`` is a planet model built on the host; pass it when the scene is
    rebuilt inside a trace, since a tabulated model built under a trace has no
    knot digest.
    """
    if model is None:
        wavelengths = jnp.array([400.0, 500.0, 600.0, 700.0])
        values = jnp.broadcast_to(jnp.asarray(contrast), (4,))
        model = GridPhysicalModel(
            wavelengths,
            jnp.array([0.0, 90.0, 180.0]),
            jnp.broadcast_to(values[None, :, None], (1, 4, 3)),
        )
    return Scene.from_classical(
        FlatStar(flux_phot_per_nm_m2=1e6),
        (Planet(jnp.ones(1), model),),
        star_mass_Msun=1.0,
        planet_mass_Msun=Mearth2kg / Msun2kg,
        dist_pc=10.0,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.atleast_1d(separation_arcsec) * 10,
        e=0.0,
        W_rad=jnp.deg2rad(sky_pa_deg),
        i_rad=0.0,
        w_rad=0.0,
        M0_rad=0.0,
        epoch_d=0.0,
        order=(0,),
    )


def make_exposure(scene, instrument, *, time_s=1.0e4):
    """A one-band exposure with a single frame of ``time_s`` seconds."""
    return cp.build_exposure(
        scene,
        instrument,
        BANDS,
        time_d=0.0,
        plan=FramePlan(frame_time_s=time_s, n_frames=1),
    )


def float_leaves(tree):
    """Count floating array leaves, for a quick structure check."""
    return sum(
        1
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating)
    )
