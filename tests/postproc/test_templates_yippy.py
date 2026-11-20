"""Synthetic convention anchors and public-data yippy integration tests."""

import os
from dataclasses import replace

import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import MultiBandCoronagraph, SourcePSFs
from optixstuff.coronagraph import AbstractTableCoronagraph
from optixstuff.grids import Frame, Image, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Band
from yippy import Coronagraph, open_yip, yip_info

from photomancy.postproc.core.detection.estimator import DetectionEstimator
from photomancy.postproc.core.detection.filters import PSFTemplateFilter
from photomancy.postproc.core.detection.samplers import ApertureSampler
from photomancy.postproc.core.detection.significance import TwoSampleTTest
from photomancy.postproc.templates.yippy import YippyTemplateProvider

_ARCHIVE = yip_info("eac1_aavc_2d")
_CACHED = _ARCHIVE.archive_path().is_file()
_OPTED_IN = os.environ.get("PHOTOMANCY_FETCH_DATA") == "1"
STAMP_SIZE = 21
INJECTION_SEPARATION_LOD = 6.0
INJECTION_AMPLITUDE = 300.0


class SyntheticCoronagraph(AbstractTableCoronagraph):
    """Analytic off-axis source; field-dependent widths expose swapped offsets."""

    grid: PixelGrid

    bandpass: Band

    @property
    def sky_trans(self):
        """Use the same geometry for sky and source products."""
        return Image(
            jnp.ones(self.grid.shape),
            self.grid,
            (),
            Measure.intensive(),
            "transmission",
        )

    @property
    def uniform_sky_profile(self):
        """Detect use of a radial sky response to prepare a point-source stamp."""
        raise AssertionError("template generation must use the source product")

    def _native_stellar_map(self, stellar_diameter_lod):
        raise AssertionError("template geometry must not require a stellar image")

    def curves(self, *args, **kwargs):
        """Catch accidental use of scalar performance by the stamp provider."""
        raise AssertionError(
            "template generation should use only point-source products"
        )

    def _native_source_psfs(self, x_lod, y_lod):
        """Return a Gaussian Image at the documented native-grid coordinate."""
        xy = self.grid.coordinates()
        x_lod, y_lod = x_lod[:, None, None], y_lod[:, None, None]
        image = Image(
            jnp.exp(
                -0.5
                * (
                    ((xy.x - x_lod) / (0.35 + 0.04 * x_lod)) ** 2
                    + ((xy.y - y_lod) / (0.30 + 0.04 * y_lod)) ** 2
                )
            ),
            self.grid,
            ("source",),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="cubic",
        )
        return SourcePSFs(image, jnp.ones(image.data.shape[0], dtype=bool))


@vv.case("psf-template", "code-verification")
@vv.seam("yippy", "photomancy")
@pytest.mark.parametrize("size", [40, 41])
@pytest.mark.parametrize(
    ("unit", "rotated"), [("lod", False), ("arcsec", False), ("arcsec", True)]
)
def test_peak_within_002_pixel_of_stamp_center(size, unit, rotated):
    """Fails if source offsets, grid center or row/column order change.

    Basis: an analytic Gaussian has its centroid at the source. A 0.02 px
    bound covers cubic/linear interpolation on these samples; a half-pixel
    mutant is 25 times larger. Field-dependent widths anchor input axes.
    Two shapes are required to distinguish even and odd center conventions.
    """
    frame = Frame(550.0, 6.0, "circumscribed")
    native = PixelGrid.regular((size, size), 0.25, "lod", frame=frame)
    if rotated:
        native = replace(native, orientation=SkyOrientation(0.0, -1))
    # A distinct science Frame makes copying native lambda/D sampling fail.
    science_frame = Frame(650.0, 4.0, "inscribed")
    step = 0.2 * 4.0 / 6.0
    # Deliberately rotate the science basis; +column points along native -y.
    science = PixelGrid(
        (size, size),
        (size - 1) / 2,
        (size - 1) / 2,
        ((0.0, step), (-step, 0.0)),
        "lod",
        frame=science_frame,
    )
    if unit == "arcsec":
        science = replace(
            science.to_arcsec(700.0),
            orientation=SkyOrientation(90.0, 1) if rotated else SkyOrientation(37.0),
        )
    # The other band has its own shape, pitch and diameter. Using reference
    # geometry at the selected wavelength displaces the off-axis template.
    other = PixelGrid.regular((25, 25), 0.45, "lod", frame=science_frame)
    bands = MultiBandCoronagraph(
        (
            SyntheticCoronagraph(other, Band(490, 510)),
            SyntheticCoronagraph(native, Band(690, 710)),
        ),
        reference_wavelength_nm=500.0 if size == 40 else 700.0,
    )
    provider = YippyTemplateProvider(bands, science, 15, wavelength_nm=700.0)
    position = jnp.array([science.center_y - 4.0, science.center_x + 6.0])
    stamp = np.asarray(provider(position))
    y, x = np.indices(stamp.shape)
    centroid = np.array([(y * stamp).sum(), (x * stamp).sum()]) / stamp.sum()
    np.testing.assert_allclose(centroid, [7.0, 7.0], atol=0.02, rtol=0)
    # Known native source (-0.8,-1.2) lod: science x width=0.252/0.2,
    # science y width=0.318/0.2. These independent second moments reject
    # swapped source inputs even if extraction happens to recenter them.
    # With explicit orientations, E=-1.2,N=+0.8 and native x=+1.2,y=+0.8;
    # science row/column widths are then 0.332/0.2 and 0.398/0.2.
    moments = (
        np.array([((y - 7) ** 2 * stamp).sum(), ((x - 7) ** 2 * stamp).sum()])
        / stamp.sum()
    )
    expected = (
        [(0.332 / 0.2) ** 2, (0.398 / 0.2) ** 2]
        if rotated
        else [(0.318 / 0.2) ** 2, (0.252 / 0.2) ** 2]
    )
    np.testing.assert_allclose(moments, expected, atol=0.30, rtol=0)


@pytest.fixture(scope="session")
def coronagraph() -> Coronagraph:
    """Load the eac1 YIP coronagraph once per test session."""
    return open_yip(_ARCHIVE.name).prepare()


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with PHOTOMANCY_FETCH_DATA=1",
)
class TestStampShape:
    """The provider returns a fixed-size, finite stamp at real separations."""

    @pytest.mark.parametrize("separation_lod", [3.0, 6.0])
    def test_stamp_shape_and_finite(self, coronagraph, separation_lod):
        """The stamp is (STAMP_SIZE, STAMP_SIZE) and finite at the separation."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        dx_px = separation_lod / coronagraph.grid.step
        stamp = provider(center_yx + jnp.array([0.0, dx_px]))
        assert stamp.shape == (STAMP_SIZE, STAMP_SIZE)
        assert bool(jnp.all(jnp.isfinite(stamp)))


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with PHOTOMANCY_FETCH_DATA=1",
)
class TestOrientationPin:
    """The stamp peak lands where the pixel-to-lod mapping says it should.

    If this fails, the pixel-to-lod mapping or axis order in the provider is
    wrong; the fix belongs in the provider, not in a widened tolerance here.
    """

    @pytest.mark.parametrize(
        ("dy_lod", "dx_lod"),
        [(0.0, 5.0), (4.0, 0.0)],
    )
    def test_peak_within_one_pixel_of_stamp_center(self, coronagraph, dy_lod, dx_lod):
        """A stamp built for a +x or +y offset peaks within 1 px of its center."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        dy_px = dy_lod / coronagraph.grid.step
        dx_px = dx_lod / coronagraph.grid.step
        position_yx = center_yx + jnp.array([dy_px, dx_px])
        stamp = np.asarray(provider(position_yx))
        peak_yx = np.unravel_index(np.argmax(stamp), stamp.shape)
        stamp_center = (STAMP_SIZE - 1) / 2.0
        assert abs(peak_yx[0] - stamp_center) <= 1.0
        assert abs(peak_yx[1] - stamp_center) <= 1.0


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with PHOTOMANCY_FETCH_DATA=1",
)
class TestSeparationDependence:
    """Stamps at different separations are genuinely different templates."""

    def test_stamps_differ_after_peak_normalization(self, coronagraph):
        """Peak-normalized stamps at 2 and 7 lambda/D are not close."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        near = np.asarray(
            provider(center_yx + jnp.array([0.0, 2.0 / coronagraph.grid.step]))
        )
        far = np.asarray(
            provider(center_yx + jnp.array([0.0, 7.0 / coronagraph.grid.step]))
        )
        near = near / near.max()
        far = far / far.max()
        assert not np.allclose(near, far)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with PHOTOMANCY_FETCH_DATA=1",
)
class TestEndToEnd:
    """The template filter detects an injected planet on the coronagraph grid."""

    def test_injected_planet_clears_detection_threshold(self, coronagraph):
        """An injected stamp at 6 lambda/D clears statistic 5 and fpf 1e-3."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        fwhm_px = 1.0 / coronagraph.grid.step
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        dx_px = INJECTION_SEPARATION_LOD / coronagraph.grid.step
        position_yx = center_yx + jnp.array([0.0, dx_px])

        stamp = provider(position_yx)
        half = STAMP_SIZE // 2
        y0, x0 = int(position_yx[0]) - half, int(position_yx[1]) - half

        image = jax.random.normal(jax.random.PRNGKey(0), coronagraph.grid.shape)
        image = image.at[y0 : y0 + STAMP_SIZE, x0 : x0 + STAMP_SIZE].add(
            INJECTION_AMPLITUDE * stamp
        )

        estimator = DetectionEstimator(
            filter=PSFTemplateFilter(provider=provider),
            sampler=ApertureSampler(fwhm=fwhm_px),
            test=TwoSampleTTest(),
        )
        stats = estimator(image, position_yx[None, :], None, center_yx)

        assert float(stats.statistic[0]) > 5.0
        assert float(stats.fpf[0]) < 1e-3
