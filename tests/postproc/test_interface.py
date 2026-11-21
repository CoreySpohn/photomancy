"""Tests for the coronagraphoto interface module."""

import math
import subprocess
import sys

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils import constants
from optixstuff.grids import Frame, Image, Measure, PixelGrid, SkyOrientation

from photomancy.postproc.interfaces.coronagraphoto import (
    analyze_observation,
    extract_image,
    get_fwhm,
    sky_offset_to_position,
)


class TestExtractImage:
    """Tests for extract_image function."""

    def test_passthrough_2d(self):
        """2D images should pass through unchanged."""
        image = jnp.ones((50, 50))
        result = extract_image(
            Image(
                image,
                PixelGrid.regular((50, 50), 0.02, "arcsec"),
                (),
                Measure.integral(),
                "electrons",
            )
        )
        assert result.shape == (50, 50)
        assert jnp.allclose(result, image)

    def test_extract_first_plane_from_3d(self):
        """3D datacubes should return first plane."""
        datacube = jnp.ones((10, 50, 50))
        result = extract_image(
            Image(
                datacube,
                PixelGrid.regular((50, 50), 0.02, "arcsec"),
                ("band",),
                Measure.integral(),
                "electrons",
            )
        )
        assert result.shape == (50, 50)


class TestGetFWHM:
    """Tests for get_fwhm function."""

    @vv.case("postproc-seam", "code-verification")
    @vv.seam("hwoutils", "photomancy")
    @pytest.mark.parametrize("unit", ["arcsec", "lod"])
    def test_fwhm_equals_diffraction_scale(self, unit):
        """Fails if nm, radians, arcsec or the Frame diameter are misread.

        Basis: typed SI/angle constants with parity checks; 8 float32 eps
        bounds conversion rounding. The grid and primary Frames differ.
        """
        nm_to_m = 1e-9
        radians_to_arcsec = 180.0 * 3600.0 / math.pi
        assert constants.nm2m == nm_to_m
        assert constants.rad2arcsec == radians_to_arcsec
        frame = Frame(550.0, 6.0, "circumscribed")
        sampling_frame = Frame(500.0, 4.0, "inscribed")
        grid = PixelGrid.regular(
            (16, 16), 0.02 if unit == "arcsec" else 0.3, unit, frame=sampling_frame
        )
        wavelength_nm = 700.0
        pitch_arcsec = (
            0.02
            if unit == "arcsec"
            else 0.3 * wavelength_nm * nm_to_m / 4.0 * radians_to_arcsec
        )
        expected = wavelength_nm * nm_to_m / 6.0 * radians_to_arcsec / pitch_arcsec
        np.testing.assert_allclose(
            get_fwhm(grid, frame, wavelength_nm),
            expected,
            rtol=8 * np.finfo(np.float32).eps,
        )

    def test_fwhm_scales_with_wavelength(self):
        """FWHM should increase with wavelength."""
        fwhm_blue = get_fwhm(
            PixelGrid.regular((16, 16), 0.02, "arcsec"),
            Frame(550.0, 6.0, "circumscribed"),
            400.0,
        )
        fwhm_red = get_fwhm(
            PixelGrid.regular((16, 16), 0.02, "arcsec"),
            Frame(550.0, 6.0, "circumscribed"),
            800.0,
        )
        assert fwhm_red > fwhm_blue

    def test_fwhm_scales_with_diameter(self):
        """FWHM should decrease with larger aperture."""
        fwhm_small = get_fwhm(
            PixelGrid.regular((16, 16), 0.02, "arcsec"),
            Frame(550.0, 4.0, "circumscribed"),
            550.0,
        )
        fwhm_large = get_fwhm(
            PixelGrid.regular((16, 16), 0.02, "arcsec"),
            Frame(550.0, 8.0, "circumscribed"),
            550.0,
        )
        assert fwhm_large < fwhm_small


class TestAnalyzeObservation:
    """Tests for analyze_observation convenience function."""

    def test_returns_snr(self):
        """Should return a finite SNR value."""
        shape = (51, 51)
        cy, cx = shape[0] / 2.0, shape[1] / 2.0

        # Create test image
        image = jnp.ones(shape) * 100.0
        planet_y, planet_x = cy + 10, cx

        y, x = jnp.ogrid[: shape[0], : shape[1]]
        sigma = 2.0
        planet = 500.0 * jnp.exp(
            -((y - planet_y) ** 2 + (x - planet_x) ** 2) / (2 * sigma**2)
        )
        image = image + planet

        snr = analyze_observation(
            image=Image(
                image,
                PixelGrid.regular(shape, 0.0218, "arcsec"),
                (),
                Measure.integral(),
                "electrons",
            ),
            frame=Frame(550.0, 6.0, "circumscribed"),
            planet_pos=(planet_y, planet_x),
            wavelength_nm=550.0,
        )

        assert jnp.isfinite(snr)
        assert snr > 0


def test_seam_contract_names_resolve_at_top_level():
    """Downstream registries resolve these exact attribute paths by import."""
    import photomancy.postproc as postproc

    for name in (
        "FrameSet",
        "DetectionStats",
        "AbstractDetectionArm",
        "MawetArm",
        "student_t_sf",
    ):
        assert hasattr(postproc, name), name
        assert name in postproc.__all__, name


def test_detection_core_names_resolve_at_top_level():
    """Composable detection names are importable from the package root."""
    import photomancy.postproc as postproc

    for name in (
        "DetectionEstimator",
        "AbstractFilter",
        "AbstractSampler",
        "AbstractTest",
        "ApertureFilter",
        "GaussianFilter",
        "ApertureSampler",
        "AnnulusSampler",
        "TwoSampleTTest",
        "AnnulusSigmaTest",
        "GrubbsTest",
        "normal_sf",
        "grubbs_fpf",
        "n_reference_apertures",
        "gaussian_kernel_1d",
        "gaussian_filter_2d",
    ):
        assert hasattr(postproc, name), name
        assert name in postproc.__all__, name


def test_template_filter_names_resolve_at_top_level():
    """PSF-template filter and template-provider names resolve at the package root."""
    import photomancy.postproc as postproc

    for name in (
        "PSFTemplateFilter",
        "extract_patch",
        "AbstractTemplateProvider",
        "ArrayTemplateProvider",
        "MatchedFilterArm",
    ):
        assert hasattr(postproc, name), name
        assert name in postproc.__all__, name


def test_plain_import_does_not_load_yippy_provider():
    """Importing photomancy.postproc must not pull in the yippy template provider.

    photomancy.postproc.templates.yippy stays unimported until something
    explicitly reaches for it. Checked in a fresh subprocess so an unrelated
    test that already imported it cannot mask a regression here.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, photomancy.postproc; "
            "assert 'photomancy.postproc.templates.yippy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_plain_photomancy_import_does_not_load_postproc():
    """``import photomancy`` stays light: post-processing loads only on request."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, photomancy; assert 'photomancy.postproc' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@vv.case("postproc-seam", "code-verification")
@vv.seam("coronagraphoto", "photomancy")
@pytest.mark.parametrize("parity", [-1, 1])
def test_rotated_observed_image_sky_position(parity):
    """Fails if sky PA, parity, row/column or a half-pixel center is wrong.

    Basis: PA=90 degrees gives x=-parity*N, y=E analytically; 2e-6 px
    is float32 rounding and rejects a 0.5-pixel mutant by >100000 times.
    A skew basis and nongeometric center also exercise the grid inverse.
    """
    grid = PixelGrid(
        (20, 24),
        10.25,
        8.75,
        ((0.02, 0.01), (0.0, 0.03)),
        "arcsec",
        orientation=SkyOrientation(90.0, parity),
    )
    observed = Image(
        jnp.zeros((2, 20, 24)), grid, ("band",), Measure.integral(), "electrons"
    )
    # E=0.09, N=0.08 -> dy=3, dx=(-parity*0.08-0.01*3)/0.02.
    expected = np.array([11.75, 10.25 + (-parity * 0.08 - 0.03) / 0.02])
    np.testing.assert_allclose(
        sky_offset_to_position(observed, 0.09, 0.08), expected, atol=2e-6, rtol=0
    )
