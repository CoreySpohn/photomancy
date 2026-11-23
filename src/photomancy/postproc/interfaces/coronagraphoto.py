"""Adapters for coronagraphoto electron Images on shared detector grids."""

import jax.numpy as jnp
from optixstuff.grids import Frame, Image, Measure, PixelGrid

from photomancy.postproc.core.detection.estimator import DetectionEstimator


def extract_image(detector_readout: Image) -> jnp.ndarray:
    """Return the 2D data, or the first band plane of an observed Image.

    Counts retain their electron units. Geometry remains on detector_readout.grid.

    Args:
        detector_readout: Observed Image of per-pixel electron counts, either 2D
            or (band, y, x).

    Returns:
        2D array of electron counts, shape (ny, nx).
    """
    if (
        detector_readout.measure != Measure.integral()
        or detector_readout.quantity != "electrons"
    ):
        raise ValueError("expected per-pixel electron counts")
    if detector_readout.data.ndim == 2:
        return detector_readout.data
    if detector_readout.data.ndim == 3:
        return detector_readout.data[0]
    raise ValueError("expected a 2D image or a (band, y, x) observation")


def get_fwhm(grid: PixelGrid, frame: Frame, wavelength_nm: float) -> float:
    """Return one lambda/D in pixels on an axis-aligned isotropic grid.

    frame is supplied by the caller from the primary's diameter convention.
    A lod grid's own Frame defines its reference angular sampling.

    Args:
        grid: Detector PixelGrid, in arcsec or lod units.
        frame: Frame that defines one lambda/D from the primary diameter.
        wavelength_nm: Wavelength in nm.

    Returns:
        Size of one lambda/D in pixels.
    """
    if grid.unit == "arcsec":
        step_arcsec = grid.step
    elif grid.unit == "lod":
        step_arcsec = grid.step * grid.frame.lod_to_arcsec(wavelength_nm)
    else:
        raise ValueError("FWHM requires an arcsec or lod grid")
    return frame.lod_to_arcsec(wavelength_nm) / step_arcsec


def sky_offset_to_position(
    image: Image, east_arcsec: float, north_arcsec: float
) -> jnp.ndarray:
    """Map sky offsets to (y, x) pixel-center coordinates of an observed Image.

    Pixel (row j, column i) is centered at (x, y) = (i, j). The oriented
    arcsec grid supplies both the star center and the detector handedness.

    Args:
        image: Observed Image on an oriented arcsec detector grid.
        east_arcsec: Offset toward sky east from the star, in arcsec.
        north_arcsec: Offset toward sky north from the star, in arcsec.

    Returns:
        (2,) array of the position (y, x) in pixel-center coordinates.
    """
    grid = image.grid
    if grid.unit != "arcsec" or grid.orientation is None:
        raise ValueError("sky offsets require an oriented arcsec detector grid")
    offset = grid.orientation.sky_to_detector(east_arcsec, north_arcsec)
    position = grid.unit_to_pixel(offset.x, offset.y)
    return jnp.stack([position.y, position.x])


def analyze_observation(
    image: Image,
    planet_pos: tuple[float, float],
    wavelength_nm: float,
    frame: Frame,
    max_apertures: int = 200,
) -> float:
    """Compute Mawet SNR in the first band of an observed electron Image.

    planet_pos is (y, x) in this image's pixel coordinates, as returned by
    sky_offset_to_position. The grid supplies the reference-aperture center;
    frame supplies the primary diameter for one lambda/D at wavelength_nm.

    Args:
        image: Observed Image of per-pixel electron counts.
        planet_pos: Planet position (y, x) in pixels of this image.
        wavelength_nm: Wavelength in nm, used to size one lambda/D.
        frame: Frame that supplies the primary diameter for lambda/D.
        max_apertures: Static reference-aperture buffer size.

    Returns:
        Mawet t-statistic of the planet position.
    """
    fwhm = get_fwhm(image.grid, frame, wavelength_nm)
    estimator = DetectionEstimator.mawet(fwhm, max_apertures=max_apertures)
    center = jnp.array([image.grid.center_y, image.grid.center_x])
    positions = jnp.asarray(planet_pos)[None, :]
    return estimator(extract_image(image), positions, center_yx=center).statistic[0]
