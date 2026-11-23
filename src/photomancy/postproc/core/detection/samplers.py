"""Noise samplers: same-radius reference apertures and full annuli."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.map_coordinates import map_coordinates

from photomancy.postproc.core.detection.base import AbstractFilter, AbstractSampler
from photomancy.postproc.core.geometry import (
    generate_aperture_coords,
    n_reference_apertures,
)


class ApertureSampler(AbstractSampler):
    """Reference apertures at the candidate's radius (Mawet et al. 2014).

    Evaluates the filter at up to max_apertures positions on the candidate's
    circle, spaced by the aperture diameter with an exclusion buffer on both
    sides of the candidate. The validity map is sampled nearest-neighbor with
    off-chip positions invalid, exactly matching the fused v1.1.1 core.

    Attributes:
        fwhm: Aperture diameter (resolution element) in pixels.
        exclusion_buffer: Gap between the candidate and the first and last
            reference aperture, in units of the angular step.
        max_apertures: Static size of the sample buffer.
    """

    fwhm: float
    exclusion_buffer: float
    max_apertures: int = eqx.field(static=True)

    def __init__(
        self,
        fwhm: float,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
    ):
        """Configure the sampler geometry.

        Args:
            fwhm: Full width at half maximum in pixels (aperture diameter).
            max_apertures: Static buffer size for the sample arrays.
            exclusion_buffer: Gap between candidate and first/last reference
                aperture in units of angular step.
        """
        self.fwhm = fwhm
        self.max_apertures = max_apertures
        self.exclusion_buffer = exclusion_buffer

    def __call__(
        self,
        filtered: jnp.ndarray,
        position_yx: jnp.ndarray,
        filt: AbstractFilter,
        validity_map: jnp.ndarray,
        center_yx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample the filter response at same-radius reference apertures.

        Args:
            filtered: Output of filt.prepare() for the current image.
            position_yx: (2,) candidate position (y, x) in pixels.
            filt: Filter evaluated at each reference aperture.
            validity_map: (ny, nx) map, 1 = usable pixel, 0 = excluded.
            center_yx: (2,) star center (y, x) in pixels.

        Returns:
            Tuple of (samples, mask, geom_ok): the (max_apertures,) filter responses
            at the reference apertures, the boolean mask of valid references, and a
            scalar bool that is True when the candidate lies at least one fwhm from
            the star center.
        """
        cy, cx = center_yx[0], center_yx[1]
        py, px = position_yx[0], position_yx[1]

        r_pix = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)
        planet_angle = jnp.arctan2(py - cy, px - cx)

        n_actual = n_reference_apertures(r_pix, self.fwhm, self.exclusion_buffer)
        ref_y, ref_x, idx_mask = generate_aperture_coords(
            (cy, cx),
            r_pix,
            planet_angle,
            n_actual,
            max_apertures=self.max_apertures,
            fwhm=self.fwhm,
            exclusion_buffer=self.exclusion_buffer,
        )

        ref_validity = map_coordinates(
            validity_map,
            jnp.stack([ref_y, ref_x]),
            order=0,
            cval=0.0,
        )
        mask = idx_mask & (ref_validity > 0.5)

        positions = jnp.stack([ref_y, ref_x], axis=1)
        samples = jax.vmap(lambda pos: filt.evaluate(filtered, pos))(positions)

        geom_ok = r_pix >= self.fwhm
        return samples, mask, geom_ok


class AnnulusSampler(AbstractSampler):
    """All pixels of a radial annulus around the candidate's radius.

    The community-standard matched-filter noise sample: every pixel of the
    prepared (filtered) image whose radius lies in [inner, outer], excluding
    a 1.5 fwhm disk around the candidate. Negative bounds select the
    automatic annulus of the v1.1.1 matched-filter core (candidate radius
    plus/minus one fwhm, clipped to the chip).

    The samples are the prepared image's own pixels, which matches filters
    whose evaluate() point-samples the prepared map. For filters whose
    prepare() is the identity (e.g. the PSF-template filter) the sigma
    test's scale is wrong: the filter response's null spread is set by the
    template normalization (sigma / sqrt(sum of squared zero-meaned
    template)) while the annulus spread is the raw-pixel sigma, so the test
    is conservative or anti-conservative depending on that normalization.
    Same-scale noise references for such filters come from ApertureSampler.

    Attributes:
        fwhm: Resolution element in pixels; the candidate exclusion disk has
            radius 1.5 fwhm and the automatic annulus half-width is one fwhm.
        annulus_inner: Inner annulus radius in pixels; negative selects the
            automatic bound.
        annulus_outer: Outer annulus radius in pixels; negative selects the
            automatic bound.
    """

    fwhm: float
    annulus_inner: float
    annulus_outer: float

    def __init__(
        self,
        fwhm: float,
        annulus_inner: float = -1.0,
        annulus_outer: float = -1.0,
    ):
        """Configure the annulus geometry.

        Args:
            fwhm: Full width at half maximum in pixels.
            annulus_inner: Inner radius in pixels; negative means automatic.
            annulus_outer: Outer radius in pixels; negative means automatic.
        """
        self.fwhm = fwhm
        self.annulus_inner = annulus_inner
        self.annulus_outer = annulus_outer

    def __call__(
        self,
        filtered: jnp.ndarray,
        position_yx: jnp.ndarray,
        filt: AbstractFilter,
        validity_map: jnp.ndarray,
        center_yx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Collect every valid annulus pixel of the prepared image.

        Args:
            filtered: Prepared image whose pixels are the noise samples.
            position_yx: (2,) candidate position (y, x) in pixels.
            filt: Unused; the samples are the prepared image's own pixels.
            validity_map: (ny, nx) map, 1 = usable pixel, 0 = excluded.
            center_yx: (2,) star center (y, x) in pixels.

        Returns:
            Tuple of (samples, mask, geom_ok): the flattened prepared image, the
            flattened boolean mask of valid annulus pixels, and a scalar bool that is
            always True.
        """
        del filt  # Annulus samples are the prepared image's own pixels.
        ny, nx = filtered.shape
        cy, cx = center_yx[0], center_yx[1]
        py, px = position_yx[0], position_yx[1]
        max_radius = jnp.minimum(ny, nx) / 2 - 1

        y_coords, x_coords = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        r_grid = jnp.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        r_planet = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)

        default_inner = jnp.maximum(r_planet - self.fwhm, self.fwhm)
        default_outer = jnp.minimum(r_planet + self.fwhm, max_radius)
        default_inner = jnp.minimum(default_inner, default_outer - 1.0)
        inner_r = jnp.where(self.annulus_inner >= 0, self.annulus_inner, default_inner)
        outer_r = jnp.where(self.annulus_outer >= 0, self.annulus_outer, default_outer)

        planet_dist = jnp.sqrt((y_coords - py) ** 2 + (x_coords - px) ** 2)
        annulus_mask = (
            (r_grid >= inner_r) & (r_grid <= outer_r) & (planet_dist > self.fwhm * 1.5)
        )
        annulus_mask = annulus_mask & (validity_map > 0.5)

        geom_ok = jnp.asarray(True)
        return filtered.ravel(), annulus_mask.ravel(), geom_ok
