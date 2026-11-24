"""The composed detection estimator: filter + sampler + significance test."""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.radial import get_center

from photomancy.postproc.contracts import DetectionStats
from photomancy.postproc.core.detection.base import (
    AbstractFilter,
    AbstractSampler,
    AbstractTest,
)
from photomancy.postproc.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_kernel_1d,
)
from photomancy.postproc.core.detection.samplers import AnnulusSampler, ApertureSampler
from photomancy.postproc.core.detection.significance import (
    AnnulusSigmaTest,
    TwoSampleTTest,
)
from photomancy.postproc.core.photometry import make_aperture_kernel


class DetectionEstimator(eqx.Module):
    """Composable detection: any filter, any sampler, any significance test.

    All three components are Equinox modules, so the composed estimator is a
    single pytree and the whole pipeline fuses into one XLA kernel under jit;
    the decomposition is purely a source-level concern. The estimator
    resolves defaults (all-valid map, geometric star center), vmaps the
    per-candidate computation, and NaNs statistic and FPF where the sampler
    reports the geometry invalid or the candidate footprint is excluded.

    Attributes:
        filter: Signal-extraction step that measures the response at a position.
        sampler: Noise-reference step that collects samples of the filter response.
        test: Significance step that turns signal and noise samples into a
            statistic, false positive fraction and degrees of freedom.
    """

    filter: AbstractFilter
    sampler: AbstractSampler
    test: AbstractTest

    @classmethod
    def mawet(
        cls,
        fwhm: float,
        soft: bool = True,
        sharpness: float = 10.0,
        fast: bool = False,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
    ) -> Self:
        """Build the Mawet et al. (2014) aperture-photometry t-test.

        Args:
            fwhm: Aperture diameter in pixels.
            soft: Use differentiable soft aperture edges.
            sharpness: Sigmoid sharpness of soft aperture edges.
            fast: Use bilinear interpolation instead of cubic.
            max_apertures: Static reference-aperture buffer size.
            exclusion_buffer: Gap between the candidate and first reference
                aperture, in units of the angular step.

        Returns:
            Estimator with an aperture filter, same-radius aperture sampler and
            two-sample t-test.
        """
        kernel = make_aperture_kernel(radius=fwhm / 2.0, soft=soft, sharpness=sharpness)
        return cls(
            filter=ApertureFilter(kernel=kernel, order=1 if fast else 3),
            sampler=ApertureSampler(
                fwhm=fwhm,
                max_apertures=max_apertures,
                exclusion_buffer=exclusion_buffer,
            ),
            test=TwoSampleTTest(),
        )

    @classmethod
    def matched_filter(
        cls,
        fwhm: float,
        fast: bool = False,
        *,
        annulus_inner: float | None = None,
        annulus_outer: float | None = None,
    ) -> Self:
        """Build a Gaussian matched filter with annular noise estimation.

        Args:
            fwhm: Gaussian full width at half maximum in pixels.
            fast: Use bilinear interpolation instead of cubic.
            annulus_inner: Inner radius of the noise annulus in pixels.
                None selects the automatic bound.
            annulus_outer: Outer radius of the noise annulus in pixels.
                None selects the automatic bound.

        Returns:
            Estimator with a Gaussian filter, annulus sampler and annulus sigma test.
        """
        return cls(
            filter=GaussianFilter(
                kernel_1d=gaussian_kernel_1d(fwhm), order=1 if fast else 3
            ),
            sampler=AnnulusSampler(
                fwhm=fwhm,
                annulus_inner=-1.0 if annulus_inner is None else annulus_inner,
                annulus_outer=-1.0 if annulus_outer is None else annulus_outer,
            ),
            test=AnnulusSigmaTest(),
        )

    @eqx.filter_jit
    def __call__(
        self,
        image: jnp.ndarray,
        positions_yx: jnp.ndarray,
        validity_map: jnp.ndarray | None = None,
        center_yx: jnp.ndarray | None = None,
    ) -> DetectionStats:
        """Detection statistics at candidate positions.

        Args:
            image: 2D science image.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            validity_map: Optional (ny, nx) mask (1 = usable, 0 = excluded).
                Measurements touching excluded pixels anywhere in their footprint
                or interpolation stencil are excluded.
            center_yx: Optional (2,) star center; geometric center when None.

        Returns:
            DetectionStats for the n candidates (statistic_kind from the
            test, fwhm_px from the sampler).
        """
        ny, nx = image.shape
        if validity_map is None:
            validity_map = jnp.ones((ny, nx))
        valid = validity_map > 0.5
        filtered = self.filter.prepare(
            jnp.where(valid, image, jnp.zeros((), image.dtype))
        )
        prepared_validity = self.filter.prepare_validity(valid)
        if center_yx is None:
            center = jnp.array(get_center(image.shape))
        else:
            center = jnp.asarray(center_yx)

        def _single(position_yx: jnp.ndarray):
            """Statistic, FPF, and dof for one candidate."""
            filt = self.filter.bind(position_yx)
            signal = filt.evaluate(filtered, position_yx)
            samples, mask, geom_ok = self.sampler(
                filtered, position_yx, filt, prepared_validity, center
            )
            statistic, fpf, dof = self.test(signal, samples, mask)
            cand_ok = filt.evaluate_validity(prepared_validity, position_yx)
            statistic = jnp.where(geom_ok & cand_ok, statistic, jnp.nan)
            fpf = jnp.where(geom_ok & cand_ok, fpf, jnp.nan)
            return statistic, fpf, dof

        statistic, fpf, dof = jax.vmap(_single)(positions_yx)
        return DetectionStats(
            positions_yx=positions_yx,
            statistic=statistic,
            fpf=fpf,
            dof=dof,
            fwhm_px=self.sampler.fwhm,
            statistic_kind=self.test.kind,
        )

    def map(
        self,
        image: jnp.ndarray,
        validity_map: jnp.ndarray | None = None,
        center_yx: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return the detection statistic at every pixel, in image shape.

        Args:
            image: 2D science image.
            validity_map: Optional (ny, nx) mask (1 = usable, 0 = excluded).
                Measurements touching excluded pixels anywhere in their footprint
                or interpolation stencil are excluded.
            center_yx: Optional (2,) star center; geometric center when None.

        Returns:
            Detection statistic at every pixel, shape (ny, nx). Entries are NaN where
            the sampler reports the geometry invalid or the test gate fails.
        """
        ny, nx = image.shape
        y_coords, x_coords = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        positions_yx = jnp.stack([y_coords.ravel(), x_coords.ravel()], axis=1)
        stats = self(image, positions_yx, validity_map, center_yx)
        return stats.statistic.reshape(image.shape)
