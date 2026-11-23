"""The post-processing seam: an abstract interface plus concrete arms.

A post-processing arm consumes a FrameSet (science frames plus metadata,
optionally with a reference FrameSet) and produces DetectionStats
(per-candidate statistic, false positive fraction, and degrees of freedom).
Downstream mission simulators register arms against AbstractDetectionArm
and thread FrameSet/DetectionStats through their pipelines; this module owns
the vocabulary and must stay import-clean of any downstream package.
"""

import abc

import equinox as eqx
import jax.numpy as jnp

from photomancy.postproc.contracts import DetectionStats, FrameSet
from photomancy.postproc.core.detection import (
    ApertureSampler,
    DetectionEstimator,
    PSFTemplateFilter,
    TwoSampleTTest,
)
from photomancy.postproc.templates.base import AbstractTemplateProvider


class AbstractDetectionArm(eqx.Module):
    """Abstract post-processing arm: FrameSet -> DetectionStats.

    Implementations are Equinox modules (PyTrees), constructed per
    configuration (e.g. per band, with a concrete resolution element), so a
    single instance JIT-compiles cleanly inside larger pipelines.
    """

    @abc.abstractmethod
    def detect(
        self,
        science: FrameSet,
        positions_yx: jnp.ndarray,
        *,
        references: FrameSet | None = None,
    ) -> DetectionStats:
        """Compute detection statistics at candidate positions.

        Args:
            science: Science frames plus metadata.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            references: Optional reference frames (for arms that difference
                against a reference library). Arms that self-reference may
                ignore it.

        Returns:
            DetectionStats for the n candidates.
        """


class MawetArm(AbstractDetectionArm):
    """Aperture-photometry detection on the coadded frame (Mawet et al. 2014).

    The statistic is the small-sample-corrected two-sample t-statistic of the
    candidate aperture against same-radius reference apertures; the FPF is
    its Student-t upper tail with (n_references - 1) degrees of freedom. The
    statistic is scale-invariant, so it is computed on the exposure-weighted
    coadd regardless of frame count. References are ignored: the noise sample
    comes from the science image itself.
    FrameSet.grid supplies the star center for reference-aperture geometry.

    Attributes:
        estimator: Composed aperture-photometry t-test estimator (filter, sampler
            and test) that the arm applies to the coadded frame.
    """

    estimator: DetectionEstimator

    def __init__(
        self,
        fwhm_px: float,
        soft: bool = True,
        sharpness: float = 10.0,
        fast: bool = False,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
    ):
        """Build the arm with a concrete resolution element.

        Args:
            fwhm_px: Resolution element (lambda/D) in pixels; must match the
                FrameSets this arm is applied to.
            soft: Use differentiable soft aperture edges.
            sharpness: Sigmoid sharpness for soft apertures.
            fast: Bilinear (True) vs cubic (False) sub-pixel sampling.
            max_apertures: Static reference-aperture buffer size.
            exclusion_buffer: Gap between test and first reference aperture,
                in angular-step units.
        """
        self.estimator = DetectionEstimator.mawet(
            fwhm=fwhm_px,
            soft=soft,
            sharpness=sharpness,
            fast=fast,
            max_apertures=max_apertures,
            exclusion_buffer=exclusion_buffer,
        )

    def detect(
        self,
        science: FrameSet,
        positions_yx: jnp.ndarray,
        *,
        references: FrameSet | None = None,
    ) -> DetectionStats:
        """Compute the Mawet t-statistic and FPF on the science coadd.

        Args:
            science: Science frames plus metadata; the arm uses their coadd.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            references: Ignored; the arm takes its noise sample from the science image.

        Returns:
            DetectionStats for the n candidates, with statistic_kind "mawet_t".
        """
        del references  # Mawet self-references; see class docstring.
        image = science.coadd()
        stats = self.estimator(
            image,
            positions_yx,
            science.valid,
            jnp.array([science.grid.center_y, science.grid.center_x]),
        )
        return DetectionStats(
            positions_yx=stats.positions_yx,
            statistic=stats.statistic,
            fpf=stats.fpf,
            dof=stats.dof,
            fwhm_px=stats.fwhm_px,
            statistic_kind="mawet_t",
        )


class MatchedFilterArm(AbstractDetectionArm):
    """PSF-template matched-filter detection on the coadded frame.

    The statistic is the small-sample-corrected two-sample t-statistic of
    the candidate's best-fit template amplitude against the same filter
    evaluated at same-radius reference apertures (exact t with
    n_references - 1 degrees of freedom; the reference spread is the
    empirical matched-filter noise normalization of Ruffio et al. 2017).
    When whiten is True and the FrameSet carries noise_variance, the
    template fit is inverse-variance weighted using the coadd-propagated
    variance sum(noise_variance) / sum(frame_time_s)**2.

    References are ignored: the noise sample comes from the science image
    itself. FrameSet.grid supplies the reference-aperture center; the
    provider must use the same science grid.

    Attributes:
        provider: Template source whose stamp size sets the fit window.
        fwhm_px: Resolution element (lambda/D) in pixels, used for the
            reference-aperture geometry.
        exclusion_buffer: Gap between the candidate and the first reference
            aperture, in angular-step units.
        order: Interpolation order for patch extraction.
        max_apertures: Static reference-aperture buffer size.
        whiten: Whether FrameSet.noise_variance, when present, inverse-variance
            weights the template fit.
    """

    provider: AbstractTemplateProvider
    fwhm_px: float
    exclusion_buffer: float
    order: int = eqx.field(static=True)
    max_apertures: int = eqx.field(static=True)
    whiten: bool = eqx.field(static=True)

    def __init__(
        self,
        provider: AbstractTemplateProvider,
        fwhm_px: float,
        order: int = 3,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
        whiten: bool = True,
    ):
        """Build the arm around a template provider.

        Args:
            provider: Template source (its stamp size sets the fit window).
            fwhm_px: Resolution element (lambda/D) in pixels, for the
                reference-aperture geometry.
            order: Interpolation order for patch extraction.
            max_apertures: Static reference-aperture buffer size.
            exclusion_buffer: Gap between test and first reference aperture,
                in angular-step units.
            whiten: Use FrameSet.noise_variance (when present) to
                inverse-variance weight the template fit.
        """
        self.provider = provider
        self.fwhm_px = fwhm_px
        self.order = order
        self.max_apertures = max_apertures
        self.exclusion_buffer = exclusion_buffer
        self.whiten = whiten

    def detect(
        self,
        science: FrameSet,
        positions_yx: jnp.ndarray,
        *,
        references: FrameSet | None = None,
    ) -> DetectionStats:
        """Compute the template-fit t-statistic and FPF on the science coadd.

        Args:
            science: Science frames plus metadata; the arm uses their coadd and, when
                whitening, the propagated noise variance.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            references: Ignored; the arm takes its noise sample from the science image.

        Returns:
            DetectionStats for the n candidates, with statistic_kind "psf_template_t".
        """
        del references  # Self-referencing arm; see class docstring.
        image = science.coadd()
        variance = None
        if self.whiten and science.noise_variance is not None:
            total_time = jnp.sum(science.frame_time_s)
            variance = jnp.sum(science.noise_variance, axis=0) / total_time**2
        composed = DetectionEstimator(
            filter=PSFTemplateFilter(
                provider=self.provider, order=self.order, noise_variance=variance
            ),
            sampler=ApertureSampler(
                fwhm=self.fwhm_px,
                max_apertures=self.max_apertures,
                exclusion_buffer=self.exclusion_buffer,
            ),
            test=TwoSampleTTest(),
        )
        stats = composed(
            image,
            positions_yx,
            science.valid,
            jnp.array([science.grid.center_y, science.grid.center_x]),
        )
        return DetectionStats(
            positions_yx=stats.positions_yx,
            statistic=stats.statistic,
            fpf=stats.fpf,
            dof=stats.dof,
            fwhm_px=stats.fwhm_px,
            statistic_kind="psf_template_t",
        )
