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
import jax
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
    configuration without a resolution element; the resolution element comes
    from the FrameSet. A single instance can be reused across observations.
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

    Frames must share one orientation. Frames at different roll are not yet
    combined; the arms do not register or derotate frames.

    Attributes:
        soft: Whether aperture edges are soft.
        sharpness: Sigmoid sharpness for soft edges.
        fast: Whether interpolation is bilinear instead of cubic.
        max_apertures: Static reference-aperture buffer size.
        exclusion_buffer: Reference gap in angular-step units.
        aperture_fwhm: Aperture diameter in multiples of science.fwhm_px.
    """

    soft: bool = eqx.field(static=True)
    sharpness: float = eqx.field(static=True)
    fast: bool = eqx.field(static=True)
    max_apertures: int = eqx.field(static=True)
    exclusion_buffer: float
    aperture_fwhm: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        soft: bool = True,
        sharpness: float = 10.0,
        fast: bool = False,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
        aperture_fwhm: float = 1.0,
    ):
        """Configure aperture photometry independently of the science resolution.

        Every option is keyword-only.

        Args:
            soft: Use differentiable soft aperture edges.
            sharpness: Sigmoid sharpness for soft apertures.
            fast: Bilinear (True) vs cubic (False) sub-pixel sampling.
            max_apertures: Static reference-aperture buffer size.
            exclusion_buffer: Gap between test and first reference aperture,
                in angular-step units.
            aperture_fwhm: Aperture diameter as a multiple of science.fwhm_px.
        """
        self.soft = soft
        self.sharpness = sharpness
        self.fast = fast
        self.max_apertures = max_apertures
        self.exclusion_buffer = exclusion_buffer
        self.aperture_fwhm = aperture_fwhm

    def estimator(self, science: FrameSet) -> DetectionEstimator:
        """Compose the aperture estimator for these science frames.

        Kernel shape requires reading science.fwhm_px as a concrete host float.
        Under jit it must be concrete (closed over, not a traced argument).

        Args:
            science: Frames supplying the resolution element in pixels.

        Returns:
            Aperture filter, reference sampler, and two-sample t-test.
        """
        # Keep the cached kernel concrete even on the first call inside jit.
        with jax.ensure_compile_time_eval():
            return DetectionEstimator.mawet(
                fwhm=self.aperture_fwhm * float(science.fwhm_px),
                soft=self.soft,
                sharpness=self.sharpness,
                fast=self.fast,
                max_apertures=self.max_apertures,
                exclusion_buffer=self.exclusion_buffer,
            )

    def detect(
        self,
        science: FrameSet,
        positions_yx: jnp.ndarray,
        *,
        references: FrameSet | None = None,
    ) -> DetectionStats:
        """Compute the Mawet t-statistic and FPF on the science coadd.

        The aperture kernel size depends on science.fwhm_px, which is read as a
        concrete host float; under jit, close over the FrameSet (or at least its
        fwhm_px) rather than passing it as a traced argument.

        Args:
            science: Science frames plus metadata; the arm uses their coadd.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            references: Ignored; the arm takes its noise sample from the science image.

        Returns:
            DetectionStats for the n candidates, with statistic_kind "mawet_t".
        """
        del references  # Mawet self-references; see class docstring.
        orientation_ok = science.single_orientation()
        image = science.coadd()
        stats = self.estimator(science)(
            image,
            positions_yx,
            science.valid,
            jnp.array([science.grid.center_y, science.grid.center_x]),
        )
        return DetectionStats(
            positions_yx=stats.positions_yx,
            statistic=jnp.where(orientation_ok, stats.statistic, jnp.nan),
            fpf=jnp.where(orientation_ok, stats.fpf, jnp.nan),
            dof=stats.dof,
            fwhm_px=science.fwhm_px,
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

    Frames must share one orientation. Frames at different roll are not yet
    combined; the arms do not register or derotate frames.

    Attributes:
        provider: Template source whose stamp size sets the fit window.
        aperture_fwhm: Reference-aperture diameter in multiples of science.fwhm_px.
        exclusion_buffer: Gap between the candidate and the first reference
            aperture, in angular-step units.
        order: Interpolation order for patch extraction.
        max_apertures: Static reference-aperture buffer size.
        whiten: Whether FrameSet.noise_variance, when present, inverse-variance
            weights the template fit.
    """

    provider: AbstractTemplateProvider
    aperture_fwhm: float
    exclusion_buffer: float
    order: int = eqx.field(static=True)
    max_apertures: int = eqx.field(static=True)
    whiten: bool = eqx.field(static=True)

    def __init__(
        self,
        provider: AbstractTemplateProvider,
        *,
        order: int = 3,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
        whiten: bool = True,
        aperture_fwhm: float = 1.0,
    ):
        """Build the arm around a template provider.

        Every option after the provider is keyword-only.

        Args:
            provider: Template source (its stamp size sets the fit window).
            order: Interpolation order for patch extraction.
            max_apertures: Static reference-aperture buffer size.
            exclusion_buffer: Gap between test and first reference aperture,
                in angular-step units.
            whiten: Use FrameSet.noise_variance (when present) to
                inverse-variance weight the template fit.
            aperture_fwhm: Reference-aperture diameter as a multiple of science.fwhm_px.
        """
        self.provider = provider
        self.aperture_fwhm = aperture_fwhm
        self.order = order
        self.max_apertures = max_apertures
        self.exclusion_buffer = exclusion_buffer
        self.whiten = whiten

    def estimator(self, science: FrameSet) -> DetectionEstimator:
        """Compose the template estimator for these science frames.

        Args:
            science: Frames supplying resolution and optional noise variance.

        Returns:
            Template filter, reference sampler, and two-sample t-test. The
            reference width can be traced under jit.
        """
        variance = None
        if self.whiten and science.noise_variance is not None:
            total_time = jnp.sum(science.frame_time_s)
            variance = jnp.sum(science.noise_variance, axis=0) / total_time**2
            if science.valid is not None:
                # Excluded pixels never enter a kept measurement; a neutral
                # value keeps a non-finite variance out of masked samples.
                variance = jnp.where(science.valid > 0.5, variance, 1.0)
        return DetectionEstimator(
            filter=PSFTemplateFilter(
                provider=self.provider, order=self.order, noise_variance=variance
            ),
            sampler=ApertureSampler(
                fwhm=self.aperture_fwhm * science.fwhm_px,
                max_apertures=self.max_apertures,
                exclusion_buffer=self.exclusion_buffer,
            ),
            test=TwoSampleTTest(),
        )

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
        orientation_ok = science.single_orientation()
        image = science.coadd()
        stats = self.estimator(science)(
            image,
            positions_yx,
            science.valid,
            jnp.array([science.grid.center_y, science.grid.center_x]),
        )
        return DetectionStats(
            positions_yx=stats.positions_yx,
            statistic=jnp.where(orientation_ok, stats.statistic, jnp.nan),
            fpf=jnp.where(orientation_ok, stats.fpf, jnp.nan),
            dof=stats.dof,
            fwhm_px=science.fwhm_px,
            statistic_kind="psf_template_t",
        )
