"""Concrete detection filters: aperture, Gaussian, and PSF-template matched."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.map_coordinates import map_coordinates

from photomancy.postproc.core.detection.base import AbstractFilter
from photomancy.postproc.core.detection.patches import extract_patch
from photomancy.postproc.core.photometry import flux_map
from photomancy.postproc.templates.base import AbstractTemplateProvider


def gaussian_kernel_1d(fwhm: float) -> jnp.ndarray:
    """Normalized 1D Gaussian kernel for separable matched filtering.

    Uses sigma = fwhm / 2.355 and an odd kernel size spanning about six sigma.

    Args:
        fwhm: Full width at half maximum in pixels (concrete Python float).

    Returns:
        1D kernel summing to 1.
    """
    sigma = fwhm / 2.355
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    return kernel_1d / jnp.sum(kernel_1d)


@jax.jit
def gaussian_filter_2d(image: jnp.ndarray, kernel_1d: jnp.ndarray) -> jnp.ndarray:
    """Apply a 2D Gaussian filter using separable convolution.

    Args:
        image: 2D image array.
        kernel_1d: 1D kernel from gaussian_kernel_1d.

    Returns:
        Filtered image, same shape as input.
    """
    pad_size = len(kernel_1d) // 2

    padded = jnp.pad(image, ((0, 0), (pad_size, pad_size)), mode="reflect")
    row_conv = jax.vmap(lambda row: jnp.convolve(row, kernel_1d, mode="valid"))(padded)

    padded = jnp.pad(row_conv, ((pad_size, pad_size), (0, 0)), mode="reflect")
    col_conv = jax.vmap(
        lambda col: jnp.convolve(col, kernel_1d, mode="valid"), in_axes=1, out_axes=1
    )(padded)

    return col_conv


class ApertureFilter(AbstractFilter):
    """Aperture-photometry filter: convolve once, sample the flux map.

    The prepared map is the aperture flux map (convolution with a circular
    kernel); evaluation samples it with sub-pixel interpolation. This is the
    Mawet et al. (2014) signal measurement.

    Attributes:
        kernel: 2D aperture kernel convolved with the image to form the flux map.
        order: Interpolation order used to sample the flux map.
    """

    kernel: jnp.ndarray
    order: int = eqx.field(static=True)

    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """Aperture flux map of the image.

        Args:
            image: 2D science image.

        Returns:
            Aperture flux map, same shape as the image.
        """
        return flux_map(image, self.kernel)

    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Sample the flux map at one position.

        Args:
            filtered: Aperture flux map from prepare().
            position_yx: (2,) position (y, x) in pixels.

        Returns:
            Scalar flux-map value interpolated at the position.
        """
        return map_coordinates(filtered, position_yx.reshape(2, 1), order=self.order)[0]


class GaussianFilter(AbstractFilter):
    """Gaussian matched filter: separable convolution plus point sampling.

    Attributes:
        kernel_1d: Normalized 1D Gaussian kernel applied along each axis.
        order: Interpolation order used to sample the filtered image.
    """

    kernel_1d: jnp.ndarray
    order: int = eqx.field(static=True)

    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """Gaussian-filtered image.

        Args:
            image: 2D science image.

        Returns:
            Gaussian-filtered image, same shape as the image.
        """
        return gaussian_filter_2d(image, self.kernel_1d)

    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Sample the filtered image at one position.

        Args:
            filtered: Filtered image from prepare().
            position_yx: (2,) position (y, x) in pixels.

        Returns:
            Scalar filtered-image value interpolated at the position.
        """
        return map_coordinates(filtered, position_yx.reshape(2, 1), order=self.order)[0]


class MapFilter(AbstractFilter):
    """Filter that reads a precomputed statistic map instead of filtering.

    Pass the statistic map (for example a likelihood-ratio z map) to the
    DetectionEstimator in place of the image: prepare() returns it unchanged
    and evaluate() samples it with sub-pixel interpolation, so a sampler
    and test calibrate the map against reference positions on the same ring
    as each candidate.

    Non-finite map pixels are replaced by zero in prepare() so they cannot
    poison the reference statistics; mark the same pixels as unusable in the
    estimator's validity_map so the samplers drop them.

    Attributes:
        order: Interpolation order used to sample the map (0 nearest, 1 bilinear).
    """

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 1):
        """Configure the interpolation.

        Args:
            order: Interpolation order for sampling the map (0 nearest,
                1 bilinear).
        """
        self.order = order

    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """The statistic map with non-finite pixels set to zero.

        Args:
            image: 2D precomputed statistic map.

        Returns:
            The map with non-finite pixels replaced by zero.
        """
        return jnp.where(jnp.isfinite(image), image, 0.0)

    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Sample the statistic map at one position.

        Args:
            filtered: Statistic map from prepare().
            position_yx: (2,) position (y, x) in pixels.

        Returns:
            Scalar map value interpolated at the position.
        """
        return map_coordinates(filtered, position_yx.reshape(2, 1), order=self.order)[0]


class PSFTemplateFilter(AbstractFilter):
    """Position-dependent PSF-template matched filter (local inner product).

    A coronagraph PSF changes shape with separation, which breaks the shift
    invariance that makes convolution a matched filter; this filter instead
    computes a normalized inner product between the local image patch and
    the candidate position's own PSF template, both zero-meaned. The
    response is the best-fit template amplitude, so injecting
    amplitude * template at the candidate adds amplitude to the response.

    With a per-pixel noise variance map the inner product is
    inverse-variance weighted (diagonal generalized least squares with a
    fitted constant background; Ruffio et al. 2017): weighted means replace
    means and weighted sums replace sums, reducing exactly to the
    unweighted form when the variance is uniform.

    The template is bound per candidate (see AbstractFilter.bind): the
    estimator queries the provider once per candidate and evaluates the
    same bound template at the candidate and at every reference position.

    Attributes:
        provider: Template source; its stamp_size sets the patch size.
        order: Interpolation order for image patch extraction.
        noise_variance: Optional (ny, nx) per-pixel variance of the detection
            image, or None for the unweighted fit.
        template: Stamp bound to the current candidate, or None until bind() is
            called.
    """

    provider: AbstractTemplateProvider
    order: int = eqx.field(static=True)
    noise_variance: jnp.ndarray | None
    template: jnp.ndarray | None

    def __init__(
        self,
        provider: AbstractTemplateProvider,
        order: int = 3,
        noise_variance: jnp.ndarray | None = None,
        template: jnp.ndarray | None = None,
    ):
        """Configure the filter.

        Args:
            provider: Template source; its stamp_size sets the patch size.
            order: Interpolation order for image patch extraction.
            noise_variance: Optional (ny, nx) per-pixel variance of the
                detection image (same units squared); enables
                inverse-variance weighting.
            template: The bound stamp; leave None (bind() fills it).
        """
        self.provider = provider
        self.order = order
        self.noise_variance = noise_variance
        self.template = template

    def bind(self, position_yx: jnp.ndarray) -> "PSFTemplateFilter":
        """Fetch and attach the template for this candidate position.

        Args:
            position_yx: (2,) candidate position (y, x) in pixels, passed to the
                template provider.

        Returns:
            Copy of this filter with template set to the provider stamp at the
            position.
        """
        return eqx.tree_at(
            lambda f: f.template,
            self,
            self.provider(position_yx),
            is_leaf=lambda x: x is None,
        )

    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """Identity: a position-dependent template admits no precompute.

        Args:
            image: 2D science image.

        Returns:
            The image unchanged.
        """
        return image

    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Best-fit template amplitude at one position.

        Args:
            filtered: Image returned by prepare().
            position_yx: (2,) position (y, x) in pixels at which the patch is taken.

        Returns:
            Scalar best-fit template amplitude, in image units per unit template
            amplitude.
        """
        if self.template is None:
            raise ValueError(
                "PSFTemplateFilter is unbound; DetectionEstimator binds it per "
                "candidate, or call filter.bind(position_yx) explicitly."
            )
        size = self.provider.stamp_size
        img_patch = extract_patch(filtered, position_yx, size, order=self.order)
        template = self.template
        if self.noise_variance is None:
            img_zm = img_patch - jnp.mean(img_patch)
            t_zm = template - jnp.mean(template)
            return jnp.sum(img_zm * t_zm) / jnp.maximum(jnp.sum(t_zm**2), 1e-10)
        variance = extract_patch(self.noise_variance, position_yx, size, order=1)
        weights = 1.0 / jnp.maximum(variance, 1e-12)
        w_sum = jnp.sum(weights)
        img_zm = img_patch - jnp.sum(weights * img_patch) / w_sum
        t_zm = template - jnp.sum(weights * template) / w_sum
        return jnp.sum(weights * img_zm * t_zm) / jnp.maximum(
            jnp.sum(weights * t_zm**2), 1e-10
        )
