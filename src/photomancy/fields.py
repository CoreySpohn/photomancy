"""Stationary Gaussian random fields in whitened coordinates.

A map-valued parameter (a wavefront error map, a surface, a background) with a Gaussian
prior of known power spectral density is best fitted through its whitened coordinates:
the leaf is a white field ``w`` with independent standard-normal entries, and the map is
``x = color(w)``. The prior term is then ``-0.5 * sum(w**2)``, every direction of ``w``
is equally scaled for a first-order optimizer, and directions the data do not constrain
stay at their prior value instead of drifting.

Convention: on an ``n x n`` grid, ``color(w) = Re(IFFT2(A * FFT2(w)))`` masked to the
support, with ``A = sqrt(PSD)`` sampled on the FFT frequency grid. For an unmasked
field ``E|FFT2(x)[k]|^2 / n^2 = A[k]^2``, so ``psd`` is the expected periodogram per
grid cell and the marginal variance of every pixel is ``mean(A**2)``.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def radial_frequency(n, diameter):
    """Radial frequency of an ``n x n`` FFT grid, in cycles per ``diameter`` samples."""
    f = np.fft.fftfreq(n) * diameter
    return np.hypot(*np.meshgrid(f, f, indexing="ij"))


class StationaryGaussianField(eqx.Module):
    """A zero-mean stationary Gaussian field on a square grid, restricted to a support.

    Attributes:
        amplitude: Square root of the PSD on the FFT frequency grid, ``(n, n)``,
            symmetric under ``k -> -k`` so the colored field is real.
        support: Boolean mask ``(n, n)``; the field is zero outside it.
    """

    amplitude: Array
    support: Array

    def __check_init__(self):  # noqa: D105
        if (
            self.amplitude.ndim != 2
            or self.amplitude.shape[0] != self.amplitude.shape[1]
        ):
            raise ValueError(
                f"amplitude must be square 2-D, got {self.amplitude.shape}"
            )
        if self.support.shape != self.amplitude.shape:
            raise ValueError(
                f"support {self.support.shape} differs from amplitude "
                f"{self.amplitude.shape}"
            )

    @classmethod
    def from_radial_psd(
        cls,
        psd: Callable,
        support,
        diameter: float,
        dtype=jnp.float32,
    ) -> "StationaryGaussianField":
        """Field whose PSD depends only on the radial frequency.

        Args:
            psd: ``k -> PSD`` (NumPy), ``k`` in cycles per ``diameter`` samples, in
                squared field units per grid cell (the module docstring's convention).
            support: Boolean ``(n, n)`` mask.
            diameter: Samples across the support's diameter (sets the frequency unit).
            dtype: Floating dtype of the amplitude.
        """
        support = np.asarray(support, bool)
        k = radial_frequency(support.shape[0], diameter)
        power = np.asarray(psd(k), float)
        if np.any(~np.isfinite(power)) or np.any(power < 0):
            raise ValueError("psd must be finite and non-negative on the grid")
        return cls(jnp.asarray(np.sqrt(power), dtype), jnp.asarray(support))

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(n, n)``."""
        return self.amplitude.shape

    def color(self, white: Array) -> Array:
        """The field ``x`` of the white coordinates ``w`` (linear in ``w``)."""
        x = jnp.real(jnp.fft.ifft2(self.amplitude * jnp.fft.fft2(white)))
        return jnp.where(self.support, x, 0.0).astype(white.dtype)

    def log_prob_white(self, white: Array) -> Array:
        """Standard-normal log density of the white coordinates, up to a constant."""
        return -0.5 * jnp.sum(white**2)

    def sample(self, key, n_samples: int | None = None) -> Array:
        """Field draws, ``(n, n)``, or ``(n_samples, n, n)`` for ``n_samples``."""
        dtype = self.amplitude.dtype
        if n_samples is None:
            return self.color(jax.random.normal(key, self.shape, dtype))
        white = jax.random.normal(key, (n_samples, *self.shape), dtype)
        return jax.vmap(self.color)(white)

    def pixel_variance(self) -> Array:
        """Marginal variance of one pixel inside the support."""
        return jnp.mean(self.amplitude**2)
