"""Template providers: fixed-size PSF stamps for a candidate position."""

import abc

import equinox as eqx
import jax.numpy as jnp


class AbstractTemplateProvider(eqx.Module):
    """Provides the expected planet signal stamp at a given image position.

    A provider returns a fixed-size square stamp -- the expected (noiseless)
    planet signal centered on the candidate's pixel position, sampled on the
    science image grid. The stamp's overall scale is arbitrary: template
    filters normalize it, so the filter response reads as best-fit template
    amplitude in image units per unit template amplitude.

    Attributes:
        stamp_size: Side length of the square template stamp, in pixels.
    """

    stamp_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(self, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Stamp for one candidate position.

        Args:
            position_yx: (2,) candidate position (y, x) in pixels; may be a
                traced value.

        Returns:
            (stamp_size, stamp_size) template stamp.
        """


class ArrayTemplateProvider(AbstractTemplateProvider):
    """A single precomputed stamp, independent of position.

    The simplest provider: appropriate when the PSF is effectively constant
    over the field (or as a controlled test double). The stamp must be
    square.

    Attributes:
        stamp: (K, K) template stamp returned for every position.
        stamp_size: Side length K of the stamp, in pixels.
    """

    stamp: jnp.ndarray
    stamp_size: int = eqx.field(static=True)

    def __init__(self, stamp: jnp.ndarray):
        """Wrap a precomputed square stamp.

        Args:
            stamp: (K, K) template stamp.
        """
        stamp = jnp.asarray(stamp)
        if stamp.ndim != 2 or stamp.shape[0] != stamp.shape[1]:
            raise ValueError(f"stamp must be square 2D; got shape {stamp.shape}")
        self.stamp = stamp
        self.stamp_size = int(stamp.shape[0])

    def __call__(self, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Return the fixed stamp regardless of position.

        Args:
            position_yx: (2,) candidate position (y, x) in pixels; ignored.

        Returns:
            The (stamp_size, stamp_size) stamp.
        """
        del position_yx
        return self.stamp
