"""Residual analysis with a known background expectation.

Implements 'Faux-RDI' and related tools for yield calculations where we assume
perfect knowledge of the background model (stellar speckles + disk).
"""

import jax.numpy as jnp


def get_perfect_residuals(
    observation: jnp.ndarray,
    expectation_model: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate residuals assuming perfect subtraction of static structure.

    This simulates 'Faux-RDI': we subtract the exact expectation value of
    the star and disk. The residual contains only the fundamental noise
    (photon + read noise) and any signals not in the model (planets).

    Args:
        observation: The noisy data (Data = Poisson(Model) + ReadNoise).
        expectation_model: The noiseless expectation of background sources
            (stellar PSF + disk, excluding planets).

    Returns:
        Residual image containing noise + unmodeled signals (planets).
    """
    return observation - expectation_model
