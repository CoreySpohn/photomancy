"""Calibrate Bayesian model evidence as a Gaussian detection significance."""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtri


def sigma_from_log_bayes_factor(log_bayes_factor):
    """Detection significance (sigma) equivalent to a natural-log Bayes factor.

    Invert the Sellke, Bayarri and Berger (2001) bound using the calibration
    applied by Benneke and Seager (2013, Section 2.6 and Table 2). For
    ``B >= 1``, solve ``B = -1 / (e * p * ln(p))`` on the ``p <= 1/e``
    branch, then use the two-sided Gaussian convention
    ``p = erfc(sigma / sqrt(2))``. This is a bound-based calibration, not a
    measured frequentist false-alarm probability.

    The inversions use fixed Newton iterations in log space, avoiding
    overflow in ``B`` and underflow in ``p`` even beyond 20 sigma. The
    function supports :func:`jax.jit` and :func:`jax.vmap`.

    Args:
        log_bayes_factor: Scalar or array of natural-log evidence ratios,
            ``log Z_with - log Z_without``, in float32 or float64. Enable
            JAX's x64 mode when float64 precision is required.

    Returns:
        Elementwise significance with the input shape and floating dtype.
        Negative inputs return NaN; zero gives the finite value at ``p = 1/e``.
        Positive infinity returns infinity. Values below ``B ~ 2.5`` (about
        2 sigma) are weak evidence. Any floor or display cap is the caller's
        choice; no clipping or capping is applied.

    References:
        Sellke, T., Bayarri, M. J., and Berger, J. O. (2001), "Calibration of
        p Values for Testing Precise Null Hypotheses", The American
        Statistician, 55, 62-71. https://doi.org/10.1198/000313001300339950

        Benneke, B. and Seager, S. (2013), "How to Distinguish between Cloudy
        Mini-Neptunes and Water/Volatile-Dominated Super-Earths", ApJ, 778,
        153. https://doi.org/10.1088/0004-637X/778/2/153
    """
    log_bayes_factor = jnp.asarray(log_bayes_factor) * 1.0
    # Substitute a regular value while solving masked inputs and the branch point.
    log_b = jnp.where(
        jnp.isfinite(log_bayes_factor) & (log_bayes_factor > 0),
        log_bayes_factor,
        1.0,
    )

    # u = -ln(p) - 1 >= 0 satisfies u - log1p(u) = ln(B).
    # The square-root start resolves the branch point; log_b dominates in the tail.
    u = log_b + jnp.sqrt(2.0) * jnp.sqrt(log_b)

    def p_step(_, u):
        # Near zero, evaluate the residual by series to avoid cancellation.
        small_u = jnp.minimum(u, 0.01)
        series = small_u**2 * (
            0.5 + small_u * (-1 / 3 + small_u * (0.25 + small_u * (-0.2 + small_u / 6)))
        )
        residual = jnp.where(u < 0.01, series, u - jnp.log1p(u)) - log_b
        return u - residual * (1 + u) / u

    u = jax.lax.fori_loop(0, 8, p_step, u)
    log_p = jnp.where(log_bayes_factor == 0, -1.0, -1.0 - u)

    # Invert log Phi(-sigma) = log(p/2), without exponentiating a tiny p.
    log_q = log_p - jnp.log(2.0)
    sigma = jnp.sqrt(2.0) * jnp.sqrt(-log_q)

    def sigma_step(_, sigma):
        log_tail = log_ndtr(-sigma)
        log_density = -0.5 * sigma**2 - 0.5 * jnp.log(2 * jnp.pi)
        hazard = jnp.exp(log_density - log_tail)
        return sigma + (log_tail - log_q) / hazard

    sigma = jax.lax.fori_loop(0, 6, sigma_step, sigma)
    # The direct lower-tail inverse avoids Newton roundoff near the endpoint.
    central_sigma = -ndtri(jnp.exp(jnp.maximum(log_q, -32.0)))
    sigma = jnp.where(log_q >= -32.0, central_sigma, sigma)
    sigma = jnp.where(jnp.isposinf(log_bayes_factor), jnp.inf, sigma)
    return jnp.where(log_bayes_factor >= 0, sigma, jnp.nan)
