"""Default prior distribution helpers for orbit fitting.

Provides eccentricity prior options and utility transforms for orbital
parameters. All transformations are pure JAX -- the NumPyro ``sample``
calls live in the model builder, not here.

Available eccentricity priors (pass as ``ecc_prior`` to
:func:`~photomancy.orbit.model.build_model`):

============== ===================================================
Name           Description
============== ===================================================
``"kipping13"`` Beta(0.867, 3.03) -- Kipping (2013) short-period
``"rayleigh"``  Rayleigh(sigma=0.3) via Weibull(sigmasqrt2, 2)
``"vaneylen19"`` Rayleigh(sigma=0.049) -- Van Eylen et al. (2019)
``"disk"``      Uniform(0, 1) on *e*, Uniform(0, 2pi) on *omega*
============== ===================================================

.. note::
   Prior versions used a Normal->Disk (Cartesian) transform for the
   ``disk``, ``rayleigh``, and ``vaneylen19`` priors. This was
   replaced with direct sampling because the Cartesian
   parameterization creates pathological geometry for HMC (exploding
   gradients at the origin, vanishing gradients at large radii).
   Additionally, the Normal(0, sigma)->Disk transform produces
   Exponential(sigma^2), not Rayleigh(sigma), on eccentricity -- a subtle
   but critical mathematical error.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# ---------------------------------------------------------------------------
# Pure-JAX transforms (no NumPyro dependency)
# ---------------------------------------------------------------------------


def eccentricity_disk_transform(e_x, e_y):
    """Normal->Disk bijective map: (x, y) -> (e, cos_w, sin_w).

    .. deprecated::
       This transform creates pathological HMC geometry (exploding
       gradients at origin, vanishing gradients at large radius).
       Use direct sampling via :func:`sample_ecc_prior` instead.

    Maps two values to eccentricity and argument of periapsis via the
    chi^2(2) CDF. When ``(e_x, e_y) ~ N(0, 1)^2``, the induced marginal
    on *e* is Uniform(0, 1). When ``(e_x, e_y) ~ N(0, sigma)^2`` with
    ``sigma < 1``, the induced marginal is Exponential(sigma^2), **not**
    Rayleigh(sigma).

    Args:
        e_x: First sample. Scalar.
        e_y: Second sample. Scalar.

    Returns:
        Tuple of ``(e, cos_w, sin_w)``.
    """
    r2 = e_x**2 + e_y**2
    r_safe = jnp.maximum(jnp.sqrt(r2), 1e-10)

    e = 1.0 - jnp.exp(-r2 / 2.0)
    cos_w = e_x / r_safe
    sin_w = e_y / r_safe

    return e, cos_w, sin_w


# ---------------------------------------------------------------------------
# Eccentricity distributions (sampler-context-free)
# ---------------------------------------------------------------------------

# The numpyro Distribution object for each named eccentricity prior. This is the
# single source of truth: the NumPyro samplers below register a ``sample`` site
# from it, and rejection samplers (e.g. photomancy.orbit.ofti) draw from it directly
# via ``.sample(key, shape)`` outside any model context.
_ECC_DISTRIBUTIONS = {
    "disk": lambda: dist.Uniform(0.0, 1.0),
    "rayleigh": lambda: dist.Weibull(scale=0.3 * jnp.sqrt(2.0), concentration=2.0),
    "vaneylen19": lambda: dist.Weibull(scale=0.049 * jnp.sqrt(2.0), concentration=2.0),
    "kipping13": lambda: dist.Beta(0.867, 3.03),
}


def ecc_distribution(name="kipping13"):
    """Return the numpyro Distribution over eccentricity for a named prior.

    Args:
        name: One of ``"kipping13"``, ``"rayleigh"``, ``"vaneylen19"``, or
            ``"disk"``. Default ``"kipping13"``.

    Returns:
        A ``numpyro.distributions.Distribution`` over eccentricity.

    Raises:
        ValueError: If *name* is not a recognized eccentricity prior.
    """
    if name not in _ECC_DISTRIBUTIONS:
        valid = ", ".join(sorted(_ECC_DISTRIBUTIONS))
        raise ValueError(f"Unknown ecc_prior={name!r}. Choose from: {valid}")
    return _ECC_DISTRIBUTIONS[name]()


# ---------------------------------------------------------------------------
# NumPyro eccentricity prior samplers
# ---------------------------------------------------------------------------

# Each function takes no arguments and registers NumPyro sample sites.
# Returns (e, cos_w, sin_w).

_ECC_PRIOR_REGISTRY = {}


def _register(name):
    """Decorator to register an eccentricity prior sampler."""

    def wrapper(fn):
        _ECC_PRIOR_REGISTRY[name] = fn
        return fn

    return wrapper


@_register("disk")
def _sample_ecc_disk():
    """Uniform(0, 1) on e, Uniform(0, 2pi) on omega.

    Samples eccentricity and argument of periapsis directly; NumPyro
    auto-applies a logit bijection for smooth HMC geometry.
    """
    e = numpyro.sample("e_raw", ecc_distribution("disk"))
    w = numpyro.sample("w_raw", dist.Uniform(0.0, 2.0 * jnp.pi))
    return e, jnp.cos(w), jnp.sin(w)


@_register("rayleigh")
def _sample_ecc_rayleigh():
    """Rayleigh(sigma=0.3) on e via Weibull(sigma*sqrt(2), 2). Mode ~ 0.3."""
    e = numpyro.sample("e_raw", ecc_distribution("rayleigh"))
    w = numpyro.sample("w_raw", dist.Uniform(0.0, 2.0 * jnp.pi))
    return e, jnp.cos(w), jnp.sin(w)


@_register("vaneylen19")
def _sample_ecc_vaneylen19():
    """Rayleigh(sigma=0.049) on e -- Van Eylen et al. (2019) for small planets."""
    e = numpyro.sample("e_raw", ecc_distribution("vaneylen19"))
    w = numpyro.sample("w_raw", dist.Uniform(0.0, 2.0 * jnp.pi))
    return e, jnp.cos(w), jnp.sin(w)


@_register("kipping13")
def _sample_ecc_kipping13():
    """Beta(0.867, 3.03) + Uniform omega -- Kipping (2013) for short-period."""
    e = numpyro.sample("e_raw", ecc_distribution("kipping13"))
    w = numpyro.sample("w_raw", dist.Uniform(0.0, 2.0 * jnp.pi))
    return e, jnp.cos(w), jnp.sin(w)


def sample_ecc_prior(ecc_prior="kipping13"):
    """Sample eccentricity and argument of periapsis under the named prior.

    This function must be called inside a NumPyro model context. It registers
    the appropriate ``numpyro.sample`` sites and returns ``(e, cos_w, sin_w)``.

    Args:
        ecc_prior: One of ``"kipping13"``, ``"rayleigh"``, ``"vaneylen19"``,
            or ``"disk"``. Default ``"kipping13"``.

    Returns:
        Tuple of ``(e, cos_w, sin_w)`` -- all scalars.

    Raises:
        ValueError: If *ecc_prior* is not a recognized name.
    """
    if ecc_prior not in _ECC_PRIOR_REGISTRY:
        valid = ", ".join(sorted(_ECC_PRIOR_REGISTRY))
        raise ValueError(f"Unknown ecc_prior={ecc_prior!r}. Choose from: {valid}")
    return _ECC_PRIOR_REGISTRY[ecc_prior]()


# Valid prior names (exported for documentation / tab-completion)
ECC_PRIOR_NAMES = tuple(sorted(_ECC_PRIOR_REGISTRY))

# Default prior bounds (for reference / convenience)
DEFAULT_PRIORS = {
    "log_P_range": (0.0, 5.0),  # log10(days): 1 day to 100,000 days
    "cos_i_range": (-1.0, 1.0),  # isotropic inclination
    "W_range": (0.0, 2 * jnp.pi),  # ascending node
    "M0_range": (0.0, 2 * jnp.pi),  # mean anomaly at epoch
    "jitter_scale": 1e-10,  # HalfNormal scale for jitter (AU/day)
    "log_Mp_range": (-2.0, 4.0),  # log10(M_earth): 0.01-10,000 M_earth
    "Ag_range": (0.0, 1.0),  # geometric albedo
    "log_Lambda_range": (-30.0, -20.0),  # log10(AU^2): photometric area
    "ecc_prior": "kipping13",  # default eccentricity prior
}
