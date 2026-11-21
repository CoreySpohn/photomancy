"""Synthetic systems, observations and tolerances shared across the test files."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import G, Mearth2kg, Msun2kg, Rearth2AU
from hwoutils.epoch import Epoch
from optixstuff.spectral import Band, Quadrature, Transmission
from orbix.equations import period_to_sma
from orbix.system import Body, ClassicalChart, SkyFrame, System
from priorities import Uniform
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.population import kipping_eccentricity, rayleigh_eccentricity
from skyscapes.scene import FlatStar, Planet, Scene

from photomancy import (
    PhotometryTerm,
    ProperMotionAnomalyTerm,
    RadialVelocityTerm,
    StellarAstrometryTerm,
)
from photomancy.measurements import (
    AstrometryData,
    PhotometryData,
    ProperMotionAnomalyData,
    RadialVelocityData,
)


def ecc_law(name):
    """The named eccentricity distribution, built on demand so it follows x64."""
    return {
        "disk": lambda: Uniform(0.0, 1.0),
        "kipping13": kipping_eccentricity,
        "rayleigh": lambda: rayleigh_eccentricity(0.3, 1.0),
        "vaneylen19": lambda: rayleigh_eccentricity(0.049, 1.0),
    }[name]()


def classical_coordinates(a, e, cos_i, W, w, tp, period_d):
    """``ClassicalChart`` coordinates at day zero of classical elements.

    ``a`` in AU, angles in radians, periapsis time ``tp`` and ``period_d`` in days:
    ``(ln a, logit e, atanh cos i, W, w, M0)`` with ``M0 = n (0 - tp)``.
    """
    mean_anomaly = (2 * jnp.pi / period_d) * (0.0 - tp)
    return jnp.stack(
        [
            jnp.log(a),
            jnp.log(e) - jnp.log1p(-e),
            jnp.arctanh(cos_i),
            jnp.asarray(W) + 0.0,
            jnp.asarray(w) + 0.0,
            mean_anomaly % (2 * jnp.pi),
        ]
    )


def _floor(*terms):
    """Float64 accumulation floor specified for comparisons of log densities."""
    return max(
        1e-10, 64 * np.finfo(float).eps * sum(np.sum(1 + np.abs(t)) for t in terms)
    )


BOUNDS = {
    "mass_Mearth": (0.01, 1e4),
    "radius_AU": (1e-5, 10**-2.5),
    "albedo": (0.01, 1.0),
}


def _orbits(mass_Mearth=0.0):
    mass_kg = Msun2kg + Mearth2kg * mass_Mearth
    period_d = jnp.array([75.0, 320.0, 1130.0, 3800.0])
    x = jnp.stack(
        [
            jnp.log(period_to_sma(period_d, mass_kg)),
            jnp.array([-3.1, -1.6, -0.2, 1.4]),
            jnp.array([1.0, 0.2, -0.4, -1.2]),
            jnp.array([0.2, 1.7, 3.4, 5.6]),
            jnp.array([0.6, 2.1, 3.1, 4.4]),
            jnp.array([0.7, 1.6, 4.2, 5.7]),
        ],
        axis=-1,
    )
    return ClassicalChart().to_orbit(x, mu_AU3_per_d2=G * mass_kg)


def _scene(orbit, *, mass_Mearth, radius_AU, albedo):
    dynamics = System(
        Body(1.0),
        Body(jnp.atleast_1d(mass_Mearth * Mearth2kg / Msun2kg)),
        jax.tree.map(lambda x: x[None], orbit),
        SkyFrame(10.0, Epoch()),
    )
    return Scene(
        FlatStar(1.0),
        (
            Planet(
                Rp_Rearth=jnp.atleast_1d(radius_AU / Rearth2AU),
                physical_model=LambertianPhysicalModel(Ag=jnp.atleast_1d(albedo)),
            ),
        ),
        dynamics,
    )


_TIMES = np.array([31.0, 120.0, 290.0, 470.0])


def _data(max_n=None):
    """Nonzero residuals and unequal correlated errors exercise every element."""
    return AstrometryData.from_separation_pa(
        time_d=_TIMES,
        separation_arcsec=[0.13, 0.11, 0.09, 0.15],
        separation_err_arcsec=[0.004] * 4,
        pa_deg=[45.0, 120.0, 210.0, 310.0],
        pa_err_deg=[1.0] * 4,
        max_n=max_n,
    )


def _new_term(kind, max_n=4):
    """Small observations with residuals and nontrivial covariance/instruments."""
    if kind == "stellar":
        return StellarAstrometryTerm(
            AstrometryData.pad(
                time_d=_TIMES,
                east_arcsec=[1e-4, -2e-4, 3e-4, 2e-4],
                north_arcsec=[-1e-4, 1e-4, 2e-4, -2e-4],
                east_err_arcsec=[5e-5] * 4,
                north_err_arcsec=[8e-5] * 4,
                corr=[0.2, -0.3, 0.1, 0.0],
                max_n=max_n,
            )
        )
    if kind == "rv":
        return RadialVelocityTerm(
            RadialVelocityData.pad(
                time_d=_TIMES,
                rv=[2.0, -4.0, 8.0, 1.0],
                rv_err=[1.5] * 4,
                inst_ids=[0, 1, 0, 1],
                n_inst=3,
                max_n=max_n,
            ),
            jitters=jnp.array([0.5, 0.8, 1.0]),
        )
    if kind == "pma":
        return ProperMotionAnomalyTerm(
            ProperMotionAnomalyData(
                t_hip=jnp.array(0.0),
                t_gaia=jnp.array(9000.0),
                gaia_window=jnp.array(1023.0),
                pm_anomaly=jnp.array([2e-7, -3e-7]),
                pm_anomaly_cov=jnp.array([[4e-14, 1e-14], [1e-14, 9e-14]]),
                valid=jnp.array(True),
            )
        )
    return PhotometryTerm(
        PhotometryData.from_dmag(
            time_d=_TIMES,
            dmag=[25.0, 25.2, 24.5, 26.0],
            dmag_err=[0.2] * 4,
            band_id=[0] * 4,
            is_upper_limit=[False, True, False, True],
            max_n=max_n,
        ),
        Quadrature.build((Band(550.0, 650.0),)),
        (Transmission.from_band(Band(550.0, 650.0)),),
    )


def rv_normalization(data):
    """``0.5 sum_j log N_j`` over the populated instruments of ``RadialVelocityData``.

    The normalization of the flat-offset RV marginal, for references frozen before
    the marginal was a normalized density.
    """
    counts = np.bincount(
        np.asarray(data.inst_ids)[np.asarray(data.valid)], minlength=data.n_inst
    )
    return 0.5 * float(np.sum(np.log(counts[counts > 0])))
