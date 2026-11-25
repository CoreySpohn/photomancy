"""Tests for photomancy.orbit module."""

import jax
import jax.numpy as jnp
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg, Rearth2AU
from kinematics import (
    predict_photometry,
    predict_relative_astrometry,
    predict_rv,
)
from orbix.kepler import Halley, kepler

from photomancy.measurements import (
    AstrometryData,
    RadialVelocityData,
    loglike_relative_astrom,
    loglike_rv_marginalized,
)

# Enable float64 for numerical stability in gradient checks

# ============================================================================
# Test fixtures -- common orbital parameters
# ============================================================================


@pytest.fixture
def circular_orbit():
    """Earth-like circular orbit around a solar-mass star."""
    return dict(
        T=365.25,  # days
        a=1.0,  # AU
        e=0.0,
        cos_i=0.0,  # edge-on (i=90deg)
        cos_w=1.0,  # omega=0
        sin_w=0.0,
        W=0.0,  # Omega=0
        tp=0.0,
        Ms=Msun2kg,
        Mp_sini=Mearth2kg,
        dist_pc=10.0,
    )


@pytest.fixture
def eccentric_orbit():
    """Moderately eccentric orbit."""
    e = 0.3
    w = jnp.pi / 4  # omega = 45deg
    return dict(
        T=200.0,
        a=0.7,
        e=e,
        cos_i=jnp.cos(jnp.radians(60.0)),
        cos_w=jnp.cos(w),
        sin_w=jnp.sin(w),
        W=jnp.pi / 3,
        tp=50.0,
        Ms=Msun2kg,
        Mp_sini=3.0 * Mearth2kg,
        dist_pc=5.0,
    )


# ============================================================================
# Differentiable Kepler solver (autodiff.py)
# ============================================================================


class TestKepler:
    """Tests for differentiable Kepler trig evaluation."""

    def test_matches_halley(self):
        """Fails if default trig evaluation disagrees with independent Halley.

        Absolute 1e-12 and relative 1e-5 tolerances retain the original
        solver parity floor.
        """
        M = jnp.linspace(0.01, 2 * jnp.pi - 0.01, 100)
        e = 0.5

        sinE_ref, cosE_ref = kepler(M, e, 0.0, solver=Halley())
        sinE, cosE = kepler(M, e, 0.0)

        assert jnp.allclose(sinE_ref, sinE, atol=1e-12)
        assert jnp.allclose(cosE_ref, cosE, atol=1e-12)

    def test_grad_finite(self):
        """Gradients w.r.t. M and e should be finite."""

        def loss(M, e):
            sinE, cosE = kepler(M, e, 0.0)
            return jnp.sum(sinE**2 + cosE**2)

        M = jnp.linspace(0.1, 6.0, 50)
        grad_M, grad_e = jax.grad(loss, argnums=(0, 1))(M, 0.5)

        assert jnp.all(jnp.isfinite(grad_M))
        assert jnp.isfinite(grad_e)

    def test_grad_circular_orbit(self):
        """At e=0: E=M, so d(sinE)/dM = cosM and d(cosE)/dM = -sinM."""

        def sinE_sum(M, e):
            sinE, _ = kepler(M, e, 0.0)
            return jnp.sum(sinE)

        def cosE_sum(M, e):
            _, cosE = kepler(M, e, 0.0)
            return jnp.sum(cosE)

        M = jnp.linspace(0.1, 6.0, 20)
        e = 0.0

        dsinE_dM = jax.grad(sinE_sum)(M, e)
        dcosE_dM = jax.grad(cosE_sum)(M, e)

        # At e=0: sinE=sinM, cosE=cosM
        # d(sum sinM_i)/dM_j = cosM_j (diagonal Jacobian -> gradient is cosM)
        assert jnp.allclose(dsinE_dM, jnp.cos(M), atol=1e-6)
        assert jnp.allclose(dcosE_dM, -jnp.sin(M), atol=1e-6)

    def test_jit(self):
        """Should be JIT-compilable."""
        f = jax.jit(kepler)
        M = jnp.array([0.5, 1.0, 2.0])
        sinE, _cosE = f(M, 0.3, 0.0)
        assert sinE.shape == (3,)
        assert jnp.all(jnp.isfinite(sinE))

    def test_grad_various_eccentricities(self):
        """Gradients should be finite across a range of eccentricities."""
        M = jnp.array([1.0, 2.0, 3.0])

        for e_val in [0.0, 0.1, 0.5, 0.8, 0.95]:

            def loss(e):
                sinE, cosE = kepler(M, e, 0.0)
                return jnp.sum(sinE + cosE)

            g = jax.grad(loss)(e_val)
            assert jnp.isfinite(g), f"Gradient not finite at e={e_val}"


# ============================================================================
# Data containers (data.py)
# ============================================================================


class TestDataContainers:
    """Tests for data container construction and properties."""

    def test_rvdata_construction(self):
        """RadialVelocityData should store fields correctly."""
        data = RadialVelocityData(
            time_d=jnp.array([0.0, 1.0, 2.0]),
            rv=jnp.array([1.0, 2.0, 3.0]),
            rv_err=jnp.array([0.1, 0.1, 0.1]),
            inst_ids=jnp.array([0, 0, 1]),
            valid=jnp.ones(3, dtype=bool),
            instruments=("a", "b"),
        )
        assert data.time_d.shape == (3,)
        assert data.n_inst == 2


# ============================================================================
# One-orbit kinematics (tests/kinematics.py)
# ============================================================================


class TestPredictRV:
    """Tests for the RV forward model."""

    def test_face_on_returns_zero(self):
        """Face-on orbit (i=0, Mp_sini=0) should produce zero RV signal."""
        times = jnp.linspace(0, 365.25, 100)
        rv = predict_rv(
            times,
            T=365.25,
            Ms=Msun2kg,
            Mp_sini=0.0,  # sin(0) = 0
            e=0.0,
            cos_w=1.0,
            sin_w=0.0,
            tp=0.0,
        )
        assert jnp.allclose(rv, 0.0, atol=1e-30)

    def test_circular_sinusoidal(self, circular_orbit):
        """Circular edge-on orbit should produce a sinusoidal RV."""
        times = jnp.linspace(0, 365.25, 200)
        rv = predict_rv(
            times,
            T=circular_orbit["T"],
            Ms=circular_orbit["Ms"],
            Mp_sini=circular_orbit["Mp_sini"],
            e=circular_orbit["e"],
            cos_w=circular_orbit["cos_w"],
            sin_w=circular_orbit["sin_w"],
            tp=circular_orbit["tp"],
        )
        # Should be periodic with the right period
        assert rv.shape == (200,)
        # Peak-to-peak amplitude should be consistent (symmetric for e=0)
        assert jnp.isclose(jnp.max(rv), -jnp.min(rv), rtol=0.01)

    def test_rv_differentiable(self, eccentric_orbit):
        """RV model should be differentiable w.r.t. orbital parameters."""
        times = jnp.array([10.0, 50.0, 100.0])

        def loss(e, tp):
            rv = predict_rv(
                times,
                T=eccentric_orbit["T"],
                Ms=eccentric_orbit["Ms"],
                Mp_sini=eccentric_orbit["Mp_sini"],
                e=e,
                cos_w=eccentric_orbit["cos_w"],
                sin_w=eccentric_orbit["sin_w"],
                tp=tp,
            )
            return jnp.sum(rv**2)

        g_e, g_tp = jax.grad(loss, argnums=(0, 1))(
            eccentric_orbit["e"], eccentric_orbit["tp"]
        )
        assert jnp.isfinite(g_e)
        assert jnp.isfinite(g_tp)

    def test_rv_jit(self, circular_orbit):
        """RV forward model should JIT-compile."""
        f = jax.jit(
            lambda t: predict_rv(
                t,
                T=circular_orbit["T"],
                Ms=circular_orbit["Ms"],
                Mp_sini=circular_orbit["Mp_sini"],
                e=circular_orbit["e"],
                cos_w=circular_orbit["cos_w"],
                sin_w=circular_orbit["sin_w"],
                tp=circular_orbit["tp"],
            )
        )
        rv = f(jnp.array([0.0, 100.0]))
        assert jnp.all(jnp.isfinite(rv))


class TestPredictAstrometry:
    """Tests for the astrometry forward model."""

    def test_output_shape(self, eccentric_orbit):
        """Should return (N,) shaped RA and DEC arrays."""
        times = jnp.linspace(0, 200.0, 50)
        ra, dec = predict_relative_astrometry(
            times,
            a=eccentric_orbit["a"],
            e=eccentric_orbit["e"],
            cos_i=eccentric_orbit["cos_i"],
            W=eccentric_orbit["W"],
            cos_w=eccentric_orbit["cos_w"],
            sin_w=eccentric_orbit["sin_w"],
            tp=eccentric_orbit["tp"],
            Ms=eccentric_orbit["Ms"],
            dist_pc=eccentric_orbit["dist_pc"],
        )
        assert ra.shape == (50,)
        assert dec.shape == (50,)
        assert jnp.all(jnp.isfinite(ra))
        assert jnp.all(jnp.isfinite(dec))

    def test_physical_scale(self, circular_orbit):
        """1 AU at 10 pc should give ~0.1 arcsec max separation."""
        times = jnp.linspace(0, 365.25, 100)
        ra, dec = predict_relative_astrometry(
            times,
            a=circular_orbit["a"],
            e=circular_orbit["e"],
            cos_i=circular_orbit["cos_i"],
            W=circular_orbit["W"],
            cos_w=circular_orbit["cos_w"],
            sin_w=circular_orbit["sin_w"],
            tp=circular_orbit["tp"],
            Ms=circular_orbit["Ms"],
            dist_pc=circular_orbit["dist_pc"],
        )
        max_sep = jnp.max(jnp.sqrt(ra**2 + dec**2))
        # 1 AU at 10 pc = 0.1 arcsec
        assert jnp.isclose(max_sep, 0.1, rtol=0.05)

    def test_differentiable(self, eccentric_orbit):
        """Astrometry should be differentiable w.r.t. semi-major axis."""
        times = jnp.array([10.0, 50.0])

        def loss(a):
            ra, dec = predict_relative_astrometry(
                times,
                a=a,
                e=eccentric_orbit["e"],
                cos_i=eccentric_orbit["cos_i"],
                W=eccentric_orbit["W"],
                cos_w=eccentric_orbit["cos_w"],
                sin_w=eccentric_orbit["sin_w"],
                tp=eccentric_orbit["tp"],
                Ms=eccentric_orbit["Ms"],
                dist_pc=eccentric_orbit["dist_pc"],
            )
            return jnp.sum(ra**2 + dec**2)

        g = jax.grad(loss)(eccentric_orbit["a"])
        assert jnp.isfinite(g)


class TestPredictPhotometry:
    """Tests for the photometry forward model."""

    def test_output_shape(self, eccentric_orbit):
        """Should return (N,) shaped alpha and dMag arrays."""
        times = jnp.linspace(0, 200.0, 30)
        # Lambda = Ag * Rp^2 ~= 0.3 * (1 Rearth)^2 in AU^2

        Lambda = 0.3 * Rearth2AU**2
        alpha, dMag = predict_photometry(
            times,
            a=eccentric_orbit["a"],
            e=eccentric_orbit["e"],
            cos_i=eccentric_orbit["cos_i"],
            W=eccentric_orbit["W"],
            cos_w=eccentric_orbit["cos_w"],
            sin_w=eccentric_orbit["sin_w"],
            tp=eccentric_orbit["tp"],
            Ms=eccentric_orbit["Ms"],
            Lambda=Lambda,
            dist_pc=eccentric_orbit["dist_pc"],
        )
        assert alpha.shape == (30,)
        assert dMag.shape == (30,)
        assert jnp.all(jnp.isfinite(alpha))
        assert jnp.all(jnp.isfinite(dMag))

    def test_dMag_positive(self, circular_orbit):
        """DMag should be positive (planet is always fainter than star)."""
        times = jnp.linspace(0, 365.25, 50)

        Lambda = 0.3 * Rearth2AU**2
        _, dMag = predict_photometry(
            times,
            a=circular_orbit["a"],
            e=circular_orbit["e"],
            cos_i=circular_orbit["cos_i"],
            W=circular_orbit["W"],
            cos_w=circular_orbit["cos_w"],
            sin_w=circular_orbit["sin_w"],
            tp=circular_orbit["tp"],
            Ms=circular_orbit["Ms"],
            Lambda=Lambda,
            dist_pc=circular_orbit["dist_pc"],
        )
        assert jnp.all(dMag > 0)


# ============================================================================
# Likelihoods (likelihoods.py)
# ============================================================================


class TestLoglikeRV:
    """Tests for the RV marginalized likelihood."""

    def test_returns_finite(self):
        """Should return a finite scalar."""
        rv_obs = jnp.array([1.0, 2.0, 3.0]) * 1e-10
        rv_model = jnp.array([1.1, 1.9, 3.1]) * 1e-10
        rv_err = jnp.ones(3) * 0.5e-10
        inst_ids = jnp.array([0, 0, 0])
        jitters = jnp.array([0.1e-10])

        ll = loglike_rv_marginalized(
            rv_obs, rv_model, rv_err, inst_ids, 1, jitters, jnp.ones(3, dtype=bool)
        )
        assert jnp.isfinite(ll)
        assert ll.shape == ()

    def test_offset_invariance(self):
        """Adding a constant offset to rv_model shouldn't change loglike.

        This is the key property of the gamma marginalization: the zero-point
        cancels for a single instrument.
        """
        rv_obs = jnp.array([1.0, 2.0, 3.0, 4.0]) * 1e-10
        rv_model = jnp.array([0.9, 2.1, 2.8, 4.2]) * 1e-10
        rv_err = jnp.ones(4) * 0.5e-10
        inst_ids = jnp.zeros(4, dtype=jnp.int32)
        jitters = jnp.array([0.1e-10])

        ll1 = loglike_rv_marginalized(
            rv_obs, rv_model, rv_err, inst_ids, 1, jitters, jnp.ones(4, dtype=bool)
        )

        # Shift model by a constant
        offset = 5.0e-10
        ll2 = loglike_rv_marginalized(
            rv_obs,
            rv_model + offset,
            rv_err,
            inst_ids,
            1,
            jitters,
            jnp.ones(4, dtype=bool),
        )

        assert jnp.isclose(ll1, ll2, rtol=1e-10)

    def test_differentiable(self):
        """Should be differentiable w.r.t. rv_model."""
        rv_obs = jnp.array([1.0, 2.0]) * 1e-10
        rv_err = jnp.ones(2) * 0.5e-10
        inst_ids = jnp.zeros(2, dtype=jnp.int32)
        jitters = jnp.array([0.1e-10])

        def loss(rv_model):
            return loglike_rv_marginalized(
                rv_obs,
                rv_model,
                rv_err,
                inst_ids,
                1,
                jitters,
                jnp.ones(2, dtype=bool),
            )

        g = jax.grad(loss)(jnp.array([1.1, 1.9]) * 1e-10)
        assert jnp.all(jnp.isfinite(g))


class TestLoglikeAstrom:
    """Tests for the astrometry likelihood."""

    def test_perfect_match(self):
        """Zero residuals should give the maximum likelihood."""
        data = AstrometryData(
            time_d=jnp.array([0.0]),
            east_arcsec=jnp.array([0.1]),
            north_arcsec=jnp.array([0.2]),
            east_err_arcsec=jnp.array([0.01]),
            north_err_arcsec=jnp.array([0.01]),
            corr=jnp.array([0.0]),
            planet=jnp.array([0]),
            valid=jnp.ones(1, dtype=bool),
        )
        ll = loglike_relative_astrom(jnp.array([0.1]), jnp.array([0.2]), data)
        assert jnp.isfinite(ll)

    def test_worse_with_residual(self):
        """Larger residuals should give lower likelihood."""
        data = AstrometryData(
            time_d=jnp.array([0.0]),
            east_arcsec=jnp.array([0.1]),
            north_arcsec=jnp.array([0.2]),
            east_err_arcsec=jnp.array([0.01]),
            north_err_arcsec=jnp.array([0.01]),
            corr=jnp.array([0.0]),
            planet=jnp.array([0]),
            valid=jnp.ones(1, dtype=bool),
        )
        ll_good = loglike_relative_astrom(jnp.array([0.1]), jnp.array([0.2]), data)
        ll_bad = loglike_relative_astrom(jnp.array([0.15]), jnp.array([0.25]), data)
        assert ll_good > ll_bad

    def test_uncorrelated_reduces_to_chi2(self):
        """With rho=0, should reduce to independent Gaussian in RA and DEC."""
        data = AstrometryData(
            time_d=jnp.array([0.0]),
            east_arcsec=jnp.array([1.0]),
            north_arcsec=jnp.array([2.0]),
            east_err_arcsec=jnp.array([0.5]),
            north_err_arcsec=jnp.array([0.5]),
            corr=jnp.array([0.0]),
            planet=jnp.array([0]),
            valid=jnp.ones(1, dtype=bool),
        )
        ra_pred = jnp.array([1.1])
        dec_pred = jnp.array([2.1])

        ll = loglike_relative_astrom(ra_pred, dec_pred, data)

        # Manual chi^2: ((1-1.1)/0.5)^2 + ((2-2.1)/0.5)^2 = 0.04 + 0.04 = 0.08
        chi2 = (1.0 - 1.1) ** 2 / 0.5**2 + (2.0 - 2.1) ** 2 / 0.5**2
        log_norm = jnp.log(0.5 * 0.5 * 1.0) + jnp.log(2 * jnp.pi)
        expected = -0.5 * chi2 - log_norm
        assert jnp.isclose(ll, expected, atol=1e-10)

    def test_differentiable(self):
        """Should be differentiable w.r.t. predictions."""
        data = AstrometryData(
            time_d=jnp.array([0.0]),
            east_arcsec=jnp.array([0.1]),
            north_arcsec=jnp.array([0.2]),
            east_err_arcsec=jnp.array([0.01]),
            north_err_arcsec=jnp.array([0.01]),
            corr=jnp.array([0.0]),
            planet=jnp.array([0]),
            valid=jnp.ones(1, dtype=bool),
        )

        def loss(ra_pred):
            return loglike_relative_astrom(ra_pred, jnp.array([0.2]), data)

        g = jax.grad(loss)(jnp.array([0.1]))
        assert jnp.all(jnp.isfinite(g))


# ============================================================================
# Prior helpers (priors.py)
# ============================================================================


class TestPriors:
    """Tests for prior distribution helpers."""

    def test_period_to_sma_earth(self):
        """Earth's period should give ~1 AU."""
        from orbix.equations import period_to_sma

        a = period_to_sma(365.25, Msun2kg)
        assert jnp.isclose(a, 1.0, rtol=0.01)

    def test_period_to_sma_differentiable(self):
        """Kepler's 3rd law should be differentiable."""
        from orbix.equations import period_to_sma

        g = jax.grad(period_to_sma)(365.25, Msun2kg)
        assert jnp.isfinite(g)
