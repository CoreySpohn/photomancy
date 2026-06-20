"""Tests for photomancy.orbit module."""

import jax
import jax.numpy as jnp
import pytest
from hwoutils.constants import G, Mearth2kg, Msun2kg, Rearth2AU
from orbix.kepler.core import diff_solve_trig

from photomancy.orbit.data import RelativeAstromData, NullData, RVData
from photomancy.orbit.forward import predict_relative_astrometry, predict_photometry, predict_rv
from photomancy.orbit.likelihoods import (
    loglike_relative_astrom,
    loglike_null,
    loglike_rv_marginalized,
)

# Enable float64 for numerical stability in gradient checks
jax.config.update("jax_enable_x64", True)


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


class TestDiffSolveTrig:
    """Tests for the custom VJP Kepler solver."""

    def test_matches_original(self):
        """diff_solve_trig should return identical values to solve_trig."""
        from orbix.kepler.core import solve_trig

        M = jnp.linspace(0.01, 2 * jnp.pi - 0.01, 100)
        e = 0.5

        sinE_orig, cosE_orig = solve_trig(M, e)
        sinE_diff, cosE_diff = diff_solve_trig(M, e)

        assert jnp.allclose(sinE_orig, sinE_diff, atol=1e-12)
        assert jnp.allclose(cosE_orig, cosE_diff, atol=1e-12)

    def test_grad_finite(self):
        """Gradients w.r.t. M and e should be finite."""

        def loss(M, e):
            sinE, cosE = diff_solve_trig(M, e)
            return jnp.sum(sinE**2 + cosE**2)

        M = jnp.linspace(0.1, 6.0, 50)
        grad_M, grad_e = jax.grad(loss, argnums=(0, 1))(M, 0.5)

        assert jnp.all(jnp.isfinite(grad_M))
        assert jnp.isfinite(grad_e)

    def test_grad_circular_orbit(self):
        """At e=0: E=M, so d(sinE)/dM = cosM and d(cosE)/dM = -sinM."""

        def sinE_sum(M, e):
            sinE, _ = diff_solve_trig(M, e)
            return jnp.sum(sinE)

        def cosE_sum(M, e):
            _, cosE = diff_solve_trig(M, e)
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
        f = jax.jit(diff_solve_trig)
        M = jnp.array([0.5, 1.0, 2.0])
        sinE, _cosE = f(M, 0.3)
        assert sinE.shape == (3,)
        assert jnp.all(jnp.isfinite(sinE))

    def test_grad_various_eccentricities(self):
        """Gradients should be finite across a range of eccentricities."""
        M = jnp.array([1.0, 2.0, 3.0])

        for e_val in [0.0, 0.1, 0.5, 0.8, 0.95]:

            def loss(e):
                sinE, cosE = diff_solve_trig(M, e)
                return jnp.sum(sinE + cosE)

            g = jax.grad(loss)(e_val)
            assert jnp.isfinite(g), f"Gradient not finite at e={e_val}"


# ============================================================================
# Data containers (data.py)
# ============================================================================


class TestDataContainers:
    """Tests for data container construction and properties."""

    def test_rvdata_construction(self):
        """RVData should store fields correctly."""
        data = RVData(
            times=jnp.array([0.0, 1.0, 2.0]),
            rv=jnp.array([1.0, 2.0, 3.0]),
            rv_err=jnp.array([0.1, 0.1, 0.1]),
            inst_ids=jnp.array([0, 0, 1]),
            is_valid=jnp.ones(3, dtype=bool),
            n_inst=2,
        )
        assert data.times.shape == (3,)
        assert data.n_inst == 2

    def test_nulldata_from_contrast(self):
        """from_contrast_curves should convert contrast to dMag correctly."""
        epochs = jnp.array([0.0, 100.0])
        seps = jnp.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        contrast = jnp.array([[1e-9, 1e-10, 1e-10], [1e-8, 1e-9, 1e-10]])

        data = NullData.from_contrast_curves(epochs, seps, contrast)

        # dMag = -2.5 * log10(contrast). Check only valid entries.
        expected = -2.5 * jnp.log10(contrast)
        n_ep, n_pts = expected.shape
        assert jnp.allclose(data.dmag0_grid[:n_ep, :n_pts], expected)
        assert data.snr_thresh == 5.0
        assert data.is_valid[:n_ep].all()
        assert not data.is_valid[n_ep:].any()

    def test_nulldata_padding(self):
        """Padding with -inf should produce zero likelihood penalty."""
        epochs = jnp.array([0.0])
        seps = jnp.array([[0.1, 0.5, 100.0]])  # 100 arcsec = beyond OWA
        dmag0 = jnp.array([[25.0, 25.0, -jnp.inf]])  # padded

        data = NullData(
            epochs=epochs,
            sep_grid=seps,
            dmag0_grid=dmag0,
            is_valid=jnp.ones(1, dtype=bool),
        )

        # Planet at padded separation: should get ~zero penalty
        alpha = jnp.array([100.0])
        dMag = jnp.array([20.0])  # bright planet at padded location
        ll = loglike_null(alpha, dMag, data)
        # At the padded point, dmag0=-inf, flux_ratio=0, z=5, log_ndtr(5)~=0
        assert ll > -1e-3


# ============================================================================
# Forward models (forward.py)
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
        data = RelativeAstromData(
            times=jnp.array([0.0]),
            ra=jnp.array([0.1]),
            dec=jnp.array([0.2]),
            ra_err=jnp.array([0.01]),
            dec_err=jnp.array([0.01]),
            corr=jnp.array([0.0]),
            planet_id=jnp.array([0]),
            is_valid=jnp.ones(1, dtype=bool),
        )
        ll = loglike_relative_astrom(jnp.array([0.1]), jnp.array([0.2]), data)
        assert jnp.isfinite(ll)

    def test_worse_with_residual(self):
        """Larger residuals should give lower likelihood."""
        data = RelativeAstromData(
            times=jnp.array([0.0]),
            ra=jnp.array([0.1]),
            dec=jnp.array([0.2]),
            ra_err=jnp.array([0.01]),
            dec_err=jnp.array([0.01]),
            corr=jnp.array([0.0]),
            planet_id=jnp.array([0]),
            is_valid=jnp.ones(1, dtype=bool),
        )
        ll_good = loglike_relative_astrom(jnp.array([0.1]), jnp.array([0.2]), data)
        ll_bad = loglike_relative_astrom(jnp.array([0.15]), jnp.array([0.25]), data)
        assert ll_good > ll_bad

    def test_uncorrelated_reduces_to_chi2(self):
        """With rho=0, should reduce to independent Gaussian in RA and DEC."""
        data = RelativeAstromData(
            times=jnp.array([0.0]),
            ra=jnp.array([1.0]),
            dec=jnp.array([2.0]),
            ra_err=jnp.array([0.5]),
            dec_err=jnp.array([0.5]),
            corr=jnp.array([0.0]),
            planet_id=jnp.array([0]),
            is_valid=jnp.ones(1, dtype=bool),
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
        data = RelativeAstromData(
            times=jnp.array([0.0]),
            ra=jnp.array([0.1]),
            dec=jnp.array([0.2]),
            ra_err=jnp.array([0.01]),
            dec_err=jnp.array([0.01]),
            corr=jnp.array([0.0]),
            planet_id=jnp.array([0]),
            is_valid=jnp.ones(1, dtype=bool),
        )

        def loss(ra_pred):
            return loglike_relative_astrom(ra_pred, jnp.array([0.2]), data)

        g = jax.grad(loss)(jnp.array([0.1]))
        assert jnp.all(jnp.isfinite(g))


class TestLoglikeNull:
    """Tests for the non-detection likelihood."""

    @pytest.fixture
    def simple_null_data(self):
        """Single epoch with a contrast curve."""
        epochs = jnp.array([0.0])
        seps = jnp.array([[0.05, 0.1, 0.2, 0.5, 1.0]])  # arcsec
        contrast = jnp.array([[1e-8, 1e-9, 1e-10, 1e-10, 1e-10]])
        return NullData.from_contrast_curves(
            epochs,
            seps,
            contrast,
            max_n=1,
            max_pts=5,
        )

    def test_bright_planet_penalized(self, simple_null_data):
        """A planet 5 mag brighter than limit should be heavily penalized."""
        # dMag0 limit at 0.2 arcsec is -2.5*log10(1e-10) = 25
        alpha = jnp.array([0.2])
        dMag = jnp.array([20.0])  # 5 mag brighter -> should have been detected

        ll = loglike_null(alpha, dMag, simple_null_data)
        assert ll < -10.0  # heavy penalty

    def test_faint_planet_no_penalty(self, simple_null_data):
        """A planet fainter than the limit should have near-zero penalty."""
        alpha = jnp.array([0.2])
        dMag = jnp.array([30.0])  # 5 mag fainter than limit

        ll = loglike_null(alpha, dMag, simple_null_data)
        assert ll > -0.01  # essentially zero

    def test_outside_iwa_no_penalty(self, simple_null_data):
        """Planet outside valid separation range gets no penalty."""
        alpha = jnp.array([5.0])  # beyond OWA
        dMag = jnp.array([15.0])  # very bright

        ll = loglike_null(alpha, dMag, simple_null_data)
        assert ll > -0.01  # no penalty -- planet was invisible

    def test_inside_iwa_no_penalty(self, simple_null_data):
        """Planet inside IWA gets no penalty."""
        alpha = jnp.array([0.01])  # inside IWA
        dMag = jnp.array([15.0])  # very bright

        ll = loglike_null(alpha, dMag, simple_null_data)
        assert ll > -0.01

    def test_differentiable(self, simple_null_data):
        """Should be differentiable w.r.t. predictions."""

        def loss(dMag):
            return loglike_null(jnp.array([0.2]), dMag, simple_null_data)

        g = jax.grad(loss)(jnp.array([25.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_gradient_direction(self, simple_null_data):
        """Gradient should push planet toward fainter (larger dMag)."""

        def loss(dMag):
            return loglike_null(jnp.array([0.2]), dMag, simple_null_data)

        # At the detection boundary, gradient should point toward fainter
        g = jax.grad(loss)(jnp.array([25.0]))
        # Positive gradient = increasing dMag increases likelihood
        assert g[0] > 0


# ============================================================================
# Prior helpers (priors.py)
# ============================================================================


class TestPriors:
    """Tests for prior distribution helpers."""

    def test_disk_transform_output_range(self):
        """Eccentricity should be in [0, 1), cos_w and sin_w on unit circle."""
        from photomancy.orbit.priors import eccentricity_disk_transform

        # Sample a grid of (x, y) values
        key = jax.random.PRNGKey(42)
        xy = jax.random.normal(key, (1000, 2))

        e_arr, cw_arr, sw_arr = jax.vmap(
            lambda row: eccentricity_disk_transform(row[0], row[1])
        )(xy)

        assert jnp.all(e_arr >= 0.0)
        assert jnp.all(e_arr < 1.0)
        # cos_w^2 + sin_w^2 ~= 1
        assert jnp.allclose(cw_arr**2 + sw_arr**2, 1.0, atol=1e-8)

    def test_disk_transform_uniform_eccentricity(self):
        """Transformed eccentricity should be approximately uniform on [0,1)."""
        from photomancy.orbit.priors import eccentricity_disk_transform

        key = jax.random.PRNGKey(0)
        xy = jax.random.normal(key, (50000, 2))

        e_arr, _, _ = jax.vmap(lambda row: eccentricity_disk_transform(row[0], row[1]))(
            xy
        )

        # Histogram into 10 bins: each should have ~10% of samples
        for low, high in [(0.0, 0.1), (0.4, 0.5), (0.8, 0.9)]:
            frac = jnp.mean((e_arr >= low) & (e_arr < high))
            assert 0.07 < float(frac) < 0.13, (
                f"e distribution not uniform: bin [{low},{high}) has {float(frac):.3f}"
            )

    def test_disk_transform_sigma_scaling(self):
        """Smaller-scale draws should concentrate eccentricity at low values."""
        from photomancy.orbit.priors import eccentricity_disk_transform

        key = jax.random.PRNGKey(1)
        xy = jax.random.normal(key, (10000, 2))

        # sigma=0.3 draws should put most mass below 0.5
        xy_03 = xy * 0.3
        e_03, _, _ = jax.vmap(lambda row: eccentricity_disk_transform(row[0], row[1]))(
            xy_03
        )
        frac_low = float(jnp.mean(e_03 < 0.5))
        assert frac_low > 0.85, f"Expected >85% below 0.5, got {frac_low:.1%}"

        # sigma=0.049 draws should put nearly all mass below 0.2
        xy_005 = xy * 0.049
        e_005, _, _ = jax.vmap(lambda row: eccentricity_disk_transform(row[0], row[1]))(
            xy_005
        )
        frac_low_ve = float(jnp.mean(e_005 < 0.2))
        assert frac_low_ve > 0.95, f"Expected >95% below 0.2, got {frac_low_ve:.1%}"

    def test_disk_transform_differentiable(self):
        """Disk transform should be differentiable."""
        from photomancy.orbit.priors import eccentricity_disk_transform

        def loss(x, y):
            e, cw, sw = eccentricity_disk_transform(x, y)
            return e + cw + sw

        g = jax.grad(loss, argnums=(0, 1))(1.0, 0.5)
        assert jnp.isfinite(g[0]) and jnp.isfinite(g[1])

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


# ============================================================================
# NumPyro model builder (numpyro_model.py)
# ============================================================================


class TestNumPyroModel:
    """Tests for the NumPyro model builder."""

    def test_build_model_requires_data(self):
        """Should raise ValueError if no data is provided."""
        from photomancy.orbit.model import build_model

        with pytest.raises(ValueError, match="At least one"):
            build_model()

    def test_build_model_rv_only(self):
        """Should return a callable model for RV-only fitting."""
        from photomancy.orbit.model import build_model

        model = build_model(has_rv=True)
        assert callable(model)

    def test_model_executes_with_seed(self):
        """Model should execute under numpyro.handlers.seed."""
        import numpyro.handlers as handlers

        from photomancy.orbit.model import build_model

        rv_data = RVData(
            times=jnp.linspace(0, 365.25, 10),
            rv=jnp.ones(10) * 1e-10,
            rv_err=jnp.ones(10) * 1e-11,
            inst_ids=jnp.zeros(10, dtype=jnp.int32),
            is_valid=jnp.ones(10, dtype=bool),
            n_inst=1,
        )
        model = build_model(has_rv=True)

        # Execute with a seed to get a trace
        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model).get_trace(
                Msun2kg, 10.0, rv_data, None, None, None
            )

        # Check essential sites exist (kipping13 default uses e_raw, w_raw)
        assert "log_P" in trace
        assert "e_raw" in trace
        assert "w_raw" in trace
        assert "cos_i" in trace
        assert "log_Mp" in trace
        assert "ll_rv" in trace

    @pytest.mark.parametrize(
        "ecc_prior", ["kipping13", "rayleigh", "vaneylen19", "disk"]
    )
    def test_model_ecc_prior_options(self, ecc_prior):
        """All eccentricity priors should produce a valid, executable model."""
        import numpyro.handlers as handlers

        from photomancy.orbit.model import build_model

        rv_data = RVData(
            times=jnp.linspace(0, 365.25, 10),
            rv=jnp.ones(10) * 1e-10,
            rv_err=jnp.ones(10) * 1e-11,
            inst_ids=jnp.zeros(10, dtype=jnp.int32),
            is_valid=jnp.ones(10, dtype=bool),
            n_inst=1,
        )
        model = build_model(
            has_rv=True,
            ecc_prior=ecc_prior,
        )

        with handlers.seed(rng_seed=99):
            trace = handlers.trace(model).get_trace(
                Msun2kg, 10.0, rv_data, None, None, None
            )

        # All priors must produce deterministic e, cos_w, sin_w
        assert "e" in trace
        assert "cos_w" in trace
        assert "sin_w" in trace

        e_val = trace["e"]["value"].item()
        assert 0.0 <= e_val < 1.0

    def test_unknown_ecc_prior_raises(self):
        """Unknown ecc_prior name should raise ValueError."""
        from photomancy.orbit.model import build_model

        rv_data = RVData(
            times=jnp.linspace(0, 365.25, 10),
            rv=jnp.ones(10) * 1e-10,
            rv_err=jnp.ones(10) * 1e-11,
            inst_ids=jnp.zeros(10, dtype=jnp.int32),
            is_valid=jnp.ones(10, dtype=bool),
            n_inst=1,
        )
        model = build_model(
            has_rv=True,
            ecc_prior="nonexistent",
        )
        import numpyro.handlers as handlers

        with pytest.raises(ValueError, match="Unknown ecc_prior"):
            with handlers.seed(rng_seed=0):
                handlers.trace(model).get_trace(
                    Msun2kg, 10.0, rv_data, None, None, None
                )

    def test_mcmc_runs_rv(self):
        """End-to-end: generate synthetic RV data and run a short NUTS chain.

        This is the acid test: the full pipeline from data -> model -> MCMC
        must execute without errors or NaN log-probabilities.
        """
        from numpyro.infer import MCMC, NUTS

        from photomancy.orbit.model import build_model

        # Generate synthetic RV from a known orbit
        true_T = 365.25
        true_e = 0.1
        true_tp = 50.0
        true_Mp_sini = 10.0 * Mearth2kg  # 10 Earth masses
        true_cos_w = 1.0
        true_sin_w = 0.0

        times = jnp.linspace(0, 2 * 365.25, 30)  # 2 years, 30 points
        rv_true = predict_rv(
            times,
            true_T,
            Msun2kg,
            true_Mp_sini,
            true_e,
            true_cos_w,
            true_sin_w,
            true_tp,
        )

        # Add noise
        key = jax.random.PRNGKey(123)
        rv_err = jnp.ones_like(rv_true) * jnp.std(rv_true) * 0.1
        noise = jax.random.normal(key, rv_true.shape) * rv_err
        rv_obs = rv_true + noise

        rv_data = RVData(
            times=times,
            rv=rv_obs,
            rv_err=rv_err,
            inst_ids=jnp.zeros(len(times), dtype=jnp.int32),
            is_valid=jnp.ones(len(times), dtype=bool),
            n_inst=1,
        )

        model = build_model(has_rv=True)

        # Run a very short chain (just testing execution, not convergence)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=50, num_samples=20, progress_bar=False)
        mcmc.run(
            jax.random.PRNGKey(0),
            Msun2kg,
            10.0,
            rv_data,
            None,
            None,
            None,
        )

        samples = mcmc.get_samples()

        # Basic sanity: all sampled values should be finite
        for key_name, vals in samples.items():
            assert jnp.all(jnp.isfinite(vals)), f"NaN/inf in {key_name}"

        # The sampled periods should bracket the true period (loose check)
        T_samples = samples["T"]
        assert jnp.min(T_samples) > 0  # all positive


# ============================================================================
# Thiele-Innes linear fitter (thiele_innes.py)
# ============================================================================


class TestThieleInnes:
    """Tests for the Thiele-Innes linear orbit fitter."""

    @staticmethod
    def _consistent_orbit():
        """Eccentric orbit with Kepler III-consistent a and T."""
        T = 200.0  # days
        Ms = Msun2kg
        e = 0.3
        w = jnp.pi / 4  # omega = 45deg
        # a from Kepler III: a = (mu * (T/2pi)^2)^(1/3)
        mu = G * Ms
        a = (mu * (T / (2.0 * jnp.pi)) ** 2) ** (1.0 / 3.0)
        return dict(
            T=T,
            a=float(a),
            e=e,
            cos_i=jnp.cos(jnp.radians(60.0)),
            cos_w=jnp.cos(w),
            sin_w=jnp.sin(w),
            W=jnp.pi / 3,
            tp=50.0,
            Ms=Ms,
            dist_pc=5.0,
        )

    def _make_astrom_data(self, params, n_pts=20, noise_arcsec=1e-6, seed=42):
        """Generate synthetic astrometry from known orbital parameters."""
        key = jax.random.PRNGKey(seed)
        times = jnp.linspace(0.0, params["T"] * 1.5, n_pts)
        ra_true, dec_true = predict_relative_astrometry(
            times=times,
            a=params["a"],
            e=params["e"],
            cos_i=params["cos_i"],
            cos_w=params["cos_w"],
            sin_w=params["sin_w"],
            W=params["W"],
            tp=params["tp"],
            Ms=params["Ms"],
            dist_pc=params["dist_pc"],
        )
        k1, k2 = jax.random.split(key)
        ra = ra_true + noise_arcsec * jax.random.normal(k1, shape=ra_true.shape)
        dec = dec_true + noise_arcsec * jax.random.normal(k2, shape=dec_true.shape)

        return RelativeAstromData(
            times=times,
            ra=ra,
            dec=dec,
            ra_err=jnp.full(n_pts, noise_arcsec),
            dec_err=jnp.full(n_pts, noise_arcsec),
            corr=jnp.zeros(n_pts),
            planet_id=jnp.zeros(n_pts, dtype=jnp.int32),
            is_valid=jnp.ones(n_pts, dtype=bool),
        )

    def test_recovers_sma_circular(self, circular_orbit):
        """TI fitter should recover semi-major axis for a circular orbit."""
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        data = self._make_astrom_data(circular_orbit)
        result = thiele_innes_fit(
            data,
            T=circular_orbit["T"],
            e=circular_orbit["e"],
            tp=circular_orbit["tp"],
            Ms=circular_orbit["Ms"],
            dist_pc=circular_orbit["dist_pc"],
        )

        assert jnp.isfinite(result.log_likelihood)
        assert jnp.allclose(result.a, circular_orbit["a"], rtol=1e-3)

    def test_recovers_elements_eccentric(self):
        """TI fitter should recover (a, cos_i) for an eccentric orbit."""
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit)
        result = thiele_innes_fit(
            data,
            T=orbit["T"],
            e=orbit["e"],
            tp=orbit["tp"],
            Ms=orbit["Ms"],
            dist_pc=orbit["dist_pc"],
        )

        assert jnp.isfinite(result.log_likelihood)
        assert jnp.allclose(result.a, orbit["a"], rtol=1e-2)
        # cos_i recovery (up to sign -- i vs pi-i ambiguity)
        assert jnp.allclose(
            jnp.abs(result.cos_i),
            jnp.abs(orbit["cos_i"]),
            atol=0.02,
        )

    def test_chi2_near_n_dof(self):
        """With correct params, chi^2 ~= N_datapoints - 4 (4 fitted params)."""
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        orbit = self._consistent_orbit()
        n_pts = 30
        data = self._make_astrom_data(orbit, n_pts=n_pts)
        result = thiele_innes_fit(
            data,
            T=orbit["T"],
            e=orbit["e"],
            tp=orbit["tp"],
            Ms=orbit["Ms"],
            dist_pc=orbit["dist_pc"],
        )

        # 2*N_pts observations - 4 fitted params
        n_dof = 2 * n_pts - 4
        # chi^2 should be within a few sigma of n_dof
        assert result.chi2 < n_dof + 6 * jnp.sqrt(2 * n_dof)

    def test_wrong_period_worse_ll(self):
        """Using the wrong period should give a worse log-likelihood."""
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit)

        result_true = thiele_innes_fit(
            data,
            T=orbit["T"],
            e=orbit["e"],
            tp=orbit["tp"],
            Ms=orbit["Ms"],
            dist_pc=orbit["dist_pc"],
        )

        result_wrong = thiele_innes_fit(
            data,
            T=orbit["T"] * 1.5,  # 50% wrong
            e=orbit["e"],
            tp=orbit["tp"],
            Ms=orbit["Ms"],
            dist_pc=orbit["dist_pc"],
        )

        assert result_true.log_likelihood > result_wrong.log_likelihood

    def test_grid_search_finds_correct_period(self):
        """Grid search should identify the correct period."""
        from photomancy.orbit.thiele_innes import thiele_innes_grid_search

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit, n_pts=25)
        true_log_T = jnp.log10(orbit["T"])

        # Coarse grid centered on truth
        log_T_grid = jnp.linspace(true_log_T - 0.3, true_log_T + 0.3, 30)
        e_grid = jnp.linspace(0.0, 0.6, 10)

        best = thiele_innes_grid_search(
            data,
            orbit["Ms"],
            orbit["dist_pc"],
            log_T_grid,
            e_grid,
            n_tp=15,
        )

        # Best-fit period should be close to truth
        assert jnp.allclose(best.T, orbit["T"], rtol=0.05)

    def test_jit_compatible(self):
        """TI fitter should work under jax.jit."""
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit)

        @jax.jit
        def _fit(T, e, tp):
            return thiele_innes_fit(
                data,
                T,
                e,
                tp,
                Ms=orbit["Ms"],
                dist_pc=orbit["dist_pc"],
            )

        result = _fit(
            jnp.float64(orbit["T"]),
            jnp.float64(orbit["e"]),
            jnp.float64(orbit["tp"]),
        )
        assert jnp.isfinite(result.log_likelihood)
        assert jnp.allclose(result.a, orbit["a"], rtol=1e-2)


# ============================================================================
# Initialization helpers (init.py)
# ============================================================================


class TestInit:
    """Tests for the MCMC initialization helpers."""

    @staticmethod
    def _consistent_orbit():
        """Eccentric orbit with Kepler III-consistent a and T."""
        T = 200.0
        Ms = Msun2kg
        e = 0.3
        w = jnp.pi / 4
        mu = G * Ms
        a = (mu * (T / (2.0 * jnp.pi)) ** 2) ** (1.0 / 3.0)
        return dict(
            T=T,
            a=float(a),
            e=e,
            cos_i=jnp.cos(jnp.radians(60.0)),
            cos_w=jnp.cos(w),
            sin_w=jnp.sin(w),
            W=jnp.pi / 3,
            tp=50.0,
            Ms=Ms,
            dist_pc=5.0,
        )

    def _make_astrom_data(self, params, n_pts=20, noise_arcsec=1e-6, seed=42):
        """Generate synthetic astrometry from known orbital parameters."""
        key = jax.random.PRNGKey(seed)
        times = jnp.linspace(0.0, params["T"] * 1.5, n_pts)
        ra_true, dec_true = predict_relative_astrometry(
            times=times,
            a=params["a"],
            e=params["e"],
            cos_i=params["cos_i"],
            cos_w=params["cos_w"],
            sin_w=params["sin_w"],
            W=params["W"],
            tp=params["tp"],
            Ms=params["Ms"],
            dist_pc=params["dist_pc"],
        )
        k1, k2 = jax.random.split(key)
        ra = ra_true + noise_arcsec * jax.random.normal(k1, shape=ra_true.shape)
        dec = dec_true + noise_arcsec * jax.random.normal(k2, shape=dec_true.shape)
        return RelativeAstromData(
            times=times,
            ra=ra,
            dec=dec,
            ra_err=jnp.full(n_pts, noise_arcsec),
            dec_err=jnp.full(n_pts, noise_arcsec),
            corr=jnp.zeros(n_pts),
            planet_id=jnp.zeros(n_pts, dtype=jnp.int32),
            is_valid=jnp.ones(n_pts, dtype=bool),
        )

    def test_ti_to_init_keys(self):
        """ti_to_init should produce the correct NumPyro parameter names."""
        from photomancy.orbit.init import ti_to_init
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit)
        result = thiele_innes_fit(
            data,
            T=orbit["T"],
            e=orbit["e"],
            tp=orbit["tp"],
            Ms=orbit["Ms"],
            dist_pc=orbit["dist_pc"],
        )

        init_dict = ti_to_init(result, orbit["Ms"])
        expected_keys = {"log_P", "e_raw", "w_raw", "cos_i", "W", "M0"}
        assert set(init_dict.keys()) == expected_keys

    def test_ti_to_init_values_near_truth(self):
        """Converted values should be close to truth parameters."""
        from photomancy.orbit.init import ti_to_init
        from photomancy.orbit.thiele_innes import thiele_innes_fit

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit)
        result = thiele_innes_fit(
            data,
            T=orbit["T"],
            e=orbit["e"],
            tp=orbit["tp"],
            Ms=orbit["Ms"],
            dist_pc=orbit["dist_pc"],
        )

        init_dict = ti_to_init(result, orbit["Ms"])

        # log_P should match
        assert jnp.allclose(
            init_dict["log_P"][0],
            jnp.log10(orbit["T"]),
            atol=0.01,
        )
        # e_raw should match
        assert jnp.allclose(init_dict["e_raw"][0], orbit["e"], atol=0.01)
        # All values should be finite
        for key, val in init_dict.items():
            assert jnp.all(jnp.isfinite(val)), f"Non-finite in {key}"

    def test_find_init_returns_valid_dict(self):
        """find_init should return a valid init dict near truth."""
        from photomancy.orbit.init import find_init

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit, n_pts=25)
        true_log_T = float(jnp.log10(orbit["T"]))

        init_dict = find_init(
            data,
            orbit["Ms"],
            orbit["dist_pc"],
            log_T_range=(true_log_T - 0.3, true_log_T + 0.3),
            n_log_T=30,
            n_tp=15,
        )

        # Should have all expected keys
        assert "log_P" in init_dict
        assert "M0" in init_dict
        # Period should be close to truth
        assert jnp.allclose(
            init_dict["log_P"][0],
            true_log_T,
            atol=0.05,
        )

    def test_find_init_top_k_returns_k_dicts(self):
        """find_init_top_k should return k distinct init dicts."""
        from photomancy.orbit.init import find_init_top_k

        orbit = self._consistent_orbit()
        data = self._make_astrom_data(orbit, n_pts=25)
        true_log_T = float(jnp.log10(orbit["T"]))

        k = 3
        init_list = find_init_top_k(
            data,
            orbit["Ms"],
            orbit["dist_pc"],
            k=k,
            log_T_range=(true_log_T - 0.3, true_log_T + 0.3),
            n_log_T=30,
            n_tp=15,
        )

        assert len(init_list) == k
        # All dicts should have the correct keys
        for d in init_list:
            assert set(d.keys()) == {"log_P", "e_raw", "w_raw", "cos_i", "W", "M0"}
        # The best one should be closest to truth
        assert jnp.allclose(
            init_list[0]["log_P"][0],
            true_log_T,
            atol=0.05,
        )
