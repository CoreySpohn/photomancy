"""Tests for the orbit diagnostics helpers added for the viz rollout."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import G
from orbix.system import FrameOrbit
from sky_reference import position_au


class _Problem:
    """A minimal problem stand-in with a traceable to_physical."""

    param_names = ("z0", "z1")

    def require(self, posterior):
        """Accept any posterior; chart checks are covered elsewhere."""

    def to_physical(self, z):
        """Map a flat z to a physical dict."""
        return {"T": 900.0 + 40.0 * z[0], "e": 0.3 * jax.nn.sigmoid(z[1])}


def test_orbits_from_samples_matches_two_body_reference():
    """Fails if posterior tracks lose their elements, mass or time origin.

    Independent NumPy/Newton positions allow 5e-14 AU absolute and 5e-14
    relative error: tens of float64 ulps for the cube root, phase reduction
    and quaternion rotation. Element/period checks allow 5e-15 relative
    error for elementary roundoff, replacing exact same-constructor equality.
    """
    from photomancy.orbit import orbits_from_samples

    rng = np.random.default_rng(0)
    K = 6
    w = rng.uniform(0.0, 2.0 * np.pi, K)
    samples = {
        "T": rng.uniform(300.0, 500.0, K),
        "e": rng.uniform(0.0, 0.5, K),
        "cos_i": rng.uniform(-1.0, 1.0, K),
        "W": rng.uniform(0.0, 2.0 * np.pi, K),
        "cos_w": np.cos(w),
        "sin_w": np.sin(w),
        "tp": rng.uniform(0.0, 100.0, K),
        "ll_total": rng.standard_normal(K),
    }
    Ms = 1.989e30
    times = np.array([-120.0, 0.0, 137.0, 690.0])
    reference_day = 1234.5

    # Both the negligible-mass default and sampled true masses set the size.
    for mass in (None, Ms * np.linspace(0.01, 0.1, K)):
        draws = samples if mass is None else dict(samples, Mp=mass)
        total_mass = Ms if mass is None else Ms + mass
        a = np.cbrt(G * total_mass * (samples["T"] / (2 * np.pi)) ** 2)
        expected = np.stack(
            [
                position_au(
                    times,
                    a=a[k],
                    **{
                        name: samples[name][k]
                        for name in ("T", "e", "cos_i", "W", "cos_w", "sin_w", "tp")
                    },
                ).T
                for k in range(K)
            ]
        )
        orbit = orbits_from_samples(draws, Ms, reference_day=reference_day)
        assert isinstance(orbit, FrameOrbit)
        np.testing.assert_allclose(orbit.a_AU, a, rtol=5e-15, atol=0)
        np.testing.assert_allclose(
            orbit.to_classical()["i_rad"],
            np.arccos(samples["cos_i"]),
            rtol=5e-15,
            atol=0,
        )
        np.testing.assert_allclose(
            orbit.n_rad_per_d, 2 * np.pi / samples["T"], rtol=5e-15, atol=0
        )
        np.testing.assert_array_equal(orbit.epoch_d, reference_day + samples["tp"])
        actual, _ = orbit.at(reference_day + times)
        np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-14)
        assert orbit.a_AU.shape == (K,)


def test_mode_scalars_pulls_one_value_per_mode():
    """mode_scalars evaluates to_physical at each mode mean with weights."""
    from photomancy.infer.legacy_posterior import MixturePosterior
    from photomancy.orbit import mode_scalars

    means = jnp.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    covs = jnp.stack([jnp.eye(2)] * 3)
    log_evidences = jnp.log(jnp.array([0.6, 0.3, 0.1]))
    posterior = MixturePosterior(means=means, covs=covs, log_evidences=log_evidences)

    values, weights = mode_scalars(posterior, _Problem(), "T")
    np.testing.assert_allclose(values, [900.0, 940.0, 860.0], rtol=1e-12)
    np.testing.assert_allclose(weights, [0.6, 0.3, 0.1], rtol=1e-6)
