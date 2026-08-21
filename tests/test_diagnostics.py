"""Tests for the orbit diagnostics helpers added for the viz rollout."""

import jax
import jax.numpy as jnp
import numpy as np


class _Problem:
    """A minimal OrbitProblem stand-in with a traceable to_physical."""

    param_names = ("z0", "z1")

    def to_physical(self, z):
        """Map a flat z to a physical dict."""
        return {"T": 900.0 + 40.0 * z[0], "e": 0.3 * jax.nn.sigmoid(z[1])}


def test_orbits_from_samples_matches_from_period():
    """The adapter reproduces KeplerianOrbit.from_period elementwise."""
    from orbix import KeplerianOrbit

    from photomancy.orbit import orbits_from_samples

    rng = np.random.default_rng(5)
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

    orbit = orbits_from_samples(samples, Ms)
    direct = KeplerianOrbit.from_period(
        T_d=samples["T"],
        e=samples["e"],
        cos_i=samples["cos_i"],
        W_rad=samples["W"],
        cos_w=samples["cos_w"],
        sin_w=samples["sin_w"],
        tp_d=samples["tp"],
        Ms_kg=Ms,
    )
    np.testing.assert_array_equal(orbit.a_AU, direct.a_AU)
    np.testing.assert_array_equal(orbit.i_rad, direct.i_rad)
    np.testing.assert_array_equal(orbit.t0_d, direct.t0_d)
    assert orbit.a_AU.shape == (K,)


def test_mode_scalars_pulls_one_value_per_mode():
    """mode_scalars evaluates to_physical at each mode mean with weights."""
    from photomancy.orbit import mode_scalars
    from photomancy.posterior import MixturePosterior

    means = jnp.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    covs = jnp.stack([jnp.eye(2)] * 3)
    log_evidences = jnp.log(jnp.array([0.6, 0.3, 0.1]))
    posterior = MixturePosterior(means=means, covs=covs, log_evidences=log_evidences)

    values, weights = mode_scalars(posterior, _Problem(), "T")
    np.testing.assert_allclose(values, [900.0, 940.0, 860.0], rtol=1e-12)
    np.testing.assert_allclose(weights, [0.6, 0.3, 0.1], rtol=1e-6)
