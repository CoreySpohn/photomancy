"""Tests for the photomancy core: scene-PyTree logdensity assembly."""

import jax.numpy as jnp

from photomancy.core import build_logdensity


def test_logdensity_peaks_at_truth():
    """build_logdensity composes prior + likelihood(forward(params))."""
    obs = 2.0

    def forward(params):
        return params["mu"]

    def likelihood(pred):
        return -0.5 * ((pred - obs) / 0.5) ** 2

    def prior(params):
        return -0.5 * (params["mu"] / 10.0) ** 2

    logdensity = build_logdensity(forward, likelihood, prior)

    ld_truth = logdensity({"mu": jnp.asarray(2.0)})
    ld_off = logdensity({"mu": jnp.asarray(-3.0)})

    assert ld_truth > ld_off
