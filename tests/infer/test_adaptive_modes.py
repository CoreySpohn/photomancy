"""AdaptiveImportance on a two-mode target: the defensive share keeps both modes.

Truth source: a mixture of two normalized Gaussians (masses 0.8 and 0.2, separation 4,
width 0.3) inside a uniform box, so the evidence is ``1 / volume`` and the mass of the
far mode is 0.2.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from priorities import Uniform

from photomancy.infer import (
    AbstractTerm,
    AdaptiveImportance,
    Fit,
    PriorProposal,
    importance_evidence,
    khat_threshold,
)

jax.config.update("jax_enable_x64", True)

HALF = 6.0
CENTERS = jnp.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
LOG_MASS = jnp.log(jnp.array([0.8, 0.2]))
WIDTH = 0.3


class _Model(eqx.Module):
    x: jax.Array


class _TwoModes(AbstractTerm):
    """Normalized two-Gaussian density of the three-vector ``x``."""

    reads = ()

    def __call__(self, model):
        sq = jnp.sum((model.x - CENTERS) ** 2, axis=-1) / WIDTH**2
        norm = -3 * jnp.log(WIDTH * jnp.sqrt(2 * jnp.pi))
        return jax.scipy.special.logsumexp(LOG_MASS - 0.5 * sq + norm)

    def draw(self, key, model):
        return self


def test_defensive_share_recovers_both_masses_of_a_two_mode_target():
    """Fails if the adaptive rounds drop the smaller mode or bias the evidence.

    Eight keys of ``AdaptiveImportance`` seeded from the prior (defensive share 0.1; the
    tempering ladder needs about eight rounds to reach the posterior, ten here):
    the mean far-mode mass is 0.2 and the mean evidence is ``1 / 12^3``, each within
    3 standard errors of the key-to-key scatter. The evidence is the importance mean of
    each key's raw weights, because a key whose k-hat is above ``khat_threshold``
    publishes a NaN evidence; the published evidence is that mean where the k-hat
    passes and NaN where it does not.
    """
    fit = Fit(
        _Model(jnp.zeros(3)),
        {"x": Uniform(-HALF * jnp.ones(3), HALF * jnp.ones(3))},
        (_TwoModes(),),
    )
    backend = AdaptiveImportance(
        PriorProposal(), n_draws=2000, n_adapt_draws=2**13, n_rounds=10, n_centers=64
    )
    mass, log_z, published, khat = [], [], [], []
    for key in jax.random.split(jax.random.key(0), 8):
        draws = backend._run(fit, key)
        w = np.asarray(jax.nn.softmax(draws.log_weights))
        far = np.asarray(draws.samples["x"])[:, 0] > 0
        mass.append(float(np.sum(w[far])))
        log_z.append(float(importance_evidence(draws.log_weights)[0]))
        published.append(float(draws.log_evidence))
        khat.append(float(draws.info.khat))
    mass, log_z = np.array(mass), np.array(log_z)
    published, passes = np.array(published), np.array(khat) <= khat_threshold(2000)
    np.testing.assert_allclose(published[passes], log_z[passes], rtol=1e-12)
    assert np.all(np.isnan(published[~passes]))
    assert abs(mass.mean() - 0.2) < 3 * mass.std(ddof=1) / np.sqrt(8)
    exact = -3 * np.log(2 * HALF)
    assert abs(log_z.mean() - exact) < 3 * log_z.std(ddof=1) / np.sqrt(8)
