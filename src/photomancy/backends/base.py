"""The Backend protocol: how inference is run on a logdensity."""

from abc import abstractmethod

import equinox as eqx


class AbstractBackend(eqx.Module):
    """Runs inference on a flat logdensity and returns a Posterior.

    Backends are ``eqx.Module`` config objects (hyperparameters as fields) with a
    pure ``run``. A backend sees only the flat ``logdensity`` -- never the scene or
    the forward model -- which is what keeps the engine forward-model agnostic.
    """

    @abstractmethod
    def run(self, logdensity, init, key=None):
        """Run inference on ``logdensity`` from ``init``, returning a Posterior.

        Args:
            logdensity: ``z -> scalar`` log-density over the flat parameter position.
            init: Initial flat position. Shape ``(d,)``.
            key: PRNG key (unused by deterministic backends such as Laplace).
        """
        raise NotImplementedError
