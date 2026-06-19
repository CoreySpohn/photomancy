"""The Backend protocol: how inference is run on a logdensity."""

from abc import abstractmethod

import equinox as eqx


class AbstractBackend(eqx.Module):
    """Runs inference on a flat logdensity and returns a Posterior.

    Backends are ``eqx.Module`` config objects (hyperparameters as fields) with a
    pure ``run``. A backend sees only the flat ``logdensity`` -- never the scene or
    the forward model -- which is what keeps the engine forward-model agnostic.

    Traced-vs-baked forward arrays: when ``logdensity`` is a ``SceneLogDensity``
    Module, its array leaves (e.g. a forward's PSF datacube) thread as traced inputs
    as long as the backend's compiled region receives the logdensity as a
    ``filter_jit`` argument. Every backend does this -- ``LaplaceBackend`` via the
    ``filter_jit``'d ``laplace_fit``, and the BlackJAX samplers (NUTS, SMC) by
    ``filter_jit``-wrapping ``run`` -- so a large coronagraph/IFS forward stays an
    input buffer instead of being baked into the kernel as a constant. A new backend
    should do the same.
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
