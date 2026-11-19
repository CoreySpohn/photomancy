"""The Backend protocol: how inference is run on a logdensity."""

import importlib
from abc import abstractmethod

import equinox as eqx
import jax


def import_extra(module, extra):
    """Import a sampler library that photomancy installs only as an extra.

    Args:
        module: Dotted module name, for example ``"blackjax.smc.resampling"``.
        extra: The photomancy extra that provides it (``"blackjax"`` or ``"jaxns"``).

    Returns:
        The imported module.

    Raises:
        ImportError: Naming the extra, when the library itself is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        package = module.partition(".")[0]
        if error.name is None or error.name.partition(".")[0] != package:
            raise
        raise ImportError(
            f"{package} is not installed. This backend needs the optional "
            f"dependency: pip install 'photomancy[{extra}]'."
        ) from error


class BackendDraws(eqx.Module):
    """What a backend hands to :func:`photomancy.infer.run`: native draws, unnormalized.

    ``run`` alone turns this into a :class:`~photomancy.infer.posterior.Posterior`:
    it normalizes the weights, evaluates ``log_target`` and fixes the dtypes.

    Attributes:
        samples: Entry dict of the draws, a flat draw axis in stored order (a leaf law
            leaf is ``(N, *leaf_shape)``, a block entry a dict of ``(N, R)`` arrays).
        log_weights: Unnormalized log weights ``(N,)`` that make the draws a weighted
            sample of the fit's density (zero for equal weights).
        log_evidence: Scalar log marginal likelihood, NaN when the backend has none.
        log_evidence_se: Scalar standard error of ``log_evidence``, NaN when absent.
        info: The backend's diagnostics module (chain index, flags, a trace) or None.
        kind: ``"monte_carlo"`` or ``"quadrature"``.
        log_target: The fit's log density at ``samples`` when the backend already
            computed it (the importance backends do), else ``None`` and ``run``
            evaluates it.
    """

    samples: dict
    log_weights: jax.Array
    log_evidence: jax.Array
    log_evidence_se: jax.Array
    info: eqx.Module | None
    kind: str = eqx.field(static=True)
    log_target: jax.Array | None = None


class AbstractBackend(eqx.Module):
    """Configuration of one inference method, run on a ``Fit`` by ``run``.

    A backend is an ``eqx.Module`` of settings (array leaves, or static counts). It is
    not callable: ``run(fit, backend, key)`` is the one public spelling, and ``run``
    builds the :class:`~photomancy.infer.posterior.Posterior`. A backend implements
    ``_run`` and returns its native draws and unnormalized log weights as
    :class:`BackendDraws`. It uses only ``Fit`` methods (``real_log_prob``, ``to_real``,
    ``from_real``, ``sample_prior``, ``log_density``, ``model_at``) and never builds its
    own density or tests what kind of entry the fit holds.

    Every backend except ``NestedSampling`` passes the fit to its compiled region as an
    argument (``filter_jit``), so large arrays inside the terms are traced inputs and
    not baked-in constants. ``NestedSampling`` closes over the fit in a jaxns model,
    runs one fit at a time and recompiles for a new fit; ``MAP`` is host driven and
    also runs one fit at a time.
    """

    @abstractmethod
    def _run(self, fit, key, *, batch_size: int | None = None) -> BackendDraws:
        """Run inference on ``fit``; return native draws and unnormalized weights.

        ``batch_size`` is the ``jax.lax.map`` chunk of density evaluations over a cloud
        of draws (``None`` evaluates every draw at once). A backend that evaluates the
        density of a cloud (the importance backends) honors it; a sampler evaluates its
        density per chain or particle inside its own loop and ignores it.
        """


class LegacyBackend(eqx.Module):
    """Runs inference on a flat logdensity and returns a legacy posterior.

    Superseded by :class:`AbstractBackend` and :func:`photomancy.infer.run`; kept for
    the orbit engines that still call it.


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
