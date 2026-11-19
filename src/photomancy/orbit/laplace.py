"""Orbit MAP and Laplace fits in flat chart coordinates."""

import jax

from photomancy.infer.backends.laplace import LaplaceBackend, LaplaceMixtureBackend
from photomancy.infer.legacy_posterior import GaussianPosterior, MixturePosterior
from photomancy.orbit.inference import ChartOrbitProblem


def map_laplace_fit(
    problem: ChartOrbitProblem,
    init: jax.Array,
    *,
    n_steps: int = 100,
    min_eigenvalue: float = 1.0,
) -> GaussianPosterior:
    """Fit a Gaussian at the MAP in the problem's flat chart coordinates.

    Pass an interior initial position from ``problem.flatten(params)``. The
    eigenvalue floor is applied to the precision matrix in these coordinates. The
    posterior declares the problem's chart and epoch.
    """
    return problem.run(
        LaplaceBackend(n_steps=n_steps, min_eigenvalue=min_eigenvalue), init
    )


def map_laplace_mixture_fit(
    problem: ChartOrbitProblem,
    init: jax.Array,
    *,
    n_steps: int = 100,
    min_eigenvalue: float = 1.0,
) -> MixturePosterior:
    """Fit one Laplace mode per row of flat chart starts, declaring chart and epoch."""
    return problem.run(
        LaplaceMixtureBackend(n_steps=n_steps, min_eigenvalue=min_eigenvalue), init
    )
