"""Projected astrometric loss differentiated through frame-orbit kinematics.

These functions assume independent Gaussian errors with one scalar uncertainty.
They do not include priors, correlated noise, parallax, or radial velocities.
Use float64 and 0 <= e < 1. The default solver is orbix's implicit-JVP Markley
solver. Features are the mean motion and eccentricity vector at an epoch plus the
sky block, so no periapsis angle enters.
"""

import jax
import jax.numpy as jnp
from orbix.kepler import Markley
from orbix.system import FrameOrbit

_DEFAULT_SOLVER = Markley()


def _terms(features, times_d, observed, sigma, epoch_d, trig_solver):
    ex, ey, n_rad_per_d, ax, ay, bx, by = features
    # Only the rate, eccentricity vector and epoch enter plane coordinates; the
    # size and orientation are carried by the sky block (ax, ay, bx, by).
    orbit = FrameOrbit(
        jnp.ones(()),
        n_rad_per_d,
        ex,
        ey,
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jnp.asarray(epoch_d),
    )
    xy, _ = orbit.in_plane(times_d, solver=trig_solver)
    x, y = xy[..., 0], xy[..., 1]
    rx = (ax * x + bx * y - observed[0]) / sigma
    ry = (ay * x + by * y - observed[1]) / sigma
    return 0.5 * jnp.sum(rx * rx + ry * ry)


def astrometric_loss(
    features, times_d, observed, sigma, *, epoch_d=0.0, trig_solver=_DEFAULT_SOLVER
):
    """Return half chi-squared for projected geometry on orbix epoch-frame axes.

    The shape is the mean motion and eccentricity vector at ``epoch_d`` and the sky
    block is on the stored mean-longitude axes (``FrameOrbit.in_plane`` and
    ``linear_map``), so no periapsis angle or time of periapsis enters.

    Args:
        features: (N, 7) columns ``(ex, ey, n_rad_per_d, Ax, Ay, Bx, By)``. The
            prediction is ``A * X + B * Y`` with ``(X, Y)`` the plane position on
            the f and g axes; ``(Ax, Ay)`` is the first sky-block column (east,
            north) and ``(Bx, By)`` the second, in the units of ``observed``.
        times_d: Nonempty (T,) observation times in days.
        observed: (N, 2, T) east/north astrometry.
        sigma: Positive scalar standard deviation.
        epoch_d: Epoch (days) at which ``(ex, ey)`` is defined.
        trig_solver: An orbix.kepler solver object.

    Returns:
        Scalar loss summed over orbits, epochs and both coordinates.
    """

    def single(feature, data):
        return _terms(feature, times_d, data, sigma, epoch_d, trig_solver)

    return jnp.sum(jax.vmap(single)(features, observed))


def make_astrometric_evaluator(trig_solver=_DEFAULT_SOLVER):
    """Build a callable returning (loss, gradient_features[N,7]).

    The callable takes the arguments of ``astrometric_loss`` (``epoch_d``
    keyword-only, default 0) and differentiates all seven columns. To fit physical
    orbital elements, obtain features and a pullback with
    ``features, pullback = jax.vjp(to_features, elements)``; then apply
    ``pullback(gradient_features)[0]``. JIT, batching and higher derivatives work
    with the default solver. XLA chooses fusion: this interface does not guarantee
    bounded GPU temporaries. Derivatives with respect to times, observed data and
    sigma are not returned. ``trig_solver`` accepts an orbix.kepler solver object
    (default Markley()).
    """

    def evaluate(features, times_d, observed, sigma, *, epoch_d=0.0):
        def single(feature, data):
            return _terms(feature, times_d, data, sigma, epoch_d, trig_solver)

        values, gradients = jax.vmap(jax.value_and_grad(single))(features, observed)
        return jnp.sum(values), gradients

    return evaluate
