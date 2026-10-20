"""Projected astrometric loss with explicit orbital-basis partials.

These functions assume independent Gaussian errors with one scalar uncertainty.
They do not include priors, correlated noise, parallax, or radial velocities.
Use float64 and 0 <= e < 1. The default solver is orbix's implicit-JVP Markley
solver.
"""

import jax
import jax.numpy as jnp
from orbix.kepler import markley_solve_trig


def _terms(features, times_d, observed, sigma, trig_solver):
    e, tp_d, n_rad_per_d, ax, ay, bx, by = features
    elapsed_d = times_d - tp_d
    sine, cosine = trig_solver(n_rad_per_d * elapsed_d, e)
    beta = jnp.sqrt((1 - e) * (1 + e))
    x, y = cosine - e, beta * sine
    rx = (ax * x + bx * y - observed[0]) / sigma
    ry = (ay * x + by * y - observed[1]) / sigma
    return e, elapsed_d, sine, cosine, beta, x, y, rx, ry


def astrometric_loss(
    features, times_d, observed, sigma, *, trig_solver=markley_solve_trig
):
    """Return half chi-squared for factored projected orbital geometry.

    Args:
        features: (N, 7) columns (e, tp_d, n_rad_per_d, Ax, Ay, Bx, By).
            Prediction is A*(cos(E)-e) + B*sqrt(1-e**2)*sin(E).
            A and B include semimajor axis and projection, with the eccentricity
            factor removed from B. They can be in AU or angular units.
            The x/y components must follow the same order as observed; for
            (RA, Dec) data use (east, north), i.e. raw orbix rows (0, 1).
        times_d: Nonempty (T,) observation times, in the same day origin as tp_d.
        observed: (N, 2, T) astrometry in the same units as A and B.
        sigma: Positive scalar standard deviation in those units.
        trig_solver: Broadcast-capable (M, e) -> (sin(E), cos(E)) callable.

    Returns:
        Scalar loss summed over orbits, epochs and both projected coordinates.
    """

    def single(feature, data):
        *_, rx, ry = _terms(feature, times_d, data, sigma, trig_solver)
        return 0.5 * jnp.sum(rx * rx + ry * ry)

    return jnp.sum(jax.vmap(single)(features, observed))


def make_astrometric_evaluator(trig_solver=markley_solve_trig):
    """Build a callable returning (loss, gradient_features[N,7]).

    The callable has the same four positional arguments as astrometric_loss.
    All seven columns are differentiated, including both geometry vectors.
    To fit physical orbital elements, obtain features and a pullback with
    ``features, pullback = jax.vjp(to_features, elements)``; then apply
    ``pullback(gradient_features)[0]``. Only the small per-orbit transformation
    uses reverse AD. Epoch residuals and partials are explicitly reduced.

    The Kepler partial dE/dM is taken from the solver's own forward derivative,
    so the analytic gradient follows whatever implicit-derivative convention the
    solver defines (for the Markley solver, a cancellation-free 1 - e*cos(E)).
    Under JIT the repeated solve is deduplicated.

    JIT, batching and higher derivatives work with the default solver. XLA
    chooses fusion: this interface does not guarantee bounded GPU temporaries.
    Derivatives with respect to times, observed data, and sigma are not returned.
    Alternative solvers must accurately solve Kepler's equation and follow its
    implicit derivative convention. Interpolated grid solvers do not satisfy
    this contract: the derivatives below are not interpolation derivatives.
    """

    def evaluate(features, times_d, observed, sigma):
        def single(feature, data):
            e, elapsed, sine, _, beta, x, y, rx, ry = _terms(
                feature, times_d, data, sigma, trig_solver
            )
            _, _, motion, ax, ay, bx, by = feature
            mean = motion * elapsed
            _, (dsine, dcosine) = jax.jvp(
                lambda m: trig_solver(m, e), (mean,), (jnp.ones_like(mean),)
            )
            qx, qy = rx / sigma, ry / sigma
            dmean = qx * (ax * dcosine + bx * beta * dsine)
            dmean += qy * (ay * dcosine + by * beta * dsine)
            de = sine * dmean - qx * (ax + bx * e * sine / beta)
            de -= qy * (ay + by * e * sine / beta)
            gradient = jnp.stack(
                (
                    jnp.sum(de),
                    -motion * jnp.sum(dmean),
                    jnp.sum(elapsed * dmean),
                    jnp.sum(qx * x),
                    jnp.sum(qy * x),
                    jnp.sum(qx * y),
                    jnp.sum(qy * y),
                )
            )
            return 0.5 * jnp.sum(rx * rx + ry * ry), gradient

        values, gradients = jax.vmap(single)(features, observed)
        return jnp.sum(values), gradients

    return evaluate
