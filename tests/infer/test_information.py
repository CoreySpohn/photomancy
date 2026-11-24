"""``Fit.information``: the observed Hessian and the Fisher matrix of a fit."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.memory import MemoryLimit
from priorities import Normal, Uniform
from scipy import integrate

jax.config.update("jax_enable_x64", True)

from photomancy.infer import AbstractTerm, Fit, ForwardTerm  # noqa: E402
from photomancy.infer._scale import SCALE_SPAN, information_scale  # noqa: E402
from photomancy.infer.likelihoods import (  # noqa: E402
    _KERNELS,
    gaussian_loglike,
    huber_loglike,
    laplace_loglike,
    student_t_loglike,
)


class _Model(eqx.Module):
    mu: jax.Array


class _Linear(eqx.Module):
    """Forward ``model -> A @ mu``."""

    matrix: jax.Array

    def __call__(self, model):
        return self.matrix @ model.mu


class _NoRoot(AbstractTerm):
    """A term with a value and no Fisher root."""

    reads = ()

    def __call__(self, model):
        return -0.5 * jnp.sum(model.mu**2)

    def draw(self, key, model):
        raise NotImplementedError


def _problem(n_data, n_param, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_data, n_param))
    sigma = rng.uniform(0.5, 2.0, size=n_data)
    data = rng.normal(size=n_data)
    mask = np.arange(n_data) % 4 != 1
    return matrix, sigma, data, mask


def _fit(matrix, likelihood, prior_sigma=3.0):
    n_param = matrix.shape[1]
    prior = Normal(jnp.zeros(n_param), prior_sigma * jnp.ones(n_param))
    term = ForwardTerm(_Linear(jnp.asarray(matrix)), likelihood)
    return Fit(_Model(jnp.zeros(n_param)), {"mu": prior}, (term,))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("shape", [(40, 5), (3, 5)], ids=["forward", "reverse"])
@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_fisher_and_observed_equal_the_linear_gaussian_closed_form(shape, batch_size):
    """For a linear Gaussian model both matrices are ``A' W A`` (plus the prior).

    The roots are longer than the five parameters (forward mode) and shorter
    (reverse mode), and every batching gives the same matrix. A Gaussian
    log-likelihood is quadratic, so its Hessian is the Fisher matrix at any point.

    Truth source: NumPy ``A' diag(mask / sigma**2) A``, and ``1 / prior_sigma**2`` on
    the diagonal for the Normal prior's Hessian.
    Tolerance basis: exact-algebra (rtol 1e-12).
    Fails if: the mask, the weights or the prior curvature is dropped, a batch is
    summed twice, or forward and reverse mode disagree.
    """
    matrix, sigma, data, mask = _problem(*shape)
    fit = _fit(matrix, gaussian_loglike(data, sigma, mask))
    x = {"mu": jnp.linspace(-1.0, 2.0, 5)}
    want = matrix.T @ (matrix * (mask / sigma**2)[:, None])
    prior = np.eye(5) / 9.0
    for kind in ("fisher", "observed"):
        for include_prior, extra in ((False, 0.0), (True, prior)):
            got = fit.information(
                x, kind=kind, include_prior=include_prior, batch_size=batch_size
            )
            np.testing.assert_allclose(got, want + extra, rtol=1e-12, atol=1e-13)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_observed_equals_jax_hessian_of_the_negative_log_density():
    """The column-by-column observed matrix is ``jax.hessian`` of the density.

    A Huber term (piecewise quadratic, so the Hessian depends on the point) at a
    point with residuals in both the core and the tails, with the prior.

    Truth source: ``jax.hessian`` of ``-fit.log_density`` over the flat entry vector.
    Tolerance basis: floating-point (rtol 1e-10; the same second derivative by a
    different contraction order).
    Fails if: the sign, the symmetrization or the prior term is wrong.
    """
    matrix, sigma, data, mask = _problem(30, 4, seed=3)
    data = data + 6.0 * (np.arange(30) % 7 == 0)
    fit = _fit(matrix, huber_loglike(data, sigma, mask))
    x = {"mu": jnp.array([0.3, -0.7, 1.2, 0.1])}
    want = jax.hessian(lambda v: -fit.log_density({"mu": v}))(x["mu"])
    for batch_size in (None, 1, 3):
        got = fit.information(x, kind="observed", batch_size=batch_size)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def _score_variance_constant(kind, shape_param):
    """``E[psi(z)**2]`` under the kernel's own normalized density, by quadrature.

    The densities are symmetric, so each integral is twice the one over ``z > 0``,
    split at the Huber knee.
    """
    kernel = _KERNELS[kind]
    psi = jax.grad(lambda z: kernel(z, shape_param))
    density = lambda z: np.exp(-float(kernel(z, shape_param)))  # noqa: E731
    knee = shape_param if kind == "huber" else 1.0

    def half(f):
        opts = {"epsabs": 1e-14, "epsrel": 1e-13}
        return (
            integrate.quad(f, 0.0, knee, **opts)[0]
            + integrate.quad(f, knee, np.inf, **opts)[0]
        )

    return half(lambda z: float(psi(z)) ** 2 * density(z)) / half(density)


@vv.case("scene-fits", "code-verification", basis="floating-point")
@pytest.mark.parametrize(
    ("builder", "kind", "shape_param"),
    [
        (laplace_loglike, "laplace", 0.0),
        (huber_loglike, "huber", 1.345),
        (lambda d, s, m: huber_loglike(d, s, m, delta=0.5), "huber", 0.5),
        (student_t_loglike, "student_t", 4.0),
    ],
    ids=["laplace", "huber", "huber-0.5", "student-t"],
)
def test_robust_fisher_is_a_constant_times_the_gaussian_one(builder, kind, shape_param):
    """A robust kernel's expected information is ``c / sigma**2``, ``c = E psi**2``.

    Truth source: ``c`` by numerical quadrature of the squared score of the kernel
    table's own function (autodiff) under its normalized density, times
    ``A' diag(mask / sigma**2) A``.
    Tolerance basis: floating-point (rtol 1e-9; adaptive quadrature to 1e-13).
    Fails if: a robust kernel is given the Gaussian information, the Huber or Student-t
    constant is wrong, or the mask is dropped.
    """
    matrix, sigma, data, mask = _problem(40, 5, seed=5)
    fit = _fit(matrix, builder(data, sigma, mask))
    c = _score_variance_constant(kind, shape_param)
    want = c * matrix.T @ (matrix * (mask / sigma**2)[:, None])
    got = fit.information({"mu": jnp.zeros(5)}, kind="fisher", include_prior=False)
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_a_fit_with_a_term_that_has_no_root_refuses_fisher_and_names_it():
    """One term without a root makes ``kind="fisher"`` unavailable for the whole fit."""
    matrix, sigma, data, _ = _problem(10, 3)
    gauss = ForwardTerm(_Linear(jnp.asarray(matrix)), gaussian_loglike(data, sigma))
    fit = Fit(
        _Model(jnp.zeros(3)),
        {"mu": Normal(jnp.zeros(3), jnp.ones(3))},
        (gauss, _NoRoot()),
    )
    x = {"mu": jnp.ones(3)}
    with pytest.raises(NotImplementedError, match=r"terms\[1\] \(_NoRoot\)"):
        fit.information(x, kind="fisher")
    assert fit.information(x, kind="observed").shape == (3, 3)


def test_a_callable_likelihood_has_no_root():
    """A ForwardTerm whose likelihood is not a masked kernel has no Fisher root."""

    class _Like(eqx.Module):
        def __call__(self, predicted):
            return -jnp.sum(predicted**2)

    fit = _fit(np.eye(3), _Like())
    with pytest.raises(NotImplementedError, match="ForwardTerm"):
        fit.information({"mu": jnp.ones(3)}, kind="fisher")


def test_an_unknown_kind_is_refused():
    """``kind`` is one of the two names."""
    fit = _fit(np.eye(3), gaussian_loglike(np.zeros(3), 1.0))
    with pytest.raises(ValueError, match="kind"):
        fit.information({"mu": jnp.ones(3)}, kind="expected")


def test_float32_is_refused():
    """With x64 off the curvature is refused rather than computed in float32."""
    fit = _fit(np.eye(3), gaussian_loglike(np.zeros(3), 1.0))
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="x64"):
        fit.information({"mu": jnp.ones(3, dtype=jnp.float32)}, kind="observed")


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_information_refuses_what_does_not_fit_max_bytes():
    """The closed-form estimate is checked before anything is built.

    Truth source: the byte counts of the definition. Fisher on 200 data and 5
    parameters (forward mode, root longer than the parameters) holds the 5 by 200
    tangent matrix and the 5 by 5 result, 8 bytes each, plus one 200-entry tangent per
    batched column.
    Tolerance basis: exact-algebra (a threshold on integers).
    Fails if: a limit below the resident estimate admits the call, one above it refuses,
    or the refusal does not say which batch fits.
    """
    matrix, sigma, data, _ = _problem(200, 5)
    fit = _fit(matrix, gaussian_loglike(data, sigma))
    x = {"mu": jnp.zeros(5)}
    fixed = (5 * 5 + 5 * 200) * 8
    per_column = 200 * 8
    with pytest.raises(MemoryLimit, match="product"):
        fit.information(x, kind="fisher", max_bytes=fixed)
    with pytest.raises(MemoryLimit, match=r"batch_size=2 fits"):
        fit.information(x, kind="fisher", max_bytes=fixed + 2 * per_column)
    ok = fit.information(
        x, kind="fisher", batch_size=2, max_bytes=fixed + 2 * per_column
    )
    assert ok.shape == (5, 5)
    # all at once is admitted exactly when every column fits; None disables the check
    assert fit.information(x, kind="fisher", max_bytes=fixed + 5 * per_column).shape
    assert fit.information(x, kind="fisher", max_bytes=None).shape == (5, 5)
    with pytest.raises(MemoryLimit):
        fit.information(x, kind="observed", max_bytes=100)


class _Two(eqx.Module):
    flux: jax.Array
    faint: jax.Array
    other: jax.Array


class _TwoForward(eqx.Module):
    """``model -> [flux * s1, faint * s2]``: two independent unit-weight data."""

    s1: float = eqx.field(static=True)
    s2: float = eqx.field(static=True)

    def __call__(self, model):
        return jnp.stack([model.flux * self.s1, model.faint * self.s2])


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_information_scale_is_the_root_norm_in_unconstrained_coordinates():
    """The scale of a coordinate is ``1 / sqrt(root norm + |prior curvature|)``.

    ``flux`` has a bounded (Uniform) law, so its unconstrained coordinate is the logit
    and the scale carries the Jacobian of the map: the diagonal of ``Fit.information``
    is not what the helper returns. ``faint`` has no data (a zero multiplier) and a
    Normal prior of width 0.5: its entry is the prior's, ``1 / 0.25``. ``other`` has
    neither data nor curvature and takes the floor, ``span`` times the smallest
    scale. The volume term of the logit, ``|d2/du2 log(s (1 - s))| = 2 s (1 - s)``,
    enters only with ``include_prior``. A fit with a term that has no root
    returns ``None``.

    Truth source: closed forms. With ``x = lo + (hi - lo) s`` and ``s`` the logistic
    of ``u``, ``dx/du = (hi - lo) s (1 - s)`` and the data entry of ``flux`` is
    ``(s1 dx/du)**2 / sigma**2``; the Normal prior's curvature is ``1 / width**2``.
    Tolerance basis: exact-algebra (rtol 1e-12, closed forms evaluated in float64).
    Fails if: the Jacobian of ``from_real`` is dropped, the prior is read without its
    absolute value, the floor is missing (an infinite scale), the dtype is not that of
    ``u``, or a term without a root does not return ``None``.
    """
    lo, hi, sigma, s1 = 0.0, 5.0, 0.5, 3.0
    like = gaussian_loglike(jnp.zeros(2), sigma)
    term = ForwardTerm(_TwoForward(s1, 0.0), like)
    laws = {
        "flux": Uniform(lo, hi),
        "faint": Normal(jnp.asarray(0.0), jnp.asarray(0.5)),
        "other": Normal(jnp.asarray(0.0), jnp.asarray(1e9)),
    }
    base = _Two(*(jnp.asarray(v, jnp.float64) for v in (1.0, 0.0, 0.0)))
    fit = Fit(base, laws, (term,))
    x = {"flux": jnp.asarray(1.5), "faint": jnp.asarray(0.0), "other": jnp.asarray(0.0)}
    u = fit.to_real(x)
    s = 1.5 / 5.0
    data = (s1 * (hi - lo) * s * (1 - s)) ** 2 / sigma**2
    got = information_scale(fit, u, include_prior=False)
    assert got["flux"].dtype == u["flux"].dtype == jnp.float64
    np.testing.assert_allclose(got["flux"], data**-0.5, rtol=1e-12)
    # no data and no prior curvature but a Normal's: the entries are the priors'
    np.testing.assert_allclose(got["faint"], got["flux"] * SCALE_SPAN, rtol=1e-12)
    np.testing.assert_allclose(got["other"], got["flux"] * SCALE_SPAN, rtol=1e-12)
    with_prior = information_scale(fit, u, include_prior=True)
    np.testing.assert_allclose(with_prior["faint"], 0.5, rtol=1e-12)
    np.testing.assert_allclose(with_prior["other"], 1e9, rtol=1e-12)
    volume = 2.0 * s * (1 - s)  # |d2/du2 log(s (1 - s))|, the logit's volume
    np.testing.assert_allclose(with_prior["flux"], (data + volume) ** -0.5, rtol=1e-12)
    custom = information_scale(fit, u, include_prior=False, span=10.0)
    np.testing.assert_allclose(custom["other"], 10.0 * got["flux"], rtol=1e-12)
    rootless = Fit(base, laws, (term, _NoRoot()))
    assert information_scale(rootless, u, include_prior=True) is None
