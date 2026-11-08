"""Stationary Gaussian fields: coloring, spectrum, variance and the support."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from photomancy.fields import StationaryGaussianField, radial_frequency

N = 32
DRAWS = 400


def _psd(k):
    return 2.0 * (1.0 + k**2) ** -1.5


def _field(support=None, scale=1.0):
    support = np.ones((N, N), bool) if support is None else support
    field = StationaryGaussianField.from_radial_psd(_psd, support, diameter=N)
    return StationaryGaussianField(field.amplitude * scale, field.support)


def _binned_periodogram(draws):
    """Mean periodogram per integer radial bin, and the cell count of each bin."""
    power = np.mean(np.abs(np.fft.fft2(np.asarray(draws, float))) ** 2, axis=0) / N**2
    k = np.rint(radial_frequency(N, N)).astype(int)
    bins = np.unique(k)
    return (
        bins,
        np.array([power[k == b].mean() for b in bins]),
        np.array([(k == b).sum() for b in bins]),
    )


def _expected_binned():
    k = radial_frequency(N, N)
    kb = np.rint(k).astype(int)
    return np.array([_psd(k)[kb == b].mean() for b in np.unique(kb)])


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
def test_draws_have_the_requested_spectrum():
    """Mean periodogram of 400 draws equals the PSD per radial bin.

    Each FFT cell's periodogram is exponential (relative std 1), and cells k and -k of a
    real field are conjugate, so a bin of M cells averaged over D draws has relative
    standard error at most sqrt(2 / (M D)); the band is 4 of those.
    """
    draws = _field().sample(jax.random.PRNGKey(0), DRAWS)
    _, got, cells = _binned_periodogram(draws)
    want = _expected_binned()
    tol = 4.0 * np.sqrt(2.0 / (cells * DRAWS))
    assert np.all(np.abs(got / want - 1.0) < tol)


@vv.case("scene-fits", "code-verification", basis="mutation")
def test_spectrum_check_rejects_a_psd_off_by_two():
    """The test above fails for an amplitude scaled by sqrt(2) (PSD off by 2)."""
    draws = _field(scale=np.sqrt(2.0)).sample(jax.random.PRNGKey(0), DRAWS)
    _, got, cells = _binned_periodogram(draws)
    tol = 4.0 * np.sqrt(2.0 / (cells * DRAWS))
    assert np.all(np.abs(got / _expected_binned() - 1.0) > tol)


@vv.case("scene-fits", "code-verification", basis="monte-carlo")
def test_pixel_variance_matches_the_draws():
    """Per-pixel variance of 400 draws is mean(A**2) at 4 sigma of sqrt(2/(D-1))."""
    field = _field()
    draws = np.asarray(field.sample(jax.random.PRNGKey(1), DRAWS), float)
    got = draws.var(axis=0, ddof=1).mean()
    want = float(field.pixel_variance())
    assert abs(got / want - 1.0) < 4.0 * np.sqrt(2.0 / (DRAWS - 1))


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_flat_unit_amplitude_is_the_identity():
    """A flat unit PSD colors white noise into itself."""
    field = StationaryGaussianField(jnp.ones((N, N)), jnp.ones((N, N), bool))
    w = jax.random.normal(jax.random.PRNGKey(2), (N, N))
    np.testing.assert_allclose(field.color(w), w, rtol=0, atol=1e-5)


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_color_is_linear_real_and_differentiable():
    """Coloring is linear, jit-stable and differentiable, and real to round-off."""
    field = _field()
    u, v = jax.random.normal(jax.random.PRNGKey(3), (2, N, N))
    np.testing.assert_allclose(
        field.color(2.0 * u - 0.5 * v),
        2.0 * field.color(u) - 0.5 * field.color(v),
        rtol=0,
        atol=1e-5,
    )
    full = jnp.fft.ifft2(field.amplitude * jnp.fft.fft2(u))
    assert float(jnp.max(jnp.abs(jnp.imag(full)))) < 1e-6 * float(
        jnp.max(jnp.abs(full))
    )
    np.testing.assert_array_equal(
        eqx.filter_jit(lambda f, w: f.color(w))(field, u), field.color(u)
    )
    check_grads(
        lambda w: jnp.sum(field.color(w) ** 2),
        (u,),
        order=1,
        modes=("rev",),
        atol=1e-2,
        rtol=1e-2,
    )


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_field_is_zero_outside_the_support():
    """The field vanishes exactly outside the support and not inside it."""
    yy, xx = np.indices((N, N))
    support = np.hypot(yy - N / 2, xx - N / 2) < N / 3
    draw = np.asarray(_field(support).sample(jax.random.PRNGKey(4)))
    assert np.all(draw[~support] == 0.0) and np.all(draw[support] != 0.0)


def test_invalid_inputs_raise():
    """Non-square, mismatched or negative inputs raise."""
    with pytest.raises(ValueError, match="square"):
        StationaryGaussianField(jnp.ones((4, 5)), jnp.ones((4, 5), bool))
    with pytest.raises(ValueError, match="support"):
        StationaryGaussianField(jnp.ones((4, 4)), jnp.ones((5, 5), bool))
    with pytest.raises(ValueError, match="non-negative"):
        StationaryGaussianField.from_radial_psd(
            lambda k: -k, np.ones((4, 4), bool), 4.0
        )


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_log_prob_of_a_colored_field_is_the_white_density():
    """On a full support log_prob(color(w)) = -0.5 |w|^2 (Parseval) to f32 round-off."""
    field = _field()
    w = jax.random.normal(jax.random.PRNGKey(5), (N, N))
    got = float(field.log_prob(field.color(w)))
    want = float(field.log_prob_white(w))
    assert abs(got / want - 1.0) < 1e-4


@vv.case("scene-fits", "code-verification", basis="floating-point")
def test_whiten_inverts_color_on_a_full_support():
    """Whiten and color invert each other on a full support, to f32 round-off."""
    field = _field()
    w = jax.random.normal(jax.random.PRNGKey(6), (N, N))
    np.testing.assert_allclose(field.whiten(field.color(w)), w, rtol=0, atol=1e-4)
    x = field.sample(jax.random.PRNGKey(7))
    scale = float(jnp.max(jnp.abs(x)))
    np.testing.assert_allclose(
        field.color(field.whiten(x)), x, rtol=0, atol=1e-5 * scale
    )
