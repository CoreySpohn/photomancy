"""A 32-pixel, one-wavelength toy coronagraph rendered with jnp, for fit tests.

A 16-sample circular pupil is zero-padded to 32 samples, so the detector has two
pixels per resolution element. The light passes an entrance-pupil path difference (a
known nominal map plus the fitted ``ote`` seam), a hard-edged focal-plane occulter, a
Lyot stop and an exit-pupil path difference (the fitted ``si`` seam), and lands on
the detector. The two seams are six low-order mode coefficients each, in nanometers,
so the fit is well posed. A source at ``(tx, ty)`` detector pixels is a linear phase
ramp in the entrance pupil. Nothing here comes from an optics library: it is only a
differentiable image with the seams, per-exposure nuisance parameters and companion
structure of a flight coronagraph.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

N_PUP, N_PAD, WAVELENGTH_NM = 16, 32, 1000.0
_MODES = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (1, 2))


def _pupil_coordinates():
    c = (N_PUP - 1) / 2.0
    y, x = np.mgrid[:N_PUP, :N_PUP] - c
    return x / (N_PUP / 2.0), y / (N_PUP / 2.0)


def _mode_basis():
    x, y = _pupil_coordinates()
    return np.stack(
        [np.cos(np.pi * kx * x) * np.cos(np.pi * ky * y) for kx, ky in _MODES]
    )


class Instrument(eqx.Module):
    """Shared optics: fixed masks, basis and nominal map; the two fitted seams."""

    pupil: jnp.ndarray
    lyot_stop: jnp.ndarray
    occulter: jnp.ndarray
    basis: jnp.ndarray
    nominal_nm: jnp.ndarray
    ote: jnp.ndarray
    si: jnp.ndarray


class Star(eqx.Module):
    """Per-exposure nuisance: star position in detector pixels and flux."""

    tilt: jnp.ndarray
    flux: jnp.ndarray


class Companion(eqx.Module):
    """A point source at a fixed offset in detector pixels, with a flux."""

    offset: jnp.ndarray
    flux: jnp.ndarray


def build_instrument():
    """Instrument with zero seam coefficients and a known 15 nm nominal map."""
    x, y = _pupil_coordinates()
    r = np.hypot(x, y)
    c = (N_PAD - 1) / 2.0
    fy, fx = np.mgrid[:N_PAD, :N_PAD] - c
    basis = jnp.asarray(_mode_basis())
    nominal = np.random.default_rng(0).normal(size=len(_MODES))
    nominal = jnp.tensordot(jnp.asarray(nominal / np.std(nominal) * 15.0), basis, 1)
    return Instrument(
        pupil=jnp.asarray((r <= 1.0).astype(float)),
        lyot_stop=jnp.asarray((r <= 0.8).astype(float)),
        occulter=jnp.asarray((np.hypot(fx, fy) > 3.0).astype(float)),
        basis=basis,
        nominal_nm=nominal,
        ote=jnp.zeros(len(_MODES)),
        si=jnp.zeros(len(_MODES)),
    )


def _pad(a):
    p = (N_PAD - N_PUP) // 2
    return jnp.pad(a, p)


def point_image(instrument, tilt):
    """Detector image of a unit source at ``tilt`` pixels."""
    k = 2.0 * jnp.pi / WAVELENGTH_NM
    ote = jnp.tensordot(instrument.ote, instrument.basis, 1)
    si = jnp.tensordot(instrument.si, instrument.basis, 1)
    c = (N_PAD - 1) / 2.0
    y, x = jnp.mgrid[:N_PAD, :N_PAD] - c
    ramp = 2.0 * jnp.pi * (tilt[0] * x + tilt[1] * y) / N_PAD
    entrance = _pad(instrument.pupil) * jnp.exp(
        1j * (_pad(k * (instrument.nominal_nm + ote)) + ramp)
    )
    focal = jnp.fft.fftshift(jnp.fft.fft2(entrance, norm="ortho"))
    lyot = jnp.fft.ifft2(jnp.fft.ifftshift(focal * instrument.occulter), norm="ortho")
    p = (N_PAD - N_PUP) // 2
    lyot = lyot[p:-p, p:-p] * instrument.lyot_stop * jnp.exp(1j * k * si)
    detector = jnp.fft.fftshift(jnp.fft.fft2(_pad(lyot), norm="ortho"))
    return jnp.abs(detector) ** 2


class StarlightRender(eqx.Module):
    """``(instrument, star, companion) -> image``: the star plus the companion."""

    def __call__(self, instrument, star, scene):
        """Render the exposure; a companion of flux 0 adds exactly 0."""
        image = star.flux * point_image(instrument, star.tilt)
        return image + scene.flux * point_image(instrument, scene.offset)


def star_only_image(instrument, star):
    """The render with no companion term at all."""
    return star.flux * point_image(instrument, star.tilt)
