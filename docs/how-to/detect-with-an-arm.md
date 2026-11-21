# Detect a planet with a detection arm

A detection arm takes the frames of one observation and a list of candidate
positions, and returns a test statistic and its false positive fraction (FPF)
at each position. Two arms ship with photomancy, `MawetArm` for aperture
photometry and `MatchedFilterArm` for a PSF-template fit. Both read the same
`FrameSet` and return the same `DetectionStats`, so changing the method changes
one line.

## The frames

The frames below are white noise on a 64 by 64 detector with a planted point
source at pixel row 32, column 50. A real `FrameSet` comes from coronagraphoto
or from your own reduction; the arm only needs the frames, their exposure times,
the pixel grid and the resolution element in pixels.

```python
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import J2000_JD
from hwoutils.epoch import Epoch
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band

from photomancy.postproc import FrameSet

rng = np.random.default_rng(0)
fwhm_px = 4.0
n = 64
y, x = np.mgrid[0:n, 0:n]
sigma = fwhm_px / 2.355
planet = 8.0 * np.exp(-((y - 32) ** 2 + (x - 50) ** 2) / (2 * sigma**2))

science = FrameSet(
    frames=jnp.asarray(rng.normal(size=(3, n, n)) + planet),
    time_d=jnp.arange(3.0),
    frame_time_s=jnp.full(3, 100.0),
    telescope_pa_deg=jnp.zeros(3),
    fwhm_px=fwhm_px,
    band=Band(495.0, 605.0),
    epoch=Epoch(J2000_JD),
    grid=PixelGrid.regular((n, n), 0.0218, "arcsec"),
)
positions = jnp.array([[32.0, 50.0], [32.0, 20.0]])  # (y, x), one on the source
```

## Aperture photometry

`MawetArm` compares the flux in the candidate aperture with the flux in
apertures at the same separation from the star, and returns the two-sample t
statistic of Mawet et al. (2014). The noise comes from the image itself, so the
arm ignores any reference frames.

```python continuation
from photomancy.postproc import MawetArm

stats = MawetArm(fwhm_px=fwhm_px).detect(science, positions)
print(stats.statistic, stats.fpf, stats.statistic_kind)
assert stats.fpf[0] < 1e-6 < stats.fpf[1]
```

The first position sits on the source and has a tiny FPF, and the second sees
only noise. `dof` holds the degrees of freedom of each t distribution, the
number of reference apertures minus one.

## Template fit

`MatchedFilterArm` fits a PSF template at each candidate and divides the fitted
amplitude by the spread of the same fit at the reference apertures. The
template comes from a provider. `ArrayTemplateProvider` holds one stamp for the
whole field, and `YippyTemplateProvider` reads the stamp for each position from
a coronagraph.

```python continuation
from photomancy.postproc import ArrayTemplateProvider, MatchedFilterArm

k = 15
yy, xx = np.mgrid[0:k, 0:k]
stamp = np.exp(-((yy - k // 2) ** 2 + (xx - k // 2) ** 2) / (2 * sigma**2))
provider = ArrayTemplateProvider(jnp.asarray(stamp / stamp.sum()))

stats = MatchedFilterArm(provider, fwhm_px=fwhm_px).detect(science, positions)
print(stats.statistic, stats.fpf, stats.statistic_kind)
assert stats.fpf[0] < 1e-6 < stats.fpf[1]
```

When the `FrameSet` carries `noise_variance`, the fit weights each pixel by its
inverse variance, which is what `whiten=True` asks for. Pass `whiten=False` to
fit with equal weights.

## Choose by the reporting convention

Use `MawetArm` when the number goes into a paper or is compared with VIP or
pyKLIP, because the convention is the community one. Use `MatchedFilterArm` when
the PSF is known and the question is how much signal a detection has, because
the template uses all the pixels of the planet instead of one aperture.
