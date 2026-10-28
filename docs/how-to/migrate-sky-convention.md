# Migrate saved fits to the east-north sky convention

Releases before photomancy 0.4.0 (with orbix 2.x) returned the sky offsets as
(north, east) in the slots labeled (RA, Dec). An astrometry-only fit made then
fitted the mirror image of the sky, so its saved orbital elements are mirrored
relative to the published convention (node measured north through east,
inclination from 0 to 180 degrees). This page converts those saved samples.

## What changes

For physical-element samples the conversion is

$$\cos i \to -\cos i, \qquad \Omega \to \left(\tfrac{\pi}{2} - \Omega\right) \bmod 2\pi,$$

and every other element (`a`, `e`, `T`, `cos_w`, `sin_w`, `tp`) is unchanged. The
first relation is the inclination change $i' = 180^\circ - i$, and the second the
node change $\Omega' = (90^\circ - \Omega) \bmod 360^\circ$. The derivation uses the
classical Thiele-Innes relations (Green 1985, *Spherical Astronomy*, ch. 17): with
$u = \omega + f$, the north and east offsets are
$N = r[\cos\Omega\cos u - \sin\Omega\cos i\sin u]$ and
$E = r[\sin\Omega\cos u + \cos\Omega\cos i\sin u]$, and the substitution above
exchanges them.

## Convert samples

```python
from photomancy.orbit import migrate_sky_convention

samples = {"T": T, "e": e, "a": a, "cos_i": cos_i, "W": W,
           "cos_w": cos_w, "sin_w": sin_w, "tp": tp}   # from an earlier fit
current = migrate_sky_convention(samples)
```

The function returns a new dict and accepts scalars, NumPy arrays and JAX arrays.
Pass the dict that `sample_physical` returned in the earlier release (or the
`SamplePosterior` columns collected into a dict).

## What cannot be converted

`migrate_sky_convention` raises `ValueError` for:

- posteriors stored in transformed sampling coordinates: the unconstrained NumPyro
  sites (`log_P`, `M0`, `e_raw`, `w_raw`), unit-cube `*_base` draws, and the
  Thiele-Innes coordinates of the few-epoch fitters (`lnP`, `tau`, `A`, `F`, `B`,
  `G`, now `east_X`, `east_Y`, `north_X`, `north_Y`);
- values outside the physical range (`|cos_i| > 1` or `W` outside $[0, 2\pi]$),
  which indicate unconstrained coordinates.

Refit those with the current release. Fits that used photometry must also be
refit, because the earlier phase angle was inverted. Convert such posteriors from
the unconstrained coordinates to physical elements with the earlier release's
`to_physical` only if you can still run it.

Epoch terms need no change. RV-only fits never used the sky chart and need no
conversion.

## Systems loaded from ExoVista

Planets loaded from ExoVista files carry orbital elements derived at load time with
the earlier convention. Do not migrate their saved elements. Rebuild every loaded
system from the original FITS file with the current skyscapes, which derives the
elements in the current chart, and discard any pickled or saved copy of the earlier
system.

## Check the result

Evaluate `predict_relative_astrometry` at the migrated elements: the track agrees
with the earlier model's output with its two columns exchanged. The test suite
checks this against an independent NumPy implementation of the Thiele-Innes
relations and against a refit of the Kervella et al. (2016) alpha Cen AB data.
