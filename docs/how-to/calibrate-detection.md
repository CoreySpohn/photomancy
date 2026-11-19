# Calibrate a statistic map and count false alarms

A detection statistic is only as good as its null. This page takes a statistic
map, for example a likelihood-ratio z map from a model fit, and answers two
questions. Is the statistic calibrated: does a threshold at a stated false
positive fraction (FPF) produce the false alarms it promises? And if so, how
many false alarms may a survey of many frames show before something is wrong?

The tools are experimental and live in `photomancy.postproc.calibration`, plus
`MapFilter` in the detection core.

## A statistic map

The map below is white noise on a 101 x 101 grid with one planted peak. Real
maps come from your own fit; the tools only need the array.

```python
import jax.numpy as jnp
import numpy as np

n = 101
center = np.array([50.0, 50.0])
rng = np.random.default_rng(0)
stat_map = rng.normal(size=(n, n))
stat_map[50, 80] += 10.0  # a planted source, 30 px from the center
```

## Find candidates

`find_candidates` lists local maxima above a threshold inside an annulus,
strongest first. Peaks closer than `min_sep_px` to a stronger one are dropped;
use one resolution element (FWHM).

```python continuation
from photomancy.postproc.calibration import find_candidates

found = find_candidates(
    stat_map, center, r_min_px=6.0, r_max_px=40.0, threshold=4.0, min_sep_px=3.0
)
print(found.positions_yx[0], found.statistic[0])
assert tuple(found.positions_yx[0]) == (50.0, 80.0)
```

Pass `validity=` (nonzero = usable) to keep bad pixels out of the search.

## Calibrate against the same ring

The raw map is not assumed to be N(0, 1). `MapFilter` lets
`DetectionEstimator` treat each candidate like an aperture measurement: it
samples the map at the candidate and at reference positions on the same ring,
then forms the two-sample t statistic of Mawet et al. (2014) and its FPF.
Pass the map where the image goes.

```python continuation
from photomancy.postproc import (
    ApertureSampler,
    DetectionEstimator,
    MapFilter,
    TwoSampleTTest,
)

estimator = DetectionEstimator(
    MapFilter(order=1), ApertureSampler(fwhm=3.0), TwoSampleTTest()
)
stats = estimator(jnp.asarray(stat_map), jnp.asarray(found.positions_yx[:1]))
print(float(stats.statistic[0]), float(stats.fpf[0]))
assert float(stats.fpf[0]) < 1e-6
```

Non-finite map pixels become zero inside `MapFilter`; also mark them in the
`validity_map` argument of the estimator so the sampler leaves them out.

## Count false alarms

On a frame with no sources, a calibrated statistic exceeds a threshold at
FPF with probability FPF in each independent trial. The number of trials in
the search annulus is the number of resolution elements, and the number of
exceedances over all trials follows a binomial distribution.

```python continuation
from photomancy.postproc.calibration import false_alarm_band, n_resolution_elements

n_trials = n_resolution_elements(6.0, 40.0, 3.0)
lo, hi = false_alarm_band(n_trials, fpf=1e-2, level=0.99)
print(n_trials, lo, hi)
assert lo <= 0.01 * n_trials <= hi
```

`n_resolution_elements` cuts the annulus into rings one FWHM wide, starting at
the inner radius, and counts `floor(2 pi r / fwhm)` elements at each ring
center; a remainder narrower than one ring is dropped. Correlated residuals
have fewer independent elements than this count, so treat it as an upper
bound on the trials.

`false_alarm_band` returns the central band of the exact binomial
distribution: with 99 percent coverage, an exceedance count outside
`[lo, hi]` is evidence that the statistic is miscalibrated, high or low.
Pool the counts of many null frames before comparing: a single frame has few
trials and a wide band relative to the count.

## Throughput and contrast

Injection and recovery turns a noise profile into a detection limit. Inject a
source with a known template at each separation (you supply the template; no
stamp is shifted), measure the recovered amplitude, and summarize.

```python continuation
from photomancy.postproc.calibration import contrast_curve, throughput_profile

separation = np.repeat([10.0, 20.0, 30.0], 6)
injected = np.full(18, 1e-6)
recovered = injected * rng.normal(np.repeat([0.7, 0.8, 0.9], 6), 0.05)
profile = throughput_profile(separation, injected, recovered)
print(profile.mean, profile.std, profile.count)

noise = np.array([4e-8, 2e-8, 1e-8])  # contrast-unit noise at each separation
limit = contrast_curve(5.0, noise, profile.mean)
print(np.asarray(limit))
assert np.all(profile.count == 6)
```

`throughput_profile` also accepts `edges=` to bin records by separation.
`contrast_curve` returns `threshold * noise / throughput`, the contrast
recovered at the threshold.
