# Post-processing

`photomancy.postproc` is the analysis companion to coronagraphoto. It handles post-processing—everything that happens after the photons hit the detector: PSF subtraction, signal extraction, and SNR estimation.

This document covers the conceptual foundation, implementation details, and practical usage.

Example scenes come from `skyscapes.datasets.fetch_scene()` and coronagraph
YIPs from `yippy.fetch_yip("eac1_aavc_2d")`.

---

## Motivation

The HWO simulation ecosystem separates **image generation** from **image analysis**:

- **coronagraphoto** simulates the physics: light propagation, coronagraph PSFs (via yippy), detector noise, zodiacal backgrounds.
- **`photomancy.postproc`** does the math: PCA subtraction, aperture photometry, detection statistics.

The likelihood-ratio map in `photomancy.detection` is the model-based counterpart:
it profiles a source amplitude under a fitted likelihood, while this package gives
the aperture and filter statistics that reporting conventions expect.

This boundary keeps the simulation agnostic to downstream analysis choices, and lets the analysis code run on any image—synthetic or real.

---

## Core Ideas

### JAX-Native Design

Everything in `photomancy.postproc` runs through JAX. That means:

1. **JIT Compilation**: Functions compile to XLA and run on CPU or GPU with no code changes.
2. **Differentiability**: You can backpropagate through the entire pipeline. This enables gradient-based optimization of coronagraph masks, observer integration times, or orbital parameters.
3. **Vectorization**: `vmap` replaces Python loops. Checking 10,000 candidate positions takes roughly the same time as checking one.

### SNR Estimation

`photomancy.postproc` implements the **Mawet et al. (2014)** method for signal-to-noise estimation:

- Places discrete apertures around the target separation
- Applies Small-Sample Statistics corrections
- Matches the methodology used in VIP and pyKLIP

Use it when you need to publish a detection significance or compare with community libraries.

---

## Implementation

### Package Layout

```
photomancy/postproc/
├── contracts.py   # FrameSet, DetectionStats
├── arms.py        # AbstractDetectionArm and the Mawet and matched-filter arms
├── core/          # Low-level JAX primitives
│   ├── geometry.py      # Radial distances, aperture coordinates
│   ├── photometry.py    # Convolution kernels, flux maps
│   ├── detection/      # Composable detection and named configurations
│   ├── pca.py           # Snapshot KLIP
│   └── statistics.py    # Masked mean, std, small-sample correction
├── calibration/   # Candidate finding, null counting, injection throughput
├── templates/     # PSF-template providers
├── pipelines/     # High-level workflows
│   └── yield_pipeline.py  # klip_subtract, calculate_yield_snr
└── interfaces/    # Adapters for other packages
    └── coronagraphoto.py
```

The `core/` modules use JAX functions and Equinox modules. An estimator precomputes its convolution kernel when constructed and can be reused across images.

### Handling Static Shapes

JAX requires array shapes to be known at trace time. This conflicts with astronomy code where aperture size depends on runtime parameters like FWHM.

The pattern we use:

1. A **Python wrapper** computes concrete integers (kernel size, aperture count) from the float FWHM.
2. It builds the necessary arrays (convolution kernels, coordinate buffers).
3. A **JIT-compiled inner function** receives those arrays and performs the actual math.

This keeps the public API flexible while satisfying JAX's constraints.

### Batched Extraction

The naive approach—looping over planet candidates—wastes work because the expensive image convolution gets repeated. Instead:

1. Compute the flux map (convolution with aperture kernel) **once**.
2. Use `jax.vmap` to extract values at many (y, x) positions in parallel.

`DetectionEstimator` implements this pattern. Each call returns a statistic, false positive fraction, and degrees of freedom for every requested position.

---

## Usage

### Disk Image Orientation

`make_simple_disk` follows the FITS display convention: row index increases
north and column index increases west. Displaying the array with
`imshow(origin="lower")` and no axis inversion shows north up and east left.
The major-axis position angle `pa_deg` is measured from north through east,
counterclockwise on this image. For example, PA 30 degrees points along
(column, row) = (-0.5, 0.866), with the opposite end of the ring on the same axis.

### Basic SNR Calculation

Requires a residual image from your observation.

```python notest
from photomancy.postproc import DetectionEstimator
import jax.numpy as jnp

# Your post-processed residual image
residual = ...  # shape (H, W)

# Planet locations in (y, x) pixel coordinates
positions = jnp.array([[100.5, 120.3]])  # (N, 2) array

# Resolution element size in pixels
fwhm = 3.5

stats = DetectionEstimator.mawet(fwhm)(residual, positions)
snr_values = stats.statistic
```

### Yield Pipelines

For mission simulations, use the subtraction primitives:

Requires science, star, and disk images from your simulation.

```python notest
from photomancy.postproc import DetectionEstimator, subtract_disk, subtract_star
import jax.numpy as jnp

# From coronagraphoto
science = noisy_observation  # counts
star_model = noiseless_star  # counts (expected)
disk_model = noiseless_disk  # counts (expected, or zeros)

# Subtract stellar PSF
residual = subtract_star(science, star_model)

# Optionally subtract disk model
residual = subtract_disk(residual, disk_model)

# Batch SNR for multiple candidates
positions = jnp.array([[100.5, 120.3], [100.5, 80.0]])
snr_values = DetectionEstimator.mawet(fwhm=3.5)(residual, positions).statistic
```

### High-Performance Pipelines

Build the estimator once to reuse its kernel in an iterative pipeline:

```python
from photomancy.postproc import DetectionEstimator
import jax

# Pre-compute aperture kernel once
estimator = DetectionEstimator.mawet(fwhm=3.5, fast=True)


# Use in JIT-compiled loop
@jax.jit
def process_cube(images, positions):
    return jax.vmap(lambda img: estimator(img, positions).statistic)(images)
```

Use `estimator.map(image)` to evaluate the statistic at every pixel.
`DetectionEstimator.matched_filter(fwhm)` uses Gaussian filtering and
annular noise estimation with the same call signature.

### Integration with coronagraphoto

The interface accepts the `Image` returned by `coronagraphoto.observe`.
Its grid carries the sampling and orientation; the caller supplies a `Frame`
from the primary to define lambda/D. Core routines continue to take arrays.

```python notest
from photomancy.postproc.interfaces import (
    analyze_observation,
    get_fwhm,
    sky_offset_to_position,
)

frame = instrument.primary.frame(wavelength_nm, convention="circumscribed")
position_yx = sky_offset_to_position(observed, east_arcsec=0.2, north_arcsec=0.1)
fwhm_px = get_fwhm(observed.grid, frame, wavelength_nm)
snr_value = analyze_observation(observed, position_yx, wavelength_nm, frame)
```

`get_fwhm` returns one lambda/D in pixels on an axis-aligned isotropic grid.
`extract_image(observed)` returns the first band plane as a 2D electron array.
All pixel coordinates use row j, column i centered at (x, y) = (i, j).

For PSF-template filtering, provide the science grid directly:

```python notest
from yippy import open_yip
from photomancy.postproc.templates.yippy import YippyTemplateProvider

coronagraph = open_yip("path/to/yip").prepare()

provider = YippyTemplateProvider(
    coronagraph,
    observed.grid,
    stamp_size=21,
    wavelength_nm=wavelength_nm,
)
stamp = provider(position_yx)
```

The provider takes the source grid from the selected imaging band's
`at_wavelength(wavelength_nm)` response, including its frame, center and full
basis. It accepts science grids in arcsec or lambda/D and
resamples through `source_psfs(..., grid=...)` using the source Image's
declared reconstruction. `order` selects interpolation for stamp extraction.
An unoriented coronagraph grid shares the science detector axes at the given
pointing. If both grids declare orientations, sampling passes through the sky
plane. An oriented native grid requires an oriented science grid.

`FrameSet(..., grid=observed.grid)` carries the star center and sampling;
its `noise_variance` remains per-pixel variance in electrons squared.
`PPConfig`, `ppfact`, `ez_ppf`, and `n_rolls` configuration fields are removed.
Analytic post-processing uses explicit residual fractions in
`optixstuff.PostProcessing`; photomancy does not call the calculator.

---

## Further Reading

- `examples/postproc_snr_methods` – SNR calculation demonstration
- `tests/postproc/test_snr_physics.py` – Physics-based validation tests
- `examples/postproc_coronagraphoto_pipeline` – End-to-end example with coronagraphoto


Detector count statistics, S/N and exposure time belong to optixstuff. This example uses aperture photon integrands, image photon
integrands, with the wavelength quadrature carried by each rate record.

```python notest
import optixstuff as ox
from optixstuff import frames_for_time

post_processing = ox.PostProcessing(
    speckle_residual_fraction=0.1, exozodi_residual_fraction=0.1
)
needed = ox.exposure_time(
    rates, detector, target_snr, frame_time_s=100.0, post_processing=post_processing
)
value = ox.snr(
    rates, detector, needed.time_s, frame_time_s=100.0, post_processing=post_processing
)
plan = frames_for_time(needed.time_s, 100.0)  # whole frames
electrons = detector.electrons(rates)
mean, variance = detector.mean_and_variance(electrons, plan)
# For image integrands (one detector pixel per spatial sample):
image_electrons = detector.electrons(image_rates)
counts = detector.draw(key, image_electrons, plan)
```

Construct `FrameSet` with a `Band` interval and an `Epoch` host value. Convert
absolute dates once on the host with `epoch.days(dates)` and pass those offsets
as `time_d`; the detector frames remain counts and their variance remains in
squared electrons.
