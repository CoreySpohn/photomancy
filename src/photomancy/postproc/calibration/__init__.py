"""Calibration tools for detection statistics (experimental).

Three pieces that turn a statistic map into a calibrated detection claim:

- candidates: local maxima of a statistic map (find_candidates).
- null: how many independent trials a search covers and how many false alarms
  a calibrated statistic may produce (n_resolution_elements, false_alarm_band).
- injection: throughput statistics from injection-recovery records and the
  contrast curve they imply (throughput_profile, contrast_curve).

These are host-side post-run helpers on NumPy arrays; none is jit-compiled.
To calibrate a statistic map against reference apertures on the same ring,
use MapFilter with DetectionEstimator.
"""

from photomancy.postproc.calibration.candidates import Candidates, find_candidates
from photomancy.postproc.calibration.injection import (
    ThroughputProfile,
    contrast_curve,
    throughput_profile,
)
from photomancy.postproc.calibration.null import false_alarm_band, n_resolution_elements

__all__ = [
    "Candidates",
    "ThroughputProfile",
    "contrast_curve",
    "false_alarm_band",
    "find_candidates",
    "n_resolution_elements",
    "throughput_profile",
]
