"""Boundary and interface-contract tests for the detection core.

Covers the edge cases deferred from the detection-core phase gate: the
abstract interfaces reject direct instantiation, the aperture sampler's
r >= fwhm geometry gate holds at exactly r == fwhm, off-chip reference
apertures are excluded through the zero-fill validity sampling rather than
zero-filled into the noise sample, and the Grubbs pool-size gate reports
degrees of freedom even where it NaNs the statistic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from photomancy.postproc import (
    AbstractFilter,
    AbstractSampler,
    AbstractTest,
    ApertureFilter,
    ApertureSampler,
    DetectionEstimator,
    GrubbsTest,
    TwoSampleTTest,
    make_aperture_kernel,
    n_reference_apertures,
)

FWHM = 5.0


def _mawet() -> DetectionEstimator:
    """Aperture + aperture + two-sample-t pairing at the module FWHM."""
    return DetectionEstimator(
        filter=ApertureFilter(
            kernel=make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0),
            order=3,
        ),
        sampler=ApertureSampler(fwhm=FWHM),
        test=TwoSampleTTest(),
    )


class TestAbstractInterfaces:
    """The detection ABCs reject direct instantiation."""

    @pytest.mark.parametrize(
        "abstract", [AbstractFilter, AbstractSampler, AbstractTest]
    )
    def test_cannot_instantiate(self, abstract):
        """Instantiating an abstract detection interface raises TypeError."""
        with pytest.raises(TypeError):
            abstract()


class TestGeometryBoundary:
    """The aperture sampler's r >= fwhm gate at the boundary."""

    def test_candidate_at_exactly_one_fwhm_is_finite(self):
        """A candidate at r == fwhm sits inside the gate: finite, dof n - 1."""
        image = jax.random.normal(jax.random.PRNGKey(0), (101, 101))
        positions = jnp.array([[50.0, 50.0 + FWHM]])
        stats = _mawet()(image, positions)
        expected_refs = int(n_reference_apertures(jnp.asarray(FWHM), FWHM, 0.5))
        assert np.isfinite(float(stats.statistic[0]))
        assert float(stats.dof[0]) == expected_refs - 1

    def test_candidate_inside_one_fwhm_is_nan(self):
        """A candidate at r < fwhm fails the gate: statistic and fpf NaN."""
        image = jax.random.normal(jax.random.PRNGKey(0), (101, 101))
        positions = jnp.array([[50.0, 50.0 + 0.9 * FWHM]])
        stats = _mawet()(image, positions)
        assert np.isnan(float(stats.statistic[0]))
        assert np.isnan(float(stats.fpf[0]))


class TestOffChipApertures:
    """Reference apertures falling off-chip are excluded from the sample."""

    def test_off_chip_references_reduce_dof(self):
        """A star near the chip edge loses the off-chip arc of references.

        Same candidate radius twice: once fully on-chip, once with the star
        10 px from the left edge so part of the reference circle leaves the
        chip. The nearest-neighbor validity sampling fills off-chip
        positions with zero, so they drop out of the mask: the statistic
        stays finite and the dof falls strictly below the on-chip count.
        """
        image = jax.random.normal(jax.random.PRNGKey(1), (101, 101))
        radius = 25.0
        estimator = _mawet()
        on_chip = estimator(
            image,
            jnp.array([[50.0, 50.0 + radius]]),
            center_yx=jnp.array([50.0, 50.0]),
        )
        clipped = estimator(
            image,
            jnp.array([[50.0, 10.0 + radius]]),
            center_yx=jnp.array([50.0, 10.0]),
        )
        full_dof = int(n_reference_apertures(jnp.asarray(radius), FWHM, 0.5)) - 1
        assert float(on_chip.dof[0]) == full_dof
        assert 2.0 <= float(clipped.dof[0]) < full_dof
        assert np.isfinite(float(clipped.statistic[0]))


class TestGrubbsDofGate:
    """Grubbs reports dof even where the pool-size gate NaNs the statistic."""

    def test_dof_reported_when_statistic_gated(self):
        """A pool of two gives NaN statistic and fpf but dof zero."""
        samples = jnp.zeros(8)
        mask = jnp.zeros(8, dtype=bool).at[0].set(True)
        statistic, fpf, dof = GrubbsTest()(jnp.asarray(5.0), samples, mask)
        assert np.isnan(float(statistic))
        assert np.isnan(float(fpf))
        assert float(dof) == 0.0

    def test_dof_tracks_pool_above_gate(self):
        """A pool of nine gives dof seven alongside a finite statistic."""
        samples = jax.random.normal(jax.random.PRNGKey(2), (8,))
        mask = jnp.ones(8, dtype=bool)
        statistic, fpf, dof = GrubbsTest()(jnp.asarray(3.0), samples, mask)
        assert np.isfinite(float(statistic))
        assert np.isfinite(float(fpf))
        assert float(dof) == 7.0
