"""Migration of saved samples fitted under the earlier (north, east) sky output.

The earlier release returned the sky offsets as (north, east). The independent
NumPy oracle of Green (1985), Spherical Astronomy, ch. 17 (the classical
Thiele-Innes constants) supplies both orders, so the old model is its reversed
output and no photomancy or orbix code stands behind the expected values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sky_reference import sky_position
from test_sky_convention import astrometry, fit, k16, reference  # noqa: F401

from photomancy.orbit import migrate_sky_convention

jax.config.update("jax_enable_x64", True)


def _elements(result):
    """Physical-element dict of a Thiele-Innes fit result."""
    return dict(
        T=result.T,
        e=result.e,
        a=result.a,
        cos_i=result.cos_i,
        W=result.W,
        cos_w=result.cos_w,
        sin_w=result.sin_w,
        tp=result.tp,
    )


def _old_sky(times, elements, dist_pc):
    """The earlier model: the same track with the two offsets in the other order."""
    east_north = sky_position(
        times, dist_pc=dist_pc, **{k: np.asarray(v) for k, v in elements.items()}
    )
    return east_north[::-1]


def test_migrated_elements_reproduce_the_old_track():
    """The current model at migrated elements equals the old model's output.

    Over random orientations: sky_position(migrated) == reversed sky_position(old).
    """
    rng = np.random.default_rng(3)
    times = np.linspace(0.0, 4000.0, 17)
    for _ in range(25):
        w = rng.uniform(0, 2 * np.pi)
        old = dict(
            T=2500.0,
            e=rng.uniform(0.0, 0.8),
            a=3.0,
            cos_i=rng.uniform(-0.95, 0.95),
            W=rng.uniform(0, 2 * np.pi),
            cos_w=np.cos(w),
            sin_w=np.sin(w),
            tp=rng.uniform(0, 2500.0),
        )
        migrated = migrate_sky_convention(old)
        np.testing.assert_allclose(
            sky_position(times, dist_pc=10.0, **migrated),
            _old_sky(times, old, 10.0),
            atol=1e-12,
            rtol=0,
        )
        for key in ("T", "e", "a", "cos_w", "sin_w", "tp"):
            assert migrated[key] == old[key]
        assert 0.0 <= migrated["W"] < 2 * np.pi


def test_migration_is_an_involution_and_accepts_arrays():
    """Applying it twice returns the input; scalars, numpy and jax arrays all work."""
    cos_i = jnp.linspace(-0.9, 0.9, 7)
    W = jnp.linspace(0.1, 6.0, 7)
    samples = dict(cos_i=cos_i, W=W, a=jnp.ones(7))
    twice = migrate_sky_convention(migrate_sky_convention(samples))
    np.testing.assert_allclose(twice["cos_i"], cos_i, atol=1e-15)
    np.testing.assert_allclose(twice["W"], W, atol=1e-12)
    scalar = migrate_sky_convention(dict(cos_i=0.25, W=0.0))
    assert scalar["cos_i"] == pytest.approx(-0.25)
    assert scalar["W"] == pytest.approx(np.pi / 2)
    assert samples["W"] is W  # the input mapping is not modified


@pytest.mark.parametrize(
    "samples",
    [
        dict(cos_i=0.1, W=1.0, log_P=3.0, M0=0.5),
        dict(cos_i=0.1, W=1.0, e_raw=0.2, w_raw=1.0),
        dict(cos_i=0.1, W=1.0, cos_i_base=0.3),
        dict(e=0.1, lnP=7.0, tau=0.2, east_X=1.0, east_Y=0.0, north_X=0.0, north_Y=1.0),
        dict(e=0.1, lnP=7.0, tau=0.2, A=1.0, F=0.0, B=0.0, G=1.0),
        dict(cos_i=3.2, W=1.0, a=1.0),
        dict(cos_i=0.2, W=-4.0, a=1.0),
        dict(a=1.0),
    ],
)
def test_transformed_or_incomplete_samples_raise(samples):
    """Posteriors in sampling coordinates need a refit, not a conversion."""
    with pytest.raises(ValueError):
        migrate_sky_convention(samples)


def test_non_mapping_raises():
    """Only a mapping of named elements is accepted."""
    with pytest.raises(TypeError):
        migrate_sky_convention(np.zeros((5, 2)))


def test_k16_old_fit_migrates_to_the_new_fit(k16):  # noqa: F811
    """A fit under the old output, migrated, matches a fit under the new one.

    Data are the Kervella et al. (2016), A&A 594, A107 (K16) alpha Cen AB track
    with Gaussian noise. The old fit minimizes chi2 of the (north, east) output
    against the observed (east, north) points, which is the current fitter run on
    the observations with their two columns (and errors) swapped. The new fit
    uses the observations directly. The two are independent solves; the migrated
    old elements must match the new ones within the fit uncertainty, which is
    measured over repeated noise draws.
    """
    times = np.linspace(0, k16["T"] * 1.4, 32)
    truth = reference(times, k16)
    sigma = np.array([1e-3, 2e-3])  # arcsec in (east, north)
    rng = np.random.default_rng(16)

    def noisy():
        return truth + rng.normal(size=truth.shape) * sigma[:, None]

    observed = noisy()
    new = fit(astrometry(times, observed, sigma[0], sigma[1]), k16)
    old = fit(astrometry(times, observed[::-1], sigma[1], sigma[0]), k16)
    migrated = migrate_sky_convention(_elements(old))

    # The migrated and directly fitted orbits trace the same sky track, whichever
    # (W + pi, omega + pi) branch each solve returned.
    def track(elements):
        fixed = {k: float(v) for k, v in elements.items() if k not in ("T", "tp")}
        return reference(times, dict(k16, **fixed))

    np.testing.assert_allclose(
        track(migrated), track(_elements(new)), atol=1e-9, rtol=0
    )

    # Fit uncertainty: scatter of the direct fit over repeated noise draws.
    draws = np.array(
        [
            [float(r.cos_i), float(r.W)]
            for r in (
                fit(astrometry(times, noisy(), sigma[0], sigma[1]), k16)
                for _ in range(40)
            )
        ]
    )
    std_cos_i, std_W = draws.std(axis=0)

    d_cos_i = abs(float(migrated["cos_i"]) - float(new.cos_i))
    d_W = min(
        abs((float(migrated["W"]) - float(new.W) + shift + np.pi) % (2 * np.pi) - np.pi)
        for shift in (0.0, np.pi)
    )
    assert d_cos_i < std_cos_i, (d_cos_i, std_cos_i)
    assert d_W < std_W, (d_W, std_W)
    # Without the migration the old elements are the mirror image of the new ones.
    assert abs(float(old.cos_i) - float(new.cos_i)) > 5 * std_cos_i
    # Both recover the published K16 inclination, not its mirror image.
    assert abs(float(migrated["cos_i"]) - k16["cos_i"]) < 5 * std_cos_i
