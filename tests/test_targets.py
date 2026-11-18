"""Target TOML routing, conversions, and composition against a system."""

import ineedvalidation as vv
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hwoutils.spectral import Band, Transmission
from test_terms import _ELEMENTS, _system

from photomancy import TermLogDensity, load_target

_EPOCH = Epoch(2451545.0)

_TOML = """
[target]
name = "X"
dist_pc = 10.0
star_mass_msun = 1.0
[[pixels]]
data_id = "img-1"
[[astrometry]]
data_id = "ground-1"
use = "term"
jd = 2460000.0
sep_mas = 300.0
sep_err_mas = [6.0, 8.0]
pa_deg = 90.0
pa_err_deg = [1.0, 1.4]
[[astrometry]]
data_id = "img-1"
use = "comparison"
cite = "Public measurement"
[[astrometry]]
data_id = "future"
use = "pending"
jd = 2460100.0
sep_mas = 320.0
sep_err_mas = 9.0
pa_deg = 80.0
pa_err_deg = 1.0
[[photometry]]
data_id = "ground-1"
use = "term"
jd = 2460000.0
filter = "L'"
dmag = 9.9
dmag_err = [0.2, 0.3]
[[photometry]]
data_id = "future"
use = "pending"
jd = 2460100.0
filter = "H"
dmag = 10.0
dmag_err = 0.2
upper_limit = true
planet = 0
[unrecognized]
ignored = true
"""


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_target_routing_and_composition(tmp_path):
    """Truth source: explicit TOML rows, polar conversion and separate term sums.

    Tolerance basis: 1e-14 relative for x64 unit/error propagation and summation;
    raw rows, counts and routing are exact. Fails if: pending/comparison rows
    enter the fit, errors are not averaged, mas are treated as arcsec, filter
    order is lost, or the loaded terms cannot score the selected system.
    """
    path = tmp_path / "target.toml"
    path.write_text(_TOML)
    bands = {
        "L'": Transmission.from_band(Band(3450.0, 3550.0)),
        "H": Transmission.from_band(Band(1550.0, 1650.0)),
    }
    loaded = load_target(path, epoch=_EPOCH, pixel_data_ids=("img-1",))
    assert loaded.target["name"] == "X"
    assert loaded.band_names == ("L'",)
    assert loaded.relative_astrometry.times.size == 1
    np.testing.assert_allclose(loaded.relative_astrometry.ra, [0.3], rtol=1e-14)
    np.testing.assert_allclose(loaded.relative_astrometry.ra_err, [0.007], rtol=1e-14)
    np.testing.assert_allclose(
        loaded.relative_astrometry.dec_err, [0.3 * np.deg2rad(1.2)], rtol=1e-14
    )
    contrast = 10 ** (-0.4 * 9.9)
    np.testing.assert_allclose(loaded.photometry.contrast, [contrast], rtol=1e-14)
    np.testing.assert_allclose(
        loaded.photometry.contrast_err, [0.4 * np.log(10) * contrast * 0.25], rtol=1e-14
    )
    assert [r["use"] for r in loaded.comparisons["astrometry"]] == [
        "comparison",
        "pending",
    ]
    assert loaded.comparisons["astrometry"][0]["cite"] == "Public measurement"
    assert loaded.comparisons["photometry"][0]["use"] == "pending"
    with pytest.raises(ValueError, match="L'"):
        loaded.terms()
    with pytest.raises(ValueError, match="L'"):
        loaded.terms(bandpasses={"H": bands["H"]})
    model = {"system": _system()}
    terms = loaded.terms(
        system=model["system"], select=lambda m: m["system"], bandpasses=bands
    )
    expected = sum(term(model)[0] for term in terms)
    np.testing.assert_allclose(TermLogDensity(terms)(model), expected, rtol=1e-14)
    pending = load_target(path, epoch=_EPOCH, include_pending=True, bandpasses=bands)
    assert pending.band_names == ("L'", "H")
    assert pending.relative_astrometry.times.size == 2
    np.testing.assert_array_equal(pending.photometry.is_upper_limit, [False, True])
    assert len(pending.terms(system=model["system"])) == 2
    assert len(pending.comparisons["astrometry"]) == 1
    assert pending.comparisons["photometry"] == []


@vv.case("scene-fits", "code-verification", basis="exact-algebra")
def test_target_guards_and_empty(tmp_path):
    """Truth source: explicit provenance IDs and the three allowed use labels.

    Tolerance basis: exact routing and exceptions. Fails if: either observable
    double-counts pixel data, pending bypasses the guard when enabled, an unknown
    use is accepted, or empty files manufacture likelihood terms.
    """
    path = tmp_path / "target.toml"
    path.write_text(_TOML)
    with pytest.raises(ValueError, match=r"astrometry.*ground-1"):
        load_target(path, epoch=_EPOCH, pixel_data_ids=("ground-1",))
    with pytest.raises(ValueError, match=r"astrometry.*future"):
        load_target(
            path, epoch=_EPOCH, pixel_data_ids=("future",), include_pending=True
        )
    load_target(path, epoch=_EPOCH, pixel_data_ids=("future",))
    path.write_text(_TOML.replace('use = "term"', 'use = "comparison"', 1))
    with pytest.raises(ValueError, match=r"photometry.*ground-1"):
        load_target(path, epoch=_EPOCH, pixel_data_ids=("ground-1",))
    path.write_text(_TOML.replace('use = "term"', 'use = "typo"', 1))
    with pytest.raises(ValueError, match="typo"):
        load_target(path, epoch=_EPOCH)
    path.write_text('[target]\nname = "empty"\n[unknown]\nx = 1\n')
    empty = load_target(path, epoch=_EPOCH)
    assert empty.relative_astrometry is None
    assert empty.photometry is None
    assert empty.band_names == ()
    assert empty.terms() == ()


@vv.case("scene-fits", "code-verification", basis="equivalence")
@pytest.mark.parametrize("include_pending", [False, True])
def test_target_epoch_invariance(tmp_path, include_pending):
    """Fails if dates reach a device before subtraction or terms mix time origins.

    Two split epochs differ by exactly 100.25 days. Time differences are exact;
    likelihoods allow 1e-10 relative for x64 propagation with shifted references.
    Pending rows exercise a second date and photometric band when enabled.
    """
    path = tmp_path / "target.toml"
    path.write_text(_TOML)
    bands = {
        "L'": Transmission.from_band(Band(3450.0, 3550.0)),
        "H": Transmission.from_band(Band(1550.0, 1650.0)),
    }
    epochs = (Epoch(2460000.0, 0.125), Epoch(2460100.0, 0.375))
    loaded = [
        load_target(path, epoch=e, include_pending=include_pending, bandpasses=bands)
        for e in epochs
    ]
    for name in ("relative_astrometry", "photometry"):
        times = [np.asarray(getattr(data, name).times) for data in loaded]
        np.testing.assert_array_equal(times[0] - times[1], 100.25)
        dates = np.array([2460000.0, 2460100.0][: times[0].size], dtype=np.float64)
        np.testing.assert_array_equal(times[0], epochs[0].days(dates))
    values = []
    for data in loaded:
        elements = _ELEMENTS.copy()
        elements[-1] += epochs[0].days_between(data.epoch)
        system = _system(elements, epoch=data.epoch)
        assert system.epoch is data.epoch
        values.append([term(system)[0] for term in data.terms(system=system)])
    np.testing.assert_allclose(values[0], values[1], rtol=1e-10, atol=0)
