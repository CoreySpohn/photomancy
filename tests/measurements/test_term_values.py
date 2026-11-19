"""Term values and predictions on the three golden systems, before and after."""

import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
from golden_systems import imaging_data, model_at, systems
from term_utils import null_dmag_term

from photomancy.measurements import DMagTerm

jax.config.update("jax_enable_x64", True)

STORED = json.loads(
    (
        Path(__file__).parent.parent / "data" / "measurement_terms_golden.json"
    ).read_text()
)
_N_POINTS = 6


@pytest.mark.parametrize("name", ["g1", "g2", "g3"])
def test_terms_equal_the_stored_values(name):
    """Fails if a term's arithmetic changed in the move to the measurements package.

    ``STORED`` was computed before the move, at the six golden chart points of each
    system. The same terms (data, kernels, path-resolved system) must return the same
    value and prediction. Both sides run the same float64 operations in the same
    order, so the bound is rounding: rtol 1e-12 on values of order 1e2 to 1e5.
    """
    scene, entry, terms = next(
        (scene, entry, terms) for n, scene, entry, terms in systems() if n == name
    )
    for i in range(_N_POINTS):
        model = model_at(scene, entry(i))
        for term in terms:
            value, prediction = STORED[name][i][type(term).__name__]
            np.testing.assert_allclose(float(term(model)), value, rtol=1e-12, atol=0)
            np.testing.assert_allclose(
                np.asarray(term.predict(model)), prediction, rtol=1e-12, atol=1e-300
            )


def test_null_epochs_are_the_old_null_term():
    """Fails if a DMagTerm of nulls differs from the deleted ``NullTerm``.

    ``NullTerm`` scored ``loglike_null`` on ``NullData``; an all-null ``ImagingData``
    on ``DMagTerm`` selects the same censored kernel on every row, so the deleted
    term is a special case. Stored ``NullTerm`` values are the reference, rtol 1e-12.
    """
    _, null = imaging_data()
    scene, entry, _ = next(
        (scene, entry, terms) for n, scene, entry, terms in systems() if n == "g3"
    )
    for i in range(_N_POINTS):
        model = model_at(scene, entry(i))
        value, prediction = STORED["g3"][i]["NullTerm"]
        term = null_dmag_term(null)
        np.testing.assert_allclose(float(term(model)), value, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            np.asarray(term.predict(model)), prediction, rtol=1e-12, atol=1e-300
        )


def test_a_dmag_term_with_a_null_epoch_refuses_to_draw():
    """Fails if a planned non-detection draws silently, leaving its EIG exactly zero.

    ``draw`` only redraws detected epochs, so an all-null term returned itself and both
    EIG estimators reported zero information; with mixed rows the predictive lacked the
    detection probability. Any valid null row refuses and names ``ReportTerm``; a term
    of detections still draws.
    """
    detections, null = imaging_data()
    scene, entry, _ = next(
        (scene, entry, terms) for n, scene, entry, terms in systems() if n == "g3"
    )
    model = model_at(scene, entry(0))
    key = jax.random.key(0)
    for term in (null_dmag_term(null), DMagTerm(detections)):
        with pytest.raises(Exception, match="ReportTerm"):
            term.draw(key, model)
    only = detections.is_detected | ~detections.is_valid
    kept = eqx.tree_at(lambda d: d.is_valid, detections, detections.is_valid & only)
    assert DMagTerm(kept).draw(key, model).data.is_detected.shape == only.shape
