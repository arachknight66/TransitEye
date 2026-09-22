from __future__ import annotations

import numpy as np
import pytest

from transiteye.domain.models import (
    AnalysisResult,
    AnalysisStatus,
    Candidate,
    Classification,
    MorphologyClass,
    Provenance,
    SourceAssessment,
    as_jsonable,
)


def test_complete_result_serializes_to_jsonable_data() -> None:
    provenance = Provenance(
        "file:///fixture.fits",
        "/fixture.fits",
        "abc",
        "tess_light_curve",
        "1",
        1,
        "TDB",
        "2457000.0",
    )
    candidate = Candidate("candidate-1", 2.0, 1000.0, 2.0, 500.0, 3, 10.0, 0.01)
    classification = Classification(
        {MorphologyClass.TRANSIT_LIKE: 1.0},
        MorphologyClass.TRANSIT_LIKE,
        SourceAssessment.UNRESOLVED,
        False,
        "model-1",
    )
    result = AnalysisResult(
        "analysis-1",
        AnalysisStatus.COMPLETE,
        provenance,
        (candidate,),
        {candidate.candidate_id: classification},
    )
    assert (
        as_jsonable(result)["classifications"]["candidate-1"]["predicted_class"] == "transit_like"
    )


def test_classification_rejects_probabilities_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        Classification(
            {MorphologyClass.TRANSIT_LIKE: 0.2},
            MorphologyClass.TRANSIT_LIKE,
            SourceAssessment.UNRESOLVED,
            False,
            "model-1",
        )


def test_analysis_rejects_classification_for_unknown_candidate() -> None:
    provenance = Provenance(
        "file:///fixture.fits", None, None, "tess_light_curve", None, None, None, None
    )
    classification = Classification(
        {MorphologyClass.TRANSIT_LIKE: 1.0},
        MorphologyClass.TRANSIT_LIKE,
        SourceAssessment.UNRESOLVED,
        False,
        "model-1",
    )
    with pytest.raises(ValueError, match="reference returned candidates"):
        AnalysisResult(
            "analysis-1", AnalysisStatus.COMPLETE, provenance, (), {"missing": classification}
        )


def test_light_curve_array_contract_is_length_safe() -> None:
    from transiteye.domain.models import LightCurve

    provenance = Provenance(
        "file:///fixture.fits", None, None, "tess_light_curve", None, None, None, None
    )
    with pytest.raises(ValueError, match="equal length"):
        LightCurve(
            np.array([1.0]),
            np.array([1.0, 2.0]),
            None,
            None,
            None,
            None,
            None,
            "PDCSAP_FLUX",
            None,
            provenance,
        )
