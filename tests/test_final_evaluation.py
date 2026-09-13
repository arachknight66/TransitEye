from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transiteye.acquisition.checksums import sha256_file
from transiteye.evaluation.calibration import (
    calibration_diagnostics,
    scientific_score_support,
)
from transiteye.evaluation.final_artifacts import (
    FinalEvaluationIdentity,
    make_final_evaluation_version,
)
from transiteye.evaluation.final_builder import _end_to_end, run_final_evaluation
from transiteye.evaluation.interpretability import shift_importance_risk
from transiteye.evaluation.scorecards import TRIAGE_VOCABULARY, scientific_triage

FINAL_VERSION = "final-evaluation-a4d1c707d285fa72a87f"
MODEL_VERSION = "model-4a58f313bc50b9c3dc41"


def test_calibration_metrics_and_bins_are_deterministic() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.9])
    first = calibration_diagnostics(labels, scores, bins=2)
    second = calibration_diagnostics(labels, scores, bins=2)
    assert first == second
    assert first["brier_score"] == pytest.approx(0.055)
    expected_log_loss = -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
    assert first["log_loss"] == pytest.approx(expected_log_loss)
    assert first["expected_calibration_error"] == pytest.approx(0.2)
    assert [row["count"] for row in first["reliability_bins"]] == [2, 2]


def test_scientific_support_is_not_a_calibration_metric() -> None:
    result = scientific_score_support(
        np.array([0.2, 0.8]), np.array([0.1, 0.3]), np.array([0.7, 0.9])
    )
    assert result["labels_used"] is False
    assert result["calibration_metrics_computed"] is False
    assert not {"brier_score", "log_loss", "expected_calibration_error"}.intersection(result)
    assert result["fractions"]["overlapping_demo_negative_support"] == 0.5
    assert result["fractions"]["overlapping_demo_positive_support"] == 0.5


def test_shifted_important_ranking_and_metrics_are_preserved() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "feature_group": ["bls", "fft"],
            "permutation_importance_mean": [0.4, 0.1],
            "permutation_importance_std": [0.01, 0.02],
        }
    )
    shift = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "standardized_median_difference": [2.0, 0.0],
            "outside_robust_support_fraction": [0.5, 0.0],
            "missingness_difference": [0.2, -0.1],
        }
    )
    first = shift_importance_risk(importance, shift)
    second = shift_importance_risk(importance, shift)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["feature"] == "a"
    assert first.iloc[0]["risk_flag"] == "high"
    assert first.iloc[0]["missingness_difference"] == pytest.approx(0.2)
    assert first.iloc[0]["outside_robust_support_fraction"] == pytest.approx(0.5)


def test_scientific_triage_uses_only_frozen_score_and_support() -> None:
    assert scientific_triage(0.9, 0.325, 1, 0) == "out_of_support"
    assert scientific_triage(0.9, 0.325, 0, 10) == "insufficient_support"
    assert scientific_triage(0.9, 0.325, 0, 0) == "model_high_score"
    assert scientific_triage(0.2, 0.325, 0, 0) == "model_mid_score"
    assert scientific_triage(0.1, 0.325, 0, 0) == "model_low_score"
    assert {
        scientific_triage(0.9, 0.325, 1, 0),
        scientific_triage(0.9, 0.325, 0, 10),
        scientific_triage(0.9, 0.325, 0, 0),
        scientific_triage(0.2, 0.325, 0, 0),
        scientific_triage(0.1, 0.325, 0, 0),
    } == TRIAGE_VOCABULARY


def test_end_to_end_keeps_recovery_and_classification_stages_distinct() -> None:
    table = pd.DataFrame(
        {
            "variant_name": ["easy", "easy", "easy"],
            "bls_recovered": [True, True, False],
            "classifier_correct_conditional": [True, False, None],
        }
    )
    result = _end_to_end(table)["easy"]
    assert result["total_injected_events"] == 3
    assert result["bls_recovered"] == 2
    assert result["classifier_correct_after_recovery"] == 1
    assert result["bls_recovery_rate"] == pytest.approx(2 / 3)
    assert result["conditional_classifier_correctness"] == pytest.approx(0.5)
    assert result["end_to_end_success_rate"] == pytest.approx(1 / 3)


def test_final_identity_is_portable_and_policy_sensitive(tmp_path: Path) -> None:
    values = ("m", "r", "df", "sf", "dd", "sd", "c", "i", "s", "ready")
    identity = FinalEvaluationIdentity(*values)
    assert make_final_evaluation_version(identity) == make_final_evaluation_version(identity)
    assert str(tmp_path) not in make_final_evaluation_version(identity)
    changed = FinalEvaluationIdentity(*values[:-1], "changed")
    assert make_final_evaluation_version(identity) != make_final_evaluation_version(changed)


def test_real_final_evaluation_replays_and_preserves_all_boundaries() -> None:
    model_directory = Path("data/models") / MODEL_VERSION
    before = {path.name: sha256_file(path) for path in model_directory.iterdir() if path.is_file()}
    result = run_final_evaluation(".")
    assert result.final_evaluation_version == FINAL_VERSION
    assert before == {
        path.name: sha256_file(path) for path in model_directory.iterdir() if path.is_file()
    }
    metadata = json.loads(
        (result.directory / "evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["official_model_modified"] is False
    assert metadata["fixed_threshold"] == 0.325
    calibration = json.loads((result.directory / "calibration.json").read_text())
    assert calibration["purpose"] == "diagnostic_only_no_calibration_transform_fitted"
    assert calibration["locked_test_role"] == "posthoc_descriptive_only"
    assert calibration["scientific_score_transfer"]["calibration_metrics_computed"] is False
    importance = pd.read_parquet(result.directory / "global_importance.parquet")
    assert importance["feature"].is_unique
    local = pd.read_parquet(result.directory / "local_explanations.parquet")
    assert local["candidate_id"].notna().all()
    assert set(local["example_role"]) == {
        "representative_demo_true_positive",
        "representative_demo_true_negative",
        "development_false_positive",
        "development_false_negative",
        "high_scoring_scientific_candidate",
        "lower_scoring_scientific_candidate",
    }
    forbidden = {"gold_candidate_label", "truth_source", "matched_event_id"}
    assert not forbidden.intersection(importance.columns)
    assert not forbidden.intersection(local.columns)
    scorecards = pd.read_parquet(result.directory / "candidate_scorecards.parquet")
    assert len(scorecards) == 450
    assert scorecards.groupby(["dataset_role", "candidate_id"]).size().eq(1).all()
    scientific = scorecards.loc[scorecards["dataset_role"] == "scientific_unlabeled"]
    assert len(scientific) == 75
    assert not forbidden.intersection(scorecards.columns)
    assert scientific["frozen_threshold"].eq(0.325).all()
    assert set(scientific["triage_status"]).issubset(TRIAGE_VOCABULARY)
    assert scientific["source_raw_checksum"].notna().all()
    benchmark = json.loads((result.directory / "benchmark_summary.json").read_text())
    frozen_metrics = json.loads((model_directory / "metrics.json").read_text())
    assert (
        benchmark["official_development_demo_modeling"]["official_locked_test"]
        == frozen_metrics["selected_test"]
    )
    assert benchmark["metric_replacement_or_cherry_picking"] is False
    assert benchmark["bls_and_classifier_stages_kept_distinct"] is True
    readiness = json.loads((result.directory / "scientific_readiness.json").read_text())
    assert readiness["demo_pipeline_status"] == "complete"
    assert readiness["scientific_readiness"] == "blocked"
    assert "independently_labeled_scientific_candidates_exist" in readiness["failed_criteria"]
    assert all("passed" in criterion for criterion in readiness["criteria"])
    checksums = json.loads((result.directory / "checksums.json").read_text())
    assert checksums["validation"] == "sha256_verified_before_analysis"
    assert any(MODEL_VERSION in path for path in checksums["upstream_artifacts"])
