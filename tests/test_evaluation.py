from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import load_config
from transiteye.evaluation.artifacts import EvaluationIdentity, make_evaluation_version
from transiteye.evaluation.builder import run_robustness_evaluation
from transiteye.evaluation.injection import (
    add_observability_diagnostics,
    injection_recovery_table,
    summarize_injection_recovery,
)
from transiteye.evaluation.metrics import safe_binary_metrics
from transiteye.evaluation.robustness import leave_one_tic_out
from transiteye.evaluation.shift import feature_shift_table, score_summary
from transiteye.evaluation.uncertainty import group_bootstrap_intervals


def _grouped_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index in range(8):
        split = "test" if group_index >= 6 else ("validation" if group_index >= 4 else "train")
        for label in ("negative", "positive"):
            rows.append(
                {
                    "candidate_id": f"c-{group_index}-{label}",
                    "object_id": f"tic-{group_index}",
                    "gold_candidate_label": label,
                    "split": split,
                    "f1": float(group_index) + (label == "positive"),
                    "f2": float(label == "positive"),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_loto_is_deterministic_and_excludes_official_test() -> None:
    config = load_config("configs/modeling/demo_mvp.yaml")
    assert config.modeling is not None
    rows = _grouped_rows()
    kwargs = {
        "modeling": config.modeling,
        "master_seed": 42,
        "fixed_threshold": 0.325,
        "excluded_tics": ("tic-6", "tic-7"),
    }
    first = leave_one_tic_out(rows, ("f1", "f2"), **kwargs)
    second = leave_one_tic_out(rows, ("f1", "f2"), **kwargs)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert set(first.development_tics) == {f"tic-{value}" for value in range(6)}
    assert not set(first.predictions["object_id"]).intersection({"tic-6", "tic-7"})
    assert first.predictions.groupby("candidate_id").size().eq(1).all()


def test_grouped_loto_rejects_wrong_test_declaration() -> None:
    config = load_config("configs/modeling/demo_mvp.yaml")
    assert config.modeling is not None
    with pytest.raises(ValueError, match="excluded TICs"):
        leave_one_tic_out(
            _grouped_rows(),
            ("f1", "f2"),
            modeling=config.modeling,
            master_seed=42,
            fixed_threshold=0.325,
            excluded_tics=("tic-7",),
        )


def test_single_class_group_metrics_are_explicitly_undefined() -> None:
    result = safe_binary_metrics(np.array([1, 1]), np.array([0.8, 0.2]), 0.5)
    assert result["pr_auc"] is None
    assert result["roc_auc"] is None
    assert result["f1"] is None
    assert (result["tp"], result["fn"]) == (1, 1)


def test_group_bootstrap_is_deterministic_and_handles_invalid_replicates() -> None:
    predictions = pd.DataFrame(
        {
            "object_id": ["tic-a", "tic-a", "tic-b", "tic-b"],
            "label": [1, 1, 0, 0],
            "score": [0.9, 0.8, 0.2, 0.1],
        }
    )
    kwargs = dict(threshold=0.5, replicates=100, confidence_level=0.95, seed=7)
    first = group_bootstrap_intervals(predictions, **kwargs)
    assert first == group_bootstrap_intervals(predictions, **kwargs)
    assert 0 < first["valid_replicates"] < first["requested_replicates"]


def test_injection_recovery_keeps_detection_and_classification_separate() -> None:
    events = pd.DataFrame(
        {
            "synthetic_event_id": ["e1", "e2"],
            "variant_id": ["v1", "v2"],
            "object_id": ["tic-a", "tic-a"],
            "variant_name": ["easy", "weak"],
            "synthetic_event_class": ["planet_like", "planet_like"],
            "difficulty": ["easy", "weak"],
            "demo_gold_class": ["positive", "positive"],
            "injected_period_days": [2.0, 5.0],
            "injected_epoch": [0.25, 0.4],
            "injected_duration_days": [0.1, 0.2],
            "injected_depth_or_amplitude": [0.02, 0.003],
        }
    )
    candidates = pd.DataFrame(
        {"candidate_id": ["c1"], "candidate_rank": [1], "observation_group_id": ["g1"]}
    )
    matches = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "synthetic_event_id": ["e1"],
            "matched": [True],
            "match_type": ["fundamental"],
        }
    )
    predictions = pd.DataFrame({"candidate_id": ["c1"], "score": [0.9], "prediction": [1]})
    result = injection_recovery_table(
        events, candidates, matches, predictions, development_tics=("tic-a",)
    )
    assert result["bls_recovered"].tolist() == [True, False]
    assert result["classifier_correct_conditional"].tolist() == [True, None]
    summary = summarize_injection_recovery(result)
    assert summary["weak"]["classifier_accuracy_conditional_on_recovery"] is None
    observed = add_observability_diagnostics(
        result, {"v1": np.arange(0.0, 6.0, 0.1), "v2": np.arange(0.0, 6.0, 0.1)}
    )
    assert observed.loc[0, "observable_event_windows"] == 3
    assert observed.loc[0, "epoch_nearest_cadence_offset"] == pytest.approx(0.5)


def test_shift_and_support_statistics_need_no_scientific_labels() -> None:
    demo = pd.DataFrame({"f1": [0.0, 1.0, 2.0, np.nan], "f2": [1.0] * 4})
    science = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "object_id": ["tic-a", "tic-b"],
            "f1": [3.0, np.nan],
            "f2": [1.0, 1.0],
        }
    )
    shift, support = feature_shift_table(demo, science, ("f1", "f2"), iqr_multiplier=1.5)
    assert shift.set_index("feature").loc["f1", "missingness_difference"] == pytest.approx(0.25)
    assert support.loc[0, "outside_train_range_count"] == 1
    assert "gold_candidate_label" not in support
    assert score_summary(pd.Series([0.1, 0.9]))["median"] == pytest.approx(0.5)


def test_evaluation_identity_is_portable_and_policy_sensitive(tmp_path: Path) -> None:
    identity = EvaluationIdentity("m", "d", "f", "sd", "sf", "split", "cfg", "grid")
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    assert make_evaluation_version(identity) == make_evaluation_version(identity)
    changed = EvaluationIdentity("m", "d", "f", "sd", "sf", "split", "changed", "grid")
    assert make_evaluation_version(identity) != make_evaluation_version(changed)


def test_real_frozen_evaluation_replays_and_preserves_official_contract() -> None:
    model_directory = Path("data/models/model-4a58f313bc50b9c3dc41")
    model_checksums = {
        path.name: sha256_file(path) for path in model_directory.iterdir() if path.is_file()
    }
    first = run_robustness_evaluation(".")
    metadata_path = first.directory / "evaluation_metadata.json"
    before = json.loads(metadata_path.read_text(encoding="utf-8"))
    second = run_robustness_evaluation(".")
    after = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert first.evaluation_version == second.evaluation_version
    assert before == after
    assert before["official_model_modified"] is False
    assert first.group_robustness.development_tics == second.group_robustness.development_tics
    assert model_checksums == {
        path.name: sha256_file(path) for path in model_directory.iterdir() if path.is_file()
    }
    stability = pd.read_parquet(first.directory / "feature_stability.parquet")
    assert {
        "remove_lomb_scargle",
        "remove_fft",
        "remove_frequency",
        "time_only",
        "bls_only",
        "candidate_period_-0.500%",
        "candidate_period_+0.500%",
    }.issubset(stability["variant"])
    injection = pd.read_parquet(first.directory / "injection_recovery.parquet")
    assert injection["observable_event_windows"].notna().all()
    shift_summary = json.loads((first.directory / "shift_summary.json").read_text(encoding="utf-8"))
    assert shift_summary["scientific_rows"] == 75
    assert shift_summary["scientific_predicted_positive"] == 75
    assert not {"accuracy", "precision", "recall", "f1"}.intersection(shift_summary)
