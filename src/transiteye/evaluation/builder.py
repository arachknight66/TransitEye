"""Orchestrate and freeze B046--B051 diagnostics without changing the model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import config_hash, load_config
from transiteye.evaluation.artifacts import (
    EvaluationIdentity,
    freeze_evaluation,
    make_evaluation_version,
)
from transiteye.evaluation.errors import error_analysis_table
from transiteye.evaluation.injection import (
    add_observability_diagnostics,
    injection_recovery_table,
    summarize_injection_recovery,
)
from transiteye.evaluation.robustness import GroupRobustnessResult, leave_one_tic_out
from transiteye.evaluation.shift import feature_shift_table, score_summary
from transiteye.evaluation.stability import feature_ablation_stability
from transiteye.evaluation.uncertainty import group_bootstrap_intervals
from transiteye.modeling.artifacts import load_frozen_model
from transiteye.modeling.inference import predict_candidates
from transiteye.provenance import derive_seed

DEMO_DATASET = "dataset-demo-a74b6aad7c21faf15b0d"
DEMO_FEATURES = "features-6278a328b8f624b2759b"
SCIENTIFIC_DATASET = "dataset-4b84e8acaa6f6b334c2f"
SCIENTIFIC_FEATURES = "features-4e9f6d54b84780a2be42"


@dataclass(frozen=True)
class EvaluationRunResult:
    directory: Path
    evaluation_version: str
    group_robustness: GroupRobustnessResult
    uncertainty: dict[str, Any]
    injection_summary: dict[str, object]
    score_distributions: dict[str, dict[str, float]]


def _load_features(root: Path, dataset: str, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = root / "data/features" / dataset / version
    metadata = json.loads((directory / "feature_metadata.json").read_text(encoding="utf-8"))
    for filename, expected in metadata["table_checksums"].items():
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"Feature checksum mismatch: {filename}")
    return pd.read_parquet(directory / "features.parquet"), pd.read_parquet(
        directory / "labels.parquet"
    )


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], json.loads(frame.to_json(orient="records")))


def run_robustness_evaluation(root: str | Path) -> EvaluationRunResult:
    """Run the frozen robustness protocol and verify exact replay when it exists."""
    repository = Path(root)
    evaluation_config = load_config(repository / "configs/evaluation/robustness_mvp.yaml")
    modeling_config = load_config(repository / "configs/modeling/demo_mvp.yaml")
    settings = evaluation_config.evaluation
    modeling = modeling_config.modeling
    if settings is None or modeling is None:
        raise ValueError("Evaluation and modeling configurations are required.")
    model_directory = repository / "data/models" / settings.model_version
    model_metadata = json.loads(
        (model_directory / "model_metadata.json").read_text(encoding="utf-8")
    )
    frozen_model = load_frozen_model(model_directory)
    if frozen_model.threshold != settings.fixed_threshold:
        raise ValueError("Evaluation threshold differs from the frozen official threshold.")

    demo_features, demo_labels = _load_features(repository, DEMO_DATASET, DEMO_FEATURES)
    scientific_features, scientific_labels = _load_features(
        repository, SCIENTIFIC_DATASET, SCIENTIFIC_FEATURES
    )
    if scientific_labels["gold_candidate_label"].isin(["positive", "negative"]).any():
        raise ValueError("Scientific rows unexpectedly contain gold binary labels.")
    split = pd.read_parquet(model_directory / "split_manifest.parquet")
    rows = demo_features.merge(demo_labels, on="candidate_id", validate="one_to_one").merge(
        split[["candidate_id", "split"]], on="candidate_id", validate="one_to_one"
    )
    official_test_tics = tuple(
        sorted(split.loc[split["split"] == "test", "object_id"].astype(str).unique())
    )
    grouped = leave_one_tic_out(
        rows,
        frozen_model.feature_names,
        modeling=modeling,
        master_seed=modeling_config.reproducibility.master_seed,
        fixed_threshold=settings.fixed_threshold,
        excluded_tics=official_test_tics,
    )
    uncertainty_seed = derive_seed(
        evaluation_config.reproducibility.master_seed, settings.bootstrap_seed_component
    )
    uncertainty = group_bootstrap_intervals(
        grouped.predictions,
        threshold=settings.fixed_threshold,
        replicates=settings.bootstrap_replicates,
        confidence_level=settings.confidence_level,
        seed=uncertainty_seed,
    )

    demo_predictions = predict_candidates(demo_features, frozen_model).merge(
        demo_features[["candidate_id", "object_id"]], on="candidate_id", validate="one_to_one"
    )
    scientific_predictions = predict_candidates(scientific_features, frozen_model).merge(
        scientific_features[["candidate_id", "object_id"]],
        on="candidate_id",
        validate="one_to_one",
    )
    dataset_directory = repository / "data/datasets" / DEMO_DATASET
    candidates = pd.read_parquet(dataset_directory / "candidates.parquet")
    events = pd.read_parquet(dataset_directory / "synthetic_events.parquet")
    matches = pd.read_parquet(dataset_directory / "candidate_event_matches.parquet")
    injection = injection_recovery_table(
        events,
        candidates,
        matches,
        demo_predictions,
        development_tics=grouped.development_tics,
    )
    synthetic_root = repository / "data/synthetic" / settings.injection_grid_identity
    timestamps_by_variant = {
        variant_id: pd.read_parquet(
            synthetic_root / "variants" / variant_id / "processed.parquet",
            columns=["time", "valid"],
        )
        .loc[lambda frame: frame["valid"], "time"]
        .to_numpy(dtype=float)
        for variant_id in injection["variant_id"].astype(str)
    }
    injection = add_observability_diagnostics(injection, timestamps_by_variant)
    injection_summary = summarize_injection_recovery(injection)
    stability = feature_ablation_stability(
        rows,
        modeling=modeling,
        master_seed=modeling_config.reproducibility.master_seed,
        fixed_threshold=settings.fixed_threshold,
        official_model=frozen_model,
        period_perturbation_fraction=settings.candidate_period_perturbation_fraction,
    )

    demo_train_gold = rows.loc[
        (rows["split"] == "train") & rows["gold_candidate_label"].isin(["positive", "negative"])
    ]
    shift, support = feature_shift_table(
        demo_train_gold,
        scientific_features,
        frozen_model.feature_names,
        iqr_multiplier=settings.robust_support_iqr_multiplier,
    )
    severity = (
        shift["standardized_median_difference"].abs().fillna(0)
        + shift["outside_robust_support_fraction"]
    )
    shift["shift_severity"] = severity
    shift = shift.sort_values(["shift_severity", "feature"], ascending=[False, True]).reset_index(
        drop=True
    )

    scored_demo = demo_predictions.merge(
        rows[["candidate_id", "gold_candidate_label", "split"]],
        on="candidate_id",
        validate="one_to_one",
    )
    score_groups = {
        "demo_train_gold_positive": scored_demo.loc[
            (scored_demo["split"] == "train") & (scored_demo["gold_candidate_label"] == "positive"),
            "score",
        ],
        "demo_train_gold_negative": scored_demo.loc[
            (scored_demo["split"] == "train") & (scored_demo["gold_candidate_label"] == "negative"),
            "score",
        ],
        "demo_validation": scored_demo.loc[scored_demo["split"] == "validation", "score"],
        "demo_locked_test": scored_demo.loc[scored_demo["split"] == "test", "score"],
        "demo_unlabeled": scored_demo.loc[
            scored_demo["gold_candidate_label"] == "unlabeled", "score"
        ],
        "scientific_unlabeled": scientific_predictions["score"],
    }
    score_distributions = {name: score_summary(values) for name, values in score_groups.items()}

    loto_for_errors = grouped.predictions[["candidate_id", "object_id", "score", "prediction"]]
    validation_for_errors = scored_demo.loc[
        (scored_demo["split"] == "validation")
        & scored_demo["gold_candidate_label"].isin(["positive", "negative"]),
        ["candidate_id", "object_id", "score", "prediction"],
    ]
    test_for_errors = scored_demo.loc[
        (scored_demo["split"] == "test")
        & scored_demo["gold_candidate_label"].isin(["positive", "negative"]),
        ["candidate_id", "object_id", "score", "prediction"],
    ]
    errors = error_analysis_table(
        (
            ("development_loto", loto_for_errors),
            ("official_validation_descriptive", validation_for_errors),
            ("official_locked_test_posthoc_descriptive", test_for_errors),
        ),
        demo_features,
        candidates,
        events,
        high_missingness_count=settings.error_high_missingness_count,
    )

    identity = EvaluationIdentity(
        settings.model_version,
        DEMO_DATASET,
        DEMO_FEATURES,
        SCIENTIFIC_DATASET,
        SCIENTIFIC_FEATURES,
        str(model_metadata["identity_inputs"]["split_id"]),
        config_hash(evaluation_config),
        settings.injection_grid_identity,
    )
    group_payload = {
        "protocol": settings.development_resampling,
        "fixed_threshold": settings.fixed_threshold,
        "official_test_tics_excluded": list(official_test_tics),
        "development_tics": list(grouped.development_tics),
        "per_tic": list(grouped.per_tic_metrics),
        "pooled": grouped.pooled_metrics,
    }
    shift_summary = {
        "scientific_rows": int(len(scientific_predictions)),
        "scientific_predicted_positive": int(scientific_predictions["prediction"].sum()),
        "scientific_predicted_positive_fraction": float(
            scientific_predictions["prediction"].mean()
        ),
        "highest_shift_features": _json_safe_records(shift.head(10)),
        "domain_discriminator": {
            "implemented": False,
            "reason": (
                "Only eight paired source TICs make a separate grouped domain classifier "
                "misleading for this narrow phase."
            ),
        },
    }
    tables = {
        "group_robustness.parquet": grouped.predictions,
        "injection_recovery.parquet": injection,
        "feature_stability.parquet": stability,
        "feature_shift.parquet": shift,
        "scientific_out_of_support.parquet": support,
        "error_analysis.parquet": errors,
    }
    payloads = {
        "group_metrics.json": group_payload,
        "uncertainty.json": uncertainty,
        "injection_summary.json": injection_summary,
        "score_distributions.json": score_distributions,
        "shift_summary.json": shift_summary,
    }
    directory = freeze_evaluation(
        identity=identity, tables=tables, payloads=payloads, root=repository / "data/evaluation"
    )
    return EvaluationRunResult(
        directory,
        make_evaluation_version(identity),
        grouped,
        uncertainty,
        injection_summary,
        score_distributions,
    )
