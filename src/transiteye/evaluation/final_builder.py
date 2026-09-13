"""Build the immutable B052--B057 evaluation and interpretation package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import load_config
from transiteye.datasets.schemas import assert_model_input_boundary
from transiteye.evaluation.calibration import (
    calibration_diagnostics,
    group_bootstrap_calibration,
    scientific_score_support,
)
from transiteye.evaluation.final_artifacts import (
    FinalEvaluationIdentity,
    freeze_final_evaluation,
    make_final_evaluation_version,
    verify_declared_checksums,
)
from transiteye.evaluation.interpretability import (
    global_feature_importance,
    local_median_perturbation,
    shift_importance_risk,
)
from transiteye.evaluation.scorecards import build_scorecards
from transiteye.evaluation.shift import feature_shift_table
from transiteye.modeling.artifacts import load_frozen_model
from transiteye.modeling.inference import predict_candidates
from transiteye.provenance import derive_seed

DEMO_DATASET = "dataset-demo-a74b6aad7c21faf15b0d"
DEMO_FEATURES = "features-6278a328b8f624b2759b"
SCIENTIFIC_DATASET = "dataset-4b84e8acaa6f6b334c2f"
SCIENTIFIC_FEATURES = "features-4e9f6d54b84780a2be42"


@dataclass(frozen=True)
class FinalEvaluationRunResult:
    directory: Path
    final_evaluation_version: str
    calibration: dict[str, Any]
    readiness: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_features(
    repository: Path, dataset: str, version: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = repository / "data/features" / dataset / version
    verify_declared_checksums(directory, "feature_metadata.json")
    return pd.read_parquet(directory / "features.parquet"), pd.read_parquet(
        directory / "labels.parquet"
    )


def _binary_labels(values: pd.Series) -> np.ndarray:
    mapping = {"negative": 0, "positive": 1}
    if not values.isin(mapping).all():
        raise ValueError("Expected only gold binary demo labels.")
    return values.map(mapping).to_numpy(dtype=int)


def _per_tic_calibration(rows: pd.DataFrame, bins: int) -> list[dict[str, Any]]:
    summaries = []
    for object_id, group in rows.groupby("object_id", sort=True):
        result = calibration_diagnostics(
            group["label"].to_numpy(), group["score"].to_numpy(), bins=bins
        )
        summaries.append(
            {
                "object_id": str(object_id),
                "candidate_count": result["candidate_count"],
                "brier_score": result["brier_score"],
                "log_loss": result["log_loss"],
                "expected_calibration_error": result["expected_calibration_error"],
                "mean_predicted_score": result["mean_predicted_score"],
                "observed_positive_fraction": result["observed_positive_fraction"],
            }
        )
    return summaries


def _select_examples(
    demo_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    loto: pd.DataFrame,
    science_predictions: pd.DataFrame,
) -> list[dict[str, str]]:
    official = demo_predictions.merge(
        labels[["candidate_id", "gold_candidate_label"]], on="candidate_id"
    ).merge(split[["candidate_id", "split"]], on="candidate_id")
    validation = official.loc[
        (official["split"] == "validation")
        & official["gold_candidate_label"].isin(["positive", "negative"])
    ].copy()
    validation["expected"] = _binary_labels(validation["gold_candidate_label"])
    roles = []
    for example_role, expected, prediction in (
        ("representative_demo_true_positive", 1, 1),
        ("representative_demo_true_negative", 0, 0),
    ):
        candidates = validation.loc[
            (validation["expected"] == expected) & (validation["prediction"] == prediction)
        ]
        chosen = candidates.sort_values(
            ["score", "candidate_id"], ascending=[expected == 0, True], kind="stable"
        ).iloc[0]
        roles.append(
            {
                "candidate_id": str(chosen["candidate_id"]),
                "dataset_role": "demo",
                "example_role": example_role,
                "selection_source": "official_validation",
            }
        )
    for example_role, expected, prediction in (
        ("development_false_positive", 0, 1),
        ("development_false_negative", 1, 0),
    ):
        chosen = (
            loto.loc[(loto["label"] == expected) & (loto["prediction"] == prediction)]
            .sort_values("candidate_id", kind="stable")
            .iloc[0]
        )
        roles.append(
            {
                "candidate_id": str(chosen["candidate_id"]),
                "dataset_role": "demo",
                "example_role": example_role,
                "selection_source": "development_loto_truth_selection_only",
            }
        )
    ordered_science = science_predictions.sort_values(["score", "candidate_id"], kind="stable")
    roles.extend(
        [
            {
                "candidate_id": str(ordered_science.iloc[-1]["candidate_id"]),
                "dataset_role": "scientific_unlabeled",
                "example_role": "high_scoring_scientific_candidate",
                "selection_source": "frozen_model_score",
            },
            {
                "candidate_id": str(ordered_science.iloc[0]["candidate_id"]),
                "dataset_role": "scientific_unlabeled",
                "example_role": "lower_scoring_scientific_candidate",
                "selection_source": "frozen_model_score",
            },
        ]
    )
    return roles


def _end_to_end(injection: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, group in injection.groupby("variant_name", sort=True):
        recovered = group.loc[group["bls_recovered"]]
        correct = int(sum(value is True for value in recovered["classifier_correct_conditional"]))
        result[str(name)] = {
            "total_injected_events": int(len(group)),
            "bls_recovered": int(group["bls_recovered"].sum()),
            "classifier_correct_after_recovery": correct,
            "bls_recovery_rate": float(group["bls_recovered"].mean()),
            "conditional_classifier_correctness": float(
                recovered["classifier_correct_conditional"].mean()
            )
            if len(recovered)
            else None,
            "end_to_end_success_rate": float(correct / len(group)),
        }
    return result


def run_final_evaluation(root: str | Path) -> FinalEvaluationRunResult:
    """Run diagnostics only, validate immutable inputs, and freeze exact outputs."""
    repository = Path(root)
    config = load_config(repository / "configs/evaluation/final_mvp.yaml")
    settings = config.final_evaluation
    if settings is None:
        raise ValueError("Final evaluation configuration is required.")
    model_directory = repository / "data/models" / settings.model_version
    robustness_directory = repository / "data/evaluation" / settings.robustness_evaluation_version
    upstream: dict[str, str] = {}
    for directory, metadata in (
        (model_directory, "model_metadata.json"),
        (robustness_directory, "evaluation_metadata.json"),
        (repository / "data/features" / DEMO_DATASET / DEMO_FEATURES, "feature_metadata.json"),
        (
            repository / "data/features" / SCIENTIFIC_DATASET / SCIENTIFIC_FEATURES,
            "feature_metadata.json",
        ),
        (repository / "data/datasets" / DEMO_DATASET, "dataset_metadata.json"),
        (repository / "data/datasets" / SCIENTIFIC_DATASET, "dataset_metadata.json"),
    ):
        upstream.update(verify_declared_checksums(directory, metadata))
    upstream = {
        str(Path(path).relative_to(repository)): checksum for path, checksum in upstream.items()
    }
    model_checksums_before = {
        path.name: sha256_file(path) for path in model_directory.iterdir() if path.is_file()
    }
    model = load_frozen_model(model_directory)
    if model.threshold != settings.fixed_threshold or model.threshold != 0.325:
        raise ValueError("Frozen threshold must remain exactly 0.325.")
    assert_model_input_boundary(model.feature_names)
    demo_features, demo_labels = _load_features(repository, DEMO_DATASET, DEMO_FEATURES)
    science_features, science_labels = _load_features(
        repository, SCIENTIFIC_DATASET, SCIENTIFIC_FEATURES
    )
    if science_labels["gold_candidate_label"].isin(["positive", "negative"]).any():
        raise ValueError("Scientific labels were introduced.")
    split = pd.read_parquet(model_directory / "split_manifest.parquet")
    demo_rows = demo_features.merge(demo_labels, on="candidate_id", validate="one_to_one").merge(
        split[["candidate_id", "split"]], on="candidate_id", validate="one_to_one"
    )
    demo_predictions = predict_candidates(demo_features, model)
    science_predictions = predict_candidates(science_features, model)
    loto = pd.read_parquet(robustness_directory / "group_robustness.parquet")
    calibration: dict[str, Any] = {
        "purpose": "diagnostic_only_no_calibration_transform_fitted",
        "locked_test_role": "posthoc_descriptive_only",
        "candidate_level_caution": (
            "Candidates within a TIC are correlated; uncertainty is dominated by six "
            "independent development TIC groups."
        ),
    }
    loto_calibration = calibration_diagnostics(
        loto["label"].to_numpy(), loto["score"].to_numpy(), bins=settings.calibration_bins
    )
    calibration["development_oof"] = loto_calibration
    calibration["development_oof_per_tic"] = _per_tic_calibration(loto, settings.calibration_bins)
    calibration["development_oof_group_bootstrap"] = group_bootstrap_calibration(
        loto,
        bins=settings.calibration_bins,
        replicates=settings.calibration_bootstrap_replicates,
        seed=derive_seed(config.reproducibility.master_seed, "calibration_group_bootstrap"),
    )
    official_scored = demo_predictions.merge(
        demo_rows[["candidate_id", "object_id", "gold_candidate_label", "split"]],
        on="candidate_id",
        validate="one_to_one",
    )
    for split_name, output_name in (
        ("validation", "official_validation"),
        ("test", "official_locked_test_descriptive"),
    ):
        rows = official_scored.loc[
            (official_scored["split"] == split_name)
            & official_scored["gold_candidate_label"].isin(["positive", "negative"])
        ]
        calibration[output_name] = calibration_diagnostics(
            _binary_labels(rows["gold_candidate_label"]),
            rows["score"].to_numpy(),
            bins=settings.calibration_bins,
        )
    demo_negative = official_scored.loc[
        official_scored["gold_candidate_label"] == "negative", "score"
    ].to_numpy()
    demo_positive = official_scored.loc[
        official_scored["gold_candidate_label"] == "positive", "score"
    ].to_numpy()
    calibration["scientific_score_transfer"] = scientific_score_support(
        science_predictions["score"].to_numpy(), demo_negative, demo_positive
    )
    validation_rows = demo_rows.loc[
        (demo_rows["split"] == "validation")
        & demo_rows["gold_candidate_label"].isin(["positive", "negative"])
    ]
    importance, importance_summary = global_feature_importance(
        model,
        validation_rows,
        _binary_labels(validation_rows["gold_candidate_label"]),
        repeats=settings.permutation_repeats,
        seed=derive_seed(config.reproducibility.master_seed, "final_permutation_importance"),
    )
    examples = _select_examples(demo_predictions, demo_labels, split, loto, science_predictions)
    local = local_median_perturbation(
        model,
        pd.concat([demo_features, science_features], ignore_index=True),
        examples,
        top_k=settings.local_top_k,
    )
    shift = pd.read_parquet(robustness_directory / "feature_shift.parquet")
    risk = shift_importance_risk(importance, shift)
    demo_train_gold = demo_rows.loc[
        (demo_rows["split"] == "train")
        & demo_rows["gold_candidate_label"].isin(["positive", "negative"])
    ]
    _, demo_support = feature_shift_table(
        demo_train_gold, demo_features, model.feature_names, iqr_multiplier=1.5
    )
    science_support = pd.read_parquet(robustness_directory / "scientific_out_of_support.parquet")
    demo_candidates = pd.read_parquet(
        repository / "data/datasets" / DEMO_DATASET / "candidates.parquet"
    )
    science_candidates = pd.read_parquet(
        repository / "data/datasets" / SCIENTIFIC_DATASET / "candidates.parquet"
    )
    demo_scorecards = build_scorecards(
        demo_features,
        demo_candidates,
        demo_predictions,
        demo_support,
        local.loc[local["dataset_role"] == "demo"],
        dataset_role="demo",
        threshold=model.threshold,
    )
    science_scorecards = build_scorecards(
        science_features,
        science_candidates,
        science_predictions,
        science_support,
        local.loc[local["dataset_role"] == "scientific_unlabeled"],
        dataset_role="scientific_unlabeled",
        threshold=model.threshold,
    )
    scorecards = (
        pd.concat([demo_scorecards, science_scorecards], ignore_index=True)
        .sort_values(["dataset_role", "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )
    demo_evaluation = (
        demo_labels.merge(
            split[["candidate_id", "split"]], on="candidate_id", validate="one_to_one"
        )
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
    )
    injection = pd.read_parquet(robustness_directory / "injection_recovery.parquet")
    end_to_end = _end_to_end(injection)
    model_metrics = _load_json(model_directory / "metrics.json")
    group_metrics = _load_json(robustness_directory / "group_metrics.json")
    uncertainty = _load_json(robustness_directory / "uncertainty.json")
    injection_summary = _load_json(robustness_directory / "injection_summary.json")
    shift_summary = _load_json(robustness_directory / "shift_summary.json")
    benchmark = {
        "headline": (
            "Grouped development-TIC robustness is the primary performance summary; "
            "the perfect two-TIC locked test is secondary."
        ),
        "benchmark_hierarchy": [
            "official_locked_test",
            "grouped_development_robustness",
            "group_bootstrap_uncertainty",
            "injection_recovery",
            "domain_shift_limitations",
        ],
        "official_development_demo_modeling": {
            "validation_comparison": model_metrics["validation"],
            "selected_model": model_metrics["selected_model"],
            "selected_ablation": model_metrics["selected_ablation"],
            "frozen_threshold": model.threshold,
            "official_locked_test": model_metrics["selected_test"],
        },
        "grouped_robustness": group_metrics,
        "group_bootstrap_uncertainty": uncertainty,
        "pipeline_recovery": injection_summary,
        "end_to_end_injection_performance": end_to_end,
        "ablation": model_metrics["ablation_winners"],
        "calibration": {
            name: calibration[name]
            for name in (
                "development_oof",
                "official_validation",
                "official_locked_test_descriptive",
            )
        },
        "domain_shift": shift_summary,
        "metric_replacement_or_cherry_picking": False,
        "bls_and_classifier_stages_kept_distinct": True,
        "real_tess_exoplanet_accuracy_claimed": False,
    }
    violation_fraction = float((science_support["outside_robust_support_count"] > 0).mean())
    labeled = science_labels["gold_candidate_label"].isin(["positive", "negative"])
    scientific_tics = int(science_features["object_id"].nunique())
    criteria = [
        (
            "independently_labeled_scientific_candidates_exist",
            bool(labeled.any()),
            f"labeled candidates={int(labeled.sum())}",
        ),
        (
            "both_positive_and_negative_real_classes_exist",
            set(science_labels.loc[labeled, "gold_candidate_label"]) == {"positive", "negative"},
            "no real binary labels",
        ),
        (
            "labels_span_multiple_tic_groups",
            scientific_tics > 1 and bool(labeled.any()),
            "labels span 0 TIC groups",
        ),
        (
            "no_catastrophic_demo_to_scientific_support_shift",
            violation_fraction <= settings.robust_support_violation_limit,
            (
                f"robust-support violation fraction={violation_fraction:.6f}; "
                f"limit={settings.robust_support_violation_limit:.6f}"
            ),
        ),
        (
            "bls_recovers_meaningful_fraction_of_real_catalog_events",
            False,
            "no independently labeled real event recovery cohort",
        ),
        ("grouped_scientific_evaluation_is_possible", False, "real labels absent"),
        (
            "no_training_evaluation_leakage",
            True,
            "scientific rows excluded from training and truth fields excluded from model features",
        ),
        (
            "sufficient_independent_tic_count",
            scientific_tics >= settings.minimum_scientific_tics,
            f"scientific TIC count={scientific_tics}; required={settings.minimum_scientific_tics}",
        ),
    ]
    readiness = {
        "scientific_readiness": "ready_for_scientific_evaluation"
        if all(item[1] for item in criteria)
        else "blocked",
        "demo_pipeline_status": "complete",
        "criteria": [
            {"criterion": name, "passed": passed, "evidence": evidence}
            for name, passed, evidence in criteria
        ],
        "failed_criteria": [name for name, passed, _ in criteria if not passed],
        "scientific_candidates_remain_unlabeled": True,
    }
    identity = FinalEvaluationIdentity(
        settings.model_version,
        settings.robustness_evaluation_version,
        DEMO_FEATURES,
        SCIENTIFIC_FEATURES,
        DEMO_DATASET,
        SCIENTIFIC_DATASET,
        settings.calibration_protocol,
        settings.interpretability_protocol,
        settings.scorecard_policy,
        settings.readiness_policy,
    )
    triage_summary = {
        str(key): int(value)
        for key, value in science_scorecards["triage_status"].value_counts().sort_index().items()
    }
    payloads = {
        "calibration.json": calibration,
        "importance_summary.json": importance_summary,
        "benchmark_summary.json": benchmark,
        "scientific_readiness.json": readiness,
        "triage_summary.json": {
            "scientific_candidate_count": len(science_scorecards),
            "statuses": triage_summary,
            "vocabulary_is_non_astrophysical": True,
        },
    }
    tables = {
        "global_importance.parquet": importance,
        "local_explanations.parquet": local,
        "shift_importance_risk.parquet": risk,
        "candidate_scorecards.parquet": scorecards,
        "demo_scorecard_evaluation.parquet": demo_evaluation,
    }
    directory = freeze_final_evaluation(
        identity=identity,
        tables=tables,
        payloads=payloads,
        upstream_checksums=upstream,
        root=repository / "data/evaluation",
    )
    model_checksums_after = {
        path.name: sha256_file(path) for path in model_directory.iterdir() if path.is_file()
    }
    if model_checksums_before != model_checksums_after:
        raise ValueError("Official model artifacts changed during final evaluation.")
    return FinalEvaluationRunResult(
        directory, make_final_evaluation_version(identity), calibration, readiness
    )
