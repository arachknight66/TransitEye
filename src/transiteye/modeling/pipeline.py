"""B040--B045 development-demo training, selection, and frozen inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import config_hash, load_config
from transiteye.features.registry import feature_names_by_groups
from transiteye.modeling.artifacts import (
    ModelIdentityInputs,
    freeze_inference,
    freeze_model,
    freeze_split,
    load_frozen_model,
    make_model_version,
)
from transiteye.modeling.inference import FrozenModel, predict_candidates
from transiteye.modeling.metrics import (
    BinaryMetrics,
    LockedTestEvaluator,
    choose_f1_threshold,
    compute_metrics,
)
from transiteye.modeling.models import (
    MODEL_ORDER,
    TrainedClassifier,
    fit_classifier,
    model_requires_scaling,
)
from transiteye.modeling.split import ModelingSplit, make_grouped_split
from transiteye.modeling.transforms import TrainOnlyTransformer


@dataclass(frozen=True)
class ValidationRun:
    ablation: str
    family: str
    feature_names: tuple[str, ...]
    transformer: TrainOnlyTransformer
    classifier: TrainedClassifier
    threshold: float
    validation_metrics: BinaryMetrics


@dataclass(frozen=True)
class ModelingRunResult:
    model_directory: Path
    split_directory: Path
    inference_directory: Path
    model: FrozenModel
    split: ModelingSplit
    validation_runs: tuple[ValidationRun, ...]
    ablation_test_metrics: dict[str, BinaryMetrics]
    scientific_predictions: pd.DataFrame
    supervised_counts: dict[str, int]


def _load_feature_artifact(
    root: Path, dataset_version: str, feature_version: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = root / "data/features" / dataset_version / feature_version
    metadata = json.loads((directory / "feature_metadata.json").read_text(encoding="utf-8"))
    if (
        metadata["dataset_version"] != dataset_version
        or metadata["feature_version"] != feature_version
    ):
        raise ValueError("Feature artifact identity mismatch.")
    for filename, expected in metadata["table_checksums"].items():
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"Feature artifact checksum mismatch: {filename}")
    return pd.read_parquet(directory / "features.parquet"), pd.read_parquet(
        directory / "labels.parquet"
    )


def _feature_groups() -> dict[str, tuple[str, ...]]:
    return {
        "time_bls": feature_names_by_groups(("time_domain", "bls")),
        "time_bls_frequency": feature_names_by_groups(
            ("time_domain", "bls", "lomb_scargle", "fft")
        ),
    }


def _labels(values: pd.Series) -> np.ndarray:
    mapped = values.map({"negative": 0, "positive": 1})
    if mapped.isna().any():
        raise ValueError("Supervised cohort contains non-binary labels.")
    return mapped.to_numpy(dtype=int)


def _fit_validation_run(
    data: pd.DataFrame,
    *,
    ablation: str,
    family: str,
    feature_names: tuple[str, ...],
    settings: Any,
    master_seed: int,
) -> ValidationRun:
    train = data.loc[data["split"] == "train"]
    validation = data.loc[data["split"] == "validation"]
    transformer = TrainOnlyTransformer(scale=model_requires_scaling(family))
    train_features = transformer.fit_transform(train.loc[:, feature_names])
    validation_features = transformer.transform(validation.loc[:, feature_names])
    classifier = fit_classifier(
        family,
        train_features,
        _labels(train["gold_candidate_label"]),
        settings=settings,
        master_seed=master_seed,
    )
    scores = classifier.scores(validation_features)
    validation_labels = _labels(validation["gold_candidate_label"])
    threshold = choose_f1_threshold(validation_labels, scores)
    metrics = compute_metrics(validation_labels, scores, threshold)
    return ValidationRun(
        ablation,
        family,
        feature_names,
        transformer,
        classifier,
        threshold,
        metrics,
    )


def _best_run(runs: list[ValidationRun]) -> ValidationRun:
    return max(
        runs,
        key=lambda run: (
            run.validation_metrics.pr_auc,
            -MODEL_ORDER.index(run.family),
        ),
    )


def _split_summary(manifest: pd.DataFrame) -> dict[str, int]:
    return {
        f"{split}_{label}": int(
            ((manifest["split"] == split) & (manifest["gold_candidate_label"] == label)).sum()
        )
        for split in ("train", "validation", "test")
        for label in ("positive", "negative", "unlabeled", "ambiguous")
    }


def run_demo_modeling(
    root: str | Path,
    *,
    demo_dataset_version: str = "dataset-demo-a74b6aad7c21faf15b0d",
    demo_feature_version: str = "features-6278a328b8f624b2759b",
    scientific_dataset_version: str = "dataset-4b84e8acaa6f6b334c2f",
    scientific_feature_version: str = "features-4e9f6d54b84780a2be42",
) -> ModelingRunResult:
    """Run the frozen development-demo protocol; test is opened after selection only."""
    repository = Path(root)
    config = load_config(repository / "configs/modeling/demo_mvp.yaml")
    settings = config.modeling
    if settings is None:
        raise ValueError("Modeling configuration is unavailable.")
    features, labels = _load_feature_artifact(
        repository, demo_dataset_version, demo_feature_version
    )
    scientific_features, scientific_labels = _load_feature_artifact(
        repository, scientific_dataset_version, scientific_feature_version
    )
    if scientific_labels["gold_candidate_label"].isin(["positive", "negative"]).any():
        raise ValueError("Scientific development rows must remain outside supervised training.")
    lineage = pd.read_parquet(
        repository / "data/datasets" / demo_dataset_version / "artifact_lineage.parquet"
    )
    split = make_grouped_split(
        features,
        labels,
        lineage,
        settings=settings,
        master_seed=config.reproducibility.master_seed,
        dataset_version=demo_dataset_version,
        feature_version=demo_feature_version,
    )
    split_directory = freeze_split(
        split.manifest,
        split_id=split.split_id,
        dataset_version=demo_dataset_version,
        feature_version=demo_feature_version,
        seed=split.seed,
        root=repository / "data/modeling",
    )
    combined = features.merge(labels, on="candidate_id", validate="one_to_one").merge(
        split.manifest[["candidate_id", "split"]], on="candidate_id", validate="one_to_one"
    )
    supervised = combined.loc[
        combined["gold_candidate_label"].isin(["positive", "negative"])
    ].copy()
    if len(supervised) != 130:
        raise ValueError("Frozen demo supervised cohort must contain exactly 130 gold rows.")

    all_runs: list[ValidationRun] = []
    winners: dict[str, ValidationRun] = {}
    for ablation, feature_names in _feature_groups().items():
        ablation_runs = [
            _fit_validation_run(
                supervised,
                ablation=ablation,
                family=family,
                feature_names=feature_names,
                settings=settings,
                master_seed=config.reproducibility.master_seed,
            )
            for family in MODEL_ORDER
        ]
        all_runs.extend(ablation_runs)
        winners[ablation] = _best_run(ablation_runs)

    selected = max(
        winners.values(),
        key=lambda run: (
            run.validation_metrics.pr_auc,
            -len(run.feature_names),
            -MODEL_ORDER.index(run.family),
        ),
    )
    # Both ablation choices, the overall selection, and all thresholds are finalized first.
    test = supervised.loc[supervised["split"] == "test"]
    test_labels = _labels(test["gold_candidate_label"])
    ablation_test_metrics: dict[str, BinaryMetrics] = {}
    for ablation, winner in winners.items():
        scores = winner.classifier.scores(
            winner.transformer.transform(test.loc[:, winner.feature_names])
        )
        ablation_test_metrics[ablation] = LockedTestEvaluator(test_labels).evaluate(
            scores, winner.threshold
        )

    training_ids = tuple(
        sorted(supervised.loc[supervised["split"] == "train", "candidate_id"].astype(str))
    )
    identity = ModelIdentityInputs(
        dataset_version=demo_dataset_version,
        feature_version=demo_feature_version,
        split_id=split.split_id,
        training_candidate_ids=training_ids,
        modeling_config_hash=config_hash(config),
        transform_hash=selected.transformer.state_hash(),
        feature_names=selected.feature_names,
        model_family=selected.family,
        hyperparameters=selected.classifier.hyperparameters,
        model_seed=selected.classifier.seed,
        threshold_policy=settings.threshold_policy,
        selected_threshold=selected.threshold,
    )
    version = make_model_version(identity)
    frozen = FrozenModel(
        version,
        selected.feature_names,
        selected.transformer,
        selected.classifier,
        selected.threshold,
        settings.statistical_role,
    )
    metrics_payload = {
        "statistical_role": settings.statistical_role,
        "primary_metric": settings.primary_metric,
        "threshold_policy": settings.threshold_policy,
        "validation": {
            f"{run.ablation}:{run.family}": {
                "threshold": run.threshold,
                **run.validation_metrics.to_dict(),
            }
            for run in all_runs
        },
        "ablation_winners": {
            name: {
                "model_family": winner.family,
                "threshold": winner.threshold,
                "validation": winner.validation_metrics.to_dict(),
                "test": ablation_test_metrics[name].to_dict(),
            }
            for name, winner in winners.items()
        },
        "selected_ablation": selected.ablation,
        "selected_model": selected.family,
        "selected_test": ablation_test_metrics[selected.ablation].to_dict(),
        "split_summary": _split_summary(split.manifest),
    }
    model_directory = freeze_model(
        frozen,
        identity=identity,
        metrics=metrics_payload,
        split_manifest=split.manifest,
        root=repository / "data/models",
    )
    loaded = load_frozen_model(model_directory)
    scientific_predictions = predict_candidates(scientific_features, loaded)
    if set(scientific_predictions["candidate_id"]) != set(scientific_features["candidate_id"]):
        raise ValueError("Scientific inference did not preserve exact candidate identities.")
    inference_directory = freeze_inference(
        scientific_predictions,
        model_version=version,
        dataset_version=scientific_dataset_version,
        root=repository / "data/inference",
    )
    return ModelingRunResult(
        model_directory,
        split_directory,
        inference_directory,
        loaded,
        split,
        tuple(all_runs),
        ablation_test_metrics,
        scientific_predictions,
        _split_summary(split.manifest),
    )
