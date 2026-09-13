"""Offline B040--B045 leakage, reproducibility, and real-artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transiteye.config import ModelingSettings, load_config
from transiteye.features.registry import feature_names_by_groups, model_feature_names
from transiteye.modeling.artifacts import (
    ModelFreezeError,
    ModelIdentityInputs,
    freeze_inference,
    freeze_model,
    freeze_split,
    load_frozen_model,
    make_model_version,
)
from transiteye.modeling.elm import ExtremeLearningMachine
from transiteye.modeling.inference import predict_candidates
from transiteye.modeling.metrics import (
    LockedTestEvaluator,
    choose_f1_threshold,
    compute_metrics,
)
from transiteye.modeling.models import MODEL_ORDER, fit_classifier, model_requires_scaling
from transiteye.modeling.pipeline import ModelingRunResult, run_demo_modeling
from transiteye.modeling.split import make_grouped_split, validate_grouped_split
from transiteye.modeling.transforms import TrainOnlyTransformer

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATASET = "dataset-demo-a74b6aad7c21faf15b0d"
DEMO_FEATURES = "features-6278a328b8f624b2759b"
SCIENTIFIC_DATASET = "dataset-4b84e8acaa6f6b334c2f"
SCIENTIFIC_FEATURES = "features-4e9f6d54b84780a2be42"


@pytest.fixture(scope="module")
def settings() -> ModelingSettings:
    value = load_config(ROOT / "configs/modeling/demo_mvp.yaml").modeling
    assert value is not None
    return value


@pytest.fixture(scope="module")
def demo_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_root = ROOT / "data/features" / DEMO_DATASET / DEMO_FEATURES
    return (
        pd.read_parquet(feature_root / "features.parquet"),
        pd.read_parquet(feature_root / "labels.parquet"),
        pd.read_parquet(ROOT / "data/datasets" / DEMO_DATASET / "artifact_lineage.parquet"),
    )


@pytest.fixture(scope="module")
def real_modeling_result() -> ModelingRunResult:
    return run_demo_modeling(ROOT)


def test_grouped_split_is_deterministic_nonempty_and_leakage_free(
    settings: ModelingSettings,
    demo_tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, labels, lineage = demo_tables
    first = make_grouped_split(
        features,
        labels,
        lineage,
        settings=settings,
        master_seed=42,
        dataset_version=DEMO_DATASET,
        feature_version=DEMO_FEATURES,
    )
    second = make_grouped_split(
        features,
        labels,
        lineage,
        settings=settings,
        master_seed=42,
        dataset_version=DEMO_DATASET,
        feature_version=DEMO_FEATURES,
    )
    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    assert first.split_id == second.split_id
    assert first.manifest.groupby("object_id")["split"].nunique().max() == 1
    assert first.manifest.groupby("source_raw_checksum")["split"].nunique().max() == 1
    assert set(first.manifest["split"]) == {"train", "validation", "test"}
    assert first.manifest.groupby("split")["object_id"].nunique().to_dict() == {
        "test": 2,
        "train": 4,
        "validation": 2,
    }


def test_split_validator_rejects_tic_and_checksum_leakage(
    settings: ModelingSettings,
    demo_tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, labels, lineage = demo_tables
    manifest = make_grouped_split(
        features,
        labels,
        lineage,
        settings=settings,
        master_seed=42,
        dataset_version=DEMO_DATASET,
        feature_version=DEMO_FEATURES,
    ).manifest
    leaked = manifest.copy()
    same_tic = leaked.index[leaked["object_id"] == leaked.loc[0, "object_id"]]
    leaked.loc[same_tic[0], "split"] = (
        "test" if leaked.loc[same_tic[1], "split"] != "test" else "train"
    )
    with pytest.raises(ValueError, match="object_id"):
        validate_grouped_split(leaked)
    leaked = manifest.copy()
    source = leaked.loc[0, "source_raw_checksum"]
    source_split = leaked.loc[0, "split"]
    target = leaked.index[leaked["split"] != source_split][0]
    leaked.loc[target, "source_raw_checksum"] = source
    with pytest.raises(ValueError, match="source_raw_checksum"):
        validate_grouped_split(leaked)


def test_only_gold_rows_form_supervised_cohort(demo_tables: tuple[pd.DataFrame, ...]) -> None:
    _, labels, _ = demo_tables
    supervised = labels[labels["gold_candidate_label"].isin(["positive", "negative"])]
    assert len(supervised) == 130
    assert supervised["gold_candidate_label"].value_counts().to_dict() == {
        "positive": 70,
        "negative": 60,
    }
    assert not supervised["gold_candidate_label"].isin(["unlabeled", "ambiguous"]).any()


def test_transformer_medians_and_scaling_are_train_only() -> None:
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0], "empty": np.nan})
    transformer = TrainOnlyTransformer(scale=True).fit(train)
    state = transformer.portable_state()
    assert transformer.removed_all_null_features == ("empty",)
    assert transformer.medians is not None
    np.testing.assert_allclose(transformer.medians, [2.0, 4.0])
    validation = pd.DataFrame({"a": [1e9], "b": [np.nan], "empty": [8.0]})
    transformed = transformer.transform(validation)
    assert np.isfinite(transformed).all()
    assert transformer.portable_state() == state


def test_test_missingness_and_validation_outlier_cannot_refit_transformer() -> None:
    train = pd.DataFrame({"a": [0.0, 2.0], "b": [10.0, 14.0]})
    transformer = TrainOnlyTransformer(scale=True).fit(train)
    expected = transformer.state_hash()
    transformer.transform(pd.DataFrame({"a": [np.nan], "b": [1e12]}))
    transformer.transform(pd.DataFrame({"a": [1e15], "b": [np.nan]}))
    assert transformer.state_hash() == expected
    with pytest.raises(ValueError, match="missing columns"):
        transformer.transform(pd.DataFrame({"a": [1.0]}))


@pytest.mark.parametrize("family", MODEL_ORDER)
def test_all_baseline_models_train_predict_and_are_deterministic(
    family: str, settings: ModelingSettings
) -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(80, 6))
    labels = (features[:, 0] - 0.5 * features[:, 1] > 0).astype(int)
    first = fit_classifier(family, features, labels, settings=settings, master_seed=42)
    second = fit_classifier(family, features, labels, settings=settings, master_seed=42)
    np.testing.assert_allclose(first.scores(features), second.scores(features), rtol=0, atol=1e-12)
    assert first.requires_scaling == model_requires_scaling(family)


def test_elm_rejects_invalid_training_contract() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExtremeLearningMachine(0, 1.0, 1)
    with pytest.raises(ValueError, match="both binary"):
        ExtremeLearningMachine(4, 1.0, 1).fit(np.ones((3, 2)), np.ones(3, dtype=int))


def test_validation_threshold_and_locked_test_contract() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold = choose_f1_threshold(labels, scores)
    assert threshold == 0.6
    metrics = compute_metrics(labels, scores, threshold)
    assert metrics.f1 == 1
    locked = LockedTestEvaluator(labels)
    assert locked.evaluate(scores, threshold).pr_auc == 1
    with pytest.raises(RuntimeError, match="already"):
        locked.evaluate(scores, threshold)


def test_registry_ablation_subsets_are_explicit_and_truth_blind() -> None:
    time_bls = feature_names_by_groups(("time_domain", "bls"))
    full = feature_names_by_groups(("time_domain", "bls", "lomb_scargle", "fft"))
    assert len(time_bls) == 41
    assert full == model_feature_names()
    assert set(time_bls) < set(full)
    assert all("catalog" not in column and "injected" not in column for column in full)
    with pytest.raises(ValueError, match="Unknown"):
        feature_names_by_groups(("truth",))


def _identity(**updates: object) -> ModelIdentityInputs:
    values: dict[str, object] = {
        "dataset_version": DEMO_DATASET,
        "feature_version": DEMO_FEATURES,
        "split_id": "model-split-a",
        "training_candidate_ids": ("cand-b", "cand-a"),
        "modeling_config_hash": "a" * 20,
        "transform_hash": "b" * 20,
        "feature_names": ("f1", "f2"),
        "model_family": "elm",
        "hyperparameters": {"hidden_units": 4},
        "model_seed": 7,
        "threshold_policy": "maximize_validation_f1",
        "selected_threshold": 0.3,
    }
    values.update(updates)
    return ModelIdentityInputs(**values)  # type: ignore[arg-type]


def test_model_version_is_portable_deterministic_and_sensitive() -> None:
    identity = _identity()
    assert make_model_version(identity) == make_model_version(identity)
    assert (Path("/relocated") / make_model_version(identity)).name == make_model_version(identity)
    assert make_model_version(_identity(model_family="svm_rbf")) != make_model_version(identity)
    assert make_model_version(_identity(split_id="model-split-b")) != make_model_version(identity)


def test_real_training_replay_and_scientific_inference_are_deterministic(
    real_modeling_result: ModelingRunResult,
) -> None:
    replay = run_demo_modeling(ROOT)
    assert replay.model.model_version == real_modeling_result.model.model_version
    assert replay.model.threshold == real_modeling_result.model.threshold
    assert replay.model.classifier.family == real_modeling_result.model.classifier.family
    assert len(replay.validation_runs) == 8
    assert replay.model.statistical_role == "development_demo"
    pd.testing.assert_frame_equal(replay.split.manifest, real_modeling_result.split.manifest)
    pd.testing.assert_frame_equal(
        replay.scientific_predictions, real_modeling_result.scientific_predictions
    )
    assert len(replay.scientific_predictions) == 75
    expected_ids = pd.read_parquet(
        ROOT / "data/features" / SCIENTIFIC_DATASET / SCIENTIFIC_FEATURES / "features.parquet"
    )["candidate_id"]
    assert set(replay.scientific_predictions["candidate_id"]) == set(expected_ids)
    metadata = json.loads(
        (replay.model_directory / "model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["identity_inputs"]["dataset_version"] == DEMO_DATASET
    assert SCIENTIFIC_DATASET not in str(metadata["identity_inputs"])


def test_shared_inference_api_preserves_candidate_ids(
    real_modeling_result: ModelingRunResult,
    demo_tables: tuple[pd.DataFrame, ...],
) -> None:
    demo_features, _, _ = demo_tables
    subset = demo_features.head(7)
    predictions = predict_candidates(subset, real_modeling_result.model)
    assert predictions["candidate_id"].tolist() == subset["candidate_id"].tolist()
    assert predictions["model_version"].nunique() == 1
    assert predictions["model_version"].iloc[0] == real_modeling_result.model.model_version
    with pytest.raises(ValueError, match="candidate identity"):
        predict_candidates(subset.drop(columns="candidate_id"), real_modeling_result.model)
    with pytest.raises(ValueError, match="lacks model features"):
        predict_candidates(
            subset.drop(columns=real_modeling_result.model.feature_names[0]),
            real_modeling_result.model,
        )


def test_unfitted_transform_and_malformed_split_are_rejected() -> None:
    transformer = TrainOnlyTransformer(scale=False)
    with pytest.raises(ValueError, match="not been fitted"):
        transformer.transform(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(ValueError, match="not been fitted"):
        transformer.portable_state()
    with pytest.raises(ValueError, match="lacks columns"):
        validate_grouped_split(pd.DataFrame({"candidate_id": ["cand-a"]}))
    with pytest.raises(ValueError, match="Unsupported"):
        model_requires_scaling("unknown")


def test_model_split_and_inference_artifact_creation_and_integrity(
    tmp_path: Path, real_modeling_result: ModelingRunResult
) -> None:
    result = real_modeling_result
    metadata = json.loads(
        (result.model_directory / "model_metadata.json").read_text(encoding="utf-8")
    )
    raw_identity = metadata["identity_inputs"]
    identity = ModelIdentityInputs(
        dataset_version=raw_identity["dataset_version"],
        feature_version=raw_identity["feature_version"],
        split_id=raw_identity["split_id"],
        training_candidate_ids=tuple(raw_identity["training_candidate_ids"]),
        modeling_config_hash=raw_identity["modeling_config_hash"],
        transform_hash=raw_identity["transform_hash"],
        feature_names=tuple(raw_identity["feature_names"]),
        model_family=raw_identity["model_family"],
        hyperparameters=raw_identity["hyperparameters"],
        model_seed=raw_identity["model_seed"],
        threshold_policy=raw_identity["threshold_policy"],
        selected_threshold=raw_identity["selected_threshold"],
    )
    split_directory = freeze_split(
        result.split.manifest,
        split_id=result.split.split_id,
        dataset_version=DEMO_DATASET,
        feature_version=DEMO_FEATURES,
        seed=result.split.seed,
        root=tmp_path / "splits",
    )
    assert (
        freeze_split(
            result.split.manifest,
            split_id=result.split.split_id,
            dataset_version=DEMO_DATASET,
            feature_version=DEMO_FEATURES,
            seed=result.split.seed,
            root=tmp_path / "splits",
        )
        == split_directory
    )
    model_directory = freeze_model(
        result.model,
        identity=identity,
        metrics={"role": "test-fixture"},
        split_manifest=result.split.manifest,
        root=tmp_path / "models",
    )
    loaded = load_frozen_model(model_directory)
    assert loaded.model_version == result.model.model_version
    inference_directory = freeze_inference(
        result.scientific_predictions,
        model_version=loaded.model_version,
        dataset_version=SCIENTIFIC_DATASET,
        root=tmp_path / "inference",
    )
    assert (
        freeze_inference(
            result.scientific_predictions,
            model_version=loaded.model_version,
            dataset_version=SCIENTIFIC_DATASET,
            root=tmp_path / "inference",
        )
        == inference_directory
    )
    (model_directory / "feature_list.json").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ModelFreezeError, match="checksum"):
        load_frozen_model(model_directory)
