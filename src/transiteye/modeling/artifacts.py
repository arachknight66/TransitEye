"""Portable B040/B045 identity and immutable artifact storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.modeling.inference import FrozenModel
from transiteye.serialization import canonical_json, content_hash


class ModelFreezeError(ValueError):
    """Raised when an immutable modeling artifact conflicts or fails integrity."""


@dataclass(frozen=True)
class ModelIdentityInputs:
    dataset_version: str
    feature_version: str
    split_id: str
    training_candidate_ids: tuple[str, ...]
    modeling_config_hash: str
    transform_hash: str
    feature_names: tuple[str, ...]
    model_family: str
    hyperparameters: dict[str, Any]
    model_seed: int
    threshold_policy: str
    selected_threshold: float

    def portable_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["training_candidate_ids"] = sorted(set(self.training_candidate_ids))
        return value


def make_model_version(identity: ModelIdentityInputs) -> str:
    return f"model-{content_hash(identity.portable_dict())}"


def freeze_split(
    manifest: pd.DataFrame,
    *,
    split_id: str,
    dataset_version: str,
    feature_version: str,
    seed: int,
    root: str | Path,
) -> Path:
    directory = Path(root) / dataset_version / split_id
    path = directory / "split_manifest.parquet"
    metadata_path = directory / "split_metadata.json"
    canonical = manifest.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(path) != metadata["split_manifest_sha256"]:
            raise ModelFreezeError("Frozen modeling split checksum mismatch.")
        pd.testing.assert_frame_equal(pd.read_parquet(path), canonical, check_dtype=False)
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    canonical.to_parquet(path, engine="pyarrow", index=False)
    metadata = {
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "split_id": split_id,
        "split_seed": seed,
        "statistical_role": "development_demo",
        "split_manifest_sha256": sha256_file(path),
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return directory


def freeze_model(
    model: FrozenModel,
    *,
    identity: ModelIdentityInputs,
    metrics: dict[str, Any],
    split_manifest: pd.DataFrame,
    root: str | Path,
) -> Path:
    """Freeze the selected trusted local model and auditable metadata."""
    version = make_model_version(identity)
    if version != model.model_version:
        raise ValueError("Frozen model version does not match its identity.")
    directory = Path(root) / version
    metadata_path = directory / "model_metadata.json"
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for filename, expected in metadata["artifact_checksums"].items():
            if sha256_file(directory / filename) != expected:
                raise ModelFreezeError(f"Frozen model checksum mismatch: {filename}")
        if metadata["model_version"] != version:
            raise ModelFreezeError("Frozen model metadata identity mismatch.")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    metrics_path = directory / "metrics.json"
    features_path = directory / "feature_list.json"
    preprocessing_path = directory / "preprocessing_state.json"
    split_path = directory / "split_manifest.parquet"
    joblib.dump(model, model_path, compress=0)
    metrics_path.write_text(canonical_json(metrics) + "\n", encoding="utf-8")
    features_path.write_text(canonical_json(list(model.feature_names)) + "\n", encoding="utf-8")
    preprocessing_path.write_text(
        canonical_json(model.transformer.portable_state()) + "\n", encoding="utf-8"
    )
    split_manifest.sort_values("candidate_id", kind="stable").to_parquet(
        split_path, engine="pyarrow", index=False
    )
    metadata = {
        "model_version": version,
        "statistical_role": model.statistical_role,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "identity_inputs": identity.portable_dict(),
        "serialization": "joblib; load only trusted local artifacts",
        "artifact_checksums": {
            path.name: sha256_file(path)
            for path in (model_path, metrics_path, features_path, preprocessing_path, split_path)
        },
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return directory


def load_frozen_model(directory: str | Path) -> FrozenModel:
    """Load a trusted local artifact after checksum and identity validation."""
    root = Path(directory)
    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    for filename, expected in metadata["artifact_checksums"].items():
        if sha256_file(root / filename) != expected:
            raise ModelFreezeError(f"Frozen model checksum mismatch: {filename}")
    model = joblib.load(root / "model.joblib")
    if not isinstance(model, FrozenModel) or model.model_version != metadata["model_version"]:
        raise ModelFreezeError("Serialized model contract is invalid.")
    return model


def freeze_inference(
    predictions: pd.DataFrame,
    *,
    model_version: str,
    dataset_version: str,
    root: str | Path,
) -> Path:
    directory = Path(root) / model_version / dataset_version
    path = directory / "predictions.parquet"
    metadata_path = directory / "inference_metadata.json"
    canonical = predictions.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(path) != metadata["predictions_sha256"]:
            raise ModelFreezeError("Frozen inference checksum mismatch.")
        pd.testing.assert_frame_equal(pd.read_parquet(path), canonical, check_dtype=False)
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    canonical.to_parquet(path, engine="pyarrow", index=False)
    metadata_path.write_text(
        canonical_json(
            {
                "model_version": model_version,
                "dataset_version": dataset_version,
                "inference_role": "unlabeled_scientific_development_smoke_test",
                "predictions_sha256": sha256_file(path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return directory
