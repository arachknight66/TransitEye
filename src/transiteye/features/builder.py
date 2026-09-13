"""Mode-agnostic B039 feature construction and immutable freezing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import config_hash, load_config
from transiteye.datasets.versioning import load_frozen_dataset
from transiteye.features.bls_features import extract_bls_features
from transiteye.features.frequency_domain import (
    extract_fft_features,
    extract_lomb_scargle_features,
    fft_spectrum,
    lomb_scargle_spectrum,
)
from transiteye.features.registry import (
    FEATURE_REGISTRY,
    model_feature_names,
    registry_hash,
)
from transiteye.features.schemas import ColumnRole
from transiteye.features.time_domain import extract_time_domain_features
from transiteye.features.validation import validate_feature_matrix
from transiteye.serialization import canonical_json, content_hash


class FeatureFreezeError(ValueError):
    """Raised when frozen feature artifacts fail immutability or integrity checks."""


@dataclass(frozen=True)
class FeatureIdentityInputs:
    dataset_version: str
    feature_config_hash: str
    registry_hash: str
    preprocessing_config_hashes: tuple[str, ...]
    bls_config_hashes: tuple[str, ...]
    generator_version: str

    def portable_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["preprocessing_config_hashes"] = sorted(set(self.preprocessing_config_hashes))
        value["bls_config_hashes"] = sorted(set(self.bls_config_hashes))
        return value


@dataclass(frozen=True)
class BuiltFeatureMatrix:
    features: pd.DataFrame
    labels: pd.DataFrame
    feature_version: str
    validation: dict[str, Any]
    identity: FeatureIdentityInputs


def make_feature_version(identity: FeatureIdentityInputs) -> str:
    return f"features-{content_hash(identity.portable_dict())}"


def _artifact_paths(root: Path, lineage: dict[str, Any], bls_config_hash: str) -> tuple[Path, Path]:
    """Resolve artifacts from portable lineage keys, never labels or dataset role."""
    variant_id = lineage.get("variant_id")
    if variant_id is not None and str(variant_id) not in {"", "None", "nan"}:
        variant_matches = list(
            root.glob(f"data/synthetic/*/variants/{variant_id}/processed.parquet")
        )
        periodogram_matches = list(
            root.glob(f"data/synthetic/*/detection/{variant_id}-periodogram.parquet")
        )
    else:
        checksum = str(lineage["source_raw_checksum"])
        key = checksum[:20]
        variant_matches = [root / "data/processed/mvp" / key / "cadences.parquet"]
        periodogram_matches = [
            root / "data/candidates" / bls_config_hash / f"{key}-periodogram.parquet"
        ]
    if len(variant_matches) != 1 or len(periodogram_matches) != 1:
        raise ValueError("Artifact lineage did not resolve to exactly one processed/BLS pair.")
    processed, periodogram = variant_matches[0], periodogram_matches[0]
    if not processed.is_file() or not periodogram.is_file():
        raise ValueError("A feature source artifact is unavailable.")
    return processed, periodogram


def _label_companion(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        entry.name
        for entry in FEATURE_REGISTRY
        if entry.role in {ColumnRole.LABEL, ColumnRole.EVALUATION}
        or entry.group == "label_provenance"
    ]
    result = pd.DataFrame({"candidate_id": candidates["candidate_id"].astype(str)})
    for column in columns:
        result[column] = candidates[column] if column in candidates else None
    return result.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def build_feature_matrix(
    root: str | Path,
    dataset_version: str,
    *,
    config_path: str | Path | None = None,
) -> BuiltFeatureMatrix:
    """Build one shared feature contract from any conforming frozen candidate dataset."""
    repository = Path(root)
    config = load_config(config_path or repository / "configs/features/mvp.yaml")
    settings = config.features
    if settings is None:
        raise ValueError("Feature configuration is unavailable.")
    dataset_root = repository / "data/datasets" / dataset_version
    dataset = load_frozen_dataset(dataset_root, expected_version=dataset_version)
    candidates = dataset.candidates.sort_values("candidate_id", kind="stable").reset_index(
        drop=True
    )
    lineage = dataset.artifact_lineage
    required = {"candidate_id", "object_id", "observation_group_id", "bls_config_hash"}
    if missing := required.difference(candidates.columns):
        raise ValueError(f"Candidate dataset lacks required columns: {sorted(missing)}")

    identity = FeatureIdentityInputs(
        dataset_version=dataset_version,
        feature_config_hash=config_hash(config),
        registry_hash=registry_hash(),
        preprocessing_config_hashes=tuple(
            sorted(candidates["preprocessing_config_hash"].astype(str).unique())
        ),
        bls_config_hashes=tuple(sorted(candidates["bls_config_hash"].astype(str).unique())),
        generator_version=settings.generator_version,
    )
    feature_version = make_feature_version(identity)
    rows: list[dict[str, Any]] = []
    for group_id, group_candidates in candidates.groupby("observation_group_id", sort=True):
        lineage_rows = lineage.loc[lineage["observation_group_id"].astype(str) == str(group_id)]
        if len(lineage_rows) != 1:
            raise ValueError(f"Observation group {group_id} lacks unique artifact lineage.")
        lineage_row = {str(key): value for key, value in lineage_rows.iloc[0].to_dict().items()}
        bls_hash = str(group_candidates["bls_config_hash"].iloc[0])
        processed_path, periodogram_path = _artifact_paths(repository, lineage_row, bls_hash)
        cadences = pd.read_parquet(processed_path)
        periodogram = pd.read_parquet(periodogram_path)
        enabled = set(settings.enabled_groups)
        ls_frequency, ls_power = (
            lomb_scargle_spectrum(cadences, settings)
            if "lomb_scargle" in enabled
            else (np.array([], dtype=float), np.array([], dtype=float))
        )
        fft_frequency, fft_power = (
            fft_spectrum(cadences, settings)
            if "fft" in enabled
            else (np.array([], dtype=float), np.array([], dtype=float))
        )
        for raw_candidate in group_candidates.to_dict(orient="records"):
            candidate = {str(key): value for key, value in raw_candidate.items()}
            row: dict[str, Any] = {
                "candidate_id": str(candidate["candidate_id"]),
                "object_id": str(candidate["object_id"]),
                "observation_group_id": str(candidate["observation_group_id"]),
                "dataset_version": dataset_version,
                "feature_version": feature_version,
                "preprocessing_config_hash": str(candidate["preprocessing_config_hash"]),
                "bls_config_hash": bls_hash,
                **dict.fromkeys(model_feature_names(), float("nan")),
            }
            if "time_domain" in enabled:
                row.update(extract_time_domain_features(candidate, cadences, settings))
            if "bls" in enabled:
                row.update(
                    extract_bls_features(
                        candidate, cadences, periodogram, group_candidates, settings
                    )
                )
            period = float(candidate["candidate_period"])
            if "lomb_scargle" in enabled:
                row.update(extract_lomb_scargle_features(ls_frequency, ls_power, period, settings))
            if "fft" in enabled:
                row.update(extract_fft_features(fft_frequency, fft_power, period, settings))
            rows.append(row)
    ordered = [
        entry.name
        for entry in FEATURE_REGISTRY
        if entry.role in {ColumnRole.IDENTITY, ColumnRole.PROVENANCE, ColumnRole.FEATURE}
        and entry.group != "label_provenance"
    ]
    features = (
        pd.DataFrame(rows, columns=ordered)
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
    )
    for column in model_feature_names():
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    labels = _label_companion(candidates)
    validation = validate_feature_matrix(features)
    label_diagnostics: dict[str, Any] = {}
    joined = features[["candidate_id", *model_feature_names()]].merge(
        labels[["candidate_id", "gold_candidate_label"]], on="candidate_id", validate="one_to_one"
    )
    for label, group in joined.groupby("gold_candidate_label", dropna=False, sort=True):
        label_diagnostics[str(label)] = {
            "candidate_rows": int(len(group)),
            "missing_fraction": {
                column: float(group[column].isna().mean()) for column in model_feature_names()
            },
        }
    validation["label_diagnostics"] = label_diagnostics
    return BuiltFeatureMatrix(
        features=features,
        labels=labels,
        feature_version=feature_version,
        validation=validation,
        identity=identity,
    )


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def freeze_feature_matrix(built: BuiltFeatureMatrix, root: str | Path) -> Path:
    """Freeze or integrity-check a deterministic B039 feature matrix."""
    directory = Path(root) / built.identity.dataset_version / built.feature_version
    features_path = directory / "features.parquet"
    labels_path = directory / "labels.parquet"
    metadata_path = directory / "feature_metadata.json"
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for filename, expected in metadata["table_checksums"].items():
            if sha256_file(directory / filename) != expected:
                raise FeatureFreezeError(f"Frozen feature checksum mismatch: {filename}")
        if sha256_file(directory / "feature_registry.json") != metadata["registry_sha256"]:
            raise FeatureFreezeError("Frozen feature registry checksum mismatch.")
        if sha256_file(directory / "feature_validation.json") != metadata["validation_sha256"]:
            raise FeatureFreezeError("Frozen feature validation checksum mismatch.")
        if (directory / "feature_validation.json").read_text(encoding="utf-8") != (
            canonical_json(built.validation) + "\n"
        ):
            raise FeatureFreezeError("Existing feature validation has conflicting content.")
        frozen = pd.read_parquet(features_path)
        labels = pd.read_parquet(labels_path)
        try:
            pd.testing.assert_frame_equal(
                _canonical_frame(built.features), frozen, check_dtype=False
            )
            pd.testing.assert_frame_equal(_canonical_frame(built.labels), labels, check_dtype=False)
        except AssertionError as exc:
            raise FeatureFreezeError("Existing feature version has conflicting content.") from exc
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    features = _canonical_frame(built.features)
    labels = _canonical_frame(built.labels)
    features.to_parquet(features_path, engine="pyarrow", index=False)
    labels.to_parquet(labels_path, engine="pyarrow", index=False)
    registry_payload = [entry.portable_dict() for entry in FEATURE_REGISTRY]
    (directory / "feature_registry.json").write_text(
        canonical_json(registry_payload) + "\n", encoding="utf-8"
    )
    (directory / "feature_validation.json").write_text(
        canonical_json(built.validation) + "\n", encoding="utf-8"
    )
    metadata = {
        "dataset_version": built.identity.dataset_version,
        "feature_version": built.feature_version,
        "identity_inputs": built.identity.portable_dict(),
        "feature_columns": list(model_feature_names()),
        "table_checksums": {
            "features.parquet": sha256_file(features_path),
            "labels.parquet": sha256_file(labels_path),
        },
        "registry_sha256": sha256_file(directory / "feature_registry.json"),
        "validation_sha256": sha256_file(directory / "feature_validation.json"),
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return directory
