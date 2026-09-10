"""Portable B034 dataset identity and immutable output storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.datasets.builder import BuiltDataset
from transiteye.serialization import canonical_json, content_hash


class DatasetFreezeError(ValueError):
    """Raised when immutable dataset artifacts conflict or fail integrity checks."""


@dataclass(frozen=True)
class DatasetIdentityInputs:
    catalog_snapshot_id: str
    catalog_snapshot_hash: str
    label_policy_hash: str
    acquisition_snapshot_id: str
    acquisition_snapshot_hash: str
    raw_checksums: tuple[str, ...]
    preprocessing_config_hash: str
    bls_config_hash: str
    matching_config_hash: str
    dataset_policy_hash: str

    def portable_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["raw_checksums"] = sorted(set(self.raw_checksums))
        return value


def make_dataset_version(inputs: DatasetIdentityInputs) -> str:
    """Hash only immutable scientific content, never paths or execution time."""
    return f"dataset-{content_hash(inputs.portable_dict())}"


def _canonical_frame(frame: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    columns = sorted(frame.columns)
    return frame[columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _write_parquet(frame: pd.DataFrame, path: Path, sort_columns: list[str]) -> str:
    canonical = _canonical_frame(frame, sort_columns)
    canonical.to_parquet(path, engine="pyarrow", index=False)
    return sha256_file(path)


def freeze_dataset(
    dataset: BuiltDataset,
    *,
    identity: DatasetIdentityInputs,
    validation_summary: dict[str, Any],
    root: str | Path,
    created_at_utc: datetime | None = None,
) -> Path:
    """Write an immutable development dataset, reusing an identical existing freeze."""
    version = make_dataset_version(identity)
    directory = Path(root) / version
    if directory.exists():
        existing = load_frozen_dataset(directory, expected_version=version)
        for current, frozen, keys in (
            (dataset.candidates, existing.candidates, ["candidate_id"]),
            (
                dataset.candidate_event_matches,
                existing.candidate_event_matches,
                ["candidate_id", "event_id"],
            ),
            (dataset.catalog_events, existing.catalog_events, ["event_id"]),
            (
                dataset.artifact_lineage,
                existing.artifact_lineage,
                ["observation_group_id"],
            ),
        ):
            try:
                pd.testing.assert_frame_equal(
                    _canonical_frame(current, keys),
                    _canonical_frame(frozen, keys),
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise DatasetFreezeError(
                    "Existing dataset version has conflicting table content."
                ) from exc
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    try:
        checksums = {
            "candidates.parquet": _write_parquet(
                dataset.candidates, directory / "candidates.parquet", ["candidate_id"]
            ),
            "candidate_event_matches.parquet": _write_parquet(
                dataset.candidate_event_matches,
                directory / "candidate_event_matches.parquet",
                ["candidate_id", "event_id"],
            ),
            "catalog_events.parquet": _write_parquet(
                dataset.catalog_events, directory / "catalog_events.parquet", ["event_id"]
            ),
            "artifact_lineage.parquet": _write_parquet(
                dataset.artifact_lineage,
                directory / "artifact_lineage.parquet",
                ["observation_group_id"],
            ),
        }
        timestamp = created_at_utc or datetime.now(UTC)
        metadata = {
            "dataset_version": version,
            "dataset_role": "development",
            "created_at_utc": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "identity_inputs": identity.portable_dict(),
            "table_checksums": checksums,
        }
        (directory / "dataset_metadata.json").write_text(
            canonical_json(metadata) + "\n", encoding="utf-8"
        )
        (directory / "validation_summary.json").write_text(
            canonical_json(validation_summary) + "\n", encoding="utf-8"
        )
    except Exception:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        raise
    return directory


def load_frozen_dataset(
    directory: str | Path, *, expected_version: str | None = None
) -> BuiltDataset:
    """Reload a frozen dataset only after verifying metadata and table checksums."""
    root = Path(directory)
    metadata = json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8"))
    if expected_version is not None and metadata["dataset_version"] != expected_version:
        raise DatasetFreezeError("Dataset version does not match the requested identity.")
    for filename, checksum in metadata["table_checksums"].items():
        if sha256_file(root / filename) != checksum:
            raise DatasetFreezeError(f"Dataset table checksum mismatch: {filename}")
    return BuiltDataset(
        pd.read_parquet(root / "candidates.parquet"),
        pd.read_parquet(root / "candidate_event_matches.parquet"),
        pd.read_parquet(root / "catalog_events.parquet"),
        pd.read_parquet(root / "artifact_lineage.parquet"),
    )


def freeze_split_manifest(
    split: pd.DataFrame,
    *,
    dataset_version: str,
    root: str | Path,
) -> Path:
    """Write an immutable, replayable TIC assignment separately from dataset tables."""
    directory = Path(root) / dataset_version
    path = directory / "split_manifest.parquet"
    metadata_path = directory / "split_metadata.json"
    canonical = _canonical_frame(split, ["object_id"])
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(path) != metadata["split_manifest_sha256"]:
            raise DatasetFreezeError("Split manifest checksum mismatch.")
        existing = pd.read_parquet(path)
        pd.testing.assert_frame_equal(existing, canonical)
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    canonical.to_parquet(path, engine="pyarrow", index=False)
    metadata = {
        "dataset_version": dataset_version,
        "dataset_role": "development",
        "split_config_hash": str(canonical["split_config_hash"].iloc[0]),
        "split_seed": int(canonical["split_seed"].iloc[0]),
        "split_manifest_sha256": sha256_file(path),
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return directory
