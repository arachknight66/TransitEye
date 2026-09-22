"""Leakage-safe source-group splits for labels and all derivative data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from transiteye.datasets.labels import LabelRecord


class DatasetSplit(StrEnum):
    """Partitions with distinct allowed uses."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    SELECTION = "selection"
    TEST = "test"
    SECTOR_SHIFT = "sector_shift"


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One source group and every derivative assigned to exactly one partition."""

    source_group_id: str
    split: DatasetSplit


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A frozen source-group partition manifest."""

    manifest_id: str
    assignments: tuple[SplitAssignment, ...]
    schema_version: str = "1"

    @classmethod
    def create(cls, assignments: tuple[SplitAssignment, ...]) -> SplitManifest:
        validate_assignments(assignments)
        canonical = json.dumps(
            [
                {"source_group_id": item.source_group_id, "split": item.split.value}
                for item in assignments
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return cls(manifest_id=hashlib.sha256(canonical).hexdigest()[:16], assignments=assignments)


def validate_assignments(assignments: tuple[SplitAssignment, ...]) -> None:
    """Ensure every source group occurs once and all required splits are represented."""

    source_groups = [assignment.source_group_id for assignment in assignments]
    if len(source_groups) != len(set(source_groups)):
        raise ValueError("A source group cannot appear in more than one split.")
    required = {
        DatasetSplit.TRAIN,
        DatasetSplit.CALIBRATION,
        DatasetSplit.SELECTION,
        DatasetSplit.TEST,
    }
    present = {assignment.split for assignment in assignments}
    missing = required.difference(present)
    if missing:
        raise ValueError(
            f"Split manifest missing required splits: {', '.join(sorted(item.value for item in missing))}"
        )


def validate_labels_against_split(
    labels: tuple[LabelRecord, ...], split_manifest: SplitManifest
) -> None:
    """Reject label/split drift before any training or calibration consumes the cohort."""

    label_groups = {label.source_group_id for label in labels}
    assigned_groups = {assignment.source_group_id for assignment in split_manifest.assignments}
    if label_groups != assigned_groups:
        missing = label_groups.difference(assigned_groups)
        unexpected = assigned_groups.difference(label_groups)
        raise ValueError(
            f"Split/label mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def write_split_manifest(manifest: SplitManifest, path: Path) -> None:
    """Write one immutable partition record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": manifest.schema_version,
        "manifest_id": manifest.manifest_id,
        "assignments": [
            {"source_group_id": item.source_group_id, "split": item.split.value}
            for item in manifest.assignments
        ],
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_split_manifest(path: Path) -> SplitManifest:
    """Read and validate a frozen partition record."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "1":
        raise ValueError("Unsupported split schema version.")
    assignments = tuple(
        SplitAssignment(item["source_group_id"], DatasetSplit(item["split"]))
        for item in document.get("assignments", [])
    )
    manifest = SplitManifest.create(assignments)
    if manifest.manifest_id != document.get("manifest_id"):
        raise ValueError("Split manifest ID does not match its contents.")
    return manifest
