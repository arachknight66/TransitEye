from __future__ import annotations

from pathlib import Path

import pytest

from transiteye.datasets.labels import (
    LabelRecord,
    LabelSource,
    MorphologyLabel,
    read_labels,
    write_labels,
)
from transiteye.datasets.splits import (
    DatasetSplit,
    SplitAssignment,
    SplitManifest,
    read_split_manifest,
    validate_labels_against_split,
    write_split_manifest,
)


def _label(source_group_id: str, label: MorphologyLabel) -> LabelRecord:
    return LabelRecord(
        source_group_id=source_group_id,
        target_id=source_group_id.removeprefix("source-"),
        label=label,
        label_source=LabelSource.MANUAL_REVIEW,
        evidence_uri="https://example.invalid/evidence",
        evidence_identifier=source_group_id,
        review_status="reviewed",
        snapshot_date="2026-09-14",
        sectors=(1,),
    )


def _assignments() -> tuple[SplitAssignment, ...]:
    return (
        SplitAssignment("source-1", DatasetSplit.TRAIN),
        SplitAssignment("source-2", DatasetSplit.CALIBRATION),
        SplitAssignment("source-3", DatasetSplit.SELECTION),
        SplitAssignment("source-4", DatasetSplit.TEST),
    )


def test_labels_and_split_manifests_round_trip(tmp_path: Path) -> None:
    labels = (
        _label("source-1", MorphologyLabel.TRANSIT_LIKE),
        _label("source-2", MorphologyLabel.ECLIPSE_LIKE),
        _label("source-3", MorphologyLabel.STELLAR_VARIABILITY),
        _label("source-4", MorphologyLabel.OTHER_ARTIFACT),
    )
    labels_path = tmp_path / "labels.json"
    split_path = tmp_path / "splits.json"
    write_labels(labels, labels_path)
    split_manifest = SplitManifest.create(_assignments())
    write_split_manifest(split_manifest, split_path)
    loaded_labels = read_labels(labels_path)
    loaded_splits = read_split_manifest(split_path)
    validate_labels_against_split(loaded_labels, loaded_splits)
    assert loaded_splits.manifest_id == split_manifest.manifest_id


def test_split_validation_rejects_source_overlap() -> None:
    with pytest.raises(ValueError, match="more than one split"):
        SplitManifest.create(
            (
                SplitAssignment("source-1", DatasetSplit.TRAIN),
                SplitAssignment("source-1", DatasetSplit.TEST),
                SplitAssignment("source-2", DatasetSplit.CALIBRATION),
                SplitAssignment("source-3", DatasetSplit.SELECTION),
            )
        )


def test_split_validation_rejects_label_drift() -> None:
    manifest = SplitManifest.create(_assignments())
    labels = (
        _label("source-1", MorphologyLabel.TRANSIT_LIKE),
        _label("source-2", MorphologyLabel.ECLIPSE_LIKE),
        _label("source-3", MorphologyLabel.STELLAR_VARIABILITY),
        _label("unexpected", MorphologyLabel.OTHER_ARTIFACT),
    )
    with pytest.raises(ValueError, match="Split/label mismatch"):
        validate_labels_against_split(labels, manifest)
