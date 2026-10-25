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


def test_frozen_pilot_manifests() -> None:
    manifest_dir = Path("manifests")
    labels_path = manifest_dir / "pilot_labels.v1.json"
    splits_path = manifest_dir / "pilot_splits.v1.json"
    products_path = manifest_dir / "pilot_products.v1.json"

    assert labels_path.is_file(), "pilot_labels.v1.json is missing"
    assert splits_path.is_file(), "pilot_splits.v1.json is missing"
    assert products_path.is_file(), "pilot_products.v1.json is missing"

    from transiteye.io.manifest import read_manifest

    labels = read_labels(labels_path)
    splits = read_split_manifest(splits_path)
    products = read_manifest(products_path)

    validate_labels_against_split(labels, splits)

    assert len(labels) == 68
    assert len(splits.assignments) == 68
    assert len(products.entries) >= 68

    # Check class distributions
    classes = {m: 0 for m in MorphologyLabel}
    ambiguous_count = 0
    for label in labels:
        classes[label.label] += 1
        if label.ambiguous:
            ambiguous_count += 1
        assert label.evidence_uri.startswith(("https://", "http://"))
        assert len(label.sectors) >= 1

    for count in classes.values():
        assert count == 17

    assert ambiguous_count == 4

    # Check split assignments and ensure zero overlap across splits
    split_map = {a.source_group_id: a.split for a in splits.assignments}
    assert len(split_map) == len(splits.assignments)

    split_counts = {s: 0 for s in DatasetSplit}
    for s in split_map.values():
        split_counts[s] += 1

    assert split_counts[DatasetSplit.TRAIN] == 36
    assert split_counts[DatasetSplit.CALIBRATION] == 8
    assert split_counts[DatasetSplit.SELECTION] == 8
    assert split_counts[DatasetSplit.TEST] == 8
    assert split_counts[DatasetSplit.SECTOR_SHIFT] == 8

