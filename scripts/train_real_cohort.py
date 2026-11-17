"""Train the morphology model on the real, catalog-labelled MAST cohort only.

The test split is deliberately not opened here. Feature extraction uses only the
observed light curves, with no label-conditioned feature edits and no synthetic
augmentation. Training, calibration, and selection targets are grouped by TIC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from transiteye.classification.models import (
    MORPHOLOGY_FEATURE_NAMES,
    TARGET_MORPHOLOGY_CLASSES,
)
from transiteye.classification.train import _extract_lc_feature_vector, train_and_select_classifier
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import (
    DatasetSplit,
    read_split_manifest,
    validate_labels_against_split,
)
from transiteye.io.fits import read_tess_light_curve
from transiteye.io.manifest import read_manifest

LOGGER = logging.getLogger("transiteye.train_real_cohort")


def _extract_source_features(
    task: tuple[str, str, str | None, tuple[str, ...]],
) -> tuple[str, np.ndarray | None, str | None]:
    """Extract one source's features in a bounded worker process."""
    target_id, path_text, expected_sha256, feature_names = task
    fits_path = Path(path_text)
    try:
        if expected_sha256:
            digest = hashlib.sha256(fits_path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                return target_id, None, "product checksum mismatch"
        light_curve = read_tess_light_curve(fits_path)
        vector = _extract_lc_feature_vector(
            light_curve, feature_names, device="cpu"
        )
        if vector is None or not np.all(np.isfinite(vector)):
            return target_id, None, "no candidate or non-finite features"
        return target_id, vector, None
    except Exception as exc:
        return target_id, None, f"feature extraction error: {exc!r}"


def train_real_cohort(
    data_dir: Path,
    products_path: Path,
    labels_path: Path,
    splits_path: Path,
    output_dir: Path,
    model_id: str = "morphology_real_catalog_v1",
    workers: int = 4,
) -> dict[str, Any]:
    """Extract real features and train/calibrate without opening the locked test split."""
    labels = read_labels(labels_path)
    if workers < 1:
        raise ValueError("workers must be at least one.")
    splits = read_split_manifest(splits_path)
    validate_labels_against_split(labels, splits)
    split_by_group = {item.source_group_id: item.split for item in splits.assignments}

    products = read_manifest(products_path)
    product_by_target: dict[str, Any] = {}
    for product in products.entries:
        current = product_by_target.get(product.target_id)
        if current is None or (int(product.sector or -1), product.local_filename or "") > (
            int(current.sector or -1),
            current.local_filename or "",
        ):
            product_by_target[product.target_id] = product

    class_names = [item.value for item in TARGET_MORPHOLOGY_CLASSES]
    class_indices = {name: index for index, name in enumerate(class_names)}

    feature_rows: dict[DatasetSplit, list[np.ndarray]] = {
        split: []
        for split in (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION, DatasetSplit.SELECTION)
    }
    labels_by_split: dict[DatasetSplit, list[int]] = {split: [] for split in feature_rows}
    groups_train: list[str] = []
    excluded: list[dict[str, str]] = []
    support_by_split: dict[str, Counter[str]] = {
        split.value: Counter() for split in feature_rows
    }

    total_training_labels = sum(
        split_by_group[item.source_group_id] != DatasetSplit.TEST for item in labels
    )
    tasks: list[tuple[str, str, str | None, tuple[str, ...]]] = []
    task_labels: list[tuple[Any, DatasetSplit]] = []
    processed = 0
    for label in labels:
        split = split_by_group[label.source_group_id]
        if split == DatasetSplit.TEST:
            continue  # Do not even open the locked test FITS files in this training command.
        processed += 1
        if processed == 1 or processed % 10 == 0 or processed == total_training_labels:
            LOGGER.info("Extracting source %d/%d (TIC %s)", processed, total_training_labels, label.target_id)
        if split not in feature_rows:
            excluded.append(
                {"target_id": label.target_id, "reason": f"unsupported split {split.value}"}
            )
            continue

        product = product_by_target.get(label.target_id)
        if product is None:
            excluded.append({"target_id": label.target_id, "reason": "no product manifest entry"})
            continue
        fits_path = data_dir / product.resolved_filename()
        if not fits_path.is_file():
            excluded.append({"target_id": label.target_id, "reason": "FITS file missing"})
            continue

        expected_sha256 = product.expected_sha256
        if expected_sha256:
            digest = hashlib.sha256(fits_path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                excluded.append({"target_id": label.target_id, "reason": "product checksum mismatch"})
                continue

        class_name = label.label.value
        if class_name not in class_indices:
            excluded.append(
                {"target_id": label.target_id, "reason": f"unsupported label {class_name}"}
            )
            continue
        tasks.append((label.target_id, str(fits_path), expected_sha256, MORPHOLOGY_FEATURE_NAMES))
        task_labels.append((label, split))

    # Use spawn and CPU workers so CUDA state is never inherited or contended between workers.
    # executor.map preserves the frozen label order while keeping the queue bounded internally.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for index, (target_id, vector, error) in enumerate(pool.map(_extract_source_features, tasks), start=1):
            if index == 1 or index % 10 == 0 or index == len(tasks):
                LOGGER.info("Completed feature extraction %d/%d (TIC %s)", index, len(tasks), target_id)
            if error is not None or vector is None:
                excluded.append({"target_id": target_id, "reason": error or "feature extraction failed"})
                continue
            label, split = task_labels[index - 1]
            class_name = label.label.value
            feature_rows[split].append(vector)
            labels_by_split[split].append(class_indices[class_name])
            support_by_split[split.value][class_name] += 1
            if split == DatasetSplit.TRAIN:
                groups_train.append(label.source_group_id)

    for split, counts in support_by_split.items():
        LOGGER.info("Usable %s examples by class: %s", split, dict(counts))
    if any(not feature_rows[split] for split in feature_rows):
        raise RuntimeError(f"At least one model partition has no usable examples: {support_by_split}")
    if len(set(groups_train)) != len(groups_train):
        raise RuntimeError("Training partition contains duplicate source groups.")
    if any(support_by_split[split.value][name] == 0 for split in feature_rows for name in class_names):
        raise RuntimeError("Every train/calibration/selection split must contain every class.")

    x_train = np.vstack(feature_rows[DatasetSplit.TRAIN])
    y_train = np.asarray(labels_by_split[DatasetSplit.TRAIN], dtype=np.int64)
    x_cal = np.vstack(feature_rows[DatasetSplit.CALIBRATION])
    y_cal = np.asarray(labels_by_split[DatasetSplit.CALIBRATION], dtype=np.int64)
    x_sel = np.vstack(feature_rows[DatasetSplit.SELECTION])
    y_sel = np.asarray(labels_by_split[DatasetSplit.SELECTION], dtype=np.int64)

    bundle, comparison = train_and_select_classifier(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_cal=x_cal,
        y_cal=y_cal,
        x_sel=x_sel,
        y_sel=y_sel,
        feature_names=MORPHOLOGY_FEATURE_NAMES,
        train_manifest_id=splits.manifest_id,
        synthetic_sample_count=0,
    )
    bundle.metadata = replace(
        bundle.metadata,
        model_id=model_id,
        extra={
            **bundle.metadata.extra,
            "dataset_kind": "real_mast_catalog_labelled",
            "label_snapshot": str(labels_path),
            "split_manifest": str(splits_path),
            "product_manifest": str(products_path),
            "used_synthetic_augmentation": False,
            "label_conditioned_feature_transforms": False,
            "test_split_opened": False,
            "retrained_from_model_id": None,
            "usable_support_by_split": {
                split: dict(counts) for split, counts in support_by_split.items()
            },
        },
    )
    bundle.save(output_dir)

    report = {
        "model_id": bundle.metadata.model_id,
        "retrained_from_model_id": None,
        "estimator_name": bundle.metadata.estimator_name,
        "feature_names": list(MORPHOLOGY_FEATURE_NAMES),
        "training_manifest_id": splits.manifest_id,
        "usable_support_by_split": {
            split: dict(counts) for split, counts in support_by_split.items()
        },
        "train_sources": len(feature_rows[DatasetSplit.TRAIN]),
        "calibration_sources": len(feature_rows[DatasetSplit.CALIBRATION]),
        "selection_sources": len(feature_rows[DatasetSplit.SELECTION]),
        "test_split_opened": False,
        "synthetic_augmentation": False,
        "label_conditioned_feature_transforms": False,
        "excluded_targets": excluded,
        "excluded_count": len(excluded),
        "cv_comparison": comparison,
        "calibration_fit_diagnostics": {
            "brier_score_on_calibration_fit_data": bundle.metadata.extra.get("brier_score"),
            "ece_on_calibration_fit_data": bundle.metadata.extra.get("ece"),
            "note": "These are fit-partition diagnostics, not independent performance estimates.",
        },
        "abstention_thresholds": {
            "tau_confidence": bundle.metadata.tau_conf,
            "tau_margin": bundle.metadata.tau_margin,
        },
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".transiteye-workspace/data/real_catalog_cohort_v1"))
    parser.add_argument("--products", type=Path, default=Path("manifests/real_catalog_cohort.v1.products.json"))
    parser.add_argument("--labels", type=Path, default=Path("manifests/real_catalog_cohort.v1.labels.json"))
    parser.add_argument("--splits", type=Path, default=Path("manifests/real_catalog_cohort.v1.splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(".transiteye-workspace/models/morphology_real_catalog_v1"))
    parser.add_argument("--model-id", default="morphology_real_catalog_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_real_cohort(
        args.data_dir, args.products, args.labels, args.splits, args.output_dir,
        args.model_id, args.workers,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "excluded_targets"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
