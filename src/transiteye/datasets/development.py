"""Replayable assembly of the accepted real B031--B034 development dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import config_hash, load_config
from transiteye.datasets.builder import BuiltDataset, build_dataset
from transiteye.datasets.lineage import build_artifact_lineage, validate_frozen_candidate_ids
from transiteye.datasets.schemas import DatasetRole
from transiteye.datasets.splitter import enrich_split_manifest, grouped_split
from transiteye.datasets.validation import validate_dataset
from transiteye.datasets.versioning import (
    DatasetIdentityInputs,
    freeze_dataset,
    freeze_split_manifest,
    load_frozen_dataset,
    make_dataset_version,
)
from transiteye.serialization import content_hash


def _one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {description}; found {len(paths)}.")
    return paths[0]


def _real_artifacts(root: Path) -> tuple[BuiltDataset, DatasetIdentityInputs, dict[str, Any]]:
    dataset_config = load_config(root / "configs/datasets/mvp.yaml")
    preprocessing_config = load_config(root / "configs/preprocessing/mvp.yaml")
    bls_config = load_config(root / "configs/bls/mvp.yaml")
    if dataset_config.dataset is None or bls_config.bls is None:
        raise ValueError("Required dataset/BLS configuration is unavailable.")

    cohort_root = _one(
        list((root / "data/manifests").glob("pilot-toi-*/selected_target_events.parquet")),
        "frozen cohort event table",
    ).parent
    cohort_metadata = json.loads((cohort_root / "metadata.json").read_text(encoding="utf-8"))
    catalog_root = root / "data/catalogs" / str(cohort_metadata["snapshot_id"])
    catalog_metadata = json.loads((catalog_root / "metadata.json").read_text(encoding="utf-8"))
    product_root = root / "data/manifests/mast-products-b011-resumable"
    product_metadata = json.loads((product_root / "metadata.json").read_text(encoding="utf-8"))
    if (
        sha256_file(product_root / "normalized_products.parquet")
        != product_metadata["products_sha256"]
    ):
        raise ValueError("Frozen product snapshot checksum mismatch.")

    receipt_path = root / "data/manifests/acq-f3156f8ed0519d054e7b/download_receipts.parquet"
    receipts = pd.read_parquet(receipt_path)
    products = pd.read_parquet(product_root / "normalized_products.parquet")
    observations = pd.read_parquet(
        root / "data/manifests/mast-canonical-tess-spoc-b011/mast_observations.parquet"
    )
    events = pd.read_parquet(cohort_root / "selected_target_events.parquet")
    preprocessing_hash = config_hash(preprocessing_config)
    candidate_root = _one(
        [path for path in (root / "data/candidates").iterdir() if path.is_dir()],
        "frozen BLS configuration directory",
    )
    bls_hash = candidate_root.name

    processed_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    for receipt in receipts.sort_values("object_id", kind="stable").to_dict(orient="records"):
        checksum = str(receipt["sha256"])
        raw_path = root / "data/raw" / str(receipt["relative_raw_path"])
        if sha256_file(raw_path) != checksum:
            raise ValueError("Raw pilot checksum no longer matches its receipt.")
        key = checksum[:20]
        processed_path = root / "data/processed/mvp" / key / "cadences.parquet"
        candidate_path = candidate_root / f"{key}-candidates.parquet"
        match_path = candidate_root / f"{key}-matches.parquet"
        for path in (processed_path, candidate_path, match_path):
            if not path.is_file():
                raise ValueError(f"Required frozen development artifact is missing: {path.name}")
        group_id = f"og-{key}"
        processed_rows.append(
            {
                "source_raw_checksum": checksum,
                "processed_checksum": sha256_file(processed_path),
                "preprocessing_config_hash": preprocessing_hash,
            }
        )
        detection_rows.append(
            {
                "source_raw_checksum": checksum,
                "observation_group_id": group_id,
                "bls_config_hash": bls_hash,
                "status": "success",
            }
        )
        candidate_frame = pd.read_parquet(candidate_path)
        candidate_frame["observation_group_id"] = group_id
        candidate_frames.append(candidate_frame)
        match_frames.append(pd.read_parquet(match_path))

    processed = pd.DataFrame(processed_rows)
    detections = pd.DataFrame(detection_rows)
    lineage = build_artifact_lineage(products, receipts, processed, detections)
    frozen_candidates = pd.concat(candidate_frames, ignore_index=True)
    validate_frozen_candidate_ids(frozen_candidates, lineage)
    frozen_matches = pd.concat(match_frames, ignore_index=True)
    matching_hash = content_hash(
        {
            "policy_version": "bls-ephemeris-match-v1",
            "period_match_tolerance": bls_config.bls.period_match_tolerance,
            "phase_match_tolerance": bls_config.bls.phase_match_tolerance,
            "harmonic_ratios": [0.5, 1.0, 2.0],
        }
    )
    dataset = build_dataset(
        frozen_candidates=frozen_candidates,
        frozen_matches=frozen_matches,
        catalog_events=events,
        observations=observations,
        products=products,
        receipts=receipts,
        artifact_lineage=lineage,
        matching_config_hash=matching_hash,
    )
    label_hash = content_hash(
        {
            "policy_version": dataset_config.dataset.label_policy_version,
            "mapping": {
                "CP": "positive",
                "KP": "positive",
                "FP": "negative",
                "FA": "negative",
                "PC": "unlabeled",
                "APC": "unlabeled",
                "UNKNOWN": "unlabeled",
            },
        }
    )
    identity = DatasetIdentityInputs(
        catalog_snapshot_id=str(catalog_metadata["snapshot_id"]),
        catalog_snapshot_hash=str(catalog_metadata["raw_response_sha256"]),
        label_policy_hash=label_hash,
        acquisition_snapshot_id=str(product_metadata["snapshot_id"]),
        acquisition_snapshot_hash=str(product_metadata["products_sha256"]),
        raw_checksums=tuple(sorted(receipts["sha256"].astype(str))),
        preprocessing_config_hash=preprocessing_hash,
        bls_config_hash=bls_hash,
        matching_config_hash=matching_hash,
        dataset_policy_hash=config_hash(dataset_config),
    )
    split = grouped_split(
        events["object_id"].astype(str).tolist(),
        settings=dataset_config.dataset,
        master_seed=dataset_config.reproducibility.master_seed,
        role=DatasetRole.DEVELOPMENT,
    )
    split = enrich_split_manifest(split, dataset.candidates, dataset.catalog_events)
    qa = validate_dataset(dataset, split)
    return dataset, identity, {"qa": qa, "split": split}


def build_current_development_dataset(root: str | Path) -> tuple[Path, Path, str]:
    """Validate, freeze, reload, and return the accepted real development dataset."""
    repository = Path(root)
    dataset, identity, context = _real_artifacts(repository)
    version = make_dataset_version(identity)
    dataset_path = freeze_dataset(
        dataset,
        identity=identity,
        validation_summary=context["qa"],
        root=repository / "data/datasets",
    )
    split_path = freeze_split_manifest(
        context["split"], dataset_version=version, root=repository / "data/splits"
    )
    replay = load_frozen_dataset(dataset_path, expected_version=version)
    for original, loaded in (
        (dataset.candidates, replay.candidates),
        (dataset.candidate_event_matches, replay.candidate_event_matches),
        (dataset.catalog_events, replay.catalog_events),
        (dataset.artifact_lineage, replay.artifact_lineage),
    ):
        pd.testing.assert_frame_equal(
            original.sort_index(axis=1)
            .sort_values(original.columns[0], kind="stable")
            .reset_index(drop=True),
            loaded.sort_index(axis=1)
            .sort_values(original.columns[0], kind="stable")
            .reset_index(drop=True),
            check_dtype=False,
        )
    return dataset_path, split_path, version
