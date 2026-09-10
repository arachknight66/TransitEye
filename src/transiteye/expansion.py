"""Execution of the frozen real-cohort expansion without scientific retuning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.acquisition.downloader import (
    AstroqueryProductFetcher,
    DownloadReceipt,
    download_product,
)
from transiteye.acquisition.expansion import (
    ExpansionManifest,
    build_expansion_manifest,
    load_expansion_manifest,
    write_expansion_manifest,
)
from transiteye.config import config_hash, load_config
from transiteye.datasets.builder import build_dataset
from transiteye.datasets.lineage import build_artifact_lineage, validate_frozen_candidate_ids
from transiteye.datasets.schemas import DatasetRole
from transiteye.datasets.splitter import enrich_split_manifest, grouped_split
from transiteye.datasets.validation import validate_dataset
from transiteye.datasets.versioning import (
    DatasetIdentityInputs,
    freeze_dataset,
    freeze_split_manifest,
    make_dataset_version,
)
from transiteye.detection.bls import run_bls
from transiteye.detection.matching import match_candidates
from transiteye.detection.peaks import extract_peaks
from transiteye.identifiers import make_observation_id
from transiteye.preprocessing.pipeline import preprocess, read_tess_lightcurve
from transiteye.preprocessing.qa import plot_preprocessing_qa
from transiteye.serialization import canonical_json, content_hash


class ExpansionExecutionError(ValueError):
    """Raised for a systemic frozen-input or immutable-output failure."""


@dataclass(frozen=True)
class ExpansionRunResult:
    manifest_id: str
    dataset_version: str
    selected_products: int
    downloaded_products: int
    preprocessed_products: int
    searched_products: int
    dataset_path: Path
    split_path: Path
    qa_summary: dict[str, Any]


def _write_table_once(frame: pd.DataFrame, path: Path, *, sort_columns: list[str]) -> None:
    """Freeze a derived table, accepting only a byte-equivalent logical replay."""
    ordered = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if path.exists():
        existing = (
            pd.read_parquet(path).sort_values(sort_columns, kind="stable").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(existing, ordered, check_dtype=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    ordered.to_parquet(partial, engine="pyarrow", index=False)
    os.replace(partial, path)


def _checkpoint(frame: pd.DataFrame, path: Path) -> None:
    """Persist resumable operational state; this is not a scientific identity artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    frame.to_parquet(partial, engine="pyarrow", index=False)
    os.replace(partial, path)


def _receipt_row(
    receipt: DownloadReceipt, *, attempts: int, error: str | None = None
) -> dict[str, object]:
    return receipt.__dict__ | {"attempts": attempts, "error": error}


def _download_selected(
    selected: pd.DataFrame,
    *,
    root: Path,
    manifest_root: Path,
    manifest_id: str,
) -> tuple[pd.DataFrame, int]:
    """Download each selected row with resumable per-row progress and terminal states."""
    final_path = manifest_root / "download_receipts.parquet"
    checkpoint_path = root / "data/interim" / manifest_id / "download_progress.parquet"
    if final_path.exists():
        receipts = pd.read_parquet(final_path)
    elif checkpoint_path.exists():
        receipts = pd.read_parquet(checkpoint_path)
    else:
        receipts = pd.DataFrame()
    if receipts.empty:
        completed_uris: set[str] = set()
    else:
        completed_uris = set(
            receipts.loc[receipts["status"].isin(["downloaded", "reused"]), "data_uri"]
            .astype(str)
            .tolist()
        )
    rows: list[dict[Any, Any]] = receipts.to_dict(orient="records")
    fetcher = AstroqueryProductFetcher()
    for _, product in selected.iterrows():
        uri = str(product["data_uri"])
        if uri in completed_uris:
            continue
        errors: list[str] = []
        for attempt in range(1, 4):
            try:
                receipt = download_product(product, fetcher=fetcher, raw_root=root / "data/raw")
                rows.append(_receipt_row(receipt, attempts=attempt))
                completed_uris.add(uri)
                break
            except Exception as exc:  # Independent archive failures are recorded, not hidden.
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        else:
            rows.append(
                {
                    "object_id": str(product["object_id"]),
                    "sector": int(product["sector"]),
                    "product_filename": str(product["product_filename"]),
                    "data_uri": uri,
                    "relative_raw_path": None,
                    "expected_size_bytes": int(product["product_size_bytes"]),
                    "actual_size_bytes": None,
                    "sha256": None,
                    "downloaded_at_utc": None,
                    "status": "terminal_failure",
                    "attempts": 3,
                    "error": " | ".join(errors),
                }
            )
        _checkpoint(pd.DataFrame(rows), checkpoint_path)
    output = pd.DataFrame(rows)
    if output["data_uri"].duplicated().any():
        raise ExpansionExecutionError("Download checkpoint has duplicate product URIs.")
    _write_table_once(output, final_path, sort_columns=["object_id", "sector", "data_uri"])
    reuse_count = 0
    for _, product in selected.iterrows():
        prior = output.loc[output["data_uri"] == str(product["data_uri"])]
        if prior.empty or prior.iloc[0]["status"] not in {"downloaded", "reused"}:
            continue
        receipt = download_product(
            product,
            fetcher=fetcher,
            raw_root=root / "data/raw",
            known_checksum=str(prior.iloc[0]["sha256"]),
        )
        if receipt.status != "reused":
            raise ExpansionExecutionError("A completed raw product was not reused on rerun.")
        reuse_count += 1
    return output, reuse_count


def _preprocess_selected(
    receipts: pd.DataFrame,
    products: pd.DataFrame,
    *,
    root: Path,
    manifest_root: Path,
    preprocessing_hash: str,
    settings: Any,
    manifest_id: str,
) -> pd.DataFrame:
    """Run the frozen preprocessing configuration once per completed raw product."""
    output_path = manifest_root / "preprocessing_receipts.parquet"
    if output_path.exists():
        return pd.read_parquet(output_path)
    rows: list[dict[str, object]] = []
    successful = receipts.loc[receipts["status"].isin(["downloaded", "reused"])]
    product_lookup = products.set_index("data_uri")
    for receipt in successful.sort_values(["object_id", "sector", "data_uri"]).to_dict(
        orient="records"
    ):
        checksum = str(receipt["sha256"])
        product = product_lookup.loc[str(receipt["data_uri"])]
        raw_path = root / "data/raw" / str(receipt["relative_raw_path"])
        processed_dir = root / "data/processed/mvp" / checksum[:20]
        cadence_path = processed_dir / "cadences.parquet"
        metadata_path = processed_dir / "metadata.json"
        try:
            observation_id = make_observation_id(
                object_id=str(receipt["object_id"]),
                sector=int(product["sector"]),
                author=str(product["author"]),
                cadence_seconds=float(product["exposure_seconds"]),
                product_id=str(receipt["data_uri"]),
                checksum_sha256=checksum,
            )
            if cadence_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("source_raw_sha256") != checksum
                    or metadata.get("preprocessing_config_hash") != preprocessing_hash
                ):
                    raise ExpansionExecutionError(
                        "Existing processed artifact conflicts with frozen lineage."
                    )
                status = "reused"
            else:
                source = read_tess_lightcurve(raw_path, observation_id=observation_id)
                processed = preprocess(
                    source,
                    gap_days=settings.gap_days,
                    trend_window=settings.trend_window_cadences,
                    positive_spike_mad=settings.positive_spike_mad,
                )
                processed_dir.mkdir(parents=True, exist_ok=True)
                partial = cadence_path.with_name(f".{cadence_path.name}.part")
                processed.data.to_parquet(partial, engine="pyarrow", index=False)
                os.replace(partial, cadence_path)
                metadata = processed.metadata | {
                    "preprocessing_config_hash": preprocessing_hash,
                    "source_product_id": str(receipt["data_uri"]),
                }
                metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
                qa_path = (
                    root / "reports/figures/expansion" / manifest_id / f"{checksum[:20]}-qa.png"
                )
                plot_preprocessing_qa(processed, qa_path)
                status = "processed"
            rows.append(
                {
                    "object_id": receipt["object_id"],
                    "data_uri": receipt["data_uri"],
                    "source_raw_checksum": checksum,
                    "observation_id": observation_id,
                    "processed_checksum": sha256_file(cadence_path),
                    "preprocessing_config_hash": preprocessing_hash,
                    "status": status,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "object_id": receipt["object_id"],
                    "data_uri": receipt["data_uri"],
                    "source_raw_checksum": checksum,
                    "observation_id": None,
                    "processed_checksum": None,
                    "preprocessing_config_hash": preprocessing_hash,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    output = pd.DataFrame(rows)
    _write_table_once(output, output_path, sort_columns=["object_id", "data_uri"])
    return output


def _matching_hash(bls_settings: Any) -> str:
    return content_hash(
        {
            "policy_version": "bls-ephemeris-match-v1",
            "period_match_tolerance": bls_settings.period_match_tolerance,
            "phase_match_tolerance": bls_settings.phase_match_tolerance,
            "harmonic_ratios": [0.5, 1.0, 2.0],
        }
    )


def _detect_selected(
    preprocessing_receipts: pd.DataFrame,
    products: pd.DataFrame,
    events: pd.DataFrame,
    *,
    root: Path,
    manifest_root: Path,
    preprocessing_hash: str,
    settings: Any,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], str]:
    """Blindly search each successful processed product, then perform post-freeze matching."""
    output_path = manifest_root / "detection_receipts.parquet"
    expected_hash = content_hash(
        {"bls": settings.model_dump(mode="json"), "preprocessing_hash": preprocessing_hash}
    )
    candidate_root = root / "data/candidates" / expected_hash
    if output_path.exists():
        receipts = pd.read_parquet(output_path)
        cached_candidate_frames = [
            pd.read_parquet(candidate_root / str(row["candidate_filename"])).assign(
                observation_group_id=str(row["observation_group_id"])
            )
            for row in receipts.loc[receipts["status"] == "success"].to_dict(orient="records")
        ]
        cached_match_frames = [
            pd.read_parquet(candidate_root / str(row["match_filename"]))
            for row in receipts.loc[receipts["status"] == "success"].to_dict(orient="records")
        ]
        return receipts, cached_candidate_frames, cached_match_frames, expected_hash

    rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    successful = preprocessing_receipts.loc[
        preprocessing_receipts["status"].isin(["processed", "reused"])
    ]
    for item in successful.sort_values(["object_id", "data_uri"]).to_dict(orient="records"):
        checksum = str(item["source_raw_checksum"])
        key = checksum[:20]
        group_id = f"og-{key}"
        candidate_path = candidate_root / f"{key}-candidates.parquet"
        periodogram_path = candidate_root / f"{key}-periodogram.parquet"
        match_path = candidate_root / f"{key}-matches.parquet"
        try:
            if candidate_path.exists() and periodogram_path.exists() and match_path.exists():
                candidates = pd.read_parquet(candidate_path)
                matches = pd.read_parquet(match_path)
                status = "reused"
            else:
                cadences = pd.read_parquet(root / "data/processed/mvp" / key / "cadences.parquet")
                result = run_bls(
                    cadences,
                    settings=settings,
                    observation_group_id=group_id,
                    preprocessing_hash=preprocessing_hash,
                )
                if result.config_hash != expected_hash:
                    raise ExpansionExecutionError(
                        "BLS result config hash differs from frozen configuration."
                    )
                candidate_root.mkdir(parents=True, exist_ok=True)
                _write_table_once(result.periodogram, periodogram_path, sort_columns=["period"])
                candidates = extract_peaks(result, settings)
                _write_table_once(candidates, candidate_path, sort_columns=["rank"])
                # Matching is intentionally executed only after BLS candidates are frozen.
                event_subset = events.loc[
                    (events["object_id"] == item["object_id"])
                    & events["period_days"].notna()
                    & events["transit_epoch_bjd"].notna()
                ]
                matches = match_candidates(candidates, event_subset, settings)
                if not matches.empty:
                    matches["object_id"] = item["object_id"]
                    matches["source_disposition"] = matches["toi_id"].map(
                        event_subset.set_index("toi_id")["source_disposition"]
                    )
                _write_table_once(matches, match_path, sort_columns=["candidate_id", "toi_id"])
                status = "success"
            candidate_frames.append(candidates.assign(observation_group_id=group_id))
            match_frames.append(matches)
            rows.append(
                {
                    "object_id": item["object_id"],
                    "data_uri": item["data_uri"],
                    "source_raw_checksum": checksum,
                    "observation_group_id": group_id,
                    "bls_config_hash": expected_hash,
                    "status": "success" if status == "success" else "reused",
                    "periodogram_filename": periodogram_path.name,
                    "candidate_filename": candidate_path.name,
                    "match_filename": match_path.name,
                    "periodogram_sha256": sha256_file(periodogram_path),
                    "candidate_sha256": sha256_file(candidate_path),
                    "match_sha256": sha256_file(match_path),
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "object_id": item["object_id"],
                    "data_uri": item["data_uri"],
                    "source_raw_checksum": checksum,
                    "observation_group_id": group_id,
                    "bls_config_hash": expected_hash,
                    "status": "failed",
                    "periodogram_filename": None,
                    "candidate_filename": None,
                    "match_filename": None,
                    "periodogram_sha256": None,
                    "candidate_sha256": None,
                    "match_sha256": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    receipts = pd.DataFrame(rows)
    _write_table_once(receipts, output_path, sort_columns=["object_id", "data_uri"])
    return receipts, candidate_frames, match_frames, expected_hash


def _build_expanded_dataset(
    *,
    root: Path,
    manifest: ExpansionManifest,
    manifest_root: Path,
    download_receipts: pd.DataFrame,
    preprocessing_receipts: pd.DataFrame,
    detection_receipts: pd.DataFrame,
    candidate_frames: list[pd.DataFrame],
    match_frames: list[pd.DataFrame],
    preprocessing_hash: str,
    bls_hash: str,
    matching_hash: str,
) -> tuple[Path, Path, str, dict[str, Any]]:
    """Regenerate B031--B034 tables from expanded frozen artifacts."""
    cohort_root = next(
        (root / "data/manifests").glob("pilot-toi-*/selected_target_events.parquet")
    ).parent
    cohort_metadata = json.loads((cohort_root / "metadata.json").read_text(encoding="utf-8"))
    catalog_metadata = json.loads(
        (root / "data/catalogs" / str(cohort_metadata["snapshot_id"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    product_root = root / "data/manifests/mast-products-b011-resumable"
    product_metadata = json.loads((product_root / "metadata.json").read_text(encoding="utf-8"))
    products = pd.read_parquet(product_root / "normalized_products.parquet")
    observations = pd.read_parquet(
        root / "data/manifests/mast-canonical-tess-spoc-b011/mast_observations.parquet"
    )
    events = pd.read_parquet(cohort_root / "selected_target_events.parquet")
    usable_downloads = download_receipts.loc[
        download_receipts["status"].isin(["downloaded", "reused"])
    ].copy()
    usable_preprocessed = preprocessing_receipts.loc[
        preprocessing_receipts["status"].isin(["processed", "reused"])
    ].copy()
    usable_detection = detection_receipts.loc[
        detection_receipts["status"].isin(["success", "reused", "failed"])
    ].copy()
    detection_lineage = usable_detection[
        ["source_raw_checksum", "observation_group_id", "bls_config_hash", "status"]
    ].rename(columns={"status": "detection_status"})
    complete = usable_downloads.merge(
        usable_preprocessed[
            ["source_raw_checksum", "processed_checksum", "preprocessing_config_hash"]
        ],
        left_on="sha256",
        right_on="source_raw_checksum",
        how="inner",
    ).merge(
        detection_lineage,
        left_on="sha256",
        right_on="source_raw_checksum",
        how="inner",
    )
    lineage = build_artifact_lineage(
        products,
        complete[["object_id", "data_uri", "sha256"]],
        complete[["sha256", "processed_checksum", "preprocessing_config_hash"]].rename(
            columns={"sha256": "source_raw_checksum"}
        ),
        complete[["sha256", "observation_group_id", "bls_config_hash", "detection_status"]].rename(
            columns={"sha256": "source_raw_checksum", "detection_status": "status"}
        ),
    )
    frozen_candidates = pd.concat(candidate_frames, ignore_index=True)
    frozen_matches = pd.concat(match_frames, ignore_index=True)
    validate_frozen_candidate_ids(frozen_candidates, lineage)
    dataset = build_dataset(
        frozen_candidates=frozen_candidates,
        frozen_matches=frozen_matches,
        catalog_events=events,
        observations=observations,
        products=products,
        receipts=usable_downloads,
        artifact_lineage=lineage,
        matching_config_hash=matching_hash,
    )
    dataset_config = load_config(root / "configs/datasets/mvp.yaml")
    if dataset_config.dataset is None:
        raise ExpansionExecutionError("Dataset configuration is unavailable.")
    label_hash = content_hash(
        {
            "policy_version": dataset_config.dataset.label_policy_version,
            "mapping": {"CP": "positive", "KP": "positive", "FP": "negative", "FA": "negative"},
        }
    )
    identity = DatasetIdentityInputs(
        catalog_snapshot_id=str(catalog_metadata["snapshot_id"]),
        catalog_snapshot_hash=str(catalog_metadata["raw_response_sha256"]),
        label_policy_hash=label_hash,
        acquisition_snapshot_id=str(product_metadata["snapshot_id"]),
        acquisition_snapshot_hash=str(product_metadata["products_sha256"]),
        raw_checksums=tuple(sorted(usable_downloads["sha256"].astype(str))),
        preprocessing_config_hash=preprocessing_hash,
        bls_config_hash=bls_hash,
        matching_config_hash=matching_hash,
        dataset_policy_hash=content_hash(
            {
                "dataset_config_hash": config_hash(dataset_config),
                "expansion_policy_hash": manifest.policy_hash,
                "materialization_schema_version": "expanded-development-v2",
            }
        ),
    )
    split = grouped_split(
        events["object_id"].astype(str).tolist(),
        settings=dataset_config.dataset,
        master_seed=dataset_config.reproducibility.master_seed,
        role=DatasetRole.DEVELOPMENT,
    )
    split = enrich_split_manifest(split, dataset.candidates, dataset.catalog_events)
    qa = validate_dataset(dataset, split)
    version = make_dataset_version(identity)
    dataset_path = freeze_dataset(
        dataset,
        identity=identity,
        validation_summary=qa,
        root=root / "data/datasets",
    )
    split_path = freeze_split_manifest(split, dataset_version=version, root=root / "data/splits")
    return dataset_path, split_path, version, qa


def run_frozen_cohort_expansion(root: str | Path) -> ExpansionRunResult:
    """Execute the frozen expansion manifest and regenerate a development-only dataset."""
    repository = Path(root)
    expansion_config = load_config(repository / "configs/acquisition/cohort_expansion.yaml")
    preprocessing_config = load_config(repository / "configs/preprocessing/mvp.yaml")
    bls_config = load_config(repository / "configs/bls/mvp.yaml")
    if (
        expansion_config.expansion is None
        or preprocessing_config.preprocessing is None
        or bls_config.bls is None
    ):
        raise ExpansionExecutionError(
            "Expansion, preprocessing, or BLS configuration is unavailable."
        )
    product_root = repository / "data/manifests/mast-products-b011-resumable"
    product_metadata = json.loads((product_root / "metadata.json").read_text(encoding="utf-8"))
    product_path = product_root / "normalized_products.parquet"
    if sha256_file(product_path) != product_metadata["products_sha256"]:
        raise ExpansionExecutionError("Frozen MAST product snapshot checksum mismatch.")
    products = pd.read_parquet(product_path)
    manifest = build_expansion_manifest(
        products,
        settings=expansion_config.expansion,
        product_snapshot_sha256=str(product_metadata["products_sha256"]),
    )
    manifest_root = write_expansion_manifest(manifest, root=repository / "data/manifests")
    _, frozen_manifest = load_expansion_manifest(manifest_root)
    if frozen_manifest.manifest_id != manifest.manifest_id:
        raise ExpansionExecutionError(
            "Expansion manifest replay differs from the frozen selection."
        )
    downloads, reuse_count = _download_selected(
        frozen_manifest.selected_products,
        root=repository,
        manifest_root=manifest_root,
        manifest_id=frozen_manifest.manifest_id,
    )
    preprocessing_hash = config_hash(preprocessing_config)
    preprocessing_receipts = _preprocess_selected(
        downloads,
        products,
        root=repository,
        manifest_root=manifest_root,
        preprocessing_hash=preprocessing_hash,
        settings=preprocessing_config.preprocessing,
        manifest_id=frozen_manifest.manifest_id,
    )
    cohort_root = next(
        (repository / "data/manifests").glob("pilot-toi-*/selected_target_events.parquet")
    ).parent
    events = pd.read_parquet(cohort_root / "selected_target_events.parquet")
    detections, candidate_frames, match_frames, bls_hash = _detect_selected(
        preprocessing_receipts,
        products,
        events,
        root=repository,
        manifest_root=manifest_root,
        preprocessing_hash=preprocessing_hash,
        settings=bls_config.bls,
    )
    dataset_path, split_path, dataset_version, qa = _build_expanded_dataset(
        root=repository,
        manifest=frozen_manifest,
        manifest_root=manifest_root,
        download_receipts=downloads,
        preprocessing_receipts=preprocessing_receipts,
        detection_receipts=detections,
        candidate_frames=candidate_frames,
        match_frames=match_frames,
        preprocessing_hash=preprocessing_hash,
        bls_hash=bls_hash,
        matching_hash=_matching_hash(bls_config.bls),
    )
    execution = {
        "manifest_id": frozen_manifest.manifest_id,
        "dataset_version": dataset_version,
        "selected_products": len(frozen_manifest.selected_products),
        "cache_reuse_count": reuse_count,
        "download_status_counts": downloads["status"].value_counts().to_dict(),
        "preprocessing_status_counts": preprocessing_receipts["status"].value_counts().to_dict(),
        "detection_status_counts": detections["status"].value_counts().to_dict(),
    }
    (manifest_root / "execution_summary.json").write_text(
        canonical_json(execution) + "\n", encoding="utf-8"
    )
    return ExpansionRunResult(
        frozen_manifest.manifest_id,
        dataset_version,
        len(frozen_manifest.selected_products),
        int(downloads["status"].isin(["downloaded", "reused"]).sum()),
        int(preprocessing_receipts["status"].isin(["processed", "reused"]).sum()),
        int(detections["status"].isin(["success", "reused"]).sum()),
        dataset_path,
        split_path,
        qa,
    )
