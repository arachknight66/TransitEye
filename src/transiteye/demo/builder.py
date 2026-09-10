"""End-to-end deterministic demo construction over immutable real TESS substrates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import BlsSettings, config_hash, load_config
from transiteye.datasets.builder import BuiltDataset
from transiteye.datasets.lineage import validate_frozen_candidate_ids
from transiteye.datasets.schemas import FORBIDDEN_MODEL_INPUT_COLUMNS, CandidateGoldLabel
from transiteye.demo.injection import inject_variant
from transiteye.demo.validation import validate_demo_dataset
from transiteye.demo.variants import (
    DemoVariantManifest,
    build_variant_manifest,
    freeze_variant_manifest,
    load_variant_manifest,
)
from transiteye.detection.bls import run_bls
from transiteye.detection.matching import match_candidates
from transiteye.detection.peaks import extract_peaks
from transiteye.preprocessing.pipeline import preprocess, read_tess_lightcurve
from transiteye.serialization import canonical_json, content_hash


class DemoBuildError(ValueError):
    """Raised when immutable inputs cannot produce a valid demo dataset."""


@dataclass(frozen=True)
class DemoBuildResult:
    manifest_id: str
    dataset_version: str
    dataset_path: Path
    split_path: Path
    qa_summary: dict[str, Any]
    acceptance: dict[str, Any]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    frame.to_parquet(partial, engine="pyarrow", index=False)
    os.replace(partial, path)


def _freeze_table(frame: pd.DataFrame, path: Path, sort_columns: list[str]) -> None:
    ordered = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if path.exists():
        existing = (
            pd.read_parquet(path).sort_values(sort_columns, kind="stable").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(existing, ordered, check_dtype=False)
        return
    _atomic_parquet(ordered, path)


def _base_products(root: Path, manifest_id: str) -> pd.DataFrame:
    manifest_root = root / "data/manifests" / manifest_id
    selected = pd.read_parquet(manifest_root / "selected_products.parquet")
    downloads = pd.read_parquet(manifest_root / "download_receipts.parquet")
    processed = pd.read_parquet(manifest_root / "preprocessing_receipts.parquet")
    base = selected.merge(
        downloads[["data_uri", "relative_raw_path", "sha256", "status"]],
        on="data_uri",
        validate="one_to_one",
    ).merge(
        processed[
            [
                "data_uri",
                "observation_id",
                "processed_checksum",
                "preprocessing_config_hash",
                "status",
            ]
        ],
        on="data_uri",
        validate="one_to_one",
        suffixes=("_download", "_preprocessing"),
    )
    if not base["status_download"].isin(["downloaded", "reused"]).all():
        raise DemoBuildError("A selected demo substrate is not downloaded.")
    if not base["status_preprocessing"].isin(["processed", "reused"]).all():
        raise DemoBuildError("A selected demo substrate lacks accepted preprocessing lineage.")
    rows: list[dict[str, Any]] = []
    for row in base.to_dict(orient="records"):
        raw_path = root / "data/raw" / str(row["relative_raw_path"])
        checksum = str(row["sha256"])
        if sha256_file(raw_path) != checksum:
            raise DemoBuildError("Base raw TESS checksum mismatch.")
        source = read_tess_lightcurve(raw_path, observation_id=str(row["observation_id"]))
        finite_time = source.data.loc[source.data["time"].notna(), "time"]
        rows.append(
            {
                "object_id": row["object_id"],
                "source_observation_id": row["observation_id"],
                "source_product_id": row["data_uri"],
                "source_raw_checksum": checksum,
                "source_preprocessing_hash": row["preprocessing_config_hash"],
                "source_processed_checksum": row["processed_checksum"],
                "relative_raw_path": row["relative_raw_path"],
                "sector": int(row["sector"]),
                "author": row["author"],
                "exposure_seconds": float(row["exposure_seconds"]),
                "time_min": float(finite_time.min()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["object_id", "source_observation_id"], kind="stable")
        .reset_index(drop=True)
    )


def _matching_hash(settings: BlsSettings) -> str:
    return content_hash(
        {
            "policy_version": "bls-ephemeris-match-v1",
            "period_match_tolerance": settings.period_match_tolerance,
            "phase_match_tolerance": settings.phase_match_tolerance,
            "harmonic_ratios": [0.5, 1.0, 2.0],
        }
    )


def _process_variants(
    root: Path,
    manifest: DemoVariantManifest,
    base: pd.DataFrame,
    *,
    preprocessing_hash: str,
    preprocessing_settings: Any,
    bls_settings: BlsSettings,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], str]:
    workspace = root / "data/synthetic" / manifest.manifest_id
    receipts_path = workspace / "artifact_lineage.parquet"
    bls_hash = content_hash(
        {"bls": bls_settings.model_dump(mode="json"), "preprocessing_hash": preprocessing_hash}
    )
    if receipts_path.exists():
        lineage = pd.read_parquet(receipts_path)
        for row in lineage.loc[lineage["detection_status"] == "success"].to_dict(orient="records"):
            checks = (
                (
                    workspace / "variants" / str(row["variant_id"]) / "source.parquet",
                    "source_sha256",
                ),
                (
                    workspace / "variants" / str(row["variant_id"]) / "processed.parquet",
                    "demo_processed_checksum",
                ),
                (
                    workspace / "detection" / str(row["periodogram_filename"]),
                    "periodogram_sha256",
                ),
                (
                    workspace / "detection" / str(row["candidate_filename"]),
                    "candidate_sha256",
                ),
                (
                    workspace / "detection" / str(row["match_filename"]),
                    "match_sha256",
                ),
            )
            for path, checksum_column in checks:
                if not path.is_file() or sha256_file(path) != str(row[checksum_column]):
                    raise DemoBuildError(f"Frozen demo artifact checksum mismatch: {path.name}")
        candidate_frames = [
            pd.read_parquet(workspace / "detection" / str(row["candidate_filename"])).assign(
                variant_id=str(row["variant_id"]),
                observation_group_id=str(row["observation_group_id"]),
            )
            for row in lineage.loc[lineage["detection_status"] == "success"].to_dict(
                orient="records"
            )
        ]
        match_frames = [
            pd.read_parquet(workspace / "detection" / str(row["match_filename"]))
            for row in lineage.loc[lineage["detection_status"] == "success"].to_dict(
                orient="records"
            )
        ]
        return lineage, candidate_frames, match_frames, bls_hash

    base_lookup = base.set_index("source_product_id")
    config = load_config(root / "configs/demo/mvp.yaml")
    if config.demo is None:
        raise DemoBuildError("Demo configuration is unavailable.")
    definitions = {variant.name: variant for variant in config.demo.variants}
    truth_lookup = manifest.synthetic_events.set_index("variant_id")
    lineage_rows: list[dict[str, Any]] = []
    candidate_frames = []
    match_frames = []
    for variant in manifest.variants.sort_values("variant_id", kind="stable").to_dict(
        orient="records"
    ):
        base_row = base_lookup.loc[str(variant["source_product_id"])]
        raw_path = root / "data/raw" / str(base_row["relative_raw_path"])
        variant_id = str(variant["variant_id"])
        event = truth_lookup.loc[variant_id]
        demo_observation_id = f"demo-obs-{content_hash({'variant_id': variant_id})}"
        group_id = f"demo-og-{content_hash({'demo_observation_id': demo_observation_id})}"
        variant_root = workspace / "variants" / variant_id
        source_path = variant_root / "source.parquet"
        processed_path = variant_root / "processed.parquet"
        metadata_path = variant_root / "metadata.json"
        candidate_path = workspace / "detection" / f"{variant_id}-candidates.parquet"
        periodogram_path = workspace / "detection" / f"{variant_id}-periodogram.parquet"
        match_path = workspace / "detection" / f"{variant_id}-matches.parquet"
        preprocessing_status = "failed"
        detection_status = "not_searched"
        error: str | None = None
        try:
            source = read_tess_lightcurve(
                raw_path, observation_id=str(variant["source_observation_id"])
            )
            derived = inject_variant(
                source,
                variant=definitions[str(variant["variant_name"])],
                epoch=(
                    None if pd.isna(event["injected_epoch"]) else float(event["injected_epoch"])
                ),
            )
            _freeze_table(derived.data, source_path, ["cadence_row"])
            processed = preprocess(
                derived,
                gap_days=preprocessing_settings.gap_days,
                trend_window=preprocessing_settings.trend_window_cadences,
                positive_spike_mad=preprocessing_settings.positive_spike_mad,
            )
            _freeze_table(processed.data, processed_path, ["cadence_row"])
            metadata = {
                "variant_id": variant_id,
                "dataset_role": "demo",
                "truth_source": "synthetic_injection",
                "source_raw_checksum": variant["source_raw_checksum"],
                "source_observation_id": variant["source_observation_id"],
                "demo_observation_id": demo_observation_id,
                "observation_group_id": group_id,
                "demo_policy_hash": manifest.policy_hash,
                "preprocessing_config_hash": preprocessing_hash,
                "source_sha256": sha256_file(source_path),
                "processed_sha256": sha256_file(processed_path),
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
            preprocessing_status = "success"
            result = run_bls(
                processed.data,
                settings=bls_settings,
                observation_group_id=group_id,
                preprocessing_hash=preprocessing_hash,
            )
            if result.config_hash != bls_hash:
                raise DemoBuildError("Demo BLS hash differs from the frozen scientific policy.")
            candidates = extract_peaks(result, bls_settings)
            _freeze_table(result.periodogram, periodogram_path, ["period"])
            _freeze_table(candidates, candidate_path, ["rank"])
            # Truth first becomes available to matching after candidate output is frozen.
            if str(event["synthetic_event_class"]) == "no_injection_control":
                matches = pd.DataFrame(
                    columns=[
                        "candidate_id",
                        "toi_id",
                        "candidate_period",
                        "catalog_period",
                        "relative_period_error",
                        "candidate_epoch",
                        "catalog_epoch",
                        "phase_error",
                        "harmonic_ratio",
                        "match_type",
                        "matched",
                    ]
                )
            else:
                generic_event = pd.DataFrame(
                    [
                        {
                            "toi_id": event["synthetic_event_id"],
                            "period_days": event["injected_period_days"],
                            "transit_epoch_bjd": event["injected_epoch"],
                        }
                    ]
                )
                matches = match_candidates(candidates, generic_event, bls_settings)
            if not matches.empty:
                matches["variant_id"] = variant_id
                matches["object_id"] = variant["object_id"]
                matches["synthetic_event_class"] = event["synthetic_event_class"]
                matches["demo_gold_class"] = event["demo_gold_class"]
            _freeze_table(matches, match_path, ["candidate_id", "toi_id"])
            candidate_frames.append(
                candidates.assign(variant_id=variant_id, observation_group_id=group_id)
            )
            match_frames.append(matches)
            detection_status = "success"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        lineage_rows.append(
            {
                "variant_id": variant_id,
                "synthetic_event_id": variant["synthetic_event_id"],
                "object_id": variant["object_id"],
                "source_observation_id": variant["source_observation_id"],
                "source_product_id": variant["source_product_id"],
                "source_raw_checksum": variant["source_raw_checksum"],
                "sector": int(base_row["sector"]),
                "source_preprocessing_hash": variant["source_preprocessing_hash"],
                "demo_observation_id": demo_observation_id,
                "observation_group_id": group_id,
                "demo_policy_hash": manifest.policy_hash,
                "preprocessing_config_hash": preprocessing_hash,
                "demo_processed_checksum": (
                    sha256_file(processed_path) if processed_path.exists() else None
                ),
                "source_sha256": sha256_file(source_path) if source_path.exists() else None,
                "bls_config_hash": bls_hash,
                "preprocessing_status": preprocessing_status,
                "detection_status": detection_status,
                "candidate_filename": candidate_path.name if candidate_path.exists() else None,
                "match_filename": match_path.name if match_path.exists() else None,
                "periodogram_filename": (
                    periodogram_path.name if periodogram_path.exists() else None
                ),
                "periodogram_sha256": (
                    sha256_file(periodogram_path) if periodogram_path.exists() else None
                ),
                "candidate_sha256": (
                    sha256_file(candidate_path) if candidate_path.exists() else None
                ),
                "match_sha256": sha256_file(match_path) if match_path.exists() else None,
                "dataset_role": "demo",
                "truth_source": "synthetic_injection",
                "error": error,
            }
        )
    lineage = pd.DataFrame(lineage_rows)
    _freeze_table(lineage, receipts_path, ["variant_id"])
    return lineage, candidate_frames, match_frames, bls_hash


def build_demo_candidate_tables(
    frozen_candidates: pd.DataFrame,
    frozen_matches: pd.DataFrame,
    events: pd.DataFrame,
    lineage: pd.DataFrame,
    *,
    matching_config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply match-based demo labels while preserving the shared candidate measurements."""
    base = frozen_candidates.merge(
        lineage[
            [
                "variant_id",
                "object_id",
                "demo_observation_id",
                "observation_group_id",
                "source_product_id",
                "source_raw_checksum",
                "preprocessing_config_hash",
                "bls_config_hash",
            ]
        ],
        on=["variant_id", "observation_group_id"],
        validate="many_to_one",
    )
    if frozen_matches.empty:
        relations = pd.DataFrame(
            columns=[
                "candidate_id",
                "event_id",
                "synthetic_event_id",
                "variant_id",
                "object_id",
                "candidate_period",
                "event_period",
                "relative_period_error",
                "candidate_epoch",
                "event_epoch",
                "phase_error",
                "match_harmonic_ratio",
                "match_type",
                "matched",
                "synthetic_event_class",
                "demo_gold_class",
                "matching_config_hash",
                "truth_source",
            ]
        )
    else:
        relations = frozen_matches.rename(
            columns={
                "toi_id": "synthetic_event_id",
                "catalog_period": "event_period",
                "catalog_epoch": "event_epoch",
                "harmonic_ratio": "match_harmonic_ratio",
            }
        ).copy()
        relations["event_id"] = relations["synthetic_event_id"]
        relations["matching_config_hash"] = matching_config_hash
        relations["truth_source"] = "synthetic_injection"
        relations = relations[
            [
                "candidate_id",
                "event_id",
                "synthetic_event_id",
                "variant_id",
                "object_id",
                "candidate_period",
                "event_period",
                "relative_period_error",
                "candidate_epoch",
                "event_epoch",
                "phase_error",
                "match_harmonic_ratio",
                "match_type",
                "matched",
                "synthetic_event_class",
                "demo_gold_class",
                "matching_config_hash",
                "truth_source",
            ]
        ]
    valid = relations.loc[relations["matched"].astype(bool)]
    priorities = {"fundamental": 0, "half_period": 1, "double_period": 1}
    rows: list[dict[str, Any]] = []
    for candidate in base.to_dict(orient="records"):
        matched = valid.loc[valid["candidate_id"] == candidate["candidate_id"]].copy()
        labels = set(matched["demo_gold_class"])
        if {"positive", "negative"}.issubset(labels):
            label = CandidateGoldLabel.AMBIGUOUS
        elif "positive" in labels:
            label = CandidateGoldLabel.POSITIVE
        elif "negative" in labels:
            label = CandidateGoldLabel.NEGATIVE
        else:
            label = CandidateGoldLabel.UNLABELED
        primary: dict[str, Any] | None = None
        if not matched.empty:
            matched["_priority"] = matched["match_type"].map(priorities).fillna(2)
            primary = (
                matched.sort_values(
                    ["_priority", "relative_period_error", "phase_error", "synthetic_event_id"],
                    kind="stable",
                )
                .iloc[0]
                .to_dict()
            )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "object_id": candidate["object_id"],
                "observation_id": candidate["demo_observation_id"],
                "observation_group_id": candidate["observation_group_id"],
                "sector": int(
                    lineage.loc[lineage["variant_id"] == candidate["variant_id"]]
                    .iloc[0]
                    .get("sector", 1)
                ),
                "source_product_id": candidate["source_product_id"],
                "source_raw_checksum": candidate["source_raw_checksum"],
                "preprocessing_config_hash": candidate["preprocessing_config_hash"],
                "bls_config_hash": candidate["bls_config_hash"],
                "candidate_rank": int(candidate["rank"]),
                "candidate_period": float(candidate["period"]),
                "candidate_duration": float(candidate["duration"]),
                "candidate_epoch": float(candidate["epoch"]),
                "candidate_depth": float(candidate["depth"]),
                "depth_uncertainty": None,
                "bls_power": float(candidate["power"]),
                "relation_to_stronger": candidate["relation_to_stronger"],
                "detection_harmonic_ratio": candidate["harmonic_ratio"],
                "catalog_match_status": "matched" if primary else "unmatched",
                "matched_event_id": primary["synthetic_event_id"] if primary else None,
                "source_disposition": None,
                "gold_candidate_label": label.value,
                "gold_training_eligible": label
                in {CandidateGoldLabel.POSITIVE, CandidateGoldLabel.NEGATIVE},
                "match_type": primary["match_type"] if primary else None,
                "match_harmonic_ratio": (primary["match_harmonic_ratio"] if primary else None),
                "matching_config_hash": matching_config_hash,
                "variant_id": candidate["variant_id"],
                "dataset_role": "demo",
                "truth_source": "synthetic_injection",
            }
        )
    candidates = (
        pd.DataFrame(rows)
        .sort_values(["object_id", "variant_id", "candidate_rank"], kind="stable")
        .reset_index(drop=True)
    )
    return candidates, relations.sort_values(
        ["candidate_id", "synthetic_event_id"], kind="stable"
    ).reset_index(drop=True)


def build_demo_event_recovery(
    events: pd.DataFrame,
    lineage: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    """Retain every injected event/control and distinguish BLS recovery from generation."""
    rows: list[dict[str, Any]] = []
    matched = relations.loc[relations["matched"].astype(bool)]
    for event in events.to_dict(orient="records"):
        variant_id = str(event["variant_id"])
        artifact = lineage.loc[lineage["variant_id"] == variant_id]
        matches = matched.loc[matched["synthetic_event_id"] == event["synthetic_event_id"]]
        fundamental = matches.loc[matches["match_type"] == "fundamental"]
        harmonic = matches.loc[
            matches["match_type"].isin(["half_period", "double_period", "harmonic"])
        ]
        if str(event["synthetic_event_class"]) == "no_injection_control":
            recovery = "searched_not_recovered"
        elif not fundamental.empty:
            recovery = "recovered_fundamental"
        elif not harmonic.empty:
            recovery = "recovered_harmonic"
        elif artifact.empty:
            recovery = "not_searched"
        elif (artifact["preprocessing_status"] == "failed").any():
            recovery = "preprocessing_failed"
        elif (artifact["detection_status"] == "failed").any():
            recovery = "detection_failed"
        elif (artifact["detection_status"] == "success").any():
            recovery = "searched_not_recovered"
        else:
            recovery = "not_searched"
        primary = None
        if not fundamental.empty:
            primary = fundamental.sort_values("candidate_id", kind="stable").iloc[0]["candidate_id"]
        elif not harmonic.empty:
            primary = harmonic.sort_values("candidate_id", kind="stable").iloc[0]["candidate_id"]
        event_row = {str(key): value for key, value in event.items()}
        rows.append(
            event_row
            | {
                "event_id": event["synthetic_event_id"],
                "internal_event_class": event["demo_gold_class"],
                "recovery_state": recovery,
                "matching_candidate_count": len(matches),
                "fundamental_candidate_count": len(fundamental),
                "harmonic_candidate_count": len(harmonic),
                "primary_candidate_id": primary,
                "contributing_observation_group_ids": canonical_json(
                    sorted(artifact["observation_group_id"].astype(str).unique())
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("synthetic_event_id", kind="stable").reset_index(drop=True)
    )


def _demo_identity(
    manifest: DemoVariantManifest,
    lineage: pd.DataFrame,
    *,
    preprocessing_hash: str,
    bls_hash: str,
    matching_hash: str,
    dataset_policy_hash: str,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "dataset_role": "demo",
        "truth_source": "synthetic_injection",
        "base_raw_checksums": sorted(set(lineage["source_raw_checksum"].astype(str))),
        "base_processed_checksums": sorted(
            set(manifest.variants["source_processed_checksum"].astype(str))
        ),
        "demo_processed_checksums": sorted(
            set(lineage["demo_processed_checksum"].dropna().astype(str))
        ),
        "preprocessing_config_hash": preprocessing_hash,
        "bls_config_hash": bls_hash,
        "matching_config_hash": matching_hash,
        "demo_policy_hash": manifest.policy_hash,
        "variant_manifest_hash": manifest.variant_manifest_hash,
        "generator_version": str(manifest.variants["generator_version"].iloc[0]),
        "dataset_construction_policy_hash": dataset_policy_hash,
    }
    return f"dataset-demo-{content_hash(identity)}", identity


def _freeze_demo_dataset(
    dataset: BuiltDataset,
    events_truth: pd.DataFrame,
    split: pd.DataFrame,
    *,
    version: str,
    identity: dict[str, Any],
    qa: dict[str, Any],
    root: Path,
) -> tuple[Path, Path]:
    dataset_root = root / "data/datasets" / version
    split_root = root / "data/splits" / version
    tables = {
        "candidates.parquet": (dataset.candidates, ["candidate_id"]),
        "candidate_event_matches.parquet": (
            dataset.candidate_event_matches,
            ["candidate_id", "event_id"],
        ),
        "catalog_events.parquet": (dataset.catalog_events, ["event_id"]),
        "artifact_lineage.parquet": (dataset.artifact_lineage, ["variant_id"]),
        "synthetic_events.parquet": (events_truth, ["synthetic_event_id"]),
    }
    if dataset_root.exists():
        metadata = json.loads((dataset_root / "dataset_metadata.json").read_text(encoding="utf-8"))
        if metadata["dataset_version"] != version or metadata["identity_inputs"] != identity:
            raise DemoBuildError("Existing demo dataset identity conflicts with replay.")
        for filename, checksum in metadata["table_checksums"].items():
            if sha256_file(dataset_root / filename) != checksum:
                raise DemoBuildError(f"Frozen demo dataset checksum mismatch: {filename}")
        for filename, (frame, keys) in tables.items():
            frozen = pd.read_parquet(dataset_root / filename)
            expected = frame.sort_values(keys, kind="stable").reset_index(drop=True)
            actual = frozen.sort_values(keys, kind="stable").reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    else:
        dataset_root.mkdir(parents=True, exist_ok=False)
        checksums: dict[str, str] = {}
        for filename, (frame, keys) in tables.items():
            _freeze_table(frame, dataset_root / filename, keys)
            checksums[filename] = sha256_file(dataset_root / filename)
        metadata = {
            "dataset_version": version,
            "dataset_role": "demo",
            "statistical_role": "development_demo",
            "truth_source": "synthetic_injection",
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "identity_inputs": identity,
            "table_checksums": checksums,
            "scientific_claim_boundary": (
                "Controlled synthetic-injection benchmark; not real TESS detection performance."
            ),
        }
        (dataset_root / "dataset_metadata.json").write_text(
            canonical_json(metadata) + "\n", encoding="utf-8"
        )
        (dataset_root / "validation_summary.json").write_text(
            canonical_json(qa) + "\n", encoding="utf-8"
        )
    if split_root.exists():
        frozen_split = pd.read_parquet(split_root / "split_manifest.parquet")
        pd.testing.assert_frame_equal(frozen_split, split, check_dtype=False)
    else:
        split_root.mkdir(parents=True, exist_ok=False)
        _atomic_parquet(split, split_root / "split_manifest.parquet")
        split_metadata = {
            "dataset_version": version,
            "dataset_role": "demo",
            "statistical_role": "development_demo",
            "grouping_unit": "source_tic",
            "split_manifest_sha256": sha256_file(split_root / "split_manifest.parquet"),
        }
        (split_root / "split_metadata.json").write_text(
            canonical_json(split_metadata) + "\n", encoding="utf-8"
        )
    return dataset_root, split_root


def evaluate_demo_acceptance(
    candidates: pd.DataFrame,
    events: pd.DataFrame,
    lineage: pd.DataFrame,
    qa: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the D008 development gate without changing generation or detector policy."""
    positives = candidates.loc[candidates["gold_candidate_label"] == "positive"]
    negatives = candidates.loc[candidates["gold_candidate_label"] == "negative"]
    required_truth_fields = {
        "synthetic_event_class",
        "demo_gold_class",
        "injected_period_days",
        "injected_epoch",
        "injected_duration_days",
        "injected_depth_or_amplitude",
        "truth_source",
    }
    criteria = {
        "explicit_demo_provenance": bool(
            candidates["dataset_role"].eq("demo").all()
            and candidates["truth_source"].eq("synthetic_injection").all()
        ),
        "positive_injections_exist": bool((events["synthetic_event_class"] == "planet_like").any()),
        "negative_confounders_exist": bool(
            events["synthetic_event_class"]
            .isin(["eclipsing_binary_like", "sinusoidal_variability_like"])
            .any()
        ),
        "controls_exist": bool((events["synthetic_event_class"] == "no_injection_control").any()),
        "all_preprocessing_success": bool(lineage["preprocessing_status"].eq("success").all()),
        "all_bls_success": bool(lineage["detection_status"].eq("success").all()),
        "positive_gold_candidates": len(positives) > 0,
        "negative_gold_candidates": len(negatives) > 0,
        "positive_source_tics_at_least_two": positives["object_id"].nunique() >= 2,
        "negative_source_tics_at_least_two": negatives["object_id"].nunique() >= 2,
        "zero_source_tic_overlap": bool(qa["zero_source_tic_overlap"]),
        "zero_source_checksum_overlap": bool(qa["zero_source_checksum_overlap"]),
        "synthetic_truth_feature_firewall": required_truth_fields.issubset(
            FORBIDDEN_MODEL_INPUT_COLUMNS
        ),
    }
    return {
        "ready": all(criteria.values()),
        "criteria": criteria,
        "positive_gold_candidates": len(positives),
        "negative_gold_candidates": len(negatives),
        "positive_source_tics": int(positives["object_id"].nunique()),
        "negative_source_tics": int(negatives["object_id"].nunique()),
    }


def build_demo_dataset(root: str | Path) -> DemoBuildResult:
    """Build and replay the complete D001--D008 demo path from frozen local inputs."""
    repository = Path(root)
    demo_config = load_config(repository / "configs/demo/mvp.yaml")
    preprocessing_config = load_config(repository / "configs/preprocessing/mvp.yaml")
    bls_config = load_config(repository / "configs/bls/mvp.yaml")
    dataset_config = load_config(repository / "configs/datasets/mvp.yaml")
    if (
        demo_config.demo is None
        or preprocessing_config.preprocessing is None
        or bls_config.bls is None
        or dataset_config.dataset is None
    ):
        raise DemoBuildError("Demo, preprocessing, BLS, or dataset configuration is unavailable.")
    base = _base_products(repository, demo_config.demo.base_expansion_manifest_id)
    manifest = build_variant_manifest(
        base, settings=demo_config.demo, master_seed=demo_config.reproducibility.master_seed
    )
    manifest_root = freeze_variant_manifest(manifest, root=repository / "data/manifests")
    frozen_manifest = load_variant_manifest(manifest_root)
    preprocessing_hash = config_hash(preprocessing_config)
    lineage, candidate_frames, match_frames, bls_hash = _process_variants(
        repository,
        frozen_manifest,
        base,
        preprocessing_hash=preprocessing_hash,
        preprocessing_settings=preprocessing_config.preprocessing,
        bls_settings=bls_config.bls,
    )
    if not candidate_frames:
        raise DemoBuildError("No demo BLS candidate artifacts were produced.")
    frozen_candidates = pd.concat(candidate_frames, ignore_index=True)
    validate_frozen_candidate_ids(frozen_candidates, lineage)
    nonempty_matches = [frame for frame in match_frames if not frame.empty]
    frozen_matches = (
        pd.concat(nonempty_matches, ignore_index=True) if nonempty_matches else pd.DataFrame()
    )
    matching_hash = _matching_hash(bls_config.bls)
    candidates, relations = build_demo_candidate_tables(
        frozen_candidates,
        frozen_matches,
        frozen_manifest.synthetic_events,
        lineage,
        matching_config_hash=matching_hash,
    )
    events = build_demo_event_recovery(frozen_manifest.synthetic_events, lineage, relations)
    split = pd.DataFrame(
        {
            "object_id": sorted(lineage["object_id"].unique()),
            "split": "development",
            "dataset_role": "demo",
            "statistical_role": "development_demo",
            "grouping_unit": "source_tic",
        }
    )
    qa = validate_demo_dataset(candidates, relations, events, lineage, split)
    dataset = BuiltDataset(candidates, relations, events, lineage)
    dataset_policy_hash = content_hash(
        {
            "scientific_dataset_config_hash": config_hash(dataset_config),
            "demo_dataset_contract_version": "demo-candidate-event-v1",
            "split_policy": demo_config.demo.split_policy,
        }
    )
    version, identity = _demo_identity(
        frozen_manifest,
        lineage,
        preprocessing_hash=preprocessing_hash,
        bls_hash=bls_hash,
        matching_hash=matching_hash,
        dataset_policy_hash=dataset_policy_hash,
    )
    dataset_path, split_path = _freeze_demo_dataset(
        dataset,
        frozen_manifest.synthetic_events,
        split,
        version=version,
        identity=identity,
        qa=qa,
        root=repository,
    )
    acceptance = evaluate_demo_acceptance(candidates, events, lineage, qa)
    (dataset_path / "acceptance_gate.json").write_text(
        canonical_json(acceptance) + "\n", encoding="utf-8"
    )
    return DemoBuildResult(
        frozen_manifest.manifest_id,
        version,
        dataset_path,
        split_path,
        qa,
        acceptance,
    )
