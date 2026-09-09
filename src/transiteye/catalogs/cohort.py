"""Deterministic target-level pilot cohort construction from frozen TOI events."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.catalogs.labels import InternalLabel
from transiteye.config import PilotCohortSettings
from transiteye.provenance import derive_seed
from transiteye.serialization import canonical_json, content_hash


class CohortSelectionError(ValueError):
    """Raised when a requested deterministic pilot cannot be selected safely."""


_REQUIRED_COLUMNS = {
    "object_id",
    "toi_id",
    "mapped_label",
    "disposition_subgroup",
    "source_disposition",
}


@dataclass(frozen=True)
class CohortResult:
    """Selected target records, preserved event lineage, and review QA."""

    targets: pd.DataFrame
    gold_events: pd.DataFrame
    selected_target_events: pd.DataFrame
    secondary_events: pd.DataFrame
    exclusions: pd.DataFrame
    qa_summary: dict[str, Any]


def _validate_events(events: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(events.columns))
    if missing:
        raise CohortSelectionError(f"Normalized catalog is missing required columns: {missing}")


def _sample_targets(target_ids: list[str], count: int, seed: int) -> list[str]:
    if count > len(target_ids):
        raise CohortSelectionError(
            f"Requested {count} targets but only {len(target_ids)} eligible targets are available."
        )
    return sorted(random.Random(seed).sample(sorted(target_ids), count))


def _target_status(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    valid = events.loc[events["object_id"].notna()]
    for object_id, group in valid.groupby("object_id", sort=True, dropna=True):
        labels = set(group["mapped_label"].astype(str))
        has_positive = InternalLabel.GOLD_POSITIVE.value in labels
        has_negative = InternalLabel.GOLD_NEGATIVE.value in labels
        if has_positive and has_negative:
            status = "mixed_gold_labels"
        elif has_positive:
            status = "positive_eligible"
        elif has_negative:
            status = "negative_eligible"
        else:
            status = "not_gold_eligible"
        records.append(
            {
                "object_id": object_id,
                "target_status": status,
                "source_event_count": len(group),
                "unique_toi_count": group["toi_id"].nunique(dropna=True),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_exclusion_reasons(events: pd.DataFrame, target_status: pd.DataFrame) -> pd.DataFrame:
    status_by_object = target_status.set_index("object_id")["target_status"].to_dict()
    records: list[dict[str, object]] = []
    for _, row in events.iterrows():
        label = str(row["mapped_label"])
        object_id = row["object_id"]
        if pd.isna(object_id):
            reason = "missing_or_invalid_tic"
        elif label == InternalLabel.UNKNOWN.value:
            reason = "unknown_disposition"
        elif label == InternalLabel.UNLABELED_SECONDARY.value:
            reason = "secondary_disposition"
        elif status_by_object.get(object_id) == "mixed_gold_labels":
            reason = "mixed_gold_labels_for_tic"
        else:
            reason = "not_selected_in_pilot"
        records.append(
            {
                "source_row_index": row.get("source_row_index"),
                "toi_id": row["toi_id"],
                "reason": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def _count_values(series: pd.Series) -> dict[str, int]:
    values = series.fillna("<missing>").astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in values.items()}


def build_catalog_qa(events: pd.DataFrame, cohort: CohortResult | None = None) -> dict[str, Any]:
    """Build a concise, serializable review summary without discarding rows."""
    _validate_events(events)
    valid_tic_events = events.loc[events["object_id"].notna()]
    unique_toi_by_tic = valid_tic_events.groupby("object_id")["toi_id"].nunique()
    summary: dict[str, Any] = {
        "total_catalog_rows": int(len(events)),
        "unique_tic_count": int(events["object_id"].nunique(dropna=True)),
        "unique_toi_count": int(events["toi_id"].nunique(dropna=True)),
        "disposition_counts": _count_values(events["source_disposition"]),
        "mapped_positive_count": int(
            (events["mapped_label"] == InternalLabel.GOLD_POSITIVE.value).sum()
        ),
        "mapped_negative_count": int(
            (events["mapped_label"] == InternalLabel.GOLD_NEGATIVE.value).sum()
        ),
        "pc_apc_count": int(
            (events["mapped_label"] == InternalLabel.UNLABELED_SECONDARY.value).sum()
        ),
        "unknown_disposition_count": int(
            (events["mapped_label"] == InternalLabel.UNKNOWN.value).sum()
        ),
        "missing_tic_count": int(events["object_id"].isna().sum()),
        "missing_period_count": int(events.get("period_days", pd.Series(dtype=float)).isna().sum()),
        "missing_epoch_count": int(
            events.get("transit_epoch_bjd", pd.Series(dtype=float)).isna().sum()
        ),
        "duplicate_toi_count": int(events.duplicated(subset=["toi_id"], keep=False).sum()),
        "tics_with_multiple_tois": int((unique_toi_by_tic > 1).sum()),
    }
    if cohort is not None:
        summary.update(
            {
                "final_pilot_cohort_size": int(len(cohort.targets)),
                "pilot_gold_event_count": int(len(cohort.gold_events)),
                "unique_tics_in_pilot": int(cohort.targets["object_id"].nunique(dropna=True)),
                "pilot_target_class_distribution": _count_values(cohort.targets["selection_class"]),
                "pilot_gold_subgroup_distribution": _count_values(
                    cohort.gold_events["disposition_subgroup"]
                ),
            }
        )
    return summary


def build_pilot_cohort(
    events: pd.DataFrame,
    *,
    settings: PilotCohortSettings,
    master_seed: int,
) -> CohortResult:
    """Select exclusive positive/negative TIC targets while retaining event lineage."""
    _validate_events(events)
    source_events = events.copy(deep=True)
    statuses = _target_status(source_events)
    positive_ids = (
        statuses.loc[statuses["target_status"] == "positive_eligible", "object_id"]
        .astype(str)
        .tolist()
    )
    negative_ids = (
        statuses.loc[statuses["target_status"] == "negative_eligible", "object_id"]
        .astype(str)
        .tolist()
    )
    positive_seed = derive_seed(master_seed, f"{settings.seed_component}:positive")
    negative_seed = derive_seed(master_seed, f"{settings.seed_component}:negative")
    selected_positive = _sample_targets(positive_ids, settings.positive_target_count, positive_seed)
    selected_negative = _sample_targets(negative_ids, settings.negative_target_count, negative_seed)
    selected_roles = {item: "positive" for item in selected_positive}
    selected_roles.update({item: "negative" for item in selected_negative})

    target_frame = pd.DataFrame(
        [
            {
                "object_id": object_id,
                "selection_class": selection_class,
                "selection_seed": positive_seed if selection_class == "positive" else negative_seed,
            }
            for object_id, selection_class in sorted(selected_roles.items())
        ]
    )
    target_frame = target_frame.merge(statuses, on="object_id", how="left", validate="one_to_one")

    selected_target_events = source_events.loc[
        source_events["object_id"].isin(selected_roles)
    ].copy()
    selected_target_events["selection_class"] = selected_target_events["object_id"].map(
        selected_roles
    )
    selected_target_events["is_gold_event"] = (
        (selected_target_events["selection_class"] == "positive")
        & (selected_target_events["mapped_label"] == InternalLabel.GOLD_POSITIVE.value)
    ) | (
        (selected_target_events["selection_class"] == "negative")
        & (selected_target_events["mapped_label"] == InternalLabel.GOLD_NEGATIVE.value)
    )
    candidate_gold_events = selected_target_events.loc[
        selected_target_events["is_gold_event"]
    ].copy()
    gold_events = candidate_gold_events.sort_values("source_row_index").drop_duplicates(
        subset=["toi_id"], keep="first"
    )
    secondary_events = source_events.loc[
        source_events["mapped_label"] == InternalLabel.UNLABELED_SECONDARY.value
    ].copy()
    exclusions = _build_exclusion_reasons(source_events, statuses)
    selected_indices = set(gold_events["source_row_index"].tolist())
    duplicate_gold_indices = set(candidate_gold_events["source_row_index"].tolist()).difference(
        selected_indices
    )
    exclusions.loc[exclusions["source_row_index"].isin(selected_indices), "reason"] = (
        "selected_gold_event"
    )
    exclusions.loc[exclusions["source_row_index"].isin(duplicate_gold_indices), "reason"] = (
        "duplicate_toi_event"
    )
    exclusions = exclusions.loc[exclusions["reason"] != "selected_gold_event"].reset_index(
        drop=True
    )

    preliminary = CohortResult(
        targets=target_frame,
        gold_events=gold_events.reset_index(drop=True),
        selected_target_events=selected_target_events.reset_index(drop=True),
        secondary_events=secondary_events.reset_index(drop=True),
        exclusions=exclusions,
        qa_summary={},
    )
    return CohortResult(
        **{
            **preliminary.__dict__,
            "qa_summary": build_catalog_qa(source_events, preliminary),
        }
    )


def _table_hash(frame: pd.DataFrame) -> str:
    records = json.loads(frame.to_json(orient="records"))
    return content_hash({"columns": frame.columns.tolist(), "records": records}, length=64)


def write_pilot_cohort_manifest(
    cohort: CohortResult,
    *,
    snapshot_id: str,
    cohort_config_hash: str,
    master_seed: int,
    root: str | Path = "data/manifests",
) -> Path:
    """Write deterministic, reviewable pilot manifests without rewriting existing content."""
    manifest_id = f"pilot-{snapshot_id}-{cohort_config_hash}"
    directory = Path(root) / manifest_id
    metadata_path = directory / "metadata.json"
    metadata = {
        "manifest_id": manifest_id,
        "snapshot_id": snapshot_id,
        "cohort_config_hash": cohort_config_hash,
        "master_seed": master_seed,
        "targets_sha256": _table_hash(cohort.targets),
        "gold_events_sha256": _table_hash(cohort.gold_events),
        "selected_target_events_sha256": _table_hash(cohort.selected_target_events),
        "secondary_events_sha256": _table_hash(cohort.secondary_events),
        "exclusions_sha256": _table_hash(cohort.exclusions),
    }
    if directory.exists():
        if metadata_path.exists() and json_load(metadata_path) == metadata:
            return directory
        raise CohortSelectionError(
            f"Pilot manifest already exists and cannot be overwritten: {manifest_id}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    try:
        cohort.targets.to_parquet(directory / "targets.parquet", engine="pyarrow", index=False)
        cohort.gold_events.to_parquet(
            directory / "gold_events.parquet", engine="pyarrow", index=False
        )
        cohort.selected_target_events.to_parquet(
            directory / "selected_target_events.parquet", engine="pyarrow", index=False
        )
        cohort.secondary_events.to_parquet(
            directory / "secondary_events.parquet", engine="pyarrow", index=False
        )
        cohort.exclusions.to_parquet(
            directory / "exclusions.parquet", engine="pyarrow", index=False
        )
        (directory / "qa.json").write_text(
            canonical_json(cohort.qa_summary) + "\n", encoding="utf-8"
        )
        metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    except Exception:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        raise
    return directory


def json_load(path: Path) -> dict[str, Any]:
    """Load a manifest metadata object for immutable-write comparisons."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CohortSelectionError("Pilot manifest metadata is malformed.")
    return value
