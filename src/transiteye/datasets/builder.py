"""Candidate and catalog-event dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from transiteye.catalogs.labels import InternalLabel, map_disposition
from transiteye.datasets.schemas import (
    CandidateDatasetRecord,
    CandidateGoldLabel,
    CatalogEventRecoveryRecord,
    RecoveryState,
)
from transiteye.serialization import canonical_json


class DatasetBuildError(ValueError):
    """Raised when frozen inputs cannot form an auditable dataset."""


@dataclass(frozen=True)
class BuiltDataset:
    candidates: pd.DataFrame
    candidate_event_matches: pd.DataFrame
    catalog_events: pd.DataFrame
    artifact_lineage: pd.DataFrame


def _label_for_disposition(value: object) -> CandidateGoldLabel:
    mapping = map_disposition(None if bool(pd.isna(cast(Any, value))) else str(value))
    if mapping.internal_label is InternalLabel.GOLD_POSITIVE:
        return CandidateGoldLabel.POSITIVE
    if mapping.internal_label is InternalLabel.GOLD_NEGATIVE:
        return CandidateGoldLabel.NEGATIVE
    return CandidateGoldLabel.UNLABELED


def _optional_float(value: object) -> float | None:
    return None if bool(pd.isna(cast(Any, value))) else float(cast(Any, value))


def build_candidate_tables(
    frozen_candidates: pd.DataFrame,
    frozen_matches: pd.DataFrame,
    artifact_lineage: pd.DataFrame,
    *,
    matching_config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate rows and an auditable many-to-many catalog relation table."""
    candidate_required = {
        "candidate_id",
        "observation_group_id",
        "rank",
        "period",
        "duration",
        "epoch",
        "power",
        "depth",
        "relation_to_stronger",
        "harmonic_ratio",
    }
    lineage_required = {
        "object_id",
        "observation_id",
        "observation_group_id",
        "sector",
        "source_product_id",
        "source_raw_checksum",
        "preprocessing_config_hash",
        "bls_config_hash",
    }
    for name, frame, required in (
        ("candidates", frozen_candidates, candidate_required),
        ("lineage", artifact_lineage, lineage_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DatasetBuildError(f"{name} missing required columns: {missing}")

    base = frozen_candidates.merge(
        artifact_lineage[list(lineage_required)],
        on="observation_group_id",
        how="left",
        validate="many_to_one",
    )
    if base["object_id"].isna().any():
        raise DatasetBuildError("Candidate has no explicit artifact lineage.")
    if base["candidate_id"].duplicated().any():
        raise DatasetBuildError("Candidate IDs must be unique before label assignment.")

    if frozen_matches.empty:
        relations = pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_object_id",
                "event_id",
                "event_object_id",
                "source_disposition",
                "candidate_period",
                "catalog_period",
                "relative_period_error",
                "candidate_epoch",
                "catalog_epoch",
                "phase_error",
                "harmonic_ratio",
                "match_type",
                "matched",
                "matching_config_hash",
            ]
        )
    else:
        match_required = {
            "candidate_id",
            "toi_id",
            "object_id",
            "source_disposition",
            "candidate_period",
            "catalog_period",
            "relative_period_error",
            "candidate_epoch",
            "catalog_epoch",
            "phase_error",
            "harmonic_ratio",
            "match_type",
            "matched",
        }
        missing = sorted(match_required.difference(frozen_matches.columns))
        if missing:
            raise DatasetBuildError(f"matches missing required columns: {missing}")
        object_lookup = base.set_index("candidate_id")["object_id"]
        relations = frozen_matches[list(match_required)].copy()
        relations = relations.rename(
            columns={
                "toi_id": "event_id",
                "object_id": "event_object_id",
                "harmonic_ratio": "match_harmonic_ratio",
            }
        )
        relations["candidate_object_id"] = relations["candidate_id"].map(object_lookup)
        if relations["candidate_object_id"].isna().any():
            raise DatasetBuildError("Catalog relation refers to an unknown candidate.")
        relations["matching_config_hash"] = matching_config_hash
        relations = relations[
            [
                "candidate_id",
                "candidate_object_id",
                "event_id",
                "event_object_id",
                "source_disposition",
                "candidate_period",
                "catalog_period",
                "relative_period_error",
                "candidate_epoch",
                "catalog_epoch",
                "phase_error",
                "match_harmonic_ratio",
                "match_type",
                "matched",
                "matching_config_hash",
            ]
        ].sort_values(["candidate_id", "event_id"], kind="stable")

    candidate_rows: list[dict[str, Any]] = []
    valid_relations = relations.loc[relations["matched"].astype(bool)]
    priority = {"fundamental": 0, "half_period": 1, "double_period": 1}
    for row in base.to_dict(orient="records"):
        matched = valid_relations.loc[valid_relations["candidate_id"] == row["candidate_id"]].copy()
        labels = {_label_for_disposition(value) for value in matched["source_disposition"]}
        gold_labels = labels.intersection(
            {CandidateGoldLabel.POSITIVE, CandidateGoldLabel.NEGATIVE}
        )
        if len(gold_labels) > 1:
            label = CandidateGoldLabel.AMBIGUOUS
        elif gold_labels:
            label = next(iter(gold_labels))
        else:
            label = CandidateGoldLabel.UNLABELED
        primary: dict[str, Any] | None = None
        if not matched.empty:
            matched["_priority"] = matched["match_type"].map(priority).fillna(2)
            matched = matched.sort_values(
                ["_priority", "relative_period_error", "phase_error", "event_id"], kind="stable"
            )
            primary = matched.iloc[0].to_dict()
        record = CandidateDatasetRecord(
            candidate_id=str(row["candidate_id"]),
            object_id=str(row["object_id"]),
            observation_id=str(row["observation_id"]),
            observation_group_id=str(row["observation_group_id"]),
            sector=int(row["sector"]),
            source_product_id=str(row["source_product_id"]),
            source_raw_checksum=str(row["source_raw_checksum"]),
            preprocessing_config_hash=str(row["preprocessing_config_hash"]),
            bls_config_hash=str(row["bls_config_hash"]),
            candidate_rank=int(row["rank"]),
            candidate_period=float(row["period"]),
            candidate_duration=float(row["duration"]),
            candidate_epoch=float(row["epoch"]),
            candidate_depth=float(row["depth"]),
            depth_uncertainty=None,
            bls_power=float(row["power"]),
            relation_to_stronger=(
                None if pd.isna(row["relation_to_stronger"]) else str(row["relation_to_stronger"])
            ),
            detection_harmonic_ratio=_optional_float(row["harmonic_ratio"]),
            catalog_match_status=("matched" if primary else "unmatched"),
            matched_event_id=(str(primary["event_id"]) if primary else None),
            source_disposition=(str(primary["source_disposition"]) if primary else None),
            gold_candidate_label=label,
            gold_training_eligible=label
            in {
                CandidateGoldLabel.POSITIVE,
                CandidateGoldLabel.NEGATIVE,
            },
            match_type=(str(primary["match_type"]) if primary else None),
            match_harmonic_ratio=(
                _optional_float(primary["match_harmonic_ratio"]) if primary else None
            ),
            matching_config_hash=matching_config_hash,
        )
        candidate_rows.append(record.model_dump(mode="json"))
    candidates = (
        pd.DataFrame(candidate_rows)
        .sort_values(["object_id", "observation_group_id", "candidate_rank"], kind="stable")
        .reset_index(drop=True)
    )
    return candidates, relations.reset_index(drop=True)


def build_catalog_event_recovery(
    catalog_events: pd.DataFrame,
    observations: pd.DataFrame,
    products: pd.DataFrame,
    receipts: pd.DataFrame,
    artifact_lineage: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_event_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate acquisition and detection evidence without losing event-level truth."""
    needed = {"object_id", "toi_id", "source_disposition", "mapped_label"}
    missing = sorted(needed.difference(catalog_events.columns))
    if missing:
        raise DatasetBuildError(f"catalog events missing required columns: {missing}")
    if catalog_events["toi_id"].duplicated().any():
        raise DatasetBuildError("Catalog event identities must be unique.")

    lightcurves = products.loc[
        products["product_filename"].fillna("").astype(str).str.lower().str.endswith("_lc.fits")
    ]
    matched = candidate_event_matches.loc[candidate_event_matches["matched"].astype(bool)].merge(
        candidates[["candidate_id", "candidate_rank"]], on="candidate_id", how="left"
    )
    event_rows: list[dict[str, Any]] = []
    for event in catalog_events.to_dict(orient="records"):
        object_id = str(event["object_id"])
        obs = observations.loc[observations["object_id"] == object_id]
        discovered = lightcurves.loc[lightcurves["object_id"] == object_id]
        downloaded = receipts.loc[receipts["object_id"] == object_id]
        groups = artifact_lineage.loc[artifact_lineage["object_id"] == object_id]
        preprocessed = groups.loc[groups["processed_checksum"].notna()]
        searched = groups.loc[groups["detection_status"] == "success"]
        failed = groups.loc[groups["detection_status"] == "failed"]
        event_matches = matched.loc[matched["event_id"] == event["toi_id"]].copy()
        fundamental = event_matches.loc[event_matches["match_type"] == "fundamental"]
        harmonic = event_matches.loc[
            event_matches["match_type"].isin(["half_period", "double_period", "harmonic"])
        ]
        primary: str | None = None
        if not fundamental.empty:
            state = RecoveryState.RECOVERED_FUNDAMENTAL
            primary = str(
                fundamental.sort_values("candidate_rank", kind="stable").iloc[0]["candidate_id"]
            )
        elif not harmonic.empty:
            state = RecoveryState.RECOVERED_HARMONIC
            primary = str(
                harmonic.sort_values("candidate_rank", kind="stable").iloc[0]["candidate_id"]
            )
        elif not searched.empty:
            state = RecoveryState.SEARCHED_NOT_RECOVERED
        elif not failed.empty:
            state = RecoveryState.DETECTION_FAILED
        elif not preprocessed.empty:
            state = RecoveryState.PREPROCESSED_NOT_SEARCHED
        elif not downloaded.empty:
            state = RecoveryState.DOWNLOADED_NOT_PREPROCESSED
        elif not discovered.empty:
            state = RecoveryState.PRODUCT_DISCOVERED_NOT_DOWNLOADED
        elif obs.empty:
            state = RecoveryState.UNAVAILABLE_ACQUISITION
        else:
            state = RecoveryState.NOT_SEARCHED
        group_ids = sorted(searched["observation_group_id"].astype(str).unique())
        record = CatalogEventRecoveryRecord(
            event_id=str(event["toi_id"]),
            object_id=object_id,
            source_disposition=str(event["source_disposition"]),
            internal_event_class=str(event["mapped_label"]),
            catalog_period=_optional_float(event.get("period_days")),
            catalog_epoch=_optional_float(event.get("transit_epoch_bjd")),
            catalog_duration_hours=_optional_float(event.get("transit_duration_hours")),
            acquisition_observation_count=len(obs),
            discovered_product_count=len(discovered),
            downloaded_product_count=len(downloaded),
            preprocessed_group_count=len(preprocessed),
            searched_group_count=len(searched),
            matching_candidate_count=len(event_matches),
            fundamental_candidate_count=len(fundamental),
            harmonic_candidate_count=len(harmonic),
            primary_candidate_id=primary,
            recovery_state=state,
            contributing_observation_group_ids=canonical_json(group_ids),
        )
        event_rows.append(record.model_dump(mode="json"))
    return (
        pd.DataFrame(event_rows)
        .sort_values(["object_id", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_dataset(
    *,
    frozen_candidates: pd.DataFrame,
    frozen_matches: pd.DataFrame,
    catalog_events: pd.DataFrame,
    observations: pd.DataFrame,
    products: pd.DataFrame,
    receipts: pd.DataFrame,
    artifact_lineage: pd.DataFrame,
    matching_config_hash: str,
) -> BuiltDataset:
    """Build all B031 tables from already-frozen upstream artifacts."""
    candidates, relations = build_candidate_tables(
        frozen_candidates,
        frozen_matches,
        artifact_lineage,
        matching_config_hash=matching_config_hash,
    )
    events = build_catalog_event_recovery(
        catalog_events,
        observations,
        products,
        receipts,
        artifact_lineage,
        candidates,
        relations,
    )
    return BuiltDataset(candidates, relations, events, artifact_lineage.copy(deep=True))
