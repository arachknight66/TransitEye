"""Adversarial duplicate, lineage, split, and label validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from transiteye.catalogs.labels import InternalLabel, map_disposition
from transiteye.datasets.builder import BuiltDataset
from transiteye.datasets.schemas import CandidateGoldLabel


class DatasetValidationError(ValueError):
    """Raised when the statistical dataset contract is violated."""


def _reject_duplicate(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    if frame.duplicated(columns, keep=False).any():
        raise DatasetValidationError(f"Duplicate {description} detected.")


def _reject_cross_split(frame: pd.DataFrame, key: str, description: str) -> None:
    if frame.empty or key not in frame:
        return
    counts = frame.dropna(subset=[key]).groupby(key)["split"].nunique()
    if (counts > 1).any():
        raise DatasetValidationError(f"{description} crosses dataset splits.")


def validate_dataset(
    dataset: BuiltDataset,
    split_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Reject leakage and semantically invalid candidate/event states."""
    candidates = dataset.candidates
    events = dataset.catalog_events
    relations = dataset.candidate_event_matches
    lineage = dataset.artifact_lineage
    _reject_duplicate(candidates, ["candidate_id"], "candidate identity")
    _reject_duplicate(events, ["event_id"], "catalog event identity")
    if not relations.empty:
        _reject_duplicate(relations, ["candidate_id", "event_id"], "candidate-event relationship")
    _reject_duplicate(
        candidates,
        ["observation_group_id", "candidate_rank"],
        "candidate rank within observation group",
    )
    if split_manifest["object_id"].duplicated().any():
        repeated = split_manifest.groupby("object_id")["split"].nunique()
        if (repeated > 1).any():
            raise DatasetValidationError("A TIC crosses dataset splits.")
        raise DatasetValidationError("Duplicate TIC split assignment detected.")
    split_map = split_manifest.set_index("object_id")["split"]
    all_objects = (
        set(candidates["object_id"]) | set(events["object_id"]) | set(lineage["object_id"])
    )
    if not all_objects.issubset(set(split_map.index)):
        raise DatasetValidationError("An artifact TIC is missing from the split manifest.")

    lineage_with_split = lineage.assign(split=lineage["object_id"].map(split_map))
    for key, description in (
        ("observation_group_id", "Observation group"),
        ("observation_id", "Observation/product identity"),
        ("mast_obs_id", "MAST observation identity"),
        ("source_product_id", "Source product identity"),
        ("source_raw_checksum", "Raw checksum/product alias"),
        ("processed_artifact_id", "Processed artifact"),
    ):
        _reject_cross_split(lineage_with_split, key, description)

    candidate_lineage = candidates.merge(
        lineage[["observation_group_id", "object_id"]],
        on="observation_group_id",
        suffixes=("_candidate", "_lineage"),
        how="left",
        validate="many_to_one",
    )
    if (
        candidate_lineage["object_id_lineage"].isna().any()
        or not candidate_lineage["object_id_candidate"]
        .eq(candidate_lineage["object_id_lineage"])
        .all()
    ):
        raise DatasetValidationError("Candidate/TIC lineage mismatch detected.")

    candidate_objects = candidates.set_index("candidate_id")["object_id"]
    event_objects = events.set_index("event_id")["object_id"]
    if not relations.empty:
        if relations["candidate_id"].map(candidate_objects).isna().any():
            raise DatasetValidationError("Relation refers to an unknown candidate.")
        if relations["event_id"].map(event_objects).isna().any():
            raise DatasetValidationError("Relation refers to an unknown event.")
        candidate_mismatch = ~relations["candidate_object_id"].eq(
            relations["candidate_id"].map(candidate_objects)
        )
        event_mismatch = ~relations["event_object_id"].eq(relations["event_id"].map(event_objects))
        if candidate_mismatch.any():
            raise DatasetValidationError("Candidate relation has inconsistent TIC lineage.")
        if event_mismatch.any():
            raise DatasetValidationError("Event relation has inconsistent TIC lineage.")
        if (~relations["candidate_object_id"].eq(relations["event_object_id"])).any():
            raise DatasetValidationError("Candidate-event match connects different TICs.")

    for frame, name in ((candidates, "candidate"), (events, "event")):
        if "split" in frame and not frame["split"].eq(frame["object_id"].map(split_map)).all():
            raise DatasetValidationError(
                f"{name.capitalize()} split is inconsistent with TIC split."
            )

    valid_relations = (
        relations.loc[relations["matched"].astype(bool)] if not relations.empty else relations
    )
    for row in candidates.to_dict(orient="records"):
        related = valid_relations.loc[valid_relations["candidate_id"] == row["candidate_id"]]
        expected: set[CandidateGoldLabel] = set()
        for disposition in related["source_disposition"]:
            mapped = map_disposition(str(disposition)).internal_label
            if mapped is InternalLabel.GOLD_POSITIVE:
                expected.add(CandidateGoldLabel.POSITIVE)
            elif mapped is InternalLabel.GOLD_NEGATIVE:
                expected.add(CandidateGoldLabel.NEGATIVE)
        if len(expected) > 1:
            label = CandidateGoldLabel.AMBIGUOUS
        elif expected:
            label = next(iter(expected))
        else:
            label = CandidateGoldLabel.UNLABELED
        if row["gold_candidate_label"] != label.value:
            raise DatasetValidationError("Candidate gold label conflicts with matched disposition.")
        eligible = label in {CandidateGoldLabel.POSITIVE, CandidateGoldLabel.NEGATIVE}
        if bool(row["gold_training_eligible"]) != eligible:
            raise DatasetValidationError(
                "Unlabeled/ambiguous candidate has invalid training eligibility."
            )
        if related.empty and row["gold_candidate_label"] == CandidateGoldLabel.NEGATIVE.value:
            raise DatasetValidationError("An unmatched candidate was encoded as gold negative.")

    return dataset_qa(dataset, split_manifest)


def dataset_qa(dataset: BuiltDataset, split_manifest: pd.DataFrame) -> dict[str, Any]:
    """Return non-model QA counts for the frozen statistical dataset."""
    candidates = dataset.candidates
    events = dataset.catalog_events
    lineage = dataset.artifact_lineage
    event_counts = events["source_disposition"].value_counts().sort_index()
    recovery_counts = events["recovery_state"].value_counts().sort_index()
    label_counts = candidates["gold_candidate_label"].value_counts().sort_index()
    match_counts = candidates["match_type"].fillna("unmatched").value_counts().sort_index()
    by_rank = candidates["candidate_rank"].value_counts().sort_index()
    events_by_tic = events.groupby("object_id").size()
    gold_sets = (
        events.loc[events["source_disposition"].isin(["CP", "KP", "FP", "FA"])]
        .groupby("object_id")["internal_event_class"]
        .nunique()
    )
    split_objects = split_manifest.groupby("split")["object_id"].nunique().sort_index()
    event_split = events.assign(
        split=events["object_id"].map(split_manifest.set_index("object_id")["split"])
    )
    candidate_split = candidates.assign(
        split=candidates["object_id"].map(split_manifest.set_index("object_id")["split"])
    )
    return {
        "catalog_events": {
            "total": len(events),
            "by_disposition": {str(k): int(v) for k, v in event_counts.items()},
            "multi_event_tics": int((events_by_tic > 1).sum()),
            "mixed_disposition_tics": int((gold_sets > 1).sum()),
        },
        "coverage": {
            "product_discovered_tics": int(
                (events["discovered_product_count"] > 0).groupby(events["object_id"]).any().sum()
            ),
            "downloaded_tics": int(lineage["object_id"].nunique()),
            "preprocessed_tics": int(
                lineage.loc[lineage["processed_checksum"].notna(), "object_id"].nunique()
            ),
            "bls_searched_tics": int(
                lineage.loc[lineage["detection_status"] == "success", "object_id"].nunique()
            ),
        },
        "recovery": {str(k): int(v) for k, v in recovery_counts.items()},
        "candidates": {
            "total": len(candidates),
            "by_label": {str(k): int(v) for k, v in label_counts.items()},
            "by_rank": {str(k): int(v) for k, v in by_rank.items()},
            "by_match_type": {str(k): int(v) for k, v in match_counts.items()},
        },
        "splits": {
            "dataset_role": str(split_manifest["dataset_role"].iloc[0]),
            "tics": {str(k): int(v) for k, v in split_objects.items()},
            "events": {str(k): int(v) for k, v in event_split.groupby("split").size().items()},
            "candidates": {
                str(k): int(v) for k, v in candidate_split.groupby("split").size().items()
            },
            "zero_tic_overlap": True,
            "zero_checksum_overlap": True,
        },
    }
