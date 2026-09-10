"""Adversarial grouping, truth-lineage, and candidate-label checks for demo mode."""

from __future__ import annotations

from typing import Any

import pandas as pd

from transiteye.datasets.schemas import assert_model_input_boundary


class DemoValidationError(ValueError):
    """Raised when demo lineage, grouping, or truth-label semantics are violated."""


def validate_demo_dataset(
    candidates: pd.DataFrame,
    relations: pd.DataFrame,
    events: pd.DataFrame,
    lineage: pd.DataFrame,
    split: pd.DataFrame,
    *,
    model_input_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reject leakage and verify match-based synthetic candidate labels."""
    try:
        assert_model_input_boundary(model_input_columns)
    except ValueError as exc:
        raise DemoValidationError(str(exc)) from exc
    for frame, key, description in (
        (candidates, "candidate_id", "candidate"),
        (events, "synthetic_event_id", "synthetic event"),
        (lineage, "variant_id", "variant"),
    ):
        if frame[key].duplicated().any():
            raise DemoValidationError(f"Duplicate {description} identity detected.")
    if split["object_id"].duplicated().any():
        raise DemoValidationError("Source TIC has duplicate split assignments.")
    split_map = split.set_index("object_id")["split"]
    if not set(lineage["object_id"]).issubset(split_map.index):
        raise DemoValidationError("Demo lineage TIC is absent from split manifest.")
    grouped = lineage.assign(split=lineage["object_id"].map(split_map))
    for key, message in (
        ("object_id", "Source TIC"),
        ("source_raw_checksum", "Source checksum"),
        ("source_observation_id", "Source observation"),
    ):
        if (grouped.groupby(key)["split"].nunique() > 1).any():
            raise DemoValidationError(f"{message} crosses demo partitions.")
    candidate_variant = candidates.set_index("candidate_id")["variant_id"]
    candidate_object = candidates.set_index("candidate_id")["object_id"]
    event_variant = events.set_index("synthetic_event_id")["variant_id"]
    event_object = events.set_index("synthetic_event_id")["object_id"]
    event_gold = events.set_index("synthetic_event_id")["demo_gold_class"]
    if not relations.empty:
        if relations["candidate_id"].map(candidate_variant).isna().any():
            raise DemoValidationError("Relation refers to unknown demo candidate.")
        if relations["synthetic_event_id"].map(event_variant).isna().any():
            raise DemoValidationError("Relation refers to unknown synthetic event.")
        if not relations["variant_id"].eq(relations["candidate_id"].map(candidate_variant)).all():
            raise DemoValidationError("Candidate relation crosses demo variants.")
        if not relations["variant_id"].eq(relations["synthetic_event_id"].map(event_variant)).all():
            raise DemoValidationError("Truth relation crosses demo variants.")
        if not relations["object_id"].eq(relations["candidate_id"].map(candidate_object)).all():
            raise DemoValidationError("Candidate relation crosses source TICs.")
        if not relations["object_id"].eq(relations["synthetic_event_id"].map(event_object)).all():
            raise DemoValidationError("Synthetic event relation crosses source TICs.")
    matched = relations.loc[relations["matched"].astype(bool)]
    for row in candidates.to_dict(orient="records"):
        related = matched.loc[matched["candidate_id"] == row["candidate_id"]]
        labels = set(related["synthetic_event_id"].map(event_gold).dropna())
        if "positive" in labels and "negative" in labels:
            expected = "ambiguous"
        elif "positive" in labels:
            expected = "positive"
        elif "negative" in labels:
            expected = "negative"
        else:
            expected = "unlabeled"
        if row["gold_candidate_label"] != expected:
            raise DemoValidationError("Demo candidate label conflicts with matched truth.")
        eligible = expected in {"positive", "negative"}
        if bool(row["gold_training_eligible"]) != eligible:
            raise DemoValidationError("Demo candidate eligibility conflicts with gold label.")
        variant_event = events.loc[events["variant_id"] == row["variant_id"]]
        if not variant_event.empty:
            synthetic_class = str(variant_event.iloc[0]["synthetic_event_class"])
            if synthetic_class == "no_injection_control" and expected != "unlabeled":
                raise DemoValidationError("No-injection control candidate cannot be gold labeled.")
        if related.empty and expected == "negative":
            raise DemoValidationError("Unmatched demo candidate was encoded as negative.")
    class_recovery = (
        events.groupby(["synthetic_event_class", "recovery_state"])
        .size()
        .rename("count")
        .reset_index()
    )
    return {
        "source_tics": int(lineage["object_id"].nunique()),
        "source_light_curves": int(lineage["source_product_id"].nunique()),
        "variants": len(lineage),
        "preprocessing": lineage["preprocessing_status"].value_counts().to_dict(),
        "detection": lineage["detection_status"].value_counts().to_dict(),
        "candidates": {
            "total": len(candidates),
            "by_label": candidates["gold_candidate_label"].value_counts().to_dict(),
        },
        "recovery_by_class": {
            f"{row['synthetic_event_class']}:{row['recovery_state']}": int(row["count"])
            for row in class_recovery.to_dict(orient="records")
        },
        "source_tics_per_partition": split.groupby("split")["object_id"].nunique().to_dict(),
        "zero_source_tic_overlap": True,
        "zero_source_checksum_overlap": True,
        "positive_source_tics": int(
            candidates.loc[candidates["gold_candidate_label"] == "positive", "object_id"].nunique()
        ),
        "negative_source_tics": int(
            candidates.loc[candidates["gold_candidate_label"] == "negative", "object_id"].nunique()
        ),
    }
