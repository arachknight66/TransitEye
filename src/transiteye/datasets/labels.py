"""Provenance-rich labels; this module never infers a negative label from catalog absence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class MorphologyLabel(StrEnum):
    """Curated labels allowed in the Phase 1 cohort."""

    TRANSIT_LIKE = "transit_like"
    ECLIPSE_LIKE = "eclipse_like"
    STELLAR_VARIABILITY = "stellar_variability"
    OTHER_ARTIFACT = "other_artifact"


class LabelSource(StrEnum):
    """Evidence sources that can justify a curated label."""

    EXOPLANET_ARCHIVE = "exoplanet_archive"
    TESS_EB_CATALOG = "tess_eb_catalog"
    VARIABLE_STAR_CATALOG = "variable_star_catalog"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One source-level label with enough evidence for an independent audit."""

    source_group_id: str
    target_id: str
    label: MorphologyLabel
    label_source: LabelSource
    evidence_uri: str
    evidence_identifier: str
    review_status: str
    snapshot_date: str
    sectors: tuple[int, ...]
    notes: str = ""
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if not self.source_group_id or not self.target_id:
            raise ValueError("source_group_id and target_id are required.")
        if not self.evidence_uri.startswith(("https://", "http://")):
            raise ValueError("evidence_uri must be an HTTP(S) URL.")
        if self.review_status not in {"verified", "reviewed", "provisional"}:
            raise ValueError("review_status must be verified, reviewed, or provisional.")
        if not self.sectors:
            raise ValueError("A label record must identify one or more sectors.")


def write_labels(labels: tuple[LabelRecord, ...], path: Path) -> None:
    """Write a small, reviewable label snapshot with stable JSON field names."""

    if len({label.source_group_id for label in labels}) != len(labels):
        raise ValueError("Phase 1 pilot labels must contain one label per source group.")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": "1", "labels": [asdict(label) for label in labels]}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_labels(path: Path) -> tuple[LabelRecord, ...]:
    """Read a label snapshot and reject malformed or duplicate-source records."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "1":
        raise ValueError("Unsupported label schema version.")
    labels = tuple(
        LabelRecord(
            source_group_id=item["source_group_id"],
            target_id=item["target_id"],
            label=MorphologyLabel(item["label"]),
            label_source=LabelSource(item["label_source"]),
            evidence_uri=item["evidence_uri"],
            evidence_identifier=item["evidence_identifier"],
            review_status=item["review_status"],
            snapshot_date=item["snapshot_date"],
            sectors=tuple(item["sectors"]),
            notes=item.get("notes", ""),
            ambiguous=bool(item.get("ambiguous", False)),
        )
        for item in document.get("labels", [])
    )
    if len({label.source_group_id for label in labels}) != len(labels):
        raise ValueError("Label snapshot contains duplicate source groups.")
    return labels
