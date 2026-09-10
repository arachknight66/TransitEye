"""Typed statistical dataset states and row contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transiteye.identifiers import normalize_object_id, validate_sha256, validate_short_hash


class CandidateGoldLabel(StrEnum):
    """Gold candidate state; unlabeled and ambiguous are never encoded as zero."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNLABELED = "unlabeled"
    AMBIGUOUS = "ambiguous"


class RecoveryState(StrEnum):
    """Event-level acquisition-to-detection outcome."""

    RECOVERED_FUNDAMENTAL = "recovered_fundamental"
    RECOVERED_HARMONIC = "recovered_harmonic"
    SEARCHED_NOT_RECOVERED = "searched_not_recovered"
    DETECTION_FAILED = "detection_failed"
    PREPROCESSED_NOT_SEARCHED = "preprocessed_not_searched"
    DOWNLOADED_NOT_PREPROCESSED = "downloaded_not_preprocessed"
    PRODUCT_DISCOVERED_NOT_DOWNLOADED = "product_discovered_not_downloaded"
    UNAVAILABLE_ACQUISITION = "unavailable_acquisition"
    NOT_SEARCHED = "not_searched"


class DatasetRole(StrEnum):
    DEVELOPMENT = "development"
    FINAL = "final"


class DatasetSplit(StrEnum):
    DEVELOPMENT = "development"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CandidateDatasetRecord(DatasetRow):
    """Minimal typed row for one BLS candidate hypothesis."""

    candidate_id: str = Field(min_length=1)
    object_id: str
    observation_id: str = Field(min_length=1)
    observation_group_id: str = Field(min_length=1)
    sector: int = Field(gt=0)
    source_product_id: str = Field(min_length=1)
    source_raw_checksum: str
    preprocessing_config_hash: str
    bls_config_hash: str
    candidate_rank: int = Field(ge=1)
    candidate_period: float = Field(gt=0)
    candidate_duration: float = Field(gt=0)
    candidate_epoch: float
    candidate_depth: float
    depth_uncertainty: float | None = None
    bls_power: float
    relation_to_stronger: str | None = None
    detection_harmonic_ratio: float | None = None
    catalog_match_status: str
    matched_event_id: str | None = None
    source_disposition: str | None = None
    gold_candidate_label: CandidateGoldLabel
    gold_training_eligible: bool
    match_type: str | None = None
    match_harmonic_ratio: float | None = None
    matching_config_hash: str

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)

    @field_validator("source_raw_checksum")
    @classmethod
    def validate_raw_checksum(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("preprocessing_config_hash", "bls_config_hash", "matching_config_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return validate_short_hash(value)


class CatalogEventRecoveryRecord(DatasetRow):
    """Minimal typed row for one frozen catalog event and its recovery state."""

    event_id: str = Field(min_length=1)
    object_id: str
    source_disposition: str
    internal_event_class: str
    catalog_period: float | None = None
    catalog_epoch: float | None = None
    catalog_duration_hours: float | None = None
    acquisition_observation_count: int = Field(ge=0)
    discovered_product_count: int = Field(ge=0)
    downloaded_product_count: int = Field(ge=0)
    preprocessed_group_count: int = Field(ge=0)
    searched_group_count: int = Field(ge=0)
    matching_candidate_count: int = Field(ge=0)
    fundamental_candidate_count: int = Field(ge=0)
    harmonic_candidate_count: int = Field(ge=0)
    primary_candidate_id: str | None = None
    recovery_state: RecoveryState
    contributing_observation_group_ids: str

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)


# Catalog truth is evaluation/label metadata and must never be promoted by a
# generic future feature-table selection operation.
FORBIDDEN_MODEL_INPUT_COLUMNS = frozenset(
    {
        "catalog_period",
        "catalog_epoch",
        "catalog_duration_hours",
        "source_disposition",
        "internal_event_class",
        "gold_candidate_label",
        "gold_training_eligible",
        "catalog_match_status",
        "matched_event_id",
        "match_type",
        "match_harmonic_ratio",
    }
)


def assert_model_input_boundary(columns: list[str] | tuple[str, ...] | set[str]) -> None:
    """Reject catalog/label fields from a future model-input column list."""
    leaked = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS.intersection(columns))
    if leaked:
        raise ValueError(f"Catalog evaluation metadata cannot be model inputs: {leaked}")
