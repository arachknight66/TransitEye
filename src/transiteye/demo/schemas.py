"""Typed provenance and truth contracts for additive demo artifacts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transiteye.identifiers import normalize_object_id, validate_sha256, validate_short_hash


class SyntheticEventClass(StrEnum):
    PLANET_LIKE = "planet_like"
    ECLIPSING_BINARY_LIKE = "eclipsing_binary_like"
    SINUSOIDAL_VARIABILITY_LIKE = "sinusoidal_variability_like"
    NO_INJECTION_CONTROL = "no_injection_control"


class DemoGoldClass(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNLABELED = "unlabeled"


class DemoRecoveryState(StrEnum):
    RECOVERED_FUNDAMENTAL = "recovered_fundamental"
    RECOVERED_HARMONIC = "recovered_harmonic"
    SEARCHED_NOT_RECOVERED = "searched_not_recovered"
    PREPROCESSING_FAILED = "preprocessing_failed"
    DETECTION_FAILED = "detection_failed"
    NOT_SEARCHED = "not_searched"


class DemoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DemoVariantRecord(DemoRecord):
    variant_id: str = Field(min_length=1)
    variant_name: str = Field(min_length=1)
    synthetic_event_id: str = Field(min_length=1)
    object_id: str
    source_observation_id: str = Field(min_length=1)
    source_product_id: str = Field(min_length=1)
    source_raw_checksum: str
    source_preprocessing_hash: str
    source_processed_checksum: str
    synthetic_event_class: SyntheticEventClass
    demo_gold_class: DemoGoldClass
    difficulty: str = Field(min_length=1)
    injection_seed: int = Field(ge=0)
    demo_policy_hash: str
    generator_version: str = Field(min_length=1)
    dataset_role: str = "demo"
    truth_source: str = "synthetic_injection"

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)

    @field_validator("source_raw_checksum", "source_processed_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("source_preprocessing_hash", "demo_policy_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return validate_short_hash(value)


class SyntheticEventRecord(DemoRecord):
    synthetic_event_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    object_id: str
    synthetic_event_class: SyntheticEventClass
    demo_gold_class: DemoGoldClass
    injected_period_days: float | None = Field(default=None, gt=0)
    injected_epoch: float | None = None
    injected_duration_days: float | None = Field(default=None, gt=0)
    injected_depth_or_amplitude: float | None = Field(default=None, gt=0, lt=1)
    injected_secondary_depth: float | None = Field(default=None, ge=0, lt=1)
    injection_seed: int = Field(ge=0)
    difficulty: str = Field(min_length=1)
    variant_name: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    dataset_role: str = "demo"
    truth_source: str = "synthetic_injection"

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)
