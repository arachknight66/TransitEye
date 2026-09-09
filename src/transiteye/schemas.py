"""Minimal typed records for data identity and lineage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from transiteye.identifiers import (
    make_candidate_id,
    make_observation_id,
    normalize_object_id,
    validate_sha256,
    validate_short_hash,
)


class LineageRecord(BaseModel):
    """Base settings shared by immutable lineage records."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RawProductRecord(LineageRecord):
    """Identity and provenance for one immutable observational product."""

    object_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    sector: int = Field(gt=0)
    author: str = Field(min_length=1)
    cadence_seconds: float = Field(gt=0)
    product_id: str = Field(min_length=1)
    checksum_sha256: str
    source_uri: str = Field(min_length=1)

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)

    @field_validator("checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def validate_observation_identity(self) -> RawProductRecord:
        expected = make_observation_id(
            object_id=self.object_id,
            sector=self.sector,
            author=self.author,
            cadence_seconds=self.cadence_seconds,
            product_id=self.product_id,
            checksum_sha256=self.checksum_sha256,
        )
        if self.observation_id != expected:
            raise ValueError("Observation ID does not match the canonical construction rule.")
        return self


class ProcessedCurveRecord(LineageRecord):
    """Lineage for one future curve derived from an immutable observation."""

    object_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    preprocessing_config_hash: str
    input_checksum_sha256: str
    output_checksum_sha256: str

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)

    @field_validator("preprocessing_config_hash")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        return validate_short_hash(value)

    @field_validator("input_checksum_sha256", "output_checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        return validate_sha256(value)


class CandidateRecord(LineageRecord):
    """Identity and lineage for one ranked future BLS candidate."""

    candidate_id: str = Field(min_length=1)
    observation_group_id: str = Field(min_length=1)
    bls_config_hash: str
    rank: int = Field(ge=1)

    @field_validator("bls_config_hash")
    @classmethod
    def validate_bls_hash(cls, value: str) -> str:
        return validate_short_hash(value)

    @model_validator(mode="after")
    def validate_candidate_identity(self) -> CandidateRecord:
        expected = make_candidate_id(
            observation_group_id=self.observation_group_id,
            bls_config_hash=self.bls_config_hash,
            rank=self.rank,
        )
        if self.candidate_id != expected:
            raise ValueError("Candidate ID does not match the canonical construction rule.")
        return self


class LabelRecord(LineageRecord):
    """A candidate label with portable source and policy references."""

    candidate_id: str = Field(min_length=1)
    label: Literal["positive", "negative", "unlabeled"]
    source_record_id: str = Field(min_length=1)
    label_policy_hash: str

    @field_validator("label_policy_hash")
    @classmethod
    def validate_policy_hash(cls, value: str) -> str:
        return validate_short_hash(value)


class SplitRecord(LineageRecord):
    """An object-level assignment to one immutable dataset partition."""

    object_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    split_policy_hash: str

    @field_validator("object_id")
    @classmethod
    def normalize_object(cls, value: str) -> str:
        return normalize_object_id(value)

    @field_validator("split_policy_hash")
    @classmethod
    def validate_policy_hash(cls, value: str) -> str:
        return validate_short_hash(value)
