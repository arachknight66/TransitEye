from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from transiteye.identifiers import (
    make_candidate_id,
    make_dataset_version,
    make_experiment_id,
    make_observation_group_id,
    make_observation_id,
    make_run_fingerprint,
    normalize_object_id,
)
from transiteye.schemas import (
    CandidateRecord,
    LabelRecord,
    ProcessedCurveRecord,
    RawProductRecord,
    SplitRecord,
)

CHECKSUM = "a" * 64
HASH = "b" * 20


def _observation_id(*, product_id: str = "product-1") -> str:
    return make_observation_id(
        object_id="TIC 123",
        sector=1,
        author="SPOC",
        cadence_seconds=120.0,
        product_id=product_id,
        checksum_sha256=CHECKSUM,
    )


def test_object_id_normalization() -> None:
    assert normalize_object_id("TIC 000123") == "tic-123"
    assert normalize_object_id(123) == "tic-123"


@pytest.mark.parametrize("value", [True, 0, "TIC invalid", 1.5])
def test_object_id_normalization_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_object_id(value)  # type: ignore[arg-type]


def test_identifiers_are_deterministic_and_path_independent() -> None:
    first = _observation_id()
    second = _observation_id()

    assert first == second
    assert "/" not in first
    assert first.startswith("obs-")


def test_identifier_changes_for_scientifically_relevant_input() -> None:
    assert _observation_id(product_id="product-1") != _observation_id(product_id="product-2")


def test_identifier_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        make_observation_id(
            object_id="TIC 123",
            sector=0,
            author="SPOC",
            cadence_seconds=120.0,
            product_id="product-1",
            checksum_sha256=CHECKSUM,
        )
    with pytest.raises(ValueError):
        make_observation_group_id([])
    with pytest.raises(ValueError):
        make_candidate_id(observation_group_id="og-example", bls_config_hash=HASH, rank=0)
    with pytest.raises(ValueError):
        make_run_fingerprint(config_hash=HASH, source_revision=None, master_seed=-1)


def test_ordered_observation_group_identity() -> None:
    first = make_observation_group_id(["obs-a", "obs-b"])
    second = make_observation_group_id(["obs-b", "obs-a"])

    assert first != second


def test_candidate_dataset_and_run_identifier_construction() -> None:
    group_id = make_observation_group_id([_observation_id()])
    candidate_id = make_candidate_id(observation_group_id=group_id, bls_config_hash=HASH, rank=1)
    dataset_id = make_dataset_version(
        catalog_snapshot_hash=HASH,
        raw_manifest_hash=HASH,
        preprocessing_config_hash=HASH,
        detection_config_hash=HASH,
        label_policy_hash=HASH,
    )
    run_id = make_run_fingerprint(config_hash=HASH, source_revision="abc123", master_seed=42)

    assert candidate_id.startswith("cand-")
    assert dataset_id.startswith("dataset-")
    assert run_id.startswith("run-")
    assert make_experiment_id(
        run_fingerprint=run_id,
        timestamp_utc=datetime(2026, 9, 9, tzinfo=UTC),
    ).startswith("exp-20260909T000000Z-")
    with pytest.raises(ValueError):
        make_experiment_id(run_fingerprint=run_id, timestamp_utc=datetime(2026, 9, 9))


def test_schema_validation_and_round_trip() -> None:
    observation_id = _observation_id()
    raw = RawProductRecord(
        object_id="TIC 123",
        observation_id=observation_id,
        sector=1,
        author="SPOC",
        cadence_seconds=120.0,
        product_id="product-1",
        checksum_sha256=CHECKSUM,
        source_uri="archive://example/product-1",
    )
    processed = ProcessedCurveRecord(
        object_id=raw.object_id,
        observation_id=raw.observation_id,
        preprocessing_config_hash=HASH,
        input_checksum_sha256=CHECKSUM,
        output_checksum_sha256="c" * 64,
    )
    group_id = make_observation_group_id([raw.observation_id])
    candidate = CandidateRecord(
        candidate_id=make_candidate_id(observation_group_id=group_id, bls_config_hash=HASH, rank=1),
        observation_group_id=group_id,
        bls_config_hash=HASH,
        rank=1,
    )
    label = LabelRecord(
        candidate_id=candidate.candidate_id,
        label="unlabeled",
        source_record_id="catalog-snapshot-placeholder",
        label_policy_hash=HASH,
    )
    split = SplitRecord(
        object_id=raw.object_id,
        dataset_version="dataset-placeholder",
        split="train",
        split_policy_hash=HASH,
    )

    assert RawProductRecord.model_validate_json(raw.model_dump_json()) == raw
    assert processed.object_id == "tic-123"
    assert label.label == "unlabeled"
    assert split.split == "train"


def test_schema_rejects_noncanonical_identity() -> None:
    with pytest.raises(ValidationError):
        RawProductRecord(
            object_id="TIC 123",
            observation_id="obs-not-canonical",
            sector=1,
            author="SPOC",
            cadence_seconds=120.0,
            product_id="product-1",
            checksum_sha256=CHECKSUM,
            source_uri="archive://example/product-1",
        )


def test_candidate_schema_rejects_noncanonical_identity() -> None:
    with pytest.raises(ValidationError):
        CandidateRecord(
            candidate_id="cand-not-canonical",
            observation_group_id="og-example",
            bls_config_hash=HASH,
            rank=1,
        )
