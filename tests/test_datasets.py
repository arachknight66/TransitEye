"""Offline B031--B034 dataset, split, lineage, and version tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from transiteye.config import ConfigurationError, DatasetSettings, load_config
from transiteye.datasets.builder import (
    BuiltDataset,
    DatasetBuildError,
    build_candidate_tables,
    build_catalog_event_recovery,
)
from transiteye.datasets.development import build_current_development_dataset
from transiteye.datasets.lineage import (
    LineageError,
    build_artifact_lineage,
    validate_frozen_candidate_ids,
)
from transiteye.datasets.schemas import (
    CandidateDatasetRecord,
    CandidateGoldLabel,
    DatasetRole,
    RecoveryState,
    assert_model_input_boundary,
)
from transiteye.datasets.splitter import SplitError, enrich_split_manifest, grouped_split
from transiteye.datasets.validation import DatasetValidationError, validate_dataset
from transiteye.datasets.versioning import (
    DatasetFreezeError,
    DatasetIdentityInputs,
    freeze_dataset,
    freeze_split_manifest,
    load_frozen_dataset,
    make_dataset_version,
)
from transiteye.identifiers import make_candidate_id

HASH_A = "a" * 20
HASH_B = "b" * 20
RAW_A = "1" * 64
RAW_B = "2" * 64


def _settings() -> DatasetSettings:
    config = load_config("configs/datasets/mvp.yaml")
    assert config.dataset is not None
    return config.dataset


def _lineage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "observation_id": "obs-1",
                "observation_group_id": "og-1",
                "mast_obs_id": "mast-1",
                "sector": 1,
                "author": "TESS-SPOC",
                "exposure_seconds": 120.0,
                "source_product_id": "mast:one",
                "product_filename": "one_lc.fits",
                "source_raw_checksum": RAW_A,
                "processed_checksum": "3" * 64,
                "processed_artifact_id": "processed-1",
                "preprocessing_config_hash": HASH_A,
                "bls_config_hash": HASH_B,
                "detection_status": "success",
            }
        ]
    )


def _candidate_rows(count: int = 8) -> pd.DataFrame:
    rows = []
    for rank in range(1, count + 1):
        rows.append(
            {
                "candidate_id": make_candidate_id(
                    observation_group_id="og-1", bls_config_hash=HASH_B, rank=rank
                ),
                "observation_group_id": "og-1",
                "rank": rank,
                "period": float(rank + 1),
                "duration": 0.1,
                "epoch": 0.2,
                "power": 1.0 / rank,
                "depth": 0.01,
                "relation_to_stronger": None if rank == 1 else "harmonic",
                "harmonic_ratio": None if rank == 1 else 2.0,
            }
        )
    return pd.DataFrame(rows)


def _match(candidate_id: str, event: str, disposition: str, kind: str) -> dict[str, object]:
    ratio = {"fundamental": 1.0, "half_period": 0.5, "double_period": 2.0}.get(kind, 1.2)
    return {
        "candidate_id": candidate_id,
        "toi_id": event,
        "object_id": "tic-1",
        "source_disposition": disposition,
        "candidate_period": ratio * 2.0,
        "catalog_period": 2.0,
        "relative_period_error": abs(ratio - 1),
        "candidate_epoch": 0.2,
        "catalog_epoch": 0.2,
        "phase_error": 0.0,
        "harmonic_ratio": ratio,
        "match_type": kind,
        "matched": True,
    }


def test_candidate_label_semantics_and_ambiguity_are_explicit() -> None:
    raw = _candidate_rows()
    ids = raw["candidate_id"].tolist()
    matches = pd.DataFrame(
        [
            _match(ids[0], "toi-cp", "CP", "fundamental"),
            _match(ids[1], "toi-kp", "KP", "half_period"),
            _match(ids[2], "toi-fp", "FP", "fundamental"),
            _match(ids[3], "toi-fa", "FA", "double_period"),
            _match(ids[4], "toi-pc", "PC", "fundamental"),
            _match(ids[6], "toi-unknown", "mystery", "fundamental"),
            _match(ids[7], "toi-conflict-a", "CP", "fundamental"),
            _match(ids[7], "toi-conflict-b", "FP", "half_period"),
        ]
    )
    candidates, relations = build_candidate_tables(
        raw, matches, _lineage(), matching_config_hash=HASH_A
    )
    assert candidates["gold_candidate_label"].tolist() == [
        "positive",
        "positive",
        "negative",
        "negative",
        "unlabeled",
        "unlabeled",
        "unlabeled",
        "ambiguous",
    ]
    assert candidates.loc[1, "match_type"] == "half_period"
    assert candidates.loc[3, "match_harmonic_ratio"] == 2.0
    assert not bool(candidates.loc[5, "gold_training_eligible"])
    assert not bool(candidates.loc[7, "gold_training_eligible"])
    assert len(relations.loc[relations["candidate_id"] == ids[7]]) == 2


def test_candidate_build_rejects_missing_lineage_and_duplicate_ids() -> None:
    with pytest.raises(DatasetBuildError, match="lineage"):
        build_candidate_tables(
            _candidate_rows(1), pd.DataFrame(), _lineage().iloc[0:0], matching_config_hash=HASH_A
        )
    duplicate = pd.concat([_candidate_rows(1), _candidate_rows(1)], ignore_index=True)
    with pytest.raises(DatasetBuildError, match="unique"):
        build_candidate_tables(duplicate, pd.DataFrame(), _lineage(), matching_config_hash=HASH_A)


def _event_inputs() -> tuple[pd.DataFrame, ...]:
    events = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "toi_id": "toi-1.01",
                "source_disposition": "CP",
                "mapped_label": "gold_positive",
                "period_days": 2.0,
                "transit_epoch_bjd": 0.2,
                "transit_duration_hours": 2.4,
            },
            {
                "object_id": "tic-1",
                "toi_id": "toi-1.02",
                "source_disposition": "FP",
                "mapped_label": "gold_negative",
                "period_days": 3.0,
                "transit_epoch_bjd": 0.3,
                "transit_duration_hours": None,
            },
            {
                "object_id": "tic-2",
                "toi_id": "toi-2.01",
                "source_disposition": "KP",
                "mapped_label": "gold_positive",
                "period_days": 4.0,
                "transit_epoch_bjd": 0.4,
                "transit_duration_hours": 3.0,
            },
            {
                "object_id": "tic-3",
                "toi_id": "toi-3.01",
                "source_disposition": "FA",
                "mapped_label": "gold_negative",
                "period_days": None,
                "transit_epoch_bjd": None,
                "transit_duration_hours": None,
            },
        ]
    )
    observations = pd.DataFrame([{"object_id": "tic-1"}, {"object_id": "tic-2"}])
    products = pd.DataFrame(
        [
            {"object_id": "tic-1", "product_filename": "one_lc.fits"},
            {"object_id": "tic-2", "product_filename": "two_lc.fits"},
        ]
    )
    receipts = pd.DataFrame([{"object_id": "tic-1"}])
    lineage = pd.concat(
        [
            _lineage(),
            _lineage().assign(
                observation_id="obs-1b",
                observation_group_id="og-1b",
                source_product_id="mast:one-b",
                source_raw_checksum=RAW_B,
                processed_artifact_id="processed-1b",
            ),
        ],
        ignore_index=True,
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": "cand-f", "candidate_rank": 2},
            {"candidate_id": "cand-h", "candidate_rank": 1},
        ]
    )
    relations = pd.DataFrame(
        [
            {
                "candidate_id": "cand-f",
                "event_id": "toi-1.01",
                "match_type": "fundamental",
                "matched": True,
            },
            {
                "candidate_id": "cand-h",
                "event_id": "toi-1.01",
                "match_type": "half_period",
                "matched": True,
            },
        ]
    )
    return events, observations, products, receipts, lineage, candidates, relations


def test_event_recovery_precedence_and_unrecovered_events_are_preserved() -> None:
    result = build_catalog_event_recovery(*_event_inputs())
    states = result.set_index("event_id")["recovery_state"].to_dict()
    assert states == {
        "toi-1.01": RecoveryState.RECOVERED_FUNDAMENTAL.value,
        "toi-1.02": RecoveryState.SEARCHED_NOT_RECOVERED.value,
        "toi-2.01": RecoveryState.PRODUCT_DISCOVERED_NOT_DOWNLOADED.value,
        "toi-3.01": RecoveryState.UNAVAILABLE_ACQUISITION.value,
    }
    first = result.set_index("event_id").loc["toi-1.01"]
    assert first["fundamental_candidate_count"] == 1
    assert first["harmonic_candidate_count"] == 1
    assert first["primary_candidate_id"] == "cand-f"
    assert first["searched_group_count"] == 2


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("downloaded", RecoveryState.DOWNLOADED_NOT_PREPROCESSED.value),
        ("preprocessed", RecoveryState.PREPROCESSED_NOT_SEARCHED.value),
        ("failed", RecoveryState.DETECTION_FAILED.value),
        ("observation_only", RecoveryState.NOT_SEARCHED.value),
    ],
)
def test_event_recovery_unavailable_stage_states(condition: str, expected: str) -> None:
    events, observations, products, receipts, lineage, candidates, relations = _event_inputs()
    one_event = events.iloc[[0]].copy()
    if condition == "downloaded":
        lineage = lineage.iloc[0:0]
    elif condition == "preprocessed":
        lineage["detection_status"] = "not_searched"
    elif condition == "failed":
        lineage["detection_status"] = "failed"
    else:
        products = products.iloc[0:0]
        receipts = receipts.iloc[0:0]
        lineage = lineage.iloc[0:0]
    result = build_catalog_event_recovery(
        one_event, observations, products, receipts, lineage, candidates, relations.iloc[0:0]
    )
    assert result.loc[0, "recovery_state"] == expected


def test_duplicate_catalog_event_is_rejected() -> None:
    inputs = list(_event_inputs())
    inputs[0] = pd.concat([inputs[0], inputs[0].iloc[[0]]], ignore_index=True)
    with pytest.raises(DatasetBuildError, match="unique"):
        build_catalog_event_recovery(*inputs)


def test_grouped_split_is_deterministic_seeded_and_never_splits_tic() -> None:
    ids = [f"tic-{index}" for index in range(1, 21)] + ["tic-1"]
    first = grouped_split(ids, settings=_settings(), master_seed=42, role=DatasetRole.FINAL)
    again = grouped_split(ids, settings=_settings(), master_seed=42, role=DatasetRole.FINAL)
    changed = grouped_split(ids, settings=_settings(), master_seed=43, role=DatasetRole.FINAL)
    pd.testing.assert_frame_equal(first, again)
    assert first["object_id"].is_unique
    assert first["split"].value_counts().to_dict() == {"train": 14, "validation": 3, "test": 3}
    assert not first[["object_id", "split"]].equals(changed[["object_id", "split"]])


def test_tiny_current_data_is_explicitly_development_only() -> None:
    split = grouped_split(
        ["tic-1", "tic-2"], settings=_settings(), master_seed=42, role=DatasetRole.DEVELOPMENT
    )
    assert set(split["split"]) == {"development"}
    assert set(split["dataset_role"]) == {"development"}
    with pytest.raises(SplitError, match="Too few"):
        grouped_split(
            ["tic-1", "tic-2"], settings=_settings(), master_seed=42, role=DatasetRole.FINAL
        )
    with pytest.raises(SplitError, match="At least"):
        grouped_split([], settings=_settings(), master_seed=42, role=DatasetRole.DEVELOPMENT)


def test_split_manifest_group_summaries_keep_mixed_truth_event_level() -> None:
    split = grouped_split(
        ["tic-1", "tic-2"], settings=_settings(), master_seed=42, role=DatasetRole.DEVELOPMENT
    )
    events = pd.DataFrame(
        [
            {"object_id": "tic-1", "internal_event_class": "gold_positive"},
            {"object_id": "tic-1", "internal_event_class": "gold_negative"},
        ]
    )
    candidates = pd.DataFrame([{"object_id": "tic-1"}])
    enriched = enrich_split_manifest(split, candidates, events).set_index("object_id")
    assert bool(enriched.loc["tic-1", "mixed_gold_event_classes"])
    assert enriched.loc["tic-1", "event_count"] == 2


def test_invalid_split_fractions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """config_version: 1
project: {name: TransitEye}
reproducibility: {master_seed: 42}
dataset:
  policy_version: v1
  label_policy_version: v1
  recovery_policy_version: v1
  ambiguity_handling: preserve_unlabeled
  development_split_behavior: all_development
  split:
    train_fraction: 0.8
    validation_fraction: 0.2
    test_fraction: 0.2
    seed_component: split
    grouped_stratification: none
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_config(path)


def _valid_dataset() -> tuple[BuiltDataset, pd.DataFrame]:
    raw = _candidate_rows(1)
    candidates, relations = build_candidate_tables(
        raw, pd.DataFrame(), _lineage(), matching_config_hash=HASH_A
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "toi-1.01",
                "object_id": "tic-1",
                "source_disposition": "CP",
                "internal_event_class": "gold_positive",
                "catalog_period": 2.0,
                "catalog_epoch": 0.2,
                "catalog_duration_hours": 2.0,
                "acquisition_observation_count": 1,
                "discovered_product_count": 1,
                "downloaded_product_count": 1,
                "preprocessed_group_count": 1,
                "searched_group_count": 1,
                "matching_candidate_count": 0,
                "fundamental_candidate_count": 0,
                "harmonic_candidate_count": 0,
                "primary_candidate_id": None,
                "recovery_state": "searched_not_recovered",
                "contributing_observation_group_ids": '["og-1"]',
            }
        ]
    )
    dataset = BuiltDataset(candidates, relations, events, _lineage())
    split = grouped_split(
        ["tic-1"], settings=_settings(), master_seed=42, role=DatasetRole.DEVELOPMENT
    )
    return dataset, split


def test_valid_dataset_passes_and_reports_qa() -> None:
    dataset, split = _valid_dataset()
    qa = validate_dataset(dataset, split)
    assert qa["candidates"]["by_label"] == {"unlabeled": 1}
    assert qa["catalog_events"]["total"] == 1
    assert qa["splits"]["zero_tic_overlap"]
    record = CandidateDatasetRecord.model_validate(dataset.candidates.iloc[0].to_dict())
    assert CandidateDatasetRecord.model_validate(record.model_dump()) == record


def _add_second_split(split: pd.DataFrame) -> pd.DataFrame:
    second = split.iloc[[0]].copy()
    second["object_id"] = "tic-2"
    second["split"] = "test"
    return pd.concat([split, second], ignore_index=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("tic_split", "TIC"),
        ("group", "Observation group"),
        ("observation", "Observation/product"),
        ("mast_observation", "MAST observation"),
        ("product", "Source product"),
        ("checksum", "Raw checksum"),
        ("processed", "Processed artifact"),
        ("duplicate_candidate", "Duplicate candidate"),
        ("duplicate_event", "Duplicate catalog event"),
        ("candidate_tic", "Candidate/TIC"),
        ("candidate_split", "Candidate split"),
        ("event_split", "Event split"),
        ("duplicate_rank", "candidate rank"),
    ],
)
def test_adversarial_duplicate_and_leakage_fixtures_fail(mutation: str, message: str) -> None:
    dataset, split = _valid_dataset()
    candidates = dataset.candidates.copy()
    events = dataset.catalog_events.copy()
    lineage = dataset.artifact_lineage.copy()
    if mutation == "tic_split":
        duplicate = split.iloc[[0]].copy()
        duplicate["split"] = "test"
        split = pd.concat([split, duplicate], ignore_index=True)
    elif mutation in {
        "group",
        "observation",
        "mast_observation",
        "product",
        "checksum",
        "processed",
    }:
        split = _add_second_split(split)
        second = lineage.iloc[[0]].copy()
        second["object_id"] = "tic-2"
        if mutation != "group":
            second["observation_group_id"] = "og-2"
        if mutation != "observation":
            second["observation_id"] = "obs-2"
        if mutation != "mast_observation":
            second["mast_obs_id"] = "mast-2"
        if mutation != "product":
            second["source_product_id"] = "mast:two"
        if mutation != "checksum":
            second["source_raw_checksum"] = RAW_B
        if mutation != "processed":
            second["processed_artifact_id"] = "processed-2"
        lineage = pd.concat([lineage, second], ignore_index=True)
    elif mutation == "duplicate_candidate":
        candidates = pd.concat([candidates, candidates], ignore_index=True)
    elif mutation == "duplicate_event":
        events = pd.concat([events, events], ignore_index=True)
    elif mutation == "candidate_tic":
        split = _add_second_split(split)
        candidates.loc[0, "object_id"] = "tic-2"
    elif mutation == "candidate_split":
        candidates["split"] = "test"
    elif mutation == "event_split":
        events["split"] = "test"
    elif mutation == "duplicate_rank":
        second = candidates.iloc[[0]].copy()
        second["candidate_id"] = "cand-other"
        candidates = pd.concat([candidates, second], ignore_index=True)
    broken = BuiltDataset(candidates, dataset.candidate_event_matches, events, lineage)
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(broken, split)


def test_cross_tic_and_inconsistent_relation_lineage_are_rejected() -> None:
    dataset, split = _valid_dataset()
    split = _add_second_split(split)
    second_event = dataset.catalog_events.iloc[[0]].copy()
    second_event["event_id"] = "toi-2.01"
    second_event["object_id"] = "tic-2"
    events = pd.concat([dataset.catalog_events, second_event], ignore_index=True)
    relation = pd.DataFrame(
        [
            {
                "candidate_id": dataset.candidates.loc[0, "candidate_id"],
                "candidate_object_id": "tic-1",
                "event_id": "toi-2.01",
                "event_object_id": "tic-2",
                "source_disposition": "CP",
                "matched": True,
            }
        ]
    )
    broken = BuiltDataset(dataset.candidates, relation, events, dataset.artifact_lineage)
    with pytest.raises(DatasetValidationError, match="different TICs"):
        validate_dataset(broken, split)
    relation.loc[0, "event_object_id"] = "tic-1"
    broken = BuiltDataset(dataset.candidates, relation, events, dataset.artifact_lineage)
    with pytest.raises(DatasetValidationError, match="Event relation"):
        validate_dataset(broken, split)


def test_invalid_label_semantics_and_ambiguous_training_state_are_rejected() -> None:
    dataset, split = _valid_dataset()
    invalid = dataset.candidates.copy()
    invalid.loc[0, "gold_candidate_label"] = CandidateGoldLabel.NEGATIVE.value
    invalid.loc[0, "gold_training_eligible"] = True
    with pytest.raises(DatasetValidationError, match="gold label"):
        validate_dataset(
            BuiltDataset(
                invalid,
                dataset.candidate_event_matches,
                dataset.catalog_events,
                dataset.artifact_lineage,
            ),
            split,
        )

    raw = _candidate_rows(1)
    candidate_id = str(raw.loc[0, "candidate_id"])
    matched_candidates, matched_relations = build_candidate_tables(
        raw,
        pd.DataFrame(
            [
                _match(candidate_id, "toi-1.01", "CP", "fundamental"),
                _match(candidate_id, "toi-conflict", "FP", "half_period"),
            ]
        ),
        _lineage(),
        matching_config_hash=HASH_A,
    )
    conflict_event = dataset.catalog_events.iloc[[0]].copy()
    conflict_event["event_id"] = "toi-conflict"
    conflict_event["source_disposition"] = "FP"
    conflict_event["internal_event_class"] = "gold_negative"
    conflict_events = pd.concat([dataset.catalog_events, conflict_event], ignore_index=True)
    matched_candidates.loc[0, "gold_training_eligible"] = True
    with pytest.raises(DatasetValidationError, match="training eligibility"):
        validate_dataset(
            BuiltDataset(
                matched_candidates,
                matched_relations,
                conflict_events,
                dataset.artifact_lineage,
            ),
            split,
        )
    invalid.loc[0, "gold_candidate_label"] = CandidateGoldLabel.AMBIGUOUS.value
    with pytest.raises(DatasetValidationError, match="gold label"):
        validate_dataset(
            BuiltDataset(
                invalid,
                dataset.candidate_event_matches,
                dataset.catalog_events,
                dataset.artifact_lineage,
            ),
            split,
        )


def test_catalog_ephemeris_feature_boundary() -> None:
    assert_model_input_boundary(["candidate_period", "bls_power"])
    with pytest.raises(ValueError, match="Catalog evaluation"):
        assert_model_input_boundary(["candidate_period", "catalog_period"])


def test_lineage_construction_and_candidate_identity() -> None:
    products = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "mast_obs_id": "10",
                "sector": 1,
                "author": "TESS-SPOC",
                "exposure_seconds": 120.0,
                "product_filename": "one_lc.fits",
                "data_uri": "mast:one",
            }
        ]
    )
    receipts = pd.DataFrame([{"object_id": "tic-1", "data_uri": "mast:one", "sha256": RAW_A}])
    processed = pd.DataFrame(
        [
            {
                "source_raw_checksum": RAW_A,
                "processed_checksum": "3" * 64,
                "preprocessing_config_hash": HASH_A,
            }
        ]
    )
    detections = pd.DataFrame(
        [
            {
                "source_raw_checksum": RAW_A,
                "observation_group_id": "og-1",
                "bls_config_hash": HASH_B,
                "status": "success",
            }
        ]
    )
    lineage = build_artifact_lineage(products, receipts, processed, detections)
    assert lineage.loc[0, "observation_id"].startswith("obs-")
    validate_frozen_candidate_ids(_candidate_rows(1), lineage)
    broken = _candidate_rows(1).assign(candidate_id="cand-bad")
    with pytest.raises(LineageError, match="candidate ID"):
        validate_frozen_candidate_ids(broken, lineage)
    with pytest.raises(LineageError, match="missing lineage"):
        build_artifact_lineage(products.drop(columns="sector"), receipts, processed, detections)


def _identity() -> DatasetIdentityInputs:
    return DatasetIdentityInputs(
        catalog_snapshot_id="toi-one",
        catalog_snapshot_hash=HASH_A,
        label_policy_hash=HASH_A,
        acquisition_snapshot_id="mast-one",
        acquisition_snapshot_hash=HASH_A,
        raw_checksums=(RAW_B, RAW_A),
        preprocessing_config_hash=HASH_A,
        bls_config_hash=HASH_A,
        matching_config_hash=HASH_A,
        dataset_policy_hash=HASH_A,
    )


@pytest.mark.parametrize(
    "field",
    [
        "catalog_snapshot_id",
        "catalog_snapshot_hash",
        "label_policy_hash",
        "acquisition_snapshot_id",
        "acquisition_snapshot_hash",
        "raw_checksums",
        "preprocessing_config_hash",
        "bls_config_hash",
        "matching_config_hash",
        "dataset_policy_hash",
    ],
)
def test_dataset_version_changes_for_every_scientific_input(field: str) -> None:
    original = _identity()
    changed_value: Any = (RAW_A,) if field == "raw_checksums" else HASH_B
    changed = replace(original, **{field: changed_value})
    assert make_dataset_version(original) != make_dataset_version(changed)


def test_dataset_version_is_order_path_and_timestamp_independent() -> None:
    identity = _identity()
    reordered = replace(identity, raw_checksums=tuple(reversed(identity.raw_checksums)))
    assert make_dataset_version(identity) == make_dataset_version(reordered)
    assert "/tmp/one" not in identity.portable_dict()


def test_dataset_and_split_freeze_round_trip_and_integrity(tmp_path: Path) -> None:
    dataset, split = _valid_dataset()
    summary = validate_dataset(dataset, split)
    first = freeze_dataset(
        dataset,
        identity=_identity(),
        validation_summary=summary,
        root=tmp_path / "one",
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    loaded = load_frozen_dataset(first, expected_version=make_dataset_version(_identity()))
    assert loaded.candidates.loc[0, "candidate_id"] == dataset.candidates.loc[0, "candidate_id"]
    assert (
        freeze_dataset(
            dataset,
            identity=_identity(),
            validation_summary=summary,
            root=tmp_path / "one",
            created_at_utc=datetime(2030, 1, 1, tzinfo=UTC),
        )
        == first
    )
    conflicting = BuiltDataset(
        dataset.candidates.assign(bls_power=99.0),
        dataset.candidate_event_matches,
        dataset.catalog_events,
        dataset.artifact_lineage,
    )
    with pytest.raises(DatasetFreezeError, match="conflicting"):
        freeze_dataset(
            conflicting,
            identity=_identity(),
            validation_summary=summary,
            root=tmp_path / "one",
        )
    split_dir = freeze_split_manifest(
        split, dataset_version=make_dataset_version(_identity()), root=tmp_path / "splits"
    )
    assert (
        freeze_split_manifest(
            split, dataset_version=make_dataset_version(_identity()), root=tmp_path / "splits"
        )
        == split_dir
    )
    (first / "candidates.parquet").write_bytes(b"corrupt")
    with pytest.raises(DatasetFreezeError, match="checksum"):
        load_frozen_dataset(first)


def test_current_development_dataset_replay_uses_only_frozen_local_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/manifests/mast-products-b011-resumable/metadata.json").is_file():
        pytest.skip("Frozen real development artifacts are not present in this checkout.")
    first = build_current_development_dataset(root)
    second = build_current_development_dataset(root)
    assert first == second
    assert first[2] == "dataset-7848048e67497fdace7d"
