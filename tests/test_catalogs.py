from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from transiteye.catalogs.cohort import (
    CohortSelectionError,
    build_catalog_qa,
    build_pilot_cohort,
    write_pilot_cohort_manifest,
)
from transiteye.catalogs.exoplanet_archive import (
    TAP_SYNC_ENDPOINT,
    TOI_FIELDS,
    CatalogResponse,
    CatalogResponseError,
    build_toi_query,
    fetch_toi_catalog,
    normalize_toi_response,
    selected_toi_fields,
)
from transiteye.catalogs.labels import (
    CatalogDisposition,
    InternalLabel,
    UnknownDispositionError,
    map_disposition,
)
from transiteye.catalogs.snapshot import (
    SnapshotImmutableError,
    load_normalized_snapshot,
    sha256_bytes,
    write_snapshot,
)
from transiteye.config import PilotCohortSettings, config_hash, load_config

HEADERS = ",".join(TOI_FIELDS)


def _row(
    *,
    tid: str,
    toi: str,
    disposition: str,
    period: str = "3.2",
    epoch: str = "2459000.1",
    duration: str = "2.5",
    depth: str = "500",
) -> str:
    return ",".join(
        [tid, toi, disposition, period, epoch, duration, depth, "2024-01-01", "2024-02-01"]
    )


def _raw_response() -> bytes:
    return (
        HEADERS
        + "\n"
        + _row(tid="100", toi="100.01", disposition="CP")
        + "\n"
        + _row(tid="100", toi="100.02", disposition="PC", period="")
        + "\n"
        + _row(tid="200", toi="200.01", disposition="KP")
        + "\n"
        + _row(tid="300", toi="300.01", disposition="FP")
        + "\n"
        + _row(tid="400", toi="400.01", disposition="FA")
        + "\n"
        + _row(tid="500", toi="500.01", disposition="APC", epoch="")
        + "\n"
        + _row(tid="", toi="600.01", disposition="CP")
        + "\n"
        + _row(tid="700", toi="700.01", disposition="UNEXPECTED")
        + "\n"
        + _row(tid="300", toi="300.01", disposition="FP")
        + "\n"
    ).encode()


def _response(raw_bytes: bytes | None = None) -> CatalogResponse:
    return CatalogResponse(
        endpoint=TAP_SYNC_ENDPOINT,
        request_url="https://example.invalid/TAP/sync?query=test",
        query=build_toi_query(),
        retrieved_at_utc=datetime(2026, 9, 9, tzinfo=UTC),
        raw_bytes=raw_bytes or _raw_response(),
    )


def test_tap_query_selects_only_stable_documented_fields() -> None:
    assert build_toi_query() == f"SELECT {','.join(TOI_FIELDS)} FROM toi ORDER BY toi"
    assert selected_toi_fields(tuple(reversed(TOI_FIELDS[:3]))) == TOI_FIELDS[:3]
    assert "*" not in build_toi_query()


def test_tap_query_rejects_unknown_or_duplicate_fields() -> None:
    with pytest.raises(ValueError):
        build_toi_query(["tid", "invalid"])
    with pytest.raises(ValueError):
        build_toi_query(["tid", "tid"])


def test_fetch_supports_offline_mocked_response() -> None:
    seen: list[str] = []

    def fake_fetcher(url: str, timeout: float) -> bytes:
        seen.append(url)
        assert timeout == 5.0
        return _raw_response()

    response = fetch_toi_catalog(
        timeout_seconds=5.0,
        fetcher=fake_fetcher,
        retrieved_at_utc=datetime(2026, 9, 9, tzinfo=UTC),
    )

    assert response.raw_bytes == _raw_response()
    assert response.query == build_toi_query()
    assert "format=csv" in seen[0]


def test_normalization_preserves_source_values_and_missing_ephemerides() -> None:
    frame = normalize_toi_response(_raw_response())

    assert len(frame) == 9
    assert frame.loc[0, "object_id"] == "tic-100"
    assert pd.isna(frame.loc[1, "period_days"])
    assert pd.isna(frame.loc[5, "transit_epoch_bjd"])
    assert frame.loc[6, "object_id"] is None
    assert frame.loc[8, "duplicate_toi"]
    assert frame.loc[0, "source_toi"] == "100.01"


def test_normalization_rejects_incomplete_or_malformed_response() -> None:
    with pytest.raises(CatalogResponseError):
        normalize_toi_response(b"tid,toi\n100,100.01\n")
    malformed_numeric = _raw_response().replace(b"3.2", b"not-a-number", 1)
    with pytest.raises(CatalogResponseError):
        normalize_toi_response(malformed_numeric)


def test_disposition_mapping_preserves_subgroups_and_surfaces_unknowns() -> None:
    assert map_disposition("FP").internal_label is InternalLabel.GOLD_NEGATIVE
    assert map_disposition("FA").subgroup == "fa"
    assert map_disposition("PC").internal_label is InternalLabel.UNLABELED_SECONDARY
    unknown = map_disposition("unexpected")
    assert unknown.disposition is CatalogDisposition.UNKNOWN
    assert unknown.internal_label is InternalLabel.UNKNOWN
    with pytest.raises(UnknownDispositionError):
        map_disposition(None, strict=True)


def test_snapshot_is_immutable_and_checksums_are_recorded(tmp_path: Path) -> None:
    response = _response()
    frame = normalize_toi_response(response.raw_bytes)
    first = write_snapshot(response, frame, root=tmp_path)
    second = write_snapshot(response, frame, root=tmp_path)

    assert not first.reused_existing
    assert second.reused_existing
    assert first.record.raw_response_sha256 == sha256_bytes(response.raw_bytes)
    assert (first.directory / "raw_response.csv").read_bytes() == response.raw_bytes
    assert first.record.normalized_table_sha256
    loaded_record, loaded_frame = load_normalized_snapshot(first.directory)
    assert loaded_record == first.record
    assert len(loaded_frame) == len(frame)

    conflicting = _response(raw_bytes=_raw_response().replace(b"2024-01-01", b"2024-03-01", 1))
    conflict_id = first.directory.parent / (
        f"toi-20260909T000000Z-{sha256_bytes(conflicting.raw_bytes)[:20]}"
    )
    conflict_id.mkdir()
    with pytest.raises(SnapshotImmutableError):
        write_snapshot(conflicting, normalize_toi_response(conflicting.raw_bytes), root=tmp_path)


def test_catalog_qa_and_pilot_preserve_multi_event_lineage() -> None:
    events = normalize_toi_response(_raw_response())
    settings = PilotCohortSettings(
        positive_target_count=2,
        negative_target_count=2,
        seed_component="test_pilot",
    )
    cohort = build_pilot_cohort(events, settings=settings, master_seed=42)

    assert set(cohort.targets["selection_class"]) == {"positive", "negative"}
    assert set(cohort.gold_events["mapped_label"]) == {
        InternalLabel.GOLD_POSITIVE.value,
        InternalLabel.GOLD_NEGATIVE.value,
    }
    assert not cohort.gold_events["mapped_label"].eq(InternalLabel.UNLABELED_SECONDARY.value).any()
    assert not cohort.gold_events["mapped_label"].eq(InternalLabel.UNKNOWN.value).any()
    assert not cohort.gold_events["toi_id"].duplicated().any()
    assert "tic-100" in set(cohort.selected_target_events["object_id"])
    tic_100_events = cohort.selected_target_events.loc[
        cohort.selected_target_events["object_id"] == "tic-100"
    ]
    assert len(tic_100_events) == 2
    qa = build_catalog_qa(events, cohort)
    assert qa["total_catalog_rows"] == len(events)
    assert qa["tics_with_multiple_tois"] >= 1
    assert qa["unknown_disposition_count"] == 1
    assert qa["missing_tic_count"] == 1
    assert qa["duplicate_toi_count"] == 2


def test_pilot_sampling_is_deterministic_and_seed_sensitive() -> None:
    rows = [HEADERS]
    for index in range(20):
        disposition = "CP" if index < 10 else "FP"
        rows.append(_row(tid=str(index + 1), toi=f"{index + 1}.01", disposition=disposition))
    events = normalize_toi_response(("\n".join(rows) + "\n").encode())
    settings = PilotCohortSettings(
        positive_target_count=4,
        negative_target_count=4,
        seed_component="sampling_test",
    )
    first = build_pilot_cohort(events, settings=settings, master_seed=42)
    second = build_pilot_cohort(events, settings=settings, master_seed=42)
    changed = build_pilot_cohort(events, settings=settings, master_seed=43)

    assert first.targets.to_dict("records") == second.targets.to_dict("records")
    assert first.targets["object_id"].tolist() != changed.targets["object_id"].tolist()


def test_cohort_rejects_unavailable_requested_target_count() -> None:
    events = normalize_toi_response(_raw_response())
    settings = PilotCohortSettings(
        positive_target_count=99,
        negative_target_count=1,
        seed_component="too_many",
    )
    with pytest.raises(CohortSelectionError):
        build_pilot_cohort(events, settings=settings, master_seed=42)


def test_pilot_manifest_is_reproducible_and_immutable(tmp_path: Path) -> None:
    events = normalize_toi_response(_raw_response())
    settings = PilotCohortSettings(
        positive_target_count=2,
        negative_target_count=2,
        seed_component="manifest_test",
    )
    cohort = build_pilot_cohort(events, settings=settings, master_seed=42)
    config = load_config("configs/catalog/toi_gold.yaml")
    manifest = write_pilot_cohort_manifest(
        cohort,
        snapshot_id="toi-test",
        cohort_config_hash=config_hash(config),
        master_seed=42,
        root=tmp_path,
    )

    assert (
        write_pilot_cohort_manifest(
            cohort,
            snapshot_id="toi-test",
            cohort_config_hash=config_hash(config),
            master_seed=42,
            root=tmp_path,
        )
        == manifest
    )
    assert pd.read_parquet(manifest / "targets.parquet", engine="pyarrow").equals(cohort.targets)
