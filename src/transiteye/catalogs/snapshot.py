"""Immutable storage for exact catalog responses and normalized tables."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from transiteye import __version__
from transiteye.catalogs.exoplanet_archive import NORMALIZATION_SCHEMA_VERSION, CatalogResponse
from transiteye.identifiers import validate_sha256
from transiteye.serialization import canonical_json


class SnapshotImmutableError(FileExistsError):
    """Raised when a requested snapshot write would overwrite immutable data."""


class SnapshotIntegrityError(ValueError):
    """Raised when a stored normalized table no longer matches its metadata."""


class CatalogSnapshotRecord(BaseModel):
    """Portable metadata describing one frozen catalog response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    request_url: str = Field(min_length=1)
    query: str = Field(min_length=1)
    retrieved_at_utc: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    raw_response_sha256: str
    normalized_table_sha256: str
    normalization_schema_version: str = Field(min_length=1)
    project_version: str = Field(min_length=1)

    @field_validator("raw_response_sha256", "normalized_table_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        return validate_sha256(value)


@dataclass(frozen=True)
class SnapshotWriteResult:
    """A newly written or detected-identical catalog snapshot."""

    record: CatalogSnapshotRecord
    directory: Path
    reused_existing: bool


def sha256_bytes(value: bytes) -> str:
    """Return a full SHA-256 digest for raw or derived file content."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a full SHA-256 digest without relying on local path identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_id(response: CatalogResponse) -> str:
    timestamp = response.retrieved_at_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"toi-{timestamp}-{sha256_bytes(response.raw_bytes)[:20]}"


def _metadata_path(directory: Path) -> Path:
    return directory / "metadata.json"


def _load_metadata(path: Path) -> CatalogSnapshotRecord:
    return CatalogSnapshotRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _find_identical_snapshot(
    root: Path, *, endpoint: str, query: str, raw_response_sha256: str
) -> tuple[CatalogSnapshotRecord, Path] | None:
    if not root.exists():
        return None
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            record = _load_metadata(metadata_path)
        except (OSError, ValueError):
            continue
        if (
            record.endpoint == endpoint
            and record.query == query
            and record.raw_response_sha256 == raw_response_sha256
        ):
            return record, metadata_path.parent
    return None


def _write_metadata(path: Path, record: CatalogSnapshotRecord) -> None:
    path.write_text(canonical_json(record.model_dump(mode="json")) + "\n", encoding="utf-8")


def write_snapshot(
    response: CatalogResponse,
    normalized_table: pd.DataFrame,
    *,
    root: str | Path = "data/catalogs",
) -> SnapshotWriteResult:
    """Write an immutable source CSV, normalized Parquet table, and metadata JSON."""
    snapshot_root = Path(root)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    raw_sha256 = sha256_bytes(response.raw_bytes)
    identical = _find_identical_snapshot(
        snapshot_root,
        endpoint=response.endpoint,
        query=response.query,
        raw_response_sha256=raw_sha256,
    )
    if identical is not None:
        record, directory = identical
        return SnapshotWriteResult(record=record, directory=directory, reused_existing=True)

    snapshot_id = _snapshot_id(response)
    final_directory = snapshot_root / snapshot_id
    if final_directory.exists():
        raise SnapshotImmutableError(f"Snapshot directory already exists: {snapshot_id}")

    temporary_directory = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=snapshot_root))
    try:
        raw_path = temporary_directory / "raw_response.csv"
        normalized_path = temporary_directory / "normalized.parquet"
        raw_path.write_bytes(response.raw_bytes)
        normalized_table.to_parquet(normalized_path, engine="pyarrow", index=False)
        record = CatalogSnapshotRecord(
            snapshot_id=snapshot_id,
            source_name="NASA Exoplanet Archive TOI TAP",
            endpoint=response.endpoint,
            request_url=response.request_url,
            query=response.query,
            retrieved_at_utc=response.retrieved_at_utc.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            row_count=len(normalized_table),
            raw_response_sha256=sha256_file(raw_path),
            normalized_table_sha256=sha256_file(normalized_path),
            normalization_schema_version=NORMALIZATION_SCHEMA_VERSION,
            project_version=__version__,
        )
        _write_metadata(_metadata_path(temporary_directory), record)
        os.replace(temporary_directory, final_directory)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return SnapshotWriteResult(record=record, directory=final_directory, reused_existing=False)


def load_normalized_snapshot(directory: str | Path) -> tuple[CatalogSnapshotRecord, pd.DataFrame]:
    """Load and verify a stored snapshot's derived Parquet checksum."""
    snapshot_directory = Path(directory)
    record = _load_metadata(_metadata_path(snapshot_directory))
    normalized_path = snapshot_directory / "normalized.parquet"
    if sha256_file(normalized_path) != record.normalized_table_sha256:
        raise SnapshotIntegrityError("Normalized snapshot checksum does not match metadata.")
    return record, pd.read_parquet(normalized_path, engine="pyarrow")
