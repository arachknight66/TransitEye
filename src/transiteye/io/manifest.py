"""Versioned, checksum-aware product manifests for reproducible acquisition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One archive product required by a frozen analysis cohort."""

    product_id: str
    data_uri: str
    target_id: str
    sector: int | None
    product_type: str
    expected_sha256: str | None = None
    local_filename: str | None = None

    def resolved_filename(self) -> str:
        """Return a stable local file name without trusting a remote path."""

        return self.local_filename or f"{self.product_id}.fits"


@dataclass(frozen=True, slots=True)
class ProductManifest:
    """An immutable collection of archive products and their discovery provenance."""

    manifest_id: str
    created_at: str
    source: str
    entries: tuple[ManifestEntry, ...]
    query: dict[str, str] = field(default_factory=dict)
    schema_version: str = "1"

    @classmethod
    def create(
        cls,
        source: str,
        entries: tuple[ManifestEntry, ...],
        query: dict[str, str] | None = None,
    ) -> ProductManifest:
        """Create a deterministic identifier from sorted entry content and query metadata."""

        payload = {
            "source": source,
            "entries": [
                entry_to_dict(entry) for entry in sorted(entries, key=lambda item: item.product_id)
            ],
            "query": query or {},
            "schema_version": "1",
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        manifest_id = hashlib.sha256(canonical).hexdigest()[:16]
        return cls(
            manifest_id=manifest_id,
            created_at=datetime.now(UTC).isoformat(),
            source=source,
            entries=entries,
            query=query or {},
        )


def entry_to_dict(entry: ManifestEntry) -> dict[str, Any]:
    return {
        "product_id": entry.product_id,
        "data_uri": entry.data_uri,
        "target_id": entry.target_id,
        "sector": entry.sector,
        "product_type": entry.product_type,
        "expected_sha256": entry.expected_sha256,
        "local_filename": entry.local_filename,
    }


def manifest_to_dict(manifest: ProductManifest) -> dict[str, Any]:
    return {
        "manifest_id": manifest.manifest_id,
        "created_at": manifest.created_at,
        "source": manifest.source,
        "entries": [entry_to_dict(entry) for entry in manifest.entries],
        "query": manifest.query,
        "schema_version": manifest.schema_version,
    }


def write_manifest(manifest: ProductManifest, path: Path) -> None:
    """Write one explicit manifest file; callers should treat it as immutable afterward."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> ProductManifest:
    """Read and validate a JSON manifest created by this module."""

    document = json.loads(path.read_text(encoding="utf-8"))
    required = {"manifest_id", "created_at", "source", "entries", "schema_version"}
    missing = required.difference(document)
    if missing:
        raise ValueError(f"Manifest is missing required fields: {', '.join(sorted(missing))}")
    entries = tuple(ManifestEntry(**entry) for entry in document["entries"])
    product_ids = [entry.product_id for entry in entries]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("Manifest product_id values must be unique.")
    return ProductManifest(
        manifest_id=document["manifest_id"],
        created_at=document["created_at"],
        source=document["source"],
        entries=entries,
        query=document.get("query", {}),
        schema_version=document["schema_version"],
    )
