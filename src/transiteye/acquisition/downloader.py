"""Sequential, checksum-verified immutable raw-product downloads."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from transiteye.acquisition.checksums import sha256_file


class DownloadIntegrityError(ValueError):
    """Raised when a raw product cannot be verified without unsafe mutation."""


class ProductFetcher(Protocol):
    """A downloader that writes exactly to the supplied temporary destination."""

    def fetch(self, data_uri: str, destination: Path) -> None: ...


class AstroqueryProductFetcher:
    """Astroquery's MAST downloader, constrained to a caller-owned partial path."""

    def fetch(self, data_uri: str, destination: Path) -> None:
        from astroquery.mast import Observations  # type: ignore[import-untyped]

        Observations.download_file(data_uri, local_path=str(destination), cache=False)


@dataclass(frozen=True)
class DownloadReceipt:
    """Portable completion record for a raw file, independent of its local absolute path."""

    object_id: str
    sector: int | None
    product_filename: str
    data_uri: str
    relative_raw_path: str
    expected_size_bytes: int | None
    actual_size_bytes: int
    sha256: str
    downloaded_at_utc: str
    status: str


def raw_destination(record: pd.Series, *, raw_root: str | Path = "data/raw") -> Path:
    """Return deterministic final raw location; the manifest remains authoritative."""
    sector = record.get("sector")
    sector_name = f"sector-{int(sector):04d}" if pd.notna(sector) else "sector-unknown"
    filename = Path(str(record["product_filename"])).name
    if filename != str(record["product_filename"]):
        raise DownloadIntegrityError("Product filename must not contain path components.")
    return Path(raw_root) / "tess" / str(record["object_id"]) / sector_name / filename


def _verify(
    path: Path, expected_size: int | None, expected_checksum: str | None
) -> tuple[int, str]:
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise DownloadIntegrityError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {size}."
        )
    checksum = sha256_file(path)
    if expected_checksum is not None and checksum != expected_checksum:
        raise DownloadIntegrityError(f"Checksum mismatch for {path.name}.")
    return size, checksum


def download_product(
    record: pd.Series,
    *,
    fetcher: ProductFetcher,
    raw_root: str | Path = "data/raw",
    known_checksum: str | None = None,
) -> DownloadReceipt:
    """Download one approved product via a partial file then atomically finalize it."""
    required = {"object_id", "product_filename", "data_uri"}
    missing = sorted(key for key in required if key not in record.index or pd.isna(record[key]))
    if missing:
        raise DownloadIntegrityError(f"Approved product is missing fields: {missing}")
    destination = raw_destination(record, raw_root=raw_root)
    expected_size = record.get("product_size_bytes")
    expected = int(expected_size) if expected_size is not None and pd.notna(expected_size) else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual_size, checksum = _verify(destination, expected, known_checksum)
        status = "reused"
    else:
        partial = destination.with_name(f".{destination.name}.part")
        if partial.exists():
            partial.unlink()
        fetcher.fetch(str(record["data_uri"]), partial)
        if not partial.exists():
            raise DownloadIntegrityError("Fetcher did not create the requested partial file.")
        actual_size, checksum = _verify(partial, expected, known_checksum)
        os.replace(partial, destination)
        status = "downloaded"
    return DownloadReceipt(
        object_id=str(record["object_id"]),
        sector=int(record["sector"]) if pd.notna(record.get("sector")) else None,
        product_filename=str(record["product_filename"]),
        data_uri=str(record["data_uri"]),
        relative_raw_path=str(destination.relative_to(Path(raw_root))),
        expected_size_bytes=expected,
        actual_size_bytes=actual_size,
        sha256=checksum,
        downloaded_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        status=status,
    )


def download_products(
    approved_products: pd.DataFrame,
    *,
    fetcher: ProductFetcher | None = None,
    raw_root: str | Path = "data/raw",
    known_checksums: dict[str, str] | None = None,
) -> list[DownloadReceipt]:
    """Acquire only rows in an already approved selected-product manifest."""
    if "download_selection_reason" not in approved_products.columns:
        raise DownloadIntegrityError(
            "Downloads require an explicit approved-pilot selection manifest."
        )
    active_fetcher = fetcher or AstroqueryProductFetcher()
    checksums = known_checksums or {}
    return [
        download_product(
            row,
            fetcher=active_fetcher,
            raw_root=raw_root,
            known_checksum=checksums.get(str(row["data_uri"])),
        )
        for _, row in approved_products.iterrows()
    ]


def receipts_frame(receipts: Iterable[DownloadReceipt]) -> pd.DataFrame:
    """Serialize receipt records without including local absolute paths."""
    return pd.DataFrame([receipt.__dict__ for receipt in receipts])
