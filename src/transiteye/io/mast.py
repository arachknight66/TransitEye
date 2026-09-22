"""MAST discovery and resumable archive downloads behind testable protocols."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from astroquery.mast import Catalogs, Observations

from transiteye.config import AcquisitionConfig
from transiteye.io.manifest import ManifestEntry, ProductManifest

LOGGER = logging.getLogger("transiteye.io.mast")


class ArchiveClient(Protocol):
    """Boundary around MAST that allows offline deterministic acquisition tests."""

    def discover(self, target_id: str, sector: int | None = None) -> tuple[ManifestEntry, ...]: ...

    def download(self, data_uri: str, destination: Path) -> None: ...


@dataclass(slots=True)
class AstroqueryMastClient:
    """MAST client for TESS delivered light-curve products."""

    def discover(self, target_id: str, sector: int | None = None) -> tuple[ManifestEntry, ...]:
        catalog_rows = Catalogs.query_criteria(catalog="TIC", ID=int(target_id))
        if len(catalog_rows) == 0:
            return ()
        target = catalog_rows[0]
        coordinates = f"{target['ra']} {target['dec']}"
        observations = Observations.query_criteria(
            obs_collection="TESS",
            coordinates=coordinates,
            radius="0.0001 deg",
        )
        if len(observations) == 0:
            return ()
        products = Observations.get_product_list(observations)
        entries: list[ManifestEntry] = []
        for product in products:
            filename = str(product["productFilename"])
            data_uri = str(product["dataURI"])
            if not filename.endswith("_lc.fits"):
                continue
            product_sector = _sector_from_filename(filename)
            if sector is not None and product_sector != sector:
                continue
            entries.append(
                ManifestEntry(
                    product_id=data_uri.replace("/", "_"),
                    data_uri=data_uri,
                    target_id=str(target_id),
                    sector=product_sector,
                    product_type="tess_light_curve",
                    local_filename=filename,
                )
            )
        return tuple(sorted(entries, key=lambda entry: (entry.sector or -1, entry.product_id)))

    def download(self, data_uri: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Observations.download_file(data_uri, local_path=str(destination), cache=False)


def _sector_from_filename(filename: str) -> int | None:
    marker = "-s"
    if marker not in filename:
        return None
    value = filename.split(marker, maxsplit=1)[1][:4]
    return int(value) if value.isdigit() else None


def build_manifest(
    client: ArchiveClient,
    target_id: str,
    sector: int | None = None,
) -> ProductManifest:
    """Discover a target's products and freeze the exact discovery result."""

    entries = client.discover(target_id, sector)
    return ProductManifest.create(
        source="mast",
        entries=entries,
        query={"target_id": target_id, "sector": "" if sector is None else str(sector)},
    )


def sha256sum(path: Path) -> str:
    """Return the content checksum used to validate completed downloads."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_manifest(
    client: ArchiveClient,
    manifest: ProductManifest,
    destination_root: Path,
    config: AcquisitionConfig,
) -> dict[str, Path]:
    """Download a frozen manifest with retries and content-addressable cache validation.

    This deliberately performs one download at a time in Phase 1. The later sector executor owns
    bounded parallelism; this function remains deterministic and independently testable.
    """

    completed: dict[str, Path] = {}
    for entry in manifest.entries:
        destination = destination_root / manifest.manifest_id / entry.resolved_filename()
        if destination.exists() and _matches_expected_checksum(destination, entry.expected_sha256):
            completed[entry.product_id] = destination
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(1, config.max_attempts + 1):
            try:
                temporary.unlink(missing_ok=True)
                client.download(entry.data_uri, temporary)
                if entry.expected_sha256 and sha256sum(temporary) != entry.expected_sha256:
                    raise ValueError(f"Checksum mismatch for {entry.product_id}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
                completed[entry.product_id] = destination
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == config.max_attempts:
                    raise
                LOGGER.warning(
                    "archive_download_retry",
                    extra={"context": {"product_id": entry.product_id, "attempt": attempt}},
                )
                time.sleep(config.retry_delay_seconds * attempt)
    return completed


def _matches_expected_checksum(path: Path, expected_sha256: str | None) -> bool:
    return expected_sha256 is None or sha256sum(path) == expected_sha256
