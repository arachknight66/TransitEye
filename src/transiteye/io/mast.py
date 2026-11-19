"""MAST discovery and resumable archive downloads behind testable protocols."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from astroquery.mast import Catalogs, Observations

from transiteye.config import AcquisitionConfig
from transiteye.io.manifest import ManifestEntry, ProductManifest, write_immutable_json

LOGGER = logging.getLogger("transiteye.io.mast")
PRODUCT_MISSIONS = {"tess_light_curve": "TESS", "kepler_light_curve": "Kepler"}


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
        status, message, _ = Observations.download_file(
            data_uri, local_path=str(destination), cache=False, verbose=False
        )
        if status != "COMPLETE":
            raise ValueError(f"MAST download {status} for {data_uri}: {message}")


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


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    """Identity and original verified content of an acquired archive product."""

    manifest_id: str
    product_id: str
    data_uri: str
    filename: str
    target_id: str
    mission: str
    sector: int | None
    provider: str | None
    sha256: str
    verified_at: str
    schema_version: str = "1"

    @classmethod
    def create(
        cls, manifest: ProductManifest, entry: ManifestEntry, digest: str
    ) -> AcquisitionReceipt:
        return cls(
            manifest.manifest_id, entry.product_id, entry.data_uri, entry.resolved_filename(),
            entry.target_id, PRODUCT_MISSIONS[entry.product_type], entry.sector, entry.provider,
            digest, datetime.now(UTC).isoformat(),
        )


def _receipt_path(product_path: Path) -> Path:
    return product_path.with_name(f"{product_path.name}.receipt.json")


def _read_receipt(
    manifest: ProductManifest, entry: ManifestEntry, product_path: Path
) -> AcquisitionReceipt | None:
    path = _receipt_path(product_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as error:
        raise ValueError(f"Invalid acquisition receipt: {path}") from error
    digest = document.get("sha256") if isinstance(document, dict) else None
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid SHA-256 in acquisition receipt: {path}")
    try:
        verified_at = document["verified_at"]
        if datetime.fromisoformat(verified_at).utcoffset() is None:
            raise ValueError("Missing timestamp offset")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamp in acquisition receipt: {path}") from error
    expected = asdict(AcquisitionReceipt.create(manifest, entry, digest))
    expected["verified_at"] = verified_at
    if document != expected:
        raise ValueError(f"Acquisition receipt identity differs from the frozen inventory: {path}")
    return AcquisitionReceipt(**document)


def verify_acquisition_receipt(
    manifest: ProductManifest, entry: ManifestEntry, product_path: Path
) -> None:
    """Check a cached product against its frozen acquisition receipt."""

    receipt = _read_receipt(manifest, entry, product_path)
    if receipt is None:
        raise ValueError(f"Missing acquisition receipt for {product_path}. Run fetch-target.")
    if not product_path.is_file():
        raise ValueError(f"Missing acquired product: {product_path}. Run fetch-target.")
    digest = sha256sum(product_path)
    if digest != receipt.sha256 or (entry.expected_sha256 and digest != entry.expected_sha256):
        raise ValueError(f"Checksum mismatch for {product_path} against its acquisition receipt.")


def acquire_manifest(
    client: ArchiveClient,
    manifest: ProductManifest,
    destination_root: Path,
    config: AcquisitionConfig,
    validate: Callable[[ManifestEntry, Path], None] | None = None,
    *,
    with_receipts: bool = False,
) -> dict[str, Path]:
    """Download a frozen manifest with retries and content-addressable cache validation.

    This deliberately performs one download at a time in Phase 1. The later sector executor owns
    bounded parallelism; this function remains deterministic and independently testable.
    """

    if with_receipts and validate is None:
        raise ValueError("Archive acquisition receipts require a product validator.")
    completed: dict[str, Path] = {}
    for entry in manifest.entries:
        destination = destination_root / manifest.manifest_id / entry.resolved_filename()
        receipt = _read_receipt(manifest, entry, destination) if with_receipts else None
        checksums = {entry.expected_sha256, receipt.sha256 if receipt else None} - {None}
        if (receipt is not None or not with_receipts) and destination.is_file() and (
            not checksums or checksums == {sha256sum(destination)}
        ):
            try:
                if validate is not None:
                    validate(entry, destination)
                completed[entry.product_id] = destination
                continue
            except Exception:
                LOGGER.warning("archive_cache_validation_failed", extra={"context": {"product_id": entry.product_id}})
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, config.max_attempts + 1):
            try:
                with TemporaryDirectory(dir=destination.parent) as temporary_root:
                    temporary = Path(temporary_root) / destination.name
                    client.download(entry.data_uri, temporary)
                    digest = sha256sum(temporary) if with_receipts or checksums else None
                    if checksums and checksums != {digest}:
                        raise ValueError(f"Checksum mismatch for {entry.product_id}")
                    if validate is not None:
                        validate(entry, temporary)
                    temporary.replace(destination)
                    # The product is published first. An interrupted receipt write
                    # leaves an untrusted file, which replay will download again.
                    if with_receipts and digest is not None:
                        _publish_receipt(manifest, entry, destination, digest)
                completed[entry.product_id] = destination
                break
            except Exception:
                if attempt == config.max_attempts:
                    raise
                LOGGER.warning(
                    "archive_download_retry",
                    extra={"context": {"product_id": entry.product_id, "attempt": attempt}},
                )
                time.sleep(config.retry_delay_seconds * attempt)
    return completed


def _publish_receipt(
    manifest: ProductManifest, entry: ManifestEntry, destination: Path, digest: str
) -> None:
    proposed = AcquisitionReceipt.create(manifest, entry, digest)
    write_immutable_json(asdict(proposed), _receipt_path(destination))
    frozen = _read_receipt(manifest, entry, destination)
    if frozen is None or frozen.sha256 != digest:
        raise ValueError(f"Checksum mismatch for {entry.product_id} against its acquisition receipt.")
