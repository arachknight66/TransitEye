"""Acquire every MAST SPOC TESS light curve for confirmed-planet host TICs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq
from download_unlabeled_mast_corpus import (
    BULK_LINE,
    MAST_BULK_ROOT,
    TIC_IN_NAME,
    _download_one,
    _sha256,
    _target_id,
    _write_json,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path(
    ".transiteye-workspace/data/mast_confirmed_exoplanet_tess_v1"
)
EXOPLANET_ARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOPLANET_QUERY = (
    "select pl_name,hostname,tic_id,discoverymethod,pl_orbper,pl_tranmid,pl_trandur,"
    "pl_trandep,disc_facility,disc_pubdate from pscomppars"
)
TIC_ID_PATTERN = re.compile(r"\bTIC\s*(\d+)\b", re.IGNORECASE)
MAX_SECTOR = 107


def _query_confirmed_planets(catalog_path: Path) -> tuple[list[dict[str, Any]], str]:
    if catalog_path.is_file():
        raw = catalog_path.read_bytes()
    else:
        query = urlencode({"query": EXOPLANET_QUERY, "format": "json"})
        with urlopen(f"{EXOPLANET_ARCHIVE_TAP}?{query}", timeout=180) as response:
            raw = response.read()
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = catalog_path.with_suffix(catalog_path.suffix + ".part")
        temporary.write_bytes(raw)
        os.replace(temporary, catalog_path)
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("NASA Exoplanet Archive returned invalid JSON") from error
    if not isinstance(records, list) or not records:
        raise ValueError("NASA Exoplanet Archive returned no confirmed-planet records")
    required = {"pl_name", "hostname", "tic_id", "discoverymethod"}
    if required - set(records[0]):
        raise ValueError("NASA Exoplanet Archive response is missing required columns")
    return records, hashlib.sha256(raw).hexdigest()


def _tic_id(value: Any) -> str | None:
    match = TIC_ID_PATTERN.search(str(value or ""))
    return str(int(match.group(1))) if match else None


def _inventory_all_sectors(
    scripts_dir: Path,
    host_tics: set[str],
    first_sector: int = 1,
    last_sector: int = MAX_SECTOR,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    sectors = list(range(first_sector, last_sector + 1))

    def fetch_and_match(sector: int) -> tuple[int, list[dict[str, Any]], str]:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"tesscurl_sector_{sector}_lc.sh"
        if not script_path.is_file():
            with urlopen(f"{MAST_BULK_ROOT}/tesscurl_sector_{sector}_lc.sh", timeout=120) as response:
                body = response.read()
            temporary = script_path.with_suffix(".sh.part")
            temporary.write_bytes(body)
            os.replace(temporary, script_path)
        script_hash = _sha256(script_path)
        matches: list[dict[str, Any]] = []
        with script_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                found = BULK_LINE.search(line)
                if not found:
                    continue
                filename = Path(found.group("filename")).name
                target_match = TIC_IN_NAME.search(filename)
                if target_match is None or int(target_match.group("sector")) != sector:
                    continue
                tic = _target_id(target_match.group("target"))
                if tic not in host_tics:
                    continue
                uri = found.group("url").split("uri=", 1)[1]
                matches.append(
                    {
                        "product_id": Path(filename).stem,
                        "target_id": tic,
                        "source_group_id": f"tic-{tic}",
                        "sector": sector,
                        "local_filename": filename,
                        "data_uri": uri,
                        "product_type": "tess_light_curve",
                        "archive_collection": "SPOC",
                        "archive_source": "MAST official TESS sector light-curve inventory",
                    }
                )
        return sector, matches, script_hash

    with ThreadPoolExecutor(max_workers=8) as pool:
        inventories = list(pool.map(fetch_and_match, sectors))
    products: list[dict[str, Any]] = []
    script_hashes: dict[str, str] = {}
    for sector, sector_products, script_hash in inventories:
        script_hashes[str(sector)] = script_hash
        products.extend(sector_products)
        LOGGER.info(
            "Sector %d: %d SPOC light curves in the official MAST inventory",
            sector,
            len(sector_products),
        )
    return products, script_hashes


def freeze_manifest(data_root: Path, last_sector: int = MAX_SECTOR) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    catalog_path = data_root / "catalogs" / "confirmed_planets_pscomppars.v1.json"
    planets, catalog_hash = _query_confirmed_planets(catalog_path)
    planets_by_tic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    planets_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ticless_rows: list[dict[str, Any]] = []
    for planet in planets:
        host = str(planet.get("hostname") or "").strip()
        tic = _tic_id(planet.get("tic_id"))
        normalized = {key: value for key, value in planet.items()}
        normalized["tic_id_normalized"] = tic
        if host:
            planets_by_host[host].append(normalized)
        if tic:
            planets_by_tic[tic].append(normalized)
        else:
            ticless_rows.append(normalized)

    scripts_dir = data_root / "source_scripts"
    inventory, script_hashes = _inventory_all_sectors(
        scripts_dir, set(planets_by_tic), 1, last_sector
    )
    seen_products: set[str] = set()
    selected: list[dict[str, Any]] = []
    for product in inventory:
        tic = product["target_id"]
        host_planets = planets_by_tic.get(tic, [])
        if not host_planets:
            continue
        product_id = product["product_id"]
        if product_id in seen_products:
            continue
        seen_products.add(product_id)
        planet_names = sorted(
            {str(record["pl_name"]) for record in host_planets if record.get("pl_name")}
        )
        transiting_planets = sorted(
            {
                str(record["pl_name"])
                for record in host_planets
                if record.get("pl_name")
                and str(record.get("discoverymethod", "")).strip().lower() == "transit"
            }
        )
        selected.append(
            {
                **product,
                "confirmed_planet_names": planet_names,
                "confirmed_planet_count": len(planet_names),
                "transiting_planet_names": transiting_planets,
                "transiting_planet_count": len(transiting_planets),
                "known_exoplanet_host": True,
                "morphology_label": None,
                "label_status": "confirmed_host_context_only",
            }
        )
    selected.sort(key=lambda item: (item["target_id"], item["sector"], item["product_id"]))

    all_host_tics = sorted(planets_by_tic)
    represented_tics = {item["target_id"] for item in selected}
    host_rows = []
    for tic in all_host_tics:
        host_planets = planets_by_tic[tic]
        host_rows.append(
            {
                "target_id": tic,
                "source_group_id": f"tic-{tic}",
                "planet_names_json": json.dumps(
                    sorted(
                        {str(record["pl_name"]) for record in host_planets if record.get("pl_name")}
                    )
                ),
                "planet_count": len(host_planets),
                "discovery_methods_json": json.dumps(
                    sorted(
                        {
                            str(record.get("discoverymethod", ""))
                            for record in host_planets
                            if record.get("discoverymethod")
                        }
                    )
                ),
                "has_transiting_confirmed_planet": any(
                    str(record.get("discoverymethod", "")).strip().lower() == "transit"
                    for record in host_planets
                ),
                "has_tess_spoc_product": tic in represented_tics,
            }
        )
    hosts_path = data_root / "confirmed_hosts.v1.parquet"
    pq.write_table(pa.Table.from_pylist(host_rows), hosts_path, compression="zstd")

    identity = {
        "dataset_id": "mast_tess_confirmed_exoplanet_hosts_v1",
        "catalog_source": "NASA Exoplanet Archive pscomppars confirmed planets",
        "catalog_sha256": catalog_hash,
        "sector_first": 1,
        "sector_last": last_sector,
        "inventory_script_sha256": script_hashes,
        "entries": selected,
        "host_record_count": len(host_rows),
        "confirmed_planet_row_count": len(planets),
        "ticless_confirmed_planet_row_count": len(ticless_rows),
        "known_planet_host_tic_count": len(planets_by_tic),
        "product_count": len(selected),
        "represented_host_tic_count": len(represented_tics),
        "unrepresented_host_tic_count": len(set(all_host_tics) - represented_tics),
        "morphology_label_policy": "None assigned; confirmed-host status is context only",
        "signal_label_policy": "Do not label every host-sector curve as a transit positive; require event-specific evidence separately",
    }
    manifest_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    manifest = {
        "schema_version": "1",
        "manifest_id": manifest_id,
        "created_at": datetime.now(UTC).isoformat(),
        **identity,
    }
    _write_json(manifest_path, manifest)
    _write_json(
        data_root / "archive_query_report.v1.json",
        {
            "schema_version": "1",
            "archive_url": EXOPLANET_ARCHIVE_TAP,
            "query": EXOPLANET_QUERY,
            "catalog_file": str(catalog_path),
            "catalog_sha256": catalog_hash,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "planet_row_count": len(planets),
            "host_name_count": len(planets_by_host),
            "rows_without_tic_id_count": len(ticless_rows),
            "host_tic_count": len(planets_by_tic),
            "all_hosts_parquet": hosts_path.name,
            "all_hosts_parquet_sha256": _sha256(hosts_path),
        },
    )
    LOGGER.info(
        "Frozen manifest %s: %d confirmed planets, %d TIC hosts, %d matched TESS SPOC products across sectors 1-%d",
        manifest_id,
        len(planets),
        len(planets_by_tic),
        len(selected),
        last_sector,
    )
    return manifest


def acquire(data_root: Path, workers: int = 16) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run with --prepare-only first to freeze the product manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = data_root / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_root / "acquisition_report.v1.json"
    report: dict[str, Any] = {
        "schema_version": "1",
        "dataset_id": manifest["dataset_id"],
        "manifest_id": manifest["manifest_id"],
        "source": "MAST Portal Download API",
        "output_directory": str(output_dir),
        "target_count": manifest["product_count"],
        "workers": min(16, max(1, workers)),
        "started_at": datetime.now(UTC).isoformat(),
        "results": {},
    }
    _write_json(report_path, report)
    entries = manifest["entries"]
    with ThreadPoolExecutor(max_workers=report["workers"]) as pool:
        futures = {
            pool.submit(_download_one, item, output_dir): item["product_id"]
            for item in entries
        }
        for done, future in enumerate(as_completed(futures), start=1):
            product_id = futures[future]
            try:
                result = future.result()
            except Exception as error:  # per-product failure is captured for resume/audit
                result = {
                    "product_id": product_id,
                    "target_id": next(
                        (item["target_id"] for item in entries if item["product_id"] == product_id),
                        None,
                    ),
                    "status": "download_failed",
                    "detail": f"{type(error).__name__}: {error}",
                }
            report["results"][product_id] = result
            if done % 250 == 0 or done == len(entries):
                report["updated_at"] = datetime.now(UTC).isoformat()
                _write_json(report_path, report)
                LOGGER.info("Processed %d/%d products", done, len(entries))

    report["finished_at"] = datetime.now(UTC).isoformat()
    report["counts"] = {
        status: sum(result["status"] == status for result in report["results"].values())
        for status in sorted({result["status"] for result in report["results"].values()})
    }
    report["failed_product_ids"] = sorted(
        product_id
        for product_id, result in report["results"].items()
        if result["status"] not in {"downloaded", "verified_existing"}
    )
    report["completed_product_count"] = len(report["results"])
    _write_json(report_path, report)
    LOGGER.info("Acquisition counts: %s", report["counts"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--last-sector", type=int, default=MAX_SECTOR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prepare_only:
        freeze_manifest(args.data_root, args.last_sector)
        return
    report = acquire(args.data_root, args.workers)
    if report["failed_product_ids"]:
        raise SystemExit(f"{len(report['failed_product_ids'])} product(s) failed; rerun to resume")


if __name__ == "__main__":
    main()
