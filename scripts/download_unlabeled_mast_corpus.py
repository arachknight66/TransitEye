"""Freeze and acquire a source-disjoint, unlabeled 50,000-target TESS corpus from MAST."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import tomllib
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from astropy.io import fits

LOGGER = logging.getLogger("transiteye.download_unlabeled_mast_corpus")
CONFIG_PATH = Path("configs/mast_unlabeled_corpus.v1.toml")
MAST_BULK_ROOT = "https://archive.stsci.edu/missions/tess/download_scripts/sector"
MAST_DOWNLOAD_ENDPOINT = "https://mast.stsci.edu/api/v0.1/Download/file/"
TIC_IN_NAME = re.compile(r"-s(?P<sector>\d{4})-(?P<target>\d{16})-")
BULK_LINE = re.compile(
    r"\bcurl\b.*?\s-o\s+(?P<filename>\S+_lc\.fits)\s+"
    r"(?P<url>https://mast\.stsci\.edu/api/v0\.1/Download/file/\?uri=mast:TESS/product/\S+_lc\.fits)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _target_id(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    result = str(value).strip()
    if not result.isdigit():
        return None
    return str(int(result))


def _known_targets(repo_root: Path, workspace_root: Path) -> set[str]:
    known: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in {"target_id", "tic_id", "ticid"}:
                    target = _target_id(child)
                    if target is not None:
                        known.add(target)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    manifests = repo_root / "manifests"
    for path in manifests.rglob("*.json"):
        try:
            visit(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not inspect manifest for source exclusions: %s", path)

    if workspace_root.exists():
        for path in workspace_root.rglob("*.fits"):
            match = TIC_IN_NAME.search(path.name)
            if match:
                target = _target_id(match.group("target"))
                if target is not None:
                    known.add(target)
    return known


def _fetch_sector_inventory(sector: int, scripts_dir: Path) -> tuple[list[dict[str, Any]], str]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"tesscurl_sector_{sector}_lc.sh"
    if not script_path.is_file():
        url = f"{MAST_BULK_ROOT}/tesscurl_sector_{sector}_lc.sh"
        with urlopen(url, timeout=120) as response:
            body = response.read()
        temporary = script_path.with_suffix(".sh.part")
        temporary.write_bytes(body)
        os.replace(temporary, script_path)

    script_hash = _sha256(script_path)
    lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: list[dict[str, Any]] = []
    for line in lines:
        match = BULK_LINE.search(line)
        if not match:
            continue
        filename = Path(match.group("filename")).name
        tic_match = TIC_IN_NAME.search(filename)
        if tic_match is None:
            continue
        target = _target_id(tic_match.group("target"))
        if target is None or int(tic_match.group("sector")) != sector:
            continue
        source_url = match.group("url")
        marker = "uri="
        uri = source_url.split(marker, 1)[1]
        entries.append(
            {
                "product_id": f"tess_{target}_s{sector:04d}",
                "target_id": target,
                "source_group_id": f"tic-{target}",
                "sector": sector,
                "local_filename": filename,
                "data_uri": uri,
                "product_type": "tess_light_curve",
                "label": None,
                "label_status": "unlabeled",
                "inventory_sector_script_sha256": script_hash,
            }
        )
    return entries, script_hash


def freeze_manifest(
    repo_root: Path,
    data_root: Path,
    sector_first: int = 1,
    sector_last: int = 13,
    target_count: int = 50_000,
    selection_seed: str = "transiteye-mast-unlabeled-v1",
) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("target_count") != target_count:
            raise ValueError("A frozen corpus manifest exists with a different target count.")
        return existing

    scripts_dir = data_root / "source_scripts"
    workspace_root = repo_root / ".transiteye-workspace"
    known = _known_targets(repo_root, workspace_root)
    all_products: dict[str, list[dict[str, Any]]] = {}
    script_hashes: dict[str, str] = {}
    sectors = range(sector_first, sector_last + 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventories = list(
            pool.map(lambda sector: _fetch_sector_inventory(sector, scripts_dir), sectors)
        )
    for sector, (products, digest) in zip(sectors, inventories, strict=True):
        script_hashes[str(sector)] = digest
        for product in products:
            if product["target_id"] in known:
                continue
            all_products.setdefault(product["target_id"], []).append(product)
        LOGGER.info(
            "Sector %d inventory: %d light curves; %d distinct eligible sources so far",
            sector,
            len(products),
            len(all_products),
        )

    if len(all_products) < target_count:
        raise RuntimeError(
            f"Only {len(all_products)} eligible new sources found; need {target_count}."
        )

    seed = selection_seed
    selected_targets = sorted(
        all_products,
        key=lambda target: hashlib.sha256(f"{seed}:{target}".encode()).hexdigest(),
    )[:target_count]
    selected: list[dict[str, Any]] = []
    for target in selected_targets:
        options = all_products[target]
        product = min(
            options,
            key=lambda item: hashlib.sha256(
                f"{seed}:product:{target}:{item['product_id']}".encode()
            ).hexdigest(),
        )
        selected.append(product)
    selected.sort(key=lambda item: item["product_id"])

    identity = {
        "dataset_id": "mast_tess_unlabeled_50000_v1",
        "selection_seed": seed,
        "sector_first": sector_first,
        "sector_last": sector_last,
        "target_count": target_count,
        "excluded_known_target_count": len(known),
        "inventory_script_sha256": script_hashes,
        "entries": selected,
        "label_policy": "unlabeled; catalog absence is not a negative label",
    }
    manifest_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    manifest = {
        "schema_version": "1",
        "manifest_id": manifest_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": identity["dataset_id"],
        "selection_seed": seed,
        "sector_range": [sector_first, sector_last],
        "target_count": target_count,
        "excluded_known_target_count": len(known),
        "inventory_script_sha256": script_hashes,
        "label_policy": identity["label_policy"],
        "use_policy": "unlabeled-only; supervised training requires a separately frozen label protocol",
        "entries": selected,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest


def _validate_fits(path: Path, item: dict[str, Any]) -> tuple[bool, str, dict[str, str]]:
    try:
        with fits.open(path, memmap=False, checksum=False) as hdul:
            if len(hdul) < 2 or hdul[1].data is None or hdul[1].data.names is None:
                return False, "missing_light_curve_table", {}
            if str(hdul[1].header.get("XTENSION", "")).strip() != "BINTABLE":
                return False, "extension_1_not_binary_table", {}
            primary, extension = hdul[0].header, hdul[1].header
            target = primary.get("TICID", extension.get("TICID"))
            sector = primary.get("SECTOR", extension.get("SECTOR"))
            if target is None or _target_id(target) != item["target_id"]:
                return False, "tic_mismatch", {}
            if sector is None or int(sector) != int(item["sector"]):
                return False, "sector_mismatch", {}
            columns = set(hdul[1].data.names)
            if "TIME" not in columns or not columns.intersection({"PDCSAP_FLUX", "SAP_FLUX"}):
                return False, "missing_time_or_flux_column", {}
            checksum_status: dict[str, str] = {}
            for hdu in hdul:
                hdu_name = str(hdu.header.get("EXTNAME", "PRIMARY"))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    checksum_value = hdu.verify_checksum()
                    datasum_value = hdu.verify_datasum()
                checksum_status[f"{hdu_name}.CHECKSUM"] = (
                    "not_present" if checksum_value == 2 else "valid" if checksum_value == 1 else "invalid"
                )
                checksum_status[f"{hdu_name}.DATASUM"] = (
                    "not_present" if datasum_value == 2 else "valid" if datasum_value == 1 else "invalid"
                )
    except Exception as error:
        return False, f"fits_error:{type(error).__name__}:{error}", {}
    return True, "verified", checksum_status


def _download_one(item: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    destination = output_dir / Path(item["local_filename"]).name
    valid, detail, checksum_status = (
        _validate_fits(destination, item) if destination.is_file() else (False, "missing", {})
    )
    if valid:
        return {
            "product_id": item["product_id"],
            "target_id": item["target_id"],
            "status": "verified_existing",
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
            "fits_internal_checksum_status": checksum_status,
        }

    partial = destination.with_name(destination.name + ".part")
    query = urlencode({"uri": item["data_uri"]})
    url = f"{MAST_DOWNLOAD_ENDPOINT}?{query}"
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "300",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "product_id": item["product_id"],
                "target_id": item["target_id"],
                "status": "download_failed",
                "detail": result.stderr[-1000:],
            }
        valid, detail, checksum_status = _validate_fits(partial, item)
        if not valid:
            partial.unlink(missing_ok=True)
            return {
                "product_id": item["product_id"],
                "target_id": item["target_id"],
                "status": "validation_failed",
                "detail": detail,
            }
        digest = _sha256(partial)
        byte_count = partial.stat().st_size
        os.replace(partial, destination)
        return {
            "product_id": item["product_id"],
            "target_id": item["target_id"],
            "status": "downloaded",
            "sha256": digest,
            "bytes": byte_count,
            "fits_internal_checksum_status": checksum_status,
        }
    except Exception as error:
        return {
            "product_id": item["product_id"],
            "target_id": item["target_id"],
            "status": "download_failed",
            "detail": f"{type(error).__name__}: {error}",
        }


def acquire(data_root: Path, workers: int = 8) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("No frozen products.v1.json; prepare the corpus first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = data_root / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_root / "acquisition_report.v1.json"
    report: dict[str, Any] = {
        "schema_version": "1",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_id": manifest["dataset_id"],
        "manifest_id": manifest["manifest_id"],
        "download_method": "MAST Portal Download API (portal may redirect to MAST object storage)",
        "output_directory": str(output_dir),
        "target_count": manifest["target_count"],
        "workers": min(16, max(1, workers)),
        "results": {},
    }
    _write_json(report_path, report)
    entries = manifest["entries"]
    with ThreadPoolExecutor(max_workers=report["workers"]) as pool:
        futures = {pool.submit(_download_one, item, output_dir): item["product_id"] for item in entries}
        for done, future in enumerate(as_completed(futures), start=1):
            product_id = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "product_id": product_id,
                    "status": "download_failed",
                    "detail": f"{type(error).__name__}: {error}",
                }
            report["results"][product_id] = result
            if done % 250 == 0 or done == len(entries):
                report["updated_at"] = datetime.now(UTC).isoformat()
                _write_json(report_path, report)
                LOGGER.info("Processed %d/%d selected products", done, len(entries))

    report["finished_at"] = datetime.now(UTC).isoformat()
    report["counts"] = {
        status: sum(1 for result in report["results"].values() if result["status"] == status)
        for status in sorted({result["status"] for result in report["results"].values()})
    }
    report["failed_product_ids"] = sorted(
        product_id
        for product_id, result in report["results"].items()
        if result["status"] not in {"downloaded", "verified_existing"}
    )
    report["completed_product_count"] = len(report["results"])
    integrity: dict[str, Counter[str]] = {}
    for result in report["results"].values():
        for key, value in result.get("fits_internal_checksum_status", {}).items():
            integrity.setdefault(key, Counter())[value] += 1
    report["fits_internal_checksum_counts"] = {
        key: dict(counts) for key, counts in sorted(integrity.items())
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(".transiteye-workspace/data/mast_unlabeled_50000_v1")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parser.add_argument("--target-count", type=int, default=int(config["target_count"]))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prepare_only:
        manifest = freeze_manifest(
            Path.cwd(),
            args.data_root,
            sector_first=int(config["sector_first"]),
            sector_last=int(config["sector_last"]),
            target_count=args.target_count,
            selection_seed=str(config["selection_seed"]),
        )
        LOGGER.info(
            "Frozen manifest %s with %d distinct targets",
            manifest["manifest_id"],
            manifest["target_count"],
        )
        return
    report = acquire(args.data_root, args.workers)
    LOGGER.info("Acquisition counts: %s", report["counts"])
    if report["failed_product_ids"]:
        raise SystemExit(f"{len(report['failed_product_ids'])} product(s) failed; rerun to resume")


if __name__ == "__main__":
    main()
