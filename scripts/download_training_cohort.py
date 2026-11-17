"""Restore the frozen development products needed to train the real-catalog model.

Only one latest-sector light curve per train, calibration, or selection source is acquired.
The locked test products are excluded before any archive request and are never opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOGGER = logging.getLogger("transiteye.download_training_cohort")
DEV_SPLITS = {"train", "calibration", "selection"}
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_fits_header(handle: Any) -> tuple[dict[str, Any], int]:
    """Read one FITS header, including its 2880-byte padding, using only stdlib."""
    header: dict[str, Any] = {}
    bytes_read = 0
    ended = False
    while not ended:
        block = handle.read(2880)
        if len(block) != 2880:
            raise ValueError("truncated FITS header")
        bytes_read += 2880
        for offset in range(0, 2880, 80):
            card = block[offset : offset + 80].decode("ascii", errors="strict")
            keyword = card[:8].strip()
            if keyword == "END":
                ended = True
                break
            if keyword and card[8:10] == "= ":
                raw = card[10:]
                in_string = False
                value_end = len(raw)
                for index, char in enumerate(raw):
                    if char == "'":
                        in_string = not in_string
                    elif char == "/" and not in_string:
                        value_end = index
                        break
                value = raw[:value_end].strip()
                if value.startswith("'"):
                    value = value[1:].split("'")[0].strip()
                elif value in {"T", "F"}:
                    value = value == "T"
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value.replace("D", "E"))
                        except ValueError:
                            pass
                header[keyword] = value
    return header, bytes_read


def _validate_fits(path: Path, item: dict[str, Any]) -> str | None:
    with path.open("rb") as handle:
        primary, primary_header_bytes = _read_fits_header(handle)
        primary_data_bytes = int(primary.get("NAXIS1", 0)) * int(primary.get("NAXIS2", 1))
        primary_data_bytes += int(primary.get("PCOUNT", 0))
        handle.seek(primary_header_bytes + ((primary_data_bytes + 2879) // 2880) * 2880)
        extension, _ = _read_fits_header(handle)
        if extension.get("XTENSION", "").strip() != "BINTABLE":
            return "extension_1_not_binary_table"
        if int(extension.get("NAXIS2", 0)) <= 0:
            return "empty_light_curve_table"
        target_id = primary.get("TICID", extension.get("TICID"))
        sector = primary.get("SECTOR", extension.get("SECTOR"))
        if target_id is None or str(int(target_id)) != str(int(item["target_id"])):
            return "tic_mismatch"
        if sector is None or int(sector) != int(item["sector"]):
            return "sector_mismatch"
        columns = {
            str(value).strip()
            for key, value in extension.items()
            if key.startswith("TTYPE")
        }
        if "TIME" not in columns:
            return "missing_time_column"
        if not columns.intersection({"PDCSAP_FLUX", "SAP_FLUX"}):
            return "missing_flux_column"
    return _sha256(path)


def _select_products(
    products_path: Path, labels_path: Path, splits_path: Path
) -> tuple[list[dict[str, Any]], int, str]:
    products = json.loads(products_path.read_text(encoding="utf-8"))["entries"]
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    split_doc = json.loads(splits_path.read_text(encoding="utf-8"))
    split_by_group = {row["source_group_id"]: row["split"] for row in split_doc["assignments"]}
    label_by_target = {str(row["target_id"]): row for row in labels}

    # Match train_real_cohort.py's deterministic latest-sector rule.
    selected_by_target: dict[str, dict[str, Any]] = {}
    for product in products:
        target = str(product["target_id"])
        current = selected_by_target.get(target)
        rank = (int(product.get("sector") or -1), product.get("local_filename") or "")
        current_rank = (
            int(current.get("sector") or -1), current.get("local_filename") or ""
        ) if current else None
        if current is None or rank > current_rank:
            selected_by_target[target] = product

    selected: list[dict[str, Any]] = []
    test_targets = 0
    for target, product in selected_by_target.items():
        label = label_by_target.get(target)
        if label is None:
            continue
        split = split_by_group.get(label["source_group_id"])
        if split == "test":
            test_targets += 1
            continue
        if split not in DEV_SPLITS:
            continue
        selected.append(
            {
                **product,
                "source_group_id": label["source_group_id"],
                "label": label["label"],
                "split": split,
            }
        )
    selected.sort(key=lambda row: (row["split"], str(row["target_id"])))
    return selected, test_targets, str(split_doc["manifest_id"])


def _check_local(path: Path, item: dict[str, Any]) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "missing"
    expected = item.get("expected_sha256")
    actual = _sha256(path)
    if expected and actual != expected:
        return False, "sha256_mismatch"
    try:
        validation_error = _validate_fits(path, item)
    except Exception as error:
        return False, f"fits_error:{type(error).__name__}:{error}"
    if validation_error != actual:
        return False, validation_error
    return True, actual


def _download_one(
    item: dict[str, Any], output_dir: Path, force_download: bool = False
) -> dict[str, Any]:
    filename = item.get("local_filename") or f"{item['product_id']}.fits"
    destination = output_dir / Path(filename).name
    valid, detail = _check_local(destination, item)
    if valid and not force_download:
        return {"product_id": item["product_id"], "status": "verified_existing", "sha256": detail}

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    try:
        query = urlencode({"uri": str(item["data_uri"])})
        portal_url = f"{MAST_DOWNLOAD_URL}?{query}"
        last_error: Exception | None = None
        attempts = 0
        for attempts in range(1, 5):
            try:
                request = Request(portal_url, headers={"User-Agent": "TransitEye/0.1"})
                with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                last_error = None
            except Exception as error:
                last_error = error
                partial.unlink(missing_ok=True)
            if last_error is None:
                break
            if attempts < 4:
                time.sleep(2 ** attempts)
        if last_error is not None:
            raise last_error
        valid, detail = _check_local(partial, item)
        if not valid:
            partial.unlink(missing_ok=True)
            return {"product_id": item["product_id"], "status": "validation_failed", "detail": detail}
        os.replace(partial, destination)
        return {
            "product_id": item["product_id"],
            "status": "downloaded",
            "sha256": detail,
            "bytes": destination.stat().st_size,
            "attempts": attempts,
        }
    except Exception as error:
        partial.unlink(missing_ok=True)
        return {
            "product_id": item["product_id"],
            "status": "download_failed",
            "detail": f"{type(error).__name__}: {error}",
        }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire(
    products_path: Path,
    labels_path: Path,
    splits_path: Path,
    output_dir: Path,
    workers: int = 4,
    force_download: bool = False,
    resume_portal: bool = False,
) -> dict[str, Any]:
    selected, excluded_test_count, split_id = _select_products(
        products_path, labels_path, splits_path
    )
    worker_count = min(16, max(1, workers))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir.parent / "real_catalog_cohort_v3_dev_acquisition.v1.json"
    manifest_path = output_dir.parent / "real_catalog_cohort_v3_dev_products.v1.json"
    manifest_entries = [
        {
            key: item.get(key)
            for key in (
                "product_id",
                "data_uri",
                "target_id",
                "sector",
                "product_type",
                "expected_sha256",
                "local_filename",
            )
        }
        for item in selected
    ]
    query = {
        "selection": "latest sector per labeled source",
        "partitions": "train,calibration,selection",
        "source_split_manifest_id": split_id,
    }
    manifest_identity = {
        "source": "MAST TESS frozen real_catalog_cohort_v3 development products",
        "entries": sorted(manifest_entries, key=lambda row: row["product_id"]),
        "query": query,
        "schema_version": "1",
    }
    manifest_id = hashlib.sha256(
        json.dumps(manifest_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    subset_manifest = {
        "manifest_id": manifest_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source": manifest_identity["source"],
        "query": query,
        "schema_version": "1",
        "source_product_manifest": str(products_path),
        "source_split_manifest_id": split_id,
        "selection_rule": "latest sector then filename per labeled source; train/calibration/selection only",
        "excluded_test_source_count": excluded_test_count,
        "entries": manifest_entries,
    }
    _write_json(manifest_path, subset_manifest)

    prior_results: dict[str, Any] = {}
    if resume_portal and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            previous.get("download_method", "").startswith("MAST Portal Download API")
            and previous.get("source_split_manifest_id") == split_id
            and previous.get("selected_count") == len(selected)
        ):
            for item in selected:
                product_id = item["product_id"]
                old_result = previous.get("results", {}).get(product_id)
                if old_result and old_result.get("status") == "downloaded":
                    valid, digest = _check_local(
                        output_dir / Path(item["local_filename"]).name, item
                    )
                    if valid and digest == old_result.get("sha256"):
                        prior_results[product_id] = old_result

    pending = [item for item in selected if item["product_id"] not in prior_results]
    report: dict[str, Any] = {
        "schema_version": "1",
        "started_at": datetime.now(UTC).isoformat(),
        "source_product_manifest": str(products_path),
        "source_label_manifest": str(labels_path),
        "source_split_manifest": str(splits_path),
        "source_split_manifest_id": split_id,
        "subset_manifest": str(manifest_path),
        "output_directory": str(output_dir),
        "selected_count": len(selected),
        "excluded_test_source_count": excluded_test_count,
        "workers": worker_count,
        "download_method": "MAST Portal Download API (portal may redirect to MAST object storage)",
        "policy": "test products were filtered before archive requests; only development products are acquired",
        "force_download": force_download or resume_portal,
        "resumed_verified_portal_count": len(prior_results),
        "results": prior_results,
    }
    _write_json(report_path, report)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_download_one, item, output_dir, force_download or resume_portal): item["product_id"]
            for item in pending
        }
        total_pending = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            product_id = futures[future]
            try:
                result = future.result()
            except Exception as error:  # Keep every product accounted for.
                result = {
                    "product_id": product_id,
                    "status": "download_failed",
                    "detail": f"{type(error).__name__}: {error}",
                }
            report["results"][product_id] = result
            if completed % 10 == 0 or completed == total_pending:
                report["updated_at"] = datetime.now(UTC).isoformat()
                _write_json(report_path, report)
                LOGGER.info(
                    "Processed %d/%d products (%d portal-verified on resume)",
                    completed,
                    total_pending,
                    len(prior_results),
                )

    report["finished_at"] = datetime.now(UTC).isoformat()
    report["counts"] = {
        status: sum(1 for row in report["results"].values() if row["status"] == status)
        for status in sorted({row["status"] for row in report["results"].values()})
    }
    report["failed_product_ids"] = sorted(
        product_id
        for product_id, row in report["results"].items()
        if row["status"] not in {"downloaded", "verified_existing"}
    )
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=Path("manifests/real_catalog_cohort.v3.products.json"))
    parser.add_argument("--labels", type=Path, default=Path("manifests/real_catalog_cohort.v3.labels.json"))
    parser.add_argument("--splits", type=Path, default=Path("manifests/real_catalog_cohort.v3.splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(".transiteye-workspace/data/real_catalog_cohort_v3_dev"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Fetch every selected product via the MAST Portal, replacing verified local copies.",
    )
    parser.add_argument(
        "--resume-portal",
        action="store_true",
        help="Keep prior checksum-verified MAST Portal downloads and fetch only the remainder.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = acquire(
        args.products,
        args.labels,
        args.splits,
        args.output_dir,
        args.workers,
        args.force_download,
        args.resume_portal,
    )
    LOGGER.info("Acquisition counts: %s", report["counts"])
    if report["failed_product_ids"]:
        raise SystemExit(f"{len(report['failed_product_ids'])} product(s) failed; see acquisition report")


if __name__ == "__main__":
    main()
