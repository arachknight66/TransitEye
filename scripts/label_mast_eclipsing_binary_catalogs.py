"""Create sector- and ephemeris-gated EB catalog proxy labels for MAST light curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path(".transiteye-workspace/data/mast_eclipsing_binary_catalogs_v1")
DEFAULT_CATALOG_ROOT = Path(".transiteye-workspace/data/eclipsing_binary_catalogs_v1")
CATALOG_FILES = {
    "MAST_TESS_EBs_v1.0": "tess_ebs_mast_s0001_s0026_v1.0.csv",
    "TESS_Ten_Thousand_new": "tess_ten_thousand_new_ebs_vizier_table3.tsv",
    "TESS_Ten_Thousand_known": "tess_ten_thousand_known_ebs_vizier_table4.tsv",
}


@dataclass(frozen=True)
class EBRecord:
    catalog: str
    record_id: str
    tic_id: str
    period_days: float | None
    primary_epoch_bjd_minus_2457000: float | None
    period_uncertainty_days: float | None
    epoch_uncertainty_days: float | None
    sectors: tuple[int, ...] | None
    raw_primary_duration_hours: float | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_tic(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid TIC identifier in catalog or manifest: {value!r}")
    return text


def _optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"--", "null", "None"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _sectors(value: Any) -> tuple[int, ...] | None:
    text = str(value or "").strip().strip('"')
    found = tuple(sorted({int(token) for token in re.findall(r"(?<!\d)\d{1,3}(?!\d)", text)}))
    return found or None


def _rounding_uncertainty(value: Any) -> float | None:
    """Return half the last printed decimal place as an explicit rounding bound."""
    text = str(value or "").strip()
    if not text or text in {"--", "null", "None"}:
        return None
    if "e" in text.lower():
        mantissa, exponent = re.split("[eE]", text)
        decimals = len(mantissa.partition(".")[2])
        scale = int(exponent) - decimals
    else:
        scale = -len(text.partition(".")[2])
    return 0.5 * 10.0**scale


def _vizier_rows(path: Path) -> list[dict[str, str]]:
    lines = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.startswith("#")]
    header_index = next((index for index, line in enumerate(lines)
                         if line.startswith("TIC\t")), None)
    if header_index is None:
        raise ValueError(f"VizieR table has no TIC header row: {path}")
    columns = lines[header_index].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1:]:
        values = line.split("\t")
        if not values or not values[0].strip().isdigit():
            continue  # VizieR includes units and separator rows between the header and records.
        rows.append({column: values[index].strip() if index < len(values) else ""
                     for index, column in enumerate(columns)})
    return rows


def load_catalog_records(catalog_root: Path) -> tuple[list[EBRecord], dict[str, str], dict[str, int]]:
    records: list[EBRecord] = []
    hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    mast_path = catalog_root / CATALOG_FILES["MAST_TESS_EBs_v1.0"]
    with mast_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    hashes[mast_path.name] = _sha256(mast_path)
    row_counts["MAST_TESS_EBs_v1.0"] = len(rows)
    for index, row in enumerate(rows):
        records.append(
            EBRecord(
                catalog="MAST_TESS_EBs_v1.0",
                record_id=f"signal-{row.get('tess_id', '')}-{row.get('signal_id', index)}-{index}",
                tic_id=_normalize_tic(row.get("tess_id", "")),
                period_days=_optional_float(row.get("period")),
                primary_epoch_bjd_minus_2457000=_optional_float(row.get("bjd0")),
                period_uncertainty_days=_optional_float(row.get("period_uncert")),
                epoch_uncertainty_days=_optional_float(row.get("bjd0_uncert")),
                sectors=_sectors(row.get("sectors")),
                raw_primary_duration_hours=None,
            )
        )

    new_path = catalog_root / CATALOG_FILES["TESS_Ten_Thousand_new"]
    rows = _vizier_rows(new_path)
    hashes[new_path.name] = _sha256(new_path)
    row_counts["TESS_Ten_Thousand_new"] = len(rows)
    for index, row in enumerate(rows):
        records.append(
            EBRecord(
                catalog="TESS_Ten_Thousand_new",
                record_id=f"table3-row-{index + 1}",
                tic_id=_normalize_tic(row.get("TIC", "")),
                period_days=_optional_float(row.get("Per")),
                primary_epoch_bjd_minus_2457000=_optional_float(row.get("T0-pri")),
                period_uncertainty_days=_rounding_uncertainty(row.get("Per")),
                epoch_uncertainty_days=_rounding_uncertainty(row.get("T0-pri")),
                sectors=_sectors(row.get("Sect")),
                raw_primary_duration_hours=_optional_float(row.get("Dur-pri")),
            )
        )

    known_path = catalog_root / CATALOG_FILES["TESS_Ten_Thousand_known"]
    rows = _vizier_rows(known_path)
    hashes[known_path.name] = _sha256(known_path)
    row_counts["TESS_Ten_Thousand_known"] = len(rows)
    for index, row in enumerate(rows):
        records.append(
            EBRecord(
                catalog="TESS_Ten_Thousand_known",
                record_id=f"table4-row-{index + 1}",
                tic_id=_normalize_tic(row.get("TIC", "")),
                period_days=_optional_float(row.get("Per-TESS")),
                primary_epoch_bjd_minus_2457000=_optional_float(row.get("T0-TESS")),
                period_uncertainty_days=_rounding_uncertainty(row.get("Per-TESS")),
                epoch_uncertainty_days=_rounding_uncertainty(row.get("T0-TESS")),
                sectors=None,
                raw_primary_duration_hours=None,
            )
        )
    return records, hashes, row_counts


def _predicted_primary_in_interval(
    record: EBRecord,
    start_bjd_minus_2457000: float,
    stop_bjd_minus_2457000: float,
) -> tuple[bool, float | None, int | None, float | None]:
    period = record.period_days
    epoch = record.primary_epoch_bjd_minus_2457000
    if period is None or period <= 0 or epoch is None or stop_bjd_minus_2457000 < start_bjd_minus_2457000:
        return False, None, None, None

    n = math.floor((start_bjd_minus_2457000 - epoch) / period) - 1
    # TESS product intervals are about one sector long; this bounded loop covers the full interval.
    for cycle in range(n, n + 500):
        predicted = epoch + cycle * period
        if predicted > stop_bjd_minus_2457000:
            break
        period_error = (abs(cycle) * record.period_uncertainty_days
                        if record.period_uncertainty_days is not None else 0.0)
        epoch_error = record.epoch_uncertainty_days or 0.0
        sigma = math.hypot(epoch_error, period_error)
        half_duration = ((record.raw_primary_duration_hours or 0.0) / 48.0)
        # The rounding and reported ephemeris uncertainties are expanded to a 3-sigma coverage window.
        tolerance = half_duration + 3.0 * sigma
        if start_bjd_minus_2457000 - tolerance <= predicted <= stop_bjd_minus_2457000 + tolerance:
            return True, predicted, cycle, sigma
    return False, None, None, None


def _read_product_interval(path: Path, target_id: str, sector: int) -> tuple[float, float]:
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        primary = hdul[0].header
        lightcurve = hdul["LIGHTCURVE"].header
        fits_tic = _normalize_tic(primary.get("TICID", lightcurve.get("TICID", "")))
        fits_sector = int(primary.get("SECTOR", lightcurve.get("SECTOR", -1)))
        if fits_tic != target_id:
            raise ValueError(f"FITS TIC {fits_tic} does not match manifest TIC {target_id}")
        if fits_sector != sector:
            raise ValueError(f"FITS sector {fits_sector} does not match manifest sector {sector}")
        start = _optional_float(lightcurve.get("TSTART"))
        stop = _optional_float(lightcurve.get("TSTOP"))
        bjdref = (_optional_float(lightcurve.get("BJDREFI")) or 0.0) + (
            _optional_float(lightcurve.get("BJDREFF")) or 0.0
        )
        if start is None or stop is None or not math.isclose(bjdref, 2457000.0, abs_tol=1e-6):
            raise ValueError("missing TSTART/TSTOP or unexpected FITS BJD reference")
        return start, stop


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_sidecar(data_root: Path, catalog_root: Path) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    acquisition_path = data_root / "acquisition_report.v1.json"
    for required in (manifest_path, acquisition_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = manifest["entries"]
    if len(entries) != manifest.get("available_mast_spoc_product_count"):
        raise ValueError("Frozen product manifest count does not match available product count")
    if any(entry.get("morphology_label") is not None for entry in entries):
        raise ValueError("Refusing to label a product manifest that already contains morphology labels")
    product_ids = [entry["product_id"] for entry in entries]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Frozen product manifest contains duplicate product IDs")
    if acquisition.get("completed_product_count") != len(entries):
        raise ValueError("MAST acquisition report does not show all manifest products completed")
    failed = [item for item in acquisition.get("results", {}).values()
              if item.get("status") not in {"downloaded", "verified_existing"}]
    if failed:
        raise ValueError(f"MAST acquisition report contains {len(failed)} non-success products")

    catalog_records, catalog_hashes, catalog_row_counts = load_catalog_records(catalog_root)
    records_by_tic: dict[str, list[EBRecord]] = defaultdict(list)
    for record in catalog_records:
        records_by_tic[record.tic_id].append(record)

    label_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    fits_errors: Counter[str] = Counter()

    for entry in entries:
        tic = _normalize_tic(entry["target_id"])
        sector = int(entry["sector"])
        product_id = str(entry["product_id"])
        product_path = data_root / "curves" / str(entry["local_filename"])
        acquisition_row = acquisition["results"].get(product_id, {})
        matching_records = records_by_tic.get(tic, [])
        sector_records = [record for record in matching_records
                          if record.sectors is not None and sector in record.sectors]
        interval_start: float | None = None
        interval_stop: float | None = None
        fits_status = "verified"
        try:
            if not product_path.is_file():
                raise FileNotFoundError(f"missing local FITS product: {product_path.name}")
            interval_start, interval_stop = _read_product_interval(product_path, tic, sector)
        except (OSError, ValueError, KeyError) as exc:
            fits_status = f"unavailable_or_invalid:{type(exc).__name__}"
            fits_errors[type(exc).__name__] += 1
            LOGGER.warning("Leaving %s unlabeled: %s", product_id, exc)

        event_supports: list[dict[str, Any]] = []
        for record in matching_records:
            if record.sectors is None:
                timing_status = "context_only_no_sector_evidence"
                event_time = None
                cycle_number = None
                timing_sigma = None
            elif sector not in record.sectors:
                timing_status = "product_sector_not_listed"
                event_time = None
                cycle_number = None
                timing_sigma = None
            elif interval_start is None or interval_stop is None:
                timing_status = "product_interval_unavailable"
                event_time = None
                cycle_number = None
                timing_sigma = None
            elif record.period_days is None or record.primary_epoch_bjd_minus_2457000 is None:
                timing_status = "catalog_primary_ephemeris_incomplete"
                event_time = None
                cycle_number = None
                timing_sigma = None
            else:
                matched, event_time, cycle_number, timing_sigma = _predicted_primary_in_interval(
                    record, interval_start, interval_stop
                )
                timing_status = (
                    "primary_eclipse_ephemeris_overlaps_product_interval"
                    if matched else "primary_eclipse_ephemeris_outside_product_interval"
                )
                if matched:
                    event_supports.append({
                        "catalog": record.catalog,
                        "record_id": record.record_id,
                        "predicted_primary_bjd_minus_2457000": event_time,
                        "cycle_number": cycle_number,
                        "combined_ephemeris_sigma_days": timing_sigma,
                    })
            event_rows.append({
                "product_id": product_id,
                "target_id": tic,
                "sector": sector,
                "catalog": record.catalog,
                "catalog_record_id": record.record_id,
                "catalog_tic_id": record.tic_id,
                "catalog_period_days": record.period_days,
                "catalog_primary_epoch_bjd_minus_2457000": record.primary_epoch_bjd_minus_2457000,
                "catalog_period_uncertainty_days": record.period_uncertainty_days,
                "catalog_epoch_uncertainty_days": record.epoch_uncertainty_days,
                "catalog_primary_duration_hours": record.raw_primary_duration_hours,
                "catalog_sectors": list(record.sectors) if record.sectors is not None else None,
                "product_tstart_bjd_minus_2457000": interval_start,
                "product_tstop_bjd_minus_2457000": interval_stop,
                "predicted_primary_bjd_minus_2457000": event_time,
                "predicted_cycle_number": cycle_number,
                "combined_ephemeris_sigma_days": timing_sigma,
                "event_match_status": timing_status,
            })

        if fits_status != "verified":
            label = None
            label_status = "unlabeled_product_identity_or_interval_invalid"
        elif not matching_records:
            label = None
            label_status = "unlabeled_no_exact_tic_catalog_match"
        elif event_supports:
            label = "eclipse_like"
            label_status = "catalog_primary_eclipse_in_sector_product_interval"
            label_counts[label] += 1
        elif sector_records and interval_start is not None:
            label = None
            label_status = "unlabeled_catalog_primary_eclipse_outside_product_interval"
        elif sector_records:
            label = None
            label_status = "unlabeled_catalog_ephemeris_incomplete"
        elif any(record.sectors is None for record in matching_records):
            label = None
            label_status = "unlabeled_catalog_context_without_sector_evidence"
        else:
            label = None
            label_status = "unlabeled_product_sector_not_listed_by_catalog"
        status_counts[label_status] += 1
        label_rows.append({
            "product_id": product_id,
            "target_id": tic,
            "source_group_id": str(entry["source_group_id"]),
            "sector": sector,
            "product_sha256": acquisition_row.get("sha256"),
            "label": label,
            "label_status": label_status,
            "label_scope": "catalog-proxy; same-TIC, listed-sector primary-eclipse ephemeris overlaps product interval",
            "catalog_match_count": len(matching_records),
            "catalog_sector_match_count": len(sector_records),
            "catalog_event_support_count": len(event_supports),
            "catalogs_same_tic_json": json.dumps(sorted({record.catalog for record in matching_records})),
            "catalog_record_ids_same_tic_json": json.dumps(sorted(record.record_id for record in matching_records)),
            "catalog_record_ids_sector_json": json.dumps(sorted(record.record_id for record in sector_records)),
            "catalog_event_supports_json": json.dumps(event_supports, sort_keys=True),
            "product_tstart_bjd_minus_2457000": interval_start,
            "product_tstop_bjd_minus_2457000": interval_stop,
            "fits_identity_status": fits_status,
        })

    label_path = data_root / "labels_eb_catalog_proxy.v1.parquet"
    events_path = data_root / "eb_catalog_event_matches.v1.jsonl"
    report_path = data_root / "eb_catalog_label_report.v1.json"
    schema = pa.schema([
        ("product_id", pa.string()),
        ("target_id", pa.string()),
        ("source_group_id", pa.string()),
        ("sector", pa.int16()),
        ("product_sha256", pa.string()),
        ("label", pa.string()),
        ("label_status", pa.string()),
        ("label_scope", pa.string()),
        ("catalog_match_count", pa.int16()),
        ("catalog_sector_match_count", pa.int16()),
        ("catalog_event_support_count", pa.int16()),
        ("catalogs_same_tic_json", pa.string()),
        ("catalog_record_ids_same_tic_json", pa.string()),
        ("catalog_record_ids_sector_json", pa.string()),
        ("catalog_event_supports_json", pa.string()),
        ("product_tstart_bjd_minus_2457000", pa.float64()),
        ("product_tstop_bjd_minus_2457000", pa.float64()),
        ("fits_identity_status", pa.string()),
    ])
    table = pa.Table.from_pylist(label_rows, schema=schema)
    _write_parquet_atomic(label_path, table)
    _write_jsonl_atomic(events_path, event_rows)

    report = {
        "schema_version": "1",
        "dataset_id": manifest["dataset_id"],
        "label_policy_id": "eb_catalog_same_sector_primary_ephemeris_proxy.v1",
        "label_snapshot_id": hashlib.sha256(
            f"{manifest['manifest_id']}:{_sha256(manifest_path)}:{json.dumps(catalog_hashes, sort_keys=True)}:eb-proxy-v1".encode()
        ).hexdigest()[:16],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "catalog_sources": [
            {"catalog": name, "file": filename, "sha256": catalog_hashes[filename],
             "row_count": catalog_row_counts[name]}
            for name, filename in CATALOG_FILES.items()
        ],
        "catalog_context_only_policy": "TESS_Ten_Thousand_known has no sector field and cannot independently assign a product label",
        "product_manifest_file": manifest_path.name,
        "product_manifest_id": manifest["manifest_id"],
        "product_manifest_sha256": _sha256(manifest_path),
        "acquisition_report_sha256": _sha256(acquisition_path),
        "label_script_sha256": _sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "astropy_version": astropy.__version__,
        "pyarrow_version": pa.__version__,
        "time_policy": "Compare catalog BJD-2457000 primary eclipse ephemerides with SPOC LIGHTCURVE TSTART/TSTOP in the same BJD-2457000 frame; expand by three propagated catalog timing sigmas and any supplied Ten Thousand primary duration.",
        "product_count": len(label_rows),
        "distinct_tic_count": len({row["target_id"] for row in label_rows}),
        "event_match_row_count": len(event_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "label_status_counts": dict(sorted(status_counts.items())),
        "invalid_or_missing_product_count": sum(fits_errors.values()),
        "invalid_or_missing_product_errors": dict(sorted(fits_errors.items())),
        "outputs": {
            "labels_parquet": label_path.name,
            "event_matches_jsonl": events_path.name,
        },
        "limitations": [
            "Labels are catalog proxies for cataloged systems and event windows, not independent morphology ground truth.",
            "This pass checks FITS target/sector identity and nominal time bounds; it does not inspect flux shape, verify eclipse depth, or localize the variable source.",
            "Unmatched, off-sector, context-only, incomplete-ephemeris, and out-of-interval products remain unlabeled; no catalog absence is a negative.",
            "All products from one TIC share a source_group_id and must remain in one future partition.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %d EB label rows; labels=%s statuses=%s", len(label_rows), label_counts, status_counts)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    report = build_sidecar(args.data_root, args.catalog_root)
    print(json.dumps({
        "product_count": report["product_count"],
        "label_counts": report["label_counts"],
        "label_status_counts": report["label_status_counts"],
        "report": str(args.data_root / "eb_catalog_label_report.v1.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
